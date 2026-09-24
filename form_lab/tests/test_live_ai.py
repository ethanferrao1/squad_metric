"""Live-AI fallbacks, caching, the session cap and the cost log, with the network mocked."""
import csv

import pandas as pd
import pytest

import live_ai
import llm_explain as lx

POOL = pd.DataFrame({'web_name': ['Out', 'In'], 'code': [1, 2]})
FACTS = {'transfer_out': {'name': 'Out', 'price_m': 4.9},
         'transfer_in': {'name': 'In', 'price_m': 4.6},
         'predicted_points_gained_rounded': 17, 'horizon_gameweeks': 5}


@pytest.fixture(autouse=True)
def files(tmp_path, monkeypatch):
    monkeypatch.setattr(live_ai, 'CACHE', tmp_path / 'explanations_live.json')
    monkeypatch.setattr(live_ai, 'LOG', tmp_path / 'llm_log.csv')
    return tmp_path


def job(key='5|swap|Out|In'):
    return {'key': key, 'kind': 'explain', 'subject': 'Out->In', 'pool': POOL,
            'facts': FACTS, 'template': 'TEMPLATE'}


def reply(text, err=None):
    def fake(facts, usage=None, **kw):
        if usage is not None:
            usage.update({'prompt_tokens': 800, 'completion_tokens': 60})
        return text, err, lx.PAID_MODEL
    return fake


def run(jobs, live=True, budget=None, memo=None, api_key='user-session-key'):
    got = {}
    live_ai.resolve(jobs, lambda k, r: got.__setitem__(k, r),
                    budget if budget is not None else {'used': 0}, 5, live, memo,
                    api_key=api_key)
    return got


def test_a_good_reply_is_live_cached_and_logged(monkeypatch, files):
    monkeypatch.setattr(lx, 'call_openrouter',
                        reply('Out has lost his place; In adds 17 points over 5 gameweeks.'))
    got = run([job()])
    assert got['5|swap|Out|In']['label'] == 'live'
    assert run([job()])['5|swap|Out|In']['label'] == 'cached'
    rows = list(csv.DictReader(open(files / 'llm_log.csv', encoding='utf-8')))
    assert len(rows) == 1 and rows[0]['prompt_tokens'] == '800'
    assert float(rows[0]['cost_usd']) == pytest.approx(800 * 0.05e-6 + 60 * 0.08e-6)


def test_a_failed_call_falls_back_to_the_template(monkeypatch):
    monkeypatch.setattr(lx, 'call_openrouter', reply(None, 'http 500'))
    assert run([job()])['5|swap|Out|In'] == {'text': 'TEMPLATE', 'label': 'template'}


def test_an_invented_number_fails_the_fact_check(monkeypatch):
    monkeypatch.setattr(lx, 'call_openrouter', reply('In adds 42 points.'))
    assert run([job()])['5|swap|Out|In']['label'] == 'template'


def test_offline_makes_no_call(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('called while offline')
    monkeypatch.setattr(lx, 'call_openrouter', boom)
    assert run([job()], live=False)['5|swap|Out|In']['label'] == 'template'


def test_the_session_cap_stops_live_calls(monkeypatch):
    monkeypatch.setattr(lx, 'call_openrouter', reply('Out is out of form.'))
    budget = {'used': live_ai.MAX_CALLS}
    assert run([job()], budget=budget)['5|swap|Out|In']['label'] == 'template'
    assert budget['used'] == live_ai.MAX_CALLS


def test_a_cache_from_another_gameweek_is_not_reused(monkeypatch):
    monkeypatch.setattr(lx, 'call_openrouter', reply('Out is out of form.'))
    run([job('4|swap|Out|In')])
    assert run([job('5|swap|Out|In')])['5|swap|Out|In']['label'] == 'live'


def test_the_log_never_holds_the_key(monkeypatch, files):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-secret-value')
    monkeypatch.setattr(lx, 'call_openrouter', reply('Out is out of form.'))
    run([job()], api_key='test-secret-value')
    assert 'test-secret-value' not in (files / 'llm_log.csv').read_text(encoding='utf-8')


def test_the_key_is_passed_as_an_argument(monkeypatch):
    seen = []

    def fake(facts, usage=None, api_key=None, **kw):
        seen.append(api_key)
        return 'Out is out of form.', None, lx.PAID_MODEL
    monkeypatch.setattr(lx, 'call_openrouter', fake)
    run([job()], api_key='from-the-session')
    assert seen == ['from-the-session']


def test_no_key_makes_no_call(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('called without a key')
    monkeypatch.setattr(lx, 'call_openrouter', boom)
    assert run([job()], api_key=None)['5|swap|Out|In']['label'] == 'template'


def test_a_rejected_key_falls_back_and_stops_calling(monkeypatch):
    calls = []

    def rejected(facts, usage=None, **kw):
        calls.append(1)
        return None, lx.REJECTED, None
    monkeypatch.setattr(lx, 'call_openrouter', rejected)
    budget, memo = {'used': 0}, {}
    assert run([job()], budget=budget, memo=memo)['5|swap|Out|In']['label'] == 'template'
    assert budget['rejected'] and not memo
    run([job('5|swap|A|B')], budget=budget, memo=memo)
    assert len(calls) == 1


def test_env_key_is_ignored_unless_allowed(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'env-value')
    monkeypatch.delenv('ALLOW_ENV_KEY', raising=False)
    assert lx.env_key() is None
    text, err, _ = lx.call_openrouter({}, backoff=())
    assert text is None and err == 'no API key'
    monkeypatch.setenv('ALLOW_ENV_KEY', '1')
    assert lx.env_key() == 'env-value'


def test_a_401_is_reported_as_rejected_without_the_key(monkeypatch):
    import io
    import urllib.error

    def unauthorised(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, 'Unauthorized', {}, io.BytesIO())
    monkeypatch.setattr(lx.urllib.request, 'urlopen', unauthorised)
    text, err, _ = lx.call_openrouter({}, backoff=(), api_key='secret-abc')
    assert text is None and err == lx.REJECTED


def test_errors_are_redacted(monkeypatch):
    lookalike = 'sk' + '-' + 'z' * 24          # built here so no key-shaped literal is committed

    def leaky(req, timeout=None):
        raise OSError(f'failed with secret-abc and {lookalike}')
    monkeypatch.setattr(lx.urllib.request, 'urlopen', leaky)
    _, err, _ = lx.call_openrouter({}, backoff=(), api_key='secret-abc')
    assert 'secret-abc' not in err and 'z' * 24 not in err and lx.MASK in err


def test_free_models_cost_nothing():
    assert live_ai.cost('google/gemma-4-31b-it:free', {'prompt_tokens': 999}) == 0


def test_facts_say_same_price_and_round_the_gain():
    row = {'out': 'Out', 'in': 'In', 'out_pred': 2.0, 'in_pred': 4.0,
           'gain': 16.7, 'cost': 0}
    pool = pd.DataFrame({'web_name': ['Out', 'In'], 'code': [1, 2], 'pos': 'MID',
                         'price': [49, 49], 'modelled': True, 'drivers': None})
    proj = pd.DataFrame({'web_name': ['Out', 'In'], 'form_ppg': [1.0, 5.0]})
    f = live_ai.explain_facts(row, pool, proj)
    assert f['same_price'] is True and f['predicted_points_gained_rounded'] == 17
    assert f['transfer_in']['recent_form_points_per_game'] == 5.0
