"""The Refresh data button: staged, validated, swapped; one at a time; the page keeps its state.

Runs on copies of both caches with a fake FPL API that serves the committed
data (with one price changed, so a swap is visible). No network.
"""
import hashlib
import io
import json
import shutil
import threading
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

import build_live
import core
import fpl_api
import images
import refresh
import refresher
from form_lab import data
from ui import ai

ROOT = Path(__file__).resolve().parents[2]
QUIET = lambda *a, **k: None                     # noqa: E731


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    src = SimpleNamespace(data=Path(fpl_api.CACHE_DIR), fl=Path(data.CACHE))
    live = SimpleNamespace(data=tmp_path / 'data' / 'cache', fl=tmp_path / 'form_lab' / 'cache')
    shutil.copytree(src.data, live.data, ignore=shutil.ignore_patterns('img', 'llm_log.csv'))
    shutil.copytree(src.fl, live.fl, ignore=shutil.ignore_patterns('.pytest_cache'))
    for mod, attr, value in ((fpl_api, 'CACHE_DIR', live.data), (data, 'CACHE', live.fl),
                             (build_live, 'OUT', live.data / 'live_pool.parquet'),
                             (core, 'BUILD_VIEWS', live.data / 'build_views.pkl'),
                             (refresh, 'STAMP', live.data / 'refresh.json'),
                             (refresh, 'STAGING', tmp_path / 'staging'),
                             (images, 'DIR', tmp_path / 'img')):
        monkeypatch.setattr(mod, attr, value)

    boot = json.loads((src.data / 'bootstrap-static.json').read_text(encoding='utf-8'))
    star = max(boot['elements'], key=lambda e: e['now_cost'])
    star['now_cost'] += 1                         # the visible change a swap brings in
    box = SimpleNamespace(live=live, star=star, calls=[], fail_at=None,
                          served={'bootstrap-static': boot})

    def fake(req, timeout=None):
        url = getattr(req, 'full_url', req)
        box.calls.append(url)
        if box.fail_at and len(box.calls) >= box.fail_at:
            raise OSError('network dropped mid-refresh')
        path = url.split('/api/', 1)[-1].strip('/') if '/api/' in url else None
        if path is None:
            raise OSError('no network in tests')
        if path in box.served:
            return io.BytesIO(json.dumps(box.served[path]).encode())
        name = path.replace('/', '_') + '.json'
        for d in (src.data, src.fl):
            if (d / name).exists():
                return io.BytesIO((d / name).read_bytes())
        return io.BytesIO(b'{"history": [], "fixtures": [], "history_past": []}')

    monkeypatch.setattr(urllib.request, 'urlopen', fake)
    refresher.reset()
    refresher.clear_caches()
    yield box
    refresher.wait(60)
    refresher.reset()
    refresher.clear_caches()


def snapshot(*dirs):
    return {str(f): hashlib.sha1(f.read_bytes()).hexdigest()
            for d in dirs for f in sorted(d.rglob('*')) if f.is_file()}


def run(**kw):
    return refresh.run(progress=QUIET, pause=0, refit=False, **kw)


def test_a_refresh_is_validated_then_swapped_in(sandbox):
    assert run() == 5
    boot = json.loads((sandbox.live.data / 'bootstrap-static.json').read_text(encoding='utf-8'))
    assert {e['code']: e['now_cost'] for e in boot['elements']}[sandbox.star['code']] \
        == sandbox.star['now_cost']
    pool = build_live.load()
    assert pool.loc[pool['code'] == sandbox.star['code'], 'price'].iloc[0] \
        == sandbox.star['now_cost']
    assert not any(refresh.STAGING.iterdir())


@pytest.mark.parametrize('failure', ['network', 'invalid'])
def test_a_failed_refresh_leaves_the_old_cache_untouched(sandbox, failure):
    before = snapshot(sandbox.live.data, sandbox.live.fl)
    if failure == 'network':
        sandbox.fail_at = 60                      # partway through the player histories
    else:
        sandbox.served['bootstrap-static'] = dict(sandbox.served['bootstrap-static'],
                                                  teams=[])
    with pytest.raises((OSError, refresh.Invalid)):
        run()
    assert snapshot(sandbox.live.data, sandbox.live.fl) == before
    assert not any(refresh.STAGING.iterdir())


def test_a_second_click_during_a_refresh_does_not_start_another(sandbox):
    gate, runs = threading.Event(), []

    def slow(progress):
        runs.append(1)
        progress('fetching players', 240, 667)
        gate.wait(10)
        return 6

    assert refresher.start(slow)[0]
    started, message = refresher.start(slow)
    assert not started and 'already running' in message
    assert refresher.status() == 'Refreshing: fetching players 240/667…'

    # another user clicking the button is told, not given a second refresh
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=600)
    at.run()
    at.button(key='refresh_wide').click()
    at.run()
    assert any('already running' in t.value for t in at.toast)

    gate.set()
    refresher.wait(10)
    assert runs == [1] and refresher.STATE['ok'] and refresher.STATE['gw'] == 6
    started, message = refresher.start(slow)       # the cooldown
    assert not started and 'recently' in message and runs == [1]


def test_a_failed_background_refresh_releases_the_lock(sandbox):
    def broken(progress):
        raise OSError('down')

    assert refresher.start(broken)[0]
    refresher.wait(10)
    assert refresher.STATE['ok'] is False and not refresher.STATE['running']
    assert refresher.start(lambda progress: 5)[0]  # no cooldown after a failure


def test_the_session_survives_a_refresh(sandbox):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=600)
    at.run()
    at.sidebar.text_input[0].input('3265946')
    at.sidebar.button[0].click()
    at.run()
    at.sidebar.text_input[1].input('user-session-key')
    at.radio(key='section').set_value('My Team')
    at.run()
    [chips] = [g for g in at.get('button_group') if g.key == 'team-pick']
    name = 'Haaland'                               # in team 3265946
    chips.set_value([name])
    at.run()
    assert not at.exception, at.exception
    panel = at.session_state['panel']

    assert refresher.start(lambda progress: run())[0]
    refresher.wait(120)
    assert refresher.STATE['ok'], 'refresh failed'
    at.run()                                       # the status line sees it finish
    assert not at.exception, at.exception

    assert at.session_state['entry'] == '3265946'
    assert at.radio(key='section').value == 'My Team'
    assert at.session_state[ai.KEY] == 'user-session-key'
    assert at.sidebar.text_input[1].value == 'user-session-key'
    assert at.session_state['panel'] == panel
    assert any('avatar-lg' in m.value and f'>{name}</div>' in m.value for m in at.markdown)
    assert any(t.value == 'Data updated to GW5' for t in at.toast)
