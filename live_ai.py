"""Live AI for the loaded team: transfer explanations and news ratings.

Advisory only: nothing here changes a suggestion. Calls run 5 at a time with
a 15 s timeout on PAID_MODEL. A failure falls back to a same-gameweek cached
result, then to the template (explanations) or the headlines alone (news).
Every call is logged with tokens and estimated cost -- never the key -- and a
session makes at most MAX_CALLS. The key is the caller's, passed to resolve()
as an argument; with none, nothing is called and every job falls back. SQUADMETRIC_FREE_AI=1 uses the free models,
for testing at no cost.
"""
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import fpl_api as api
import llm_explain as lx
from form_lab import form, sentiment as fs

WORKERS, TIMEOUT, MAX_CALLS = 5, 15, 200
PRICE_IN, PRICE_OUT = 0.05e-6, 0.08e-6          # PAID_MODEL, dollars per token
EST_TOKENS = {'explain': (900, 90), 'news': (700, 130)}
NEWS_PAUSE = 0.3
LOG = api.CACHE_DIR / 'llm_log.csv'
CACHE = api.CACHE_DIR / 'explanations_live.json'
LOG_COLUMNS = ['time', 'gw', 'kind', 'subject', 'model', 'prompt_tokens',
               'completion_tokens', 'cost_usd', 'latency_s', 'outcome']

EXPLAIN = (
    "You explain one Fantasy Premier League transfer to its manager in at most "
    "two sentences. Lead with why the outgoing player is being replaced: turn his "
    "top_drivers into plain-English reasons, and use his recent form and any "
    "injury_news. Then give the gain in one short clause: "
    "predicted_points_gained_rounded points over horizon_gameweeks gameweeks, or, "
    "if part_of_plan is present, the plan's plan_points_gained_rounded. Write "
    "prices as £Xm from price_m; if same_price is true say 'at the same price', "
    "otherwise use price_difference_abs_m. Whole-number points only. Use only the "
    "facts given, invent nothing, and mention no player other than the two named."
)


def free_mode():
    return os.environ.get('SQUADMETRIC_FREE_AI') == '1'


def model_kw():
    """The model for live calls: PAID_MODEL, or the free list when testing."""
    if free_mode():
        return {'model': lx.MODEL, 'models': lx.FALLBACKS}
    return {'model': lx.PAID_MODEL, 'models': None}


def cost(model, usage):
    """Estimated dollars for one call; free models cost nothing."""
    if not usage or ':free' in str(model):
        return 0.0
    return (usage.get('prompt_tokens', 0) * PRICE_IN
            + usage.get('completion_tokens', 0) * PRICE_OUT)


def estimate(jobs):
    """(calls, dollars) a list of jobs would cost if none were cached."""
    dollars = sum(EST_TOKENS[j['kind']][0] * PRICE_IN + EST_TOKENS[j['kind']][1] * PRICE_OUT
                  for j in jobs)
    return len(jobs), dollars


def explain_facts(row, pool, proj, plan=None):
    """llm_explain's facts plus what the live prompt needs, all checkable."""
    facts = lx.build_facts(row, pool)
    form_ppg = proj.drop_duplicates('web_name').set_index('web_name').get('form_ppg')
    for side, name in (('transfer_out', row['out']), ('transfer_in', row['in'])):
        if form_ppg is not None and name in form_ppg.index and form_ppg[name] == form_ppg[name]:
            facts[side]['recent_form_points_per_game'] = round(float(form_ppg[name]), 1)
    facts['predicted_points_gained_rounded'] = int(round(float(row['gain'])))
    facts['price_difference_abs_m'] = abs(round(float(row['cost']) / 10, 1))
    facts['same_price'] = int(row['cost']) == 0
    facts['horizon_gameweeks'] = form.SWAP_HORIZON
    if plan:
        facts['part_of_plan'] = {'transfers': plan['transfers'],
                                 'plan_points_gained_rounded': int(round(plan['net']))}
    return facts


def _read_cache():
    try:
        return json.loads(CACHE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def _log(gw, job, model, usage, latency, outcome):
    new = not LOG.exists()
    with LOG.open('a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_COLUMNS)
        w.writerow([datetime.now(timezone.utc).isoformat(timespec='seconds'), gw,
                    job['kind'], job['subject'], model or '',
                    usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0),
                    f'{cost(model, usage):.8f}', f'{latency:.2f}', lx.redact(outcome)])


def _cached(job, gw):
    if job['kind'] == 'explain':
        hit = _read_cache().get(job['key'])
        return {'text': hit['text'], 'label': 'cached'} if hit else None
    news = fs.cached_news(job['code'])
    f = fs.VERDICT_DIR / f"{job['code']}.json"
    if news and news.get('gw') == gw and f.exists():
        v = json.loads(f.read_text(encoding='utf-8'))
        if v.get('gw') == gw:
            return {'verdict': v['verdict'], 'headlines': news['headlines'], 'label': 'cached'}
    return None


def _fallback(job):
    if job['kind'] == 'explain':
        return {'text': job['template'], 'label': 'template'}
    news = fs.cached_news(job['code'])
    return {'verdict': None, 'headlines': news['headlines'] if news else [],
            'label': 'headlines only'}


def _work(job, gw, api_key):
    """The network part of a job; runs in a worker thread and writes nothing shared."""
    usage, start = {}, time.time()
    call = dict(timeout=TIMEOUT, backoff=(), usage=usage, api_key=api_key)
    if job['kind'] == 'explain':
        text, err, used = lx.call_openrouter(job['facts'], system=EXPLAIN,
                                             **model_kw(), **call)
        ok = bool(text) and lx.fact_check(text, job['facts'], job['pool'])[0]
        return {'text': text if ok else None, 'model': used, 'usage': usage,
                'latency': time.time() - start, 'called': True,
                'outcome': 'ok' if ok else (err or 'failed fact check')}
    items = fs.headlines(job['code'], refresh=True, pause=NEWS_PAUSE, gw=gw)
    kw = model_kw()
    verdict, reason = fs.judge(items, job['names'], kw['model'] if kw['models'] is None
                               else None, **call)
    return {'verdict': verdict, 'headlines': items, 'reason': reason,
            'model': kw['model'], 'usage': usage, 'latency': time.time() - start,
            'called': bool(items), 'outcome': reason}


def _finish(job, res, gw):
    """Cache and log a finished call (main thread); returns what to show."""
    if res.get('called'):
        _log(gw, job, res['model'], res['usage'], res['latency'], res['outcome'])
    if job['kind'] == 'explain':
        if not res.get('text'):
            return _fallback(job)
        cache = _read_cache()
        cache[job['key']] = {'text': res['text'], 'model': res['model']}
        CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding='utf-8')
        return {'text': res['text'], 'label': 'live'}
    if res['reason'].startswith(fs.UNAVAILABLE):
        return {'verdict': None, 'headlines': res['headlines'], 'label': 'headlines only'}
    fs.VERDICT_DIR.mkdir(parents=True, exist_ok=True)
    (fs.VERDICT_DIR / f"{job['code']}.json").write_text(json.dumps(
        {'code': job['code'], 'verdict': res['verdict'], 'reason': res['reason'],
         'n_headlines': len(res['headlines']), 'gw': gw}, ensure_ascii=False),
        encoding='utf-8')
    return {'verdict': res['verdict'], 'headlines': res['headlines'], 'label': 'live'}


def resolve(jobs, on_result, budget, gw, live=True, memo=None, api_key=None):
    """Answer every job: session memo, same-GW cache, live call, else a fallback.

    `on_result(key, result)` is called on this thread as each one lands, so a
    page can fill its placeholders as results arrive. `budget` is a dict with
    'used', shared across a session. Live calls need `api_key`; once OpenRouter
    rejects it, budget['rejected'] is set and nothing more is called.
    """
    memo = {} if memo is None else memo
    live = live and bool(api_key) and not budget.get('rejected')
    pending = []
    for job in jobs:
        hit = memo.get(job['key']) or _cached(job, gw)
        if hit:
            memo[job['key']] = hit
            on_result(job['key'], hit)
        elif not live or budget['used'] >= MAX_CALLS:
            on_result(job['key'], _fallback(job))
        else:
            budget['used'] += 1
            pending.append(job)
    if not pending:
        return
    with ThreadPoolExecutor(WORKERS) as ex:
        futures = {ex.submit(_work, job, gw, api_key): job for job in pending}
        for fut in as_completed(futures):
            job = futures[fut]
            rejected = False
            try:
                res = fut.result()
                rejected = lx.REJECTED in str(res.get('outcome', ''))
                result = _finish(job, res, gw)
            except Exception:
                result = _fallback(job)
            if rejected:
                budget['rejected'] = True     # not memoised: a new key asks again
            else:
                memo[job['key']] = result
            on_result(job['key'], result)
