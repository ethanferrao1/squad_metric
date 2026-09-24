"""LLM-written reasons for a suggested transfer.

The model never sees the pool, the pipeline, or free text -- only a small
facts dict built here. Everything it writes is then checked back against
those facts, and anything that fails falls back to the template. So the
worst case is the wording we already had, never a fabricated stat.

Bring your own key: the OpenRouter key is passed in as `api_key` (the app
keeps it in the browser session only). It is never written to disk, a log, a
cache or an error string. OPENROUTER_API_KEY in the environment or .env is
ignored unless ALLOW_ENV_KEY=1 (for local scripts such as pregenerate.py).
Without a key, or offline, every call returns the template.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

import pandas as pd

import data_io as io

MASK = '***'
# anything shaped like an OpenRouter or other bearer key, for redaction
KEY_SHAPE = re.compile(r'sk-[A-Za-z0-9_-]{8,}')

MODEL = 'google/gemma-4-31b-it:free'

MODEL = 'google/gemma-4-31b-it:free'

# Handed to OpenRouter as its `models` list, so one request can move down the
# list when the primary is rate limited or down. General instruction-tuned
# models only: a reasoning-FIRST model (glm-5.2, inkling, the nemotron
# reasoning line) would spend the 160-token budget thinking, and the stray
# figures in a thinking trace are exactly what fact_check rejects.
FALLBACKS = [
    'qwen/qwen3.8-27b:free',
    'google/gemma-4-26b-a4b-it:free',
]

# Only used by `python pregenerate.py --paid`. A general instruct model with
# no thinking mode, at $0.05/M in and $0.08/M out -- well under the 0.50
# ceiling, and a few hundred explanations cost a fraction of a cent. Nothing
# reaches this unless --paid is passed.
PAID_MODEL = 'mistralai/mistral-small-24b-instruct-2501'

URL = 'https://openrouter.ai/api/v1/chat/completions'
TIMEOUT = 20

# Seconds to wait after each 429 before trying again. Four attempts in all;
# the free tier refuses in bursts, so backing off is usually enough.
RETRY_BACKOFF = (5, 15, 45)

CACHE = io.DATA_DIR / 'cache' / 'explanations.json'

SYSTEM = (
    "You are a calm, evidence-led Fantasy Premier League analyst. "
    "Explain the suggested transfer in at most two sentences. "
    "Use ONLY the facts in the JSON you are given. "
    "Never invent statistics, fixtures, opponents, form or injury details. "
    "Every number you write must appear in the facts. "
    "Mention no player other than the two named. "
    "No hype, no exclamation marks, no advice beyond the facts."
)


# ---------------------------------------------------------------- facts

def _player_facts(name, pos, price_tenths, pred, no_history, drivers, boot_row):
    f = {
        'name': name,
        'position': pos,
        'price_m': round(price_tenths / 10, 1),
        'predicted_points': round(float(pred), 1),
        'no_pl_history': bool(no_history),
    }
    if drivers:
        f['top_drivers'] = [
            {'reason': d['feature'], 'value': d['value'],
             'points_contribution': d['shap']}
            for d in drivers
        ]
    if boot_row:
        news = (boot_row.get('news') or '').strip()
        if news:
            f['injury_news'] = news
        chance = boot_row.get('chance_of_playing_next_round')
        if chance is not None:
            f['chance_of_playing_next_round'] = chance
    return f


def _boot_index():
    """code -> the bootstrap row, from the cached API response only."""
    try:
        import fpl_api
        boot = fpl_api.bootstrap()
    except Exception:
        return {}
    return {e['code']: e for e in boot['elements']}


def build_facts(row, pool, boot=None):
    """The complete input to the model. Nothing else is sent."""
    import build_live as bl

    boot = _boot_index() if boot is None else boot
    by_name = pool.drop_duplicates('web_name').set_index('web_name')

    def side(name, pred):
        if name not in by_name.index:
            return {'name': name, 'predicted_points': round(float(pred), 1)}
        p = by_name.loc[name]
        return _player_facts(
            name, p['pos'], p['price'], pred,
            not bool(p.get('modelled', True)),
            bl.player_drivers(pool, p['code']),
            boot.get(p['code']))

    return {
        'transfer_out': side(row['out'], row['out_pred']),
        'transfer_in': side(row['in'], row['in_pred']),
        'predicted_points_gained': round(float(row['gain']), 1),
        'price_difference_m': round(float(row['cost']) / 10, 1),
    }


# ---------------------------------------------------------------- fact check

_NUM = re.compile(r'-?\d+(?:\.\d+)?')


def _numbers_in(obj, acc=None):
    acc = [] if acc is None else acc
    if isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in(v, acc)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, str):
        acc += [float(m) for m in _NUM.findall(obj)]
    return acc


def fact_check(text, facts, pool=None):
    """(ok, reason). Every number must be in the facts to 1dp, and no player
    other than the two named may be mentioned."""
    allowed = _numbers_in(facts)
    for tok in _NUM.findall(text):
        x = float(tok)
        if not any(abs(x - a) <= 0.051 or round(x, 1) == round(a, 1)
                   for a in allowed):
            return False, f'number {tok} is not in the facts'

    if pool is not None:
        named = {facts['transfer_out']['name'], facts['transfer_in']['name']}
        for other in pool['web_name'].dropna().unique():
            if other in named or len(str(other)) < 4:
                continue
            if re.search(rf'\b{re.escape(str(other))}\b', text):
                return False, f'mentions another player: {other}'
    return True, 'ok'


# ---------------------------------------------------------------- cache

def _load_cache():
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(c):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(c, indent=1, ensure_ascii=False),
                     encoding='utf-8')


def cache_key(gw, out, name_in):
    return f'{gw}|{out}|{name_in}'


# ---------------------------------------------------------------- the key

REJECTED = 'key rejected (401)'


def env_key():
    """OPENROUTER_API_KEY from the environment or .env, only if ALLOW_ENV_KEY=1."""
    if os.environ.get('ALLOW_ENV_KEY') != '1':
        return None
    try:
        from dotenv import load_dotenv
        load_dotenv(io.DATA_DIR.parent / '.env')
    except ImportError:
        pass
    return os.environ.get('OPENROUTER_API_KEY') or None


def redact(text, key=None):
    """`text` with the key, and anything shaped like a key, masked."""
    text = str(text)
    if key:
        text = text.replace(key, MASK)
    return KEY_SHAPE.sub(MASK, text)


KEY_URL = 'https://openrouter.ai/api/v1/key'


def check_key(api_key, timeout=8):
    """True if OpenRouter accepts the key, False if it rejects it, None if it
    could not be checked (offline, timeout). Costs nothing; never raises."""
    req = urllib.request.Request(KEY_URL, headers={
        'Authorization': f'Bearer {(api_key or "").strip()}'})
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except urllib.error.HTTPError as e:
        return False if e.code in (401, 403) else None
    except Exception:
        return None


# ---------------------------------------------------------------- the call

def call_openrouter(facts, model=MODEL, models=FALLBACKS, timeout=TIMEOUT,
                    backoff=RETRY_BACKOFF, system=SYSTEM, usage=None,
                    api_key=None):
    """(text, error, model_used). Never raises: a 429, a timeout or a missing
    key all come back as an error string for the caller to fall back on. A
    401 comes back as REJECTED. The error string never contains the key.

    `api_key` is the caller's key; without one, env_key() (opt-in only).

    Two separate defences against the free tier's rate limiting. `models` is
    OpenRouter's own fallback routing -- one request, and it moves down the
    list if the primary refuses. `backoff` then retries the whole request
    after each 429, waiting the given seconds between attempts. Only when
    every model and every retry has refused does this give up and let the
    caller fall back to the template.

    `model_used` is what OpenRouter reports actually served the request,
    which with a fallback list is not necessarily `model`.

    `system` defaults to the transfer-explanation prompt; another caller can
    pass its own and reuse the key handling, fallbacks and backoff.
    """
    key = (api_key or '').strip() or env_key()
    if not key:
        return None, 'no API key', None

    payload = {
        'model': model,
        'messages': [{'role': 'system', 'content': system},
                     {'role': 'user', 'content': json.dumps(facts)}],
        'temperature': 0.2,
        'max_tokens': 160,
        # every current free model ships a configurable thinking mode. Left
        # on, the trace eats the token budget before the answer starts and
        # leaks numbers the facts never contained.
        'reasoning': {'enabled': False},
    }
    if models:
        payload['models'] = [model] + [m for m in models if m != model]
    body = json.dumps(payload).encode()

    # one attempt per wait, plus a final attempt with nothing left to wait for
    for wait in (*backoff, None):
        req = urllib.request.Request(
            URL, data=body,
            headers={'Authorization': f'Bearer {key}',
                     'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                return None, REJECTED, None
            if e.code != 429:
                return None, f'http {e.code}', None
            if wait is None:
                return None, (f'rate limited (429) after {len(backoff)} '
                              f'retries'), None
            time.sleep(wait)
            continue
        except Exception as e:                  # timeout, DNS, offline
            return None, redact(f'{type(e).__name__}: {e}', key), None

        try:
            text = data['choices'][0]['message']['content'].strip()
        except (KeyError, IndexError):
            return None, 'malformed response', None
        if usage is not None:
            usage.update(data.get('usage') or {})
        return text, None, data.get('model') or model


# ---------------------------------------------------------------- entry point

def explain(row, pool, template, gw=None, use_cache=True, model=MODEL,
            models=FALLBACKS, api_key=None):
    """(text, source, detail). `source` is 'llm', 'cache' or 'template'.

    `detail` says which model wrote it when `source` is 'llm' or 'cache', and
    why the model was not used when it is 'template' -- a 429 that outlasted
    the retries reads differently from a reply that failed the fact check,
    and that distinction is the whole diagnostic when a pregenerate run comes
    back all template.

    `template` is the fallback string, passed in so this module never has to
    know how the template is worded.
    """
    if gw is None:
        try:
            import fpl_api
            gw = fpl_api.current_gw()
        except Exception:
            gw = 0

    key = cache_key(gw, row['out'], row['in'])
    cache = _load_cache() if use_cache else {}
    if key in cache:
        # 'cache' rather than 'llm' so a rerun is visibly free, not a fresh call
        return cache[key]['text'], 'cache', cache[key].get('model')

    facts = build_facts(row, pool)
    text, err, used = call_openrouter(facts, model=model, models=models,
                                      api_key=api_key)
    if text is None:
        return template, 'template', err

    ok, reason = fact_check(text, facts, pool)
    if not ok:
        return template, 'template', f'failed fact check: {reason}'

    if use_cache:
        cache[key] = {'text': text, 'source': 'llm', 'model': used,
                      'fact_check': reason}
        _save_cache(cache)
    return text, 'llm', used
