"""Advisory news sentiment for players in a suggested swap.

Headlines come from Google News RSS (no key, no search API), and an LLM
labels them. Everything is cached under form_lab/cache/, so a second run --
and the demo -- works offline. Nothing here feeds the projection: sentiment
is shown beside a swap, never inside its gain or its verdict.
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd

from form_lab import data

NEWS_DIR = data.CACHE / 'news'
VERDICT_DIR = data.CACHE / 'sentiment'
LABELS = data.CACHE.parent / 'sentiment_labels.csv'

RSS = 'https://news.google.com/rss/search'
WINDOW_DAYS = 7
MAX_HEADLINES = 8
RSS_SCAN = 40                # items read before the names-him filter
PAUSE = 1.0
TIMEOUT = 20
UA = 'Mozilla/5.0'

SENTIMENTS = ('positive', 'neutral', 'negative')
FLAGS = ('injury', 'rotation', 'suspension', 'transfer', 'role_change')
CONFIDENCE = ('low', 'med', 'high')

NEUTRAL = {'sentiment': 'neutral', 'flags': [], 'confidence': 'low',
           'summary': 'No recent news', 'evidence': []}

# prefix marking a verdict the model never actually gave -- a 429, a timeout
# or a missing key. These are not cached, so the next run asks again.
UNAVAILABLE = 'llm unavailable: '

JUDGE = (
    "You label Premier League news headlines about ONE named player. "
    "Use only the numbered headlines you are given. "
    "Never infer anything they do not say, and never mention another player. "
    "Reply with JSON only, no prose and no code fences, with exactly these "
    "keys: sentiment (positive, neutral or negative), flags (a list drawn "
    f"from {list(FLAGS)}, empty if none apply), confidence (low, med or "
    "high), summary (one sentence about this player), evidence (the numbers "
    "of the headlines you used)."
)


def _player(code):
    """(full name, club, web_name) from the live bootstrap."""
    import fpl_api
    boot = fpl_api.bootstrap()
    clubs = {t['id']: t['name'] for t in boot['teams']}
    for e in boot['elements']:
        if e['code'] == code:
            full = f"{e['first_name']} {e['second_name']}".strip()
            return full, clubs.get(e['team'], ''), e['web_name']
    raise KeyError(f'code {code} not in bootstrap')


def _surnames(full, web_name):
    """The name parts a headline about this player should contain."""
    parts = {web_name, full.split()[-1] if full else ''}
    parts |= set(re.split(r'[-\s]', full.split()[-1])) if full else set()
    return {p for p in parts if len(p) >= 3}


def mentions_player(title, full, web_name):
    """True if the headline names the player, not just his club."""
    low = title.lower()
    return any(re.search(rf'\b{re.escape(p.lower())}\b', low)
               for p in _surnames(full, web_name))


def _parse_rss(xml_text, max_headlines=MAX_HEADLINES):
    """(title, source, date) for each item, newest first."""
    root = ET.fromstring(xml_text)
    out, seen = [], set()
    for item in root.iterfind('./channel/item'):
        title = (item.findtext('title') or '').strip()
        # Google repeats the same rolling live-blog headline several times
        fingerprint = title.lower()
        if not title or fingerprint in seen:
            continue
        seen.add(fingerprint)

        src = item.find('source')
        out.append({
            'title': title,
            'source': (src.text or '').strip() if src is not None else '',
            'date': (item.findtext('pubDate') or '').strip(),
            'link': (item.findtext('link') or '').strip(),
        })
        if len(out) >= max_headlines:
            break
    return out


def cached_news(code):
    """The cached news file for a player, or None."""
    f = NEWS_DIR / f'{int(code)}.json'
    return json.loads(f.read_text(encoding='utf-8')) if f.exists() else None


def headlines(code, refresh=False, pause=PAUSE, gw=None):
    """Up to 8 headlines from the last 7 days, cached per player.

    A failed fetch keeps whatever was cached rather than overwriting it.
    """
    code = int(code)                    # codes arrive as numpy ints from frames
    f = NEWS_DIR / f'{code}.json'
    if f.exists() and not refresh:
        return json.loads(f.read_text(encoding='utf-8'))['headlines']

    full, _, web = _player(code)
    query = f'"{full}" when:{WINDOW_DAYS}d'
    url = f'{RSS}?{urllib.parse.urlencode({"q": query, "hl": "en-GB", "gl": "GB", "ceid": "GB:en"})}'
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            # filter before capping, so club roundups don't use up the eight
            items = _parse_rss(r.read().decode('utf-8', 'replace'),
                               max_headlines=RSS_SCAN)
    except Exception:
        old = cached_news(code)
        return old['headlines'] if old else []
    items = [h for h in items
             if mentions_player(h['title'], full, web)][:MAX_HEADLINES]

    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({
        'code': code, 'query': query, 'headlines': items, 'gw': gw,
        'fetched': datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
        encoding='utf-8')
    time.sleep(pause)
    return items


def _numbered(items):
    return '\n'.join(f'{i}. {h["title"]} ({h["source"]}, {h["date"]})'
                     for i, h in enumerate(items, 1))


def _other_names(subject_names):
    """Other players' surnames, for the no-other-player check.

    web_name only -- pulling in first names too rejected summaries over a
    common given name that happened to be someone else's web_name.
    """
    import fpl_api
    return {e['web_name'] for e in fpl_api.bootstrap()['elements']
            if e['web_name'] and len(e['web_name']) >= 4
            and e['web_name'] not in subject_names}


def validate(raw, n_headlines, subject_names=()):
    """(verdict, reason). Anything that fails comes back as the neutral verdict."""
    text = re.sub(r'^```(?:json)?|```$', '', str(raw).strip(),
                  flags=re.MULTILINE).strip()
    try:
        v = json.loads(text)
    except json.JSONDecodeError:
        return dict(NEUTRAL), 'not JSON'
    if not isinstance(v, dict):
        return dict(NEUTRAL), 'not an object'

    if v.get('sentiment') not in SENTIMENTS:
        return dict(NEUTRAL), f'bad sentiment {v.get("sentiment")!r}'
    flags = v.get('flags') or []
    if not isinstance(flags, list) or any(f not in FLAGS for f in flags):
        return dict(NEUTRAL), f'bad flags {flags!r}'
    if v.get('confidence') not in CONFIDENCE:
        return dict(NEUTRAL), f'bad confidence {v.get("confidence")!r}'

    evidence = v.get('evidence') or []
    if not isinstance(evidence, list):
        return dict(NEUTRAL), 'evidence is not a list'
    for e in evidence:
        if not isinstance(e, int) or not 1 <= e <= n_headlines:
            return dict(NEUTRAL), f'evidence {e!r} is not a headline number'

    summary = str(v.get('summary') or '').strip()
    if not summary:
        return dict(NEUTRAL), 'empty summary'
    for other in _other_names(set(subject_names)):
        if re.search(rf'\b{re.escape(other)}\b', summary):
            return dict(NEUTRAL), f'summary mentions {other}'

    return {'sentiment': v['sentiment'], 'flags': list(flags),
            'confidence': v['confidence'], 'summary': summary,
            'evidence': list(evidence)}, 'ok'


def _chat(system, facts, model=None, **call):
    """(text, error) from llm_explain's OpenRouter client, with our prompt.

    Key handling, model fallbacks, timeout and 429 backoff all come from
    llm_explain unless `call` overrides them; only the system prompt differs.
    """
    import llm_explain as lx

    over = {'model': model, 'models': None} if model else {}
    text, err, _ = lx.call_openrouter(facts, system=system, **over, **call)
    return text, err


def judge(items, subject_names=(), model=None, **call):
    """(verdict, reason) for a list of headlines. Never raises."""
    if not items:
        return dict(NEUTRAL), 'no headlines'

    subject = next(iter(subject_names), 'the player')
    facts = {'player': subject,
             'headlines': [{'n': i, 'title': h['title'],
                            'source': h['source'], 'date': h['date']}
                           for i, h in enumerate(items, 1)]}
    text, err = _chat(JUDGE, facts, model, **call)
    if text is None:
        # marked so assess() does not cache a 429 as though it were a verdict
        return dict(NEUTRAL), f'{UNAVAILABLE}{err}'
    return validate(text, len(items), subject_names)


def assess(code, refresh=False, model=None):
    """Cached verdict for one player. Offline once cached."""
    code = int(code)
    f = VERDICT_DIR / f'{code}.json'
    if f.exists() and not refresh:
        return json.loads(f.read_text(encoding='utf-8'))['verdict']

    items = headlines(code)
    full, _, web = _player(code)
    verdict, reason = judge(items, (web, full), model)

    if reason.startswith(UNAVAILABLE):
        return verdict                  # transient, so leave the cache empty

    VERDICT_DIR.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({'code': code, 'verdict': verdict,
                             'reason': reason, 'n_headlines': len(items)},
                            ensure_ascii=False), encoding='utf-8')
    return verdict


def write_template(codes, path=LABELS):
    """A CSV with the headlines filled in and the label columns blank."""
    rows = []
    for code in codes:
        items = headlines(code)
        full, club, web = _player(code)
        rows.append({
            'code': code, 'web_name': web, 'full_name': full, 'club': club,
            'n_headlines': len(items),
            'headlines': ' || '.join(f'{i}. {h["title"]}'
                                     for i, h in enumerate(items, 1)),
            'sentiment': '', 'flags': '',
        })
    d = pd.DataFrame(rows)
    d.to_csv(path, index=False, encoding='utf-8')
    print(f'wrote {path} ({len(d)} players, '
          f'{int((d["n_headlines"] > 0).sum())} with headlines)')
    print('fill in `sentiment` (positive/neutral/negative) and `flags` '
          '(semicolon separated, blank for none)')
    return d


def agreement(path=LABELS, model=None):
    """LLM-versus-you agreement on sentiment and on flags."""
    if not path.exists():
        raise FileNotFoundError(f'{path} not found -- run --template first')
    d = pd.read_csv(path)
    done = d[d['sentiment'].astype(str).str.strip() != '']
    if done.empty:
        print(f'{path} has no filled-in labels yet')
        return pd.DataFrame()

    rows = []
    for _, r in done.iterrows():
        v = assess(int(r['code']), model=model)
        mine = set(str(r['flags'] or '').replace(',', ';').split(';')) - {''}
        theirs = set(v['flags'])
        rows.append({
            'web_name': r['web_name'],
            'mine': str(r['sentiment']).strip().lower(),
            'llm': v['sentiment'],
            'my_flags': ';'.join(sorted(mine)),
            'llm_flags': ';'.join(sorted(theirs)),
            'flag_jaccard': (len(mine & theirs) / len(mine | theirs)
                             if (mine | theirs) else 1.0),
        })
    out = pd.DataFrame(rows)
    out['match'] = out['mine'] == out['llm']

    print(out.to_string(index=False))
    print(f'\nlabelled: {len(out)}')
    print(f'sentiment agreement: {100 * out["match"].mean():.1f}%')
    print(f'mean flag overlap  : {100 * out["flag_jaccard"].mean():.1f}%')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--template', type=int, metavar='ENTRY_ID',
                    help='build the labelling CSV from this team and its candidates')
    ap.add_argument('--n', type=int, default=20, help='players in the template')
    ap.add_argument('--agreement', action='store_true',
                    help='score the filled-in CSV against the LLM')
    ap.add_argument('--paid', action='store_true',
                    help='judge with llm_explain.PAID_MODEL instead of the free models')
    args = ap.parse_args()

    paid = None
    if args.paid:
        import llm_explain as lx
        paid = lx.PAID_MODEL
        print(f'--paid: billing {paid}')

    if args.agreement:
        agreement(model=paid)
    elif args.template:
        from form_lab import eval as ev
        import build_live as bl
        import fpl_api as fa
        pool = bl.load()
        team, _ = fa.load_team(args.template)
        extra = [c for c in ev.demo_codes(pool, team) if c not in team]
        write_template((list(team) + extra)[:args.n])
    else:
        ap.print_help()
