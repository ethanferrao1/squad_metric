"""Read a real FPL team from the public API.

Every response is cached as JSON under data/cache/, so once a team has been
fetched the demo runs with no network at all. Pass refresh=True to re-fetch.

While refresh.py builds a new cache, its thread sees a staging overlay:
writes go to the staging copy and reads look there first, then in the live
cache. Every other thread keeps reading the live cache untouched.
"""
import json
import threading
import urllib.request

import data_io as io

BASE = 'https://fantasy.premierleague.com/api'
CACHE_DIR = io.DATA_DIR / 'cache'
UA = 'Mozilla/5.0'          # the API 403s an unset user-agent
TIMEOUT = 30

_local = threading.local()


def set_staging(roots):
    """{live cache dir: staging dir} for this thread only; None to stop."""
    _local.roots = roots or {}


def overlay(root):
    """Where this thread writes cache files for `root`."""
    return getattr(_local, 'roots', {}).get(root, root)


def cached_file(root, name):
    """The file to read for `name` under `root`, staged copy first; None if absent."""
    for d in dict.fromkeys((overlay(root), root)):
        if (d / name).exists():
            return d / name
    return None


def _name(path):
    return path.strip('/').replace('/', '_') + '.json'


def _cache_path(path):
    return CACHE_DIR / _name(path)


def _get(path, refresh=False):
    """GET `path` under the API, via the on-disk cache.

    A cached copy is used unless `refresh` is set; if the network is
    unavailable and nothing is cached, the underlying URLError propagates.
    """
    f = cached_file(CACHE_DIR, _name(path))
    if f and not refresh:
        return json.loads(f.read_text(encoding='utf-8'))

    req = urllib.request.Request(f'{BASE}/{path.strip("/")}/',
                                 headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = json.load(r)

    out = overlay(CACHE_DIR) / _name(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data), encoding='utf-8')
    return data


def bootstrap(refresh=False):
    return _get('bootstrap-static', refresh)


def picks(entry_id, gw, refresh=False):
    """The raw picks for one entry and gameweek: slot, captain and vice flags."""
    return _get(f'entry/{entry_id}/event/{gw}/picks', refresh)


def cached_event(entry_id):
    """The latest gameweek whose picks for this entry are on disk, or None."""
    import re
    events = [int(m.group(1)) for f in CACHE_DIR.glob(f'entry_{entry_id}_event_*_picks.json')
              if (m := re.match(rf'entry_{entry_id}_event_(\d+)_picks', f.stem))]
    return max(events) if events else None


def current_gw(boot=None):
    """The gameweek in progress, falling back to the last finished one."""
    boot = boot if boot is not None else bootstrap()
    cur = [e['id'] for e in boot['events'] if e.get('is_current')]
    if cur:
        return cur[0]
    done = [e['id'] for e in boot['events'] if e.get('finished')]
    return max(done) if done else 1


def element_codes(boot=None):
    """FPL element id -> the stable `code` the pipeline keys on. The id is
    per-season and gets reused; the code follows the player for his career."""
    boot = boot if boot is not None else bootstrap()
    return {e['id']: e['code'] for e in boot['elements']}


def load_team(entry_id, gw=None, refresh=False):
    """The 15 picks and the bank for one FPL entry.

    Returns (codes, bank): player `code`s in pick order, and money in the
    bank in tenths of a million.
    """
    boot = bootstrap(refresh)
    gw = gw if gw is not None else current_gw(boot)

    picks_path = f'entry/{entry_id}/event/{gw}/picks'
    # new picks mean a new gameweek, so the cached bank is stale too
    new_week = refresh or not _cache_path(picks_path).exists()
    picks = _get(picks_path, refresh)
    lut = element_codes(boot)
    codes = [lut[p['element']] for p in picks['picks']
             if p['element'] in lut]

    hist = _get(f'entry/{entry_id}/history', new_week)
    row = next((h for h in hist['current'] if h['event'] == gw), None)
    if row is None:
        row = hist['current'][-1] if hist['current'] else {}
    bank = row.get('bank', picks.get('entry_history', {}).get('bank', 0))

    return codes, bank
