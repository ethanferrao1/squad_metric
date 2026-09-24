"""Bring every cached input up to date for the current gameweek.

    python refresh.py [--images]

Also run by the app's Refresh data button (refresher.py), in a thread.
Updates bootstrap-static, the fixture list, the live pool (prices,
availability, positions, new signings), the gameweek history of every
buyable player and every player in a cached team, and the Build tab's season
replay, then stamps the gameweek and time into data/cache/refresh.json.

Nothing live is touched until the end: everything is written to a staging
folder, validated as a whole, and only then moved over the live cache file
by file (each move an atomic rename). Any failure leaves the old cache as it
was. The model is refitted only when every historical season is present in
data/; otherwise the fitted predictions are kept (build_live.refresh_live).
"""
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import build_live
import data_io as io
import fpl_api as api
import rater as rt
from form_lab import data

STAMP = api.CACHE_DIR / 'refresh.json'
STAGING = io.DATA_DIR.parent / '.refresh-staging'
# swapped last, in this order, so a reader never sees a new gameweek before its data
LAST = ('live_pool.parquet', 'build_views.pkl', 'bootstrap-static.json', 'refresh.json')


class Invalid(RuntimeError):
    """The staged cache failed validation; nothing was swapped in."""


def stamp():
    """{'gw', 'refreshed'} from the last refresh, or None if never run."""
    return json.loads(STAMP.read_text()) if STAMP.exists() else None


def team_players():
    """Codes of every player in every cached team, as last cached."""
    import re
    codes = set()
    for f in api.CACHE_DIR.glob('entry_*_event_*_picks.json'):
        m = re.match(r'entry_(\d+)_event_(\d+)_picks', f.stem)
        if m:
            codes |= set(api.load_team(int(m.group(1)), int(m.group(2)))[0])
    return codes


def can_refit():
    """True if every season the model trains on is in data/."""
    need = ([f'merged_gw_{s}.csv' for s in io.FEATURE_SEASONS]
            + [f'players_raw_20{s}.csv' for s in io.SEASONS])
    return all((io.DATA_DIR / n).exists() for n in need)


def _say(phase, done=None, total=None):
    print(f'{phase} {done}/{total}' if total else phase, flush=True)


def build(roots, progress=_say, pause=data.PAUSE, refit=None):
    """Write a complete refreshed cache into the staging `roots`. Returns (gw, player ids)."""
    stage = roots[api.CACHE_DIR]
    progress('fetching the player list')
    boot = api.bootstrap(refresh=True)
    gw = api.current_gw(boot)
    progress('fetching fixtures')
    data.fixtures(refresh=True)

    progress('rebuilding the player pool')
    pool_path = stage / build_live.OUT.name
    if can_refit() if refit is None else refit:
        pool = build_live.build(pool_path)
    else:
        pool = build_live.refresh_live(build_live.load(), pool_path)

    ids = {e['code']: e['id'] for e in boot['elements']}
    wanted = set(rt.available(rt._modelled(pool))['code']) | team_players()
    players = sorted(ids[c] for c in wanted if c in ids)
    for i, pid in enumerate(players, 1):
        data.element_summary(pid, pause, refresh=True)
        progress('fetching players', i, len(players))

    progress('replaying the season')
    import core
    pd.to_pickle(core.compute_build_views(pool, offline=True), stage / 'build_views.pkl')
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    (stage / STAMP.name).write_text(json.dumps({'gw': gw, 'refreshed': now}))
    return gw, players


def validate(roots, gw, players):
    """Raise Invalid unless the staged cache is complete and consistent with `gw`."""
    stage, fl = roots[api.CACHE_DIR], roots[data.CACHE]

    def need(ok, what):
        if not ok:
            raise Invalid(what)

    boot = json.loads((stage / 'bootstrap-static.json').read_text(encoding='utf-8'))
    need(len(boot.get('elements', [])) >= 400 and len(boot.get('teams', [])) == 20
         and boot.get('events'), 'bootstrap is incomplete')
    need(api.current_gw(boot) == gw, 'bootstrap gameweek changed during the refresh')

    fixtures = json.loads((fl / 'fixtures.json').read_text(encoding='utf-8'))
    need(isinstance(fixtures, list) and len(fixtures) >= 300, 'fixture list is incomplete')
    need(max((f.get('event') or 0) for f in fixtures) >= min(gw, 38),
         'fixtures end before the current gameweek')

    pool = pd.read_parquet(stage / build_live.OUT.name)
    need({'code', 'element_type', 'team_code', 'price', 'pred', 'modelled',
          'web_name', 'chance'} <= set(pool.columns), 'pool is missing columns')
    need(len(pool) >= 400 and pool.loc[pool['modelled'], 'pred'].notna().all(),
         'pool is incomplete')
    codes = {e['code'] for e in boot['elements']}
    need(pool['code'].isin(codes).mean() >= 0.9, 'pool does not match the bootstrap')

    missing = [p for p in players if not (fl / f'element-summary_{p}.json').exists()]
    need(not missing, f'{len(missing)} player histories missing')

    views = pd.read_pickle(stage / 'build_views.pkl')
    need(views is None or views['gw'] <= gw, 'season replay is ahead of the bootstrap')
    need(json.loads((stage / STAMP.name).read_text())['gw'] == gw, 'stamp gameweek mismatch')


def swap(roots):
    """Move every staged file over its live counterpart, the LAST files last."""
    moves = [(f, live / f.relative_to(stage))
             for live, stage in roots.items() for f in stage.rglob('*') if f.is_file()]
    moves.sort(key=lambda m: LAST.index(m[0].name) if m[0].name in LAST else -1)
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)


def run(progress=_say, pause=data.PAUSE, refit=None):
    """Refresh into staging, validate, swap in. Returns the new gameweek.

    Raises on any failure, with the live cache untouched.
    """
    STAGING.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='run-', dir=STAGING))
    roots = {api.CACHE_DIR: tmp / 'data', data.CACHE: tmp / 'form_lab'}
    try:
        api.set_staging(roots)
        try:
            gw, players = build(roots, progress, pause, refit)
        finally:
            api.set_staging(None)
        progress('checking the new data')
        validate(roots, gw, players)
        progress('swapping in')
        swap(roots)
        return gw
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(images=False):
    start = time.time()
    gw = run()
    if images:
        import images as im
        fetched, missing = im.fetch_all()
        print(f'images: {fetched} new, {missing} unavailable (placeholder shown)')
    print(f'\nrefreshed for GW{gw} ({time.time() - start:.0f}s)')


if __name__ == '__main__':
    main(images='--images' in sys.argv[1:])
