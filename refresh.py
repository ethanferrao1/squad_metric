"""Bring every cached input up to date for the current gameweek.

    python refresh.py

Run before a demo. Updates bootstrap-static, the fixture list, the live pool
(model predictions, today's prices, availability, new signings), the
gameweek history of every buyable player and every player in a cached team,
and any player photo, club badge or shirt not yet cached. Then stamps the
gameweek and time into data/cache/refresh.json for the app header.
"""
import json
import time
from datetime import datetime, timezone

import build_live
import fpl_api as api
import rater as rt
from form_lab import data

STAMP = api.CACHE_DIR / 'refresh.json'


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


def main(pause=data.PAUSE):
    start = time.time()
    boot = api.bootstrap(refresh=True)
    gw = api.current_gw(boot)
    print(f'bootstrap: GW{gw}, {len(boot["elements"])} players', flush=True)

    data.fixtures(refresh=True)
    print('fixtures: updated', flush=True)

    pool = build_live.build()

    ids = {e['code']: e['id'] for e in boot['elements']}
    wanted = set(rt.available(rt._modelled(pool))['code']) | team_players()
    players = sorted(ids[c] for c in wanted if c in ids)
    for i, pid in enumerate(players, 1):
        data.element_summary(pid, pause, refresh=True)
        if i % 50 == 0 or i == len(players):
            print(f'  element-summaries {i}/{len(players)}', flush=True)

    import images
    fetched, missing = images.fetch_all(boot=boot)
    print(f'images: {fetched} new, {missing} unavailable (placeholder shown)',
          flush=True)

    import pandas as pd
    import core
    views = core.compute_build_views(build_live.load(), offline=True)
    pd.to_pickle(views, core.BUILD_VIEWS)
    print(f"build views: replayed to GW{views['gw'] if views else 0}", flush=True)

    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    STAMP.write_text(json.dumps({'gw': gw, 'refreshed': now}))
    print(f'\nrefreshed for GW{gw} at {now} ({time.time() - start:.0f}s)')


if __name__ == '__main__':
    main()
