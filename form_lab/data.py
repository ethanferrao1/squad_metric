"""Gameweek histories, season-model priors and ages.

Read-only on the main project: everything written lands in form_lab/cache/.
"""
import json
import time
import urllib.request

import numpy as np
import pandas as pd

import data_io as io

CACHE = io.DATA_DIR.parent / 'form_lab' / 'cache'
PRIORS = CACHE / 'priors.parquet'
SEASON_GWS = 38
API = 'https://fantasy.premierleague.com/api'
UA = 'Mozilla/5.0'
PAUSE = 1.0
TIMEOUT = 30


def _season_start(season):
    """First August of a season label like '24_25'."""
    return pd.Timestamp(f'20{season[:2]}-08-01')


def gw_history(season):
    """Per player, per gameweek: points, minutes, price, transfers, ownership.

    Doubles are summed into one gameweek row. A gameweek his team played but
    he did not is already a zero-minute row in the file; a blank gameweek has
    no row at all, which is what we want.
    """
    gw = io.read_gw(season)
    players = io.read_players(season)
    gw = gw.merge(players[['id', 'code', 'element_type']],
                  left_on='element', right_on='id')

    agg = gw.groupby(['code', 'round']).agg(
        points=('total_points', 'sum'),
        minutes=('minutes', 'sum'),
        price=('value', 'last'),
        transfers_in=('transfers_in', 'sum'),
        transfers_out=('transfers_out', 'sum'),
        selected=('selected', 'last'),
        element_type=('element_type', 'first'),
        fixtures=('total_points', 'size'),
    ).reset_index().rename(columns={'round': 'gw'})
    return agg.sort_values(['code', 'gw']).reset_index(drop=True)


def priors(refresh=False):
    """Season-model points prediction for each (season entered, player).

    run_milp.build()'s out-of-fold predictions, rekeyed from the season the
    features come from to the season they predict into -- so a backtest of
    24_25 never sees a model that trained on that player.
    """
    if PRIORS.exists() and not refresh:
        return pd.read_parquet(PRIORS)

    import run_milp
    trans, _, _ = run_milp.build()
    out = pd.DataFrame({
        'season': trans['next_season'].to_numpy(),
        'code': trans['code'].to_numpy(),
        'prior_pts': trans['oof_pts'].to_numpy(),
    })
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PRIORS, index=False)
    return out


def debut_season(season):
    """code -> True if the player appears in no players_raw before `season`."""
    earlier = io.SEASONS[:io.SEASONS.index(season)]
    seen = set()
    for s in earlier:
        try:
            seen |= set(io.read_players(s)['code'])
        except (FileNotFoundError, IOError):
            continue
    codes = io.read_players(season)['code']
    return pd.Series(~codes.isin(seen).to_numpy(), index=codes.to_numpy())


def ages(season):
    """code -> age in years at the start of `season` (NaN where dob is missing)."""
    players = io.read_players(season)
    if 'date_of_birth' not in players.columns:
        return pd.Series(np.nan, index=players['code'].to_numpy())
    dob = pd.to_datetime(players['date_of_birth'], format='%d/%m/%Y',
                         errors='coerce')
    years = (_season_start(season) - dob).dt.days / 365.25
    return pd.Series(years.to_numpy(), index=players['code'].to_numpy())


# set when a live fetch fails and the caller fell back to the cache
STATUS = {'network_failed': False}


def _get(path, refresh=False):
    """GET one API path, via form_lab's own cache."""
    f = CACHE / (path.strip('/').replace('/', '_') + '.json')
    if f.exists() and not refresh:
        return json.loads(f.read_text(encoding='utf-8'))

    req = urllib.request.Request(f'{API}/{path.strip("/")}/',
                                 headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = json.load(r)
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data), encoding='utf-8')
    return data


def fixtures(refresh=False):
    """The season's fixture list, cached."""
    return _get('fixtures', refresh)


def teams_with_fixture(gw):
    """Team ids with at least one fixture in `gw`."""
    out = set()
    for f in fixtures():
        if f.get('event') == gw:
            out |= {f['team_h'], f['team_a']}
    return out


def player_teams():
    """code -> team id, from the live bootstrap."""
    import fpl_api
    return {e['code']: e['team'] for e in fpl_api.bootstrap()['elements']}


def has_fixture(codes, gw):
    """code -> True if his club plays in `gw`."""
    teams = teams_with_fixture(gw)
    by_code = player_teams()
    return pd.Series([by_code.get(c) in teams for c in codes],
                     index=list(codes))


def live_regulars(codes, gw, offline=True):
    """Codes with 60+ minutes in at least half their team's last 5 games.
    A player with no history to judge by does not qualify."""
    from form_lab import backtest
    return backtest.regular_starters(live_history(codes, offline=offline),
                                     gw + 1)


def element_summary(element_id, pause=PAUSE, refresh=False):
    """One player's live gameweek history, cached. Pauses only on a real fetch."""
    f = CACHE / f'element-summary_{element_id}.json'
    fresh = refresh or not f.exists()
    data = _get(f'element-summary/{element_id}', refresh)
    if fresh:
        time.sleep(pause)
    return data


def is_cached(element_id):
    """True if this player's gameweek history is already on disk."""
    return (CACHE / f'element-summary_{element_id}.json').exists()


def live_history(codes, pause=PAUSE, offline=False):
    """gw_history-shaped frame for `codes`, from the live element-summary API.

    Missing histories are fetched and cached. With `offline` they are skipped
    instead; and if a fetch fails, the rest are skipped too and
    STATUS['network_failed'] is set, so the caller can say it fell back.
    """
    import fpl_api
    boot = fpl_api.bootstrap()
    by_code = {e['code']: e for e in boot['elements']}

    rows = []
    for code in codes:
        e = by_code.get(code)
        if e is None or (offline and not is_cached(e['id'])):
            continue
        try:
            summary = element_summary(e['id'], pause)
        except OSError:
            STATUS['network_failed'] = offline = True
            continue
        for h in summary['history']:
            rows.append({
                'code': code, 'gw': h['round'], 'points': h['total_points'],
                'minutes': h['minutes'], 'price': h['value'],
                'transfers_in': h['transfers_in'],
                'transfers_out': h['transfers_out'],
                'selected': h['selected'],
                'element_type': e['element_type'],
            })
    if not rows:
        return pd.DataFrame(columns=['code', 'gw', 'points', 'minutes', 'price',
                                     'transfers_in', 'transfers_out',
                                     'selected', 'element_type'])
    d = pd.DataFrame(rows)
    return d.groupby(['code', 'gw'], as_index=False).agg(
        points=('points', 'sum'), minutes=('minutes', 'sum'),
        price=('price', 'last'), transfers_in=('transfers_in', 'sum'),
        transfers_out=('transfers_out', 'sum'), selected=('selected', 'last'),
        element_type=('element_type', 'first'))


if __name__ == '__main__':
    p = priors()
    print(f'priors cached: {len(p)} rows, seasons {sorted(p["season"].unique())}')
