"""Fit the live model once and cache the 26/27 pool to disk, with the SHAP
drivers behind each prediction.

Run this whenever prices or the roster move (daily is plenty):

    python build_live.py

app.py then reads the parquet and never fits a model, or runs SHAP, at
request time.
"""
import json

import numpy as np
import pandas as pd

import data_io as io
import rater as rt

OUT = io.DATA_DIR / 'cache' / 'live_pool.parquet'
N_DRIVERS = 3


# Raw feature name -> something a human can read. Anything missing falls
# back to the underscores-to-spaces version, so a new feature degrades to
# readable rather than breaking.
GLOSSARY = {
    'total_points': 'total points last season',
    'pts': 'points from gameweek data last season',
    'minutes': 'minutes last season',
    'mins': 'minutes last season',
    'apps': 'appearances last season',
    'starts': 'starts last season',
    'now_cost': 'current price',
    'value_start': 'price at the start of last season',
    'value_end': 'price at the end of last season',
    'value_change': 'price change over last season',
    'goals_scored': 'goals last season',
    'assists': 'assists last season',
    'clean_sheets': 'clean sheets last season',
    'selected_by_percent': 'share of managers who owned him',
    'points_per_game': 'points per game last season',
    'pts_per_app': 'points per appearance last season',
    'pts_sd': 'how much his scores varied',
    'ceiling_top5': 'average of his five best returns',
    'haul_rate': 'share of games he returned 10+ points',
    'blank_rate': 'share of games he returned 2 or fewer points',
    'bps': 'bonus points system total',
    'influence': 'influence rating',
    'creativity': 'creativity rating',
    'threat': 'threat rating',
    'form_apps': 'appearances in his last ten games',
    'form_mins': 'average minutes in his last ten games',
    'form_mins_tot': 'total minutes in his last ten games',
    'form_pts': 'average points per match in his last ten games',
    'form_pts_sd': 'how much his last ten scores varied',
    'form_bps': 'average bonus points in his last ten games',
    'form_threat': 'average threat in his last ten games',
    'form_mins_share': 'share of his minutes that came recently',
    # these are per-MATCH means over each half of the season, not totals
    'early_pts': 'points per match in the first half of last season',
    'late_pts': 'points per match in the second half of last season',
    'early_mins': 'minutes per match in the first half of last season',
    'late_mins': 'minutes per match in the second half of last season',
    'trend_pts': 'change in points per match from the first to second half '
                 'of last season',
    'trend_mins': 'change in minutes per match from the first to second half '
                  'of last season',
    'early_ppa': 'points per appearance early last season',
    'late_ppa': 'points per appearance late last season',
    'trend_ppa': 'change in points per appearance across last season',
    'pts_slope': 'the trend in his scores across last season',
    'pts_p90': 'points per 90 minutes',
    'bps_p90': 'bonus points per 90 minutes',
    'influence_p90': 'influence per 90 minutes',
    'creativity_p90': 'creativity per 90 minutes',
    'threat_p90': 'threat per 90 minutes',
    'age': 'age',
    'tenure': 'seasons in the league',
    'career_mins': 'career minutes',
    'career_pts': 'career points',
    'career_pts_avg': 'career points per season',
    'prior_pts_max': 'his best season so far',
    'pts_vs_prior_best': 'last season against his best season',
    'next_team_gf': 'goals his new club scored last season',
    'next_team_ga': 'goals his new club conceded last season',
    'next_team_gd': 'his new club goal difference last season',
    'rank_in_club_mn': 'where he ranks for minutes among teammates in his position',
    'rank_pct_mn': 'his minutes rank among teammates as a percentage',
    'n_competitors_mn': 'how many teammates compete for his position',
    'price_share_mn': 'his share of the minutes in his position at the club',
    'gap_to_top_mn': 'the minutes gap to the top player in his position',
    'pos_1': 'being a goalkeeper',
    'pos_2': 'being a defender',
    'pos_3': 'being a midfielder',
    'pos_4': 'being a forward',
}


def plain(name):
    return GLOSSARY.get(name, name.replace('_', ' '))


def drivers(model, X_live, live, n=N_DRIVERS):
    """code -> the n features that moved each prediction most, by |SHAP|.

    TreeExplainer is exact for a forest, so these are the actual additive
    contributions to that player's predicted points, not an approximation.
    """
    import shap

    sv = shap.TreeExplainer(model).shap_values(X_live)
    sv = np.asarray(sv)
    if sv.ndim == 3:                     # (rows, features, outputs)
        sv = sv[:, :, 0]

    cols = list(X_live.columns)
    out = {}
    for i, code in enumerate(live['code'].to_numpy()):
        row = sv[i]
        top = np.argsort(np.abs(row))[::-1][:n]
        out[code] = [
            {'feature': plain(cols[j]),
             'raw_feature': cols[j],
             'value': round(float(X_live.iloc[i, j]), 2),
             'shap': round(float(row[j]), 2)}
            for j in top
        ]
    return out


def add_availability(pool):
    """Attach FPL's own fitness flags: `chance` (chance_of_playing_next_round,
    null when nothing is flagged) and `news` (the injury note, '' if none)."""
    try:
        import fpl_api
        boot = {e['code']: e for e in fpl_api.bootstrap()['elements']}
    except Exception as e:
        print(f'  no bootstrap available ({e}); everyone counts as fit')
        pool['chance'] = np.nan
        pool['news'] = ''
        return pool

    pool = pool.copy()
    pool['chance'] = [boot.get(c, {}).get('chance_of_playing_next_round')
                      for c in pool['code']]
    pool['chance'] = pd.to_numeric(pool['chance'], errors='coerce')
    pool['news'] = [(boot.get(c, {}).get('news') or '').strip()
                    for c in pool['code']]
    n_out = int((pool['chance'] < 50).sum())
    print(f'  availability: {n_out} player(s) below 50% chance of playing')
    return pool


def add_live_positions(pool, positions=None):
    """FPL's current positions. Modelled rows carry last season's, and FPL
    reclassifies players between seasons (Wieffer MID -> DEF)."""
    if positions is None:
        try:
            import fpl_api
            positions = {e['code']: e['element_type']
                         for e in fpl_api.bootstrap()['elements']}
        except Exception as e:
            print(f'  no bootstrap available ({e}); keeping model positions')
            return pool

    pool = pool.copy()
    live = pool['code'].map(positions)
    live = live.where(live.isin(list(rt.POS)))
    moved = int((live.notna() & (live != pool['element_type'])).sum())
    pool['element_type'] = live.fillna(pool['element_type']).astype(int)
    pool['pos'] = pool['element_type'].map(rt.POS)
    pool['rating'] = pool.groupby('element_type')['pred'].rank(pct=True) * 100
    print(f'  positions: {moved} player(s) moved to their current FPL position')
    return pool


def add_live_prices(pool):
    """Today's prices from the bootstrap. The modelled players are otherwise
    priced from the players_raw snapshot, which does not move with the game."""
    try:
        import fpl_api
        now = {e['code']: e['now_cost'] for e in fpl_api.bootstrap()['elements']}
    except Exception as e:
        print(f'  no bootstrap available ({e}); keeping snapshot prices')
        return pool

    pool = pool.copy()
    live = pool['code'].map(now)
    moved = int((live.notna() & (live != pool['price'])).sum())
    pool['price'] = live.fillna(pool['price'])
    pool['value'] = pool['pred'] / (pool['price'] / 10)
    print(f'  prices: {moved} player(s) repriced from the live bootstrap')
    return pool


def build(path=OUT):
    """Fit, score, explain, save."""
    model, X_live, live, cands = rt.live_parts()
    pool = rt.live_pool(parts=(model, X_live, live, cands))

    print(f'computing SHAP for {len(X_live)} modelled players...')
    d = drivers(model, X_live, live)

    # parquet handles a JSON string cleanly. Only modelled players get
    # drivers: a few rows survive the feature build but get dropped from the
    # candidate pool for want of a club or price, and their SHAP values
    # describe a prediction the pool no longer carries.
    pool['drivers'] = [
        json.dumps(d[c]) if (m and c in d) else None
        for c, m in zip(pool['code'], pool['modelled'])
    ]

    # availability, straight from the cached bootstrap, so the app never has
    # to call the API to know who is injured
    pool = add_availability(pool)
    pool = add_live_positions(pool)
    pool = add_live_prices(pool)

    path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(path, index=False)
    print(f'wrote {path}  ({len(pool)} players, '
          f'{int(pool["modelled"].sum())} modelled, '
          f'{int((~pool["modelled"]).sum())} without PL history, '
          f'{int(pool["drivers"].notna().sum())} with drivers)')
    return pool


def load(path=OUT):
    """The cached pool. Raises if build() has never been run."""
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found -- run `python build_live.py` first')
    return pd.read_parquet(path)


def player_drivers(pool, code):
    """The stored drivers for one player, as a list of dicts ([] if none)."""
    s = pool.loc[pool['code'] == code, 'drivers']
    if not len(s) or pd.isna(s.iloc[0]):
        return []
    return json.loads(s.iloc[0])


if __name__ == '__main__':
    build()
