"""End-to-end squad evaluation.

Sits on top of data_io / features / panel / model / optimise. Nothing in those
modules changes. Paste as notebook cells, or import and call main().
"""

import numpy as np
import pandas as pd

import data_io as io
import features as fx
import model as md
import optimise as op
import panel as pn


# ---------------------------------------------------------------- cell 1

def build():
    """Pipeline + both models. Returns (trans, X, groups, oof_pct, oof_pts)."""
    trans = pn.build_all()
    X, y, groups, season = pn.make_xy(trans)

    oof_pct = md.oof(X, y, groups)                       # ranking model
    oof_pts = md.oof(X, trans['target_pts'], groups)     # MILP input, points scale

    trans = trans.copy()
    trans['oof_pct'] = oof_pct
    trans['oof_pts'] = oof_pts
    return trans, X, groups


# ---------------------------------------------------------------- cell 2

def candidates(trans, clubs):
    """One candidate frame per feature season.

    Two things that are easy to get wrong:
      price     - START of season t+1, from that season's gameweek file. Using
                  players_raw.now_cost would be the END-of-season price, which
                  already reflects how well the player did.
      team_code - the club he turns out for in t+1, not t. The 3-per-club limit
                  applies to the squad you actually field.
    """
    out = {}
    for t in sorted(trans['season'].unique()):
        rows = trans[trans['season'] == t]
        t1 = rows['next_season'].iloc[0]

        if t1 in io.BAD_FEATURE_SEASONS:
            # merged_gw is corrupt there, so no trustworthy start prices
            print(f'  skipping {t} -> {t1}: no reliable prices')
            continue

        gw1 = io.read_gw(t1)
        gw1 = gw1.merge(io.read_players(t1)[['id', 'code']],
                        left_on='element', right_on='id')
        gw1['kickoff_time'] = pd.to_datetime(gw1['kickoff_time'],
                                             format='mixed', utc=True)
        price = op.start_prices(gw1)

        c = pd.DataFrame({
            'code': rows['code'].to_numpy(),
            'element_type': rows['element_type'].to_numpy(),
            'team_code': rows['code'].map(clubs[t1]).to_numpy(),
            'price': rows['code'].map(price).to_numpy(),
            'actual_pts': rows['target_pts'].to_numpy(),
            'pred': rows['oof_pts'].to_numpy(),
            'last_pts': rows['total_points'].to_numpy(),
        })

        before = len(c)
        c = c.dropna(subset=['price', 'team_code', 'element_type'])
        if before != len(c):
            print(f'  {t}: dropped {before - len(c)} of {before} '
                  f'candidates with no price or club')
        out[t] = c.reset_index(drop=True)
    return out


# ---------------------------------------------------------------- cell 3

STRATEGIES = {
    'model':       'pred',
    'persistence': 'last_pts',
    'market':      'price',        # buy the most expensive squad affordable
    'hindsight':   'actual_pts',   # ceiling, not achievable
}


def evaluate(cands_by_season, strategies=STRATEGIES, base='persistence'):
    strategies = {k: v for k, v in strategies.items()
                  if all(v in c.columns for c in cands_by_season.values())}
    long = op.compare(cands_by_season, strategies)
    order = [k for k in strategies]

    table = long.pivot(index='season', columns='strategy', values='actual')[order]
    spend = long.pivot(index='season', columns='strategy', values='spend')[order]

    summary = pd.DataFrame({'mean': table.mean(), 'median': table.median(),
                            'worst': table.min(), 'mean_spend': spend.mean()})
    # how much of the achievable gap above the baseline each strategy closes
    gap = table['hindsight'] - table[base]
    summary['pct_of_ceiling'] = (
        table.sub(table[base], axis=0)).div(gap, axis=0).mean()

    for m in order:
        if m in (base, 'hindsight'):
            continue
        wins = int((table[m] > table[base]).sum())
        print(f'{m:14s} beats {base} in {wins}/{len(table)} seasons')
    return table, summary, spend


# ---------------------------------------------------------------- all of it

def main():
    trans, X, groups = build()
    _, clubs = pn.club_tables()
    cands = candidates(trans, clubs)
    table, summary, spend = evaluate(cands)
    print('\nsquad points by season:')
    print(table.round(0).to_string())
    print('\nsummary:')
    print(summary.round(3).to_string())
    return trans, cands, table, summary, spend


if __name__ == '__main__':
    main()