"""A clean holdout: train on everything up to and including the 24/25
target, then predict the one season the model has never seen, 24/25 -> 25/26.

Unlike the out-of-fold backtest, no fold of this model ever saw a 25/26
outcome, and no 25/26 row influenced feature construction. It is the closest
thing here to "what would it have done in August 2025".

    python holdout.py

Reads the pipeline; changes none of it.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import model as md
import optimise as op
import panel as pn
import run_milp as rm

TEST_SEASON = '24_25'          # features from here...
TEST_TARGET = '25_26'          # ...outcome here
POS = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
TOP_K = 10

STRATEGIES = {
    'model':              'pred',
    'persistence':        'last_pts',
    'points_per_million': 'ppm',
    'market':             'price',
    'hindsight':          'actual_pts',
}


def fit_predict(test_season=TEST_SEASON, test_target=None):
    """Train on every other transition, predict the held-out one.

    Defaults to 24/25 -> 25/26. Pass an earlier `test_season` to hold out an
    earlier season instead; the target is inferred from the panel.
    """
    trans = pn.build_all()
    X, y, groups, season = pn.make_xy(trans)

    is_test = trans['season'] == test_season
    assert is_test.any(), f'no {test_season} rows to hold out'
    target = test_target or trans.loc[is_test, 'next_season'].iloc[0]
    assert (trans.loc[is_test, 'next_season'] == target).all()

    # everything whose OUTCOME is the held-out season or later is dropped, so
    # no row the model trains on can know how that season turned out
    is_future = trans['next_season'] >= target
    model = md.make().fit(X[~is_future], trans.loc[~is_future, 'target_pts'])
    test = trans[is_test].copy()
    test['pred'] = model.predict(X[is_test])

    train_seasons = sorted(trans.loc[~is_future, 'next_season'].unique())
    return test, train_seasons


def ranking_table(test):
    """Spearman, top-10 hit rate and MAE, per position and overall."""
    t = test.copy()
    t['pos'] = t['element_type'].map(POS)

    # groupby sorts alphabetically; walk POS order explicitly so the labels
    # can never drift from the rows
    groups = [(p, t[t['pos'] == p]) for p in POS.values()] + [('Overall', t)]

    rows = []
    for label, g in groups:
        hit = chance = np.nan
        if label != 'Overall' and len(g) >= 2 * TOP_K:
            top_actual = set(g.nlargest(TOP_K, 'target_pts')['code'])
            top_pred = set(g.nlargest(TOP_K, 'pred')['code'])
            hit = len(top_actual & top_pred)
            # picking k of n at random hits k*k/n of the true top k
            chance = TOP_K * TOP_K / len(g)
        rows.append({
            'pos': label,
            'n': len(g),
            f'top{TOP_K}_chance': chance,
            'spearman': g['target_pts'].corr(g['pred'], method='spearman'),
            'spearman_base': g['target_pts'].corr(g['total_points'],
                                                  method='spearman'),
            f'top{TOP_K}_hit': hit,
            'mae_pts': mean_absolute_error(g['target_pts'], g['pred']),
            'mae_base': mean_absolute_error(g['target_pts'], g['total_points']),
        })
    return pd.DataFrame(rows).set_index('pos')


def squad_table(test):
    """MILP squad points for each strategy on the held-out season."""
    _, clubs = pn.club_tables()
    cands = rm.candidates(test.assign(oof_pts=test['pred']), clubs)
    c = cands[TEST_SEASON].copy()
    c['ppm'] = c['last_pts'] / (c['price'] / 10)      # points per million

    rows = {}
    for label, col in STRATEGIES.items():
        r = op.run(c, col)
        rows[label] = {'squad_pts': r['actual'], 'spend': r['spend'] / 10}
    return pd.DataFrame(rows).T


def main():
    test, train_seasons = fit_predict()
    rank = ranking_table(test)
    squad = squad_table(test)

    base = squad.loc['persistence', 'squad_pts']
    ceil = squad.loc['hindsight', 'squad_pts']
    squad['vs_persistence'] = squad['squad_pts'] - base
    squad['pct_of_ceiling'] = (squad['squad_pts'] - base) / (ceil - base)

    w = 66
    print()
    print('=' * w)
    print(f'HOLDOUT  {TEST_SEASON} -> {TEST_TARGET}'.center(w))
    print(f'trained on targets {train_seasons[0]}-{train_seasons[-1]} '
          f'({len(train_seasons)} seasons), {len(test)} players'.center(w))
    print('=' * w)
    print('\nRANKING                    (base = last season\'s points)')
    print('-' * w)
    print(rank.round(3).to_string())
    print(f'\nSQUAD POINTS               (MILP, 100.0m, XI + captain)')
    print('-' * w)
    print(squad.round(2).to_string())
    print('=' * w)
    return rank, squad


if __name__ == '__main__':
    pd.set_option('display.width', 140)
    main()
