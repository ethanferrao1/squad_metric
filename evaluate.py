"""Evaluation. Decision metric is paired within-season Spearman against the
persistence baseline; R2/MAE and top-k are reported alongside it."""

from math import comb

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, r2_score

POS = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}


def frame(trans, preds):
    """One tidy table: identifiers, actual, and a column per predictor."""
    e = pd.DataFrame({
        'season': trans['season'].to_numpy(),
        'code': trans['code'].to_numpy(),
        'pos': trans['element_type'].map(POS).fillna('OTH').to_numpy(),
        'actual': trans['target'].to_numpy(),
        'actual_pts': trans['target_pts'].to_numpy(),
    })
    for name, p in preds.items():
        e[name] = np.asarray(p)
    return e


def season_spearman(e, col):
    """Within-season rank correlation, one value per season."""
    return e.groupby('season').apply(
        lambda g: g['actual'].corr(g[col], method='spearman'),
        include_groups=False)


def spearman_table(e, preds):
    return pd.DataFrame({c: season_spearman(e, c) for c in preds})


def paired(rho, model, base):
    """Model minus baseline, one pair per season. Sign test and Wilcoxon;
    Wilcoxon uses the SIZE of each gain, the sign test throws that away."""
    d = rho[model] - rho[base]
    wins = int((d > 0).sum())
    n = len(d) - int((d == 0).sum())
    return {
        'model': model,
        'wins': f'{wins}/{len(d)}',
        'mean_delta': d.mean(),
        'median_delta': d.median(),
        'worst': d.min(),
        'sign_p': sum(comb(n, i) for i in range(wins, n + 1)) / 2 ** n,
        'wilcoxon_p': wilcoxon(d, alternative='greater').pvalue,
    }


def error_table(e, preds):
    """R2 and MAE on the percentile target. Row-wise, so directly comparable."""
    return pd.DataFrame([
        {'model': c,
         'r2': r2_score(e['actual'], e[c]),
         'mae': mean_absolute_error(e['actual'], e[c]),
         'pred_sd': e[c].std()}
        for c in preds
    ]).set_index('model').assign(actual_sd=e['actual'].std())


def capture(e, preds, ks=(5, 10)):
    """Top-k overlap per position. A season x position pool smaller than 2k is
    skipped: picking 10 from 12 is near-trivial. `chance` is k^2/n."""
    recs = []
    for (season, pos), g in e.groupby(['season', 'pos']):
        n = len(g)
        for k in ks:
            if n < 2 * k:
                continue
            top = set(g.nlargest(k, 'actual')['code'])
            row = {'pos': pos, 'k': k, 'of': k, 'chance': k * k / n}
            for c in preds:
                row[c] = len(top & set(g.nlargest(k, c)['code']))
            recs.append(row)

    cap = pd.DataFrame(recs)
    tot = cap.groupby(['pos', 'k']).sum(numeric_only=True)
    tot['chance'] = tot['chance'] / tot['of']      # summed hits -> rate
    for c in preds:
        tot[f'{c}_rate'] = tot[c] / tot['of']
        tot[f'{c}_lift'] = tot[f'{c}_rate'] / tot['chance']
    return tot


def points_captured(e, preds, k=15):
    """Share of the best possible top-k points haul each predictor's own top-k
    actually scored. The closest correlation metric to squad selection."""
    rows = []
    for c in preds:
        v = [g.nlargest(k, c)['actual_pts'].sum() /
             g.nlargest(k, 'actual_pts')['actual_pts'].sum()
             for _, g in e.groupby('season')]
        rows.append({'model': c, 'pts_captured': np.mean(v)})
    return pd.DataFrame(rows).set_index('model')


def report(trans, preds, base='baseline', k=15):
    """Everything, in one call."""
    e = frame(trans, preds)
    rho = spearman_table(e, preds)
    others = [c for c in preds if c != base]
    return {
        'spearman': rho,
        'paired': pd.DataFrame([paired(rho, m, base) for m in others])
                    .set_index('model'),
        'error': error_table(e, preds),
        'capture': capture(e, preds),
        'points_captured': points_captured(e, preds, k=k),
    }