"""Ranking report: out-of-fold model vs the persistence baseline, on the
within-season percentile target. Decision metric is paired Spearman.

    python rank_report.py
"""
import pandas as pd

import evaluate as ev
import model as md
import panel as pn


def build():
    """Pipeline + out-of-fold percentile predictions."""
    trans = pn.build_all()
    X, y, groups, season = pn.make_xy(trans)
    oof_pct = md.oof(X, y, groups)
    return trans, X, oof_pct


def main():
    trans, X, oof_pct = build()
    print(f'trans {trans.shape}   X {X.shape}')
    print(f'seasons: {sorted(trans["season"].unique())}\n')

    rep = ev.report(trans, {'baseline': trans['base_pct'], 'model': oof_pct})

    print('--- per-season Spearman ---')
    print(rep['spearman'].round(4).to_string())
    print('\n--- paired (model - baseline) ---')
    print(rep['paired'].round(4).to_string())
    print('\n--- error (percentile target) ---')
    print(rep['error'].round(4).to_string())
    print('\n--- points captured (top 15) ---')
    print(rep['points_captured'].round(4).to_string())
    return rep


if __name__ == '__main__':
    pd.set_option('display.width', 140)
    main()
