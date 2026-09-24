"""Predict INTO the live 2026/27 season and build a squad. Additive only --
doesn't touch data_io/features/panel/model/evaluate/optimise.
"""
import numpy as np
import pandas as pd

import data_io as io
import features as fx
import model as md
import optimise as op
import panel as pn

T = io.FEATURE_SEASONS[-1]          # '25_26' -- most recent COMPLETE season
T1 = io.CURRENT_SEASON              # '26_27'


def fit_full_model():
    """Fit on ALL historical transitions, no holdout. This is deployment,
    not evaluation -- there is no season left to protect for testing."""
    trans = pn.build_all()
    X, y, groups, season = pn.make_xy(trans)
    model = md.make().fit(X, trans['target_pts'])
    return model, X.columns, trans


def live_features(trans_columns):
    """Season-25/26 rows only, built the SAME way panel.build_all() builds
    every other season, with next_season set to 26/27 so add_pecking ranks
    each player against his ACTUAL 26/27 club roster. Column order must
    match `trans_columns` exactly (see panel.make_xy's docstring on why
    order matters with max_features=0.5 + a fixed seed)."""
    panel = pn.build_panel()
    stats, clubs = pn.club_tables()

    live = panel[panel['season'] == T].copy()
    live = live[live['mins'] >= pn.MIN_MINUTES].reset_index(drop=True)
    live['next_season'] = T1

    live = pn.add_team_features(live, stats, clubs)
    live = fx.add_pecking(live, clubs, io)

    # Mirror make_xy()'s exact construction: dummies on the base columns,
    # THEN the pecking (_mn) columns appended last.
    peck_cols = [c for c in live.columns if c.endswith('_mn')]
    base = live.drop(columns=peck_cols)
    frame = pd.get_dummies(base, columns=['element_type'], prefix='pos')
    X_live = (frame.drop(columns=pn.DROP, errors='ignore')
                   .select_dtypes(include=['number', 'bool']))
    X_live = pd.concat([X_live, live[peck_cols]], axis=1)

    # Defensive: catch a silent column mismatch before it reaches the model.
    missing = set(trans_columns) - set(X_live.columns)
    extra = set(X_live.columns) - set(trans_columns)
    assert not missing, f'live features missing columns the model expects: {missing}'
    if extra:
        print(f'dropping {len(extra)} columns not seen in training: {extra}')
    X_live = X_live[list(trans_columns)]   # exact order match

    return X_live, live


def build_candidates(live, preds):
    """code, position, club, BOTH prices (see note below), and the
    prediction. `last_pts` is 25/26's total points -- shown for reference,
    not as a strategy to optimise against (there's no real persistence
    baseline for a season that hasn't happened)."""
    _, clubs = pn.club_tables()
    players1 = io.read_players(T1)

    c = pd.DataFrame({
        'code': live['code'].to_numpy(),
        'element_type': live['element_type'].to_numpy(),
        'team_code': live['code'].map(clubs[T1]).to_numpy(),
        'current_price': live['code'].map(
            players1.set_index('code')['now_cost']).to_numpy(),
        'last_pts': live['total_points'].to_numpy(),
        'pred': preds,
    })

    # Start-of-season price too, if a partial 26/27 gw file is available --
    # useful to compare "what it would have cost in August" vs "what it
    # costs to buy today, mid-season".
    try:
        gw1 = io.read_gw(T1).merge(players1[['id', 'code']],
                                   left_on='element', right_on='id')
        gw1['kickoff_time'] = pd.to_datetime(gw1['kickoff_time'],
                                             format='mixed', utc=True)
        c['start_price'] = c['code'].map(op.start_prices(gw1))
    except FileNotFoundError:
        c['start_price'] = np.nan

    return c.dropna(subset=['team_code', 'current_price'])


def main(price_col='current_price'):
    """price_col: 'current_price' (buy TODAY, mid-season -- default, since
    that's the actionable number) or 'start_price' (what it would have cost
    in August, for reference only)."""
    model, trans_columns, trans = fit_full_model()
    X_live, live = live_features(trans_columns)
    preds = model.predict(X_live)

    cands = build_candidates(live, preds)
    cands = cands.rename(columns={price_col: 'price'})

    squad = op.pick_squad(cands, 'pred')
    xi, captain = op.best_xi(squad, 'pred')

    names = io.read_players(T1).set_index('code')['web_name']
    show = squad.assign(web_name=squad['code'].map(names)).sort_values(
        ['element_type', 'pred'], ascending=[True, False])
    print(show[['web_name', 'element_type', 'price', 'pred', 'last_pts']]
          .to_string(index=False))
    print(f"\ncaptain: {names.get(captain, captain)}")
    print(f"spend: {squad['price'].sum() / 10:.1f}m / 100.0m")
    print(f"predicted XI total (captain doubled): "
          f"{xi['pred'].sum() + xi.loc[xi['code'] == captain, 'pred'].sum():.1f}")
    return squad, xi, captain


if __name__ == '__main__':
    main()
