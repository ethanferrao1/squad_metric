"""Panel assembly: per-season rows -> season t features with a season t+1 target."""

import numpy as np
import pandas as pd

import data_io as io
import features as fx

SEASON_COLS = ['code', 'web_name', 'element_type', 'team_code', 'now_cost',
               'total_points', 'minutes', 'goals_scored', 'assists',
               'clean_sheets', 'selected_by_percent', 'points_per_game']

MIN_MINUTES = 450

TEAM_COLS = ['next_team_gf', 'next_team_ga', 'next_team_gd']


# ---------------------------------------------------------------- panels

def build_season_panel():
    """One row per player per season from players_raw. ALL seasons, including
    21_22: players_raw is sound there, only merged_gw is corrupt."""
    frames = []
    for s in io.SEASONS:
        p = io.read_players(s)[SEASON_COLS].copy()
        for c in SEASON_COLS[2:]:
            p[c] = pd.to_numeric(p[c], errors='coerce')
        p['season'] = s
        frames.append(p)
    panel = pd.concat(frames, ignore_index=True)
    assert not panel.duplicated(['code', 'season']).any()
    return panel


def build_gw_panel():
    """Gameweek-derived features. FEATURE_SEASONS only."""
    return pd.concat(
        [fx.gw_features(io.read_gw(s), io.read_players(s), s)
         for s in io.FEATURE_SEASONS],
        ignore_index=True)


def build_dob():
    """code -> date of birth, resolved across every file that carries one."""
    frames = []
    for s in io.SEASONS:
        for path in (io.DATA_DIR / f'players_20{s}_with_dob.csv',
                     io.DATA_DIR / f'players_raw_20{s}.csv'):
            if not path.exists():
                continue
            d = io._read_csv(path)
            if 'date_of_birth' in d.columns:
                frames.append(pd.DataFrame(
                    {'code': d['code'], 'd': fx.parse_dob(d['date_of_birth'])}))
                break
    if not frames:
        raise FileNotFoundError(f'no date_of_birth column found under {io.DATA_DIR}')
    dob = pd.concat(frames, ignore_index=True).dropna()
    return dob.groupby('code')['d'].agg(lambda s: s.mode().iloc[0])


def build_panel():
    """Season panel + career + age + gameweek features."""
    season_panel = fx.add_career(build_season_panel())
    season_panel = fx.add_age(season_panel, build_dob())

    panel = season_panel.merge(build_gw_panel(), on=['code', 'season'],
                               how='inner', suffixes=('', '_gw'))
    assert not panel.columns.duplicated().any()
    assert not panel.duplicated(['code', 'season']).any()
    return panel


# ---------------------------------------------------------------- club context

def club_tables():
    """Per season: club goal stats, and each player's club in his first match."""
    stats, clubs = {}, {}
    for s in io.SEASONS:
        players = io.read_players(s)
        if s in io.BAD_FEATURE_SEASONS or s == io.CURRENT_SEASON:
            # merged_gw is untrustworthy (21_22) or partial/absent (current
            # season); players_raw's own team_code is the post-window roster
            clubs[s] = players.set_index('code')['team_code']
            stats[s] = None
            continue
        gw = fx.match_clubs(io.read_gw(s), players)
        stats[s] = fx.club_season_stats(gw)
        gw = gw.merge(players[['id', 'code']], left_on='element', right_on='id')
        gw['kickoff_time'] = pd.to_datetime(gw['kickoff_time'],
                                            format='mixed', utc=True)
        clubs[s] = fx.first_club(gw)
    return stats, clubs


def add_team_features(trans, stats, clubs):
    """The club he turns out for in t+1, as that club performed in t.
    Post-window: known in August, not in May."""
    nxt = trans.apply(
        lambda r: clubs[r['next_season']].get(r['code'], np.nan), axis=1)
    cur_stats = trans['season'].map(lambda s: stats[s])

    def look(code_series, col):
        return [np.nan if (st is None or pd.isna(tc) or tc not in st.index)
                else st.loc[tc, col]
                for tc, st in zip(code_series, cur_stats)]

    trans['next_team_gf'] = look(nxt, 'gf')
    trans['next_team_ga'] = look(nxt, 'ga')
    trans['next_team_gd'] = look(nxt, 'gd')
    return trans


# ---------------------------------------------------------------- transitions

def build_transitions(panel):
    """Features from t, target = total_points in t+1. Inner join, so only
    players present in both seasons survive."""
    frames = []
    for t, t1 in zip(io.SEASONS, io.SEASONS[1:]):
        if t in io.BAD_FEATURE_SEASONS:
            continue
        if t1 == io.CURRENT_SEASON:     # in progress: not a real target yet
            continue
        X = panel[panel['season'] == t]
        if X.empty:
            continue
        y = (panel.loc[panel['season'] == t1, ['code', 'total_points']]
                  .rename(columns={'total_points': 'target_pts'}))
        f = X.merge(y, on='code', how='inner')
        f['next_season'] = t1
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def prepare(trans, min_minutes=MIN_MINUTES):
    """Minutes floor, then the within-season percentile target and the
    persistence baseline on that same scale."""
    trans = trans[trans['mins'] >= min_minutes].reset_index(drop=True)
    trans['target'] = trans.groupby('season')['target_pts'].rank(pct=True)
    trans['base_pct'] = trans.groupby('season')['total_points'].rank(pct=True)
    return trans


DROP = ['target', 'target_pts', 'base_pct', 'code', 'season', 'next_season',
        'web_name', 'team_code']


def make_xy(trans, drop_extra=()):
    """X, y, groups, season. Position one-hot; identifiers dropped.

    Pecking-order (`*_mn`) columns are pulled out and re-appended LAST, after
    the position dummies -- this must match squad_rater.ipynb's original
    construction order (dummies first, pecking concatenated after) exactly,
    because RandomForestRegressor's max_features=0.5 + fixed random_state
    means column ORDER changes which features get sampled at each split,
    changing the fitted model even though nothing about the data changed.
    """
    peck_cols = [c for c in trans.columns if c.endswith('_mn')]
    base = trans.drop(columns=peck_cols)

    frame = pd.get_dummies(base, columns=['element_type'], prefix='pos')
    X = (frame.drop(columns=DROP + list(drop_extra), errors='ignore')
              .select_dtypes(include=['number', 'bool']))
    X = pd.concat([X, trans[peck_cols]], axis=1)
    return X, trans['target'], trans['code'], trans['season']


def build_all():
    """The whole pipeline, one call."""
    panel = build_panel()
    stats, clubs = club_tables()
    trans = build_transitions(panel)
    trans = add_team_features(trans, stats, clubs)
    trans = prepare(trans)
    trans = fx.add_pecking(trans, clubs, io)
    return trans