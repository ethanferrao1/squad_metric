"""Season-level features. Everything here is computed from season t only,
except the team block, which uses the club the player turns out for in t+1
(known in August, after the window)."""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- gameweek

def gw_features(gw, players, season):
    """~24k match rows -> one row per player, keyed by `code`."""
    gw = gw.copy()
    gw['kickoff_time'] = pd.to_datetime(gw['kickoff_time'], format='mixed', utc=True)
    gw = gw.sort_values(['element', 'kickoff_time'])

    gw['played'] = gw['minutes'] > 0
    gw['started'] = gw['minutes'] >= 60

    gw['match_no'] = gw.groupby('element').cumcount()
    n = gw.groupby('element')['match_no'].transform('size')
    gw['is_late'] = gw['match_no'] >= n / 2

    avail = gw.groupby('element').agg(
        apps=('played', 'sum'),
        starts=('started', 'sum'),
        mins=('minutes', 'sum'),
        pts=('total_points', 'sum'),
        value_start=('value', 'first'),
        value_end=('value', 'last'),
    )
    avail['value_change'] = avail['value_end'] - avail['value_start']

    played = gw[gw['played']]
    shape = played.groupby('element').agg(
        pts_per_app=('total_points', 'mean'),
        pts_sd=('total_points', 'std'),
        ceiling_top5=('total_points', lambda s: s.nlargest(5).mean()),
        haul_rate=('total_points', lambda s: (s >= 10).mean()),
        blank_rate=('total_points', lambda s: (s <= 2).mean()),
        bps=('bps', 'sum'),
        influence=('influence', 'sum'),
        creativity=('creativity', 'sum'),
        threat=('threat', 'sum'),
    )

    form = gw.groupby('element').tail(10).groupby('element').agg(
        form_apps=('played', 'sum'),
        form_mins=('minutes', 'mean'),
        form_mins_tot=('minutes', 'sum'),
        form_pts=('total_points', 'mean'),
        form_pts_sd=('total_points', 'std'),
        form_bps=('bps', 'mean'),
        form_threat=('threat', 'mean'),
    )

    early = gw[~gw['is_late']].groupby('element').agg(
        early_pts=('total_points', 'mean'), early_mins=('minutes', 'mean'))
    late = gw[gw['is_late']].groupby('element').agg(
        late_pts=('total_points', 'mean'), late_mins=('minutes', 'mean'))
    trend = early.join(late, how='outer')
    trend['trend_pts'] = trend['late_pts'] - trend['early_pts']
    trend['trend_mins'] = trend['late_mins'] - trend['early_mins']

    ep = played[~played['is_late']].groupby('element')['total_points'].mean()
    lp = played[played['is_late']].groupby('element')['total_points'].mean()
    trend['early_ppa'], trend['late_ppa'] = ep, lp
    trend['trend_ppa'] = trend['late_ppa'] - trend['early_ppa']

    # OLS slope of points across played matches
    p = played[['element', 'total_points']].copy()
    p['xi'] = played.groupby('element').cumcount()
    xc = p['xi'] - p.groupby('element')['xi'].transform('mean')
    yc = p['total_points'] - p.groupby('element')['total_points'].transform('mean')
    sl = pd.DataFrame({'element': p['element'], 'num': xc * yc, 'den': xc ** 2}) \
           .groupby('element').sum()
    trend['pts_slope'] = sl['num'] / sl['den'].replace(0, np.nan)

    feats = avail.join(shape).join(form).join(trend)

    mins = feats['mins'].replace(0, np.nan)
    for c in ['pts', 'bps', 'influence', 'creativity', 'threat']:
        feats[f'{c}_p90'] = feats[c] / mins * 90
    feats['form_mins_share'] = feats['form_mins_tot'] / mins

    feats = feats.reset_index().merge(
        players[['id', 'code']].drop_duplicates(),
        left_on='element', right_on='id', how='inner')
    feats['season'] = season
    return feats.drop(columns=['element', 'id'])


# ---------------------------------------------------------------- club

def match_clubs(gw, players):
    """Add `team_code` per match row. Recovered from the fixture, not from the
    players_raw snapshot, so January movers are attributed correctly."""
    gw = gw.copy()
    home = gw[~gw['was_home']].groupby('fixture')['opponent_team'].first()
    away = gw[gw['was_home']].groupby('fixture')['opponent_team'].first()
    gw['team'] = np.where(gw['was_home'],
                          gw['fixture'].map(home), gw['fixture'].map(away))
    code = players.drop_duplicates('team').set_index('team')['team_code']
    gw['team_code'] = gw['team'].map(code)
    return gw


def club_season_stats(gw):
    """Goals for/against per club, from one row per (club, fixture)."""
    g = gw.dropna(subset=['team_code']).drop_duplicates(['team_code', 'fixture'])
    gf = np.where(g['was_home'], g['team_h_score'], g['team_a_score'])
    ga = np.where(g['was_home'], g['team_a_score'], g['team_h_score'])
    out = pd.DataFrame({'team_code': g['team_code'], 'gf': gf, 'ga': ga}) \
            .groupby('team_code').sum()
    out['gd'] = out['gf'] - out['ga']
    return out


def first_club(gw):
    """Each player's club in his FIRST match of the season, by `code`."""
    g = gw.sort_values(['code', 'kickoff_time'])
    return g.groupby('code')['team_code'].first()


# ---------------------------------------------------------------- pecking order

def pecking_order(squad, by, group=('team_code', 'element_type')):
    """Rank each row of `squad` against its peer group (default: same club,
    same position) by column `by`. Peers missing `by` or a group key are
    dropped from the ranking, not zero-filled."""
    g = list(group)
    s = squad.dropna(subset=g + [by]).copy()
    grp = s.groupby(g)[by]
    s['rank_in_club'] = grp.rank(ascending=False, method='min')
    s['n_competitors'] = grp.transform('size')
    s['rank_pct'] = s['rank_in_club'] / s['n_competitors']
    s['price_share'] = s[by] / grp.transform('sum').replace(0, np.nan)
    s['gap_to_top'] = grp.transform('max') - s[by]
    return s.set_index('code')[['rank_in_club', 'rank_pct', 'n_competitors',
                                'price_share', 'gap_to_top']]


def add_pecking(trans, clubs, io_module, suffix='_mn'):
    """Add minutes-ranked pecking-order columns to `trans` (already built by
    panel.build_transitions/prepare). Peer group = the FULL next-season
    roster from players_raw (not just the 450+-minute modelled pool), ranked
    on each player's OWN season-t minutes (known before t+1 starts, so no
    leakage). A peer missing a season-t row is dropped from the ranking, not
    zero-filled -- new signings/promotions aren't penalised, they're just
    not counted as competitors yet.

    `io_module` is passed in (rather than imported here) to avoid a circular
    import between features.py and data_io.py -- pass `data_io` from the
    caller.
    """
    out = []
    for t in sorted(trans['season'].unique()):
        rows = trans[trans['season'] == t]
        t1 = rows['next_season'].iloc[0]
        players = io_module.read_players(t1)
        sq = players[['code', 'element_type']].copy()
        sq['team_code'] = sq['code'].map(clubs[t1])
        sq['mins'] = sq['code'].map(rows.set_index('code')['mins'])
        out.append(pecking_order(sq, 'mins').reindex(rows['code'])
                                            .set_index(rows.index))
    peck = pd.concat(out).reindex(trans.index).add_suffix(suffix)
    return trans.join(peck)


# ---------------------------------------------------------------- career

def add_career(panel):
    """Tenure and career totals, counted up to and including season t."""
    panel = panel.sort_values(['code', 'season']).reset_index(drop=True)
    g = panel.groupby('code')
    panel['tenure'] = g.cumcount() + 1
    panel['career_mins'] = g['minutes'].cumsum()
    panel['career_pts'] = g['total_points'].cumsum()
    panel['career_pts_avg'] = panel['career_pts'] / panel['tenure']
    panel['prior_pts_max'] = g['total_points'].transform(
        lambda s: s.shift().expanding().max())
    panel['pts_vs_prior_best'] = panel['total_points'] - panel['prior_pts_max']
    return panel


# ---------------------------------------------------------------- age

def parse_dob(s):
    """with_dob files are YYYY-MM-DD; players_raw_2025_26 is DD/MM/YYYY."""
    return pd.to_datetime(s, format='%Y-%m-%d', errors='coerce').fillna(
           pd.to_datetime(s, format='%d/%m/%Y', errors='coerce'))


def add_age(panel, dob, lo=16, hi=42):
    """Age at 1 August of season t. Implausible values are nulled: the dob
    source was name-matched and common names collided."""
    panel = panel.merge(dob.rename('date_of_birth'), on='code', how='left')
    ref = pd.to_datetime('20' + panel['season'].str[:2] + '-08-01')
    panel['age'] = (ref - panel['date_of_birth']).dt.days / 365.25
    panel.loc[~panel['age'].between(lo, hi), 'age'] = np.nan
    return panel.drop(columns=['date_of_birth'])