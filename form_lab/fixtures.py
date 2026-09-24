"""Fixture difficulty over the next five gameweeks, from two sources.

(a) 'fpl': FPL's own ratings as they stood before each deadline. The vaastav
    archive re-commits fixtures.csv through the season, so its git history
    gives the ratings a manager could actually see at the time.
(b) 'own': the opponent's goals over their last 10 games before the deadline
    -- goals scored for GK/DEF, goals conceded for MID/FWD -- plus a home/away
    term from last season's home advantage.

Both are z-scored across the window's fixtures, so K means the same thing for
each: the fractional change in projection per standard deviation of difficulty.
"""
import json
import urllib.request

import numpy as np
import pandas as pd

import data_io as io
from form_lab import data

REPO = 'vaastav/Fantasy-Premier-League'
DIR = data.CACHE / 'fdr'
HORIZON = 5
STRENGTH_GAMES = 10
UA = {'User-Agent': 'form-lab'}
SOURCES = ('fpl', 'own')


def _label(season):
    return f'20{season[:2]}-{season[3:]}'


def _get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                timeout=30) as r:
        return r.read()


def commits(season):
    """[(scrape time, sha)] for the season's fixtures.csv, oldest first."""
    f = DIR / season / 'commits.json'
    if not f.exists():
        url = (f'https://api.github.com/repos/{REPO}/commits'
               f'?path=data/{_label(season)}/fixtures.csv&per_page=100')
        raw = json.loads(_get(url))
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps([(c['commit']['author']['date'], c['sha'])
                                 for c in raw]))
    return sorted((pd.Timestamp(t), s) for t, s in json.loads(f.read_text()))


def version(season, sha):
    """fixture id -> (home side's difficulty, away side's) in one commit."""
    f = DIR / season / f'{sha}.csv'
    if not f.exists():
        f.write_bytes(_get(f'https://raw.githubusercontent.com/{REPO}/{sha}'
                           f'/data/{_label(season)}/fixtures.csv'))
    d = pd.read_csv(f)
    return dict(zip(d['id'], zip(d['team_h_difficulty'],
                                 d['team_a_difficulty'])))


def rows(season):
    """One row per player per fixture, with stable club codes for both sides."""
    g = io.read_gw(season).merge(
        io.read_players(season)[['id', 'code', 'element_type']],
        left_on='element', right_on='id')
    g = g[g['element_type'].isin([1, 2, 3, 4])]
    g['kickoff'] = pd.to_datetime(g['kickoff_time'], format='mixed', utc=True)

    # every fixture has exactly two opponent ids; a row's own club is the other
    lo = g.groupby('fixture')['opponent_team'].transform('min')
    hi = g.groupby('fixture')['opponent_team'].transform('max')
    g['team'] = np.where(g['opponent_team'] == lo, hi, lo)

    ids = io.read_players(season).groupby('team')['team_code'].first()
    g['club'] = g['team'].map(ids)
    g['opp'] = g['opponent_team'].map(ids)
    g['scored'] = np.where(g['was_home'], g['team_h_score'], g['team_a_score'])
    g['conceded'] = np.where(g['was_home'], g['team_a_score'], g['team_h_score'])
    return g.drop_duplicates(['code', 'fixture'])[
        ['code', 'round', 'fixture', 'was_home', 'club', 'opp', 'kickoff',
         'element_type', 'scored', 'conceded']]


def deadlines(r):
    """gw -> first kickoff of that gameweek."""
    return r.groupby('round')['kickoff'].min().to_dict()


def team_log(season):
    """Every club game this season and last, for rolling opponent strength."""
    prev = io.SEASONS[io.SEASONS.index(season) - 1]
    both = pd.concat([rows(prev), rows(season)])
    return (both.drop_duplicates(['fixture', 'club', 'kickoff'])
            [['club', 'kickoff', 'scored', 'conceded']].sort_values('kickoff'))


def home_advantage(season):
    """Last season's home-minus-away goals per game."""
    prev = rows(io.SEASONS[io.SEASONS.index(season) - 1])
    games = prev.drop_duplicates(['fixture', 'club'])
    return (games.loc[games['was_home'], 'scored'].mean()
            - games.loc[~games['was_home'], 'scored'].mean())


def strength(log, before, games=STRENGTH_GAMES):
    """club -> (goals scored, conceded) per game over its last 10 before `before`."""
    past = log[log['kickoff'] < before].groupby('club').tail(games)
    return past.groupby('club')[['scored', 'conceded']].mean()


def _zscore(d, keys):
    """Standardise `d` over distinct (fixture, side) pairs, so no club counts twice."""
    base = d[~keys.duplicated()]
    sd = base.std()
    return (d - base.mean()) / sd if sd and sd > 0 else d * 0


def asof_ratings(season):
    """gw -> the FPL ratings as last scraped before that gameweek's deadline."""
    r = rows(season)
    snaps = commits(season)
    out = {}
    for gw, dl in deadlines(r).items():
        known = [s for t, s in snaps if t < dl]
        out[gw] = version(season, known[-1] if known else snaps[0][1])
    return out


def difficulty(season, source, horizon=HORIZON):
    """gw -> code -> mean z-scored difficulty of his fixtures in gw..gw+horizon-1."""
    r = rows(season)
    dl = deadlines(r)
    if source == 'fpl':
        ratings = asof_ratings(season)
    else:
        log, ha = team_log(season), home_advantage(season)

    out = {}
    for gw in sorted(dl):
        win = r[(r['round'] >= gw) & (r['round'] < gw + horizon)].copy()
        if win.empty:
            continue
        keys = win[['fixture', 'club']].apply(tuple, axis=1)

        if source == 'fpl':
            rate = ratings[gw]
            win['d'] = [rate.get(f, (np.nan, np.nan))[0 if h else 1]
                        for f, h in zip(win['fixture'], win['was_home'])]
            win['z'] = _zscore(win['d'], keys)
        else:
            st = strength(log, dl[gw])
            venue = np.where(win['was_home'], -ha / 2, ha / 2)
            attack = win['opp'].map(st['scored'])
            leaky = -win['opp'].map(st['conceded'])
            back = win['element_type'].isin([1, 2])
            win['d'] = np.where(back, attack, leaky) + venue
            win['z'] = np.nan
            for mask in (back, ~back):
                win.loc[mask, 'z'] = _zscore(win.loc[mask, 'd'], keys[mask])
        out[gw] = win.groupby('code')['z'].mean().fillna(0.0)
    return out


def adjust(ppg, opp_z, k):
    """adj_ppg = ppg * (1 + K * (league average - his fixtures)); the average is 0."""
    z = pd.Series(opp_z).reindex(pd.Series(ppg).index).fillna(0.0)
    return pd.Series(ppg) * (1 - k * z)


def revisions(season):
    """Share of fixtures whose rating differs between the first and last scrape."""
    snaps = commits(season)
    first, last = version(season, snaps[0][1]), version(season, snaps[-1][1])
    common = set(first) & set(last)
    changed = sum(first[f] != last[f] for f in common)
    return changed, len(common)
