"""Free hit: a one-gameweek squad, played when the week breaks the normal XI.

    python -m form_lab.freehit        # the three-season comparison

Triggers use only pre-deadline data: (a) 3+ starters with no fixture,
(b) 3+ starters who played 0 minutes in each of their team's last 2 games,
(c) 4+ of the XI in one fixture between two top-6 sides by rolling goal
difference. Any trigger also needs the FH XI to beat the current XI by
FH_THRESHOLD on the one-gameweek projection. Fallback: the window's last
gameweek not taken by triple captain or bench boost.
"""
import pandas as pd

from form_lab import backtest as bt, fixtures as fx

THRESHOLDS = (5, 10, 15)
TRIGGER_SETS = ('a', 'b', 'c', 'abc')
MIN_BLANK, MIN_OUT, MIN_STACK = 3, 3, 4
TOP, STALE_GAMES, STRENGTH_GAMES = 6, 2, 10
FALLBACK_OFFSET = 2          # triple captain falls back to the last GW, bench boost the one before
BASELINE = 'A: no free hit'


def blank_starters(starters, nfix, gw):
    """Starters whose club has no fixture this gameweek."""
    return [c for c in starters if nfix.get((c, gw), 0) == 0]


def stale(hist, gw, games=STALE_GAMES):
    """Codes on 0 minutes in each of their team's last `games` games before `gw`."""
    last = hist[hist['gw'] < gw].groupby('code').tail(games)
    g = last.groupby('code')['minutes']
    ok = (g.size() >= games) & (g.max() == 0)
    return set(ok[ok].index)


def top_clubs(log, before, n=TOP, games=STRENGTH_GAMES):
    """The n clubs with the best goal difference over their last `games`."""
    st = fx.strength(log, before, games)
    return set((st['scored'] - st['conceded']).nlargest(n).index)


def big_game_stack(xi, rows_at_gw, top):
    """Most of the XI in any one fixture between two top clubs."""
    r = rows_at_gw[rows_at_gw['code'].isin(xi) & rows_at_gw['club'].isin(top)
                   & rows_at_gw['opp'].isin(top)]
    return int(r.groupby('fixture')['code'].nunique().max()) if len(r) else 0


def budget(squad, bank, prices):
    """Selling value plus bank: what a free hit can spend."""
    return bank + sum(bt.sell_price(b, prices.get(c, b)) for c, b in squad.items())


def season_context(season):
    """Fixture counts, fixture rows, club log and deadlines, built once."""
    rows = fx.rows(season)
    return {'nfix': bt.known_fixtures(season, None), 'rows': rows,
            'log': fx.team_log(season), 'deadlines': fx.deadlines(rows)}


def check(gw, hi, squad, bank, proj, prices, pos, clubs, eligible, hist, ctx, cfg):
    """The free hit to play this gameweek, or None."""
    nfix = ctx['nfix']
    starters = bt.xi_members(squad, proj, pos)
    out = stale(hist, gw)
    fired = []
    if 'a' in cfg['triggers'] and len(blank_starters(starters, nfix, gw)) >= MIN_BLANK:
        fired.append('a')
    if 'b' in cfg['triggers'] and len(starters & out) >= MIN_OUT:
        fired.append('b')
    if 'c' in cfg['triggers']:
        top = top_clubs(ctx['log'], ctx['deadlines'][gw])
        at = ctx['rows'][ctx['rows']['round'] == gw]
        if big_game_stack(starters, at, top) >= MIN_STACK:
            fired.append('c')
    fallback = gw == hi - FALLBACK_OFFSET
    if not fired and not fallback:
        return None

    value = {c: v * nfix.get((c, gw), 0) * (c not in out) for c, v in proj.items()}
    spend = budget(squad, bank, prices)
    cands = pd.DataFrame({'code': [c for c in value if c in prices and c in clubs
                                   and pos.get(c) in (1, 2, 3, 4)]})
    cands['element_type'] = cands['code'].map(pos)
    cands['team_code'] = cands['code'].map(clubs)
    cands['price'] = cands['code'].map(prices)
    cands['pred'] = cands['code'].map(value)
    try:
        new = bt.pick_squad(cands.dropna(), 'pred', eligible, budget=int(spend))
    except RuntimeError:
        return None
    new = {int(r.code): int(r.price) for r in new.itertuples()}
    gain = bt.xi_value(new, value, pos) - bt.xi_value(squad, value, pos)
    if (fired and gain >= cfg['threshold']) or (not fired and gain > 0):
        return {'squad': new, 'budget': spend, 'value_gain': gain,
                'trigger': ''.join(fired) or 'fallback'}
    return None


def variants():
    out = {BASELINE: None}
    for t in THRESHOLDS:
        for trig in TRIGGER_SETS:
            out[f'FH {trig} >= {t}'] = {'threshold': t, 'triggers': trig}
    return out


def compare(seasons=None):
    """Every variant over every season: blend, chips and hits, fixed wildcard."""
    seasons = seasons or bt.SEASONS
    rows = []
    for season in seasons:
        hist, clubs, pos, pri = bt.season_data(season)
        for name, cfg in variants().items():
            d, _ = bt.simulate('blend', season, hist, clubs, pos, pri,
                               chips=True, free_hit=cfg)
            fh = d[d['chip'] == 'freehit']
            rows.append({'season': season, 'variant': name,
                         'total': int(d['points'].sum()),
                         'fh': ' '.join(f"GW{r.gw}({r.fh_trigger})" for r in fh.itertuples()),
                         'fh_gain': int(fh['fh_gain'].sum())})
            print(f"  {season} {name:14s} {rows[-1]['total']:5d}  {rows[-1]['fh']}  "
                  f"{rows[-1]['fh_gain']:+d}", flush=True)
    return pd.DataFrame(rows)


def decide(table, need=30, wins=2):
    """Adopt a variant only with +need over all seasons AND `wins` season wins."""
    wide = table.pivot(index='variant', columns='season', values='total')
    base = wide.loc[BASELINE]
    verdicts = {v: (int(wide.loc[v].sum() - base.sum()), int((wide.loc[v] > base).sum()))
                for v in wide.index.drop(BASELINE)}
    passing = {v: m for v, (m, w) in verdicts.items() if m >= need and w >= wins}
    return (max(passing, key=passing.get) if passing else BASELINE), verdicts


def main():
    from form_lab import data
    table = compare()
    wide = table.pivot(index='variant', columns='season', values='total')
    wide['3-season'] = wide.sum(axis=1)
    wide['FH gain'] = table.groupby('variant')['fh_gain'].sum()
    wide = wide.sort_values('3-season', ascending=False)
    print('\n' + wide.to_string())
    winner, verdicts = decide(table)
    print(f'\nadopt: {winner}')
    res = data.CACHE.parent / 'results'
    table.to_csv(res / 'freehit_backtest.csv', index=False)
    wide.to_csv(res / 'freehit_summary.csv')
    return table, winner, verdicts


if __name__ == '__main__':
    main()
