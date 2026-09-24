"""Multi-transfer planning and a data-driven wildcard, as MILPs.

transfer_plan() chooses every transfer at once over the whole pool, so a
two-for-two that moves money between positions is found; the greedy search in
backtest.best_move makes one swap at a time. The objective is the best XI's
5-gameweek projection, less SWITCH_MARGIN per transfer (so a plan stays
tentative) and 4 per hit.

    python -m form_lab.plan          # the three-season comparison
"""
import pulp

import optimise as op
from form_lab import form
from form_lab.backtest import HIT_COST
WC_MULTIPLES = (3, 5, 8)


def _problem(squad, bank, sell, free, max_hits, value, prices, pos, clubs,
             eligible, unavailable, margin, hit_cost, max_transfers, count):
    owned = set(squad)
    cands = [c for c in set(value) | owned
             if c in pos and c in clubs and pos[c] in op.SQUAD
             and (c in owned or (c in prices and c not in unavailable))]
    x = {c: pulp.LpVariable(f'x{c}', cat='Binary') for c in cands}
    s = {c: pulp.LpVariable(f's{c}', cat='Binary') for c in cands}
    h = pulp.LpVariable('hits', lowBound=0, upBound=max_hits, cat='Integer')
    buys = pulp.lpSum(x[c] for c in cands if c not in owned)

    prob = pulp.LpProblem('plan', pulp.LpMaximize)
    prob += (pulp.lpSum(value.get(c, 0.0) * s[c] for c in cands)
             - margin * buys - hit_cost * h)
    for p, n in op.SQUAD.items():
        group = [c for c in cands if pos[c] == p]
        prob += pulp.lpSum(x[c] for c in group) == n
        prob += pulp.lpSum(s[c] for c in group) >= op.XI_MIN[p]
        prob += pulp.lpSum(s[c] for c in group) <= op.XI_MAX[p]
    prob += pulp.lpSum(s.values()) == 11
    for club in {clubs[c] for c in cands}:
        prob += pulp.lpSum(x[c] for c in cands if clubs[c] == club) <= op.MAX_PER_CLUB
    prob += (pulp.lpSum(prices[c] * x[c] for c in cands if c not in owned)
             <= bank + pulp.lpSum(sell[c] * (1 - x[c]) for c in owned))
    prob += buys <= free + h
    if max_transfers is not None:
        prob += buys <= max_transfers
    if count is not None:
        prob += buys == count
    for c in cands:
        prob += s[c] <= x[c]
        # bench fodder: anyone bought onto the bench must be a regular starter
        if c not in owned and eligible is not None and c not in eligible:
            prob += x[c] <= s[c]
    return prob, x, s, h, owned


def _solve(squad, bank, sell, free, max_hits, count=None, **kw):
    prob, x, s, h, owned = _problem(squad, bank, sell, free, max_hits,
                                    count=count, **kw)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != 'Optimal':
        return None
    pick = {c for c, v in x.items() if v.value() > 0.5}
    xi = {c for c, v in s.items() if v.value() > 0.5}
    outs, ins = sorted(owned - pick), sorted(pick - owned)
    prices = kw['prices']
    return {'outs': outs, 'ins': ins, 'xi': xi, 'transfers': len(ins),
            'hits': int(round(h.value() or 0)),
            'cost': sum(prices[c] for c in ins) - sum(sell[c] for c in outs),
            'value': sum(kw['value'].get(c, 0.0) for c in xi),
            'objective': pulp.value(prob.objective)}


def transfer_plan(squad, bank, selling_prices, free_transfers, max_hits, *,
                  value, prices, pos, clubs, eligible=None, unavailable=(),
                  margin=form.SWITCH_MARGIN, hit_cost=HIT_COST,
                  max_transfers=None, by_count=True):
    """(best plan, {transfer count: plan}) for this gameweek.

    `value` is each player's 5-gameweek projection, `prices` what a player
    costs to buy, `selling_prices` what each owned player fetches. A plan
    has outs, ins, cost, gain (the best XI's value over keeping the squad),
    hits and transfers. Hits are the transfers beyond `free_transfers`.
    """
    kw = dict(value=value, prices=prices, pos=pos, clubs=clubs,
              eligible=eligible, unavailable=set(unavailable), margin=margin,
              hit_cost=hit_cost, max_transfers=max_transfers)
    args = (squad, bank, selling_prices, free_transfers, max_hits)
    base = _solve(*args, count=0, **kw)
    by = {0: base}
    if by_count:
        for n in range(1, free_transfers + max_hits + 1):
            by[n] = _solve(*args, count=n, **kw)
        best = max((p for p in by.values() if p), key=lambda p: p['objective'])
    else:
        best = _solve(*args, **kw)
    for p in by.values():
        if p:
            p['gain'] = p['value'] - base['value']
    best['gain'] = best['value'] - base['value']
    return best, by


def wildcard_plan(squad, bank, selling_prices, **kw):
    """The squad a wildcard would build: unlimited transfers, no hit or margin."""
    kw.update(margin=0, hit_cost=0, max_transfers=None, by_count=False)
    best, _ = transfer_plan(squad, bank, selling_prices, 15, 0, **kw)
    return best


def wildcard_gain(normal, wildcard, hit_cost=HIT_COST):
    """Points a wildcard adds over the best normal plan, net of its hits."""
    return wildcard['value'] - (normal['value'] - hit_cost * normal['hits'])


VARIANTS = {'A: greedy + fixed WC': dict(transfers='greedy', wc_threshold=None),
            'B: plan + fixed WC': dict(transfers='plan', wc_threshold=None)}
VARIANTS.update({f'C: plan + WC >= {k}x margin':
                 dict(transfers='plan', wc_threshold=k * form.SWITCH_MARGIN)
                 for k in WC_MULTIPLES})


def compare(seasons=None, variants=VARIANTS):
    """Every variant over every season, chips and hits on, blend projections."""
    import pandas as pd
    from form_lab import backtest as bt

    seasons = seasons or bt.SEASONS
    rows = []
    for season in seasons:
        hist, clubs, pos, pri = bt.season_data(season)
        for name, kw in variants.items():
            d, chip_pts = bt.simulate('blend', season, hist, clubs, pos, pri,
                                      chips=True, **kw)
            wc = ' '.join(f'GW{g}' for g, c in zip(d['gw'], d['chip'])
                          if c == 'wildcard')
            rows.append({'season': season, 'variant': name,
                         'total': int(d['points'].sum()),
                         'transfers': int(d['transfers'].sum()),
                         'hits': int(d['hits'].sum()), 'wildcards': wc})
            print(f'  {season} {name:28s} {rows[-1]["total"]:5d}  '
                  f'tr {rows[-1]["transfers"]:3d}  hits {rows[-1]["hits"]:2d}  '
                  f'WC {wc}', flush=True)
    return pd.DataFrame(rows)


def decide(table, baseline='A: greedy + fixed WC', need=50, wins=2):
    """Adopt a variant only if it beats A by `need` over all seasons AND wins
    at least `wins` seasons; otherwise keep A."""
    wide = table.pivot(index='variant', columns='season', values='total')
    base = wide.loc[baseline]
    verdicts = {}
    for v in wide.index.drop(baseline):
        margin = wide.loc[v].sum() - base.sum()
        won = int((wide.loc[v] > base).sum())
        verdicts[v] = (margin, won, margin >= need and won >= wins)
    passing = {v: m for v, (m, _, ok) in verdicts.items() if ok}
    return (max(passing, key=passing.get) if passing else baseline), verdicts


def main():
    import pandas as pd
    from form_lab import data

    table = compare()
    wide = table.pivot(index='variant', columns='season', values='total')
    wide['3-season'] = wide.sum(axis=1)
    agg = table.groupby('variant')[['transfers', 'hits']].sum()
    wc = table.pivot(index='variant', columns='season', values='wildcards')
    wc.columns = [f'WC {c}' for c in wc.columns]
    out = pd.concat([wide, agg, wc], axis=1)
    print('\n' + out.to_string())

    winner, verdicts = decide(table)
    print()
    for v, (m, won, ok) in verdicts.items():
        print(f'  {v:28s} {m:+5d} vs A, wins {won}/3 -> '
              f'{"passes" if ok else "fails"}')
    print(f'\nadopt: {winner}')

    res = data.CACHE.parent / 'results'
    res.mkdir(parents=True, exist_ok=True)
    table.to_csv(res / 'plan_backtest.csv', index=False)
    out.to_csv(res / 'plan_backtest_summary.csv')
    return out, winner


if __name__ == '__main__':
    main()
