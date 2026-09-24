"""Squad selection. MILP picks 15 under the FPL constraints, then the best
valid XI is chosen from those 15 and the captain doubled.

Prices are START-of-season (the `value` at the player's first match of t+1),
not `now_cost` from players_raw, which is an end-of-season figure and would
leak how well the player did that year.
"""

import itertools

import pandas as pd
import pulp

BUDGET = 1000           # tenths of a million
SQUAD = {1: 2, 2: 5, 3: 5, 4: 3}            # GK, DEF, MID, FWD
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
MAX_PER_CLUB = 3


def start_prices(gw_next):
    """code -> price at his first match of the season, in tenths."""
    g = gw_next.sort_values('kickoff_time')
    return g.groupby('code')['value'].first()


# ---------------------------------------------------------------- squad

def pick_squad(cands, points_col, budget=BUDGET, max_per_club=MAX_PER_CLUB):
    """15 players under budget, position and club constraints, chosen to
    maximise the best STARTING XI's `points_col` -- not the sum across all 15.

    Top-heavy by construction: only the 11 starters score, so the solver has
    no reason to spend on the bench and fills it with the cheapest legal
    bodies. The squad-level constraints (budget, 2/5/5/3, 3-per-club) still
    govern which 15 can be bought; they just don't drive the objective.

    Formation bounds come from XI_MIN/XI_MAX, the same constants best_xi()
    enumerates, so the two cannot disagree about what a legal XI is.
    """
    c = cands.reset_index(drop=True)
    x = pulp.LpVariable.dicts('x', c.index, cat='Binary')      # in the 15
    s = pulp.LpVariable.dicts('start', c.index, cat='Binary')  # in the XI

    prob = pulp.LpProblem('squad', pulp.LpMaximize)
    prob += pulp.lpSum(c.loc[i, points_col] * s[i] for i in c.index)
    prob += pulp.lpSum(c.loc[i, 'price'] * x[i] for i in c.index) <= budget

    for pos, n in SQUAD.items():
        idx = c.index[c['element_type'] == pos]
        prob += pulp.lpSum(x[i] for i in idx) == n

    for club, idx in c.groupby('team_code').groups.items():
        prob += pulp.lpSum(x[i] for i in idx) <= max_per_club

    # you can only start someone you own
    for i in c.index:
        prob += s[i] <= x[i]

    prob += pulp.lpSum(s[i] for i in c.index) == 11
    for pos in SQUAD:
        idx = c.index[c['element_type'] == pos]
        prob += pulp.lpSum(s[i] for i in idx) >= XI_MIN[pos]
        prob += pulp.lpSum(s[i] for i in idx) <= XI_MAX[pos]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != 'Optimal':
        raise RuntimeError(f'solver status: {pulp.LpStatus[prob.status]}')

    return c.loc[[i for i in c.index if x[i].value() > 0.5]]


def formations():
    for d, m, f in itertools.product(range(3, 6), range(2, 6), range(1, 4)):
        if d + m + f == 10:
            yield {1: 1, 2: d, 3: m, 4: f}


def best_xi(squad, points_col):
    """Highest-scoring valid XI from the 15, plus the captain (the highest
    predicted player in that XI)."""
    best, best_total = None, -1e18
    for shape in formations():
        rows = [squad[squad['element_type'] == pos].nlargest(n, points_col)
                for pos, n in shape.items()]
        xi = pd.concat(rows)
        total = xi[points_col].sum()
        if total > best_total:
            best, best_total = xi, total
    captain = best.loc[best[points_col].idxmax(), 'code']
    return best, captain


def score(xi, captain, actual_col='actual_pts'):
    """Actual points of the XI, captain doubled."""
    return xi[actual_col].sum() + xi.loc[xi['code'] == captain, actual_col].sum()


def run(cands, points_col, actual_col='actual_pts', **kw):
    squad = pick_squad(cands, points_col, **kw)
    xi, captain = best_xi(squad, points_col)
    return {'squad': squad, 'xi': xi, 'captain': captain,
            'predicted': xi[points_col].sum(),
            'actual': score(xi, captain, actual_col),
            'spend': int(squad['price'].sum())}


def compare(cands_by_season, strategies, actual_col='actual_pts'):
    """One row per season per strategy. `strategies` maps a label to the
    column in `cands` to maximise."""
    rows = []
    for season, cands in cands_by_season.items():
        for label, col in strategies.items():
            r = run(cands, col, actual_col)
            rows.append({'season': season, 'strategy': label,
                         'actual': r['actual'], 'spend': r['spend'],
                         'captain': r['captain']})
    return pd.DataFrame(rows)