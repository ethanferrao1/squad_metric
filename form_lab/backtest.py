"""Full-season simulations of 23/24, 24/25 and 25/26.

    python -m form_lab.backtest

Each strategy plays each season under the same rules from the same GW1
budget. The prior for a season comes from holdout.fit_predict with that
season held out, so none of its outcomes reach the model; every gameweek
decision uses only earlier gameweeks and fixtures knowable in advance.

No free hit, no injury data (a player is unavailable only if he recorded 0
minutes), and simplified auto-subs.
"""
import json

import numpy as np
import pandas as pd

import data_io as io
import optimise as op
from form_lab import data, form

# 22_23 is absent on purpose: its prior would need 21_22 features, and 21_22
# is a bad feature season, so no model prediction for 22_23 exists.
SEASONS = ['23_24', '24_25', '25_26']
HORIZON = 5
MAX_FREE = 5
CAND_PER_SLOT = 10
CHASER_WINDOW = 5

HIT_COST = 4
MAX_HITS = 2

HALVES = [(1, 19), (20, 38)]

# the wildcard plays on these gameweeks, one per half, and nowhere else
WILDCARD_FALLBACK = {1: 16, 2: 35}
BB_MIN_DOUBLES = 4
FIXTURE_LOOKAHEAD = 4

# (source, K) for the blend's fixture adjustment (form_lab.fixtures); None = off
FIXTURE = None

# bench fodder must be a regular starter: 60+ minutes in half his last 5 games
BENCH_WINDOW = 5
START_MINUTES = 60

# fallback order when a chip never triggers: the last gameweek of its window,
# then the one before, so two pending chips do not collide on the same deadline
CHIP_PRIORITY = ['tcaptain', 'bboost']

# Chip sets per season. From 2025/26 every chip resets at the halfway point;
# before that only the wildcard did. The 24/25 Assistant Manager chip is not
# modelled -- it scores from a real manager's club, which this sim has no
# concept of.
SINGLE = {'wildcard': HALVES, 'bboost': [(1, 38)], 'tcaptain': [(1, 38)],
          'freehit': [(2, 38)]}
DOUBLE = {'wildcard': HALVES, 'bboost': HALVES, 'tcaptain': HALVES,
          'freehit': [(2, 19), (20, 38)]}
CHIP_RULES = {'23_24': SINGLE, '24_25': SINGLE, '25_26': DOUBLE}

# adopted from form_lab.freehit (+110 over 3 seasons); pass free_hit=None to switch off
FREE_HIT = {'threshold': 15, 'triggers': 'a'}
FREE_HIT_NOTE = ('free-hit rule chosen as best of 12 variants, expect slight '
                 'optimism')

# reference palette, categorical slots in their documented order
COLOURS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb'

STRATEGIES = ['blend', 'form only', 'no transfers', 'form chaser']


def feature_season(season):
    """The season whose features predict `season`."""
    return io.SEASONS[io.SEASONS.index(season) - 1]


def chart_path(season):
    return data.CACHE.parent / f'backtest_{season}.png'


def prior_points(season, refresh=False):
    """code -> predicted points for `season`, from a model that never saw it."""
    cache = data.CACHE / f'holdout_prior_{season}.parquet'
    if cache.exists() and not refresh:
        return pd.read_parquet(cache).set_index('code')['prior_pts']

    import holdout as ho
    test, train_seasons = ho.fit_predict(feature_season(season))
    print(f'{season}: prior trained on targets '
          f'{train_seasons[0]}-{train_seasons[-1]}')
    assert season not in train_seasons, 'the prior saw the simulated season'
    assert all(s < season for s in train_seasons), 'a later season leaked in'

    out = pd.DataFrame({'code': test['code'].to_numpy(),
                        'prior_pts': test['pred'].to_numpy()})
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out.set_index('code')['prior_pts']


def season_data(season):
    """History, club and position lookups, and a prior for every player."""
    hist = data.gw_history(season)
    players = io.read_players(season).drop_duplicates('code')
    clubs = players.set_index('code')['team_code']
    pos = hist.drop_duplicates('code').set_index('code')['element_type']

    # 24/25 added element_type 5, the manager. op.SQUAD constrains only 1-4,
    # so leaving them in lets the MILP buy any number of them.
    pos = pos[pos.isin(op.SQUAD)]
    hist = hist[hist['code'].isin(pos.index)]

    pri = prior_points(season).reindex(hist['code'].unique())
    # a player with no prior is a promoted or newly signed one; give him the
    # median for his position so he can still be bought, not a forecast
    med = pri.groupby(pos.reindex(pri.index)).transform('median')
    pri = pri.fillna(med).fillna(pri.median())

    hist = hist[hist['code'].isin(clubs.dropna().index)]
    return hist, clubs, pos, pri


def projections(hist, gw, pri, strategy):
    """code -> the number this strategy ranks players by, using only GW < gw."""
    snap = form.snapshot(hist, gw, pri)
    if snap.empty:
        return pd.Series(dtype=float)

    if strategy == 'form only':
        return snap.set_index('code')['form_ppg'].fillna(
            snap.set_index('code')['prior_ppg'])
    if strategy == 'form chaser':
        past = hist[hist['gw'] < gw]
        recent = past.groupby('code').tail(CHASER_WINDOW)
        return recent.groupby('code')['points'].mean()

    p = form.project(snap, w_max=form.W_MAX, market_weight=form.MARKET_WEIGHT)
    return p.set_index('code')['blended_ppg']


def gw_prices(hist, gw):
    """code -> price during `gw`, falling back to his last known price."""
    upto = hist[hist['gw'] <= gw].sort_values('gw')
    return upto.groupby('code')['price'].last()


def sell_price(buy, now):
    """FPL's rule: you keep half of any rise, rounded down to 0.1m."""
    return int(buy + (now - buy) // 2) if now > buy else int(now)


def legal_shape(counts):
    """True if these position counts are a playable XI."""
    return any(counts == shape for shape in op.formations())


def apply_autosubs(xi, bench, minutes, pos):
    """Swap out starters who did not play, keeping the formation legal."""
    xi, bench = list(xi), list(bench)
    for starter in [c for c in xi if minutes.get(c, 0) == 0]:
        for sub in bench:
            if minutes.get(sub, 0) == 0:
                continue
            trial = [c for c in xi if c != starter] + [sub]
            counts = pd.Series([pos[c] for c in trial]).value_counts().to_dict()
            if counts.get(1, 0) == 1 and legal_shape(
                    {k: v for k, v in counts.items()}):
                xi, bench = trial, [c for c in bench if c != sub] + [starter]
                break
    return xi, bench


def pick_xi(squad_codes, proj, pos, playing, bench_order=None):
    """Best legal XI from those with a fixture, plus captain and vice.

    The bench is ordered by `bench_order` (the blended projection) when given.
    """
    avail = [c for c in squad_codes if c in playing]
    frame = pd.DataFrame({'code': avail,
                          'element_type': [pos[c] for c in avail],
                          'p': [proj.get(c, 0.0) for c in avail]})
    if len(frame) < 11 or not legal_enough(frame):
        # too few with a fixture to field a legal XI; take the best available
        frame = pd.DataFrame({'code': list(squad_codes),
                              'element_type': [pos[c] for c in squad_codes],
                              'p': [proj.get(c, 0.0) for c in squad_codes]})
    xi, captain = op.best_xi(frame, 'p')
    order = xi.sort_values('p', ascending=False)['code'].tolist()
    vice = order[1] if len(order) > 1 else captain
    bench = [c for c in squad_codes if c not in set(xi['code'])]
    rank = bench_order if bench_order is not None else proj
    bench.sort(key=lambda c: -rank.get(c, 0.0))
    return xi['code'].tolist(), bench, captain, vice


def legal_enough(frame):
    """Can a legal XI be built from this frame at all?"""
    have = frame['element_type'].value_counts().to_dict()
    return any(all(have.get(p, 0) >= n for p, n in shape.items())
               for shape in op.formations())


FORMATIONS = list(op.formations())


def xi_value(codes, proj, pos):
    """Projected points of the best XI from these 15 -- the transfer objective.

    The same answer op.best_xi gives, without building a frame per call: this
    runs once per candidate transfer, tens of thousands of times a season.
    """
    by_pos = {1: [], 2: [], 3: [], 4: []}
    for c in codes:
        by_pos[pos[c]].append(proj.get(c, 0.0))
    for v in by_pos.values():
        v.sort(reverse=True)

    best = -1e9
    for shape in FORMATIONS:
        if any(len(by_pos[p]) < n for p, n in shape.items()):
            continue
        best = max(best, sum(sum(by_pos[p][:n]) for p, n in shape.items()))
    return best


def xi_members(codes, proj, pos):
    """The codes in the best XI from these players, as xi_value picks it."""
    by_pos = {1: [], 2: [], 3: [], 4: []}
    for c in codes:
        by_pos[pos[c]].append((proj.get(c, 0.0), c))
    for v in by_pos.values():
        v.sort(reverse=True)

    best, members = -1e9, set()
    for shape in FORMATIONS:
        if any(len(by_pos[p]) < n for p, n in shape.items()):
            continue
        total = sum(v for p, n in shape.items() for v, _ in by_pos[p][:n])
        if total > best:
            best = total
            members = {c for p, n in shape.items() for _, c in by_pos[p][:n]}
    return members


def regular_starters(hist, gw, prev=None, window=BENCH_WINDOW):
    """Codes with 60+ minutes in at least half their team's last 5 games.

    Before any game this season: 60+ minutes in half of last season's games.
    """
    past = hist[hist['gw'] < gw]
    if past.empty:
        if prev is None or prev.empty:
            return set()
        starts = (prev['minutes'] >= START_MINUTES).groupby(prev['code']).sum()
        return set(starts[starts >= data.SEASON_GWS / 2].index)

    recent = past.groupby('code').tail(window)
    started = (recent['minutes'] >= START_MINUTES).groupby(recent['code'])
    ok = started.sum() * 2 >= started.size()
    return set(ok[ok].index)


def rank_by_position(proj, pos):
    """position -> codes ranked by projection, built once per gameweek."""
    out = {1: [], 2: [], 3: [], 4: []}
    for code, value in sorted(proj.items(), key=lambda kv: -kv[1]):
        p = pos.get(code)
        if p in out:
            out[p].append(code)
    return out


def best_move(squad, bank, proj, prices, pos, clubs, margin, ranked,
              chaser=False, paid=False, eligible=None):
    """The single best legal transfer, or None if none clears the margin.

    The chaser ignores the margin on a free transfer, by design, but a paid
    one still has to clear it -- otherwise it buys -4s all season. A player
    bought onto the bench must be a regular starter (`eligible`).
    """
    owned = set(squad)
    base = xi_value(owned, proj, pos)
    club_count = pd.Series([clubs[c] for c in owned]).value_counts()

    best = None
    for out_code in squad:
        sell = sell_price(squad[out_code], prices.get(out_code, squad[out_code]))
        budget = bank + sell
        same_pos = []
        for c in ranked[pos[out_code]]:
            if c in owned or prices.get(c, 1e9) > budget:
                continue
            same_pos.append(c)
            if len(same_pos) >= CAND_PER_SLOT:
                break

        for in_code in same_pos:
            club = clubs[in_code]
            held = club_count.get(club, 0) - (1 if clubs[out_code] == club else 0)
            if held >= op.MAX_PER_CLUB:
                continue

            if chaser:
                gain = (proj.get(in_code, 0) - proj.get(out_code, 0)) * HORIZON
                if gain <= 0 or (paid and gain < margin):
                    continue
            else:
                trial = (owned - {out_code}) | {in_code}
                gain = (xi_value(trial, proj, pos) - base) * HORIZON
                if gain < margin:
                    continue
            if best is not None and gain <= best['gain']:
                continue
            if eligible is not None and in_code not in eligible:
                trial = (owned - {out_code}) | {in_code}
                if in_code not in xi_members(trial, proj, pos):
                    continue
            if best is None or gain > best['gain']:
                best = {'out': out_code, 'in': in_code, 'gain': gain,
                        'sell': sell, 'buy': int(prices[in_code])}
    return best


def half_of(gw):
    """1 for the first half of the season, 2 for the second."""
    return 1 if gw <= HALVES[0][1] else 2


def open_chips(season, gw, used):
    """Chips whose window contains `gw` and which are still unused there."""
    out = []
    for chip, windows in CHIP_RULES[season].items():
        for lo, hi in windows:
            if lo <= gw <= hi and (chip, lo) not in used:
                out.append((chip, lo))
    return out


def known_counts(rows, lookahead=FIXTURE_LOOKAHEAD):
    """(code, gw) -> fixtures knowable `lookahead` gameweeks before the deadline.

    merged_gw holds only the final schedule, but a fixture id still encodes
    its original round (ids run 10 per round). A fixture played in its
    original round is known all season; a rearranged one counts only if its
    original slot passed `lookahead`+ gameweeks earlier. Duplicate rows share
    an id and count once.
    """
    r = rows.drop_duplicates(['code', 'round', 'fixture'])
    if lookahead is None:
        # optimistic: every fixture in the final schedule counts as known
        return r.groupby(['code', 'round']).size().to_dict()
    orig = np.ceil(r['fixture'] / 10)
    known = (orig == r['round']) | (orig <= r['round'] - lookahead)
    return r[known].groupby(['code', 'round']).size().to_dict()


def known_fixtures(season, lookahead=FIXTURE_LOOKAHEAD):
    """known_counts for a whole season's merged_gw."""
    g = io.read_gw(season).merge(io.read_players(season)[['id', 'code']],
                                 left_on='element', right_on='id')
    return known_counts(g[['code', 'round', 'fixture']], lookahead)


def tc_ready(captain, gw, known):
    """The captain has a known double gameweek."""
    return known.get((captain, gw), 0) >= 2


def bb_ready(squad, bench, gw, known, min_doubles=BB_MIN_DOUBLES):
    """4+ of the 15 have a known double and every bench player has a fixture."""
    doubles = sum(1 for c in squad if known.get((c, gw), 0) >= 2)
    return (doubles >= min_doubles
            and all(known.get((c, gw), 0) >= 1 for c in bench))


def pick_squad(cands, col, eligible, budget=op.BUDGET):
    """op.pick_squad, plus: anyone not a regular starter must be in the XI."""
    import pulp

    c = cands.reset_index(drop=True)
    x = pulp.LpVariable.dicts('x', c.index, cat='Binary')
    s = pulp.LpVariable.dicts('s', c.index, cat='Binary')
    prob = pulp.LpProblem('squad', pulp.LpMaximize)
    prob += pulp.lpSum(c.at[i, col] * s[i] for i in c.index)
    prob += pulp.lpSum(c.at[i, 'price'] * x[i] for i in c.index) <= budget
    prob += pulp.lpSum(s[i] for i in c.index) == 11
    for p, n in op.SQUAD.items():
        idx = c.index[c['element_type'] == p]
        prob += pulp.lpSum(x[i] for i in idx) == n
        prob += pulp.lpSum(s[i] for i in idx) >= op.XI_MIN[p]
        prob += pulp.lpSum(s[i] for i in idx) <= op.XI_MAX[p]
    for _, idx in c.groupby('team_code').groups.items():
        prob += pulp.lpSum(x[i] for i in idx) <= op.MAX_PER_CLUB
    for i in c.index:
        prob += s[i] <= x[i]
        if c.at[i, 'code'] not in eligible:
            prob += x[i] <= s[i]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != 'Optimal':
        raise RuntimeError(f'solver status: {pulp.LpStatus[prob.status]}')
    return c.loc[[i for i in c.index if x[i].value() > 0.5]]


def wildcard_squad(squad, bank, proj, prices, pos, clubs, eligible):
    """The MILP rebuild a wildcard buys, at selling prices plus the bank."""
    budget = bank + sum(sell_price(buy, prices.get(c, buy))
                        for c, buy in squad.items())
    codes = [c for c in proj if c in prices and c in clubs and pos.get(c)]
    cands = pd.DataFrame({'code': codes})
    cands['element_type'] = cands['code'].map(pos)
    cands['team_code'] = cands['code'].map(clubs)
    cands['price'] = cands['code'].map(prices)
    cands['pred'] = cands['code'].map(proj)
    cands = cands.dropna()
    if len(cands) < 30:
        return None

    try:
        built = pick_squad(cands, 'pred', eligible, budget=int(budget))
    except RuntimeError:
        return None
    return {int(r.code): int(r.price) for r in built.itertuples()}


def assert_legal(squad, bank, clubs, pos):
    """The squad must stay a squad you could actually own."""
    assert len(squad) == 15, f'{len(squad)} players'
    counts = pd.Series([pos[c] for c in squad]).value_counts().to_dict()
    assert counts == op.SQUAD, f'illegal shape {counts}'
    per_club = pd.Series([clubs[c] for c in squad]).value_counts()
    assert per_club.max() <= op.MAX_PER_CLUB, f'{per_club.max()} from one club'
    assert bank >= 0, f'bank {bank}'


def apply_plan(squad, bank, plan, prices, sells):
    """Make a plan's transfers in place. Returns (bank, transfers made)."""
    for c in plan['outs']:
        bank += sells[c]
        del squad[c]
    for c in plan['ins']:
        squad[c] = int(prices[c])
        bank -= int(prices[c])
    return bank, len(plan['ins'])


def gw_points(codes, proj, pos, playing, minutes, points, order):
    """What a squad scores this gameweek: best XI, auto-subs, captain doubled."""
    xi, bench, cap, vice = pick_xi(list(codes), proj, pos, playing, order)
    xi, bench = apply_autosubs(xi, bench, minutes, pos)
    scorer = cap if minutes.get(cap, 0) > 0 else vice
    return sum(points.get(c, 0) for c in xi) + points.get(scorer, 0)


def simulate(strategy, season, hist, clubs, pos, pri, margin=None, chips=False,
             transfers='greedy', wc_threshold=None, free_hit=FREE_HIT, start=None):
    """Play the whole season with one strategy. Returns (gameweeks, chip points).

    `transfers` is 'greedy' (one best swap at a time) or 'plan' (the MILP in
    form_lab.plan). `wc_threshold` None plays the wildcard on its fixed
    gameweeks; a number plays it when the MILP rebuild beats the best normal
    plan by that many points over the horizon. `free_hit` is None (never) or
    {'threshold', 'triggers'} for form_lab.freehit.
    """
    from form_lab import plan as mp
    fh_ctx = None
    if free_hit and chips:
        from form_lab import freehit as fhm
        fh_ctx = fhm.season_context(season)

    margin = form.SWITCH_MARGIN if margin is None else margin
    chaser = strategy == 'form chaser'
    transfers_on = strategy != 'no transfers'
    use_plan = transfers == 'plan' or wc_threshold is not None
    pos_d, clubs_d = pos.to_dict(), clubs.dropna().to_dict()
    used_chips, chip_pts = set(), {}
    prev = data.gw_history(feature_season(season))
    known = known_fixtures(season, FIXTURE_LOOKAHEAD) if chips else {}
    fix = None
    if FIXTURE and strategy == 'blend':
        from form_lab import fixtures
        fix = fixtures.difficulty(season, FIXTURE[0])

    gw1 = hist[hist['gw'] == 1]
    if start is None:
        start = op.start_prices(
            io.read_gw(season).merge(io.read_players(season)[['id', 'code']],
                                     left_on='element', right_on='id')
            .assign(kickoff_time=lambda d: pd.to_datetime(
                d['kickoff_time'], format='mixed', utc=True)))

    cands = pd.DataFrame({'code': gw1['code'].to_numpy()})
    cands['element_type'] = cands['code'].map(pos)
    cands['team_code'] = cands['code'].map(clubs)
    cands['price'] = cands['code'].map(start)
    cands['pred'] = cands['code'].map(pri)
    cands = cands.dropna()

    first = pick_squad(cands, 'pred', regular_starters(hist, 1, prev))
    squad = {int(r.code): int(r.price) for r in first.itertuples()}
    bank = op.BUDGET - sum(squad.values())
    free = 1

    rows = []
    for gw in range(1, int(hist['gw'].max()) + 1):
        this = hist[hist['gw'] == gw]
        playing = set(this['code'])
        points = this.set_index('code')['points'].to_dict()
        minutes = this.set_index('code')['minutes'].to_dict()
        prices = gw_prices(hist, gw).to_dict()
        series = projections(hist, gw, pri, strategy) if gw > 1 \
            else pri.reindex(list(squad)).fillna(0)
        if fix is not None and gw > 1:
            series = fixtures.adjust(series, fix.get(gw, {}), FIXTURE[1])
        proj = {int(k): float(v) for k, v in series.items() if pd.notna(v)}
        ranked = rank_by_position(proj, pos)
        eligible = regular_starters(hist, gw, prev)
        if strategy in ('blend', 'no transfers') or gw == 1:
            order = proj
        else:
            order = projections(hist, gw, pri, 'blend').to_dict()

        normal = None
        if use_plan and gw > 1 and transfers_on:
            sells = {c: sell_price(b, prices.get(c, b)) for c, b in squad.items()}
            plan_kw = dict(value={c: v * HORIZON for c, v in proj.items()},
                           prices=prices, pos=pos_d, clubs=clubs_d,
                           eligible=eligible)
            normal, _ = mp.transfer_plan(squad, bank, sells, min(free, MAX_FREE),
                                         MAX_HITS, margin=margin, by_count=False,
                                         **plan_kw)

        chip, window = None, None
        if chips and gw > 1 and transfers_on:
            if wc_threshold is None:
                chip, window = pick_wildcard(season, gw, used_chips, squad,
                                             bank, proj, prices, pos, clubs,
                                             eligible)
            else:
                lo = next((l for c, l in open_chips(season, gw, used_chips)
                           if c == 'wildcard'), None)
                if lo is not None:
                    wc = mp.wildcard_plan(squad, bank, sells, **plan_kw)
                    if mp.wildcard_gain(normal, wc) >= wc_threshold:
                        chip, window = 'wildcard plan', (lo, wc)

        fh, saved = None, None
        start_squad, start_bank = sorted(squad), bank
        if fh_ctx and chip is None and gw > 1:
            lo = next((l for c, l in open_chips(season, gw, used_chips)
                       if c == 'freehit'), None)
            if lo is not None:
                hi = next(h for l, h in CHIP_RULES[season]['freehit'] if l == lo)
                fh = fhm.check(gw, hi, squad, bank, proj, prices, pos_d, clubs_d,
                               eligible, hist, fh_ctx, free_hit)
                if fh:
                    chip = 'freehit'
                    used_chips.add(('freehit', lo))

        made, hits = 0, 0
        if chip == 'freehit':
            # one week only: bought at today's prices, reverted after scoring
            saved = (dict(squad), bank)
            squad = dict(fh['squad'])
            bank = fh['budget'] - sum(squad.values())
            free = min(MAX_FREE, free + 1)
        elif chip == 'wildcard plan':
            lo, wc = window
            bank, made = apply_plan(squad, bank, wc, prices, sells)
            chip = 'wildcard'
            used_chips.add(('wildcard', lo))
            chip_pts['wildcard'] = chip_pts.get('wildcard', 0)
            free = 1
        elif chip == 'wildcard':
            new = window[1]
            bank += sum(sell_price(b, prices.get(c, b))
                        for c, b in squad.items()) - sum(new.values())
            made = sum(1 for c in new if c not in squad)
            squad, window = new, window[0]
            used_chips.add(('wildcard', window))
            chip_pts['wildcard'] = chip_pts.get('wildcard', 0)
            free = 1                      # a wildcard week consumes no transfer
        elif gw > 1 and transfers_on and transfers == 'plan':
            bank, made = apply_plan(squad, bank, normal, prices, sells)
            hits = max(0, made - min(free, MAX_FREE))
            free = min(MAX_FREE, max(0, free - made) + 1)
        elif gw > 1 and transfers_on:
            while True:
                allowed = min(free, MAX_FREE)
                extra = made - allowed
                if extra >= MAX_HITS:
                    break
                # a hit has to clear the margin AND pay back its own 4 points
                paid = made >= allowed
                need = margin + HIT_COST if paid else margin
                mv = best_move(squad, bank, proj, prices, pos, clubs,
                               need, ranked, chaser, paid, eligible)
                if mv is None:
                    break
                bank += mv['sell'] - mv['buy']
                del squad[mv['out']]
                squad[mv['in']] = mv['buy']
                made += 1
                if made > allowed:
                    hits += 1
            free = min(MAX_FREE, max(0, free - made) + 1)
        elif gw > 1:
            free = min(MAX_FREE, free + 1)

        assert_legal(squad, bank, clubs, pos)

        picked, picked_bench, cap, vice = pick_xi(list(squad), proj, pos,
                                                  playing, order)

        # decided at the deadline, on the picked bench -- before auto-subs,
        # which depend on minutes nobody knows yet
        if chips and chip is None:
            chip, window = pick_squad_chip(season, gw, used_chips, cap,
                                           list(squad), picked_bench, known)
            if chip:
                used_chips.add((chip, window[0]))

        xi, bench = apply_autosubs(picked, picked_bench, minutes, pos)

        scorer = cap if minutes.get(cap, 0) > 0 else vice
        multiplier = 2 if chip == 'tcaptain' else 1
        gw_pts = sum(points.get(c, 0) for c in xi) \
            + points.get(scorer, 0) * multiplier
        if chip == 'bboost':
            gw_pts += sum(points.get(c, 0) for c in bench)
        gw_pts -= HIT_COST * hits

        if chip in ('tcaptain', 'bboost'):
            extra_pts = (points.get(scorer, 0) if chip == 'tcaptain'
                         else sum(points.get(c, 0) for c in bench))
            chip_pts[chip] = chip_pts.get(chip, 0) + extra_pts

        rows.append({'gw': gw, 'points': gw_pts, 'transfers': made,
                     'hits': hits, 'chip': chip or '',
                     'captain_points': points.get(scorer, 0) * multiplier,
                     'squad_value': sum(prices.get(c, squad[c])
                                        for c in squad) + bank,
                     # kept for form_lab.diagnose, which replays the season
                     'xi': list(xi), 'bench': list(bench), 'scorer': scorer,
                     'squad': list(squad), 'picked': list(picked),
                     'captain': cap, 'vice': vice,
                     'squad_start': start_squad, 'bank_start': start_bank,
                     'fh_trigger': fh['trigger'] if fh else '',
                     'fh_gain': (gw_pts - gw_points(saved[0], proj, pos, playing,
                                                    minutes, points, order))
                     if saved else 0})
        if saved:
            squad, bank = saved
    return pd.DataFrame(rows), chip_pts


def pick_wildcard(season, gw, used, squad, bank, proj, prices, pos, clubs,
                  eligible):
    """('wildcard', (window_start, new_squad)) on this half's wildcard week.

    Fixed gameweeks, not a gain threshold: a rebuild's projected gain is
    inflated early in a half, so a gain trigger fired at once and churned the
    squad for nothing.
    """
    for chip, lo in open_chips(season, gw, used):
        if chip != 'wildcard' or gw != WILDCARD_FALLBACK[1 if lo == 1 else 2]:
            continue
        new = wildcard_squad(squad, bank, proj, prices, pos, clubs, eligible)
        if new is not None:
            return 'wildcard', (lo, new)
    return None, None


def pick_squad_chip(season, gw, used, captain, squad, bench, known):
    """Triple captain or bench boost on the first known double, else the fallback.

    At most one per gameweek; triple captain has priority when both qualify.
    """
    pending = [(c, lo) for c, lo in open_chips(season, gw, used)
               if c in CHIP_PRIORITY]
    pending.sort(key=lambda cl: CHIP_PRIORITY.index(cl[0]))

    for rank, (chip, lo) in enumerate(pending):
        hi = next(h for l, h in CHIP_RULES[season][chip] if l == lo)
        # each pending chip gets its own fallback deadline, latest first
        fallback = hi - rank
        ready = (tc_ready(captain, gw, known) if chip == 'tcaptain'
                 else bb_ready(squad, bench, gw, known))
        if ready or gw == fallback:
            return chip, (lo, hi)
    return None, None


def average_manager(season):
    """FPL's average entry score per gameweek, if data/ carries it."""
    gw = io.read_gw(season)
    for col in gw.columns:
        if 'average' in col.lower() and 'entry' in col.lower():
            return gw.groupby('round')[col].first()
    return None


def chart(curves, season, benchmarks=None, path=None):
    """Cumulative points by gameweek, one line per strategy."""
    path = chart_path(season) if path is None else path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for (name, series), colour in zip(curves.items(), COLOURS):
        ax.plot(series.index, series.to_numpy(), linewidth=2,
                color=colour, label=name, solid_capstyle='round')

    # end labels, nudged apart so close finishers stay readable. The gap is a
    # share of the AXIS height, not of the spread -- three points apart is
    # still two overlapping labels.
    finals = sorted((s.iloc[-1], s.index[-1]) for s in curves.values())
    gap = 0.032 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    last_y = None
    for value, x in finals:
        last_y = value if last_y is None else max(value, last_y + gap)
        ax.annotate(f'{value:.0f}', xy=(x + 0.4, last_y), color=INK,
                    fontsize=9, va='center', annotation_clip=False)
    # Real managers give a season TOTAL, not a per-gameweek curve, so these are
    # horizontal finish levels rather than invented trajectories.
    for i, (name, total) in enumerate(sorted((benchmarks or {}).items(),
                                             key=lambda kv: -kv[1])):
        ax.axhline(total, color=MUTED, linewidth=1.2,
                   linestyle=('--' if i == 0 else ':'), label=name)

    ax.set_title(f'Cumulative points, 20{season[:2]}/{season[3:]} simulation',
                 color=INK, fontsize=13, loc='left', pad=14)
    ax.set_xlabel('Gameweek', color=MUTED, fontsize=10)
    ax.set_ylabel('Points', color=MUTED, fontsize=10)
    ax.grid(axis='y', color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlim(1, max(s.index[-1] for s in curves.values()) + 1.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#c3c2b7')
    ax.tick_params(colors=MUTED, labelsize=9)
    # opaque backing: the benchmark lines span the axes and cross the legend
    leg = ax.legend(fontsize=9, labelcolor=INK, loc='upper left',
                    frameon=True, facecolor=SURFACE, edgecolor='none',
                    framealpha=1)
    leg.set_zorder(5)
    fig.subplots_adjust(right=0.88)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def run_season(season, chips=False):
    """Every strategy over one season. Returns (results, curves)."""
    hist, clubs, pos, pri = season_data(season)
    results, curves = {}, {}
    for s in STRATEGIES:
        d, chip_pts = simulate(s, season, hist, clubs, pos, pri, chips=chips)
        results[s] = {
            'total': int(d['points'].sum()),
            'transfers': int(d['transfers'].sum()),
            'hits': int(d['hits'].sum()),
            'chip_pts': int(sum(chip_pts.values())),
            'chips': ' '.join(sorted(c for c in d['chip'] if c)) or '-',
        }
        curves[s] = d.set_index('gw')['points'].cumsum()
        print(f'  {season} {"chips" if chips else "plain":5s} {s:12s} '
              f'{results[s]["total"]:5d} pts, {results[s]["transfers"]:3d} tr, '
              f'{results[s]["hits"]:2d} hits, chips: {results[s]["chips"]}')
    return results, curves


def main(seasons=SEASONS, managers=True):
    rows = []
    for season in seasons:
        for chips in (True, False):
            results, curves = run_season(season, chips=chips)
            ranks = {}
            if managers:
                bench, ranks = manager_benchmark(season, results)
                if chips:
                    chart(curves, season, bench)
                    print(f'  chart: {chart_path(season)}')
            for s, r in results.items():
                rows.append({'season': season,
                             'mode': 'chips+hits' if chips else 'plain',
                             'strategy': s, **r,
                             'top_pct': ranks.get(s, np.nan)})

    table = pd.DataFrame(rows)
    wide = table.pivot_table(index='strategy', columns=['season', 'mode'],
                             values=['total', 'top_pct', 'transfers', 'hits',
                                     'chip_pts'], aggfunc='first')
    print('\nper season\n')
    print(table.to_string(index=False))

    totals = (table[table['mode'] == 'chips+hits']
              .groupby('strategy')[['total', 'transfers', 'hits', 'chip_pts']]
              .sum().sort_values('total', ascending=False))
    plain = (table[table['mode'] == 'plain'].groupby('strategy')['total'].sum()
             .rename('total_plain'))
    totals = totals.join(plain)
    print(f'\n{len(seasons)}-season totals (chips+hits)\n')
    print(totals.to_string())

    out = data.CACHE.parent / 'results'
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'per_season.csv', index=False)
    totals.to_csv(out / 'totals.csv')
    wide.to_csv(out / 'per_season_wide.csv')
    print(f'\ncsv: {out}')

    print(f'\nwinner: {decide(totals)}')
    notes = [FREE_HIT_NOTE,
             'no injury data (0 minutes only), simplified auto-subs',
             '22_23 omitted (no prior: 21_22 is a bad feature season)']
    (out / 'NOTES.txt').write_text('\n'.join(notes) + '\n', encoding='utf-8')
    print('\n' + '\n'.join(notes))
    return table, totals, wide


def decide(totals, tolerance=50):
    """Highest multi-season total, or the thriftier of a near-tie."""
    top = totals.sort_values('total', ascending=False)
    if len(top) > 1 and top['total'].iloc[0] - top['total'].iloc[1] < tolerance:
        pair = top.head(2).sort_values('transfers')
        return (f'{pair.index[0]} (within {tolerance} of '
                f'{top.index[0] if pair.index[0] != top.index[0] else top.index[1]}'
                f', fewer transfers)')
    return f'{top.index[0]} ({int(top["total"].iloc[0])} pts)'


def manager_benchmark(season, results):
    """(lines for the chart, top-% per strategy) from the sampled managers."""
    from form_lab import managers as mg
    try:
        sample = mg.load(season)
    except FileNotFoundError:
        print(f'  no manager sample for {season} '
              f'-- run: python -m form_lab.managers')
        return {}, {}

    stats = mg.summary(sample)
    print(f'  {season} managers: n={stats["n"]}, median={stats["median"]:.0f}, '
          f'mean={stats["mean"]:.0f}, p25={stats["p25"]:.0f}, '
          f'p75={stats["p75"]:.0f}')
    lines = {'median manager': stats['median'], 'mean manager': stats['mean']}
    tops = {s: mg.top_pct(sample, r['total']) for s, r in results.items()}
    return lines, tops


if __name__ == '__main__':
    main()
