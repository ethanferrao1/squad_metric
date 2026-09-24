"""Squad rater: score a 15-man squad against what the MILP could achieve at
the same budget, rate each player within his position, and suggest transfers.

Lifted out of notebooks/squad_rater.ipynb so the same code can rate a
historical pool (out-of-fold predictions, known outcomes) or the live 26/27
pool (a full-data fit, no outcomes yet).
"""
import numpy as np
import pandas as pd

import data_io as io
import optimise as op
import predict_2026_27 as p2627

BUDGET = op.BUDGET          # 1000 tenths = £100.0m
POS = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}

# how far either side of a player's price to look for comparable modelled
# players when imputing a prediction, in tenths of a million
PRICE_BAND = 5


# ---------------------------------------------------------------- pools

def decorate(p, names):
    """Attach name, position label, within-position rating and value."""
    p = p.copy()
    p['web_name'] = p['code'].map(names)
    p['pos'] = p['element_type'].map(POS)
    # 0-100: where he sits among players in his own position
    p['rating'] = p.groupby('element_type')['pred'].rank(pct=True) * 100
    p['value'] = p['pred'] / (p['price'] / 10)      # projected points per £m
    return p.reset_index(drop=True)


def pool_for(season, base_cands, names):
    """Candidates for one historical season, with names and ratings attached.

    `base_cands` is run_milp.candidates(...) output, `names` the
    (season, code) -> web_name lookup.
    """
    return decorate(base_cands[season], names.loc[season])


def _impute_pred(price, element_type, modelled, band=PRICE_BAND):
    """Median prediction of modelled players at the same position and a
    similar price. Widens to the whole position, then the whole pool, if
    that band is empty."""
    same_pos = modelled[modelled['element_type'] == element_type]
    near = same_pos[(same_pos['price'] - price).abs() <= band]
    for pool in (near, same_pos, modelled):
        if len(pool):
            return pool['pred'].median()
    return np.nan


def api_roster():
    """The live API's player list, shaped like players_raw. Returns None if
    the API is neither cached nor reachable.

    players_raw_2026_27 is a snapshot and runs ~68 players behind the live
    game -- mid-season registrations and late signings. Those are exactly
    the players a real squad is most likely to contain that the snapshot
    lacks, so top the roster up from the API where we can. fpl_api caches,
    so this costs nothing offline once fetched.
    """
    try:
        import fpl_api
        boot = fpl_api.bootstrap()
    except Exception:
        return None
    return pd.DataFrame(boot['elements'])[
        ['code', 'element_type', 'team_code', 'now_cost', 'web_name']]


def roster(season=None):
    """players_raw for `season`, topped up with any API-only players."""
    season = season or p2627.T1
    players = io.read_players(season)
    extra = api_roster()
    if extra is not None:
        extra = extra[~extra['code'].isin(set(players['code']))]
        if len(extra):
            players = pd.concat([players, extra], ignore_index=True)
    return players


def unmodelled(modelled, season=None, players=None):
    """Everyone on the roster the model has no view on: new signings,
    promoted players, and anyone under the 450-minute floor.

    They get the median prediction of modelled players at the same position
    and a similar price -- a placeholder so a real squad containing them can
    still be rated, not a forecast. `modelled=False` marks them so the
    report can say so. `players` overrides the roster (the live API's, say).
    """
    players = roster(season) if players is None else players

    have = set(modelled['code'])
    p = players[~players['code'].isin(have)].copy()
    p['element_type'] = pd.to_numeric(p['element_type'], errors='coerce')
    p = p[p['element_type'].isin(POS)]
    p['price'] = pd.to_numeric(p['now_cost'], errors='coerce')
    p = p.dropna(subset=['price', 'team_code'])

    out = pd.DataFrame({
        'code': p['code'].to_numpy(),
        'element_type': p['element_type'].astype(int).to_numpy(),
        'team_code': p['team_code'].to_numpy(),
        'price': p['price'].to_numpy(),
        'last_pts': np.nan,
        'pred': [_impute_pred(pr, et, modelled)
                 for pr, et in zip(p['price'], p['element_type'])],
    })
    out['modelled'] = False
    return out.dropna(subset=['pred']).reset_index(drop=True)


def live_parts():
    """Fit on all history, build the live feature matrix, score it.

    Returns (model, X_live, live, cands) so a caller that needs the fitted
    model or the feature matrix -- SHAP, say -- doesn't have to fit a second
    time. `live` is row-aligned with `X_live` and carries `code`.
    """
    model, cols, trans = p2627.fit_full_model()
    X_live, live = p2627.live_features(cols)
    c = p2627.build_candidates(live, model.predict(X_live))
    c = c.rename(columns={'current_price': 'price'})
    c = c[['code', 'element_type', 'team_code', 'price', 'last_pts', 'pred']]
    c['modelled'] = True
    return model, X_live, live, c


def live_pool(include_unmodelled=True, parts=None):
    """The 26/27 pool: model fitted on all history, predicting into the live
    season, priced at what it costs to buy today."""
    model, X_live, live, c = parts if parts is not None else live_parts()

    if include_unmodelled:
        c = pd.concat([c, unmodelled(c)], ignore_index=True)

    names = roster().drop_duplicates('code').set_index('code')['web_name']
    return decorate(c, names)


MIN_CHANCE = 50


def available(pool):
    """Players fit enough to be worth buying.

    FPL's `chance_of_playing_next_round` is null when nothing has been
    flagged, which means fit -- only an explicit figure below MIN_CHANCE
    rules a player out. Filters who you can BUY; it never removes anyone
    from a squad you already own.
    """
    if 'chance' not in pool.columns:
        return pool
    c = pool['chance']
    return pool[c.isna() | (c >= MIN_CHANCE)]


def _modelled(pool):
    """The part of the pool the MILP is allowed to reason about. Imputed
    predictions are placeholders, so they never set the benchmark or get
    suggested as transfers in."""
    return pool[pool['modelled']] if 'modelled' in pool.columns else pool


# ---------------------------------------------------------------- rating

def squad_points(sq, col='pred'):
    """Best valid XI from the 15, captain doubled."""
    xi, captain = op.best_xi(sq, col)
    return xi[col].sum() + xi.loc[xi['code'] == captain, col].sum(), xi, captain


def optimum(pool, budget=BUDGET, col='pred'):
    """What the MILP achieves at the same budget. The denominator.

    Same filter as the Build tab: only modelled, buyable players. A
    benchmark built from players FPL says are unlikely to play is one no
    manager could actually assemble, so it depressed every rating against
    a squad that was never available.
    """
    total, _, _ = squad_points(
        op.pick_squad(available(_modelled(pool)), col, budget=budget), col)
    return total


def rate_squad(codes, pool, budget=BUDGET, bank=None):
    """Rate a 15-man squad against the MILP optimum at the same budget.

    `bank` is for a live team: pass what the API says is in the bank and the
    budget becomes the squad's value at TODAY's prices plus that bank, i.e.
    the money actually available. Without it a squad that has appreciated
    prices out above 100.0m and reports a negative bank, and suggested
    transfers get costed against money that isn't there. Historical squads
    leave `bank` as None and keep the flat 100.0m budget, unchanged.
    """
    sq = pool[pool['code'].isin(codes)].copy()
    if len(sq) != 15:
        missing = set(codes) - set(sq['code'])
        raise ValueError(f'{len(sq)} of 15 found in pool; missing {missing}')

    shape = sq['element_type'].value_counts().reindex(op.SQUAD).to_dict()
    if shape != op.SQUAD:
        raise ValueError(f'illegal shape {shape}, need {op.SQUAD}')

    spend = int(sq['price'].sum())
    if bank is not None:
        budget = spend + int(bank)

    total, xi, captain = squad_points(sq)
    best = optimum(pool, budget)

    sq['in_xi'] = sq['code'].isin(xi['code'])
    sq['captain'] = sq['code'] == captain
    return {
        'score': total,
        'optimum': best,
        'pct': 100 * total / best,
        'budget': budget,
        'spend': spend,
        'bank': budget - spend,
        'captain': captain,
        'players': sq.sort_values(['element_type', 'pred'],
                                  ascending=[True, False]),
    }


# ---------------------------------------------------------------- suggestions

def suggest(rating, pool, budget=BUDGET, k=3, per_slot=3):
    sq = rating['players']
    base = rating['score']
    bank = budget - sq['price'].sum()
    held = set(sq['code'])

    # never suggest buying someone FPL says is unlikely to play
    pool = available(_modelled(pool))

    rows = []
    for i, p in sq.iterrows():
        rest = sq.drop(index=i)
        clubs_held = rest['team_code'].value_counts()

        cand = pool[(pool['element_type'] == p['element_type']) &
                    (~pool['code'].isin(held)) &
                    (pool['price'] <= bank + p['price'])]
        cand = cand[[clubs_held.get(t, 0) < op.MAX_PER_CLUB
                     for t in cand['team_code']]]
        if cand.empty:
            continue

        for j in cand.nlargest(per_slot, 'pred').index:
            total, _, _ = squad_points(pd.concat([rest, pool.loc[[j]]]))
            rows.append({
                'out': p['web_name'], 'out_pred': p['pred'],
                'in': pool.at[j, 'web_name'], 'in_pred': pool.at[j, 'pred'],
                'pos': p['pos'],
                'cost': int(pool.at[j, 'price'] - p['price']),
                'gain': total - base,
            })

    d = pd.DataFrame(rows)
    if d.empty:
        return d
    # one row per incoming player, so the list is three distinct moves
    return (d[d['gain'] > 0].sort_values('gain', ascending=False)
             .drop_duplicates('in').head(k).reset_index(drop=True))


# ---------------------------------------------------------------- report

def report(codes, pool, budget=BUDGET, bank=None):
    r = rate_squad(codes, pool, budget, bank)
    budget = r['budget']            # the live budget, if `bank` was given

    print(f"squad {r['score']:.0f}   optimum {r['optimum']:.0f}   "
          f"{r['pct']:.1f}% of achievable")
    print(f"spend {r['spend'] / 10:.1f}m   bank {r['bank'] / 10:.1f}m")

    p = r['players']
    cols = ['web_name', 'pos', 'price', 'pred', 'rating', 'value',
            'in_xi', 'captain']
    show = p[cols].copy()
    show['price'] = show['price'] / 10

    has_unmodelled = 'modelled' in p.columns and not p['modelled'].all()
    if has_unmodelled:
        # an imputed prediction is a placeholder, so don't dress it up as a
        # rating the model stands behind
        show['note'] = np.where(p['modelled'], '', 'no PL history')
    print()
    print(show.round(1).to_string(index=False))

    known = p[p['modelled']] if 'modelled' in p.columns else p
    weak = known[known['rating'] < 50]
    if len(weak):
        print(f"\nbelow the 50th percentile for their position: "
              f"{', '.join(weak['web_name'].astype(str))}")
    if has_unmodelled:
        n = int((~p['modelled']).sum())
        print(f"\n{n} player(s) with no PL history: predictions are the median "
              f"for their position and price, not a forecast. They are "
              f"excluded from the optimum and from transfer suggestions.")

    s = suggest(r, pool, budget)
    print('\nsuggested transfers:')
    print(s.round(1).to_string(index=False) if len(s)
          else '  none improve the squad')
    return r, s
