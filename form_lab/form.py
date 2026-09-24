"""Form-adjusted projection: the pre-season model blended with recent scoring.

Everything here is a pure function over frames, so the eval and the demo
share exactly the rules the tests check.
"""
import math

import numpy as np
import pandas as pd

SEASON_GWS = 38
MAX_FORM_GAMES = 10

# how far the blend can ever move off the prior, reached at MAX_FORM_GAMES.
# 0.8 won the backtest; 0.7-0.9 are within 0.005 MAE of each other.
W_MAX = 0.8

# breakout exception: a cheap player scoring far above his prior skips the ramp.
# Off by default -- it cost MAE and gained nothing on Spearman in the backtest.
BREAKOUT = False
CHEAP_PRICE = 55
BREAKOUT_PCT = 90
BREAKOUT_RATIO = 1.5

YOUTH_AGE = 24.0
YOUTH_BOOST = 0.10

# market signals. Off: worth ~0.012 MAE over a 5-gameweek horizon, but cost
# 77 points over a full simulated season by firing extra transfers.
MARKET_WEIGHT = 0.0
PRICE_MOVE = 3
TRANSFER_SHARE = 0.05
MARKET_WINDOW = 3
MARKET_MOVER_PCT = 98

# the blend's backtest MAE, in points per gameweek
BLEND_MAE = 1.132

# swaps are judged over the next five gameweeks, the horizon the backtest
# actually measured
SWAP_HORIZON = 5
STRONG_MULTIPLE = 2.0


def snapshot(hist, upto_gw, priors, ages=None, debut=None,
             max_games=MAX_FORM_GAMES, window=MARKET_WINDOW):
    """What was knowable about each player before `upto_gw`.

    `hist` is a gw_history frame, `priors` a code -> season points mapping.
    Only gameweeks strictly before `upto_gw` are used, so a backtest at that
    gameweek cannot see its own answer.
    """
    past = hist[hist['gw'] < upto_gw]
    if past.empty:
        return pd.DataFrame()

    g = past.groupby('code')
    last = g.last()
    recent = g.tail(max_games).groupby('code')['points'].mean()
    # price drift across the market window, not since August
    opened = past.groupby('code').tail(window + 1).groupby('code')['price'].first()

    snap = pd.DataFrame({
        'code': last.index,
        'element_type': last['element_type'].to_numpy(),
        'n_games': g.size().reindex(last.index).to_numpy(),
        'form_ppg': recent.reindex(last.index).to_numpy(),
        'price': last['price'].to_numpy(),
        'price_change': (last['price'] - opened.reindex(last.index)).to_numpy(),
        'net_transfers': (last['transfers_in']
                          - last['transfers_out']).to_numpy(),
        'net_transfer_share': ((last['transfers_in'] - last['transfers_out'])
                               / last['selected'].replace(0, np.nan)).to_numpy(),
    })
    snap['prior_ppg'] = prior_ppg(snap['code'].map(priors).to_numpy())
    snap['age'] = snap['code'].map(ages) if ages is not None else np.nan
    snap['no_pl_history'] = (snap['code'].map(debut).fillna(False)
                             if debut is not None else False)
    return snap.dropna(subset=['prior_ppg']).reset_index(drop=True)


def prior_ppg(prior_pts):
    """Season-model points spread evenly over the season."""
    return np.asarray(prior_pts, dtype=float) / SEASON_GWS


def form_ppg(points, max_games=MAX_FORM_GAMES):
    """Mean of his last `max_games` scores, zeros included."""
    recent = list(points)[-max_games:]
    return float(np.mean(recent)) if recent else np.nan


def weight(n_games, w_max=W_MAX, max_games=MAX_FORM_GAMES):
    """Certainty ramp: 0 with no games, w_max at max_games, flat after."""
    n = np.clip(np.asarray(n_games, dtype=float), 0, max_games)
    return w_max * n / max_games


def youth_adjust(prior, age, no_pl_history, boost=YOUTH_BOOST,
                 max_age=YOUTH_AGE):
    """Lift the prior for an under-24 with no Premier League season behind him."""
    prior = np.asarray(prior, dtype=float)
    age = pd.to_numeric(pd.Series(age), errors='coerce').to_numpy()
    young = (age < max_age) & np.asarray(no_pl_history, dtype=bool)
    return np.where(np.isnan(age), prior, np.where(young, prior * (1 + boost),
                                                   prior))


def breakout_mask(snap, cheap=CHEAP_PRICE, pct=BREAKOUT_PCT,
                  ratio=BREAKOUT_RATIO):
    """Cheap, top-decile-for-his-position form, and well clear of his prior."""
    form, prior = snap['form_ppg'], snap['prior_ppg']
    # rank within position on this gameweek's snapshot, not across the season
    pctile = form.groupby(snap['element_type']).rank(pct=True) * 100
    return ((snap['price'] <= cheap) & (pctile >= pct) &
            (form >= ratio * prior) & form.notna()).to_numpy()


def market_signal(snap):
    """Price drift and net transfers, each normalised to roughly [-1, 1]."""
    price_move = snap['price_change'].fillna(0) / max(PRICE_MOVE, 1)
    flow = snap['net_transfer_share'].fillna(0) / max(TRANSFER_SHARE, 1e-9)
    return np.clip(price_move + flow, -2, 2).to_numpy() / 2


def is_market_mover(snap, pct=MARKET_MOVER_PCT):
    """Top 2% of this pool by |net transfers|, or a 0.3m price move.

    Ranked within the snapshot rather than against a fixed share of
    ownership, which fires for most of the pool while ownership is still low.
    """
    flow = snap['net_transfers'].abs() if 'net_transfers' in snap.columns \
        else snap['net_transfer_share'].abs()
    busy = (flow.rank(pct=True) * 100) >= pct
    return (busy | (snap['price_change'].abs() >= PRICE_MOVE)).to_numpy()


def project(snap, w_max=W_MAX, breakout=BREAKOUT, market_weight=MARKET_WEIGHT,
            youth=True):
    """Blended points per gameweek for a snapshot of players.

    `snap` needs code, element_type, prior_ppg, form_ppg, n_games, price, age,
    no_pl_history, price_change, net_transfer_share. Returns a copy with
    blended_ppg and the flags behind it.
    """
    out = snap.copy()
    prior = out['prior_ppg'].to_numpy(dtype=float)
    if youth:
        prior = youth_adjust(prior, out['age'], out['no_pl_history'])
    out['adj_prior_ppg'] = prior

    w = weight(out['n_games'].to_numpy(), w_max)
    out['breakout'] = breakout_mask(out) if breakout else False
    # a breakout player gets the full weight now rather than waiting out the ramp
    w = np.where(out['breakout'], w_max, w)

    form = out['form_ppg'].to_numpy(dtype=float)
    w = np.where(np.isnan(form), 0.0, w)
    out['form_weight'] = w

    blended = (1 - w) * prior + w * np.nan_to_num(form)
    if market_weight:
        blended = blended + market_weight * market_signal(out)
    out['blended_ppg'] = blended
    out['market_mover'] = is_market_mover(out)
    return out


def switch_margin(mae_ppg=BLEND_MAE, horizon=SWAP_HORIZON):
    """Points of noise to clear before a swap is worth it.

    The blend's per-gameweek error over the horizon, times root two because
    the gain is a difference between two independently mispredicted players.
    """
    return float(mae_ppg) * float(horizon) * math.sqrt(2)


SWITCH_MARGIN = switch_margin()


def label(gain, margin=SWITCH_MARGIN, strong=STRONG_MULTIPLE):
    """'strong', 'hold' or None when the gain does not clear the margin."""
    if gain < margin:
        return None
    return 'strong' if gain >= strong * margin else 'hold'


def cover_verdict(gain, margin=SWITCH_MARGIN):
    """For an unavailable starter: a transfer only if it beats simply starting
    the bench player by the margin, else make the substitution."""
    return 'injury cover' if gain >= margin else 'substitute instead'


def gain_points(ppg_in, ppg_out, gws_remaining):
    """Projected points a swap adds over the rest of the season."""
    return (ppg_in - ppg_out) * gws_remaining


def _weeks(codes, avail, horizon):
    """[(code -> multiplier, weight)]: each gameweek's availability, with
    identical gameweeks merged so each distinct one is scored once."""
    weeks = []
    for t in range(horizon):
        m = {c: (avail or {}).get(c, [1.0] * horizon)[t] for c in codes}
        for w in weeks:
            if w[0] == m:
                w[1] += 1
                break
        else:
            weeks.append([m, 1])
    return weeks


def suggest_swaps(codes, projected, pool, gws_remaining, bank=0,
                  margin=SWITCH_MARGIN, k=3, per_slot=3, include_below=False,
                  eligible=None, avail=None, only_out=None):
    """Swaps whose projected gain over the next `gws_remaining` clears `margin`.

    Scored the same way rater.suggest scores a transfer: re-pick the best XI
    with the swap made, on blended points per gameweek. With `avail` (code ->
    per-gameweek multipliers, next first) each gameweek's XI is picked on
    that gameweek's availability, so an injured starter is worth nothing to
    keep. With `eligible`, a player who would land on the bench must be a
    regular starter. `only_out` limits who may be sold.
    """
    import optimise as op
    import rater as rt

    proj = projected.set_index('code')
    sq = pool[pool['code'].isin(codes)].copy()
    sq['blended_ppg'] = sq['code'].map(proj['blended_ppg'])
    sq = sq.dropna(subset=['blended_ppg'])

    cand_pool = pool[pool['code'].isin(proj.index) &
                     ~pool['code'].isin(set(codes))].copy()
    cand_pool['blended_ppg'] = cand_pool['code'].map(proj['blended_ppg'])
    cand_pool = rt.available(rt._modelled(cand_pool)).dropna(
        subset=['blended_ppg'])

    weeks = _weeks(set(sq['code']) | set(cand_pool['code']), avail,
                   gws_remaining)
    mean_m = {c: sum(m[c] * w for m, w in weeks) / gws_remaining
              for c in weeks[0][0]}
    cand_pool['window_ppg'] = (cand_pool['blended_ppg']
                               * cand_pool['code'].map(mean_m))

    def value(frame):
        """(points over the window, the last gameweek's XI)."""
        total, xi = 0.0, None
        for m, w in weeks:
            f = frame.assign(p=frame['blended_ppg'] * frame['code'].map(m))
            pts, xi, _ = rt.squad_points(f, 'p')
            total += pts * w
        return total, xi

    base, _ = value(sq)
    rows = []
    for i, p in sq.iterrows():
        if only_out is not None and p['code'] not in only_out:
            continue
        rest = sq.drop(index=i)
        clubs_held = rest['team_code'].value_counts()
        cand = cand_pool[
            (cand_pool['element_type'] == p['element_type']) &
            (cand_pool['price'] <= bank + p['price'])]
        cand = cand[[clubs_held.get(t, 0) < op.MAX_PER_CLUB
                     for t in cand['team_code']]]
        if cand.empty:
            continue

        for j in cand.nlargest(per_slot, 'window_ppg').index:
            total, xi = value(pd.concat([rest, cand_pool.loc[[j]]]))
            benched = cand_pool.at[j, 'code'] not in set(xi['code'])
            if eligible is not None and benched \
                    and cand_pool.at[j, 'code'] not in eligible:
                continue
            gain = total - base
            tag = label(gain, margin)
            if tag is None and not include_below:
                continue
            rows.append({
                'out': p['web_name'], 'in': cand_pool.at[j, 'web_name'],
                'pos': p.get('pos'), 'gain_pts': gain, 'gain': gain,
                'out_pred': float(p['blended_ppg'] * mean_m[p['code']]),
                'in_pred': float(cand_pool.at[j, 'window_ppg']),
                'verdict': tag or 'below margin',
                'cost': int(cand_pool.at[j, 'price'] - p['price']),
            })

    d = pd.DataFrame(rows)
    if d.empty:
        return d
    return (d.sort_values('gain_pts', ascending=False)
             .drop_duplicates('in').head(k).reset_index(drop=True))


def captain(xi, projected, unavailable=()):
    """Highest blended points per gameweek among those with a fixture who can
    play; never anyone in `unavailable`."""
    ok = set(projected.loc[projected['has_fixture'], 'code']) - set(unavailable)
    playing = xi[xi['code'].isin(ok)]
    if playing.empty:
        return None
    ppg = projected.set_index('code')['blended_ppg']
    return playing.assign(p=playing['code'].map(ppg)).nlargest(1, 'p')['code'].iloc[0]
