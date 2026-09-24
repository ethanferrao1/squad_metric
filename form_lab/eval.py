"""Backtest the blend against prior-only and form-only.

    python -m form_lab.eval

At every gameweek from GW2 it predicts points per gameweek over the next
five, scores the three projections, and sweeps the blend's settings.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from form_lab import data, form

SEASONS = ['24_25', '25_26']
FIRST_GW = 2
HORIZON = 5
W_GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MARKET_GRID = [0.0, 0.2, 0.3, 0.4, 0.6]
MIN_PLAYERS = 30


def target_ppg(hist, gw, horizon=HORIZON):
    """Actual points per gameweek over the next `horizon` gameweeks."""
    win = hist[(hist['gw'] >= gw) & (hist['gw'] < gw + horizon)]
    return win.groupby('code')['points'].mean()


def snapshots(season, horizon=HORIZON):
    """(gw, snap, actual) for every scorable gameweek of a season."""
    hist = data.gw_history(season)
    pri = data.priors()
    pri = pri[pri['season'] == season].set_index('code')['prior_pts']
    age = data.ages(season)
    debut = data.debut_season(season)

    last_gw = int(hist['gw'].max())
    out = []
    for gw in range(FIRST_GW, last_gw - horizon + 2):
        snap = form.snapshot(hist, gw, pri, age, debut)
        actual = target_ppg(hist, gw, horizon)
        snap = snap[snap['code'].isin(actual.index)]
        if len(snap) < MIN_PLAYERS:
            continue
        out.append((gw, snap.reset_index(drop=True),
                    snap['code'].map(actual).to_numpy()))
    return out


def score(pred, actual):
    """(MAE, Spearman) for one gameweek's cross-section."""
    ok = ~(np.isnan(pred) | np.isnan(actual))
    if ok.sum() < MIN_PLAYERS:
        return np.nan, np.nan
    rho = spearmanr(pred[ok], actual[ok]).statistic
    return float(np.mean(np.abs(pred[ok] - actual[ok]))), float(rho)


def _summarise(rows, **tags):
    mae = [r[0] for r in rows if not np.isnan(r[0])]
    rho = [r[1] for r in rows if not np.isnan(r[1])]
    return {**tags, 'mae': np.mean(mae), 'spearman': np.mean(rho),
            'gws': len(mae)}


def run(seasons=SEASONS, w_grid=W_GRID, market_grid=MARKET_GRID):
    """Sweep the blend's settings and score every arm on both seasons."""
    frames = {s: snapshots(s) for s in seasons}
    results = []

    baselines = {'prior only': [], 'form only (last 10)': []}
    for s in seasons:
        for _, snap, actual in frames[s]:
            baselines['prior only'].append(
                score(snap['prior_ppg'].to_numpy(), actual))
            baselines['form only (last 10)'].append(
                score(snap['form_ppg'].to_numpy(), actual))
    for name, rows in baselines.items():
        results.append(_summarise(rows, model=name, w_max=np.nan,
                                  breakout='-', market=np.nan))

    for w in w_grid:
        for brk in (False, True):
            for mw in market_grid:
                rows = []
                for s in seasons:
                    for _, snap, actual in frames[s]:
                        p = form.project(snap, w_max=w, breakout=brk,
                                         market_weight=mw)
                        rows.append(score(p['blended_ppg'].to_numpy(), actual))
                results.append(_summarise(
                    rows, model='blend', w_max=w,
                    breakout='on' if brk else 'off', market=mw))

    return pd.DataFrame(results)


BUCKETS = [('GW2-4', 2, 4), ('GW5-10', 5, 10), ('GW11+', 11, 99)]


def bucket_of(gw):
    return next(n for n, lo, hi in BUCKETS if lo <= gw <= hi)


def by_phase(seasons=SEASONS, w_max=None, market=None):
    """MAE and Spearman for each projection, split by how deep the season is."""
    import form_lab.form as f
    w_max = f.W_MAX if w_max is None else w_max
    market = f.MARKET_WEIGHT if market is None else market

    rows = []
    for s in seasons:
        for gw, snap, actual in snapshots(s):
            blend = form.project(snap, w_max=w_max, market_weight=market)
            for name, pred in (
                    ('prior only', snap['prior_ppg'].to_numpy()),
                    ('form only (last 10)', snap['form_ppg'].to_numpy()),
                    ('blend', blend['blended_ppg'].to_numpy())):
                mae, rho = score(pred, actual)
                rows.append({'phase': bucket_of(gw), 'model': name,
                             'mae': mae, 'spearman': rho})

    d = pd.DataFrame(rows).dropna()
    out = d.groupby(['phase', 'model']).agg(
        mae=('mae', 'mean'), spearman=('spearman', 'mean'),
        gws=('mae', 'size')).reset_index()
    order = {n: i for i, (n, _, _) in enumerate(BUCKETS)}
    return out.sort_values(['phase', 'mae'],
                           key=lambda c: c.map(order) if c.name == 'phase'
                           else c).reset_index(drop=True)


def remaining_gws(seasons=SEASONS):
    """Mean gameweeks left across the backtested gameweeks, for the margin."""
    spans = []
    for s in seasons:
        hist = data.gw_history(s)
        last = int(hist['gw'].max())
        spans += [last - gw + 1
                  for gw in range(FIRST_GW, last - HORIZON + 2)]
    return float(np.mean(spans))


def main(seasons=SEASONS):
    table = run(seasons)
    blends = table[table['model'] == 'blend']
    best = blends.loc[blends['mae'].idxmin()]
    best_rho = blends.loc[blends['spearman'].idxmax()]

    # the market arms are only worth keeping if they beat the same blend at 0
    flat = blends[blends['market'] == 0]['mae'].min()
    market_helps = blends[blends['market'] > 0]['mae'].min() < flat

    show = pd.concat([
        table[table['model'] != 'blend'],
        blends[blends['market'] == 0].sort_values('mae'),
        blends[blends['market'] > 0].sort_values('mae').head(3),
    ])
    print(f'backtest {", ".join(seasons)} | GW{FIRST_GW}+, '
          f'{HORIZON}-gameweek horizon\n')
    print(show[['model', 'w_max', 'breakout', 'market', 'mae', 'spearman',
                'gws']].round(4).to_string(index=False))

    margin = form.switch_margin(best['mae'])
    print(f'\nbest MAE      : {best["model"]} w_max={best["w_max"]} '
          f'breakout={best["breakout"]} market={best["market"]} '
          f'-> {best["mae"]:.4f}')
    print(f'best Spearman : w_max={best_rho["w_max"]} '
          f'breakout={best_rho["breakout"]} market={best_rho["market"]} '
          f'-> {best_rho["spearman"]:.4f}')
    print(f'market signals help: {"yes" if market_helps else "no"}')
    print(f'SWITCH_MARGIN : {best["mae"]:.3f} ppg x {form.SWAP_HORIZON} '
          f'gameweeks x sqrt(2) = {margin:.1f} pts')

    print('\nby phase of season:')
    print(by_phase(seasons).round(4).to_string(index=False))
    return table


K_GRID = [0, 0.05, 0.1, 0.15, 0.2]


def fixture_grid(seasons=SEASONS, k_grid=K_GRID):
    """The blend with a fixture adjustment, per source and K, on the 5-GW eval."""
    from form_lab import fixtures as fx

    frames = {s: snapshots(s) for s in seasons}
    diff = {(s, src): fx.difficulty(s, src)
            for s in seasons for src in fx.SOURCES}

    rows = []
    for src in fx.SOURCES:
        for k in k_grid:
            scores = []
            for s in seasons:
                for gw, snap, actual in frames[s]:
                    blend = form.project(snap).set_index('code')['blended_ppg']
                    adj = fx.adjust(blend, diff[(s, src)].get(gw, {}), k)
                    scores.append(score(adj.to_numpy(), actual))
            rows.append(_summarise(scores, source=src, k=k))
    return pd.DataFrame(rows)


TOP_PER_POS = 25


def demo_codes(pool, codes, top_per_pos=TOP_PER_POS):
    """The squad, plus the best few per position as swap candidates.

    Each new player costs an API call, so the demo shortlists on the season
    model rather than pulling histories for the whole pool.
    """
    best = (rt_available(pool).sort_values('pred', ascending=False)
            .groupby('element_type').head(top_per_pos))
    return sorted(set(codes) | set(best['code']))


def rt_available(pool):
    import rater as rt
    return rt.available(rt._modelled(pool))


def demo(entry_id, margin=None, top_per_pos=TOP_PER_POS, news=True, model=None):
    """Blended projections, captain and suggested swaps for one cached team."""
    import build_live as bl
    import fpl_api as api
    import rater as rt

    pool = bl.load()
    codes, bank = api.load_team(entry_id)
    gw = api.current_gw()
    next_gw = gw + 1

    wanted = demo_codes(pool, codes, top_per_pos)
    print(f'fetching gameweek history for {len(wanted)} players '
          f'(cached after the first run)...')
    hist = data.live_history(wanted)
    by_code = pool.set_index('code')
    snap = form.snapshot(hist, next_gw, by_code['pred'])
    snap['price'] = snap['code'].map(by_code['price'])
    snap['has_fixture'] = snap['code'].map(
        data.has_fixture(snap['code'], next_gw))
    proj = form.project(snap)
    proj['web_name'] = proj['code'].map(by_code['web_name'])

    movers = int(proj['market_mover'].sum())
    print(f'entry {entry_id}, GW{gw}, next GW{next_gw}, '
          f'bank {bank / 10:.1f}m')
    print(f'market movers: {movers} of {len(proj)} projected players '
          f'({100 * movers / len(proj):.1f}%)\n')

    mine = proj[proj['code'].isin(codes)].sort_values('blended_ppg',
                                                      ascending=False)
    mine = mine.assign(price_m=mine['price'] / 10)
    cols = ['web_name', 'element_type', 'price_m', 'n_games', 'prior_ppg',
            'form_ppg', 'form_weight', 'blended_ppg', 'has_fixture',
            'breakout', 'market_mover']
    print(mine[cols].round(2).to_string(index=False))

    sq = pool[pool['code'].isin(codes)].copy()
    sq['blended_ppg'] = sq['code'].map(proj.set_index('code')['blended_ppg'])
    _, xi, _ = rt.squad_points(sq.dropna(subset=['blended_ppg']),
                               'blended_ppg')
    cap = form.captain(xi, proj)
    print(f'\ncaptain (GW{next_gw}): '
          f'{by_code["web_name"].get(cap, cap) if cap else "no XI fixture"}')

    margin = form.SWITCH_MARGIN if margin is None else margin
    swaps = form.suggest_swaps(codes, proj, pool, form.SWAP_HORIZON,
                               bank=bank, margin=margin)
    print(f'\nsuggested swaps over the next {form.SWAP_HORIZON} gameweeks '
          f'(margin {margin:.1f} pts):')
    if len(swaps):
        if news:
            swaps = _attach_news(swaps, proj, pool, model)
        print(swaps.round(2).to_string(index=False))
        return proj, swaps

    # nothing qualifies: show the closest misses so the margin is calibratable
    print('  none clear the margin')
    near = form.suggest_swaps(codes, proj, pool, form.SWAP_HORIZON, bank=bank,
                              margin=margin, include_below=True)
    if len(near):
        if news:
            near = _attach_news(near, proj, pool, model)
        print('\nclosest misses (not suggested):')
        print(near.round(2).to_string(index=False))
    return proj, swaps


def _attach_news(swaps, proj, pool, model=None):
    """Advisory sentiment beside each swap. Never touches gain or verdict."""
    from form_lab import sentiment

    names = {}
    for side in ('out', 'in'):
        for n in swaps[side]:
            names.setdefault(n, proj.loc[proj['web_name'] == n, 'code'].iloc[0])

    verdicts = {n: sentiment.assess(c, model=model) for n, c in names.items()}
    for side in ('out', 'in'):
        swaps[f'{side}_news'] = [
            f"{verdicts[n]['sentiment']}"
            + (f" ({','.join(verdicts[n]['flags'])})"
               if verdicts[n]['flags'] else '')
            for n in swaps[side]]
    return swaps


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--demo', type=int, metavar='ENTRY_ID',
                    help='print projections and swaps for one cached team')
    ap.add_argument('--paid', action='store_true',
                    help='judge news with llm_explain.PAID_MODEL')
    args = ap.parse_args()

    paid = None
    if args.paid:
        import llm_explain as lx
        paid = lx.PAID_MODEL
        print(f'--paid: billing {paid}')
    demo(args.demo, model=paid) if args.demo else main()

