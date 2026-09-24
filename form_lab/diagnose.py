"""Where the blend loses points: five diagnostics per season.

    python -m form_lab.diagnose

Replays the simulation and measures blanks left in the XI, captaincy regret,
chip timing against hindsight, the early-season split against form only, and
players kept in the XI while clearly not playing.
"""
import numpy as np
import pandas as pd

from form_lab import backtest as bt

STRATEGY = 'blend'
REFERENCE = 'form only'
EARLY = 6
INJURY_RUN = 2


def replay(season, strategy=STRATEGY, chips=True):
    """(gameweek frame, chip points, history) for one strategy."""
    hist, clubs, pos, pri = bt.season_data(season)
    frame, chip_pts = bt.simulate(strategy, season, hist, clubs, pos, pri,
                                  chips=chips)
    return frame, chip_pts, hist, pos


def gw_lookup(hist):
    """gw -> {code: (minutes, points)} for every gameweek played."""
    out = {}
    for gw, g in hist.groupby('gw'):
        out[int(gw)] = dict(zip(g['code'], zip(g['minutes'], g['points'])))
    return out


def blanks_left_in(frame, lookup):
    """Starters on 0 minutes that auto-subs could not replace."""
    n = 0
    for r in frame.itertuples():
        by_code = lookup.get(r.gw, {})
        n += sum(1 for c in r.xi if by_code.get(c, (0, 0))[0] == 0)
    return n


def captaincy(frame, lookup):
    """(captain points, best-in-XI points) -- the armband's regret."""
    got = best = 0
    for r in frame.itertuples():
        by_code = lookup.get(r.gw, {})
        scores = [by_code.get(c, (0, 0))[1] for c in r.xi]
        if not scores:
            continue
        got += by_code.get(r.scorer, (0, 0))[1]
        best += max(scores)
    return got, best


def chip_timing(season, frame, lookup):
    """Each chip's actual return against the best gameweek in its window.

    The counterfactual uses the squad actually held that week, which is what
    you would really have had; it is not a fixed-squad comparison.
    """
    by_gw = {int(r.gw): r for r in frame.itertuples()}
    rows = []
    for r in frame.itertuples():
        if not r.chip or r.chip == 'wildcard':
            continue
        lo, hi = next((l, h) for l, h in bt.CHIP_RULES[season][r.chip]
                      if l <= r.gw <= h)
        actual = chip_value(r.chip, by_gw[r.gw], lookup.get(r.gw, {}))
        best, best_gw = actual, r.gw
        for gw in range(lo, hi + 1):
            if gw not in by_gw:
                continue
            v = chip_value(r.chip, by_gw[gw], lookup.get(gw, {}))
            if v > best:
                best, best_gw = v, gw
        rows.append({'chip': r.chip, 'played_gw': r.gw, 'actual': actual,
                     'best_gw': best_gw, 'best': best, 'missed': best - actual})
    return pd.DataFrame(rows)


def chip_value(chip, row, by_code):
    """Extra points this chip returns for that gameweek's squad."""
    if chip == 'tcaptain':
        return by_code.get(row.scorer, (0, 0))[1]
    if chip == 'bboost':
        return sum(by_code.get(c, (0, 0))[1] for c in row.bench)
    return 0


def phase_split(frame, reference, early=EARLY):
    """Points per gameweek early and late, against the reference strategy."""
    rows = []
    for label, lo, hi in (('GW1-6', 1, early), ('GW7-38', early + 1, 99)):
        a = frame[(frame['gw'] >= lo) & (frame['gw'] <= hi)]['points']
        b = reference[(reference['gw'] >= lo) &
                      (reference['gw'] <= hi)]['points']
        rows.append({'phase': label, 'gws': len(a),
                     'blend_ppg': a.mean(), 'form_only_ppg': b.mean(),
                     'diff_ppg': a.mean() - b.mean(),
                     'diff_total': a.sum() - b.sum()})
    return pd.DataFrame(rows)


def stale_starts(frame, hist, run=INJURY_RUN):
    """XI selections made after a player's last `run` team games were blanks."""
    played = {}
    for code, g in hist.sort_values('gw').groupby('code'):
        played[code] = list(zip(g['gw'], g['minutes']))

    n = 0
    for r in frame.itertuples():
        for c in r.xi:
            prior = [m for gw, m in played.get(c, []) if gw < r.gw]
            if len(prior) >= run and all(m == 0 for m in prior[-run:]):
                n += 1
    return n


CAPTAIN_RULES = ('projection', 'most selected', 'top priced')


def captain_rules(frame, hist):
    """Captain points under three rules, on the same picked XI each week.

    Ownership is the last figure before the deadline (GW1 uses GW1's own);
    price is the price that gameweek. The vice scores if the captain blanks.
    """
    lookup = gw_lookup(hist)
    h = hist.sort_values('gw')
    out = dict.fromkeys(CAPTAIN_RULES, 0)
    for r in frame.itertuples():
        src = h[h['gw'] < r.gw] if r.gw > 1 else h[h['gw'] == 1]
        sel = src.groupby('code')['selected'].last()
        price = h[h['gw'] <= r.gw].groupby('code')['price'].last()
        played = lookup.get(r.gw, {})
        picks = {
            'projection': [r.captain, r.vice],
            'most selected': sorted(r.picked, key=lambda c: -sel.get(c, 0))[:2],
            'top priced': sorted(r.picked, key=lambda c: -price.get(c, 0))[:2],
        }
        for rule, (cap, vice) in picks.items():
            who = cap if played.get(cap, (0, 0))[0] > 0 else vice
            out[rule] += played.get(who, (0, 0))[1]
    return out


def chip_weeks(frame):
    """'TC7 BB26 WC16' -- which chip went where."""
    short = {'tcaptain': 'TC', 'bboost': 'BB', 'wildcard': 'WC'}
    return ' '.join(f'{short[r.chip]}{r.gw}' for r in frame.itertuples()
                    if r.chip)


def compare(seasons=bt.SEASONS, strategies=(STRATEGY, REFERENCE)):
    """Before vs after for the bench and chip changes, plus the captain rules."""
    from form_lab import managers as mg

    out = bt.data.CACHE.parent / 'results'
    before = pd.read_csv(out / 'before.csv')
    rows, caps = [], []
    for season in seasons:
        sample = mg.load(season)
        for strategy in strategies:
            frame, chip_pts, hist, _ = replay(season, strategy)
            total = int(frame['points'].sum())
            rows.append({
                'season': season, 'strategy': strategy, 'total': total,
                'top_pct': mg.top_pct(sample, total),
                'blanks': blanks_left_in(frame, gw_lookup(hist)),
                'chip_pts': int(sum(v for k, v in chip_pts.items()
                                    if k != 'wildcard')),
                'transfers': int(frame['transfers'].sum()),
                'chips': chip_weeks(frame)})
            if strategy == STRATEGY:
                caps.append({'season': season, **captain_rules(frame, hist)})
            print(f'  {season} {strategy} done', flush=True)

    after = pd.DataFrame(rows)
    both = before.merge(after, on=['season', 'strategy'],
                        suffixes=('_before', '_after'))
    cols = ['season', 'strategy']
    for m in ('total', 'top_pct', 'blanks', 'chip_pts'):
        cols += [f'{m}_before', f'{m}_after']
    table = both[cols]
    sums = table.groupby('strategy')[[c for c in cols[2:]
                                      if not c.startswith('top_pct')]].sum()
    sums = sums.reset_index().assign(season='3-season')
    table = pd.concat([table, sums], ignore_index=True)[cols]

    print('\nbefore vs after (chips+hits)\n')
    print(table.to_string(index=False))
    print('\nchip weeks after\n')
    print(after[['season', 'strategy', 'chips']].to_string(index=False))

    cap = pd.DataFrame(caps)
    cap = pd.concat([cap, cap.drop(columns='season').sum().to_frame().T
                     .assign(season='3-season')], ignore_index=True)
    print(f'\ncaptain points by rule ({STRATEGY}, report only)\n')
    print(cap[['season', *CAPTAIN_RULES]].to_string(index=False))

    totals = (after.groupby('strategy')[['total', 'transfers']].sum()
              .sort_values('total', ascending=False))
    print(f'\n3-season totals after\n\n{totals.to_string()}')
    print(f'\nwinner: {bt.decide(totals)}')

    table.to_csv(out / 'before_after.csv', index=False)
    after.to_csv(out / 'after.csv', index=False)
    cap.to_csv(out / 'captain_rules.csv', index=False)
    return table, cap, totals


def for_season(season):
    """Every diagnostic for one season."""
    frame, chip_pts, hist, _ = replay(season)
    ref, _, _, _ = replay(season, REFERENCE)
    lookup = gw_lookup(hist)

    got, best = captaincy(frame, lookup)
    chips = chip_timing(season, frame, lookup)
    phases = phase_split(frame, ref)

    summary = {
        'season': season,
        'total': int(frame['points'].sum()),
        'blanks_in_xi': blanks_left_in(frame, lookup),
        'captain_pts': got,
        'best_captain_pts': best,
        'captain_regret': best - got,
        'chip_pts': int(chips['actual'].sum()) if len(chips) else 0,
        'chip_missed': int(chips['missed'].sum()) if len(chips) else 0,
        'early_diff': round(phases.loc[0, 'diff_total'], 1),
        'late_diff': round(phases.loc[1, 'diff_total'], 1),
        'stale_starts': stale_starts(frame, hist),
    }
    return summary, chips, phases


def main(seasons=bt.SEASONS):
    summaries, chip_frames, phase_frames = [], [], []
    for season in seasons:
        s, chips, phases = for_season(season)
        summaries.append(s)
        chip_frames.append(chips.assign(season=season))
        phase_frames.append(phases.assign(season=season))
        print(f'  {season} done', flush=True)

    table = pd.DataFrame(summaries)
    totals = table.drop(columns='season').sum()
    totals['season'] = '3-season'
    table = pd.concat([table, pd.DataFrame([totals])[table.columns]],
                      ignore_index=True)

    print(f'\ndiagnostics for "{STRATEGY}" (chips+hits)\n')
    print(table.to_string(index=False))

    print('\nchip timing against the best gameweek in each window\n')
    print(pd.concat(chip_frames).to_string(index=False))

    print(f'\nper-gameweek scoring vs "{REFERENCE}"\n')
    print(pd.concat(phase_frames).round(2).to_string(index=False))

    out = bt.data.CACHE.parent / 'results'
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'diagnostics.csv', index=False)
    pd.concat(chip_frames).to_csv(out / 'chip_timing.csv', index=False)
    pd.concat(phase_frames).to_csv(out / 'phase_split.csv', index=False)
    print(f'\ncsv: {out}')
    return table


if __name__ == '__main__':
    import sys
    compare() if '--compare' in sys.argv else main()
