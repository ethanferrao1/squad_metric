"""The live 26/27 season replayed with the backtest's own logic, from the pre-season squad.

Transfers follow backtest.simulate exactly (greedy swaps, SWITCH_MARGIN, hits).
Chips are left out: their windows have not closed yet. Every gameweek's
projection is checked to use only gameweeks before its deadline.
"""
import pandas as pd

import fpl_api as api
from form_lab import backtest as bt, data

SEASON = '26_27'


def finished_gw(boot=None):
    """The last gameweek FPL has marked finished (0 before GW1 ends)."""
    boot = boot or api.bootstrap()
    done = [e['id'] for e in boot['events'] if e.get('finished')]
    return max(done) if done else 0


def average_scores(boot=None):
    """FPL's average manager score per finished gameweek."""
    boot = boot or api.bootstrap()
    return {e['id']: e['average_entry_score'] for e in boot['events']
            if e.get('finished') and e.get('average_entry_score') is not None}


def _inputs(pool, offline):
    by = pool[pool['modelled']].drop_duplicates('code').set_index('code')
    last = finished_gw()
    hist = data.live_history(list(by.index), offline=offline)
    hist = hist[hist['gw'] <= last].reset_index(drop=True)
    start = hist.sort_values('gw').groupby('code')['price'].first()
    return hist, by['team_code'], by['element_type'], by['pred'], start, last


def assert_pre_deadline(hist, pri, last):
    """Each gameweek's projection must equal one built from earlier gameweeks only."""
    for gw in range(2, last + 1):
        full = bt.projections(hist, gw, pri, 'blend')
        cut = bt.projections(hist[hist['gw'] < gw], gw, pri, 'blend')
        assert full.sort_index().equals(cut.sort_index()), f'GW{gw} saw later data'


def transfer_log(rows):
    """[(gw, outs, ins)] from the squads before and after each deadline."""
    log = []
    for r in rows.itertuples():
        before, after = set(r.squad_start), set(r.squad)
        if before != after:
            log.append((r.gw, sorted(before - after), sorted(after - before)))
    return log


def replay(pool, offline=True):
    """Followed, held and average-manager points for the live season so far."""
    hist, clubs, pos, pri, start, last = _inputs(pool, offline)
    if last == 0 or hist.empty:
        return None
    assert_pre_deadline(hist, pri, last)
    followed, _ = bt.simulate('blend', SEASON, hist, clubs, pos, pri, start=start)
    held, _ = bt.simulate('no transfers', SEASON, hist, clubs, pos, pri, start=start)
    avg = average_scores()
    per_gw = pd.DataFrame({'gw': followed['gw'], 'followed': followed['points'],
                           'held': held['points'].to_numpy(),
                           'average': followed['gw'].map(avg)})
    return {'gw': last, 'preseason': followed.iloc[0]['squad_start'],
            'followed': followed, 'per_gw': per_gw, 'log': transfer_log(followed)}
