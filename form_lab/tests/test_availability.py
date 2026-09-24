"""Availability rules and their effect on swaps and the captain."""
import pandas as pd
import pytest

import availability as av
from form_lab import form

# 1-4-4-2 XI at 5.0 per gameweek; the four on the bench at 1.0
XI = {1: 1, 2: 4, 3: 4, 4: 2}
BENCH = {1: 1, 2: 1, 3: 1, 4: 1}


def squad_and_pool():
    """A squad plus one spare per position at 4.0 per gameweek."""
    rows, ppg, code = [], {}, 1
    for group, value in ((XI, 5.0), (BENCH, 1.0), ({1: 1, 2: 1, 3: 1, 4: 1}, 4.0)):
        for pos, n in group.items():
            for _ in range(n):
                rows.append({'code': code, 'element_type': pos,
                             'team_code': code, 'price': 50, 'pred': 100.0,
                             'web_name': f'p{code}', 'pos': str(pos),
                             'modelled': True})
                ppg[code] = value
                code += 1
    pool = pd.DataFrame(rows)
    squad = list(range(1, 16))
    proj = pd.DataFrame({'code': list(ppg), 'blended_ppg': list(ppg.values()),
                         'has_fixture': True})
    return squad, pool, proj


INJURED = 6          # the first midfielder in the XI


def test_doubtful_at_75_is_scaled_for_one_gameweek_only():
    assert av.multipliers('d', 75) == [0.75, 1.0, 1.0, 1.0, 1.0]


def test_suspended_is_out_for_one_gameweek_only():
    assert av.multipliers('s', 0) == [0.0, 1.0, 1.0, 1.0, 1.0]


def test_injured_with_no_chance_is_out_for_the_whole_window():
    assert av.multipliers('i', 0) == [0.0] * 5
    assert av.multipliers('u', None) == [0.0] * 5


def test_available_is_unchanged():
    assert av.multipliers('a', None) == [1.0] * 5


def test_injured_with_some_chance_is_treated_as_doubtful():
    assert av.multipliers('i', 25) == [0.25, 1.0, 1.0, 1.0, 1.0]


def test_season_factor_counts_missed_gameweeks():
    assert av.season_factor({'mult': [0.0] * 5}) == pytest.approx(1 - 5 / 38)
    assert av.season_factor({'mult': [1.0] * 5}) == 1.0


def test_identical_later_gameweeks_are_scored_once():
    weeks = form._weeks({1, 2}, {1: av.multipliers('d', 75)}, 5)
    assert sorted(w for _, w in weeks) == [1, 4]


def test_an_injured_starter_produces_a_replacement():
    squad, pool, proj = squad_and_pool()
    fit = form.suggest_swaps(squad, proj, pool, 5)
    assert f'p{INJURED}' not in set(fit.get('out', []))

    hurt = form.suggest_swaps(squad, proj, pool, 5,
                              avail={INJURED: av.multipliers('i', 0)})
    top = hurt.iloc[0]
    assert top['out'] == f'p{INJURED}'
    # he was worth 5.0, the bench midfielder who covers him 1.0, the spare 4.0
    assert top['gain_pts'] == pytest.approx((4.0 - 1.0) * 5)


def test_a_suspended_starter_costs_only_one_gameweek():
    squad, pool, proj = squad_and_pool()
    out = form.suggest_swaps(squad, proj, pool, 5, include_below=True,
                             only_out={INJURED},
                             avail={INJURED: av.multipliers('s', 0)})
    # one week his cover beats the bench, four weeks the spare is worse than him
    assert out.iloc[0]['gain_pts'] == pytest.approx((4 - 1) - 4 * (5 - 4))


def test_a_small_cover_gain_means_substitute_instead():
    """The gain is already measured against starting the bench player."""
    assert form.cover_verdict(2.2) == 'substitute instead'
    assert form.cover_verdict(form.SWITCH_MARGIN) == 'injury cover'


def test_live_positions_override_last_seasons():
    import build_live
    pool = pd.DataFrame({'code': [1, 2, 3], 'element_type': [3, 3, 2],
                         'pred': [50.0, 40.0, 30.0]})
    out = build_live.add_live_positions(pool, {1: 2, 2: 3, 3: 5})
    assert list(out['element_type']) == [2, 3, 2]   # 5 (manager) is ignored
    assert list(out['pos']) == ['DEF', 'MID', 'DEF']


def test_an_injured_player_is_never_captain():
    squad, pool, proj = squad_and_pool()
    proj.loc[proj['code'] == INJURED, 'blended_ppg'] = 99.0
    xi = pool[pool['code'].isin(range(1, 12))]
    assert form.captain(xi, proj) == INJURED
    assert form.captain(xi, proj, unavailable={INJURED}) != INJURED
