"""Tests for the diagnostics, on hand-made gameweek traces."""
import pandas as pd

from form_lab import diagnose as dg


def frame(rows):
    return pd.DataFrame(rows)


def test_blanks_counts_only_starters_on_zero_minutes():
    f = frame([{'gw': 1, 'xi': [1, 2, 3], 'bench': [4], 'scorer': 1, 'chip': ''}])
    lookup = {1: {1: (90, 5), 2: (0, 0), 3: (60, 2)}}
    assert dg.blanks_left_in(f, lookup) == 1


def test_blanks_ignores_the_bench():
    f = frame([{'gw': 1, 'xi': [1], 'bench': [2, 3], 'scorer': 1, 'chip': ''}])
    lookup = {1: {1: (90, 5), 2: (0, 0), 3: (0, 0)}}
    assert dg.blanks_left_in(f, lookup) == 0


def test_captaincy_regret_is_the_gap_to_the_best_starter():
    f = frame([{'gw': 1, 'xi': [1, 2], 'bench': [], 'scorer': 1, 'chip': ''}])
    lookup = {1: {1: (90, 3), 2: (90, 12)}}
    got, best = dg.captaincy(f, lookup)
    assert (got, best) == (3, 12)


def test_captaincy_has_no_regret_when_the_captain_is_top():
    f = frame([{'gw': 1, 'xi': [1, 2], 'bench': [], 'scorer': 2, 'chip': ''}])
    lookup = {1: {1: (90, 3), 2: (90, 12)}}
    got, best = dg.captaincy(f, lookup)
    assert got == best == 12


def test_chip_value_of_a_bench_boost_is_the_bench():
    row = next(frame([{'gw': 1, 'xi': [1], 'bench': [2, 3], 'scorer': 1}])
               .itertuples())
    assert dg.chip_value('bboost', row, {2: (90, 4), 3: (90, 6)}) == 10


def test_chip_value_of_a_triple_captain_is_the_captain():
    row = next(frame([{'gw': 1, 'xi': [1], 'bench': [2], 'scorer': 1}])
               .itertuples())
    assert dg.chip_value('tcaptain', row, {1: (90, 9)}) == 9


def test_wildcard_has_no_direct_chip_value():
    row = next(frame([{'gw': 1, 'xi': [1], 'bench': [2], 'scorer': 1}])
               .itertuples())
    assert dg.chip_value('wildcard', row, {1: (90, 9)}) == 0


def test_captain_rules_score_each_rule_on_the_same_xi():
    f = frame([{'gw': 2, 'picked': [1, 2, 3], 'captain': 1, 'vice': 2}])
    hist = pd.DataFrame({
        'code': [1, 2, 3] * 2, 'gw': [1] * 3 + [2] * 3,
        'minutes': [90] * 6, 'points': [0, 0, 0, 3, 8, 12],
        'selected': [100, 900, 50, 100, 900, 50],
        'price': [60, 50, 130, 60, 50, 130]})
    got = dg.captain_rules(f, hist)
    assert got == {'projection': 3, 'most selected': 8, 'top priced': 12}


def test_captain_rules_fall_back_to_the_vice():
    f = frame([{'gw': 2, 'picked': [1, 2], 'captain': 1, 'vice': 2}])
    hist = pd.DataFrame({
        'code': [1, 2] * 2, 'gw': [1, 1, 2, 2], 'minutes': [90, 90, 0, 90],
        'points': [0, 0, 0, 7], 'selected': [9, 1, 9, 1],
        'price': [90, 50, 90, 50]})
    assert dg.captain_rules(f, hist)['projection'] == 7


def hist_for(code, minutes_by_gw):
    return pd.DataFrame({'code': code, 'gw': list(minutes_by_gw),
                         'minutes': list(minutes_by_gw.values())})


def test_stale_start_after_two_consecutive_blanks():
    f = frame([{'gw': 4, 'xi': [1], 'bench': [], 'scorer': 1, 'chip': ''}])
    hist = hist_for(1, {1: 90, 2: 0, 3: 0})
    assert dg.stale_starts(f, hist) == 1


def test_one_blank_is_not_stale():
    f = frame([{'gw': 4, 'xi': [1], 'bench': [], 'scorer': 1, 'chip': ''}])
    hist = hist_for(1, {1: 90, 2: 90, 3: 0})
    assert dg.stale_starts(f, hist) == 0


def test_a_return_to_playing_clears_the_run():
    f = frame([{'gw': 5, 'xi': [1], 'bench': [], 'scorer': 1, 'chip': ''}])
    hist = hist_for(1, {1: 0, 2: 0, 3: 0, 4: 90})
    assert dg.stale_starts(f, hist) == 0


def test_blank_gameweeks_do_not_count_as_zero_minutes():
    """A gameweek his club did not play has no row, so it breaks no run."""
    f = frame([{'gw': 6, 'xi': [1], 'bench': [], 'scorer': 1, 'chip': ''}])
    hist = hist_for(1, {1: 90, 4: 0, 5: 0})
    assert dg.stale_starts(f, hist) == 1
