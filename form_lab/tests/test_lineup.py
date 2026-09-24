"""The next-gameweek lineup optimiser, on a hand-made squad."""
import pytest

import lineup

# 2 GK, 5 DEF, 5 MID, 3 FWD; codes by position, points if he plays
POS = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2,
       8: 3, 9: 3, 10: 3, 11: 3, 12: 3, 13: 4, 14: 4, 15: 4}
BASE = {c: float(20 - c) for c in POS}      # lower code = better player
FIT = {c: 1.0 for c in POS}


def counts(xi):
    return {p: sum(POS[c] == p for c in xi) for p in (1, 2, 3, 4)}


def test_best_xi_is_legal():
    xi = lineup.best(list(POS), BASE, FIT, POS)['xi']
    n = counts(xi)
    assert len(xi) == 11 and n[1] == 1
    assert 3 <= n[2] <= 5 and 2 <= n[3] <= 5 and 1 <= n[4] <= 3


def test_bench_starts_with_the_spare_keeper():
    b = lineup.best(list(POS), BASE, FIT, POS)['bench']
    assert POS[b[0]] == 1 and len(b) == 4


def test_captain_and_vice_differ():
    b = lineup.best(list(POS), BASE, FIT, POS)
    assert b['captain'] != b['vice']


def test_an_unavailable_player_is_never_captain():
    play = {**FIT, 3: 0.0}                 # the best outfield player is out
    base = {**BASE, 3: 99.0}
    b = lineup.best(list(POS), base, play, POS)
    assert b['captain'] != 3
    assert 3 not in b['xi']


def test_a_ruled_out_starter_is_subbed_off():
    current = {'xi': [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14],
               'bench': [2, 7, 12, 15], 'captain': 3, 'vice': 4}
    play = {**FIT, 3: 0.0}
    c = lineup.compare(current, BASE, play, POS)
    assert 3 not in c['best']['xi']
    assert c['total'] > 0
    assert c['best']['captain'] != 3


def test_steps_add_up_to_the_total():
    current = {'xi': [2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14],
               'bench': [1, 15, 12, 7], 'captain': 14, 'vice': 13}
    c = lineup.compare(current, BASE, FIT, POS)
    assert sum(c['steps'].values()) == pytest.approx(c['total'])
    assert c['total'] > 0


def test_the_best_lineup_reports_no_gain():
    b = lineup.best(list(POS), BASE, FIT, POS)
    current = {k: b[k] for k in ('xi', 'bench', 'captain', 'vice')}
    assert lineup.compare(current, BASE, FIT, POS)['total'] == pytest.approx(0)


def test_a_ruled_out_player_is_benched_even_when_auto_sub_would_cover_him():
    current = {'xi': [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14],
               'bench': [2, 7, 12, 15], 'captain': 3, 'vice': 4}
    play = {**FIT, 6: 0.0}                 # a weak defender, easily covered
    c = lineup.compare(current, BASE, play, POS)
    assert 6 not in c['best']['xi']
    assert c['total'] >= 0


def test_a_zero_value_vice_change_is_not_suggested():
    """With a certain captain the vice is worth nothing; keep the manager's."""
    b = lineup.best(list(POS), BASE, FIT, POS)
    other = next(c for c in b['xi'] if c not in (b['captain'], b['vice']))
    current = {**{k: b[k] for k in ('xi', 'bench', 'captain')}, 'vice': other}
    c = lineup.compare(current, BASE, FIT, POS)
    assert c['best']['vice'] == other
    assert c['total'] == pytest.approx(0)


def test_vice_matters_only_when_the_captain_may_not_play():
    xi = [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14]
    bench = [2, 7, 12, 15]
    e = lambda v, play: lineup.expected_points(xi, bench, 3, v, BASE, play, POS)
    assert e(4, FIT) == e(14, FIT)
    doubtful = {**FIT, 3: 0.5}
    assert e(4, doubtful) > e(14, doubtful)
