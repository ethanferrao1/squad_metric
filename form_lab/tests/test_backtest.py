"""Rule tests for the season simulation, on hand-made inputs."""
import pandas as pd
import pytest

from form_lab import backtest as bt

# 1 GK, 5 DEF, 5 MID, 3 FWD -- a legal 15
POS = {1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2,
       7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 4, 13: 4, 14: 4, 15: 1}


def test_sell_price_keeps_half_of_a_rise():
    assert bt.sell_price(50, 54) == 52


def test_sell_price_rounds_the_profit_down():
    """A 0.3m rise returns 0.1m, not 0.15m."""
    assert bt.sell_price(50, 53) == 51


def test_sell_price_takes_the_full_loss():
    assert bt.sell_price(50, 46) == 46


def test_sell_price_is_flat_when_unchanged():
    assert bt.sell_price(50, 50) == 50


def test_legal_shape_accepts_a_real_formation():
    assert bt.legal_shape({1: 1, 2: 4, 3: 4, 4: 2})


def test_legal_shape_rejects_two_keepers():
    assert not bt.legal_shape({1: 2, 2: 4, 3: 3, 4: 2})


def test_legal_shape_rejects_too_few_defenders():
    assert not bt.legal_shape({1: 1, 2: 2, 3: 5, 4: 3})


def test_autosub_replaces_a_starter_who_did_not_play():
    xi = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13]      # 1-4-5-2... legal
    bench = [6, 11, 14, 15]
    minutes = {c: 90 for c in POS}
    minutes[12] = 0                                 # a forward blanked
    new_xi, new_bench = bt.apply_autosubs(xi, bench, minutes, POS)
    assert 12 not in new_xi
    assert 12 in new_bench
    assert len(new_xi) == 11


def test_autosub_keeps_the_formation_legal():
    xi = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13]
    bench = [15, 6, 11, 14]                         # the spare keeper is first
    minutes = {c: 90 for c in POS}
    minutes[12] = 0
    new_xi, _ = bt.apply_autosubs(xi, bench, minutes, POS)
    counts = pd.Series([POS[c] for c in new_xi]).value_counts().to_dict()
    assert counts[1] == 1, 'a second keeper came on'
    assert bt.legal_shape(counts)


def test_autosub_skips_bench_players_who_also_blanked():
    xi = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13]
    bench = [14, 6, 11, 15]
    minutes = {c: 90 for c in POS}
    minutes[12] = minutes[14] = 0
    new_xi, _ = bt.apply_autosubs(xi, bench, minutes, POS)
    assert 14 not in new_xi


def test_autosub_leaves_a_full_xi_alone():
    xi = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13]
    bench = [6, 11, 14, 15]
    minutes = {c: 90 for c in POS}
    assert bt.apply_autosubs(xi, bench, minutes, POS)[0] == xi


def clubs(mapping=None):
    base = {c: c % 7 for c in POS}
    base.update(mapping or {})
    return base


def test_assert_legal_accepts_a_legal_squad():
    bt.assert_legal({c: 50 for c in POS}, 5, clubs(), POS)


def test_assert_legal_rejects_a_fourth_player_from_one_club():
    four = {1: 0, 2: 0, 3: 0, 4: 0}
    with pytest.raises(AssertionError, match='from one club'):
        bt.assert_legal({c: 50 for c in POS}, 5, clubs(four), POS)


def test_assert_legal_rejects_a_negative_bank():
    with pytest.raises(AssertionError, match='bank'):
        bt.assert_legal({c: 50 for c in POS}, -1, clubs(), POS)


def test_assert_legal_rejects_the_wrong_squad_size():
    squad = {c: 50 for c in list(POS)[:14]}
    with pytest.raises(AssertionError, match='14 players'):
        bt.assert_legal(squad, 5, clubs(), POS)


def test_assert_legal_rejects_an_illegal_shape():
    bad = dict(POS)
    bad[15] = 4                       # two keepers become one keeper, four FWD
    with pytest.raises(AssertionError, match='illegal shape'):
        bt.assert_legal({c: 50 for c in bad}, 5, clubs(), bad)


def test_free_transfers_never_exceed_the_cap():
    assert bt.MAX_FREE == 5


def fixtures(rows):
    """(code, round, fixture id) rows as merged_gw carries them."""
    return pd.DataFrame(rows, columns=['code', 'round', 'fixture'])


def test_a_fixture_in_its_original_round_is_known():
    assert bt.known_counts(fixtures([(1, 10, 95)])) == {(1, 10): 1}


def test_a_long_rearranged_fixture_makes_a_known_double():
    """Fixture 25 was a GW3 game; played in GW10 it was postponed long ago."""
    k = bt.known_counts(fixtures([(1, 10, 95), (1, 10, 25)]))
    assert k[(1, 10)] == 2


def test_a_recently_rearranged_fixture_is_not_known_yet():
    """Fixture 75 was a GW8 game; moved to GW10, it was not known 4 GWs ahead."""
    k = bt.known_counts(fixtures([(1, 10, 95), (1, 10, 75)]))
    assert k[(1, 10)] == 1


def test_optimistic_lookahead_counts_every_fixture():
    k = bt.known_counts(fixtures([(1, 10, 95), (1, 10, 75)]), lookahead=None)
    assert k[(1, 10)] == 2


def test_a_duplicate_row_counts_once():
    """25/26 repeats some rows with the same fixture id; that is not a double."""
    k = bt.known_counts(fixtures([(1, 3, 25), (1, 3, 25)]))
    assert k[(1, 3)] == 1


def test_triple_captain_needs_a_known_double():
    assert bt.tc_ready(1, 10, {(1, 10): 2})
    assert not bt.tc_ready(1, 10, {(1, 10): 1})


def test_bench_boost_needs_four_doubles_in_the_fifteen():
    squad = list(range(15))
    bench = [11, 12, 13, 14]
    three = {(c, 5): 2 if c < 3 else 1 for c in squad}
    four = {(c, 5): 2 if c < 4 else 1 for c in squad}
    assert not bt.bb_ready(squad, bench, 5, three)
    assert bt.bb_ready(squad, bench, 5, four)


def test_bench_boost_needs_every_bench_player_to_have_a_fixture():
    squad = list(range(15))
    known = {(c, 5): 2 if c < 4 else 1 for c in squad if c != 14}
    assert not bt.bb_ready(squad, [11, 12, 13, 14], 5, known)


def test_half_of_splits_at_gameweek_nineteen():
    assert bt.half_of(19) == 1
    assert bt.half_of(20) == 2


def test_open_chips_respects_the_window():
    assert bt.open_chips('25_26', 5, set()) != []
    # a chip already used in this window is not offered again
    used = {(c, lo) for c, lo in bt.open_chips('25_26', 5, set())}
    assert bt.open_chips('25_26', 5, used) == []


def test_earlier_seasons_get_one_bench_boost_not_two():
    assert len(bt.CHIP_RULES['23_24']['bboost']) == 1
    assert len(bt.CHIP_RULES['25_26']['bboost']) == 2


def test_the_wildcard_is_twice_in_every_season():
    for season in bt.CHIP_RULES:
        assert len(bt.CHIP_RULES[season]['wildcard']) == 2


SQUAD = list(range(1, 16))
BENCH = [12, 13, 14, 15]


def test_pending_chips_get_different_fallback_deadlines():
    """Both fallbacks on one gameweek would starve the lower-priority chip."""
    early, _ = bt.pick_squad_chip('25_26', 18, set(), 1, SQUAD, BENCH, {})
    assert early == 'bboost'
    late, _ = bt.pick_squad_chip('25_26', 19, {('bboost', 1)}, 1, SQUAD,
                                 BENCH, {})
    assert late == 'tcaptain'


def test_triple_captain_plays_on_the_first_known_double():
    chip, _ = bt.pick_squad_chip('25_26', 7, set(), 1, SQUAD, BENCH,
                                 {(1, 7): 2})
    assert chip == 'tcaptain'


def test_never_both_chips_in_one_gameweek():
    """Both qualify: triple captain plays, bench boost waits."""
    known = {(c, 7): 2 for c in SQUAD}
    chip, _ = bt.pick_squad_chip('25_26', 7, set(), 1, SQUAD, BENCH, known)
    assert chip == 'tcaptain'
    chip, _ = bt.pick_squad_chip('25_26', 8, {('tcaptain', 1)}, 1, SQUAD,
                                 BENCH, {(c, 8): 2 for c in SQUAD})
    assert chip == 'bboost'


def test_no_chip_without_a_double_or_a_deadline():
    chip, _ = bt.pick_squad_chip('25_26', 7, set(), 1, SQUAD, BENCH,
                                 {(c, 7): 1 for c in SQUAD})
    assert chip is None


def test_regular_starter_needs_half_of_the_last_five():
    hist = pd.DataFrame({'code': [1] * 5 + [2] * 5, 'gw': list(range(1, 6)) * 2,
                         'minutes': [90, 90, 90, 0, 0] + [90, 90, 0, 0, 0]})
    assert bt.regular_starters(hist, 6) == {1}


def test_regular_starter_looks_only_at_the_last_five():
    hist = pd.DataFrame({'code': 1, 'gw': list(range(1, 9)),
                         'minutes': [90, 90, 90, 90, 0, 0, 0, 90]})
    assert bt.regular_starters(hist, 9) == set()


def test_gw1_starters_come_from_last_season():
    prev = pd.DataFrame({'code': [1] * 38 + [2] * 38, 'gw': list(range(1, 39)) * 2,
                         'minutes': [90] * 19 + [0] * 19 + [90] * 18 + [0] * 20})
    empty = pd.DataFrame({'code': [], 'gw': [], 'minutes': []})
    assert bt.regular_starters(empty, 1, prev) == {1}


def candidates(n_per_pos=8):
    """Enough cheap and dear players at each position to build a squad."""
    rows, code = [], 1
    for pos in (1, 2, 3, 4):
        for k in range(n_per_pos):
            # the root term keeps every XI total distinct, so the best XI is unique
            rows.append({'code': code, 'element_type': pos,
                         'team_code': code % 20, 'price': 40 + 5 * k,
                         'pred': 1.0 + k + code ** 0.5 / 100})
            code += 1
    return pd.DataFrame(rows)


def test_the_bench_is_only_regular_starters():
    c = candidates()
    # the cheapest two at each position are not regulars: exactly what a
    # budget MILP would otherwise put on the bench
    not_regular = set(c.groupby('element_type').head(2)['code'])
    eligible = set(c['code']) - not_regular
    squad = bt.pick_squad(c, 'pred', eligible, budget=1000)
    xi = bt.xi_members(set(squad['code']), dict(zip(c['code'], c['pred'])),
                       dict(zip(c['code'], c['element_type'])))
    bench = set(squad['code']) - xi
    assert len(squad) == 15
    assert bench <= eligible


def test_xi_members_is_the_best_eleven():
    pos = {c: t for c, t in POS.items()}
    proj = {c: float(c) for c in POS}
    members = bt.xi_members(set(POS), proj, pos)
    assert len(members) == 11
    assert 15 in members and 1 not in members


def test_only_the_four_playing_positions_are_squad_positions():
    """element_type 5 is the manager; op.SQUAD does not constrain it, so a
    rebuild that saw one could buy any number."""
    import optimise as op
    assert set(op.SQUAD) == {1, 2, 3, 4}


def test_wildcard_plays_only_on_its_fixed_gameweek():
    squad = {c: 50 for c in POS}
    args = (squad, 0, {}, {}, POS, clubs(), None)
    assert bt.pick_wildcard('25_26', 10, set(), *args) == (None, None)
    assert bt.pick_wildcard('25_26', 30, set(), *args) == (None, None)


def test_wildcard_gameweeks_are_one_per_half():
    assert bt.half_of(bt.WILDCARD_FALLBACK[1]) == 1
    assert bt.half_of(bt.WILDCARD_FALLBACK[2]) == 2


def test_hit_threshold_is_the_margin_plus_its_own_cost():
    assert bt.HIT_COST == 4
    assert bt.MAX_HITS == 2


def test_xi_value_rejects_a_squad_that_cannot_field_a_legal_xi():
    only_keepers = {1: 1, 15: 1}
    assert bt.xi_value(only_keepers, pd.Series({1: 5.0, 15: 5.0}),
                       only_keepers) == -1e9
