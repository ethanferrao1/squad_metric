"""Free hit mechanics and triggers."""
import pandas as pd
import pytest

from form_lab import backtest as bt, freehit as fh


def test_blank_trigger_counts_starters_without_a_fixture():
    nfix = {(1, 7): 1, (2, 7): 0, (3, 7): 0, (4, 7): 0}
    assert len(fh.blank_starters({1, 2, 3, 4}, nfix, 7)) == 3


def test_unavailable_trigger_needs_zero_minutes_in_both_of_the_last_two_games():
    hist = pd.DataFrame({'code': [1, 1, 1, 2, 2, 2], 'gw': [1, 2, 3] * 2,
                         'minutes': [90, 0, 0, 0, 0, 90]})
    assert fh.stale(hist, 4) == {1}


def test_big_game_trigger_counts_the_xi_in_one_top_fixture():
    rows = pd.DataFrame({'code': [1, 2, 3, 4, 5], 'fixture': [9, 9, 9, 9, 8],
                         'club': [10, 10, 11, 11, 12], 'opp': [11, 11, 10, 10, 13]})
    assert fh.big_game_stack({1, 2, 3, 4, 5}, rows, top={10, 11}) == 4
    assert fh.big_game_stack({1, 2, 3, 4, 5}, rows, top={10, 12}) == 0


def test_top_clubs_rank_by_rolling_goal_difference():
    t = pd.Timestamp('2024-09-01', tz='UTC')
    log = pd.DataFrame({'club': [1, 2, 3], 'kickoff': [t] * 3,
                        'scored': [3, 1, 2], 'conceded': [0, 2, 2]})
    assert fh.top_clubs(log, t + pd.Timedelta(days=1), n=2) == {1, 3}


def test_budget_is_selling_value_plus_bank():
    """A 0.4m rise sells for half of it; a fall is taken in full."""
    squad = {1: 50, 2: 60}
    assert fh.budget(squad, 5, {1: 54, 2: 57}) == 5 + 52 + 57


@pytest.fixture(scope='module')
def season():
    # a full-season replay: needs the historical season CSVs in data/, which
    # are not committed (see README, "Data sources")
    try:
        hist, clubs, pos, pri = bt.season_data('24_25')
    except FileNotFoundError as e:
        pytest.skip(f'historical season data not present: {e.filename}')
    d, _ = bt.simulate('blend', '24_25', hist, clubs, pos, pri, chips=True,
                       free_hit={'threshold': -1e9, 'triggers': 'abc'})
    return d


def test_the_free_hit_is_played_and_reverts_exactly(season):
    d = season.set_index('gw')
    fh_gws = list(d.index[d['chip'] == 'freehit'])
    assert fh_gws
    for g in fh_gws:
        assert d.at[g + 1, 'squad_start'] == d.at[g, 'squad_start']
        assert d.at[g + 1, 'bank_start'] == d.at[g, 'bank_start']
        assert sorted(d.at[g, 'squad']) != d.at[g, 'squad_start']


def test_no_two_chips_share_a_gameweek_and_counts_follow_the_config(season):
    chips = season.loc[season['chip'] != '', 'chip']
    assert chips.index.is_unique
    allowed = {c: len(w) for c, w in bt.CHIP_RULES['24_25'].items()}
    for chip, n in chips.value_counts().items():
        assert n <= allowed[chip]
