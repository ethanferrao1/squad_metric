"""Tests for the fixture adjustment, on hand-made inputs."""
import pandas as pd
import pytest

from form_lab import fixtures as fx


def test_harder_fixtures_lower_the_projection():
    adj = fx.adjust(pd.Series({1: 4.0, 2: 4.0}), {1: 1.0, 2: -1.0}, 0.1)
    assert adj[1] == pytest.approx(3.6)
    assert adj[2] == pytest.approx(4.4)


def test_k_zero_changes_nothing():
    ppg = pd.Series({1: 4.0})
    assert fx.adjust(ppg, {1: 2.0}, 0)[1] == 4.0


def test_a_player_without_fixtures_in_the_window_is_unadjusted():
    assert fx.adjust(pd.Series({1: 4.0}), {}, 0.2)[1] == 4.0


def log(rows):
    return pd.DataFrame(rows, columns=['club', 'kickoff', 'scored', 'conceded'])


def test_strength_uses_only_games_before_the_deadline():
    t = pd.Timestamp
    g = log([(1, t('2024-08-10', tz='UTC'), 1, 1),
             (1, t('2024-08-17', tz='UTC'), 5, 0)])
    st = fx.strength(g, t('2024-08-15', tz='UTC'))
    assert st.at[1, 'scored'] == 1


def test_strength_uses_the_last_ten_games():
    t = pd.Timestamp('2024-08-01', tz='UTC')
    g = log([(1, t + pd.Timedelta(days=i), 9 if i == 0 else 1, 0)
             for i in range(11)])
    st = fx.strength(g, t + pd.Timedelta(days=30))
    assert st.at[1, 'scored'] == 1


def test_zscore_counts_each_fixture_side_once():
    """A club with many players in the window must not dominate the mean."""
    d = pd.Series([1.0, 1.0, 1.0, 3.0])
    keys = pd.Series([(10, 1), (10, 1), (10, 1), (11, 2)])
    z = fx._zscore(d, keys)
    assert z.iloc[0] == pytest.approx(-z.iloc[3])
