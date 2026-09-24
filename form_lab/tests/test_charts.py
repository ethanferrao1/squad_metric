"""The points chart copes with blanks, zero appearances and comparisons."""
import numpy as np
import pandas as pd

from ui import charts


def log(points, minutes=None):
    n = len(points)
    return pd.DataFrame({'gw': range(1, n + 1), 'points': points,
                         'minutes': minutes if minutes is not None else [90] * n,
                         'opp': ['ARS (H)'] * n})


def test_no_appearances_at_all():
    fig = charts.points_chart(log([np.nan] * 5, [0] * 5), None, 'New')
    assert not [t for t in fig.data if len(t.x or [])]
    assert fig.layout.annotations[0].text == 'No appearances yet'


def test_an_empty_history():
    fig = charts.points_chart(pd.DataFrame(columns=['gw', 'points', 'minutes', 'opp']))
    assert fig.layout.annotations


def test_zero_minute_games_plot_as_zero_not_gaps():
    fig = charts.points_chart(log([0, 0, 0], [0, 0, 0]), 2.0, 'Bench')
    assert list(fig.data[0].y) == [0, 0, 0]


def test_a_blank_gameweek_leaves_a_gap():
    fig = charts.points_chart(log([6, np.nan, 2]), 3.0, 'Saka')
    points = fig.data[0]
    assert np.isnan(points.y[1]) and points.connectgaps is False


def test_average_and_projection_are_drawn_dashed_and_dotted():
    fig = charts.points_chart(log([6, 2, 8]), 4.5, 'Saka')
    dashes = [t.line.dash for t in fig.data]
    assert 'dash' in dashes and 'dot' in dashes


def test_compare_overlays_a_second_player():
    fig = charts.points_chart(log([6, 2]), 4.0, 'Out',
                              compare=(log([1, 9]), 5.0, 'In'))
    assert {t.name for t in fig.data} >= {'Out', 'In'}
    assert fig.layout.showlegend
