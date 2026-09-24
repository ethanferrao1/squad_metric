"""Tests for the real-manager benchmark, on hand-made samples."""
import numpy as np
import pandas as pd
import pytest

from form_lab import managers as mg


def sample(totals, ranks=None):
    d = pd.DataFrame({'entry_id': range(len(totals)), 'total_points': totals,
                      'rank': ranks if ranks is not None else
                      list(range(len(totals), 0, -1))})
    d['rank_percentage'] = d['rank'].rank(pct=True) * 100
    return d.sort_values('total_points').reset_index(drop=True)


def test_summary_reports_the_quartiles():
    s = mg.summary(sample([100, 200, 300, 400]))
    assert s['n'] == 4
    assert s['median'] == 250
    assert s['p25'] == 175
    assert s['p75'] == 325


def test_top_pct_of_the_best_score_is_small():
    """Beating everyone means finishing in the top 0%."""
    assert mg.top_pct(sample([100, 200, 300, 400]), 500) == 0.0


def test_top_pct_of_the_worst_score_is_everyone():
    assert mg.top_pct(sample([100, 200, 300, 400]), 50) == 100.0


def test_top_pct_at_the_median():
    assert mg.top_pct(sample([100, 200, 300, 400]), 300) == 50.0


def test_estimated_rank_interpolates_between_managers():
    s = sample([100, 300], ranks=[2000, 1000])
    assert mg.estimated_rank(s, 200) == pytest.approx(1500)


def test_estimated_rank_clamps_beyond_the_sample():
    s = sample([100, 300], ranks=[2000, 1000])
    assert mg.estimated_rank(s, 10_000) == 1000
    assert mg.estimated_rank(s, 0) == 2000


def test_estimated_rank_needs_two_points():
    assert np.isnan(mg.estimated_rank(sample([100]), 100))


def test_past_rows_keeps_only_the_requested_seasons():
    hist = {'past': [{'season_name': '2023/24', 'total_points': 1, 'rank': 1},
                     {'season_name': '2024/25', 'total_points': 2100,
                      'rank': 500}]}
    out = mg._past_rows(hist, ('24_25', '25_26'))
    assert out == {'24_25': (2100, 500)}


def test_past_rows_of_a_missing_entry_is_empty():
    assert mg._past_rows(None, ('24_25',)) == {}


def test_past_rows_handles_an_entry_with_no_history():
    assert mg._past_rows({'past': []}, ('24_25',)) == {}
