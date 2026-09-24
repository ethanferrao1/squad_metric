"""Rule tests for the form blend, on hand-made inputs."""
import numpy as np
import pandas as pd
import pytest

from form_lab import form


def hist(code, points, prices=None, gw0=1):
    """A gameweek history for one player, one row per gameweek."""
    n = len(points)
    prices = prices if prices is not None else [50] * n
    return pd.DataFrame({
        'code': code, 'gw': range(gw0, gw0 + n), 'points': points,
        'minutes': [90] * n, 'price': prices, 'transfers_in': [0] * n,
        'transfers_out': [0] * n, 'selected': [1000] * n, 'element_type': 3,
    })


def snap_row(**over):
    row = {'code': 1, 'element_type': 3, 'prior_ppg': 2.0, 'form_ppg': 2.0,
           'n_games': 10, 'price': 50, 'age': 28.0, 'no_pl_history': False,
           'price_change': 0, 'net_transfer_share': 0.0}
    row.update(over)
    return pd.DataFrame([row])


def test_weight_ramps_to_w_max_over_ten_games():
    assert form.weight(0) == 0
    assert form.weight(5) == pytest.approx(form.W_MAX / 2)
    assert form.weight(10) == pytest.approx(form.W_MAX)


def test_weight_caps_at_w_max():
    for n in (11, 20, 38):
        assert form.weight(n) == pytest.approx(form.W_MAX)


def test_early_gameweeks_stay_centred_on_the_prior():
    """Four games in, the prior still carries most of the weight."""
    assert form.weight(4) < 0.5


def test_form_never_uses_more_than_ten_games():
    """The 20 early hauls are outside the window and must not count."""
    points = [20] * 20 + [1] * 10
    assert form.form_ppg(points) == pytest.approx(1.0)


def test_form_counts_zeros_when_the_team_played():
    assert form.form_ppg([6, 0, 0, 0]) == pytest.approx(1.5)


def test_snapshot_counts_blanks_as_absent_not_zero():
    """A missing gameweek row is a blank, so it is not a zero in the mean."""
    h = hist(1, [6, 6])                      # gw 1 and 2 only
    h = pd.concat([h, hist(1, [6], gw0=5)])  # gw 5; 3 and 4 are blanks
    s = form.snapshot(h, 10, {1: 76.0})
    assert s['n_games'].iloc[0] == 3
    assert s['form_ppg'].iloc[0] == pytest.approx(6.0)


def test_snapshot_ignores_gameweeks_at_or_after_the_cutoff():
    s = form.snapshot(hist(1, [1, 1, 9, 9]), 3, {1: 76.0})
    assert s['n_games'].iloc[0] == 2
    assert s['form_ppg'].iloc[0] == pytest.approx(1.0)


def test_blend_moves_off_the_prior_with_the_weight():
    s = form.project(snap_row(prior_ppg=2.0, form_ppg=6.0, n_games=10),
                     w_max=0.5, breakout=False)
    assert s['blended_ppg'].iloc[0] == pytest.approx(0.5 * 2.0 + 0.5 * 6.0)


def test_blend_is_the_prior_with_no_games():
    s = form.project(snap_row(form_ppg=np.nan, n_games=0), breakout=False)
    assert s['blended_ppg'].iloc[0] == pytest.approx(2.0)


def test_breakout_triggers_for_a_cheap_player():
    s = form.project(snap_row(price=form.CHEAP_PRICE, prior_ppg=1.0,
                              form_ppg=5.0, n_games=2), breakout=True)
    assert bool(s['breakout'].iloc[0])
    assert s['form_weight'].iloc[0] == pytest.approx(form.W_MAX)


def test_breakout_does_not_trigger_for_an_expensive_player():
    """Same form, same ratio, only the price differs."""
    s = form.project(snap_row(price=form.CHEAP_PRICE + 1, prior_ppg=1.0,
                              form_ppg=5.0, n_games=2), breakout=True)
    assert not bool(s['breakout'].iloc[0])
    assert s['form_weight'].iloc[0] == pytest.approx(form.weight(2))


def test_breakout_needs_form_well_above_the_prior():
    below = form.BREAKOUT_RATIO * 2.0 - 0.1
    s = form.project(snap_row(price=40, prior_ppg=2.0, form_ppg=below,
                              n_games=2), breakout=True)
    assert not bool(s['breakout'].iloc[0])


def test_breakout_is_off_by_default():
    """The backtest did not justify it, so it must be opt-in."""
    s = form.project(snap_row(price=form.CHEAP_PRICE, prior_ppg=1.0,
                              form_ppg=5.0, n_games=2))
    assert not bool(s['breakout'].iloc[0])


def test_youth_boost_applies_to_under_24_with_no_pl_history():
    s = form.project(snap_row(age=21.0, no_pl_history=True, n_games=0,
                              form_ppg=np.nan))
    assert s['blended_ppg'].iloc[0] == pytest.approx(2.0 * (1 + form.YOUTH_BOOST))


def test_youth_boost_skips_an_older_player():
    s = form.project(snap_row(age=27.0, no_pl_history=True, n_games=0,
                              form_ppg=np.nan))
    assert s['blended_ppg'].iloc[0] == pytest.approx(2.0)


def test_youth_boost_skips_a_young_player_with_pl_history():
    s = form.project(snap_row(age=21.0, no_pl_history=False, n_games=0,
                              form_ppg=np.nan))
    assert s['blended_ppg'].iloc[0] == pytest.approx(2.0)


def test_market_weight_of_zero_leaves_the_blend_alone():
    kw = dict(price_change=10, net_transfer_share=0.5)
    off = form.project(snap_row(**kw), market_weight=0.0)
    on = form.project(snap_row(**kw), market_weight=0.2)
    assert off['blended_ppg'].iloc[0] != pytest.approx(on['blended_ppg'].iloc[0])
    assert off['blended_ppg'].iloc[0] == pytest.approx(2.0)


def test_market_mover_flag_is_independent_of_the_weight():
    s = form.project(snap_row(price_change=form.PRICE_MOVE), market_weight=0.0)
    assert bool(s['market_mover'].iloc[0])


def test_no_swap_below_the_margin():
    assert form.label(form.SWITCH_MARGIN - 0.01) is None


def test_swap_at_the_margin_is_a_hold():
    assert form.label(form.SWITCH_MARGIN) == 'hold'


def test_big_gain_is_strong():
    assert form.label(form.SWITCH_MARGIN * form.STRONG_MULTIPLE) == 'strong'


def test_gain_scales_with_gameweeks_remaining():
    assert form.gain_points(3.0, 2.0, 10) == pytest.approx(10.0)


def test_switch_margin_scales_with_the_horizon():
    """Per-gameweek error over the horizon, widened for a two-player difference."""
    assert form.switch_margin(0.5, 10) == pytest.approx(5.0 * np.sqrt(2))


def test_switch_margin_default_is_the_five_gameweek_horizon():
    assert form.SWITCH_MARGIN == pytest.approx(
        form.BLEND_MAE * form.SWAP_HORIZON * np.sqrt(2))


def test_market_mover_fires_on_a_price_move_alone():
    snap = pd.DataFrame({'net_transfers': [0, 0], 'price_change': [0, 3]})
    assert list(form.is_market_mover(snap)) == [False, True]


def test_market_mover_takes_only_the_busiest_by_transfers():
    """At the 98th percentile, only the top three of a hundred qualify."""
    snap = pd.DataFrame({'net_transfers': list(range(100)),
                         'price_change': [0] * 100})
    fired = form.is_market_mover(snap)
    assert fired.sum() == 3
    assert fired[-1] and not fired[0]


def test_market_mover_uses_absolute_transfers():
    """A big sell-off is a mover too, not just a big buy."""
    snap = pd.DataFrame({'net_transfers': [-500, 0, 1, 2],
                         'price_change': [0] * 4})
    assert form.is_market_mover(snap)[0]
