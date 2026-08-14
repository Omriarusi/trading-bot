"""Indicator correctness.

These are checked against hand-computed values rather than against another
library, so a silent behaviour change in pandas cannot quietly move the
strategy's entry threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot import indicators


def test_rsi_is_100_when_price_only_rises():
    close = pd.Series([10, 11, 12, 13, 14, 15, 16], dtype=float)
    rsi = indicators.wilder_rsi(close, period=3)
    assert rsi.iloc[-1] == pytest.approx(100.0)


def test_rsi_is_0_when_price_only_falls():
    close = pd.Series([16, 15, 14, 13, 12, 11, 10], dtype=float)
    rsi = indicators.wilder_rsi(close, period=3)
    assert rsi.iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_rsi_is_50_when_price_never_moves():
    close = pd.Series([10.0] * 10)
    rsi = indicators.wilder_rsi(close, period=3)
    # No gains and no losses is genuinely neutral, not undefined. Returning
    # NaN here would silently drop the symbol from every screen.
    assert rsi.iloc[-1] == pytest.approx(50.0)


def test_rsi_warms_up_before_emitting():
    close = pd.Series(np.arange(10, 30), dtype=float)
    rsi = indicators.wilder_rsi(close, period=5)
    assert rsi.iloc[:5].isna().all()
    assert rsi.iloc[5:].notna().all()


def test_rsi_stays_in_bounds_on_noisy_data():
    rng = np.random.default_rng(7)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 500))))
    rsi = indicators.wilder_rsi(close, period=3).dropna()
    assert rsi.between(0, 100).all()


def test_rsi_rejects_period_below_two():
    with pytest.raises(ValueError, match="at least 2"):
        indicators.wilder_rsi(pd.Series([1.0, 2.0]), period=1)


def test_true_range_uses_the_overnight_gap():
    # Day 2 gaps up from a close of 10 to a range of 14-13. The intraday range
    # is only 1, but the move from the prior close is 4 — that is the risk.
    high = pd.Series([11.0, 14.0])
    low = pd.Series([9.0, 13.0])
    close = pd.Series([10.0, 13.5])
    tr = indicators.true_range(high, low, close)
    assert tr.iloc[1] == pytest.approx(4.0)


def test_atr_of_a_constant_range_equals_that_range():
    n = 40
    high = pd.Series([102.0] * n)
    low = pd.Series([98.0] * n)
    close = pd.Series([100.0] * n)
    atr = indicators.atr(high, low, close, period=20)
    assert atr.iloc[-1] == pytest.approx(4.0, rel=1e-6)


def test_sma_emits_nothing_until_the_window_is_full():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = indicators.sma(series, window=3)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(3.0)


def test_rate_of_change_is_a_percentage():
    close = pd.Series([100.0, 110.0, 120.0])
    roc = indicators.rate_of_change(close, periods=2)
    assert roc.iloc[2] == pytest.approx(20.0)


def test_rolling_percentile_ranks_within_its_own_window():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = indicators.rolling_percentile(series, window=5)
    # The final value is the largest of the five, so it ranks at the top.
    assert result.iloc[-1] == pytest.approx(100.0)


def test_rolling_percentile_never_looks_ahead():
    """A value's rank must not change when future data is appended."""
    series = pd.Series([5.0, 3.0, 8.0, 1.0, 9.0, 2.0, 7.0, 4.0])
    partial = indicators.rolling_percentile(series.iloc[:6], window=4)
    full = indicators.rolling_percentile(series, window=4)
    pd.testing.assert_series_equal(partial, full.iloc[:6], check_names=False)


def test_realised_volatility_is_zero_for_a_flat_series():
    close = pd.Series([100.0] * 30)
    vol = indicators.realised_volatility(close, window=10)
    assert vol.iloc[-1] == pytest.approx(0.0, abs=1e-12)
