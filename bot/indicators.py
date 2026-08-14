"""Technical indicators.

Pure functions over pandas Series. No I/O, no config, no state — which makes
them cheap to test and impossible to accidentally give a lookahead bias, as
long as callers respect the rule that a value at index ``t`` may only be acted
on at ``t+1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    Wilder's original formulation, not the simple-moving-average variant.
    They diverge meaningfully at the short periods this strategy uses, and
    every published parameterisation of a 2-4 period RSI assumes Wilder's.

    Returns values in [0, 100]; NaN until ``period`` deltas are available.
    """
    if period < 2:
        raise ValueError(f"RSI period must be at least 2, got {period}")

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # alpha = 1/period is Wilder's smoothing, equivalent to an EMA of span
    # 2*period-1. min_periods ensures no value is emitted before warmup.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # A window with no losses gives avg_loss == 0 and rs == inf; RSI is 100.
    # A window with no gains gives rs == 0 and RSI is 0, which falls out
    # naturally. Both must be handled explicitly because 0/0 is NaN.
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi.where(avg_gain.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's True Range: the widest of today's range and the overnight gaps."""
    prev_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """Average True Range with Wilder's smoothing.

    This is the volatility unit the position sizer works in: stops are placed
    a multiple of ATR away, so a volatile stock gets fewer shares for the same
    dollar risk. That is what makes risk comparable across the universe.
    """
    if period < 1:
        raise ValueError(f"ATR period must be at least 1, got {period}")
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average, emitting nothing until the window is full."""
    if window < 1:
        raise ValueError(f"SMA window must be at least 1, got {window}")
    return series.rolling(window=window, min_periods=window).mean()


def rate_of_change(close: pd.Series, periods: int) -> pd.Series:
    """Percentage change over ``periods`` bars, as a percentage."""
    if periods < 1:
        raise ValueError(f"ROC periods must be at least 1, got {periods}")
    return close.pct_change(periods=periods) * 100.0


def realised_volatility(close: pd.Series, window: int, annualise: bool = True) -> pd.Series:
    """Standard deviation of log returns over a rolling window.

    Used as the regime volatility gate. Log returns rather than simple returns
    so that the measure is symmetric in up and down moves.
    """
    if window < 2:
        raise ValueError(f"Volatility window must be at least 2, got {window}")
    log_returns = np.log(close / close.shift(1))
    vol = log_returns.rolling(window=window, min_periods=window).std()
    if annualise:
        vol = vol * np.sqrt(252.0)
    return vol * 100.0


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of each value within its own trailing window.

    Expressed in [0, 100]. Uses only past and present values, so it is safe to
    act on the value at ``t`` at ``t+1``.
    """
    if window < 2:
        raise ValueError(f"Percentile window must be at least 2, got {window}")

    def _rank(window_values: np.ndarray) -> float:
        current = window_values[-1]
        if np.isnan(current):
            return np.nan
        valid = window_values[~np.isnan(window_values)]
        if valid.size == 0:
            return np.nan
        return float((valid <= current).sum()) / valid.size * 100.0

    return series.rolling(window=window, min_periods=window // 2).apply(_rank, raw=True)


def dollar_volume(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Average daily traded value over a window, in currency units.

    The liquidity screen. A $1,400 account cannot move a market, but a wide
    spread on a thin name costs more than the strategy's entire edge.
    """
    return (close * volume).rolling(window=window, min_periods=max(2, window // 2)).mean()
