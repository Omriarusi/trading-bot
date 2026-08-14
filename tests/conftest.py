"""Shared fixtures: synthetic price data and a default config.

Synthetic rather than recorded data, so the whole suite runs with no network
and no vendored files, and so a test can construct exactly the market
condition it wants to check.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bot.config import Config


@pytest.fixture
def cfg() -> Config:
    """Default configuration, matching the committed defaults."""
    return Config()


def make_bars(
    closes: list[float] | np.ndarray,
    start: str = "2020-01-01",
    volume: float = 5_000_000,
    range_pct: float = 0.02,
) -> pd.DataFrame:
    """Build an OHLCV frame from a close series.

    Highs and lows are placed symmetrically around the close so the ATR is a
    predictable fraction of price, which keeps sizing assertions readable.
    """
    closes = np.asarray(closes, dtype=float)
    index = pd.bdate_range(start=start, periods=len(closes))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(closes, opens) * (1 + range_pct / 2),
            "low": np.minimum(closes, opens) * (1 - range_pct / 2),
            "close": closes,
            "volume": np.full(len(closes), volume),
        },
        index=index,
    )


def trending_series(n: int = 400, start: float = 100.0, daily_drift: float = 0.0008,
                    noise: float = 0.01, seed: int = 42) -> np.ndarray:
    """A geometric random walk with positive drift."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(daily_drift, noise, n)
    return start * np.exp(np.cumsum(returns))


def falling_series(n: int = 400, start: float = 100.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(-0.0015, 0.012, n)
    return start * np.exp(np.cumsum(returns))


@pytest.fixture
def uptrend_bars() -> pd.DataFrame:
    """400 bars in a steady uptrend — a valid backdrop for entries."""
    return make_bars(trending_series())


@pytest.fixture
def downtrend_bars() -> pd.DataFrame:
    return make_bars(falling_series())


@pytest.fixture
def pullback_bars() -> pd.DataFrame:
    """An uptrend ending in a sharp multi-day drop.

    This is the exact setup the strategy is built to buy: price well above its
    long-term average, but short-term oversold.
    """
    trend = trending_series(n=380, daily_drift=0.0012, noise=0.008)
    # Five consecutive down days, deep enough to push a 3-period RSI far below
    # its entry threshold while leaving price above the 200-day average.
    drop = trend[-1] * np.array([0.97, 0.945, 0.925, 0.91, 0.90])
    return make_bars(np.concatenate([trend, drop]))


@pytest.fixture
def risk_on_benchmark() -> pd.DataFrame:
    """A benchmark comfortably above its 200-day average."""
    return make_bars(trending_series(n=500, daily_drift=0.0006, noise=0.007, seed=1))


@pytest.fixture
def risk_off_benchmark() -> pd.DataFrame:
    """A benchmark that has fallen below its 200-day average."""
    up = trending_series(n=300, daily_drift=0.0010, noise=0.007, seed=2)
    down = up[-1] * np.exp(np.cumsum(np.full(200, -0.004)))
    return make_bars(np.concatenate([up, down]))


@pytest.fixture
def liquid_cfg(cfg: Config) -> Config:
    """Config whose liquidity floor suits the synthetic fixtures."""
    return replace(cfg, universe=replace(cfg.universe, min_avg_dollar_volume=1_000_000))
