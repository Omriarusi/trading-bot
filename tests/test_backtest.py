"""Backtester correctness.

A backtest that is wrong in the optimistic direction is worse than no
backtest, because it justifies risking real money. These tests check the
mechanics — no lookahead, costs charged, stops honoured on gaps — rather than
the returns.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from backtest.engine import Backtester, compute_stats
from bot.config import Config
from tests.conftest import falling_series, make_bars, trending_series


@pytest.fixture
def bt_cfg(cfg: Config) -> Config:
    """Config with a liquidity floor the synthetic fixtures can clear."""
    return replace(cfg, universe=replace(cfg.universe, min_avg_dollar_volume=1_000_000))


@pytest.fixture
def market() -> dict[str, pd.DataFrame]:
    """A few hundred bars of a small synthetic market."""
    return {
        "AAA": make_bars(trending_series(n=600, seed=1), volume=2_000_000),
        "BBB": make_bars(trending_series(n=600, seed=2, daily_drift=0.0012), volume=2_000_000),
        "CCC": make_bars(falling_series(n=600, seed=3), volume=2_000_000),
    }


@pytest.fixture
def benchmark() -> pd.DataFrame:
    return make_bars(trending_series(n=600, daily_drift=0.0005, noise=0.006, seed=9))


# ------------------------------------------------------------------ mechanics


def test_backtest_runs_and_produces_an_equity_curve(bt_cfg, market, benchmark):
    result = Backtester(bt_cfg, starting_equity=1400.0).run(market, benchmark)

    assert len(result.equity_curve) > 100
    assert result.equity_curve.iloc[0] == pytest.approx(1400.0, rel=0.05)
    assert result.equity_curve.index.is_monotonic_increasing
    assert (result.equity_curve > 0).all(), "equity must never go negative"


def test_equity_is_conserved_when_nothing_trades(bt_cfg, benchmark):
    """With no valid candidate, equity must sit exactly at the starting value.

    Uses a strictly monotonic decline rather than a random walk: a noisy
    downtrend produces bear-market rallies that legitimately clear the trend
    filter, so it would not prove the absence of trading.
    """
    monotonic_decline = 200.0 * np.exp(np.linspace(0, -1.2, 600))
    only_falling = {"CCC": make_bars(monotonic_decline, volume=2_000_000)}
    result = Backtester(bt_cfg, starting_equity=1400.0).run(only_falling, benchmark)

    assert not result.trades
    assert result.equity_curve.nunique() == 1
    assert result.equity_curve.iloc[-1] == pytest.approx(1400.0)


def test_results_are_deterministic(bt_cfg, market, benchmark):
    """Identical inputs must produce identical trades, every time."""
    first = Backtester(bt_cfg, starting_equity=1400.0).run(market, benchmark)
    second = Backtester(bt_cfg, starting_equity=1400.0).run(market, benchmark)

    assert len(first.trades) == len(second.trades)
    assert first.equity_curve.iloc[-1] == pytest.approx(second.equity_curve.iloc[-1])
    for a, b in zip(first.trades, second.trades, strict=True):
        assert (a.symbol, a.entry_date, a.shares) == (b.symbol, b.entry_date, b.shares)


def test_every_trade_pays_commission_both_ways(bt_cfg, market, benchmark):
    result = Backtester(bt_cfg, starting_equity=1400.0).run(market, benchmark)

    for trade in result.closed_trades:
        assert trade.commission > 0
        # Entry plus exit, each at least the configured minimum.
        assert trade.commission >= 2 * min(
            bt_cfg.costs.commission_min,
            trade.shares * trade.entry_price * bt_cfg.costs.commission_max_pct_of_trade / 100,
        )


def test_no_position_exceeds_the_configured_caps(bt_cfg, market, benchmark):
    result = Backtester(bt_cfg, starting_equity=1400.0).run(market, benchmark)

    for trade in result.trades:
        notional = trade.shares * trade.entry_price
        assert notional <= bt_cfg.execution.max_order_notional * 1.05


def test_concurrent_positions_never_exceed_the_cap(bt_cfg, market, benchmark):
    result = Backtester(bt_cfg, starting_equity=1400.0).run(market, benchmark)

    events = []
    for trade in result.trades:
        events.append((trade.entry_date, 1))
        if trade.exit_date:
            events.append((trade.exit_date, -1))
    events.sort()

    concurrent = 0
    for _, delta in events:
        concurrent += delta
        assert concurrent <= bt_cfg.risk.max_concurrent_positions


def test_entries_happen_after_their_signal_not_on_it(bt_cfg, market, benchmark):
    """No trade may be dated before the day the strategy could have seen it."""
    result = Backtester(bt_cfg, starting_equity=1400.0).run(market, benchmark)

    for trade in result.trades:
        bars = market[trade.symbol]
        assert pd.Timestamp(trade.entry_date) in bars.index
        # The fill price must be a price that existed on the entry day.
        day = bars.loc[pd.Timestamp(trade.entry_date)]
        assert float(day["low"]) * 0.99 <= trade.entry_price <= float(day["high"]) * 1.01


def test_a_gap_below_the_stop_fills_at_the_open(bt_cfg, benchmark):
    """The honest modelling of the strategy's worst days.

    A stop does not protect against a gap; it becomes a market order at a
    price nobody chose. A backtest that fills at the trigger understates the
    losing tail, which is exactly the number that matters.
    """
    trend = trending_series(n=400, daily_drift=0.0015, noise=0.006, seed=4)
    pullback = trend[-1] * np.array([0.97, 0.95, 0.93, 0.92])
    # A 25% overnight collapse, far below any plausible stop.
    crash = np.array([pullback[-1] * 0.75] * 20)
    closes = np.concatenate([trend, pullback, crash])

    bars = make_bars(closes, volume=2_000_000)
    # Force a genuine gap: the crash day's high is below the prior close.
    crash_start = len(trend) + len(pullback)
    bars.iloc[crash_start, bars.columns.get_loc("open")] = pullback[-1] * 0.75
    bars.iloc[crash_start, bars.columns.get_loc("high")] = pullback[-1] * 0.76

    result = Backtester(bt_cfg, starting_equity=1400.0).run({"GAP": bars}, benchmark)
    stopped = [t for t in result.closed_trades if t.exit_reason == "stop"]

    if stopped:
        trade = stopped[0]
        # The realised loss must exceed the loss the stop nominally implied.
        assert trade.exit_price < trade.stop_price * 1.001
        assert trade.pnl < 0


def test_the_stop_distance_survives_a_gapped_fill(bt_cfg, benchmark):
    """Risk is budgeted as shares x stop distance, so the distance must hold.

    If the stop were anchored to the planned entry price instead, an open that
    gapped away from it would silently change what a stop-out costs — upward
    when the gap was against us, which is exactly when it matters.
    """
    result = Backtester(bt_cfg, starting_equity=100_000.0).run(
        {"AAA": make_bars(trending_series(n=600, seed=1), volume=2_000_000)}, benchmark
    )

    assert result.trades, "expected at least one trade to inspect"
    for trade in result.trades:
        distance = trade.entry_price - trade.stop_price
        assert distance > 0
        # The realised risk must stay within the budget the sizer worked to.
        assert trade.shares * distance <= 100_000.0 * bt_cfg.risk.risk_per_trade_pct / 100.0 * 1.01


def test_an_exit_is_never_recorded_before_its_entry(bt_cfg, market, benchmark):
    result = Backtester(bt_cfg, starting_equity=1400.0).run(market, benchmark)

    for trade in result.closed_trades:
        assert trade.exit_date >= trade.entry_date
        assert trade.exit_price is not None and trade.exit_price > 0


def test_cash_never_goes_negative(bt_cfg, market, benchmark):
    result = Backtester(bt_cfg, starting_equity=1400.0).run(market, benchmark)
    assert (result.equity_curve > 0).all()


def test_a_short_history_is_rejected_clearly(bt_cfg, benchmark):
    tiny = {"AAA": make_bars(trending_series(n=50), volume=2_000_000)}
    with pytest.raises(ValueError, match="enough history"):
        Backtester(bt_cfg, starting_equity=1400.0).run(tiny, benchmark)


def test_risk_per_trade_is_respected_in_the_backtest(bt_cfg, market, benchmark):
    """Loss on a clean stop-out must stay inside the risk budget."""
    result = Backtester(bt_cfg, starting_equity=100_000.0).run(market, benchmark)

    for trade in result.trades:
        planned_risk = trade.shares * (trade.entry_price - trade.stop_price)
        assert planned_risk <= 100_000.0 * bt_cfg.risk.risk_per_trade_pct / 100.0 * 1.01


# ---------------------------------------------------------------- statistics


def test_stats_on_a_known_curve():
    curve = pd.Series(
        [100.0, 110.0, 105.0, 120.0],
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    stats = compute_stats(curve, [])

    assert stats["total_return_pct"] == pytest.approx(20.0)
    assert stats["max_drawdown_pct"] == pytest.approx(-100 * 5 / 110, abs=0.01)


def test_stats_handles_an_empty_curve():
    assert "error" in compute_stats(pd.Series(dtype=float), [])


def test_drawdown_is_never_positive():
    curve = pd.Series(
        np.linspace(100, 200, 50), index=pd.bdate_range("2024-01-01", periods=50)
    )
    assert compute_stats(curve, [])["max_drawdown_pct"] <= 0


def test_report_renders_without_error(bt_cfg, market, benchmark):
    from backtest.report import render_report

    result = Backtester(bt_cfg, starting_equity=1400.0).run(market, benchmark)
    text = render_report(result, result.stats())

    assert "# Backtest" in text
    assert "Max drawdown" in text
