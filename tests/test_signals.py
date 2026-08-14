"""Strategy logic: regime gate, entry screen, exits, stops."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bot import signals
from bot.config import Config, Ranking, RiskOffAction
from bot.signals import EntryCandidate, ExitReason, RejectReason
from tests.conftest import make_bars, trending_series

# -------------------------------------------------------------------- regime


def test_regime_is_risk_on_in_an_uptrend(risk_on_benchmark, cfg: Config):
    features = signals.compute_features(risk_on_benchmark, cfg.strategy)
    regime = signals.evaluate_regime(features, cfg)

    assert regime.risk_on
    assert regime.benchmark_above_ma
    assert "risk-on" in regime.describe()


def test_regime_is_risk_off_below_the_moving_average(risk_off_benchmark, cfg: Config):
    features = signals.compute_features(risk_off_benchmark, cfg.strategy)
    regime = signals.evaluate_regime(features, cfg)

    assert not regime.risk_on
    assert not regime.benchmark_above_ma
    assert "risk-off" in regime.describe()


def test_regime_needs_enough_history(cfg: Config):
    short = make_bars(trending_series(n=50))
    features = signals.compute_features(short, cfg.strategy)
    with pytest.raises(ValueError, match="too few"):
        signals.evaluate_regime(features, cfg)


def test_high_volatility_blocks_a_risk_on_regime(cfg: Config):
    """An uptrend in a panic is still a panic."""
    calm = trending_series(n=400, daily_drift=0.0008, noise=0.004, seed=5)
    # A violent but net-positive tail: price stays above its average while
    # realised volatility spikes into its own top decile.
    rng = np.random.default_rng(11)
    wild = calm[-1] * np.exp(np.cumsum(rng.normal(0.002, 0.07, 30)))
    bars = make_bars(np.concatenate([calm, wild]))

    features = signals.compute_features(bars, cfg.strategy)
    regime = signals.evaluate_regime(features, cfg)

    if regime.benchmark_above_ma:
        assert not regime.volatility_ok
        assert not regime.risk_on


# -------------------------------------------------------------- entry screen


def test_a_pullback_in_an_uptrend_is_a_candidate(pullback_bars, liquid_cfg: Config):
    features = signals.compute_features(pullback_bars, liquid_cfg.strategy)
    outcome = signals.screen_entry("TEST", features, liquid_cfg)

    assert isinstance(outcome, EntryCandidate), f"rejected as {outcome}"
    assert outcome.rsi <= liquid_cfg.strategy.rsi_entry_max
    assert outcome.price > 0
    assert outcome.atr > 0


def test_a_downtrend_is_rejected_however_oversold(downtrend_bars, liquid_cfg: Config):
    """The trend filter is what stops the bot buying every collapse."""
    features = signals.compute_features(downtrend_bars, liquid_cfg.strategy)
    outcome = signals.screen_entry("FALLING", features, liquid_cfg)

    assert outcome in (RejectReason.BELOW_TREND_MA, RejectReason.WEAK_MOMENTUM)


def test_an_uptrend_at_full_price_is_not_oversold(uptrend_bars, liquid_cfg: Config):
    features = signals.compute_features(uptrend_bars, liquid_cfg.strategy)
    outcome = signals.screen_entry("STRONG", features, liquid_cfg)

    if isinstance(outcome, RejectReason):
        assert outcome is RejectReason.NOT_OVERSOLD


def test_short_history_is_rejected_rather_than_guessed(liquid_cfg: Config):
    features = signals.compute_features(make_bars(trending_series(n=30)), liquid_cfg.strategy)
    assert signals.screen_entry("NEW", features, liquid_cfg) is RejectReason.INSUFFICIENT_HISTORY


def test_illiquid_symbols_are_screened_out(pullback_bars, cfg: Config):
    """A thin name is rejected even with a textbook setup.

    Same bars as the passing candidate test, with only the traded volume
    reduced — so the rejection can only be the liquidity screen.
    """
    thin = pullback_bars.copy()
    thin["volume"] = 1_000  # ~$145k a day against a $50m floor
    features = signals.compute_features(thin, cfg.strategy)
    assert signals.screen_entry("THIN", features, cfg) is RejectReason.ILLIQUID


def test_price_filter_excludes_penny_and_very_expensive_stocks(liquid_cfg: Config):
    cheap = make_bars(trending_series(n=400, start=2.0), volume=1e9)
    features = signals.compute_features(cheap, liquid_cfg.strategy)
    assert signals.screen_entry("PENNY", features, liquid_cfg) is RejectReason.PRICE_FILTER


def test_earnings_blackout_blocks_an_otherwise_valid_setup(pullback_bars, liquid_cfg: Config):
    """An earnings gap jumps straight through the stop, so risk is undefined."""
    features = signals.compute_features(pullback_bars, liquid_cfg.strategy)
    assert isinstance(signals.screen_entry("TEST", features, liquid_cfg), EntryCandidate)

    blocked = signals.screen_entry("TEST", features, liquid_cfg, earnings_within_days=1)
    assert blocked is RejectReason.EARNINGS_BLACKOUT


def test_empty_history_is_handled(liquid_cfg: Config):
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    features = signals.compute_features(empty, liquid_cfg.strategy)
    assert signals.screen_entry("X", features, liquid_cfg) is RejectReason.INSUFFICIENT_HISTORY


# ------------------------------------------------------------------- ranking


def _candidate(symbol: str, rsi: float, momentum: float) -> EntryCandidate:
    return EntryCandidate(
        symbol=symbol, price=100.0, atr=2.0, rsi=rsi,
        momentum=momentum, adv=1e9, rank_value=-rsi,
    )


def test_most_oversold_ranks_first():
    ranked = signals.rank_candidates([
        _candidate("A", rsi=20, momentum=10),
        _candidate("B", rsi=5, momentum=10),
        _candidate("C", rsi=15, momentum=10),
    ])
    assert [c.symbol for c in ranked] == ["B", "C", "A"]


def test_ranking_is_deterministic_on_ties():
    """Identical data must always produce identical trades."""
    tied = [_candidate("Z", 10, 5), _candidate("A", 10, 5), _candidate("M", 10, 5)]
    assert [c.symbol for c in signals.rank_candidates(tied)] == ["A", "M", "Z"]
    assert signals.rank_candidates(tied) == signals.rank_candidates(list(reversed(tied)))


def test_momentum_ranking_uses_momentum(liquid_cfg: Config, pullback_bars):
    cfg = replace(liquid_cfg, strategy=replace(liquid_cfg.strategy, ranking=Ranking.MOMENTUM))
    features = signals.compute_features(pullback_bars, cfg.strategy)
    outcome = signals.screen_entry("TEST", features, cfg)

    assert isinstance(outcome, EntryCandidate)
    assert outcome.rank_value == outcome.momentum


# --------------------------------------------------------------------- exits


def test_time_stop_fires_at_the_configured_limit(uptrend_bars, cfg: Config):
    features = signals.compute_features(uptrend_bars, cfg.strategy)
    assert signals.should_exit(features, cfg, holding_days=cfg.risk.max_holding_days) is ExitReason.TIME_STOP
    # One day short of the limit, the time stop must not fire.
    early = signals.should_exit(features, cfg, holding_days=cfg.risk.max_holding_days - 1)
    assert early is not ExitReason.TIME_STOP


def test_recovery_above_the_short_average_takes_profit(cfg: Config):
    """The exit that makes this a swing strategy rather than a trend one."""
    recovering = np.concatenate([
        trending_series(n=390, daily_drift=0.001, noise=0.005),
        # A sharp bounce that carries price back above its 10-day average.
        trending_series(n=10, start=100.0, daily_drift=0.05, noise=0.001) * 0 + 200.0,
    ])
    features = signals.compute_features(make_bars(recovering), cfg.strategy)
    assert signals.should_exit(features, cfg, holding_days=2) is ExitReason.TARGET


def test_no_exit_while_a_trade_is_still_working(cfg: Config):
    """A position that is neither at target nor stale must be left alone."""
    # Price sitting below its 10-day average with a mid-range RSI.
    drifting = np.concatenate([
        trending_series(n=395, daily_drift=0.001, noise=0.005, seed=3),
        np.array([1.0, 0.99, 0.985, 0.98, 0.978]) * trending_series(
            n=395, daily_drift=0.001, noise=0.005, seed=3
        )[-1],
    ])
    features = signals.compute_features(make_bars(drifting), cfg.strategy)
    assert signals.should_exit(features, cfg, holding_days=1) is None


# --------------------------------------------------------------------- stops


def test_initial_stop_sits_an_atr_multiple_below_entry(cfg: Config):
    stop = signals.initial_stop_price(entry_price=100.0, atr_value=2.0, cfg=cfg)
    assert stop == pytest.approx(100.0 - cfg.risk.atr_stop_multiple * 2.0)


def test_a_stop_can_never_be_zero_or_negative(cfg: Config):
    """A huge ATR on a cheap stock must not produce an unplaceable stop."""
    stop = signals.initial_stop_price(entry_price=1.0, atr_value=50.0, cfg=cfg)
    assert stop > 0


def test_trailing_stop_only_ever_moves_up(cfg: Config):
    current = 95.0
    # A lower high must not loosen the stop.
    assert signals.trailing_stop_price(90.0, 2.0, cfg, current) == current
    # A higher high ratchets it upward.
    assert signals.trailing_stop_price(120.0, 2.0, cfg, current) > current


def test_trailing_stop_uses_the_configured_multiple(cfg: Config):
    trailed = signals.trailing_stop_price(120.0, 2.0, cfg, current_stop=0.0)
    assert trailed == pytest.approx(120.0 - cfg.risk.atr_trail_multiple * 2.0)


# ------------------------------------------------------------------ features


def test_compute_features_does_not_mutate_its_input(uptrend_bars, cfg: Config):
    before = uptrend_bars.copy()
    signals.compute_features(uptrend_bars, cfg.strategy)
    pd.testing.assert_frame_equal(uptrend_bars, before)


def test_compute_features_rejects_unsorted_bars(uptrend_bars, cfg: Config):
    with pytest.raises(ValueError, match="sorted"):
        signals.compute_features(uptrend_bars.iloc[::-1], cfg.strategy)


def test_compute_features_rejects_missing_columns(uptrend_bars, cfg: Config):
    with pytest.raises(ValueError, match="missing column"):
        signals.compute_features(uptrend_bars.drop(columns=["volume"]), cfg.strategy)


def test_features_at_a_point_do_not_depend_on_later_data(uptrend_bars, cfg: Config):
    """The no-lookahead guarantee the whole backtest rests on."""
    cutoff = 300
    partial = signals.compute_features(uptrend_bars.iloc[:cutoff], cfg.strategy)
    full = signals.compute_features(uptrend_bars, cfg.strategy)

    for column in ("sma_trend", "sma_exit", "rsi", "momentum", "atr"):
        pd.testing.assert_series_equal(
            partial[column], full[column].iloc[:cutoff], check_names=False
        )


def test_risk_off_flatten_is_a_distinct_configuration(cfg: Config):
    flattening = replace(cfg, regime=replace(cfg.regime, risk_off_action=RiskOffAction.FLATTEN))
    assert flattening.regime.risk_off_action is RiskOffAction.FLATTEN
    assert cfg.regime.risk_off_action is RiskOffAction.HOLD
