"""Position sizing and circuit breakers.

The most important tests in the repository. A strategy bug costs a trade; a
sizing bug costs the account.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from bot import risk
from bot.config import AccountType, Config
from bot.risk import HaltReason, PortfolioState, Sizing, SizingRejection


def portfolio(
    equity: float = 1400.0, cash: float | None = None, gross: float = 0.0,
    positions: int = 0, peak: float | None = None, session_start: float | None = None,
) -> PortfolioState:
    cash = equity if cash is None else cash
    return PortfolioState(
        equity=equity, cash=cash, settled_cash=cash, gross_exposure=gross,
        open_position_count=positions, peak_equity=peak or equity,
        equity_at_session_start=session_start or equity,
    )


# ------------------------------------------------------------------- sizing


@pytest.fixture
def unconstrained_cfg(cfg: Config) -> Config:
    """Config whose secondary ceilings are lifted out of the way.

    The sizer takes the minimum of five ceilings. To test that the *risk
    budget* is computed correctly, the other four must not bind — otherwise
    the test passes or fails for reasons unrelated to what it claims to check.
    Each of those other ceilings has its own test below, against the real
    config.
    """
    return replace(
        cfg,
        risk=replace(cfg.risk, max_position_pct=100.0, min_position_notional=0.0),
        execution=replace(cfg.execution, max_order_notional=1e9),
    )


def test_position_risk_matches_the_configured_budget(unconstrained_cfg: Config):
    """The whole point of the sizer: a stop-out costs the budgeted amount."""
    state = portfolio(equity=1400.0)
    sizing = risk.size_position("AAPL", entry_price=100.0, stop_price=95.0,
                                adv=1e9, state=state, cfg=unconstrained_cfg)

    assert isinstance(sizing, Sizing)
    # 2% of $1,400 is $28; a $5 stop distance allows 5 shares ($25 at risk).
    assert sizing.shares == 5
    assert sizing.risk_dollars == pytest.approx(25.0)
    assert sizing.risk_pct_of(1400.0) <= unconstrained_cfg.risk.risk_per_trade_pct


def test_a_wider_stop_produces_a_smaller_position(unconstrained_cfg: Config):
    """Volatility must reduce size, not dollar risk. This is the core idea."""
    state = portfolio(equity=10_000.0)
    tight = risk.size_position("A", 100.0, 98.0, 1e9, state, unconstrained_cfg)
    wide = risk.size_position("B", 100.0, 90.0, 1e9, state, unconstrained_cfg)

    assert isinstance(tight, Sizing) and isinstance(wide, Sizing)
    # A stop five times wider must give roughly a fifth of the shares.
    assert wide.shares == pytest.approx(tight.shares / 5, rel=0.1)
    # Both risk the same amount despite very different position sizes.
    assert tight.risk_dollars == pytest.approx(wide.risk_dollars, rel=0.05)


def test_every_ceiling_binds_at_the_real_account_size(cfg: Config):
    """At $1,400 the concentration cap is the binding constraint, by design.

    Documents the actual behaviour: a 5% stop on a $100 stock is limited to
    30% of equity, not to the 2% risk budget, which would allow more.
    """
    state = portfolio(equity=1400.0)
    sizing = risk.size_position("AAPL", 100.0, 95.0, 1e9, state, cfg)

    assert isinstance(sizing, Sizing)
    assert sizing.shares == 4  # floor($1,400 * 30% / $100)
    assert sizing.notional <= 1400.0 * cfg.risk.max_position_pct / 100.0
    assert sizing.risk_dollars < 1400.0 * cfg.risk.risk_per_trade_pct / 100.0


def test_concentration_cap_overrides_a_very_tight_stop(cfg: Config):
    """A tight stop must not be a licence to buy the whole account."""
    state = portfolio(equity=10_000.0)
    sizing = risk.size_position("AAPL", 100.0, 99.9, 1e9, state, cfg)

    assert isinstance(sizing, Sizing)
    max_notional = 10_000.0 * cfg.risk.max_position_pct / 100.0
    assert sizing.notional <= max_notional


def test_sizing_never_spends_more_than_settled_cash(cfg: Config):
    state = portfolio(equity=1400.0, cash=300.0, gross=1100.0)
    sizing = risk.size_position("AAPL", 100.0, 90.0, 1e9, state, cfg)

    if isinstance(sizing, Sizing):
        assert sizing.notional <= 300.0
    else:
        assert sizing in (SizingRejection.INSUFFICIENT_CASH, SizingRejection.BELOW_MIN_NOTIONAL)


def test_order_notional_cap_is_a_hard_backstop(cfg: Config):
    """Even with a huge account, no single order exceeds the configured cap."""
    state = portfolio(equity=1_000_000.0)
    sizing = risk.size_position("AAPL", 100.0, 95.0, 1e12, state, cfg)

    assert isinstance(sizing, Sizing)
    assert sizing.notional <= cfg.execution.max_order_notional


def test_stop_at_or_above_entry_is_rejected(cfg: Config):
    state = portfolio()
    assert risk.size_position("A", 100.0, 100.0, 1e9, state, cfg) is SizingRejection.INVALID_STOP
    assert risk.size_position("A", 100.0, 105.0, 1e9, state, cfg) is SizingRejection.INVALID_STOP
    assert risk.size_position("A", 100.0, 0.0, 1e9, state, cfg) is SizingRejection.INVALID_STOP


def test_position_limit_blocks_further_entries(cfg: Config):
    state = portfolio(positions=cfg.risk.max_concurrent_positions)
    result = risk.size_position("A", 100.0, 95.0, 1e9, state, cfg)
    assert result is SizingRejection.POSITION_LIMIT_REACHED


def test_tiny_positions_are_rejected_rather_than_traded(cfg: Config):
    """Below the minimum notional, commission eats the edge."""
    state = portfolio(equity=200.0)
    result = risk.size_position("A", 50.0, 45.0, 1e9, state, cfg)
    assert result in (SizingRejection.BELOW_MIN_NOTIONAL, SizingRejection.ZERO_SHARES)


def test_illiquid_names_cannot_be_bought(cfg: Config):
    """A $2,000/day stock cannot absorb even a small order."""
    state = portfolio(equity=100_000.0)
    result = risk.size_position("THIN", 100.0, 95.0, adv=2000.0, state=state, cfg=cfg)
    assert result in (SizingRejection.EXCEEDS_ADV_LIMIT, SizingRejection.BELOW_MIN_NOTIONAL)


def test_exposure_headroom_limits_the_last_position(cfg: Config):
    """Sizing must respect the gross exposure cap, not just the cash balance."""
    state = portfolio(equity=1400.0, cash=1400.0, gross=1300.0)
    sizing = risk.size_position("A", 20.0, 18.0, 1e9, state, cfg)

    headroom = 1400.0 * cfg.risk.max_gross_exposure_pct / 100.0 - 1300.0
    if isinstance(sizing, Sizing):
        assert sizing.notional <= headroom + 1e-9


@pytest.mark.parametrize("risk_pct", [0.5, 1.0, 2.0, 3.0, 5.0])
def test_risk_budget_is_honoured_across_settings(unconstrained_cfg: Config, risk_pct: float):
    cfg = replace(unconstrained_cfg, risk=replace(unconstrained_cfg.risk, risk_per_trade_pct=risk_pct))
    state = portfolio(equity=50_000.0)
    sizing = risk.size_position("A", 100.0, 90.0, 1e9, state, cfg)

    assert isinstance(sizing, Sizing)
    assert sizing.risk_pct_of(50_000.0) <= risk_pct + 1e-9


def test_the_sizer_never_exceeds_any_ceiling(unconstrained_cfg: Config):
    """Property check across a wide grid of prices, stops, and equities.

    Guards the invariant that matters: whatever the inputs, the realised risk
    stays inside the budget and the position inside every cap.
    """
    cfg = replace(
        unconstrained_cfg,
        risk=replace(unconstrained_cfg.risk, max_position_pct=30.0),
        execution=replace(unconstrained_cfg.execution, max_order_notional=500.0),
    )
    for equity in (500.0, 1400.0, 5000.0, 50_000.0):
        for price in (5.0, 47.3, 100.0, 380.0):
            for stop_pct in (0.5, 2.0, 5.0, 15.0):
                stop = price * (1 - stop_pct / 100.0)
                sizing = risk.size_position("X", price, stop, 1e12, portfolio(equity), cfg)
                if not isinstance(sizing, Sizing):
                    continue
                assert sizing.risk_dollars <= equity * cfg.risk.risk_per_trade_pct / 100.0 + 1e-6
                assert sizing.notional <= equity * cfg.risk.max_position_pct / 100.0 + 1e-6
                assert sizing.notional <= cfg.execution.max_order_notional + 1e-6
                assert sizing.shares >= 1


# --------------------------------------------------------- circuit breakers


def test_drawdown_halt_trips_at_the_threshold(cfg: Config):
    state = portfolio(equity=1000.0, peak=1400.0)  # -28.6%
    assert risk.check_halts(state, cfg, 0) is HaltReason.MAX_DRAWDOWN


def test_manual_override_clears_the_drawdown_halt(cfg: Config):
    cfg = replace(cfg, circuit_breakers=replace(cfg.circuit_breakers, manual_override=True))
    state = portfolio(equity=1000.0, peak=1400.0)
    assert risk.check_halts(state, cfg, 0) is not HaltReason.MAX_DRAWDOWN


def test_daily_loss_halt_trips_on_a_bad_session(cfg: Config):
    state = portfolio(equity=1300.0, peak=1400.0, session_start=1400.0)  # -7.1%
    assert risk.check_halts(state, cfg, 0) is HaltReason.DAILY_LOSS


def test_stale_data_halts_trading(cfg: Config):
    state = portfolio()
    assert risk.check_halts(state, cfg, data_staleness_days=10) is HaltReason.STALE_DATA
    assert risk.check_halts(state, cfg, data_staleness_days=3) is None


def test_a_healthy_account_is_not_halted(cfg: Config):
    assert risk.check_halts(portfolio(equity=1500.0, peak=1500.0), cfg, 1) is None


def test_drawdown_takes_precedence_over_daily_loss(cfg: Config):
    """The more severe breaker must be the one reported."""
    state = portfolio(equity=900.0, peak=1400.0, session_start=1000.0)
    assert risk.check_halts(state, cfg, 0) is HaltReason.MAX_DRAWDOWN


# ----------------------------------------------------------------- pricing


def test_marketable_limits_cross_the_spread_in_the_right_direction(cfg: Config):
    assert risk.marketable_limit_price(100.0, "BUY", cfg) > 100.0
    assert risk.marketable_limit_price(100.0, "SELL", cfg) < 100.0


def test_marketable_limit_offset_is_bounded(cfg: Config):
    buy = risk.marketable_limit_price(100.0, "BUY", cfg)
    assert buy == pytest.approx(100.0 * (1 + cfg.execution.marketable_limit_offset_pct / 100), abs=0.01)


def test_marketable_limit_rejects_an_unknown_side(cfg: Config):
    with pytest.raises(ValueError, match="BUY or SELL"):
        risk.marketable_limit_price(100.0, "SHORT", cfg)


def test_slippage_guard_flags_a_bad_fill(cfg: Config):
    assert not risk.slippage_exceeded(100.0, 100.5, cfg)
    assert risk.slippage_exceeded(100.0, 103.0, cfg)


# -------------------------------------------------------------- commission


def test_commission_respects_the_stated_minimum(cfg: Config):
    # 5 shares at $0.0035 is under two cents, so the $0.35 minimum applies —
    # unless the percentage cap binds first, which it does on small trades.
    assert cfg.costs.commission(5, 100.0) == pytest.approx(0.35)


def test_commission_is_capped_as_a_share_of_a_small_trade(cfg: Config):
    """IBKR charges the lower of the minimum and 1% of trade value."""
    commission = cfg.costs.commission(1, 10.0)
    assert commission <= 10.0 * cfg.costs.commission_max_pct_of_trade / 100.0


def test_no_commission_on_an_empty_order(cfg: Config):
    assert cfg.costs.commission(0, 100.0) == 0.0


def test_margin_only_features_are_off_in_a_cash_account(cfg: Config):
    assert cfg.account.type is AccountType.CASH
    assert not cfg.account.can_short
