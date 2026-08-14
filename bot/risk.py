"""Position sizing and portfolio risk limits.

Sizing is driven by the stop, not by the price. The question this module
answers is never "how many shares can I afford" but "how many shares can I
hold such that being wrong costs a known, survivable amount".

Every limit here is a hard gate. If a check fails the trade does not happen;
none of them are advisory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from bot.config import Config


class SizingRejection(StrEnum):
    """Why a candidate could not be turned into an order."""

    INVALID_STOP = "invalid_stop"
    ZERO_SHARES = "zero_shares"
    BELOW_MIN_NOTIONAL = "below_min_notional"
    INSUFFICIENT_CASH = "insufficient_cash"
    POSITION_LIMIT_REACHED = "position_limit_reached"
    EXPOSURE_LIMIT_REACHED = "exposure_limit_reached"
    EXCEEDS_ADV_LIMIT = "exceeds_adv_limit"
    COMMISSION_TOO_HIGH = "commission_too_high"

    def explain(self) -> str:
        return {
            SizingRejection.INVALID_STOP: "stop price is not below the entry price",
            SizingRejection.ZERO_SHARES: "risk budget is too small for even one share",
            SizingRejection.BELOW_MIN_NOTIONAL: "position would be too small to justify commission",
            SizingRejection.INSUFFICIENT_CASH: "not enough settled cash",
            SizingRejection.POSITION_LIMIT_REACHED: "already holding the maximum number of positions",
            SizingRejection.EXPOSURE_LIMIT_REACHED: "gross exposure limit reached",
            SizingRejection.EXCEEDS_ADV_LIMIT: "position would be too large relative to average daily volume",
            SizingRejection.COMMISSION_TOO_HIGH: "commission would consume too much of the trade",
        }[self]


class HaltReason(StrEnum):
    DAILY_LOSS = "daily_loss"
    MAX_DRAWDOWN = "max_drawdown"
    STALE_DATA = "stale_data"


@dataclass(frozen=True)
class Sizing:
    """A sized, cost-aware order ready to be sent."""

    symbol: str
    shares: int
    entry_price: float
    stop_price: float

    @property
    def notional(self) -> float:
        return self.shares * self.entry_price

    @property
    def risk_dollars(self) -> float:
        """Loss if the stop fills exactly at its trigger price."""
        return self.shares * (self.entry_price - self.stop_price)

    def risk_pct_of(self, equity: float) -> float:
        return self.risk_dollars / equity * 100.0 if equity > 0 else 0.0


@dataclass(frozen=True)
class PortfolioState:
    """Everything the risk checks need to know about the account right now.

    Sourced from the broker on every run rather than from local state, so a
    manual trade or a partial fill cannot desynchronise the bot's view.
    """

    equity: float
    cash: float
    settled_cash: float
    gross_exposure: float
    open_position_count: int
    peak_equity: float
    equity_at_session_start: float

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity * 100.0)

    @property
    def session_pnl_pct(self) -> float:
        if self.equity_at_session_start <= 0:
            return 0.0
        return (
            (self.equity - self.equity_at_session_start)
            / self.equity_at_session_start
            * 100.0
        )

    @property
    def gross_exposure_pct(self) -> float:
        if self.equity <= 0:
            return 0.0
        return self.gross_exposure / self.equity * 100.0


def check_halts(
    state: PortfolioState, cfg: Config, data_staleness_days: int
) -> HaltReason | None:
    """Portfolio-wide stop conditions, checked before anything else runs.

    A halt blocks *new* positions. Existing positions keep their resting stop
    orders at the broker, because cancelling protection during a drawdown is
    the opposite of risk management.
    """
    cb = cfg.circuit_breakers

    if data_staleness_days > cb.max_data_staleness_days:
        return HaltReason.STALE_DATA

    if state.drawdown_pct >= cb.max_drawdown_halt_pct and not cb.manual_override:
        return HaltReason.MAX_DRAWDOWN

    if state.session_pnl_pct <= -cb.daily_loss_halt_pct:
        return HaltReason.DAILY_LOSS

    return None


def size_position(
    symbol: str,
    entry_price: float,
    stop_price: float,
    adv: float,
    state: PortfolioState,
    cfg: Config,
) -> Sizing | SizingRejection:
    """Convert a candidate into a share count, or explain why it cannot be.

    The share count is the minimum of five independent ceilings. Taking the
    minimum rather than the first binding constraint means adding a new limit
    can only ever make the position smaller.
    """
    risk, execution, universe = cfg.risk, cfg.execution, cfg.universe

    if entry_price <= 0 or stop_price <= 0 or stop_price >= entry_price:
        return SizingRejection.INVALID_STOP

    if state.open_position_count >= risk.max_concurrent_positions:
        return SizingRejection.POSITION_LIMIT_REACHED

    if state.gross_exposure_pct >= risk.max_gross_exposure_pct:
        return SizingRejection.EXPOSURE_LIMIT_REACHED

    # Ceiling 1: the risk budget. This is the primary sizing rule; the rest
    # only ever cut it down.
    risk_dollars = state.equity * risk.risk_per_trade_pct / 100.0
    stop_distance = entry_price - stop_price
    shares_by_risk = math.floor(risk_dollars / stop_distance)

    # Ceiling 2: concentration. Independent of the stop, so a very tight stop
    # cannot produce an outsized position.
    shares_by_concentration = math.floor(
        state.equity * risk.max_position_pct / 100.0 / entry_price
    )

    # Ceiling 3: available cash, and the remaining room under the gross
    # exposure cap, whichever binds first.
    exposure_headroom = max(
        0.0, state.equity * risk.max_gross_exposure_pct / 100.0 - state.gross_exposure
    )
    spendable = min(state.settled_cash, exposure_headroom)
    shares_by_cash = math.floor(spendable / entry_price)

    # Ceiling 4: market impact. Irrelevant at this account size for liquid
    # names, but it is what stops the bot if the universe is ever widened.
    shares_by_adv = (
        math.floor(adv * universe.max_pct_of_adv / 100.0 / entry_price)
        if adv > 0
        else shares_by_risk
    )

    # Ceiling 5: the absolute per-order backstop against a sizing bug.
    shares_by_order_cap = math.floor(execution.max_order_notional / entry_price)

    shares = min(
        shares_by_risk,
        shares_by_concentration,
        shares_by_cash,
        shares_by_adv,
        shares_by_order_cap,
    )

    if shares <= 0:
        if shares_by_cash <= 0:
            return SizingRejection.INSUFFICIENT_CASH
        if shares_by_adv <= 0:
            return SizingRejection.EXCEEDS_ADV_LIMIT
        return SizingRejection.ZERO_SHARES

    notional = shares * entry_price
    if notional < risk.min_position_notional:
        return SizingRejection.BELOW_MIN_NOTIONAL

    # A round trip pays commission twice. If that is a large share of the
    # position, the strategy's edge is spent before the trade begins.
    round_trip_commission = 2 * cfg.costs.commission(shares, entry_price)
    if round_trip_commission > notional * cfg.costs.commission_max_pct_of_trade / 100.0:
        return SizingRejection.COMMISSION_TOO_HIGH

    return Sizing(
        symbol=symbol,
        shares=shares,
        entry_price=round(entry_price, 2),
        stop_price=round(stop_price, 2),
    )


def marketable_limit_price(reference_price: float, side: str, cfg: Config) -> float:
    """Limit price set through the touch, so it fills like a market order.

    A true market order on a small account is how you discover what a bad
    print looks like. This crosses the spread by a controlled amount and no
    more, so a stale quote or a gap cannot produce an arbitrary fill.
    """
    offset = cfg.execution.marketable_limit_offset_pct / 100.0
    if side.upper() == "BUY":
        return round(reference_price * (1 + offset), 2)
    if side.upper() == "SELL":
        return round(reference_price * (1 - offset), 2)
    raise ValueError(f"side must be BUY or SELL, got {side!r}")


def slippage_exceeded(decision_price: float, fill_price: float, cfg: Config) -> bool:
    """Whether a fill landed further from the decision price than tolerated."""
    if decision_price <= 0:
        return True
    drift = abs(fill_price - decision_price) / decision_price * 100.0
    return drift > cfg.execution.max_slippage_pct
