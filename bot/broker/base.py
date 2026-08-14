"""Broker interface.

The engine talks only to this abstraction, which is what lets the identical
decision code run against a simulator, a paper account, and a live account.
The concrete implementations live alongside: ``paper.py`` and ``ibkr.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    LIMIT = "LMT"
    MARKET = "MKT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"


class OrderStatus(StrEnum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)

    @property
    def is_working(self) -> bool:
        return self in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        )


class BrokerError(RuntimeError):
    """A broker operation failed. Never raised for a merely rejected order."""


class OrderRejected(BrokerError):
    """The broker refused the order outright."""


@dataclass(frozen=True)
class AccountSummary:
    """Account-level figures, as reported by the broker."""

    account_id: str
    equity: float
    cash: float
    settled_cash: float
    buying_power: float
    currency: str = "USD"
    is_margin: bool = False

    @property
    def account_type(self) -> str:
        return "margin" if self.is_margin else "cash"


@dataclass(frozen=True)
class Position:
    symbol: str
    shares: int
    average_cost: float
    market_price: float
    conid: int | None = None

    @property
    def market_value(self) -> float:
        return self.shares * self.market_price

    @property
    def unrealised_pnl(self) -> float:
        return self.shares * (self.market_price - self.average_cost)

    @property
    def unrealised_pnl_pct(self) -> float:
        if self.average_cost <= 0:
            return 0.0
        return (self.market_price - self.average_cost) / self.average_cost * 100.0


@dataclass(frozen=True)
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    shares: int
    status: OrderStatus
    limit_price: float | None = None
    stop_price: float | None = None
    filled_shares: int = 0
    average_fill_price: float | None = None
    conid: int | None = None

    @property
    def is_protective_stop(self) -> bool:
        """A resting sell stop, i.e. the thing guarding an open position."""
        return self.side is OrderSide.SELL and self.order_type in (
            OrderType.STOP,
            OrderType.STOP_LIMIT,
        )


@dataclass(frozen=True)
class OrderResult:
    """Outcome of a submission attempt."""

    accepted: bool
    order_id: str | None = None
    stop_order_id: str | None = None
    message: str = ""

    @classmethod
    def rejected(cls, message: str) -> OrderResult:
        return cls(accepted=False, message=message)


@dataclass
class TradeRecord:
    """A completed or in-flight trade, for the run log."""

    symbol: str
    side: OrderSide
    shares: int
    price: float
    timestamp: datetime
    reason: str = ""
    commission: float = 0.0
    order_id: str | None = None


@dataclass
class SettlementLot:
    """Cash from a sale, and the date it becomes usable.

    Only meaningful for cash accounts. Spending unsettled proceeds is a
    good-faith violation, and repeated violations get the account restricted
    to settled-cash-only trading for 90 days.
    """

    amount: float
    settles_on: date
    from_symbol: str = ""


class Broker(ABC):
    """Operations the engine needs from a brokerage account."""

    name: str

    @abstractmethod
    def get_account_summary(self) -> AccountSummary:
        """Current equity, cash, and settled cash."""

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Open positions. This is the engine's source of truth, not local state."""

    @abstractmethod
    def get_open_orders(self) -> list[Order]:
        """Working orders, including resting protective stops."""

    @abstractmethod
    def place_entry_with_stop(
        self,
        symbol: str,
        shares: int,
        limit_price: float,
        stop_price: float,
    ) -> OrderResult:
        """Buy ``shares``, attaching a resting stop that survives this process.

        The stop must be submitted as a child of the entry so that it becomes
        live automatically on fill. A stop that only exists in the bot's memory
        is not a stop: a scheduled job that fails to start leaves the position
        completely unprotected.
        """

    @abstractmethod
    def place_protective_stop(
        self, symbol: str, shares: int, stop_price: float
    ) -> OrderResult:
        """Attach a standalone resting stop to a position already held.

        Needed when an entry filled but its child stop did not, or when a
        position was opened outside the bot. Distinct from the bracket path
        because there is no parent order to hang it from.
        """

    @abstractmethod
    def place_exit(
        self, symbol: str, shares: int, limit_price: float, reason: str = ""
    ) -> OrderResult:
        """Close (or reduce) a long position with a marketable limit order."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a working order. Returns whether the broker accepted it."""

    @abstractmethod
    def modify_stop(self, order_id: str, new_stop_price: float) -> bool:
        """Move a resting stop. Used to ratchet a trailing stop upward."""

    def health_check(self) -> tuple[bool, str]:
        """Confirm the connection works before any order is considered."""
        try:
            summary = self.get_account_summary()
        except BrokerError as exc:
            return False, f"{self.name}: {exc}"
        return True, (
            f"{self.name}: account {summary.account_id} "
            f"({summary.account_type}), equity ${summary.equity:,.2f}"
        )


@dataclass
class BrokerState:
    """Snapshot of everything read from the broker in one run."""

    summary: AccountSummary
    positions: list[Position] = field(default_factory=list)
    open_orders: list[Order] = field(default_factory=list)

    def position_for(self, symbol: str) -> Position | None:
        return next((p for p in self.positions if p.symbol == symbol), None)

    def protective_stop_for(self, symbol: str) -> Order | None:
        return next(
            (
                o
                for o in self.open_orders
                if o.symbol == symbol and o.is_protective_stop and o.status.is_working
            ),
            None,
        )

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions)

    def unprotected_positions(self) -> list[Position]:
        """Open positions with no working stop order.

        Should always be empty. A non-empty result means an entry filled but
        its child stop did not, and the engine repairs it before doing
        anything else.
        """
        return [p for p in self.positions if self.protective_stop_for(p.symbol) is None]
