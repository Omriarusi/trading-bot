"""Simulated broker backed by a JSON file.

Used for ``dry_run`` mode and by the tests. It implements the same interface
as the live adapter, including resting stop orders that trigger against
subsequent price updates, so an engine bug shows up here rather than in a real
account.

It is a simulator, not a backtester: it models fills optimistically at the
requested limit price and applies commission, but it does not model spread,
queue position, or partial fills.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from bot.broker.base import (
    AccountSummary,
    Broker,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from bot.config import CostsConfig

log = logging.getLogger(__name__)


class PaperBroker(Broker):
    """In-memory account with optional persistence between runs."""

    name = "paper"

    def __init__(
        self,
        starting_cash: float = 1400.0,
        costs: CostsConfig | None = None,
        state_path: Path | None = None,
        account_id: str = "PAPER",
    ):
        self.costs = costs or CostsConfig()
        self.state_path = state_path
        self.account_id = account_id

        self.cash = starting_cash
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._next_order_id = 1
        self.fills: list[dict] = []
        self.realised_pnl = 0.0
        self._cost_basis: dict[str, float] = {}

        if state_path and state_path.exists():
            self._load()

    # ---------------------------------------------------------------- reads

    def get_account_summary(self) -> AccountSummary:
        equity = self.cash + sum(p.market_value for p in self._positions.values())
        return AccountSummary(
            account_id=self.account_id,
            equity=equity,
            cash=self.cash,
            settled_cash=self.cash,
            buying_power=self.cash,
            is_margin=False,
        )

    def get_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.shares != 0]

    def get_open_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.status.is_working]

    # --------------------------------------------------------------- writes

    def place_entry_with_stop(
        self, symbol: str, shares: int, limit_price: float, stop_price: float
    ) -> OrderResult:
        if shares <= 0:
            return OrderResult.rejected(f"{symbol}: refusing to order {shares} shares")
        if stop_price >= limit_price:
            return OrderResult.rejected(
                f"{symbol}: stop {stop_price} is not below entry {limit_price}"
            )

        commission = self.costs.commission(shares, limit_price)
        cost = shares * limit_price + commission
        if cost > self.cash:
            return OrderResult.rejected(
                f"{symbol}: needs ${cost:.2f} but only ${self.cash:.2f} available"
            )

        # The simulator assumes the marketable limit fills immediately, which
        # is the point of pricing it through the touch.
        self.cash -= cost
        existing = self._positions.get(symbol)
        if existing:
            total_shares = existing.shares + shares
            blended = (
                existing.average_cost * existing.shares + limit_price * shares
            ) / total_shares
            self._positions[symbol] = Position(symbol, total_shares, blended, limit_price)
        else:
            self._positions[symbol] = Position(symbol, shares, limit_price, limit_price)

        entry_id = self._record_order(
            symbol, OrderSide.BUY, OrderType.LIMIT, shares, OrderStatus.FILLED,
            limit_price=limit_price, filled_shares=shares, average_fill_price=limit_price,
        )
        stop_id = self._record_order(
            symbol, OrderSide.SELL, OrderType.STOP, shares, OrderStatus.SUBMITTED,
            stop_price=stop_price,
        )
        self._log_fill(symbol, OrderSide.BUY, shares, limit_price, commission, "entry")
        self._save()

        return OrderResult(
            accepted=True,
            order_id=entry_id,
            stop_order_id=stop_id,
            message=f"bought {shares} {symbol} at {limit_price:.2f}, stop {stop_price:.2f}",
        )

    def place_protective_stop(
        self, symbol: str, shares: int, stop_price: float
    ) -> OrderResult:
        position = self._positions.get(symbol)
        if position is None or position.shares <= 0:
            return OrderResult.rejected(f"{symbol}: no position to protect")
        if stop_price <= 0:
            return OrderResult.rejected(f"{symbol}: invalid stop {stop_price}")

        stop_id = self._record_order(
            symbol, OrderSide.SELL, OrderType.STOP, min(shares, position.shares),
            OrderStatus.SUBMITTED, stop_price=stop_price,
        )
        self._save()
        return OrderResult(
            accepted=True, order_id=stop_id, stop_order_id=stop_id,
            message=f"stop for {shares} {symbol} at {stop_price:.2f}",
        )

    def place_exit(
        self, symbol: str, shares: int, limit_price: float, reason: str = ""
    ) -> OrderResult:
        position = self._positions.get(symbol)
        if position is None or position.shares <= 0:
            return OrderResult.rejected(f"{symbol}: no position to close")

        shares = min(shares, position.shares)
        commission = self.costs.commission(shares, limit_price)
        proceeds = shares * limit_price - commission
        self.cash += proceeds
        self.realised_pnl += shares * (limit_price - position.average_cost) - commission

        remaining = position.shares - shares
        if remaining > 0:
            self._positions[symbol] = Position(
                symbol, remaining, position.average_cost, limit_price
            )
        else:
            del self._positions[symbol]
            # A closed position must not leave its stop resting, or the next
            # trigger would open an unintended short.
            for order in list(self._orders.values()):
                if order.symbol == symbol and order.is_protective_stop and order.status.is_working:
                    self._orders[order.order_id] = _with_status(order, OrderStatus.CANCELLED)

        order_id = self._record_order(
            symbol, OrderSide.SELL, OrderType.LIMIT, shares, OrderStatus.FILLED,
            limit_price=limit_price, filled_shares=shares, average_fill_price=limit_price,
        )
        self._log_fill(symbol, OrderSide.SELL, shares, limit_price, commission, reason)
        self._save()

        return OrderResult(
            accepted=True,
            order_id=order_id,
            message=f"sold {shares} {symbol} at {limit_price:.2f} ({reason})",
        )

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None or not order.status.is_working:
            return False
        self._orders[order_id] = _with_status(order, OrderStatus.CANCELLED)
        self._save()
        return True

    def modify_stop(self, order_id: str, new_stop_price: float) -> bool:
        order = self._orders.get(order_id)
        if order is None or not order.status.is_working or not order.is_protective_stop:
            return False
        self._orders[order_id] = Order(
            order_id=order.order_id, symbol=order.symbol, side=order.side,
            order_type=order.order_type, shares=order.shares, status=order.status,
            limit_price=order.limit_price, stop_price=new_stop_price,
            filled_shares=order.filled_shares, average_fill_price=order.average_fill_price,
        )
        self._save()
        return True

    # ------------------------------------------------------------ simulation

    def mark_to_market(self, prices: dict[str, float]) -> None:
        """Update position marks. Does not trigger stops."""
        for symbol, price in prices.items():
            position = self._positions.get(symbol)
            if position and price > 0:
                self._positions[symbol] = Position(
                    symbol, position.shares, position.average_cost, price, position.conid
                )
        self._save()

    def trigger_stops(self, lows: dict[str, float]) -> list[str]:
        """Fill any resting stop whose trigger was touched.

        Fills at the trigger price, which flatters the result: a real stop
        becomes a market order and in a gap down fills materially lower. The
        backtester models that properly; this is only for smoke-testing.
        """
        triggered = []
        for order in list(self._orders.values()):
            if not (order.is_protective_stop and order.status.is_working):
                continue
            low = lows.get(order.symbol)
            if low is not None and order.stop_price is not None and low <= order.stop_price:
                self._orders[order.order_id] = _with_status(order, OrderStatus.FILLED)
                self.place_exit(order.symbol, order.shares, order.stop_price, reason="stop")
                triggered.append(order.symbol)
        return triggered

    # -------------------------------------------------------------- helpers

    def _record_order(
        self, symbol: str, side: OrderSide, order_type: OrderType, shares: int,
        status: OrderStatus, **kwargs,
    ) -> str:
        order_id = str(self._next_order_id)
        self._next_order_id += 1
        self._orders[order_id] = Order(
            order_id=order_id, symbol=symbol, side=side, order_type=order_type,
            shares=shares, status=status, **kwargs,
        )
        return order_id

    def _log_fill(
        self, symbol: str, side: OrderSide, shares: int, price: float,
        commission: float, reason: str,
    ) -> None:
        self.fills.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "symbol": symbol,
                "side": side.value,
                "shares": shares,
                "price": round(price, 4),
                "commission": round(commission, 4),
                "reason": reason,
            }
        )

    def _save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cash": self.cash,
            "realised_pnl": self.realised_pnl,
            "next_order_id": self._next_order_id,
            "positions": [
                {
                    "symbol": p.symbol, "shares": p.shares,
                    "average_cost": p.average_cost, "market_price": p.market_price,
                }
                for p in self._positions.values()
            ],
            "orders": [
                {
                    "order_id": o.order_id, "symbol": o.symbol, "side": o.side.value,
                    "order_type": o.order_type.value, "shares": o.shares,
                    "status": o.status.value, "limit_price": o.limit_price,
                    "stop_price": o.stop_price, "filled_shares": o.filled_shares,
                }
                for o in self._orders.values()
            ],
            "fills": self.fills[-500:],
        }
        self.state_path.write_text(json.dumps(payload, indent=2))

    def _load(self) -> None:
        try:
            payload = json.loads(self.state_path.read_text())
        except (OSError, ValueError) as exc:
            log.warning("paper state at %s is unreadable (%s); starting fresh", self.state_path, exc)
            return

        self.cash = float(payload.get("cash", self.cash))
        self.realised_pnl = float(payload.get("realised_pnl", 0.0))
        self._next_order_id = int(payload.get("next_order_id", 1))
        self.fills = payload.get("fills", [])
        self._positions = {
            row["symbol"]: Position(
                row["symbol"], int(row["shares"]),
                float(row["average_cost"]), float(row["market_price"]),
            )
            for row in payload.get("positions", [])
        }
        self._orders = {
            row["order_id"]: Order(
                order_id=row["order_id"], symbol=row["symbol"],
                side=OrderSide(row["side"]), order_type=OrderType(row["order_type"]),
                shares=int(row["shares"]), status=OrderStatus(row["status"]),
                limit_price=row.get("limit_price"), stop_price=row.get("stop_price"),
                filled_shares=int(row.get("filled_shares", 0)),
            )
            for row in payload.get("orders", [])
        }


def _with_status(order: Order, status: OrderStatus) -> Order:
    return Order(
        order_id=order.order_id, symbol=order.symbol, side=order.side,
        order_type=order.order_type, shares=order.shares, status=status,
        limit_price=order.limit_price, stop_price=order.stop_price,
        filled_shares=order.filled_shares, average_fill_price=order.average_fill_price,
    )
