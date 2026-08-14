"""Interactive Brokers adapter over the Client Portal Web API.

Authentication is OAuth 1.0a, which is the only IBKR scheme that works fully
headless — no TWS, no IB Gateway, no interactive login. That is what makes a
scheduled job on a runner viable at all.

Signing is delegated to ``ibind``; implementing IBKR's Diffie-Hellman live
session token exchange by hand would be a large amount of security-critical
code for no benefit.

Credentials are read from the environment, never from the repository:

    IBKR_ACCOUNT_ID          e.g. U1234567 (or DU… for paper)
    IBKR_CONSUMER_KEY        the 9-character key registered in the portal
    IBKR_ACCESS_TOKEN
    IBKR_ACCESS_TOKEN_SECRET
    IBKR_DH_PRIME            hex prime extracted from dhparam.pem
    IBKR_SIGNATURE_KEY       PEM body of private_signature.pem
    IBKR_ENCRYPTION_KEY      PEM body of private_encryption.pem
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import UTC, datetime

from bot.broker.base import (
    AccountSummary,
    Broker,
    BrokerError,
    Order,
    OrderRejected,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from bot.universe_list import PRIIPS_BLOCKED_SIGNALS_ONLY

log = logging.getLogger(__name__)

# IBKR asks the same handful of confirmation questions on every order. Each is
# answered deliberately: the risk they describe is already bounded elsewhere.
_ORDER_ANSWERS: dict[str, bool] = {
    # We deliberately trade without a paid market-data subscription; prices
    # come from the data layer, and orders are marketable limits, so an
    # unpriced order can still never fill at an arbitrary level.
    "MISSING_MARKET_DATA": True,
    # Stop orders can gap through their trigger. Accepted knowingly: the
    # alternative is holding no stop at all.
    "STOP_ORDER_RISKS": True,
    "TRIGGER_AND_FILL": True,
    "PRICE_PERCENTAGE_CONSTRAINT": True,
    "TICK_SIZE_LIMIT": True,
    "CASH_QUANTITY": True,
    "CASH_QUANTITY_ORDER": True,
    # Size and value limits are refused. Hitting one means the sizing logic
    # produced something larger than IBKR's own sanity threshold, which is a
    # bug, not a trade to confirm.
    "ORDER_SIZE_LIMIT": False,
    "ORDER_VALUE_LIMIT": False,
    "SIZE_MODIFICATION_LIMIT": False,
    "MANDATORY_CAP_PRICE": False,
    "DISRUPTIVE_ORDERS": False,
    "MULTIPLE_ACCOUNTS": False,
    "CLOSE_POSITION": True,
}

_REQUIRED_ENV = (
    "IBKR_ACCOUNT_ID",
    "IBKR_CONSUMER_KEY",
    "IBKR_ACCESS_TOKEN",
    "IBKR_ACCESS_TOKEN_SECRET",
    "IBKR_DH_PRIME",
    "IBKR_SIGNATURE_KEY",
    "IBKR_ENCRYPTION_KEY",
)


def missing_credentials() -> list[str]:
    """Environment variables required for a live connection that are unset."""
    return [name for name in _REQUIRED_ENV if not os.environ.get(name)]


class IbkrBroker(Broker):
    """Live or paper IBKR account over the Web API."""

    name = "ibkr"

    def __init__(self, account_id: str | None = None, timeout: float = 20.0):
        missing = missing_credentials()
        if missing:
            raise BrokerError(
                "Missing IBKR credentials: "
                + ", ".join(missing)
                + ". See docs/IBKR_SETUP.md for how to generate them."
            )

        self.account_id = account_id or os.environ["IBKR_ACCOUNT_ID"]
        self._conid_cache: dict[str, int] = {}
        self._client = self._build_client(timeout)

    def _build_client(self, timeout: float):
        # Imported lazily so that the rest of the bot — backtests, dry runs,
        # tests — does not require the OAuth dependency chain to be installed.
        try:
            from ibind import IbkrClient
            from ibind.oauth.oauth1a import OAuth1aConfig
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise BrokerError(
                "ibind is not installed. Run: pip install 'ibind[oauth]'"
            ) from exc

        oauth_config = OAuth1aConfig(
            access_token=os.environ["IBKR_ACCESS_TOKEN"],
            access_token_secret=os.environ["IBKR_ACCESS_TOKEN_SECRET"],
            consumer_key=os.environ["IBKR_CONSUMER_KEY"],
            dh_prime=os.environ["IBKR_DH_PRIME"],
            # Keys arrive as environment variables rather than files so that
            # nothing secret is ever written to the runner's disk.
            signature_key=_normalise_pem(os.environ["IBKR_SIGNATURE_KEY"]),
            encryption_key=_normalise_pem(os.environ["IBKR_ENCRYPTION_KEY"]),
        )

        try:
            return IbkrClient(
                account_id=self.account_id,
                use_oauth=True,
                oauth_config=oauth_config,
                timeout=timeout,
            )
        except Exception as exc:
            raise BrokerError(f"IBKR authentication failed: {exc}") from exc

    # ---------------------------------------------------------------- reads

    def get_account_summary(self) -> AccountSummary:
        data = self._unwrap(self._client.portfolio_summary(self.account_id), "portfolio_summary")
        if not isinstance(data, dict):
            raise BrokerError(f"portfolio_summary returned {type(data).__name__}")

        equity = _amount(data, "netliquidation")
        cash = _amount(data, "totalcashvalue", "availablefunds")
        settled = _amount(data, "settledcash")
        buying_power = _amount(data, "buyingpower", "availablefunds")

        # A cash account reports no meaningful margin buying power. Treating
        # a mis-parsed field as margin would let the sizer spend money that
        # is not there, so the default on ambiguity is the stricter one.
        excess_liquidity = _amount(data, "excessliquidity")
        is_margin = buying_power > equity * 1.05 and excess_liquidity > 0

        if equity <= 0:
            raise BrokerError(
                f"portfolio_summary reported equity of {equity}; refusing to "
                "trade against an implausible account value"
            )

        return AccountSummary(
            account_id=self.account_id,
            equity=equity,
            cash=cash,
            # In a cash account, unsettled proceeds must not be spent. When
            # IBKR does not report settled cash, assume none of it is settled
            # beyond available funds.
            settled_cash=settled if settled > 0 else min(cash, buying_power),
            buying_power=buying_power,
            currency=str(data.get("currency", "USD") or "USD"),
            is_margin=is_margin,
        )

    def get_positions(self) -> list[Position]:
        positions: list[Position] = []
        # The endpoint pages at 30 rows; loop until a short page comes back.
        for page in range(10):
            data = self._unwrap(
                self._client.positions(account_id=self.account_id, page=page), "positions"
            )
            if not data:
                break
            rows = data if isinstance(data, list) else [data]
            for row in rows:
                position = _parse_position(row)
                if position is not None and position.shares != 0:
                    positions.append(position)
            if len(rows) < 30:
                break
        return positions

    def get_open_orders(self) -> list[Order]:
        data = self._unwrap(
            self._client.live_orders(account_id=self.account_id, force=True), "live_orders"
        )
        rows = data.get("orders", []) if isinstance(data, dict) else (data or [])
        orders = [_parse_order(row) for row in rows]
        return [o for o in orders if o is not None and o.status.is_working]

    def resolve_conid(self, symbol: str) -> int:
        """Map a ticker to IBKR's contract id, cached for the run."""
        cached = self._conid_cache.get(symbol)
        if cached is not None:
            return cached

        data = self._unwrap(
            self._client.stock_conid_by_symbol(symbol), f"stock_conid_by_symbol({symbol})"
        )
        conid = data.get(symbol) if isinstance(data, dict) else data
        if not conid:
            raise BrokerError(f"No contract id found for {symbol}")
        self._conid_cache[symbol] = int(conid)
        return int(conid)

    # --------------------------------------------------------------- writes

    def place_entry_with_stop(
        self, symbol: str, shares: int, limit_price: float, stop_price: float
    ) -> OrderResult:
        """Submit a buy with an attached protective stop, as one bracket.

        The stop is a *child* order, so IBKR activates it automatically when
        the entry fills. This is the single most important property of the
        whole system: protection does not depend on this process ever running
        again.
        """
        self._assert_tradable(symbol)
        if shares <= 0:
            return OrderResult.rejected(f"{symbol}: refusing to order {shares} shares")
        if stop_price >= limit_price:
            return OrderResult.rejected(
                f"{symbol}: stop {stop_price} is not below entry {limit_price}"
            )

        try:
            from ibind import OrderRequest
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise BrokerError("ibind is not installed") from exc

        conid = self.resolve_conid(symbol)
        # A deterministic-prefixed unique id lets a duplicate submission be
        # spotted in IBKR's order history after a retry.
        parent_coid = f"e{uuid.uuid4().hex[:16]}"

        entry = OrderRequest(
            conid=conid,
            side=OrderSide.BUY.value,
            quantity=shares,
            order_type=OrderType.LIMIT.value,
            price=limit_price,
            acct_id=self.account_id,
            coid=parent_coid,
            tif="DAY",
            outside_rth=False,
        )
        protective_stop = OrderRequest(
            conid=conid,
            side=OrderSide.SELL.value,
            quantity=shares,
            order_type=OrderType.STOP.value,
            aux_price=stop_price,
            acct_id=self.account_id,
            parent_id=parent_coid,
            # Good-till-cancelled: the stop must outlive the session that
            # created it, otherwise it expires overnight and the position is
            # naked at the next open.
            tif="GTC",
        )

        result = self._submit([entry, protective_stop], f"entry {symbol}")
        ids = _order_ids(result)
        return OrderResult(
            accepted=True,
            order_id=ids[0] if ids else None,
            stop_order_id=ids[1] if len(ids) > 1 else None,
            message=f"bought {shares} {symbol} at limit {limit_price:.2f}, stop {stop_price:.2f}",
        )

    def place_protective_stop(
        self, symbol: str, shares: int, stop_price: float
    ) -> OrderResult:
        """Place a standalone GTC sell stop against an existing position."""
        self._assert_tradable(symbol)
        if shares <= 0 or stop_price <= 0:
            return OrderResult.rejected(
                f"{symbol}: invalid protective stop ({shares} shares at {stop_price})"
            )

        try:
            from ibind import OrderRequest
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise BrokerError("ibind is not installed") from exc

        stop_order = OrderRequest(
            conid=self.resolve_conid(symbol),
            side=OrderSide.SELL.value,
            quantity=shares,
            order_type=OrderType.STOP.value,
            aux_price=stop_price,
            acct_id=self.account_id,
            coid=f"s{uuid.uuid4().hex[:16]}",
            tif="GTC",
        )

        result = self._submit(stop_order, f"protective stop {symbol}")
        ids = _order_ids(result)
        return OrderResult(
            accepted=True,
            stop_order_id=ids[0] if ids else None,
            order_id=ids[0] if ids else None,
            message=f"stop for {shares} {symbol} at {stop_price:.2f}",
        )

    def place_exit(
        self, symbol: str, shares: int, limit_price: float, reason: str = ""
    ) -> OrderResult:
        """Close a long with a marketable limit order.

        Any resting stop for the symbol is cancelled first. Selling while the
        stop is still live risks both filling and leaving a short position,
        which a cash account cannot hold.
        """
        self._assert_tradable(symbol)
        if shares <= 0:
            return OrderResult.rejected(f"{symbol}: refusing to sell {shares} shares")

        try:
            from ibind import OrderRequest
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise BrokerError("ibind is not installed") from exc

        for order in self.get_open_orders():
            if order.symbol == symbol and order.is_protective_stop:
                self.cancel_order(order.order_id)
                # IBKR processes the cancel asynchronously; give it a moment
                # so the sell is not rejected as an over-sell.
                time.sleep(1.0)

        exit_order = OrderRequest(
            conid=self.resolve_conid(symbol),
            side=OrderSide.SELL.value,
            quantity=shares,
            order_type=OrderType.LIMIT.value,
            price=limit_price,
            acct_id=self.account_id,
            coid=f"x{uuid.uuid4().hex[:16]}",
            tif="DAY",
            outside_rth=False,
        )

        result = self._submit(exit_order, f"exit {symbol} ({reason})")
        ids = _order_ids(result)
        return OrderResult(
            accepted=True,
            order_id=ids[0] if ids else None,
            message=f"sold {shares} {symbol} at limit {limit_price:.2f} ({reason})",
        )

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._unwrap(self._client.cancel_order(order_id, account_id=self.account_id), "cancel_order")
            return True
        except BrokerError as exc:
            log.warning("cancel_order(%s) failed: %s", order_id, exc)
            return False

    def modify_stop(self, order_id: str, new_stop_price: float) -> bool:
        """Raise a resting stop to a new trigger price."""
        try:
            from ibind import OrderRequest
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise BrokerError("ibind is not installed") from exc

        existing = next((o for o in self.get_open_orders() if o.order_id == order_id), None)
        if existing is None:
            log.warning("modify_stop: order %s is no longer working", order_id)
            return False

        amended = OrderRequest(
            conid=existing.conid,
            side=OrderSide.SELL.value,
            quantity=existing.shares,
            order_type=OrderType.STOP.value,
            aux_price=new_stop_price,
            acct_id=self.account_id,
            tif="GTC",
        )
        try:
            self._unwrap(
                self._client.modify_order(order_id, amended, _ORDER_ANSWERS, self.account_id),
                "modify_order",
            )
            return True
        except BrokerError as exc:
            log.warning("modify_stop(%s -> %.2f) failed: %s", order_id, new_stop_price, exc)
            return False

    # -------------------------------------------------------------- helpers

    def _assert_tradable(self, symbol: str) -> None:
        """Refuse instruments the account is legally barred from buying.

        Under PRIIPs an IBIE account cannot buy US-domiciled ETFs. The
        benchmark is read as a signal and must never leak into an order, so
        this is asserted at the last possible moment rather than trusted to
        the universe definition.
        """
        if symbol.upper() in PRIIPS_BLOCKED_SIGNALS_ONLY:
            raise OrderRejected(
                f"{symbol} is a signal-only instrument and must never be traded "
                "(US-domiciled ETFs are blocked for EEA accounts under PRIIPs)"
            )

    def _submit(self, order_request, description: str):
        try:
            result = self._client.place_order(order_request, _ORDER_ANSWERS, self.account_id)
        except Exception as exc:
            raise OrderRejected(f"{description}: {exc}") from exc

        data = getattr(result, "data", result)
        # IBKR reports business-level failures inside a 200 response, so the
        # payload has to be inspected rather than trusting the status code.
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            if isinstance(row, dict) and row.get("error"):
                raise OrderRejected(f"{description}: {row['error']}")
        log.info("submitted %s: %s", description, data)
        return result

    @staticmethod
    def _unwrap(result, description: str):
        if result is None:
            raise BrokerError(f"{description}: no response")
        data = getattr(result, "data", result)
        if isinstance(data, dict) and data.get("error"):
            raise BrokerError(f"{description}: {data['error']}")
        return data


def _normalise_pem(value: str) -> str:
    r"""Restore newlines in a PEM key passed through an environment variable.

    GitHub Actions secrets preserve newlines, but a key pasted into a shell or
    a ``.env`` file often arrives with literal ``\n``. Both forms are accepted
    so a setup mistake fails loudly at signing time rather than silently.
    """
    text = value.strip()
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    return text


def _amount(data: dict, *keys: str) -> float:
    """Read a numeric field from IBKR's summary payload.

    Values arrive either as a bare number or wrapped as ``{"amount": x}``,
    and key casing is inconsistent across endpoints.
    """
    lowered = {str(k).lower(): v for k, v in data.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if isinstance(value, dict):
            value = value.get("amount")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.replace(",", "").replace("$", ""))
            except ValueError:
                continue
    return 0.0


def _parse_position(row: dict) -> Position | None:
    if not isinstance(row, dict):
        return None
    symbol = row.get("contractDesc") or row.get("ticker") or row.get("symbol")
    if not symbol:
        return None
    try:
        shares = int(float(row.get("position", 0) or 0))
    except (TypeError, ValueError):
        return None

    market_price = float(row.get("mktPrice", 0) or 0)
    average_cost = float(row.get("avgCost", 0) or 0)
    # A zero mark usually means a stale snapshot rather than a worthless
    # holding; fall back to cost so exposure is not understated.
    if market_price <= 0:
        market_price = average_cost

    return Position(
        symbol=str(symbol).split()[0].upper(),
        shares=shares,
        average_cost=average_cost,
        market_price=market_price,
        conid=int(row["conid"]) if row.get("conid") else None,
    )


_STATUS_MAP = {
    "PendingSubmit": OrderStatus.PENDING,
    "PreSubmitted": OrderStatus.SUBMITTED,
    "Submitted": OrderStatus.SUBMITTED,
    "Filled": OrderStatus.FILLED,
    "Cancelled": OrderStatus.CANCELLED,
    "PendingCancel": OrderStatus.SUBMITTED,
    "Inactive": OrderStatus.CANCELLED,
    "Rejected": OrderStatus.REJECTED,
}

_TYPE_MAP = {
    "LMT": OrderType.LIMIT,
    "LIMIT": OrderType.LIMIT,
    "MKT": OrderType.MARKET,
    "MARKET": OrderType.MARKET,
    "STP": OrderType.STOP,
    "STOP": OrderType.STOP,
    "STOP_LIMIT": OrderType.STOP_LIMIT,
    "STP LMT": OrderType.STOP_LIMIT,
}


def _parse_order(row: dict) -> Order | None:
    if not isinstance(row, dict):
        return None
    order_id = row.get("orderId") or row.get("order_id")
    symbol = row.get("ticker") or row.get("symbol")
    if not order_id or not symbol:
        return None

    raw_status = str(row.get("status") or row.get("order_status") or "")
    raw_type = str(row.get("origOrderType") or row.get("orderType") or "").upper()
    filled = float(row.get("filledQuantity", 0) or 0)
    remaining = float(row.get("remainingQuantity", 0) or 0)

    status = _STATUS_MAP.get(raw_status, OrderStatus.SUBMITTED)
    if status is OrderStatus.SUBMITTED and filled > 0 and remaining > 0:
        status = OrderStatus.PARTIALLY_FILLED

    return Order(
        order_id=str(order_id),
        symbol=str(symbol).upper(),
        side=OrderSide.SELL if str(row.get("side", "")).upper() in ("S", "SELL") else OrderSide.BUY,
        order_type=_TYPE_MAP.get(raw_type, OrderType.LIMIT),
        shares=int(filled + remaining) or int(float(row.get("totalSize", 0) or 0)),
        status=status,
        limit_price=_optional_float(row.get("price")),
        stop_price=_optional_float(row.get("stop_price") or row.get("auxPrice")),
        filled_shares=int(filled),
        average_fill_price=_optional_float(row.get("avgPrice")),
        conid=int(row["conid"]) if row.get("conid") else None,
    )


def _optional_float(value) -> float | None:
    if value in (None, "", "--"):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _order_ids(result) -> list[str]:
    """Pull the broker's order ids out of a submission response."""
    data = getattr(result, "data", result)
    rows = data if isinstance(data, list) else [data]
    ids = []
    for row in rows:
        if isinstance(row, dict):
            value = row.get("order_id") or row.get("orderId")
            if value:
                ids.append(str(value))
    return ids


def utc_now() -> datetime:
    return datetime.now(UTC)
