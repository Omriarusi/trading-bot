"""Broker adapters: the simulator's behaviour and the live adapter's parsing.

The IBKR adapter cannot be exercised against a real account here, so its
tests target the parts that historically break silently: parsing IBKR's
inconsistent payload shapes, and the guard that keeps a PRIIPs-blocked
instrument from ever being sent as an order.
"""

from __future__ import annotations

import pytest

from bot.broker.base import BrokerState, Order, OrderSide, OrderStatus, OrderType, Position
from bot.broker.paper import PaperBroker
from bot.config import CostsConfig


@pytest.fixture
def broker() -> PaperBroker:
    return PaperBroker(starting_cash=1400.0, costs=CostsConfig())


# ----------------------------------------------------------------- simulator


def test_an_entry_creates_a_position_and_a_resting_stop(broker):
    result = broker.place_entry_with_stop("AAPL", 3, 100.0, 90.0)

    assert result.accepted
    assert result.stop_order_id is not None

    position = broker.get_positions()[0]
    assert position.symbol == "AAPL" and position.shares == 3

    stops = [o for o in broker.get_open_orders() if o.is_protective_stop]
    assert len(stops) == 1 and stops[0].stop_price == 90.0


def test_an_entry_that_exceeds_cash_is_rejected(broker):
    result = broker.place_entry_with_stop("AAPL", 100, 100.0, 90.0)

    assert not result.accepted
    assert "available" in result.message
    assert not broker.get_positions()


def test_a_stop_at_or_above_the_entry_is_rejected(broker):
    assert not broker.place_entry_with_stop("AAPL", 3, 100.0, 100.0).accepted
    assert not broker.place_entry_with_stop("AAPL", 3, 100.0, 110.0).accepted


def test_commission_is_charged_on_both_sides(broker):
    broker.place_entry_with_stop("AAPL", 3, 100.0, 90.0)
    cash_after_buy = broker.cash
    broker.place_exit("AAPL", 3, 100.0, reason="target")

    # A flat round trip must lose exactly two commissions.
    round_trip_cost = 1400.0 - broker.cash
    assert round_trip_cost == pytest.approx(2 * broker.costs.commission(3, 100.0), abs=0.01)
    assert cash_after_buy < 1400.0


def test_closing_a_position_cancels_its_stop(broker):
    """A resting stop on a closed position would open an unintended short."""
    broker.place_entry_with_stop("AAPL", 3, 100.0, 90.0)
    broker.place_exit("AAPL", 3, 105.0, reason="target")

    assert not broker.get_positions()
    assert not [o for o in broker.get_open_orders() if o.is_protective_stop]


def test_a_partial_exit_leaves_the_remainder_held(broker):
    broker.place_entry_with_stop("AAPL", 4, 100.0, 90.0)
    broker.place_exit("AAPL", 2, 105.0, reason="scale out")

    assert broker.get_positions()[0].shares == 2


def test_selling_more_than_held_sells_only_what_is_held(broker):
    broker.place_entry_with_stop("AAPL", 3, 100.0, 90.0)
    result = broker.place_exit("AAPL", 10, 105.0)

    assert result.accepted
    assert not broker.get_positions()


def test_exiting_an_unheld_symbol_is_rejected(broker):
    assert not broker.place_exit("MSFT", 1, 100.0).accepted


def test_a_protective_stop_needs_a_position(broker):
    assert not broker.place_protective_stop("AAPL", 3, 90.0).accepted

    broker.place_entry_with_stop("AAPL", 3, 100.0, 90.0)
    assert broker.place_protective_stop("AAPL", 3, 92.0).accepted


def test_a_stop_can_be_raised_but_the_order_must_exist(broker):
    broker.place_entry_with_stop("AAPL", 3, 100.0, 90.0)
    stop = next(o for o in broker.get_open_orders() if o.is_protective_stop)

    assert broker.modify_stop(stop.order_id, 95.0)
    assert not broker.modify_stop("nonexistent", 95.0)

    updated = next(o for o in broker.get_open_orders() if o.is_protective_stop)
    assert updated.stop_price == 95.0


def test_a_triggered_stop_closes_the_position(broker):
    broker.place_entry_with_stop("AAPL", 3, 100.0, 90.0)
    triggered = broker.trigger_stops({"AAPL": 88.0})

    assert triggered == ["AAPL"]
    assert not broker.get_positions()
    assert broker.realised_pnl < 0


def test_a_stop_that_is_not_touched_does_not_fire(broker):
    broker.place_entry_with_stop("AAPL", 3, 100.0, 90.0)
    assert broker.trigger_stops({"AAPL": 95.0}) == []
    assert broker.get_positions()


def test_equity_reflects_the_current_mark(broker):
    broker.place_entry_with_stop("AAPL", 3, 100.0, 90.0)
    broker.mark_to_market({"AAPL": 120.0})

    # Equity is cash plus the marked position, less the commission paid.
    expected = broker.cash + 3 * 120.0
    assert broker.get_account_summary().equity == pytest.approx(expected)


def test_state_persists_across_instances(tmp_path):
    path = tmp_path / "paper.json"
    first = PaperBroker(starting_cash=1400.0, costs=CostsConfig(), state_path=path)
    first.place_entry_with_stop("AAPL", 3, 100.0, 90.0)

    second = PaperBroker(starting_cash=9999.0, costs=CostsConfig(), state_path=path)
    assert second.cash == pytest.approx(first.cash)
    assert second.get_positions()[0].symbol == "AAPL"
    assert [o for o in second.get_open_orders() if o.is_protective_stop]


def test_corrupt_paper_state_falls_back_to_a_fresh_account(tmp_path):
    path = tmp_path / "paper.json"
    path.write_text("not json at all")

    broker = PaperBroker(starting_cash=1400.0, costs=CostsConfig(), state_path=path)
    assert broker.cash == 1400.0


def test_health_check_reports_the_account(broker):
    healthy, message = broker.health_check()
    assert healthy and "PAPER" in message


# --------------------------------------------------------------- broker state


def _position(symbol: str) -> Position:
    return Position(symbol=symbol, shares=10, average_cost=100.0, market_price=110.0)


def _stop(symbol: str) -> Order:
    return Order(
        order_id="1", symbol=symbol, side=OrderSide.SELL, order_type=OrderType.STOP,
        shares=10, status=OrderStatus.SUBMITTED, stop_price=95.0,
    )


def test_unprotected_positions_are_identified():
    from bot.broker.base import AccountSummary

    state = BrokerState(
        summary=AccountSummary("U1", 1400.0, 500.0, 500.0, 500.0),
        positions=[_position("AAPL"), _position("MSFT")],
        open_orders=[_stop("AAPL")],
    )

    assert [p.symbol for p in state.unprotected_positions()] == ["MSFT"]
    assert state.protective_stop_for("AAPL") is not None
    assert state.gross_exposure == pytest.approx(2200.0)


def test_a_filled_stop_no_longer_counts_as_protection():
    """A stop that has already fired is not protecting anything."""
    from bot.broker.base import AccountSummary

    filled = Order(
        order_id="1", symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.STOP,
        shares=10, status=OrderStatus.FILLED, stop_price=95.0,
    )
    state = BrokerState(
        summary=AccountSummary("U1", 1400.0, 500.0, 500.0, 500.0),
        positions=[_position("AAPL")],
        open_orders=[filled],
    )

    assert [p.symbol for p in state.unprotected_positions()] == ["AAPL"]


# ------------------------------------------------------------- ibkr adapter


def test_priips_blocked_instruments_are_refused_before_any_request():
    """SPY drives the regime filter and must never reach the order path.

    Asserted at the execution layer rather than trusted to the universe
    definition, so a future edit to the ticker list cannot produce an order
    an IBIE account is legally barred from placing.
    """
    from bot.broker.base import OrderRejected
    from bot.broker.ibkr import IbkrBroker

    broker = IbkrBroker.__new__(IbkrBroker)  # no network, no credentials

    for symbol in ("SPY", "QQQ", "TQQQ", "spy"):
        with pytest.raises(OrderRejected, match="signal-only"):
            broker._assert_tradable(symbol)

    broker._assert_tradable("AAPL")  # a single stock is fine


def test_missing_credentials_are_reported_by_name(monkeypatch):
    from bot.broker.ibkr import missing_credentials

    for name in ("IBKR_ACCOUNT_ID", "IBKR_CONSUMER_KEY", "IBKR_ACCESS_TOKEN",
                 "IBKR_ACCESS_TOKEN_SECRET", "IBKR_DH_PRIME",
                 "IBKR_SIGNATURE_KEY", "IBKR_ENCRYPTION_KEY"):
        monkeypatch.delenv(name, raising=False)

    assert "IBKR_ACCOUNT_ID" in missing_credentials()
    monkeypatch.setenv("IBKR_ACCOUNT_ID", "DU123")
    assert "IBKR_ACCOUNT_ID" not in missing_credentials()


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"netliquidation": 1400.0}, 1400.0),
        ({"netliquidation": {"amount": 1400.0}}, 1400.0),
        ({"NetLiquidation": {"amount": 1400.0}}, 1400.0),
        ({"netliquidation": "1,400.00"}, 1400.0),
        ({}, 0.0),
    ],
)
def test_amount_parsing_handles_ibkrs_payload_shapes(payload, expected):
    """IBKR returns bare numbers, wrapped amounts, and formatted strings."""
    from bot.broker.ibkr import _amount

    assert _amount(payload, "netliquidation") == expected


def test_position_parsing_falls_back_to_cost_when_unmarked():
    """A zero mark is a stale snapshot, not a worthless holding."""
    from bot.broker.ibkr import _parse_position

    position = _parse_position(
        {"contractDesc": "AAPL", "position": 5, "avgCost": 100.0, "mktPrice": 0}
    )
    assert position is not None
    assert position.market_price == 100.0  # exposure not understated


def test_position_parsing_rejects_unusable_rows():
    from bot.broker.ibkr import _parse_position

    assert _parse_position({}) is None
    assert _parse_position({"position": 5}) is None
    assert _parse_position("not a dict") is None


def test_order_parsing_marks_a_partial_fill():
    from bot.broker.ibkr import _parse_order

    order = _parse_order({
        "orderId": "7", "ticker": "AAPL", "side": "B", "orderType": "LMT",
        "status": "Submitted", "filledQuantity": 2, "remainingQuantity": 3,
    })
    assert order is not None
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert order.shares == 5


def test_order_parsing_identifies_a_protective_stop():
    from bot.broker.ibkr import _parse_order

    order = _parse_order({
        "orderId": "8", "ticker": "AAPL", "side": "SELL", "orderType": "STP",
        "status": "PreSubmitted", "remainingQuantity": 5, "auxPrice": 95.0,
    })
    assert order is not None and order.is_protective_stop
    assert order.stop_price == 95.0


def test_pem_keys_survive_being_flattened_into_an_env_var():
    r"""A key pasted through a shell often arrives with literal \n."""
    from bot.broker.ibkr import _normalise_pem

    flattened = "-----BEGIN PRIVATE KEY-----\\nABC\\n-----END PRIVATE KEY-----"
    assert "\n" in _normalise_pem(flattened)
    assert "\\n" not in _normalise_pem(flattened)

    proper = "-----BEGIN PRIVATE KEY-----\nABC\n-----END PRIVATE KEY-----"
    assert _normalise_pem(proper) == proper
