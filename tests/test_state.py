"""Persistent state: sessions, reconciliation, and surviving a fresh container.

Every scheduled run starts on a new machine with nothing but the repository,
so anything these tests get wrong shows up as the bot forgetting a stop or
mis-measuring its own drawdown.
"""

from __future__ import annotations

from datetime import date

from bot.state import BotState, PositionState


def test_a_missing_state_file_yields_a_fresh_state(tmp_path):
    state = BotState.load(tmp_path / "absent.json")
    assert state.run_count == 0
    assert state.positions == {}


def test_state_round_trips_through_disk(tmp_path):
    path = tmp_path / "state.json"
    state = BotState()
    state.begin_session(1400.0, date(2026, 1, 5))
    state.open_position("AAPL", entry_price=150.0, stop=140.0,
                        stop_order_id="42", today=date(2026, 1, 5))
    state.save(path)

    reloaded = BotState.load(path)
    assert reloaded.peak_equity == 1400.0
    assert reloaded.run_count == 1
    assert reloaded.positions["AAPL"].entry_price == 150.0
    assert reloaded.positions["AAPL"].current_stop == 140.0
    assert reloaded.positions["AAPL"].stop_order_id == "42"


def test_corrupt_state_does_not_stop_the_bot(tmp_path):
    """Losing bookkeeping costs a trailing-stop update, not a position.

    The broker still holds the real picture, so the run must continue rather
    than abort and leave positions unmanaged.
    """
    path = tmp_path / "state.json"
    path.write_text("{ this is not json")

    state = BotState.load(path)
    assert state.run_count == 0


def test_a_state_file_from_a_future_version_is_discarded(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"version": 999, "peak_equity": 5000.0}')

    state = BotState.load(path)
    assert state.peak_equity == 0.0


# ------------------------------------------------------------------ sessions


def test_the_daily_baseline_resets_once_per_day():
    """The daily-loss breaker measures against this, so it must not drift."""
    state = BotState()
    state.begin_session(1400.0, date(2026, 3, 2))
    assert state.equity_at_session_start == 1400.0

    # A second run the same day keeps the morning's baseline.
    state.begin_session(1320.0, date(2026, 3, 2))
    assert state.equity_at_session_start == 1400.0

    # The next day rebases.
    state.begin_session(1320.0, date(2026, 3, 3))
    assert state.equity_at_session_start == 1320.0


def test_peak_equity_only_ratchets_upward():
    state = BotState()
    state.begin_session(1400.0, date(2026, 3, 2))
    state.begin_session(1800.0, date(2026, 3, 3))
    state.begin_session(1200.0, date(2026, 3, 4))

    assert state.peak_equity == 1800.0


def test_a_first_run_does_not_report_a_full_drawdown():
    """Starting the peak at zero would make the first run look catastrophic."""
    state = BotState()
    state.begin_session(1400.0, date(2026, 3, 2))
    assert state.peak_equity == 1400.0


def test_halts_are_recorded_and_cleared():
    state = BotState()
    state.record_halt("max_drawdown", date(2026, 3, 2))
    assert state.halted_reason == "max_drawdown"
    assert state.halted_on == "2026-03-02"

    state.clear_halt()
    assert state.halted_reason is None


# ------------------------------------------------------------- reconciliation


def test_a_position_gone_from_the_broker_is_dropped():
    """A stop that fired overnight must not linger as a phantom holding."""
    state = BotState()
    state.open_position("AAPL", 150.0, 140.0, today=date(2026, 3, 1))
    state.open_position("MSFT", 400.0, 380.0, today=date(2026, 3, 1))

    changes = state.reconcile({"MSFT"}, date(2026, 3, 5))

    assert set(state.positions) == {"MSFT"}
    assert any("AAPL" in c for c in changes)


def test_an_unknown_broker_position_is_adopted():
    state = BotState()
    changes = state.reconcile({"NVDA"}, date(2026, 3, 5))

    assert "NVDA" in state.positions
    assert any("NVDA" in c for c in changes)
    # Adopted with placeholder prices; the engine fills them from the broker.
    assert state.positions["NVDA"].entry_price == 0.0


def test_reconciling_an_accurate_state_changes_nothing():
    state = BotState()
    state.open_position("AAPL", 150.0, 140.0, today=date(2026, 3, 1))

    assert state.reconcile({"AAPL"}, date(2026, 3, 5)) == []
    assert set(state.positions) == {"AAPL"}


def test_holding_days_counts_from_the_open_date():
    position = PositionState(
        symbol="AAPL", opened_on="2026-03-01", entry_price=150.0,
        initial_stop=140.0, current_stop=140.0, highest_close=150.0,
    )
    assert position.holding_days(date(2026, 3, 13)) == 12
    # A clock skew must not produce a negative holding period.
    assert position.holding_days(date(2026, 2, 20)) == 0


def test_closing_a_position_is_idempotent():
    state = BotState()
    state.open_position("AAPL", 150.0, 140.0, today=date(2026, 3, 1))
    state.close_position("AAPL")
    state.close_position("AAPL")  # must not raise
    assert "AAPL" not in state.positions
