"""End-to-end engine behaviour against a simulated broker.

These tests are about the *order of operations* and the guard rails, not about
whether the strategy makes money. They assert the properties that must hold
however the market moves.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from bot.broker.paper import PaperBroker
from bot.config import Config, ExecutionMode
from bot.data import Bars
from bot.engine import Engine
from bot.state import BotState


class FakeRepository:
    """Serves pre-built bars, so no test touches the network."""

    def __init__(self, frames: dict[str, object], benchmark_symbol: str = "SPY"):
        self._frames = frames
        self._benchmark = benchmark_symbol
        self.requested: list[str] = []

    def get(self, symbol: str, use_cache: bool = True) -> Bars:
        self.requested.append(symbol)
        frame = self._frames.get(symbol)
        if frame is None:
            from bot.data import DataError

            raise DataError(f"{symbol}: not in fixture")
        return Bars(symbol=symbol, frame=frame, source="fake")

    def get_many(self, symbols: list[str], use_cache: bool = True):
        ok, failed = {}, {}
        for symbol in symbols:
            try:
                ok[symbol] = self.get(symbol)
            except Exception as exc:
                failed[symbol] = str(exc)
        return ok, failed


@pytest.fixture
def engine_cfg(cfg: Config) -> Config:
    """Dry-run config whose universe matches the fixtures."""
    return replace(
        cfg,
        universe=replace(cfg.universe, list_name="test_universe", min_avg_dollar_volume=1_000_000),
        execution=replace(cfg.execution, mode=ExecutionMode.PAPER),
    )


@pytest.fixture(autouse=True)
def register_test_universe(monkeypatch):
    """Point the universe lookup at the fixture symbols."""
    from bot import universe_list

    monkeypatch.setitem(universe_list.UNIVERSES, "test_universe", ("GOOD", "BAD"))


@pytest.fixture
def frames(pullback_bars, downtrend_bars, risk_on_benchmark):
    return {"GOOD": pullback_bars, "BAD": downtrend_bars, "SPY": risk_on_benchmark}


def build_engine(cfg, frames, broker=None, state=None, today=None):
    broker = broker or PaperBroker(starting_cash=cfg.account.assumed_equity, costs=cfg.costs)
    return Engine(
        cfg=cfg,
        broker=broker,
        repository=FakeRepository(frames),
        state=state or BotState(),
        today=today or date(2021, 7, 1),
    ), broker


# ------------------------------------------------------------------- entries


def test_a_valid_setup_produces_an_entry_with_a_stop(engine_cfg, frames, tmp_path, monkeypatch):
    monkeypatch.setattr("bot.engine._state_path", lambda cfg: tmp_path / "state.json")
    engine, broker = build_engine(engine_cfg, frames)
    report = engine.run()

    entries = [a for a in report.actions if a.kind == "ENTRY"]
    assert entries, f"expected an entry; got {[a.render() for a in report.actions]}"
    assert entries[0].symbol == "GOOD"
    assert entries[0].executed

    # Every position must carry a resting stop the moment it exists.
    positions = broker.get_positions()
    assert len(positions) == 1
    stops = [o for o in broker.get_open_orders() if o.is_protective_stop]
    assert len(stops) == 1
    assert stops[0].symbol == "GOOD"
    assert stops[0].stop_price < positions[0].average_cost


def test_the_downtrending_symbol_is_never_bought(engine_cfg, frames, tmp_path, monkeypatch):
    monkeypatch.setattr("bot.engine._state_path", lambda cfg: tmp_path / "state.json")
    engine, broker = build_engine(engine_cfg, frames)
    engine.run()

    assert all(p.symbol != "BAD" for p in broker.get_positions())


def test_dry_run_places_no_orders(cfg, frames, tmp_path, monkeypatch):
    monkeypatch.setattr("bot.engine._state_path", lambda cfg: tmp_path / "state.json")
    dry = replace(
        cfg,
        universe=replace(cfg.universe, list_name="test_universe", min_avg_dollar_volume=1_000_000),
        execution=replace(cfg.execution, mode=ExecutionMode.DRY_RUN),
    )
    engine, broker = build_engine(dry, frames)
    report = engine.run()

    assert any(a.kind == "ENTRY" for a in report.actions), "should still plan an entry"
    assert not broker.get_positions(), "dry run must not touch the account"


def test_risk_off_regime_blocks_new_entries(engine_cfg, frames, risk_off_benchmark, tmp_path, monkeypatch):
    monkeypatch.setattr("bot.engine._state_path", lambda cfg: tmp_path / "state.json")
    frames = {**frames, "SPY": risk_off_benchmark}
    engine, broker = build_engine(engine_cfg, frames)
    report = engine.run()

    assert report.regime is not None and not report.regime.risk_on
    assert not [a for a in report.actions if a.kind == "ENTRY"]
    assert not broker.get_positions()


def test_position_cap_is_respected(engine_cfg, frames, tmp_path, monkeypatch):
    monkeypatch.setattr("bot.engine._state_path", lambda cfg: tmp_path / "state.json")
    capped = replace(engine_cfg, risk=replace(engine_cfg.risk, max_concurrent_positions=1))
    engine, broker = build_engine(capped, frames)
    engine.run()

    assert len(broker.get_positions()) <= 1


def test_entries_stop_when_cash_runs_out(engine_cfg, frames, tmp_path, monkeypatch):
    """Sizing must account for cash already committed earlier in the run."""
    monkeypatch.setattr("bot.engine._state_path", lambda cfg: tmp_path / "state.json")
    broker = PaperBroker(starting_cash=300.0, costs=engine_cfg.costs)
    engine, _ = build_engine(engine_cfg, frames, broker=broker)
    engine.run()

    assert broker.cash >= 0, "the bot must never overdraw the account"


# ------------------------------------------------------- circuit breakers


def test_drawdown_halt_blocks_entries(engine_cfg, frames, tmp_path, monkeypatch):
    monkeypatch.setattr("bot.engine._state_path", lambda cfg: tmp_path / "state.json")
    state = BotState(peak_equity=5000.0)  # against $1,400 equity: -72%
    engine, broker = build_engine(engine_cfg, frames, state=state)
    report = engine.run()

    assert report.halt is not None
    assert not [a for a in report.actions if a.kind == "ENTRY"]
    assert not broker.get_positions()


def test_stale_data_halts_trading(engine_cfg, frames, tmp_path, monkeypatch):
    """A data source stuck in the past must not be traded on."""
    monkeypatch.setattr("bot.engine._state_path", lambda cfg: tmp_path / "state.json")
    engine, broker = build_engine(engine_cfg, frames, today=date(2030, 1, 1))
    report = engine.run()

    assert report.halt is not None
    assert not broker.get_positions()


def test_halt_does_not_cancel_protection_on_open_positions(engine_cfg, frames, tmp_path, monkeypatch):
    """A circuit breaker stops new risk; it must not abandon open trades."""
    monkeypatch.setattr("bot.engine._state_path", lambda cfg: tmp_path / "state.json")
    broker = PaperBroker(starting_cash=1400.0, costs=engine_cfg.costs)
    broker.place_entry_with_stop("GOOD", 2, 140.0, 130.0)
    stops_before = [o for o in broker.get_open_orders() if o.is_protective_stop]

    state = BotState(peak_equity=100_000.0)
    engine, _ = build_engine(engine_cfg, frames, broker=broker, state=state)
    report = engine.run()

    assert report.halt is not None
    stops_after = [o for o in broker.get_open_orders() if o.is_protective_stop]
    assert len(stops_after) >= len(stops_before) - 1  # only an exit may remove one


# ------------------------------------------------------------ stop repair


def test_an_unprotected_position_gets_a_stop(engine_cfg, frames, tmp_path, monkeypatch):
    """The single most important repair the engine performs."""
    monkeypatch.setattr("bot.engine._state_path", lambda cfg: tmp_path / "state.json")
    broker = PaperBroker(starting_cash=1400.0, costs=engine_cfg.costs)
    broker.place_entry_with_stop("GOOD", 2, 140.0, 130.0)
    # Simulate the child stop having been rejected or cancelled.
    for order in broker.get_open_orders():
        if order.is_protective_stop:
            broker.cancel_order(order.order_id)
    assert not [o for o in broker.get_open_orders() if o.is_protective_stop]

    engine, _ = build_engine(engine_cfg, frames, broker=broker)
    report = engine.run()

    repairs = [a for a in report.actions if a.kind == "REPAIR-STOP"]
    assert repairs and repairs[0].symbol == "GOOD"
    assert repairs[0].executed
    assert [o for o in broker.get_open_orders() if o.is_protective_stop]


def test_repair_uses_an_atr_based_stop_below_the_market(engine_cfg, frames, tmp_path, monkeypatch):
    monkeypatch.setattr("bot.engine._state_path", lambda cfg: tmp_path / "state.json")
    broker = PaperBroker(starting_cash=1400.0, costs=engine_cfg.costs)
    broker.place_entry_with_stop("GOOD", 2, 140.0, 130.0)
    for order in broker.get_open_orders():
        if order.is_protective_stop:
            broker.cancel_order(order.order_id)

    engine, _ = build_engine(engine_cfg, frames, broker=broker, state=BotState())
    engine.run()

    stop = next(o for o in broker.get_open_orders() if o.is_protective_stop)
    position = next(p for p in broker.get_positions() if p.symbol == "GOOD")
    assert 0 < stop.stop_price < position.market_price


# --------------------------------------------------------------- reconciling


def test_a_position_closed_at_the_broker_is_forgotten(engine_cfg, frames, tmp_path, monkeypatch):
    """A stop that fired while the bot was offline must not linger in state."""
    monkeypatch.setattr("bot.engine._state_path", lambda cfg: tmp_path / "state.json")
    state = BotState()
    state.open_position("GONE", entry_price=100.0, stop=90.0, today=date(2021, 6, 1))

    engine, _ = build_engine(engine_cfg, frames, state=state)
    report = engine.run()

    assert "GONE" not in state.positions
    assert any("GONE" in w for w in report.warnings)


def test_a_manually_opened_position_is_adopted(engine_cfg, frames, tmp_path, monkeypatch):
    monkeypatch.setattr("bot.engine._state_path", lambda cfg: tmp_path / "state.json")
    broker = PaperBroker(starting_cash=1400.0, costs=engine_cfg.costs)
    broker.place_entry_with_stop("GOOD", 2, 140.0, 130.0)

    state = BotState()
    engine, _ = build_engine(engine_cfg, frames, broker=broker, state=state)
    report = engine.run()

    assert "GOOD" in state.positions or not broker.get_positions()
    assert any("GOOD" in w for w in report.warnings)


# ---------------------------------------------------------------- resilience


def test_a_broker_outage_ends_the_run_without_crashing(engine_cfg, frames, tmp_path, monkeypatch):
    monkeypatch.setattr("bot.engine._state_path", lambda cfg: tmp_path / "state.json")

    class DeadBroker(PaperBroker):
        name = "dead"

        def get_account_summary(self):
            from bot.broker.base import BrokerError

            raise BrokerError("connection refused")

    engine, _ = build_engine(engine_cfg, frames, broker=DeadBroker())
    report = engine.run()

    assert report.warnings
    assert not [a for a in report.actions if a.executed]


def test_a_missing_benchmark_prevents_trading(engine_cfg, frames, tmp_path, monkeypatch):
    """No regime read means no basis for a trade."""
    monkeypatch.setattr("bot.engine._state_path", lambda cfg: tmp_path / "state.json")
    without_benchmark = {k: v for k, v in frames.items() if k != "SPY"}
    engine, broker = build_engine(engine_cfg, without_benchmark)
    report = engine.run()

    assert not broker.get_positions()
    assert any("enchmark" in w for w in report.warnings)


def test_a_symbol_with_no_data_does_not_stop_the_run(engine_cfg, frames, tmp_path, monkeypatch):
    monkeypatch.setattr("bot.engine._state_path", lambda cfg: tmp_path / "state.json")
    partial = {k: v for k, v in frames.items() if k != "BAD"}
    engine, _broker = build_engine(engine_cfg, partial)
    report = engine.run()

    assert report.data_failures  # BAD is missing
    assert [a for a in report.actions if a.kind == "ENTRY"]  # GOOD still trades


def test_the_report_renders(engine_cfg, frames, tmp_path, monkeypatch):
    monkeypatch.setattr("bot.engine._state_path", lambda cfg: tmp_path / "state.json")
    engine, _ = build_engine(engine_cfg, frames)
    markdown = engine.run().to_markdown()

    assert "# Trading bot run" in markdown
    assert "Equity" in markdown


def test_state_survives_a_save_and_load_cycle(engine_cfg, frames, tmp_path, monkeypatch):
    """A fresh container must pick up exactly where the last run left off."""
    path = tmp_path / "state.json"
    monkeypatch.setattr("bot.engine._state_path", lambda cfg: path)

    broker = PaperBroker(
        starting_cash=1400.0, costs=engine_cfg.costs, state_path=tmp_path / "acct.json"
    )
    engine, _ = build_engine(engine_cfg, frames, broker=broker)
    engine.run()

    reloaded = BotState.load(path)
    assert reloaded.run_count >= 1
    assert reloaded.peak_equity > 0
    assert set(reloaded.positions) == {p.symbol for p in broker.get_positions()}
