"""The trading engine: one scheduled run, start to finish.

Ordering in :meth:`Engine.run` is not arbitrary. Protecting what is already
held comes before managing it, which comes before opening anything new. If the
run dies partway through, it dies having done the most important work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import pandas as pd

from bot import risk, signals
from bot.broker.base import Broker, BrokerError, BrokerState, Position
from bot.config import Config, ExecutionMode, RiskOffAction
from bot.data import Bars, DataError, PriceRepository
from bot.risk import HaltReason, PortfolioState, Sizing, SizingRejection
from bot.signals import EntryCandidate, ExitReason, RegimeState, RejectReason
from bot.state import BotState
from bot.universe_list import get_universe

log = logging.getLogger(__name__)


@dataclass
class PlannedAction:
    """A decision the engine made, and what came of it."""

    kind: str
    symbol: str
    detail: str
    executed: bool = False
    error: str | None = None

    def render(self) -> str:
        if self.error:
            return f"[FAILED] {self.kind} {self.symbol}: {self.detail} — {self.error}"
        marker = "[DONE]" if self.executed else "[PLANNED]"
        return f"{marker} {self.kind} {self.symbol}: {self.detail}"


@dataclass
class RunReport:
    """Everything that happened in one run, for the log and the run summary."""

    started_utc: str
    mode: str
    equity: float = 0.0
    cash: float = 0.0
    open_positions: int = 0
    regime: RegimeState | None = None
    halt: HaltReason | None = None
    actions: list[PlannedAction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    universe_size: int = 0
    candidates_found: int = 0
    data_failures: dict[str, str] = field(default_factory=dict)
    rejections: dict[str, str] = field(default_factory=dict)

    def add(self, action: PlannedAction) -> PlannedAction:
        self.actions.append(action)
        return action

    def warn(self, message: str) -> None:
        log.warning(message)
        self.warnings.append(message)

    @property
    def executed_actions(self) -> list[PlannedAction]:
        return [a for a in self.actions if a.executed]

    def to_markdown(self) -> str:
        """Render for the GitHub Actions run summary."""
        lines = [
            f"# Trading bot run — {self.started_utc}",
            "",
            f"**Mode:** `{self.mode}`  ",
            f"**Equity:** ${self.equity:,.2f}  ",
            f"**Cash:** ${self.cash:,.2f}  ",
            f"**Open positions:** {self.open_positions}",
            "",
        ]

        if self.halt:
            lines += [f"## ⛔ Halted: {self.halt.value}", ""]
        if self.regime:
            lines += [f"**Regime:** {self.regime.describe()}", ""]

        lines += [
            "## Actions",
            "",
        ]
        if self.actions:
            lines += [f"- {a.render()}" for a in self.actions]
        else:
            lines.append("- No action taken.")

        lines += [
            "",
            "## Scan",
            "",
            f"- Universe screened: {self.universe_size}",
            f"- Candidates passing every filter: {self.candidates_found}",
        ]
        if self.data_failures:
            lines.append(f"- Symbols with no usable data: {len(self.data_failures)}")
        if self.warnings:
            lines += ["", "## Warnings", ""] + [f"- {w}" for w in self.warnings]
        return "\n".join(lines)


class Engine:
    """Runs the strategy once against a broker."""

    def __init__(
        self,
        cfg: Config,
        broker: Broker,
        repository: PriceRepository,
        state: BotState,
        today: date | None = None,
    ):
        self.cfg = cfg
        self.broker = broker
        self.repository = repository
        self.state = state
        self.today = today or datetime.now(UTC).date()

    # ------------------------------------------------------------------ run

    def run(self) -> RunReport:
        report = RunReport(
            started_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            mode=self.cfg.execution.mode.value,
        )

        healthy, message = self.broker.health_check()
        if not healthy:
            report.warn(f"Broker unavailable: {message}")
            return report
        log.info(message)

        broker_state = self._read_broker_state(report)
        if broker_state is None:
            return report

        report.equity = broker_state.summary.equity
        report.cash = broker_state.summary.cash
        report.open_positions = len(broker_state.positions)

        self.state.begin_session(broker_state.summary.equity, self.today)
        for change in self.state.reconcile({p.symbol for p in broker_state.positions}, self.today):
            report.warn(change)
        self._adopt_broker_costs(broker_state)

        # Prices are loaded first because every subsequent decision needs them,
        # including the ATR that sizes a repaired stop. Nothing here can place
        # an order, so it costs nothing in risk terms to do it before repair.
        held = sorted({p.symbol for p in broker_state.positions})
        bars, benchmark = self._load_data(held, report)

        # First action of the run: an open position without a stop is the only
        # situation that can produce an unbounded loss, so it is fixed before
        # anything else is considered.
        self._repair_unprotected(broker_state, bars, report)

        if benchmark is None:
            report.warn("No benchmark data; refusing to trade without a regime read.")
            return report

        staleness = benchmark.staleness_days(self.today)
        portfolio = self._portfolio_state(broker_state)

        halt = risk.check_halts(portfolio, self.cfg, staleness)
        if halt:
            report.halt = halt
            self.state.record_halt(halt.value, self.today)
            report.warn(self._describe_halt(halt, portfolio, staleness))
        elif self.state.halted_reason:
            self.state.clear_halt()
            log.info("Circuit breaker cleared; trading resumes.")

        try:
            benchmark_features = signals.compute_features(benchmark.frame, self.cfg.strategy)
            report.regime = signals.evaluate_regime(benchmark_features, self.cfg)
            log.info("Regime: %s", report.regime.describe())
        except ValueError as exc:
            report.warn(f"Cannot evaluate regime ({exc}); treating as risk-off.")
            report.regime = None

        # Step 2: manage what is already held. This runs even when halted —
        # a circuit breaker stops new risk, it does not abandon open trades.
        self._manage_positions(broker_state, bars, report)

        # Step 3: open new positions, only if everything is clear.
        if self._may_open_positions(report):
            self._open_positions(broker_state, report)

        self.state.save(_state_path(self.cfg))
        return report

    # -------------------------------------------------------- broker reading

    def _read_broker_state(self, report: RunReport) -> BrokerState | None:
        try:
            return BrokerState(
                summary=self.broker.get_account_summary(),
                positions=self.broker.get_positions(),
                open_orders=self.broker.get_open_orders(),
            )
        except BrokerError as exc:
            report.warn(f"Could not read account state: {exc}")
            return None

    def _adopt_broker_costs(self, broker_state: BrokerState) -> None:
        """Fill in placeholder prices for positions adopted from the broker."""
        for position in broker_state.positions:
            local = self.state.positions.get(position.symbol)
            if local and local.entry_price <= 0:
                local.entry_price = position.average_cost
                local.highest_close = max(position.average_cost, position.market_price)

    def _portfolio_state(self, broker_state: BrokerState) -> PortfolioState:
        summary = broker_state.summary
        return PortfolioState(
            equity=summary.equity,
            cash=summary.cash,
            settled_cash=summary.settled_cash,
            gross_exposure=broker_state.gross_exposure,
            open_position_count=len(broker_state.positions),
            peak_equity=max(self.state.peak_equity, summary.equity),
            equity_at_session_start=self.state.equity_at_session_start or summary.equity,
        )

    def _describe_halt(self, halt: HaltReason, portfolio: PortfolioState, staleness: int) -> str:
        cb = self.cfg.circuit_breakers
        if halt is HaltReason.MAX_DRAWDOWN:
            return (
                f"HALTED: drawdown {portfolio.drawdown_pct:.1f}% has reached the "
                f"{cb.max_drawdown_halt_pct:.0f}% limit. No new positions until "
                "circuit_breakers.manual_override is set to true. Existing stops "
                "remain live at the broker."
            )
        if halt is HaltReason.DAILY_LOSS:
            return (
                f"HALTED for today: down {portfolio.session_pnl_pct:.1f}% this "
                f"session against a {cb.daily_loss_halt_pct:.0f}% limit."
            )
        return (
            f"HALTED: price data is {staleness} days old, beyond the "
            f"{cb.max_data_staleness_days}-day limit. This usually means the data "
            "source is broken rather than the market being closed."
        )

    # --------------------------------------------------------- data loading

    def _load_data(
        self, held: list[str], report: RunReport
    ) -> tuple[dict[str, Bars], Bars | None]:
        universe = list(get_universe(self.cfg.universe.list_name))
        # Held symbols always get fetched, even if they have since dropped out
        # of the universe, or the bot could not decide when to exit them.
        symbols = list(dict.fromkeys(universe + held))
        report.universe_size = len(symbols)

        bars, failures = self.repository.get_many(symbols)
        report.data_failures = failures
        if failures:
            report.warn(f"No usable price data for {len(failures)} symbol(s).")

        missing_held = [s for s in held if s not in bars]
        if missing_held:
            report.warn(
                f"Held position(s) {missing_held} have no price data; they cannot "
                "be evaluated for exit this run. Their broker stops remain live."
            )

        try:
            benchmark = self.repository.get(self.cfg.regime.benchmark)
        except DataError as exc:
            report.warn(f"Benchmark {self.cfg.regime.benchmark} unavailable: {exc}")
            return bars, None

        return bars, benchmark

    # ------------------------------------------------------ position repair

    def _repair_unprotected(
        self, broker_state: BrokerState, bars: dict[str, Bars], report: RunReport
    ) -> None:
        """Attach a stop to any position that has none.

        Happens when an entry fills but its child stop is rejected, or when a
        position is opened manually. Either way the position is running with
        undefined risk until this fixes it.
        """
        for position in broker_state.unprotected_positions():
            stop = self._repair_stop_price(position, bars)
            action = report.add(
                PlannedAction(
                    kind="REPAIR-STOP",
                    symbol=position.symbol,
                    detail=f"{position.shares} shares unprotected; placing stop at {stop:.2f}",
                )
            )
            report.warn(
                f"{position.symbol}: position had no working stop order. Placing one now."
            )

            if not self.cfg.execution.mode.sends_orders:
                continue
            try:
                result = self.broker.place_protective_stop(
                    position.symbol, position.shares, stop
                )
                action.executed = result.accepted
                if result.accepted:
                    local = self.state.positions.get(position.symbol)
                    if local:
                        local.current_stop = stop
                        local.stop_order_id = result.stop_order_id
                else:
                    action.error = result.message
            except BrokerError as exc:
                action.error = str(exc)

    def _repair_stop_price(self, position: Position, bars: dict[str, Bars]) -> float:
        """Choose a stop for a position found without one.

        Preference order: the stop the bot remembers, then a fresh ATR-based
        stop, then a fixed percentage. The last is a crude fallback, but any
        stop beats leaving the position naked while waiting for better data.
        """
        local = self.state.positions.get(position.symbol)
        if local and 0 < local.current_stop < position.market_price:
            return local.current_stop

        symbol_bars = bars.get(position.symbol)
        if symbol_bars is not None:
            features = signals.compute_features(symbol_bars.frame, self.cfg.strategy)
            atr_value = features.iloc[-1].get("atr")
            if atr_value == atr_value and atr_value and atr_value > 0:  # not NaN
                return signals.initial_stop_price(
                    position.market_price, float(atr_value), self.cfg
                )

        return max(round(position.market_price * 0.90, 2), 0.01)

    # --------------------------------------------------- position management

    def _manage_positions(
        self, broker_state: BrokerState, bars: dict[str, Bars], report: RunReport
    ) -> None:
        flatten_all = (
            report.regime is not None
            and not report.regime.risk_on
            and self.cfg.regime.risk_off_action is RiskOffAction.FLATTEN
        )

        for position in broker_state.positions:
            symbol = position.symbol
            local = self.state.positions.get(symbol)
            symbol_bars = bars.get(symbol)

            if symbol_bars is None:
                continue

            features = signals.compute_features(symbol_bars.frame, self.cfg.strategy)
            holding_days = local.holding_days(self.today) if local else 0

            reason: ExitReason | None = None
            if flatten_all:
                reason = ExitReason.REGIME_FLATTEN
            else:
                reason = signals.should_exit(features, self.cfg, holding_days)

            if reason is not None:
                self._exit_position(position, reason, features, report)
                continue

            self._update_trailing_stop(position, broker_state, features, report)

    def _exit_position(
        self, position: Position, reason: ExitReason, features, report: RunReport
    ) -> None:
        price = float(features.iloc[-1]["close"])
        limit = risk.marketable_limit_price(price, "SELL", self.cfg)
        action = report.add(
            PlannedAction(
                kind="EXIT",
                symbol=position.symbol,
                detail=(
                    f"{position.shares} shares at ~{limit:.2f} ({reason.value}); "
                    f"P&L {position.unrealised_pnl_pct:+.1f}%"
                ),
            )
        )

        if not self.cfg.execution.mode.sends_orders:
            return
        try:
            result = self.broker.place_exit(
                position.symbol, position.shares, limit, reason.value
            )
            action.executed = result.accepted
            if result.accepted:
                self.state.close_position(position.symbol)
            else:
                action.error = result.message
        except BrokerError as exc:
            action.error = str(exc)

    def _update_trailing_stop(
        self, position: Position, broker_state: BrokerState, features, report: RunReport
    ) -> None:
        """Ratchet the resting stop upward as the trade moves in our favour."""
        local = self.state.positions.get(position.symbol)
        if local is None:
            return

        last = features.iloc[-1]
        close = float(last["close"])
        if pd.isna(last.get("atr")):
            return
        atr_value = float(last["atr"])
        if atr_value <= 0:
            return

        local.highest_close = max(local.highest_close, close)
        new_stop = signals.trailing_stop_price(
            local.highest_close, atr_value, self.cfg, local.current_stop
        )

        # Only act on a meaningful move. Rewriting the stop for a cent of
        # improvement burns API calls and risks a window with no protection.
        if new_stop <= local.current_stop * 1.002:
            return

        stop_order = broker_state.protective_stop_for(position.symbol)
        action = report.add(
            PlannedAction(
                kind="TRAIL-STOP",
                symbol=position.symbol,
                detail=f"raising stop {local.current_stop:.2f} → {new_stop:.2f}",
            )
        )

        if not self.cfg.execution.mode.sends_orders:
            local.current_stop = new_stop
            return

        if stop_order is None:
            action.error = "no working stop order to modify"
            return
        try:
            if self.broker.modify_stop(stop_order.order_id, new_stop):
                local.current_stop = new_stop
                local.stop_order_id = stop_order.order_id
                action.executed = True
            else:
                action.error = "broker rejected the modification"
        except BrokerError as exc:
            action.error = str(exc)

    # -------------------------------------------------------------- entries

    def _may_open_positions(self, report: RunReport) -> bool:
        if report.halt is not None:
            return False
        if report.regime is None:
            report.warn("No regime read available; not opening new positions.")
            return False
        if not report.regime.risk_on:
            log.info("Risk-off regime; no new entries. %s", report.regime.describe())
            return False
        return True

    def _open_positions(self, broker_state: BrokerState, report: RunReport) -> None:
        bars, _ = self.repository.get_many(list(get_universe(self.cfg.universe.list_name)))
        held = {p.symbol for p in broker_state.positions}

        candidates: list[EntryCandidate] = []
        rejections: dict[str, str] = {}

        for symbol, symbol_bars in bars.items():
            if symbol in held:
                continue
            features = signals.compute_features(symbol_bars.frame, self.cfg.strategy)
            outcome = signals.screen_entry(symbol, features, self.cfg)
            if isinstance(outcome, RejectReason):
                rejections[symbol] = outcome.value
            else:
                candidates.append(outcome)

        report.candidates_found = len(candidates)
        report.rejections = rejections
        if not candidates:
            log.info("No symbol passed every entry filter.")
            return

        portfolio = self._portfolio_state(broker_state)
        slots = self.cfg.risk.max_concurrent_positions - portfolio.open_position_count

        for candidate in signals.rank_candidates(candidates):
            if slots <= 0:
                break
            sizing = self._size(candidate, portfolio)
            if isinstance(sizing, SizingRejection):
                log.debug("%s not sized: %s", candidate.symbol, sizing.explain())
                continue
            if self._enter(sizing, candidate, report):
                slots -= 1
                portfolio = _after_entry(portfolio, sizing)

    def _size(self, candidate: EntryCandidate, portfolio: PortfolioState) -> Sizing | SizingRejection:
        stop = signals.initial_stop_price(candidate.price, candidate.atr, self.cfg)
        return risk.size_position(
            symbol=candidate.symbol,
            entry_price=candidate.price,
            stop_price=stop,
            adv=candidate.adv,
            state=portfolio,
            cfg=self.cfg,
        )

    def _enter(self, sizing: Sizing, candidate: EntryCandidate, report: RunReport) -> bool:
        limit = risk.marketable_limit_price(sizing.entry_price, "BUY", self.cfg)
        action = report.add(
            PlannedAction(
                kind="ENTRY",
                symbol=sizing.symbol,
                detail=(
                    f"{sizing.shares} shares at ~{limit:.2f} "
                    f"(${sizing.notional:.0f}, stop {sizing.stop_price:.2f}, "
                    f"risking ${sizing.risk_dollars:.2f} = "
                    f"{sizing.risk_pct_of(report.equity):.1f}% of equity; "
                    f"RSI {candidate.rsi:.0f}, momentum {candidate.momentum:+.0f}%)"
                ),
            )
        )

        if not self.cfg.execution.mode.sends_orders:
            return True

        try:
            result = self.broker.place_entry_with_stop(
                sizing.symbol, sizing.shares, limit, sizing.stop_price
            )
        except BrokerError as exc:
            action.error = str(exc)
            return False

        action.executed = result.accepted
        if not result.accepted:
            action.error = result.message
            return False

        self.state.open_position(
            symbol=sizing.symbol,
            entry_price=limit,
            stop=sizing.stop_price,
            stop_order_id=result.stop_order_id,
            today=self.today,
        )
        return True


def _after_entry(portfolio: PortfolioState, sizing: Sizing) -> PortfolioState:
    """Portfolio state as it will be once an entry fills.

    Applied between entries in the same run so the second trade is sized
    against the cash the first one just consumed, rather than against a stale
    snapshot that would let the bot overcommit.
    """
    return PortfolioState(
        equity=portfolio.equity,
        cash=portfolio.cash - sizing.notional,
        settled_cash=portfolio.settled_cash - sizing.notional,
        gross_exposure=portfolio.gross_exposure + sizing.notional,
        open_position_count=portfolio.open_position_count + 1,
        peak_equity=portfolio.peak_equity,
        equity_at_session_start=portfolio.equity_at_session_start,
    )


def _state_path(cfg: Config):
    from bot.config import REPO_ROOT

    return REPO_ROOT / "state" / "portfolio.json"


def build_broker(cfg: Config):
    """Instantiate the broker implied by the execution mode."""
    from pathlib import Path

    from bot.broker.ibkr import IbkrBroker
    from bot.broker.paper import PaperBroker
    from bot.config import REPO_ROOT

    if cfg.execution.mode is ExecutionMode.DRY_RUN:
        return PaperBroker(
            starting_cash=cfg.account.assumed_equity,
            costs=cfg.costs,
            state_path=Path(REPO_ROOT / "state" / "paper_account.json"),
        )
    return IbkrBroker()
