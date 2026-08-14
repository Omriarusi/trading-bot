"""Persistent state that the broker cannot tell us.

The broker is the source of truth for positions, orders, and cash. A handful
of facts are not derivable from it — when a position was opened, the equity
high-water mark, whether a circuit breaker has tripped — and those live here,
committed back to the repository after each run so the next scheduled run on a
fresh container picks up where this one left off.

Anything that *can* be read from the broker deliberately is not stored, so the
two can never disagree.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

log = logging.getLogger(__name__)

STATE_VERSION = 1


@dataclass
class PositionState:
    """Bookkeeping for one open position."""

    symbol: str
    opened_on: str
    entry_price: float
    initial_stop: float
    current_stop: float
    highest_close: float
    stop_order_id: str | None = None

    @property
    def opened_date(self) -> date:
        return date.fromisoformat(self.opened_on)

    def holding_days(self, as_of: date | None = None) -> int:
        """Calendar days held.

        Calendar rather than trading days: it is the figure that survives a
        missed run, and the time stop is a coarse control where the
        difference does not matter.
        """
        reference = as_of or datetime.now(UTC).date()
        return max(0, (reference - self.opened_date).days)


@dataclass
class BotState:
    version: int = STATE_VERSION
    peak_equity: float = 0.0
    last_run_utc: str | None = None
    last_session_date: str | None = None
    equity_at_session_start: float = 0.0
    halted_reason: str | None = None
    halted_on: str | None = None
    positions: dict[str, PositionState] = field(default_factory=dict)
    run_count: int = 0

    # ------------------------------------------------------------ lifecycle

    def begin_session(self, equity: float, today: date | None = None) -> None:
        """Roll the daily baseline forward if this is the first run of a day.

        The daily-loss circuit breaker measures against this baseline, so it
        must reset once per session and not on every run.
        """
        today = today or datetime.now(UTC).date()
        if self.last_session_date != today.isoformat():
            self.last_session_date = today.isoformat()
            self.equity_at_session_start = equity

        # A first-ever run has no history, so the peak starts at today's value
        # rather than zero, which would report a 100% drawdown.
        self.peak_equity = max(self.peak_equity, equity)
        self.run_count += 1
        self.last_run_utc = datetime.now(UTC).isoformat(timespec="seconds")

    def record_halt(self, reason: str, today: date | None = None) -> None:
        self.halted_reason = reason
        self.halted_on = (today or datetime.now(UTC).date()).isoformat()

    def clear_halt(self) -> None:
        self.halted_reason = None
        self.halted_on = None

    # ------------------------------------------------------------ positions

    def open_position(
        self, symbol: str, entry_price: float, stop: float,
        stop_order_id: str | None = None, today: date | None = None,
    ) -> None:
        self.positions[symbol] = PositionState(
            symbol=symbol,
            opened_on=(today or datetime.now(UTC).date()).isoformat(),
            entry_price=entry_price,
            initial_stop=stop,
            current_stop=stop,
            highest_close=entry_price,
            stop_order_id=stop_order_id,
        )

    def close_position(self, symbol: str) -> None:
        self.positions.pop(symbol, None)

    def reconcile(self, broker_symbols: set[str], today: date | None = None) -> list[str]:
        """Align local bookkeeping with what the broker actually holds.

        Two kinds of drift are possible and both are handled here: a position
        the broker no longer has (a stop fired while the bot was not running)
        and one the bot has never seen (a manual trade). Returns a
        human-readable list of what changed, for the run log.
        """
        changes: list[str] = []

        for symbol in set(self.positions) - broker_symbols:
            state = self.positions.pop(symbol)
            changes.append(
                f"{symbol}: closed at the broker while the bot was not running "
                f"(opened {state.opened_on}, stop was {state.current_stop:.2f})"
            )

        for symbol in broker_symbols - set(self.positions):
            changes.append(
                f"{symbol}: held at the broker but unknown to the bot; adopting it"
            )
            # Adopted with placeholder prices; the engine overwrites them from
            # the broker's actual cost basis immediately after reconciling.
            self.positions[symbol] = PositionState(
                symbol=symbol,
                opened_on=(today or datetime.now(UTC).date()).isoformat(),
                entry_price=0.0, initial_stop=0.0, current_stop=0.0, highest_close=0.0,
            )

        return changes

    # ----------------------------------------------------------- persistence

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["positions"] = {k: asdict(v) for k, v in self.positions.items()}
        # Sorted keys keep the committed diff readable run to run.
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> BotState:
        """Read state, falling back to a fresh instance on any problem.

        A corrupt state file must not stop the bot from managing live
        positions: the broker still holds the real picture, and losing the
        bookkeeping costs a trailing-stop update, not a position.
        """
        if not path.exists():
            return cls()
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            log.error("state file %s is unreadable (%s); starting fresh", path, exc)
            return cls()

        if payload.get("version") != STATE_VERSION:
            log.warning(
                "state file version %s does not match %s; starting fresh",
                payload.get("version"), STATE_VERSION,
            )
            return cls()

        positions = {
            symbol: PositionState(**row)
            for symbol, row in (payload.get("positions") or {}).items()
        }
        known = {f for f in cls.__dataclass_fields__ if f != "positions"}
        return cls(positions=positions, **{k: v for k, v in payload.items() if k in known})
