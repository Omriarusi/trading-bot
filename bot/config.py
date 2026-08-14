"""Typed configuration loading and validation.

The config file is the only place risk parameters are defined. This module
turns it into frozen dataclasses and refuses to build a config that would let
the bot do something reckless, so a bad edit fails at startup rather than
halfway through a trading session.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class ConfigError(ValueError):
    """Raised when the config file is missing, malformed, or unsafe."""


class ExecutionMode(StrEnum):
    DRY_RUN = "dry_run"
    PAPER = "paper"
    LIVE = "live"

    @property
    def sends_orders(self) -> bool:
        """Whether this mode actually transmits orders to a broker."""
        return self in (ExecutionMode.PAPER, ExecutionMode.LIVE)

    @property
    def risks_real_money(self) -> bool:
        return self is ExecutionMode.LIVE


class AccountType(StrEnum):
    CASH = "cash"
    MARGIN = "margin"


class RiskOffAction(StrEnum):
    HOLD = "hold"
    FLATTEN = "flatten"


class Ranking(StrEnum):
    MOST_OVERSOLD = "most_oversold"
    MOMENTUM = "momentum"


@dataclass(frozen=True)
class AccountConfig:
    entity: str = "IBIE"
    type: AccountType = AccountType.CASH
    base_currency: str = "USD"
    assumed_equity: float = 1400.0

    @property
    def priips_restricted(self) -> bool:
        """True for EEA/UK entities, where US-domiciled ETFs are blocked.

        IB LLC (US) clients can trade ETFs; IBIE/IBCE/IBUK clients cannot,
        so the universe must be restricted to single stocks.
        """
        return self.entity.upper() in {"IBIE", "IBCE", "IBUK", "IBLUX", "IBHU"}

    @property
    def can_short(self) -> bool:
        return self.type is AccountType.MARGIN


@dataclass(frozen=True)
class ExecutionConfig:
    mode: ExecutionMode = ExecutionMode.DRY_RUN
    max_order_notional: float = 500.0
    marketable_limit_offset_pct: float = 0.30
    max_slippage_pct: float = 1.5


@dataclass(frozen=True)
class CostsConfig:
    commission_per_share: float = 0.0035
    commission_min: float = 0.35
    commission_max_pct_of_trade: float = 1.0
    slippage_bps: float = 5.0

    def commission(self, shares: int, price: float) -> float:
        """IBKR tiered US stock commission for one order."""
        if shares <= 0:
            return 0.0
        raw = shares * self.commission_per_share
        capped = min(
            max(raw, self.commission_min),
            shares * price * self.commission_max_pct_of_trade / 100.0,
        )
        # The percentage cap can fall below the stated minimum on very small
        # trades; IBKR charges the lower of the two, so the cap wins.
        return round(capped, 4)


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_pct: float = 2.0
    max_position_pct: float = 30.0
    min_position_notional: float = 250.0
    max_concurrent_positions: int = 4
    max_gross_exposure_pct: float = 95.0
    atr_stop_multiple: float = 2.5
    atr_trail_multiple: float = 3.0
    max_holding_days: int = 12


@dataclass(frozen=True)
class CircuitBreakerConfig:
    daily_loss_halt_pct: float = 6.0
    max_drawdown_halt_pct: float = 25.0
    max_data_staleness_days: int = 5
    manual_override: bool = False


@dataclass(frozen=True)
class RegimeConfig:
    benchmark: str = "SPY"
    trend_ma_days: int = 200
    vol_lookback_days: int = 10
    vol_percentile_block: float = 90.0
    risk_off_action: RiskOffAction = RiskOffAction.HOLD


@dataclass(frozen=True)
class StrategyConfig:
    name: str = "trend_pullback"
    trend_ma_days: int = 200
    momentum_lookback_days: int = 126
    min_momentum_pct: float = 0.0
    rsi_period: int = 3
    rsi_entry_max: float = 25.0
    exit_ma_days: int = 10
    rsi_exit_min: float = 70.0
    ranking: Ranking = Ranking.MOST_OVERSOLD


@dataclass(frozen=True)
class UniverseConfig:
    min_price: float = 5.0
    max_price: float = 400.0
    min_avg_dollar_volume: float = 50_000_000
    adv_lookback_days: int = 20
    max_pct_of_adv: float = 1.0
    earnings_blackout_days: int = 3
    list_name: str = "liquid_us_large_cap"


@dataclass(frozen=True)
class DataConfig:
    sources: tuple[str, ...] = ("stooq", "yahoo")
    history_days: int = 400
    cache_dir: str = ".cache/prices"


@dataclass(frozen=True)
class NotificationsConfig:
    write_run_summary: bool = True
    log_dir: str = "state/logs"


@dataclass(frozen=True)
class Config:
    account: AccountConfig = field(default_factory=AccountConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    circuit_breakers: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)


T = TypeVar("T")


def _build(cls: type[T], raw: Mapping[str, Any] | None, path: str) -> T:
    """Instantiate a dataclass from a mapping, coercing and rejecting typos.

    An unknown key is an error rather than a silent no-op: a misspelled
    ``risk_per_trade_pct`` would otherwise leave the default in place and
    trade a size the user never chose.
    """
    if raw is None:
        return cls()  # type: ignore[call-arg]
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path}: expected a mapping, got {type(raw).__name__}")

    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - set(known)
    if unknown:
        raise ConfigError(
            f"{path}: unknown option(s) {sorted(unknown)}. "
            f"Valid options: {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        kwargs[name] = _coerce(known[name].type, value, f"{path}.{name}")
    return cls(**kwargs)  # type: ignore[call-arg]


def _coerce(annotation: Any, value: Any, path: str) -> Any:
    """Convert a YAML scalar to the annotated type, with clear errors."""
    if isinstance(annotation, str):
        # ``from __future__ import annotations`` makes every annotation a
        # string; resolve it against this module's namespace.
        annotation = eval(annotation, globals())

    origin = get_origin(annotation)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list")
        (item_type, _) = get_args(annotation)
        return tuple(_coerce(item_type, v, f"{path}[]") for v in value)

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value)
        except ValueError:
            allowed = [e.value for e in annotation]
            raise ConfigError(f"{path}: {value!r} is not one of {allowed}") from None

    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value

    if annotation in (int, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return annotation(value)

    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value

    if is_dataclass(annotation):
        return _build(annotation, value, path)

    return value


def _validate(cfg: Config) -> None:
    """Reject configurations that are internally inconsistent or unsafe.

    These are not style checks. Each one corresponds to a way the bot could
    lose money that no amount of strategy quality would compensate for.
    """
    errors: list[str] = []
    r, cb, u, s, x = (
        cfg.risk,
        cfg.circuit_breakers,
        cfg.universe,
        cfg.strategy,
        cfg.execution,
    )

    if not 0 < r.risk_per_trade_pct <= 10:
        errors.append(
            f"risk.risk_per_trade_pct must be in (0, 10], got {r.risk_per_trade_pct}. "
            "Above 10% of equity on a single stop-out, a normal losing streak "
            "is an account-ending event."
        )
    if not 0 < r.max_position_pct <= 100:
        errors.append("risk.max_position_pct must be in (0, 100]")
    if r.max_concurrent_positions < 1:
        errors.append("risk.max_concurrent_positions must be at least 1")
    if r.atr_stop_multiple <= 0:
        errors.append("risk.atr_stop_multiple must be positive; a stop is mandatory")
    if r.atr_trail_multiple < r.atr_stop_multiple:
        errors.append(
            f"risk.atr_trail_multiple ({r.atr_trail_multiple}) is tighter than "
            f"atr_stop_multiple ({r.atr_stop_multiple}); the trailing stop would "
            "immediately tighten below the initial stop and cut winners short."
        )
    if r.max_holding_days < 1:
        errors.append("risk.max_holding_days must be at least 1")

    # A position at the size cap, stopped out, must not breach the daily halt,
    # otherwise a single ordinary loss halts the bot for the day.
    worst_single_loss = r.max_position_pct * r.risk_per_trade_pct / 100.0
    if worst_single_loss > cb.daily_loss_halt_pct:
        errors.append(
            f"circuit_breakers.daily_loss_halt_pct ({cb.daily_loss_halt_pct}%) is "
            f"below the loss from a single stopped-out position "
            f"(~{worst_single_loss:.1f}%); the bot would halt on a routine loss."
        )

    # All positions stopping out at once must stay inside the drawdown halt.
    worst_day = r.risk_per_trade_pct * r.max_concurrent_positions
    if worst_day > cb.max_drawdown_halt_pct:
        errors.append(
            f"{r.max_concurrent_positions} positions at "
            f"{r.risk_per_trade_pct}% risk each can lose {worst_day:.0f}% in one "
            f"day, which exceeds max_drawdown_halt_pct "
            f"({cb.max_drawdown_halt_pct}%). Reduce risk per trade or positions."
        )

    if r.min_position_notional * r.max_concurrent_positions > cfg.account.assumed_equity:
        errors.append(
            f"{r.max_concurrent_positions} positions of at least "
            f"${r.min_position_notional:.0f} need "
            f"${r.min_position_notional * r.max_concurrent_positions:.0f}, but the "
            f"account has ${cfg.account.assumed_equity:.0f}. The bot could never "
            "fill its target position count."
        )

    if x.max_order_notional < r.min_position_notional:
        errors.append(
            f"execution.max_order_notional (${x.max_order_notional:.0f}) is below "
            f"risk.min_position_notional (${r.min_position_notional:.0f}); every "
            "order would be rejected."
        )

    if r.max_gross_exposure_pct > 100 and cfg.account.type is AccountType.CASH:
        errors.append(
            "risk.max_gross_exposure_pct above 100% requires margin, but "
            "account.type is 'cash'."
        )

    if not 0 < s.rsi_entry_max < 100:
        errors.append("strategy.rsi_entry_max must be in (0, 100)")
    if s.rsi_exit_min <= s.rsi_entry_max:
        errors.append(
            f"strategy.rsi_exit_min ({s.rsi_exit_min}) must exceed rsi_entry_max "
            f"({s.rsi_entry_max}); otherwise entries exit on the same bar."
        )
    if s.rsi_period < 2:
        errors.append("strategy.rsi_period must be at least 2")
    if s.trend_ma_days < 2:
        errors.append("strategy.trend_ma_days must be at least 2")

    if u.min_price >= u.max_price:
        errors.append("universe.min_price must be below universe.max_price")
    if not 0 < u.max_pct_of_adv <= 20:
        errors.append("universe.max_pct_of_adv must be in (0, 20]")

    # Indicators need enough history to warm up before the first signal.
    warmup = max(s.trend_ma_days, s.momentum_lookback_days, cfg.regime.trend_ma_days)
    # Calendar days to trading days is roughly 252/365.
    trading_days = int(cfg.data.history_days * 252 / 365)
    if trading_days < warmup + 20:
        errors.append(
            f"data.history_days ({cfg.data.history_days} calendar days "
            f"~= {trading_days} trading days) is too short to warm up a "
            f"{warmup}-day indicator. Use at least "
            f"{int((warmup + 40) * 365 / 252)}."
        )

    if errors:
        raise ConfigError(
            "Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def load(path: str | Path | None = None) -> Config:
    """Load, coerce, and validate the config file.

    Environment overrides are applied for the two settings that must be
    controllable from CI without editing a committed file.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")

    with cfg_path.open() as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, Mapping):
        raise ConfigError(f"{cfg_path}: top level must be a mapping")

    cfg = _build(Config, raw, "config")

    # Execution mode is overridable so the same commit can run dry in CI and
    # paper on a schedule without a config edit racing the scheduler.
    mode_override = os.environ.get("BOT_EXECUTION_MODE")
    if mode_override:
        try:
            mode = ExecutionMode(mode_override.strip().lower())
        except ValueError:
            raise ConfigError(
                f"BOT_EXECUTION_MODE={mode_override!r} is not one of "
                f"{[m.value for m in ExecutionMode]}"
            ) from None
        cfg = _replace_execution(cfg, mode)

    if os.environ.get("BOT_MANUAL_OVERRIDE", "").strip().lower() in {"1", "true", "yes"}:
        cfg = _replace_override(cfg, True)

    _validate(cfg)
    return cfg


def _replace_execution(cfg: Config, mode: ExecutionMode) -> Config:
    from dataclasses import replace

    return replace(cfg, execution=replace(cfg.execution, mode=mode))


def _replace_override(cfg: Config, value: bool) -> Config:
    from dataclasses import replace

    return replace(
        cfg, circuit_breakers=replace(cfg.circuit_breakers, manual_override=value)
    )
