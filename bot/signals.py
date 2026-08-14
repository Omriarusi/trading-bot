"""Strategy logic: turn price history into entry and exit decisions.

The same functions serve the backtester and the live engine. That is
deliberate — if the two used different code paths, the backtest would be
measuring something other than what actually trades, which is the most common
way a "profitable" strategy turns out to lose money.

Convention throughout: a signal computed from the bar at index ``t`` is acted
on at ``t+1``. Nothing here reads beyond ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from bot import indicators
from bot.config import Config, Ranking, StrategyConfig

# Column names produced by compute_features.
FEATURE_COLUMNS = (
    "sma_trend",
    "sma_exit",
    "rsi",
    "momentum",
    "atr",
    "adv",
)


class ExitReason(StrEnum):
    TARGET = "target"
    STOP = "stop"
    TRAILING_STOP = "trailing_stop"
    TIME_STOP = "time_stop"
    REGIME_FLATTEN = "regime_flatten"
    CIRCUIT_BREAKER = "circuit_breaker"


class RejectReason(StrEnum):
    """Why a symbol was not a valid entry candidate. Logged for diagnosis."""

    INSUFFICIENT_HISTORY = "insufficient_history"
    BELOW_TREND_MA = "below_trend_ma"
    WEAK_MOMENTUM = "weak_momentum"
    NOT_OVERSOLD = "not_oversold"
    PRICE_FILTER = "price_filter"
    ILLIQUID = "illiquid"
    EARNINGS_BLACKOUT = "earnings_blackout"
    NO_ATR = "no_atr"


@dataclass(frozen=True)
class EntryCandidate:
    """A symbol that passed every entry filter, ready to be sized."""

    symbol: str
    price: float
    atr: float
    rsi: float
    momentum: float
    adv: float
    rank_value: float

    @property
    def atr_pct(self) -> float:
        """ATR as a percentage of price — the trade's volatility in relative terms."""
        return self.atr / self.price * 100.0 if self.price > 0 else 0.0


@dataclass(frozen=True)
class ExitDecision:
    symbol: str
    reason: ExitReason
    price: float


@dataclass(frozen=True)
class RegimeState:
    """Portfolio-level risk gate derived from the benchmark index."""

    risk_on: bool
    benchmark_above_ma: bool
    volatility_ok: bool
    benchmark_price: float
    benchmark_ma: float
    volatility: float
    volatility_percentile: float

    def describe(self) -> str:
        if self.risk_on:
            return (
                f"risk-on (benchmark {self.benchmark_price:.2f} above its "
                f"{self.benchmark_ma:.2f} moving average, volatility at the "
                f"{self.volatility_percentile:.0f}th percentile)"
            )
        reasons = []
        if not self.benchmark_above_ma:
            reasons.append(
                f"benchmark {self.benchmark_price:.2f} below its MA "
                f"{self.benchmark_ma:.2f}"
            )
        if not self.volatility_ok:
            reasons.append(
                f"volatility {self.volatility:.1f}% at the "
                f"{self.volatility_percentile:.0f}th percentile"
            )
        return "risk-off (" + "; ".join(reasons) + ")"


def compute_features(bars: pd.DataFrame, strategy: StrategyConfig, adv_window: int = 20) -> pd.DataFrame:
    """Attach indicator columns to an OHLCV frame.

    ``bars`` must be indexed by date ascending with columns open/high/low/
    close/volume. The returned frame is a copy; the input is not mutated.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars is missing column(s): {sorted(missing)}")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("bars must be sorted by date ascending")

    out = bars.copy()
    out["sma_trend"] = indicators.sma(out["close"], strategy.trend_ma_days)
    out["sma_exit"] = indicators.sma(out["close"], strategy.exit_ma_days)
    out["rsi"] = indicators.wilder_rsi(out["close"], strategy.rsi_period)
    out["momentum"] = indicators.rate_of_change(
        out["close"], strategy.momentum_lookback_days
    )
    out["atr"] = indicators.atr(out["high"], out["low"], out["close"], period=20)
    out["adv"] = indicators.dollar_volume(out["close"], out["volume"], adv_window)
    return out


def evaluate_regime(benchmark_features: pd.DataFrame, cfg: Config) -> RegimeState:
    """Decide whether the portfolio may open new long positions.

    This single gate does more for the return profile than any entry rule.
    Buying pullbacks works while the market trends up and fails badly in a
    sustained decline, where every dip keeps going.
    """
    regime = cfg.regime
    if benchmark_features.empty:
        raise ValueError("benchmark history is empty; cannot evaluate regime")

    close = benchmark_features["close"]
    ma = indicators.sma(close, regime.trend_ma_days)
    vol = indicators.realised_volatility(close, regime.vol_lookback_days)
    # One year of trailing volatility gives the percentile a stable reference
    # without reaching so far back that a structural shift dominates it.
    vol_pct = indicators.rolling_percentile(vol, window=252)

    last_close = float(close.iloc[-1])
    last_ma = ma.iloc[-1]
    last_vol = vol.iloc[-1]
    last_vol_pct = vol_pct.iloc[-1]

    if pd.isna(last_ma):
        raise ValueError(
            f"benchmark has {len(close)} bars, too few for a "
            f"{regime.trend_ma_days}-day moving average"
        )

    above_ma = last_close > float(last_ma)
    # An unknown volatility percentile must not block trading on its own;
    # early in a data series there is simply no history to rank against.
    vol_ok = pd.isna(last_vol_pct) or float(last_vol_pct) < regime.vol_percentile_block

    return RegimeState(
        risk_on=above_ma and vol_ok,
        benchmark_above_ma=above_ma,
        volatility_ok=vol_ok,
        benchmark_price=last_close,
        benchmark_ma=float(last_ma),
        volatility=float(last_vol) if pd.notna(last_vol) else float("nan"),
        volatility_percentile=float(last_vol_pct) if pd.notna(last_vol_pct) else float("nan"),
    )


def screen_entry(
    symbol: str,
    features: pd.DataFrame,
    cfg: Config,
    earnings_within_days: int | None = None,
) -> EntryCandidate | RejectReason:
    """Apply every entry filter to one symbol's latest bar.

    Returns a candidate when all filters pass, or the first failing reason.
    Ordering is deliberate: cheap structural checks first, so the logged
    reason points at the most fundamental disqualifier.
    """
    strategy, universe = cfg.strategy, cfg.universe

    if features.empty:
        return RejectReason.INSUFFICIENT_HISTORY

    last = features.iloc[-1]

    if pd.isna(last.get("sma_trend")) or pd.isna(last.get("rsi")):
        return RejectReason.INSUFFICIENT_HISTORY
    if pd.isna(last.get("atr")) or float(last["atr"]) <= 0:
        # Without ATR there is no stop distance, and without a stop there is
        # no defined risk, so the position cannot be sized at all.
        return RejectReason.NO_ATR

    price = float(last["close"])
    if not (universe.min_price <= price <= universe.max_price):
        return RejectReason.PRICE_FILTER

    adv = float(last["adv"]) if pd.notna(last.get("adv")) else 0.0
    if adv < universe.min_avg_dollar_volume:
        return RejectReason.ILLIQUID

    if earnings_within_days is not None and earnings_within_days <= universe.earnings_blackout_days:
        # An earnings gap opens straight through a stop. The realised loss is
        # unbounded by the stop distance, which breaks position sizing.
        return RejectReason.EARNINGS_BLACKOUT

    if price <= float(last["sma_trend"]):
        return RejectReason.BELOW_TREND_MA

    momentum = float(last["momentum"]) if pd.notna(last.get("momentum")) else float("nan")
    if pd.isna(momentum) or momentum < strategy.min_momentum_pct:
        return RejectReason.WEAK_MOMENTUM

    rsi = float(last["rsi"])
    if rsi > strategy.rsi_entry_max:
        return RejectReason.NOT_OVERSOLD

    rank_value = -rsi if strategy.ranking is Ranking.MOST_OVERSOLD else momentum

    return EntryCandidate(
        symbol=symbol,
        price=price,
        atr=float(last["atr"]),
        rsi=rsi,
        momentum=momentum,
        adv=adv,
        rank_value=rank_value,
    )


def rank_candidates(candidates: list[EntryCandidate]) -> list[EntryCandidate]:
    """Order candidates best-first.

    Ties break on symbol so that a run is reproducible: given the same data,
    the bot must always choose the same trades.
    """
    return sorted(candidates, key=lambda c: (-c.rank_value, c.symbol))


def should_exit(
    features: pd.DataFrame,
    cfg: Config,
    holding_days: int,
) -> ExitReason | None:
    """Decide whether a held position should be closed on the next bar.

    Stop-losses are not evaluated here. They live as resting orders at the
    broker so that they still fire when this process is not running, which is
    the whole reason a scheduled bot can be trusted with an open position.
    """
    if features.empty:
        return None

    last = features.iloc[-1]
    strategy = cfg.strategy

    if holding_days >= cfg.risk.max_holding_days:
        return ExitReason.TIME_STOP

    close = float(last["close"])
    exit_ma = last.get("sma_exit")
    rsi = last.get("rsi")

    if pd.notna(exit_ma) and close > float(exit_ma):
        return ExitReason.TARGET
    if pd.notna(rsi) and float(rsi) >= strategy.rsi_exit_min:
        return ExitReason.TARGET

    return None


def initial_stop_price(entry_price: float, atr_value: float, cfg: Config) -> float:
    """Stop price for a new long position.

    Floored just above zero so a bad ATR can never produce a negative or zero
    stop, which the broker would reject and which would leave the position
    unprotected.
    """
    stop = entry_price - cfg.risk.atr_stop_multiple * atr_value
    return max(round(stop, 2), 0.01)


def trailing_stop_price(
    highest_close_since_entry: float, atr_value: float, cfg: Config, current_stop: float
) -> float:
    """Updated trailing stop, which may only ever move up.

    Ratcheting in one direction is what makes the trail a risk reduction. A
    stop that could loosen would re-expose capital that had already been
    locked in.
    """
    trailed = highest_close_since_entry - cfg.risk.atr_trail_multiple * atr_value
    return max(round(trailed, 2), current_stop)
