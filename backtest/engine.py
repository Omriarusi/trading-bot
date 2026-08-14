"""Event-driven backtester.

Calls the same :mod:`bot.signals` and :mod:`bot.risk` code the live engine
does. That is the whole point: a backtest of a reimplementation measures the
reimplementation.

Modelling choices, all deliberately pessimistic:

* A signal from the close of day *t* is executed at the **open** of day *t+1*.
  Nothing is ever filled at a price that was known when the decision was made.
* A stop is checked against each day's **low**. On a gap down it fills at the
  open, not at the trigger — which is what actually happens and is where most
  of the strategy's worst days come from.
* Commission follows the configured IBKR schedule and is charged both ways.
* Slippage is charged on every fill, against the trade.

What it still does not model: partial fills, borrow costs, spread widening in
stress, and survivorship bias in the universe. Treat the output as an upper
bound on what the strategy would have earned, not a forecast.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from bot import indicators, risk, signals
from bot.config import Config, RiskOffAction
from bot.risk import PortfolioState, Sizing, SizingRejection
from bot.signals import ExitReason, RejectReason

log = logging.getLogger(__name__)


@dataclass
class Trade:
    symbol: str
    entry_date: date
    entry_price: float
    shares: int
    stop_price: float
    exit_date: date | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    commission: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return self.shares * (self.exit_price - self.entry_price) - self.commission

    @property
    def return_pct(self) -> float:
        cost = self.shares * self.entry_price
        return self.pnl / cost * 100.0 if cost > 0 else 0.0

    @property
    def holding_days(self) -> int:
        if self.exit_date is None:
            return 0
        return (self.exit_date - self.entry_date).days


@dataclass
class OpenPosition:
    symbol: str
    shares: int
    entry_price: float
    entry_date: date
    stop_price: float
    highest_close: float
    trade: Trade
    entry_commission: float


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade]
    benchmark_curve: pd.Series | None = None
    config_summary: dict = field(default_factory=dict)
    skipped_symbols: list[str] = field(default_factory=list)

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if not t.is_open]

    def stats(self) -> dict:
        """Performance statistics for the run."""
        return compute_stats(self.equity_curve, self.closed_trades, self.benchmark_curve)


def compute_stats(
    equity: pd.Series, trades: list[Trade], benchmark: pd.Series | None = None
) -> dict:
    """Summarise an equity curve and its trades.

    Deliberately reports drawdown and the losing tail alongside return. A
    strategy is only usable if its worst stretch is survivable, and that is
    not visible in a total-return number.
    """
    if equity.empty or len(equity) < 2:
        return {"error": "equity curve too short to evaluate"}

    start_value, end_value = float(equity.iloc[0]), float(equity.iloc[-1])
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)

    total_return = (end_value / start_value - 1) * 100.0
    cagr = ((end_value / start_value) ** (1 / years) - 1) * 100.0 if start_value > 0 else 0.0

    daily = equity.pct_change().dropna()
    ann_vol = float(daily.std() * np.sqrt(252) * 100.0) if len(daily) > 1 else 0.0
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0

    downside = daily[daily < 0]
    sortino = (
        float(daily.mean() / downside.std() * np.sqrt(252))
        if len(downside) > 1 and downside.std() > 0
        else 0.0
    )

    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak * 100.0
    max_dd = float(drawdown.min())

    # Longest stretch spent below a previous high — the figure that decides
    # whether a strategy is psychologically survivable.
    underwater = (equity < running_peak).astype(int)
    longest_dd_days = int(
        underwater.groupby((underwater != underwater.shift()).cumsum()).cumsum().max()
    ) if len(underwater) else 0

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))

    stats = {
        "start_equity": round(start_value, 2),
        "end_equity": round(end_value, 2),
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "annual_volatility_pct": round(ann_vol, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "longest_drawdown_days": longest_dd_days,
        "calmar": round(cagr / abs(max_dd), 2) if max_dd < 0 else 0.0,
        "trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100.0, 1) if trades else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else 0.0,
        "avg_win_pct": round(float(np.mean([t.return_pct for t in wins])), 2) if wins else 0.0,
        "avg_loss_pct": round(float(np.mean([t.return_pct for t in losses])), 2) if losses else 0.0,
        "avg_holding_days": round(float(np.mean([t.holding_days for t in trades])), 1) if trades else 0.0,
        "total_commission": round(sum(t.commission for t in trades), 2),
        "worst_trade_pct": round(min((t.return_pct for t in trades), default=0.0), 2),
        "best_trade_pct": round(max((t.return_pct for t in trades), default=0.0), 2),
    }

    if benchmark is not None and not benchmark.empty:
        aligned = benchmark.reindex(equity.index).ffill().dropna()
        if len(aligned) > 1:
            bench_return = (float(aligned.iloc[-1]) / float(aligned.iloc[0]) - 1) * 100.0
            bench_peak = aligned.cummax()
            stats["benchmark_return_pct"] = round(bench_return, 2)
            stats["benchmark_max_drawdown_pct"] = round(
                float(((aligned - bench_peak) / bench_peak * 100.0).min()), 2
            )
            stats["excess_return_pct"] = round(total_return - bench_return, 2)

    return stats


class Backtester:
    """Replays the strategy day by day over historical bars."""

    def __init__(self, cfg: Config, starting_equity: float | None = None):
        self.cfg = cfg
        self.starting_equity = starting_equity or cfg.account.assumed_equity

    def run(
        self,
        bars: dict[str, pd.DataFrame],
        benchmark_bars: pd.DataFrame,
        start: date | None = None,
        end: date | None = None,
    ) -> BacktestResult:
        features, skipped = self._prepare(bars)
        if not features:
            raise ValueError("No symbol had enough history to backtest")

        eligible_by_date = self._eligible_by_date(features)
        benchmark_features = self._prepare_benchmark(benchmark_bars)
        calendar = self._calendar(features, benchmark_features, start, end)
        if len(calendar) < 2:
            raise ValueError(
                f"Only {len(calendar)} tradable day(s) in range; nothing to simulate"
            )

        cash = self.starting_equity
        positions: dict[str, OpenPosition] = {}
        trades: list[Trade] = []
        equity_points: list[tuple[pd.Timestamp, float]] = []
        halted_until_recovery = False
        peak_equity = self.starting_equity

        # Pending decisions from yesterday's close, executed at today's open.
        pending_entries: list[Sizing] = []
        pending_exits: list[tuple[str, ExitReason]] = []

        for today in calendar:
            today_date = today.date()

            # --- 1. Stops fire first. A stop that would have triggered before
            # a planned discretionary exit must win, or the backtest reports
            # losses smaller than reality.
            for symbol in list(positions):
                position = positions[symbol]
                bar = self._bar(features, symbol, today)
                if bar is None:
                    continue
                if float(bar["low"]) <= position.stop_price:
                    # A gap below the stop fills at the open, not the trigger.
                    fill = min(float(bar["open"]), position.stop_price)
                    cash += self._close(position, fill, today_date, ExitReason.STOP, trades)
                    del positions[symbol]

            # --- 2. Yesterday's exit decisions, at today's open.
            for symbol, reason in pending_exits:
                position = positions.get(symbol)
                if position is None:
                    continue
                bar = self._bar(features, symbol, today)
                if bar is None:
                    continue
                fill = self._apply_slippage(float(bar["open"]), "SELL")
                cash += self._close(position, fill, today_date, reason, trades)
                del positions[symbol]
            pending_exits = []

            # --- 3. Yesterday's entry decisions, at today's open.
            for sizing in pending_entries:
                bar = self._bar(features, sizing.symbol, today)
                if bar is None or sizing.symbol in positions:
                    continue
                fill = self._apply_slippage(float(bar["open"]), "BUY")
                # An open that gapped past the intended entry invalidates the
                # setup; a real bot's marketable limit would not have filled.
                if risk.slippage_exceeded(sizing.entry_price, fill, self.cfg):
                    continue
                commission = self.cfg.costs.commission(sizing.shares, fill)
                cost = sizing.shares * fill + commission
                if cost > cash:
                    continue

                # The stop travels with the fill, keeping the *distance* the
                # sizer budgeted rather than the price it happened to pick.
                # Shares were chosen so that shares x distance equals the risk
                # budget, so preserving the distance preserves the risk;
                # anchoring the stop to the planned entry instead would silently
                # change how much a stop-out costs whenever the open gapped.
                stop_distance = sizing.entry_price - sizing.stop_price
                stop = max(round(fill - stop_distance, 2), 0.01)
                cash -= cost
                trade = Trade(
                    symbol=sizing.symbol, entry_date=today_date, entry_price=fill,
                    shares=sizing.shares, stop_price=stop, commission=commission,
                )
                trades.append(trade)
                positions[sizing.symbol] = OpenPosition(
                    symbol=sizing.symbol, shares=sizing.shares, entry_price=fill,
                    entry_date=today_date, stop_price=stop, highest_close=fill,
                    trade=trade, entry_commission=commission,
                )
            pending_entries = []

            # --- 4. Mark to market.
            holdings_value = 0.0
            for symbol, position in positions.items():
                bar = self._bar(features, symbol, today)
                price = float(bar["close"]) if bar is not None else position.entry_price
                holdings_value += position.shares * price
            equity = cash + holdings_value
            equity_points.append((today, equity))
            peak_equity = max(peak_equity, equity)

            drawdown_pct = (peak_equity - equity) / peak_equity * 100.0 if peak_equity > 0 else 0.0
            if drawdown_pct >= self.cfg.circuit_breakers.max_drawdown_halt_pct:
                halted_until_recovery = True
            elif drawdown_pct < self.cfg.circuit_breakers.max_drawdown_halt_pct / 2:
                # Resume only after recovering half the drawdown, so the bot
                # does not switch on and off around the threshold.
                halted_until_recovery = False

            # --- 5. Decide, on today's close, what to do at tomorrow's open.
            regime = self._regime_at(benchmark_features, today)

            for symbol, position in positions.items():
                history = self._decision_bar(features, symbol, today)
                if history is None or history.empty:
                    continue
                bar = history.iloc[-1]
                close = float(bar["close"])
                position.highest_close = max(position.highest_close, close)

                atr_value = bar.get("atr")
                if pd.notna(atr_value) and float(atr_value) > 0:
                    position.stop_price = signals.trailing_stop_price(
                        position.highest_close, float(atr_value), self.cfg, position.stop_price
                    )

                if regime is not None and not regime.risk_on and (
                    self.cfg.regime.risk_off_action is RiskOffAction.FLATTEN
                ):
                    pending_exits.append((symbol, ExitReason.REGIME_FLATTEN))
                    continue

                holding_days = (today_date - position.entry_date).days
                reason = signals.should_exit(history, self.cfg, holding_days)
                if reason is not None:
                    pending_exits.append((symbol, reason))

            exiting = {s for s, _ in pending_exits}
            can_open = (
                regime is not None
                and regime.risk_on
                and not halted_until_recovery
                and len(positions) - len(exiting) < self.cfg.risk.max_concurrent_positions
            )
            if can_open:
                pending_entries = self._select_entries(
                    features, today, positions, exiting, cash, equity, peak_equity,
                    eligible_today=eligible_by_date.get(today, []),
                )

        equity_curve = pd.Series(
            dict(equity_points), name="equity"
        ).sort_index()

        benchmark_curve = benchmark_features["close"].reindex(equity_curve.index).ffill()

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            benchmark_curve=benchmark_curve,
            skipped_symbols=skipped,
            config_summary={
                "risk_per_trade_pct": self.cfg.risk.risk_per_trade_pct,
                "max_positions": self.cfg.risk.max_concurrent_positions,
                "atr_stop_multiple": self.cfg.risk.atr_stop_multiple,
                "atr_trail_multiple": self.cfg.risk.atr_trail_multiple,
                "rsi_period": self.cfg.strategy.rsi_period,
                "rsi_entry_max": self.cfg.strategy.rsi_entry_max,
                "max_holding_days": self.cfg.risk.max_holding_days,
                "starting_equity": self.starting_equity,
            },
        )

    # -------------------------------------------------------------- internals

    def _prepare(self, bars: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[str]]:
        """Compute features once per symbol, dropping anything too short."""
        warmup = max(
            self.cfg.strategy.trend_ma_days, self.cfg.strategy.momentum_lookback_days
        )
        prepared: dict[str, pd.DataFrame] = {}
        skipped: list[str] = []
        for symbol, frame in bars.items():
            if len(frame) < warmup + 20:
                skipped.append(symbol)
                continue
            try:
                prepared[symbol] = signals.compute_features(frame, self.cfg.strategy)
            except ValueError as exc:
                log.warning("%s: %s", symbol, exc)
                skipped.append(symbol)
        return prepared, skipped

    def _eligible_by_date(
        self, features: dict[str, pd.DataFrame]
    ) -> dict[pd.Timestamp, list[str]]:
        """Pre-index the days on which each symbol could possibly be a candidate.

        The daily loop otherwise asks ``screen_entry`` about every symbol on
        every day: 500 symbols over 2,300 days is 1.2 million calls per
        backtest, and the sweep runs eighteen of them. On a typical day only a
        handful of names are oversold, so almost all of that work is spent
        confirming a rejection.

        The two conditions used here are *necessary* conditions inside
        ``screen_entry`` — it rejects outright on either — so a symbol filtered
        out here could never have become a candidate. That keeps this a pure
        speed-up rather than a second, drifting copy of the entry rules;
        ``screen_entry`` still makes every actual decision.
        """
        threshold = self.cfg.strategy.rsi_entry_max
        by_date: dict[pd.Timestamp, list[str]] = {}

        for symbol, frame in features.items():
            eligible = (
                frame["rsi"].notna()
                & frame["sma_trend"].notna()
                & (frame["close"] > frame["sma_trend"])
                & (frame["rsi"] <= threshold)
            )
            for when in frame.index[eligible]:
                by_date.setdefault(when, []).append(symbol)

        return by_date

    def _prepare_benchmark(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["ma"] = indicators.sma(out["close"], self.cfg.regime.trend_ma_days)
        vol = indicators.realised_volatility(out["close"], self.cfg.regime.vol_lookback_days)
        out["vol"] = vol
        out["vol_pct"] = indicators.rolling_percentile(vol, window=252)
        return out

    def _calendar(
        self, features: dict[str, pd.DataFrame], benchmark: pd.DataFrame,
        start: date | None, end: date | None,
    ) -> pd.DatetimeIndex:
        """Trading days on which a decision is possible.

        Starts only once the benchmark's own trend filter has warmed up;
        before that there is no regime read and no basis for a trade.
        """
        index = benchmark.index[benchmark["ma"].notna()]
        union = pd.DatetimeIndex([])
        for frame in features.values():
            union = union.union(frame.index)
        index = index.intersection(union)

        if start is not None:
            index = index[index >= pd.Timestamp(start)]
        if end is not None:
            index = index[index <= pd.Timestamp(end)]
        return pd.DatetimeIndex(index).sort_values()

    @staticmethod
    def _bar(features: dict[str, pd.DataFrame], symbol: str, when: pd.Timestamp):
        frame = features.get(symbol)
        if frame is None or when not in frame.index:
            return None
        return frame.loc[when]

    @staticmethod
    def _decision_bar(features: dict[str, pd.DataFrame], symbol: str, when: pd.Timestamp):
        """The single bar a decision is allowed to see, as a one-row frame.

        The decision functions read only the latest row — every lookback is
        already baked into the indicator columns by ``compute_features``. So
        handing them one row is not a shortcut: it makes looking forward
        structurally impossible rather than merely intended, and it removes a
        prefix copy that ran once per symbol per day. Across a 187-symbol,
        2,300-day backtest that copy dominated the runtime.
        """
        frame = features.get(symbol)
        if frame is None or when not in frame.index:
            return None
        return frame.loc[[when]]

    def _regime_at(self, benchmark: pd.DataFrame, when: pd.Timestamp):
        if when not in benchmark.index:
            return None
        row = benchmark.loc[when]
        if pd.isna(row["ma"]):
            return None
        vol_pct = row["vol_pct"]
        vol_ok = pd.isna(vol_pct) or float(vol_pct) < self.cfg.regime.vol_percentile_block
        above_ma = float(row["close"]) > float(row["ma"])
        return signals.RegimeState(
            risk_on=above_ma and vol_ok,
            benchmark_above_ma=above_ma,
            volatility_ok=vol_ok,
            benchmark_price=float(row["close"]),
            benchmark_ma=float(row["ma"]),
            volatility=float(row["vol"]) if pd.notna(row["vol"]) else float("nan"),
            volatility_percentile=float(vol_pct) if pd.notna(vol_pct) else float("nan"),
        )

    def _select_entries(
        self, features: dict[str, pd.DataFrame], today: pd.Timestamp,
        positions: dict[str, OpenPosition], exiting: set[str],
        cash: float, equity: float, peak_equity: float,
        eligible_today: list[str] | None = None,
    ) -> list[Sizing]:
        candidates = []
        # Only symbols that cleared the necessary conditions are worth asking
        # about; the rest cannot pass screen_entry by construction.
        for symbol in eligible_today if eligible_today is not None else features:
            if symbol in positions:
                continue
            history = self._decision_bar(features, symbol, today)
            if history is None or history.empty:
                continue
            outcome = signals.screen_entry(symbol, history, self.cfg)
            if not isinstance(outcome, RejectReason):
                candidates.append(outcome)

        if not candidates:
            return []

        state = PortfolioState(
            equity=equity, cash=cash, settled_cash=cash,
            gross_exposure=equity - cash,
            open_position_count=len(positions) - len(exiting),
            peak_equity=peak_equity, equity_at_session_start=equity,
        )

        planned: list[Sizing] = []
        slots = self.cfg.risk.max_concurrent_positions - state.open_position_count
        for candidate in signals.rank_candidates(candidates):
            if slots <= 0:
                break
            stop = signals.initial_stop_price(candidate.price, candidate.atr, self.cfg)
            sizing = risk.size_position(
                candidate.symbol, candidate.price, stop, candidate.adv, state, self.cfg
            )
            if isinstance(sizing, SizingRejection):
                continue
            planned.append(sizing)
            slots -= 1
            state = PortfolioState(
                equity=state.equity,
                cash=state.cash - sizing.notional,
                settled_cash=state.settled_cash - sizing.notional,
                gross_exposure=state.gross_exposure + sizing.notional,
                open_position_count=state.open_position_count + 1,
                peak_equity=state.peak_equity,
                equity_at_session_start=state.equity_at_session_start,
            )
        return planned

    def _apply_slippage(self, price: float, side: str) -> float:
        """Move the fill against us by the configured amount."""
        drift = self.cfg.costs.slippage_bps / 10_000.0
        return price * (1 + drift) if side == "BUY" else price * (1 - drift)

    def _close(
        self, position: OpenPosition, fill: float, when: date,
        reason: ExitReason, trades: list[Trade],
    ) -> float:
        """Close a position, returning the cash proceeds."""
        commission = self.cfg.costs.commission(position.shares, fill)
        position.trade.exit_date = when
        position.trade.exit_price = fill
        position.trade.exit_reason = reason.value
        position.trade.commission = position.entry_commission + commission
        return position.shares * fill - commission
