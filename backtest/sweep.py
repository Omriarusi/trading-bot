"""Compare a small number of pre-registered strategy variants.

Deliberately not a grid search. With ~1,000 trades of history, searching a
large parameter space finds the settings that best fit *this* history and
tells you almost nothing about the next nine years. Each variant here is a
hypothesis about a specific weakness in the baseline, written down before the
result is known, so a win is evidence rather than a coincidence.

The baseline measured over 2017-2026 returned +34.4% against a +272.8%
benchmark, with commission at 152% of net profit and average wins (+2.53%)
smaller than average losses (-3.81%). Every variant below targets one of
those two facts: too much cost per unit of edge, or winners cut too early.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import date

import pandas as pd

from backtest.engine import Backtester
from bot.config import Config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Variant:
    name: str
    hypothesis: str
    apply: object  # Callable[[Config], Config]


def _original_baseline() -> Config:
    """The strategy exactly as first implemented, pinned regardless of config.yaml.

    Once `combined` was adopted into the committed config, deriving these
    variants from the *loaded* config rather than from this fixed reference
    silently collapsed several of them into identical configurations: the
    first sweep run after adoption produced byte-identical results — the
    same 483 trades, the same $338 of commission — for baseline, winners_run,
    fewer_bigger, selective, and combined, because every one of those
    variants' changes was already present in the file being mutated.
    Re-applying a no-op change is still a no-op. Pinning to a fixed reference
    keeps the historical comparison meaningful no matter what is currently
    committed.
    """
    return Config()


def _let_winners_run(cfg: Config) -> Config:
    """Hand the exit to the trailing stop instead of the moving-average target.

    Tests whether the poor win/loss ratio is caused by the exit rather than
    the entry. Holding longer also spreads a fixed commission over a larger
    move, which attacks the cost problem at the same time.
    """
    return replace(
        cfg,
        strategy=replace(cfg.strategy, exit_on_ma_recross=False, rsi_exit_min=80.0),
        risk=replace(cfg.risk, max_holding_days=30, atr_trail_multiple=2.5),
    )


def _fewer_bigger_positions(cfg: Config) -> Config:
    """Concentrate into fewer, larger positions.

    Commission has a fixed floor per order, so at this account size cost as a
    percentage falls roughly in proportion to position size. Three positions
    of ~$450 pay materially less than four of ~$340 for the same exposure.
    """
    return replace(
        cfg,
        risk=replace(
            cfg.risk,
            max_concurrent_positions=3,
            max_position_pct=40.0,
            min_position_notional=400.0,
        ),
        execution=replace(cfg.execution, max_order_notional=650.0),
    )


def _more_selective(cfg: Config) -> Config:
    """Demand a deeper pullback and stronger trend.

    Fewer trades means less total commission. If the edge per trade rises
    enough to offset the lost opportunities, selectivity is worth more than
    frequency at this account size.
    """
    return replace(
        cfg,
        strategy=replace(cfg.strategy, rsi_entry_max=15.0, min_momentum_pct=10.0),
    )


def _combined(cfg: Config) -> Config:
    """All three together — the variant most likely to be adopted."""
    return _more_selective(_fewer_bigger_positions(_let_winners_run(cfg)))


def _higher_risk(cfg: Config) -> Config:
    """The combined variant sized more aggressively.

    Included because the account owner asked for a high-risk profile. Whether
    this is sensible depends entirely on whether the combined variant has a
    positive edge: leverage on a losing strategy just loses faster.
    """
    cfg = _combined(cfg)
    return replace(cfg, risk=replace(cfg.risk, risk_per_trade_pct=3.5))


def _trend_ma_50(cfg: Config) -> Config:
    """Judge the trend on a 50-day average instead of a 200-day one.

    A faster filter admits stocks that have only recently turned up, which
    means more candidates and earlier entries. The cost is that a 50-day
    average also turns down faster and whipsaws in choppy markets, so it
    permits buying dips in names that are rolling over rather than pulling
    back. Which effect dominates is an empirical question.
    """
    return replace(cfg, strategy=replace(cfg.strategy, trend_ma_days=50))


def _volume_capitulation(cfg: Config) -> Config:
    """Require the entry day to trade at least 1.5x its average volume.

    The capitulation reading: heavy selling into a pullback means the sellers
    are spending themselves, and the bounce is closer.
    """
    return replace(cfg, strategy=replace(cfg.strategy, min_volume_ratio=1.5))


def _volume_quiet(cfg: Config) -> Config:
    """Require the entry day to trade no more than its average volume.

    The opposite reading, equally commonly held: a pullback nobody is panicking
    about is an orderly one, and heavy volume on a down day is distribution
    rather than exhaustion.
    """
    return replace(cfg, strategy=replace(cfg.strategy, max_volume_ratio=1.0))


def _combined_ma50(cfg: Config) -> Config:
    """The adopted strategy with the faster trend filter."""
    return _trend_ma_50(_combined(cfg))


def _combined_capitulation(cfg: Config) -> Config:
    """The adopted strategy with the high-volume requirement."""
    return _volume_capitulation(_combined(cfg))


def _combined_quiet(cfg: Config) -> Config:
    """The adopted strategy with the low-volume requirement."""
    return _volume_quiet(_combined(cfg))


def _combined_fast_exit(cfg: Config) -> Config:
    """The adopted strategy, forced out after 3 days instead of 30.

    True scalping — holds measured in seconds to minutes, dozens of trades a
    day — cannot run on this repository at all: a twice-daily scheduled job
    cannot react on that timescale, free data sources give at most a few
    weeks of intraday history (not enough for a trustworthy backtest), and
    the marketable-limit crossing this bot already uses to avoid bad fills
    (0.3% each way, 0.6% round trip) alone exceeds a typical scalp target of
    0.1-0.3%. This is the fastest turnover actually testable within the
    daily-bar architecture: not scalping, but the nearest honest analogue.
    The trailing stop stays at combined's 2.5 ATR — it can never sit tighter
    than the initial stop, so 3 days is rarely enough room for it to have
    ratcheted up at all. The 3-day cap does essentially all of the work here.
    """
    return replace(cfg, risk=replace(cfg.risk, max_holding_days=3))


VARIANTS: tuple[Variant, ...] = (
    # Fixed historical progression: each ignores the config passed in and
    # starts from _original_baseline(), so this group stays a stable
    # reference point no matter what config.yaml currently trades.
    Variant("baseline", "the original strategy (fixed, ignores config.yaml)",
            lambda c: _original_baseline()),
    Variant("winners_run", "trailing stop exit instead of the MA target",
            lambda c: _let_winners_run(_original_baseline())),
    Variant("fewer_bigger", "3 larger positions to cut commission drag",
            lambda c: _fewer_bigger_positions(_original_baseline())),
    Variant("selective", "deeper pullback and stronger trend required",
            lambda c: _more_selective(_original_baseline())),
    Variant("combined", "winners_run + fewer_bigger + selective",
            lambda c: _combined(_original_baseline())),
    Variant("combined_hot", "combined, sized at 3.5% risk per trade",
            lambda c: _higher_risk(_original_baseline())),
    Variant("ma50", "trend judged on a 50-day average, alone (vs. the original baseline)",
            lambda c: _trend_ma_50(_original_baseline())),
    # Requested changes, tested against whatever config.yaml currently
    # trades — these three intentionally use the passed-in cfg, so they
    # track "the adopted strategy plus this one change" even as the
    # adopted strategy itself changes over time.
    Variant("combined_ma50", "currently adopted strategy, 50-day trend filter", _combined_ma50),
    Variant("combined_capitulation", "currently adopted strategy, volume >= 1.5x average", _combined_capitulation),
    Variant("combined_quiet", "currently adopted strategy, volume <= 1.0x average", _combined_quiet),
    Variant("combined_fast_exit", "currently adopted strategy, 3-day max hold (fastest honest analogue to scalping)", _combined_fast_exit),
)


def run_sweep(
    cfg: Config,
    bars: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    starting_equity: float,
    start: date | None = None,
    end: date | None = None,
) -> list[tuple[Variant, dict]]:
    """Backtest every variant over identical data."""
    results: list[tuple[Variant, dict]] = []
    for variant in VARIANTS:
        log.info("running variant: %s", variant.name)
        try:
            variant_cfg = variant.apply(cfg)
            result = Backtester(variant_cfg, starting_equity=starting_equity).run(
                bars, benchmark, start=start, end=end
            )
            results.append((variant, result.stats()))
        except (ValueError, KeyError) as exc:
            log.error("variant %s failed: %s", variant.name, exc)
            results.append((variant, {"error": str(exc)}))
    return results


def run_split_sweep(
    cfg: Config,
    bars: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    starting_equity: float,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, list[tuple[Variant, dict]]]:
    """Run the sweep over the full period and over each half separately.

    This is the check that decides whether a variant's advantage is real.
    Choosing the best of six variants on one sample will always produce a
    winner, whether or not any of them has an edge — so the question is not
    "which scored highest" but "did the same one score highest in both
    halves". A variant that wins the first half and loses the second was
    fitted to the first half.

    The split is a single chronological cut rather than a random one:
    randomly interleaving days would let a variant learn from the future,
    which is exactly what this is meant to detect.
    """
    index = benchmark.index
    first_day = pd.Timestamp(start) if start else index[0]
    last_day = pd.Timestamp(end) if end else index[-1]
    midpoint = index[(index >= first_day) & (index <= last_day)]
    if len(midpoint) < 4:
        raise ValueError("not enough history to split into two periods")
    cut = midpoint[len(midpoint) // 2]

    log.info(
        "split: %s..%s then %s..%s",
        first_day.date(), cut.date(), cut.date(), last_day.date(),
    )

    ranges = {
        "full": (first_day.date(), last_day.date()),
        "first_half": (first_day.date(), cut.date()),
        "second_half": (cut.date(), last_day.date()),
    }
    periods: dict[str, list[tuple[Variant, dict]]] = {name: [] for name in ranges}

    # One Backtester per variant, reused across all three ranges, so the
    # indicator columns for a 500-symbol universe are computed once rather
    # than three times.
    for variant in VARIANTS:
        log.info("running variant: %s", variant.name)
        backtester = Backtester(variant.apply(cfg), starting_equity=starting_equity)
        for name, (begin, finish) in ranges.items():
            try:
                result = backtester.run(bars, benchmark, start=begin, end=finish)
                periods[name].append((variant, result.stats()))
            except (ValueError, KeyError) as exc:
                log.error("variant %s failed over %s: %s", variant.name, name, exc)
                periods[name].append((variant, {"error": str(exc)}))

    return periods


def render_split(periods: dict[str, list[tuple[Variant, dict]]]) -> str:
    """Report each variant's rank in both halves, and flag the unstable ones."""
    lines = [
        "# Out-of-sample check",
        "",
        "Picking the best of six variants on a single sample always produces a "
        "winner, edge or no edge. What matters is whether the same variant wins "
        "in both halves of the history. One that wins the first half and loses "
        "the second was fitted to the first half.",
        "",
        "| Variant | Full CAGR | 1st half CAGR | 2nd half CAGR | 1st DD | 2nd DD | Consistent |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]

    def by_name(results: list[tuple[Variant, dict]]) -> dict[str, dict]:
        return {v.name: s for v, s in results}

    full = by_name(periods["full"])
    first = by_name(periods["first_half"])
    second = by_name(periods["second_half"])

    for variant in VARIANTS:
        f, a, b = (
            full.get(variant.name, {}),
            first.get(variant.name, {}),
            second.get(variant.name, {}),
        )
        if any("error" in d or not d for d in (f, a, b)):
            lines.append(f"| `{variant.name}` | — | — | — | — | — | n/a |")
            continue

        # "Consistent" means profitable in both halves, not merely on average.
        # A variant carried entirely by one good stretch is not a strategy.
        consistent = a["cagr_pct"] > 0 and b["cagr_pct"] > 0
        lines.append(
            f"| `{variant.name}` | {f['cagr_pct']:+.1f}% | {a['cagr_pct']:+.1f}% "
            f"| {b['cagr_pct']:+.1f}% | {a['max_drawdown_pct']:.1f}% "
            f"| {b['max_drawdown_pct']:.1f}% | {'yes' if consistent else 'NO'} |"
        )

    lines += [
        "",
        "_A variant marked `NO` lost money in one of the two halves. Whatever "
        "its full-period figure says, it has not shown that it works in more "
        "than one market._",
        "",
    ]
    return "\n".join(lines)


def render_sweep(results: list[tuple[Variant, dict]]) -> str:
    """Format the comparison, ranked by risk-adjusted return."""
    lines = [
        "# Strategy comparison",
        "",
        "Identical data and identical costs across every variant. Ranked by "
        "Calmar (return per unit of worst drawdown), because at this account "
        "size surviving the bad stretch matters more than the headline return.",
        "",
        "| Variant | Return | CAGR | Max DD | Sharpe | Calmar | Trades | Win% | PF | Commission | Comm/PnL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    ok = [(v, s) for v, s in results if "error" not in s]
    ranked = sorted(ok, key=lambda pair: pair[1].get("calmar", 0), reverse=True)

    for variant, stats in ranked:
        net = stats["end_equity"] - stats["start_equity"]
        comm_ratio = (
            f"{stats['total_commission'] / abs(net) * 100:.0f}%" if net else "n/a"
        )
        lines.append(
            f"| `{variant.name}` | {stats['total_return_pct']:+.1f}% "
            f"| {stats['cagr_pct']:+.1f}% | {stats['max_drawdown_pct']:.1f}% "
            f"| {stats['sharpe']:.2f} | {stats['calmar']:.2f} "
            f"| {stats['trades']} | {stats['win_rate_pct']:.0f}% "
            f"| {stats['profit_factor']:.2f} | ${stats['total_commission']:,.0f} "
            f"| {comm_ratio} |"
        )

    for variant, stats in results:
        if "error" in stats:
            lines.append(f"| `{variant.name}` | failed: {stats['error']} |")

    if ok:
        benchmark_return = ok[0][1].get("benchmark_return_pct")
        if benchmark_return is not None:
            lines += [
                "",
                f"**Benchmark buy-and-hold over the same period: "
                f"{benchmark_return:+.1f}%** "
                f"(max drawdown {ok[0][1].get('benchmark_max_drawdown_pct', 0):.1f}%)",
            ]

    lines += ["", "## What each variant tests", ""]
    for variant in VARIANTS:
        lines.append(f"- **`{variant.name}`** — {variant.hypothesis}")

    lines += [
        "",
        "---",
        "",
        "_Six pre-registered variants, not a grid search. Searching a large "
        "parameter space over ~1,000 trades finds what fits this history, not "
        "what works next. Treat a variant that wins by a small margin as noise; "
        "only a large, consistent gap is evidence._",
    ]
    return "\n".join(lines)
