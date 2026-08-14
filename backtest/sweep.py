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


VARIANTS: tuple[Variant, ...] = (
    Variant("baseline", "current committed configuration", lambda c: c),
    Variant("winners_run", "trailing stop exit instead of the MA target", _let_winners_run),
    Variant("fewer_bigger", "3 larger positions to cut commission drag", _fewer_bigger_positions),
    Variant("selective", "deeper pullback and stronger trend required", _more_selective),
    Variant("combined", "winners_run + fewer_bigger + selective", _combined),
    Variant("combined_hot", "combined, sized at 3.5% risk per trade", _higher_risk),
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
