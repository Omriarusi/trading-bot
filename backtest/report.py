"""Human-readable backtest output."""

from __future__ import annotations

from collections import Counter

from backtest.engine import BacktestResult


def render_report(result: BacktestResult, stats: dict) -> str:
    """Format a backtest as markdown, readable in a terminal or a CI summary."""
    if "error" in stats:
        return f"Backtest failed: {stats['error']}"

    equity = result.equity_curve
    lines = [
        "# Backtest",
        "",
        f"**Period:** {equity.index[0].date()} → {equity.index[-1].date()} "
        f"({len(equity)} trading days)  ",
        f"**Universe:** {len(set(t.symbol for t in result.trades))} symbols traded",
        "",
        "## Result",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Starting equity | ${stats['start_equity']:,.2f} |",
        f"| Ending equity | ${stats['end_equity']:,.2f} |",
        f"| Total return | {stats['total_return_pct']:+.1f}% |",
        f"| CAGR | {stats['cagr_pct']:+.1f}% |",
        f"| **Max drawdown** | **{stats['max_drawdown_pct']:.1f}%** |",
        f"| Longest drawdown | {stats['longest_drawdown_days']} days |",
        f"| Annualised volatility | {stats['annual_volatility_pct']:.1f}% |",
        f"| Sharpe | {stats['sharpe']:.2f} |",
        f"| Sortino | {stats['sortino']:.2f} |",
        f"| Calmar | {stats['calmar']:.2f} |",
    ]

    if "benchmark_return_pct" in stats:
        lines += [
            "",
            "## Against buy-and-hold",
            "",
            "| Metric | Strategy | Benchmark |",
            "| --- | ---: | ---: |",
            f"| Total return | {stats['total_return_pct']:+.1f}% | "
            f"{stats['benchmark_return_pct']:+.1f}% |",
            f"| Max drawdown | {stats['max_drawdown_pct']:.1f}% | "
            f"{stats['benchmark_max_drawdown_pct']:.1f}% |",
            "",
            f"Excess return: **{stats['excess_return_pct']:+.1f}%**",
        ]

    lines += [
        "",
        "## Trades",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Closed trades | {stats['trades']} |",
        f"| Win rate | {stats['win_rate_pct']:.1f}% |",
        f"| Profit factor | {stats['profit_factor']:.2f} |",
        f"| Average win | {stats['avg_win_pct']:+.2f}% |",
        f"| Average loss | {stats['avg_loss_pct']:+.2f}% |",
        f"| Best / worst | {stats['best_trade_pct']:+.1f}% / {stats['worst_trade_pct']:+.1f}% |",
        f"| Average hold | {stats['avg_holding_days']:.1f} days |",
        f"| Total commission | ${stats['total_commission']:,.2f} |",
    ]

    # Commission as a share of the final result is the number that decides
    # whether a strategy is viable at this account size.
    net_profit = stats["end_equity"] - stats["start_equity"]
    if stats["total_commission"] > 0:
        lines.append(
            f"| Commission vs. net P&L | "
            f"{stats['total_commission'] / abs(net_profit) * 100:.0f}% |"
            if net_profit != 0
            else "| Commission vs. net P&L | n/a |"
        )

    reasons = Counter(t.exit_reason for t in result.closed_trades if t.exit_reason)
    if reasons:
        lines += ["", "## Exits by reason", "", "| Reason | Count | Avg return |", "| --- | ---: | ---: |"]
        for reason, count in reasons.most_common():
            matching = [t for t in result.closed_trades if t.exit_reason == reason]
            avg = sum(t.return_pct for t in matching) / len(matching)
            lines.append(f"| {reason} | {count} | {avg:+.2f}% |")

    worst = sorted(result.closed_trades, key=lambda t: t.return_pct)[:5]
    if worst:
        lines += [
            "",
            "## Five worst trades",
            "",
            "| Symbol | Entry | Exit | Return | Reason |",
            "| --- | --- | --- | ---: | --- |",
        ]
        for trade in worst:
            lines.append(
                f"| {trade.symbol} | {trade.entry_date} | {trade.exit_date} | "
                f"{trade.return_pct:+.1f}% | {trade.exit_reason} |"
            )

    if result.skipped_symbols:
        lines += [
            "",
            f"_{len(result.skipped_symbols)} symbol(s) skipped for insufficient history._",
        ]

    lines += [
        "",
        "---",
        "",
        "_Backtests overstate live results: they assume every intended fill happened, "
        "and the universe is today's liquid names, which excludes companies that "
        "were liquid then and are not now. Treat the drawdown as the honest number "
        "and the return as optimistic._",
    ]
    return "\n".join(lines)
