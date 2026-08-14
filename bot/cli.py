"""Command-line entry point.

    python -m bot.cli validate-config
    python -m bot.cli check-account
    python -m bot.cli scan
    python -m bot.cli run
    python -m bot.cli backtest --start 2019-01-01
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from bot.config import REPO_ROOT, Config, ConfigError
from bot.config import load as load_config

log = logging.getLogger("bot")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _write_step_summary(markdown: str) -> None:
    """Append to the GitHub Actions run summary when running in CI."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(markdown + "\n")
    except OSError as exc:
        log.warning("could not write run summary: %s", exc)


# ------------------------------------------------------------------ commands


def cmd_validate_config(args) -> int:
    cfg = load_config(args.config)
    print("Configuration is valid.\n")
    print(f"  Mode                 {cfg.execution.mode.value}")
    print(f"  Entity               {cfg.account.entity} "
          f"({'PRIIPs-restricted, single stocks only' if cfg.account.priips_restricted else 'ETFs permitted'})")
    print(f"  Account type         {cfg.account.type.value}")
    print(f"  Assumed equity       ${cfg.account.assumed_equity:,.2f}")
    print(f"  Risk per trade       {cfg.risk.risk_per_trade_pct}% "
          f"(${cfg.account.assumed_equity * cfg.risk.risk_per_trade_pct / 100:,.2f} per stop-out)")
    print(f"  Max positions        {cfg.risk.max_concurrent_positions}")
    print(f"  Worst modelled day   "
          f"-{cfg.risk.risk_per_trade_pct * cfg.risk.max_concurrent_positions:.0f}% "
          f"(every position stopping out at once)")
    print(f"  Drawdown halt        {cfg.circuit_breakers.max_drawdown_halt_pct}%")
    print(f"  Universe             {cfg.universe.list_name}")
    return 0


def cmd_check_account(args) -> int:
    """Report what the broker actually says about the account.

    Run this before trusting any config value: entity, account type, and
    permissions are facts about the account, not preferences.
    """
    from bot.broker.ibkr import IbkrBroker, missing_credentials

    missing = missing_credentials()
    if missing:
        print("Cannot connect. Missing environment variable(s):")
        for name in missing:
            print(f"  - {name}")
        print("\nSee docs/IBKR_SETUP.md for how to obtain these.")
        return 1

    cfg = load_config(args.config)
    broker = IbkrBroker()
    summary = broker.get_account_summary()

    print(f"Account          {summary.account_id}")
    print(f"Type             {summary.account_type}")
    print(f"Currency         {summary.currency}")
    print(f"Net liquidation  ${summary.equity:,.2f}")
    print(f"Cash             ${summary.cash:,.2f}")
    print(f"Settled cash     ${summary.settled_cash:,.2f}")
    print(f"Buying power     ${summary.buying_power:,.2f}")

    positions = broker.get_positions()
    print(f"\nOpen positions   {len(positions)}")
    for position in positions:
        print(
            f"  {position.symbol:6s} {position.shares:5d} @ ${position.average_cost:8.2f} "
            f"→ ${position.market_price:8.2f}  ({position.unrealised_pnl_pct:+.1f}%)"
        )

    orders = broker.get_open_orders()
    print(f"\nWorking orders   {len(orders)}")
    for order in orders:
        price = order.stop_price if order.is_protective_stop else order.limit_price
        print(
            f"  {order.symbol:6s} {order.side.value:4s} {order.order_type.value:7s} "
            f"{order.shares:5d} @ {price}  [{order.status.value}]"
        )

    unprotected = [
        p for p in positions
        if not any(o.symbol == p.symbol and o.is_protective_stop for o in orders)
    ]
    if unprotected:
        print("\nWARNING: these positions have no resting stop order:")
        for position in unprotected:
            print(f"  - {position.symbol}")
        print("The next `run` will place stops for them.")

    if summary.account_type != cfg.account.type.value:
        print(
            f"\nNOTE: config says account.type is '{cfg.account.type.value}' but the "
            f"broker reports '{summary.account_type}'. Update config.yaml to match."
        )
    return 0


def cmd_scan(args) -> int:
    """Screen the universe and print what would be traded, touching no broker."""
    from bot import signals
    from bot.data import PriceRepository
    from bot.signals import RejectReason
    from bot.universe_list import get_universe

    cfg = load_config(args.config)
    repository = PriceRepository(cfg.data)

    benchmark = repository.get(cfg.regime.benchmark)
    benchmark_features = signals.compute_features(benchmark.frame, cfg.strategy)
    regime = signals.evaluate_regime(benchmark_features, cfg)
    print(f"Regime: {regime.describe()}\n")

    symbols = list(get_universe(cfg.universe.list_name))
    bars, failures = repository.get_many(symbols)
    print(f"Fetched {len(bars)}/{len(symbols)} symbols "
          f"({len(failures)} failed)\n")

    candidates = []
    reject_counts: dict[str, int] = {}
    for symbol, symbol_bars in bars.items():
        features = signals.compute_features(symbol_bars.frame, cfg.strategy)
        outcome = signals.screen_entry(symbol, features, cfg)
        if isinstance(outcome, RejectReason):
            reject_counts[outcome.value] = reject_counts.get(outcome.value, 0) + 1
        else:
            candidates.append(outcome)

    print(f"{'SYMBOL':<8}{'PRICE':>9}{'RSI':>7}{'MOM%':>8}{'ATR%':>7}{'STOP':>9}")
    print("-" * 48)
    for candidate in signals.rank_candidates(candidates):
        stop = signals.initial_stop_price(candidate.price, candidate.atr, cfg)
        print(
            f"{candidate.symbol:<8}{candidate.price:>9.2f}{candidate.rsi:>7.1f}"
            f"{candidate.momentum:>8.1f}{candidate.atr_pct:>7.1f}{stop:>9.2f}"
        )
    if not candidates:
        print("(no candidates)")

    print("\nRejections by reason:")
    for reason, count in sorted(reject_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<24} {count}")
    return 0


def cmd_run(args) -> int:
    """Execute one full trading run."""
    from bot.data import PriceRepository
    from bot.engine import Engine, build_broker
    from bot.state import BotState

    cfg = load_config(args.config)

    if cfg.execution.mode.risks_real_money and not args.i_understand_this_is_live:
        print(
            "Refusing to run in live mode without --i-understand-this-is-live.\n"
            "This mode places real orders with real money."
        )
        return 2

    log.info("Starting run in %s mode", cfg.execution.mode.value)

    state_path = REPO_ROOT / "state" / "portfolio.json"
    state = BotState.load(state_path)
    broker = build_broker(cfg)
    repository = PriceRepository(cfg.data)

    engine = Engine(cfg, broker, repository, state)
    report = engine.run()

    print("\n" + report.to_markdown())
    if cfg.notifications.write_run_summary:
        _write_step_summary(report.to_markdown())

    log_dir = REPO_ROOT / cfg.notifications.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    (log_dir / f"run-{stamp}.md").write_text(report.to_markdown() + "\n")

    # A run that could not read the account is a failure the scheduler should
    # surface, not a quiet no-op.
    if not report.actions and report.warnings and report.equity == 0:
        return 1
    return 0


def cmd_backtest(args) -> int:
    from backtest.engine import Backtester
    from backtest.report import render_report
    from bot.data import PriceRepository
    from bot.universe_list import get_universe

    cfg = load_config(args.config)
    if args.risk_per_trade or args.max_positions or args.years:
        cfg = _apply_overrides(cfg, args)

    repository = PriceRepository(cfg.data)
    symbols = list(get_universe(args.universe or cfg.universe.list_name))

    log.info("Fetching %d symbols plus benchmark...", len(symbols))
    bars, failures = repository.get_many(symbols)
    if failures:
        log.warning("%d symbol(s) unavailable", len(failures))
    if not bars:
        print("No price data available; cannot backtest.")
        return 1

    benchmark = repository.get(cfg.regime.benchmark)

    backtester = Backtester(cfg, starting_equity=args.equity or cfg.account.assumed_equity)
    result = backtester.run(
        {s: b.frame for s, b in bars.items()},
        benchmark.frame,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
    )

    stats = result.stats()
    print(render_report(result, stats))

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
        log.info("Wrote statistics to %s", path)

    _write_step_summary(render_report(result, stats))
    return 0


def _apply_overrides(cfg: Config, args) -> Config:
    """Apply sweep overrides without touching the committed config file."""
    from dataclasses import replace

    risk_cfg = cfg.risk
    if args.risk_per_trade:
        risk_cfg = replace(risk_cfg, risk_per_trade_pct=args.risk_per_trade)
    if args.max_positions:
        risk_cfg = replace(risk_cfg, max_concurrent_positions=args.max_positions)

    data_cfg = cfg.data
    if args.years:
        data_cfg = replace(data_cfg, history_days=int(args.years * 365))

    return replace(cfg, risk=risk_cfg, data=data_cfg)


def _parse_date(value: str | None) -> date | None:
    return datetime.strptime(value, "%Y-%m-%d").date() if value else None


# --------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bot", description="IBKR swing trading bot")
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate-config", help="check config.yaml for errors").set_defaults(
        func=cmd_validate_config
    )
    sub.add_parser("check-account", help="report what IBKR says about the account").set_defaults(
        func=cmd_check_account
    )
    sub.add_parser("scan", help="screen the universe without trading").set_defaults(
        func=cmd_scan
    )

    run = sub.add_parser("run", help="execute one trading run")
    run.add_argument(
        "--i-understand-this-is-live",
        action="store_true",
        help="required confirmation when execution.mode is 'live'",
    )
    run.set_defaults(func=cmd_run)

    bt = sub.add_parser("backtest", help="replay the strategy over history")
    bt.add_argument("--start", help="YYYY-MM-DD")
    bt.add_argument("--end", help="YYYY-MM-DD")
    bt.add_argument("--years", type=float, help="how much history to fetch")
    bt.add_argument("--equity", type=float, help="starting equity")
    bt.add_argument("--universe", help="named universe to trade")
    bt.add_argument("--risk-per-trade", type=float, help="override risk %% per trade")
    bt.add_argument("--max-positions", type=int, help="override concurrent position cap")
    bt.add_argument("--json-out", help="write statistics as JSON to this path")
    bt.set_defaults(func=cmd_backtest)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
