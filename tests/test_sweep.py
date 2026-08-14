"""Every sweep variant must produce a configuration the bot would accept.

A variant that violates a safety rule would be measured, reported, and
possibly adopted, without the config validator ever seeing it.
"""

from __future__ import annotations

import pytest

import bot.config as config_module
from backtest.sweep import VARIANTS, render_sweep
from bot.config import Config


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.name)
def test_variant_produces_a_valid_config(variant):
    config_module._validate(variant.apply(Config()))


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.name)
def test_variant_keeps_a_stop_and_a_position_limit(variant):
    cfg = variant.apply(Config())
    assert cfg.risk.atr_stop_multiple > 0
    assert cfg.risk.max_concurrent_positions >= 1
    assert cfg.risk.risk_per_trade_pct > 0


def test_variant_names_are_unique():
    names = [v.name for v in VARIANTS]
    assert len(names) == len(set(names))


def test_render_handles_a_failed_variant():
    results = [(VARIANTS[0], {"error": "not enough history"})]
    assert "failed" in render_sweep(results)


def test_render_ranks_by_calmar():
    stats = {
        "total_return_pct": 10.0, "cagr_pct": 2.0, "max_drawdown_pct": -10.0,
        "sharpe": 0.5, "trades": 100, "win_rate_pct": 50.0, "profit_factor": 1.1,
        "total_commission": 50.0, "start_equity": 1400.0, "end_equity": 1540.0,
    }
    worse = {**stats, "calmar": 0.1}
    better = {**stats, "calmar": 0.9}
    output = render_sweep([(VARIANTS[0], worse), (VARIANTS[1], better)])

    assert output.index(f"`{VARIANTS[1].name}`") < output.index(f"`{VARIANTS[0].name}`")


def test_split_flags_a_variant_that_only_worked_in_one_half():
    """The whole point of the check: catch a variant carried by one stretch."""
    from backtest.sweep import render_split

    def stats(cagr: float) -> dict:
        return {
            "cagr_pct": cagr, "max_drawdown_pct": -10.0, "total_return_pct": cagr * 5,
            "sharpe": 0.5, "calmar": 0.5, "trades": 100, "win_rate_pct": 55.0,
            "profit_factor": 1.2, "total_commission": 100.0,
            "start_equity": 1400.0, "end_equity": 1600.0,
        }

    steady, lucky = VARIANTS[0], VARIANTS[1]
    output = render_split({
        "full": [(steady, stats(8.0)), (lucky, stats(9.0))],
        "first_half": [(steady, stats(7.0)), (lucky, stats(25.0))],
        "second_half": [(steady, stats(9.0)), (lucky, stats(-6.0))],
    })

    lines = {line.split("|")[1].strip(): line for line in output.splitlines() if "|" in line}
    assert "yes" in lines[f"`{steady.name}`"]
    assert "NO" in lines[f"`{lucky.name}`"]


def test_split_tolerates_a_failed_variant():
    from backtest.sweep import render_split

    output = render_split({
        "full": [(VARIANTS[0], {"error": "boom"})],
        "first_half": [(VARIANTS[0], {"error": "boom"})],
        "second_half": [(VARIANTS[0], {"error": "boom"})],
    })
    assert "n/a" in output
