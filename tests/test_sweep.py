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
