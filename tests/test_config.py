"""Configuration loading and the safety validations.

Each rejection test corresponds to a config that would let the bot lose money
in a way no strategy quality could offset.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import bot.config as config_module
from bot.config import AccountType, Config, ConfigError, ExecutionMode


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(body))
    return path


BASE = """
    account:
      entity: IBIE
      type: cash
      assumed_equity: 1400.0
    risk:
      risk_per_trade_pct: 2.0
      max_concurrent_positions: 4
      min_position_notional: 250.0
"""


def test_defaults_are_valid():
    config_module._validate(Config())


def test_committed_config_is_valid():
    """The config the bot actually ships with must pass every check."""
    config_module._validate(config_module.load())


def test_loads_a_minimal_file(tmp_path):
    cfg = config_module.load(write_config(tmp_path, BASE))
    assert cfg.account.assumed_equity == 1400.0
    assert cfg.risk.risk_per_trade_pct == 2.0
    assert cfg.execution.mode is ExecutionMode.DRY_RUN  # default


def test_unknown_option_is_an_error_not_a_no_op(tmp_path):
    """A typo must fail loudly rather than silently keep the default.

    Misspelling ``risk_per_trade_pct`` would otherwise leave the default in
    place and trade a size the user never chose.
    """
    path = write_config(tmp_path, """
        risk:
          risk_per_trade_pcnt: 8.0
    """)
    with pytest.raises(ConfigError, match="unknown option"):
        config_module.load(path)


def test_wrong_type_is_rejected(tmp_path):
    path = write_config(tmp_path, """
        risk:
          max_concurrent_positions: "four"
    """)
    with pytest.raises(ConfigError, match="expected a number"):
        config_module.load(path)


def test_invalid_enum_value_lists_the_valid_ones(tmp_path):
    path = write_config(tmp_path, """
        execution:
          mode: yolo
    """)
    with pytest.raises(ConfigError, match="dry_run"):
        config_module.load(path)


def test_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        config_module.load(tmp_path / "nope.yaml")


# ---------------------------------------------------------------- validation


def test_absurd_risk_per_trade_is_rejected(tmp_path):
    path = write_config(tmp_path, """
        risk:
          risk_per_trade_pct: 50.0
    """)
    with pytest.raises(ConfigError, match="risk_per_trade_pct"):
        config_module.load(path)


def test_zero_risk_per_trade_is_rejected(tmp_path):
    path = write_config(tmp_path, "risk:\n  risk_per_trade_pct: 0.0\n")
    with pytest.raises(ConfigError, match="risk_per_trade_pct"):
        config_module.load(path)


def test_missing_stop_is_rejected(tmp_path):
    """Trading without a stop is the one thing that is never configurable."""
    path = write_config(tmp_path, """
        risk:
          atr_stop_multiple: 0.0
    """)
    with pytest.raises(ConfigError, match="stop is mandatory"):
        config_module.load(path)


def test_trailing_stop_tighter_than_the_initial_stop_is_rejected(tmp_path):
    path = write_config(tmp_path, """
        risk:
          atr_stop_multiple: 3.0
          atr_trail_multiple: 1.0
    """)
    with pytest.raises(ConfigError, match="tighter than"):
        config_module.load(path)


def test_combined_risk_exceeding_the_drawdown_halt_is_rejected(tmp_path):
    """Four positions at 8% risk can lose 32% in a day against a 25% halt."""
    path = write_config(tmp_path, """
        risk:
          risk_per_trade_pct: 8.0
          max_concurrent_positions: 4
          max_position_pct: 25.0
        circuit_breakers:
          max_drawdown_halt_pct: 25.0
          daily_loss_halt_pct: 30.0
    """)
    with pytest.raises(ConfigError, match="exceeds max_drawdown_halt_pct"):
        config_module.load(path)


def test_daily_halt_below_a_single_stop_out_is_rejected(tmp_path):
    """A breaker that trips on one ordinary loss is a broken breaker."""
    path = write_config(tmp_path, """
        risk:
          risk_per_trade_pct: 3.0
          max_position_pct: 100.0
        circuit_breakers:
          daily_loss_halt_pct: 1.0
    """)
    with pytest.raises(ConfigError, match="halt on a routine loss"):
        config_module.load(path)


def test_unreachable_position_count_is_rejected(tmp_path):
    """Four $250 minimum positions do not fit in a $400 account."""
    path = write_config(tmp_path, """
        account:
          assumed_equity: 400.0
        risk:
          min_position_notional: 250.0
          max_concurrent_positions: 4
    """)
    with pytest.raises(ConfigError, match="could never"):
        config_module.load(path)


def test_leverage_in_a_cash_account_is_rejected(tmp_path):
    path = write_config(tmp_path, """
        account:
          type: cash
        risk:
          max_gross_exposure_pct: 150.0
    """)
    with pytest.raises(ConfigError, match="requires margin"):
        config_module.load(path)


def test_exit_threshold_below_entry_threshold_is_rejected(tmp_path):
    """Otherwise every entry exits on the bar it was opened."""
    path = write_config(tmp_path, """
        strategy:
          rsi_entry_max: 80.0
          rsi_exit_min: 70.0
    """)
    with pytest.raises(ConfigError, match="must exceed rsi_entry_max"):
        config_module.load(path)


def test_history_too_short_for_the_indicators_is_rejected(tmp_path):
    path = write_config(tmp_path, """
        data:
          history_days: 100
        strategy:
          trend_ma_days: 200
    """)
    with pytest.raises(ConfigError, match="too short to warm up"):
        config_module.load(path)


def test_order_cap_below_the_minimum_position_is_rejected(tmp_path):
    path = write_config(tmp_path, """
        execution:
          max_order_notional: 100.0
        risk:
          min_position_notional: 250.0
    """)
    with pytest.raises(ConfigError, match="every"):
        config_module.load(path)


# ------------------------------------------------------------- env overrides


def test_execution_mode_can_be_overridden_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_EXECUTION_MODE", "paper")
    cfg = config_module.load(write_config(tmp_path, BASE))
    assert cfg.execution.mode is ExecutionMode.PAPER


def test_an_invalid_mode_override_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_EXECUTION_MODE", "nonsense")
    with pytest.raises(ConfigError, match="BOT_EXECUTION_MODE"):
        config_module.load(write_config(tmp_path, BASE))


def test_manual_override_can_be_set_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_MANUAL_OVERRIDE", "true")
    cfg = config_module.load(write_config(tmp_path, BASE))
    assert cfg.circuit_breakers.manual_override


# -------------------------------------------------------------- entity rules


@pytest.mark.parametrize("entity", ["IBIE", "IBCE", "IBUK", "ibie"])
def test_eea_entities_are_priips_restricted(entity):
    cfg = config_module._build(
        config_module.AccountConfig, {"entity": entity}, "account"
    )
    assert cfg.priips_restricted


@pytest.mark.parametrize("entity", ["IBLLC", "IB LLC", "IBKR"])
def test_us_entities_are_not_priips_restricted(entity):
    cfg = config_module._build(
        config_module.AccountConfig, {"entity": entity}, "account"
    )
    assert not cfg.priips_restricted


def test_execution_mode_capabilities():
    assert not ExecutionMode.DRY_RUN.sends_orders
    assert ExecutionMode.PAPER.sends_orders
    assert not ExecutionMode.PAPER.risks_real_money
    assert ExecutionMode.LIVE.sends_orders
    assert ExecutionMode.LIVE.risks_real_money


def test_cash_accounts_cannot_short():
    assert not config_module.AccountConfig(type=AccountType.CASH).can_short
    assert config_module.AccountConfig(type=AccountType.MARGIN).can_short
