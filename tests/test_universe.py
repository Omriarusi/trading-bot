"""Universe definitions.

The universe decides what the bot is allowed to buy, so the guarantees here
are about legality and liquidity rather than performance.
"""

from __future__ import annotations

import pytest

from bot.universe_list import (
    PRIIPS_BLOCKED_SIGNALS_ONLY,
    UNIVERSES,
    get_universe,
)


def test_every_universe_is_deduplicated_and_stable():
    for name, symbols in UNIVERSES.items():
        if not symbols:
            continue
        resolved = get_universe(name)
        assert len(resolved) == len(set(resolved)), f"{name} has duplicates"
        assert resolved == get_universe(name), f"{name} is not order-stable"


def test_no_universe_contains_a_priips_blocked_instrument():
    """An IBIE account cannot buy US-domiciled ETFs.

    Including one would generate orders the broker rejects on every run, and
    the execution layer would refuse them anyway — so catch it here, where the
    fix is a one-line edit rather than a puzzling production failure.
    """
    for name, symbols in UNIVERSES.items():
        if not symbols:
            continue
        blocked = set(get_universe(name)) & PRIIPS_BLOCKED_SIGNALS_ONLY
        assert not blocked, f"{name} contains signal-only instruments: {sorted(blocked)}"


def test_the_benchmark_is_never_tradable():
    """SPY drives the regime filter and must not also be a trade candidate."""
    for name, symbols in UNIVERSES.items():
        if symbols:
            assert "SPY" not in get_universe(name)


def test_unknown_universe_names_are_rejected():
    with pytest.raises(KeyError, match="Unknown universe"):
        get_universe("does_not_exist")


def test_an_empty_universe_explains_how_to_fix_it(monkeypatch):
    """Before the first refresh, sp500 is empty; the error must say so."""
    monkeypatch.setitem(UNIVERSES, "empty_for_test", ())
    with pytest.raises(KeyError, match="refresh_sp500"):
        get_universe("empty_for_test")


def test_symbols_look_like_us_tickers():
    for name, symbols in UNIVERSES.items():
        if not symbols:
            continue
        for symbol in get_universe(name):
            assert symbol.isupper(), f"{name}: {symbol} is not upper-case"
            assert 1 <= len(symbol) <= 7, f"{name}: {symbol} has an implausible length"
            assert symbol.replace(".", "").isalpha(), f"{name}: {symbol} is malformed"
