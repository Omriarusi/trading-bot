"""Data layer: validation, fallback, and behaviour when a provider is down."""

from __future__ import annotations

import pandas as pd
import pytest

from bot.config import DataConfig
from bot.data import (
    _SOURCE_FAILURE_THRESHOLD,
    DataError,
    PriceRepository,
    PriceSource,
    _validate_frame,
)


def good_frame(rows: int = 30) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=rows)
    return pd.DataFrame(
        {
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.5, "volume": 1_000_000.0,
        },
        index=index,
    )


# ------------------------------------------------------------- validation


def test_valid_bars_pass_through():
    frame = _validate_frame("AAA", good_frame())
    assert len(frame) == 30
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]


def test_empty_data_is_rejected():
    with pytest.raises(DataError, match="no rows"):
        _validate_frame("AAA", pd.DataFrame())


def test_missing_columns_are_rejected():
    with pytest.raises(DataError, match="missing column"):
        _validate_frame("AAA", good_frame().drop(columns=["volume"]))


def test_negative_prices_are_dropped():
    """A zero or negative price silently produces an enormous position."""
    frame = good_frame()
    frame.iloc[5, frame.columns.get_loc("close")] = -10.0
    assert len(_validate_frame("AAA", frame)) == 29


def test_internally_inconsistent_bars_are_dropped():
    """A high below the low means the source mangled the row."""
    frame = good_frame()
    frame.iloc[3, frame.columns.get_loc("high")] = 50.0
    assert len(_validate_frame("AAA", frame)) == 29


def test_out_of_order_bars_are_sorted():
    frame = good_frame().iloc[::-1]
    assert _validate_frame("AAA", frame).index.is_monotonic_increasing


def test_duplicate_dates_keep_the_last_observation():
    frame = pd.concat([good_frame(5), good_frame(5)])
    assert len(_validate_frame("AAA", frame)) == 5


def test_a_frame_of_only_bad_rows_is_rejected():
    frame = good_frame(3)
    frame[["open", "high", "low", "close"]] = -1.0
    with pytest.raises(DataError, match="no valid bars"):
        _validate_frame("AAA", frame)


# --------------------------------------------------------------- fallback


class StubSource(PriceSource):
    def __init__(self, name: str, fail: bool = False):
        self.name = name
        self.fail = fail
        self.calls = 0

    def fetch(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        self.calls += 1
        if self.fail:
            raise DataError(f"{self.name}: down")
        return good_frame()


@pytest.fixture
def repo(tmp_path) -> PriceRepository:
    return PriceRepository(DataConfig(), cache_dir=tmp_path / "cache")


def test_a_failing_source_falls_through_to_the_next(repo):
    primary, backup = StubSource("primary", fail=True), StubSource("backup")
    repo._sources = [primary, backup]

    bars = repo.get("AAA")
    assert bars.source == "backup"
    assert primary.calls == 1 and backup.calls == 1


def test_all_sources_failing_raises(repo):
    repo._sources = [StubSource("a", fail=True), StubSource("b", fail=True)]
    with pytest.raises(DataError, match="all sources failed"):
        repo.get("AAA")


def test_results_are_memoised_within_a_run(repo):
    source = StubSource("only")
    repo._sources = [source]

    repo.get("AAA")
    repo.get("AAA")
    assert source.calls == 1


def test_get_many_reports_failures_without_stopping(repo):
    class Selective(PriceSource):
        name = "selective"

        def fetch(self, symbol: str, lookback_days: int) -> pd.DataFrame:
            if symbol == "BAD":
                raise DataError("delisted")
            return good_frame()

    repo._sources = [Selective()]
    ok, failed = repo.get_many(["AAA", "BAD", "CCC"])

    assert set(ok) == {"AAA", "CCC"}
    assert "BAD" in failed


def test_get_many_handles_an_empty_list(repo):
    assert repo.get_many([]) == ({}, {})


# ------------------------------------------------- provider-down behaviour


def test_a_down_source_is_abandoned_rather_than_retried_for_every_symbol(repo):
    """The fix for a real outage: a dead provider must not eat the run window.

    Retrying every symbol through its full backoff across a ~170 symbol
    universe takes minutes, which can consume the entire scheduled window and
    leave open positions unmanaged.
    """
    down = StubSource("down", fail=True)
    repo._sources = [down]

    symbols = [f"S{i}" for i in range(100)]
    ok, failed = repo.get_many(symbols)

    assert not ok
    assert len(failed) == 100
    # Only the symbols before the breaker tripped should have hit the network.
    assert down.calls <= _SOURCE_FAILURE_THRESHOLD + _MAX_WORKERS_SLACK


def test_a_working_source_is_never_abandoned(repo):
    """A run of bad tickers must not disable a healthy provider."""
    class MostlyGood(PriceSource):
        name = "mostly_good"

        def __init__(self):
            self.calls = 0

        def fetch(self, symbol: str, lookback_days: int) -> pd.DataFrame:
            self.calls += 1
            if symbol.startswith("BAD"):
                raise DataError("delisted")
            return good_frame()

    source = MostlyGood()
    repo._sources = [source]

    # Interleave failures so no run of them reaches the threshold.
    symbols = [f"{'BAD' if i % 2 else 'OK'}{i}" for i in range(60)]
    ok, failed = repo.get_many(symbols)

    assert len(ok) == 30
    assert len(failed) == 30
    assert source.calls == 60  # never skipped


def test_the_cache_is_used_on_a_second_repository(tmp_path):
    cache = tmp_path / "cache"
    source = StubSource("only")

    first = PriceRepository(DataConfig(), cache_dir=cache)
    first._sources = [source]
    first.get("AAA")

    second = PriceRepository(DataConfig(), cache_dir=cache)
    second._sources = [source]
    bars = second.get("AAA")

    assert bars.source == "cache"
    assert source.calls == 1


def test_a_repository_with_no_known_source_is_rejected(tmp_path):
    with pytest.raises(DataError, match="No usable data sources"):
        PriceRepository(DataConfig(sources=("nonsense",)), cache_dir=tmp_path)


# The breaker is checked per thread, so a few extra calls can be in flight
# when it trips. The point is that it is bounded, not that it is exact.
_MAX_WORKERS_SLACK = 8
