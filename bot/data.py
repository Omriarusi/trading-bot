"""Market data retrieval.

Price data comes from free public sources, not from IBKR. That decoupling is
deliberate: it means the bot needs no paid market-data subscription, and a
market-data outage degrades signal generation without touching the ability to
manage or exit existing positions.

Every source is behind the same interface and tried in order, so one going
down is a fallback rather than an outage.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from bot.config import REPO_ROOT, DataConfig

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Sources are rate-limited and unsupported; be a polite client.
_REQUEST_TIMEOUT = 20
_RETRY_DELAYS = (1.0, 3.0, 7.0)
_USER_AGENT = "Mozilla/5.0 (compatible; trading-bot/1.0)"

# Concurrent fetches. Enough to hide network latency across a ~170 symbol
# universe, small enough to stay a well-behaved client of a free endpoint.
_MAX_FETCH_WORKERS = 8

# Consecutive failures after which a source is abandoned for the run. A
# handful of bad tickers is normal; this many in a row means the provider
# is down, and retrying every remaining symbol just burns the run window.
_SOURCE_FAILURE_THRESHOLD = 12


class DataError(RuntimeError):
    """Raised when price history cannot be obtained or fails validation."""


@dataclass(frozen=True)
class Bars:
    """Validated OHLCV history for one symbol."""

    symbol: str
    frame: pd.DataFrame
    source: str

    @property
    def last_date(self) -> date:
        return self.frame.index[-1].date()

    def staleness_days(self, as_of: date | None = None) -> int:
        """Calendar days between the last bar and today.

        Compared against a threshold rather than requiring freshness, because
        weekends and holidays legitimately produce a stale-looking series.
        """
        reference = as_of or datetime.now(UTC).date()
        return (reference - self.last_date).days


def _validate_frame(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise and sanity-check a raw price frame.

    Bad data is more dangerous than no data: a zero or negative price silently
    produces an enormous position, and an out-of-order index makes every
    indicator meaningless. Both are rejected here rather than downstream.
    """
    if frame.empty:
        raise DataError(f"{symbol}: no rows returned")

    missing = set(OHLCV_COLUMNS) - set(frame.columns)
    if missing:
        raise DataError(f"{symbol}: missing column(s) {sorted(missing)}")

    frame = frame[OHLCV_COLUMNS].copy()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])

    if frame.empty:
        raise DataError(f"{symbol}: every row failed numeric parsing")

    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    non_positive = (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if non_positive.any():
        log.warning(
            "%s: dropping %d bar(s) with non-positive prices",
            symbol,
            int(non_positive.sum()),
        )
        frame = frame[~non_positive]

    # high must bound the bar; a violation means the source mangled the row.
    inconsistent = (frame["high"] < frame["low"]) | (
        frame["high"] < frame[["open", "close"]].max(axis=1)
    ) | (frame["low"] > frame[["open", "close"]].min(axis=1))
    if inconsistent.any():
        log.warning(
            "%s: dropping %d internally inconsistent bar(s)",
            symbol,
            int(inconsistent.sum()),
        )
        frame = frame[~inconsistent]

    if frame.empty:
        raise DataError(f"{symbol}: no valid bars survived validation")

    frame["volume"] = frame["volume"].fillna(0.0)
    return frame


class PriceSource(ABC):
    """One provider of daily OHLCV history."""

    name: str

    @abstractmethod
    def fetch(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Return a date-indexed OHLCV frame, or raise DataError."""

    def _get(self, url: str, params: dict | None = None) -> requests.Response:
        """HTTP GET with bounded retries on transient failures."""
        last_error: Exception | None = None
        for attempt, delay in enumerate((0.0, *_RETRY_DELAYS)):
            if delay:
                time.sleep(delay)
            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=_REQUEST_TIMEOUT,
                    headers={"User-Agent": _USER_AGENT},
                )
            except requests.RequestException as exc:
                last_error = exc
                log.debug("%s: attempt %d failed: %s", self.name, attempt + 1, exc)
                continue

            # 4xx other than rate limiting will not fix themselves on retry.
            if response.status_code == 429 or response.status_code >= 500:
                last_error = DataError(f"HTTP {response.status_code}")
                log.debug("%s: attempt %d got HTTP %d", self.name, attempt + 1, response.status_code)
                continue
            if not response.ok:
                raise DataError(f"{self.name}: HTTP {response.status_code} for {url}")
            return response

        raise DataError(f"{self.name}: request failed after retries ({last_error})")


class StooqSource(PriceSource):
    """Stooq CSV endpoint. No key, no quota, split- and dividend-adjusted."""

    name = "stooq"
    BASE_URL = "https://stooq.com/q/d/l/"

    def fetch(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        start = (datetime.now(UTC).date() - timedelta(days=lookback_days))
        response = self._get(
            self.BASE_URL,
            params={
                "s": self._stooq_symbol(symbol),
                "d1": start.strftime("%Y%m%d"),
                "d2": datetime.now(UTC).date().strftime("%Y%m%d"),
                "i": "d",
            },
        )

        text = response.text.strip()
        # Stooq answers an unknown ticker with a 200 and a plain-text notice
        # rather than an error status, so the body has to be inspected.
        if not text or text.lower().startswith("no data") or "," not in text.split("\n")[0]:
            raise DataError(f"stooq: no data for {symbol}")

        frame = pd.read_csv(io.StringIO(text))
        frame.columns = [c.strip().lower() for c in frame.columns]
        if "date" not in frame.columns:
            raise DataError(f"stooq: unexpected columns {list(frame.columns)} for {symbol}")

        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).set_index("date")
        return frame

    @staticmethod
    def _stooq_symbol(symbol: str) -> str:
        """Stooq namespaces US listings with a .us suffix."""
        s = symbol.lower().replace("-", ".")
        return s if "." in s and s.endswith(".us") else f"{s}.us"


class YahooSource(PriceSource):
    """Yahoo Finance chart endpoint. Unofficial, so used as a fallback."""

    name = "yahoo"
    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"

    def fetch(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        response = self._get(
            f"{self.BASE_URL}{symbol.upper()}",
            params={
                "range": self._range_for(lookback_days),
                "interval": "1d",
                "events": "div,split",
            },
        )

        try:
            payload = response.json()
            result = payload["chart"]["result"][0]
            timestamps = result["timestamp"]
            quote = result["indicators"]["quote"][0]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise DataError(f"yahoo: unexpected payload for {symbol} ({exc})") from exc

        frame = pd.DataFrame(
            {
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": quote.get("close"),
                "volume": quote.get("volume"),
            },
            index=pd.to_datetime(timestamps, unit="s", utc=True).tz_localize(None).normalize(),
        )

        # Prefer split/dividend-adjusted closes when present, since the
        # indicators assume a continuous price series.
        adjclose = result.get("indicators", {}).get("adjclose")
        if adjclose and adjclose[0].get("adjclose"):
            adj = pd.Series(adjclose[0]["adjclose"], index=frame.index)
            ratio = (adj / frame["close"]).replace([float("inf")], pd.NA)
            if ratio.notna().any():
                for col in ("open", "high", "low", "close"):
                    frame[col] = frame[col] * ratio
        return frame

    @staticmethod
    def _range_for(lookback_days: int) -> str:
        for threshold, label in ((30, "1mo"), (90, "3mo"), (180, "6mo"), (365, "1y"), (730, "2y"), (1825, "5y")):
            if lookback_days <= threshold:
                return label
        return "10y"


SOURCES: dict[str, type[PriceSource]] = {
    "stooq": StooqSource,
    "yahoo": YahooSource,
}


class PriceRepository:
    """Fetches history with fallback across sources and an on-disk cache.

    The cache is keyed by symbol and holds the full series. On a scheduled run
    it turns a cold start of a 150-symbol universe into a handful of requests,
    which matters because these endpoints throttle aggressively.
    """

    def __init__(self, cfg: DataConfig, cache_dir: Path | None = None):
        self.cfg = cfg
        self.cache_dir = cache_dir or (REPO_ROOT / cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._sources = [SOURCES[name]() for name in cfg.sources if name in SOURCES]
        if not self._sources:
            raise DataError(
                f"No usable data sources in {cfg.sources}; known: {sorted(SOURCES)}"
            )
        self._memo: dict[str, Bars] = {}
        # get() is called from several threads by get_many(). The lock guards
        # the memo; the worst a race would cost is a duplicate HTTP request,
        # but duplicating requests is exactly what annoys a throttled endpoint.
        self._lock = threading.Lock()
        self._consecutive_failures: dict[str, int] = {}

    def get(self, symbol: str, use_cache: bool = True) -> Bars:
        """Fetch one symbol, trying each source in order."""
        with self._lock:
            if symbol in self._memo:
                return self._memo[symbol]

        if use_cache:
            cached = self._read_cache(symbol)
            if cached is not None:
                with self._lock:
                    self._memo[symbol] = cached
                return cached

        errors: list[str] = []
        for source in self._sources:
            if self._is_tripped(source.name):
                errors.append(f"{source.name}: skipped, source marked down")
                continue
            try:
                raw = source.fetch(symbol, self.cfg.history_days)
                frame = _validate_frame(symbol, raw)
                bars = Bars(symbol=symbol, frame=frame, source=source.name)
                self._write_cache(symbol, bars)
                with self._lock:
                    self._memo[symbol] = bars
                    self._consecutive_failures[source.name] = 0
                return bars
            except DataError as exc:
                errors.append(f"{source.name}: {exc}")
                self._record_failure(source.name)
            except Exception as exc:
                errors.append(f"{source.name}: unexpected {type(exc).__name__}: {exc}")
                self._record_failure(source.name)

        raise DataError(f"{symbol}: all sources failed ({'; '.join(errors)})")

    def _record_failure(self, source_name: str) -> None:
        """Count a failure, tripping the source's breaker at the threshold."""
        with self._lock:
            count = self._consecutive_failures.get(source_name, 0) + 1
            self._consecutive_failures[source_name] = count
            if count == _SOURCE_FAILURE_THRESHOLD:
                log.error(
                    "data source %r failed %d times in a row; skipping it for the "
                    "rest of this run",
                    source_name,
                    count,
                )

    def _is_tripped(self, source_name: str) -> bool:
        """Whether a source has been abandoned for this run.

        Without this, a provider outage costs every symbol its full retry
        budget. Across a ~170 symbol universe that is several minutes of
        sleeping, which can consume the entire scheduled window and leave open
        positions unmanaged. Failing fast preserves the time to do the part
        that matters: reconcile with the broker and manage what is held.
        """
        with self._lock:
            return (
                self._consecutive_failures.get(source_name, 0)
                >= _SOURCE_FAILURE_THRESHOLD
            )

    def get_many(self, symbols: list[str], use_cache: bool = True) -> tuple[dict[str, Bars], dict[str, str]]:
        """Fetch many symbols, returning what succeeded and why the rest failed.

        A partial universe is a normal, tradable outcome. One delisted ticker
        must not stop the bot from managing the rest of the portfolio.

        Fetched in parallel. Sequentially, a ~170 symbol universe takes long
        enough that a scheduled run can miss its window entirely — and these
        endpoints throttle, so the retry backoff compounds the delay. The pool
        is deliberately small: enough to hide the latency, not enough to look
        like abuse of a free service.
        """
        ok: dict[str, Bars] = {}
        failed: dict[str, str] = {}
        if not symbols:
            return ok, failed

        workers = min(_MAX_FETCH_WORKERS, len(symbols))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self.get, symbol, use_cache): symbol for symbol in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    ok[symbol] = future.result()
                except DataError as exc:
                    failed[symbol] = str(exc)
                    log.warning("data: %s", exc)
                # A source raising something unexpected must not end the run.
                except Exception as exc:
                    failed[symbol] = f"unexpected {type(exc).__name__}: {exc}"
                    log.warning("data: %s failed unexpectedly: %s", symbol, exc)

        log.info("fetched %d/%d symbols", len(ok), len(symbols))
        return ok, failed

    def _cache_path(self, symbol: str) -> Path:
        safe = symbol.upper().replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe}.csv"

    def _read_cache(self, symbol: str) -> Bars | None:
        path = self._cache_path(symbol)
        if not path.exists():
            return None
        # A cache written earlier today is current; anything older gets
        # refetched so a scheduled run never trades on yesterday's prices.
        age = time.time() - path.stat().st_mtime
        if age > 6 * 3600:
            return None
        try:
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
            return Bars(symbol=symbol, frame=_validate_frame(symbol, frame), source="cache")
        except (OSError, ValueError, DataError) as exc:
            log.debug("cache read failed for %s: %s", symbol, exc)
            return None

    def _write_cache(self, symbol: str, bars: Bars) -> None:
        try:
            bars.frame.to_csv(self._cache_path(symbol))
        except OSError as exc:
            log.debug("cache write failed for %s: %s", symbol, exc)
