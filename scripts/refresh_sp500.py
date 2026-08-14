"""Regenerate the S&P 500 universe from a live constituent list.

Run this rather than editing ``bot/sp500.py`` by hand. Index membership
changes several times a year, and hand-maintaining it means the bot slowly
drifts into trading names that left the index and missing ones that joined.

    python scripts/refresh_sp500.py

Writes ``bot/sp500.py`` with a dated snapshot. Committing that file is what
makes a backtest reproducible: the universe is then a fact about a specific
date, not whatever the internet returned the day someone happened to run it.

Requires network access, so it will not run in a restricted sandbox. The
`Refresh S&P 500` workflow runs it on a GitHub runner.
"""

from __future__ import annotations

import io
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "bot" / "sp500.py"

_USER_AGENT = "Mozilla/5.0 (compatible; trading-bot/1.0)"
_TIMEOUT = 30

# The index has had 500-505 members for decades (a few companies have two
# share classes). Anything far outside that means the source changed shape and
# the result must not be written.
_MIN_EXPECTED = 480
_MAX_EXPECTED = 520


def from_wikipedia() -> list[str]:
    """Current constituents from Wikipedia. Updated within a day of a change."""
    import pandas as pd

    response = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": _USER_AGENT},
        timeout=_TIMEOUT,
    )
    response.raise_for_status()

    for table in pd.read_html(io.StringIO(response.text)):
        columns = {str(c).strip().lower() for c in table.columns}
        if "symbol" in columns:
            return [str(s).strip().upper() for s in table["Symbol"].tolist()]
    raise RuntimeError("no table with a Symbol column found on the Wikipedia page")


def from_datahub() -> list[str]:
    """Fallback: a CSV snapshot hosted on GitHub. Lags Wikipedia slightly."""
    response = requests.get(
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
        "main/data/constituents.csv",
        headers={"User-Agent": _USER_AGENT},
        timeout=_TIMEOUT,
    )
    response.raise_for_status()

    import pandas as pd

    frame = pd.read_csv(io.StringIO(response.text))
    column = next((c for c in frame.columns if c.strip().lower() == "symbol"), None)
    if column is None:
        raise RuntimeError(f"no Symbol column in CSV, got {list(frame.columns)}")
    return [str(s).strip().upper() for s in frame[column].tolist()]


def normalise(symbols: list[str]) -> list[str]:
    """Clean, validate, and sort the raw ticker list.

    Class shares appear as ``BRK.B`` on Wikipedia and ``BRK-B`` elsewhere; the
    data layer expects the dotted form, and Stooq maps it correctly.
    """
    cleaned: list[str] = []
    for raw in symbols:
        symbol = raw.replace("-", ".").upper().strip()
        # A valid US ticker is letters, optionally with a single class suffix.
        if not re.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?", symbol):
            print(f"  skipping malformed symbol {raw!r}", file=sys.stderr)
            continue
        cleaned.append(symbol)

    unique = sorted(set(cleaned))
    if not _MIN_EXPECTED <= len(unique) <= _MAX_EXPECTED:
        raise RuntimeError(
            f"got {len(unique)} symbols, expected {_MIN_EXPECTED}-{_MAX_EXPECTED}. "
            "The source probably changed shape; refusing to overwrite the "
            "universe with something that may be wrong."
        )
    return unique


def render(symbols: list[str], source: str) -> str:
    """Render the module, wrapped so the committed diff stays readable."""
    lines: list[str] = []
    row: list[str] = []
    for symbol in symbols:
        row.append(f'"{symbol}",')
        if len(row) == 8:
            lines.append("    " + " ".join(row))
            row = []
    if row:
        lines.append("    " + " ".join(row))

    today = datetime.now(UTC).date().isoformat()
    return f'''"""S&P 500 constituents — generated, do not edit by hand.

Snapshot taken {today} from {source}.
Regenerate with: python scripts/refresh_sp500.py

Important when reading any backtest run against this list: these are the
constituents *today*, not on each historical date. Companies enter the index
after performing well and leave after performing badly, so backtesting
today's membership over past years quietly assumes foreknowledge of which
companies would succeed. It inflates historical returns by an amount this
repository does not attempt to estimate. It does not affect forward trading,
which is the actual use.
"""

from __future__ import annotations

SP500_SNAPSHOT_DATE = "{today}"

SP500: tuple[str, ...] = (
{chr(10).join(lines)}
)
'''


def main() -> int:
    sources = [("Wikipedia", from_wikipedia), ("datasets/s-and-p-500-companies", from_datahub)]

    for name, fetch in sources:
        try:
            print(f"Fetching from {name}...")
            symbols = normalise(fetch())
        # Any failure here means try the next source, not abort.
        except Exception as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            continue

        OUTPUT.write_text(render(symbols, name))
        print(f"Wrote {len(symbols)} symbols to {OUTPUT.relative_to(REPO_ROOT)}")
        print(f"  first five: {symbols[:5]}")
        return 0

    print("Every source failed; bot/sp500.py left unchanged.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
