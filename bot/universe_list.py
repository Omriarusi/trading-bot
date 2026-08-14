"""Tradable universe definitions.

Kept as an explicit, reviewable list rather than scraped at runtime. A
universe that changes silently between runs makes a backtest impossible to
reproduce, and survivorship bias creeps in the moment the list is derived from
"today's index members".

Every entry is a US-listed common stock. There are deliberately no ETFs: under
PRIIPs, an IBKR Ireland (IBIE) account cannot buy US-domiciled ETFs, so
including them would generate orders the broker rejects. Index ETFs are still
used as *signals* — see ``regime.benchmark`` — just never traded.
"""

from __future__ import annotations

# Large- and mega-cap US names with consistently deep liquidity. Chosen for
# tight spreads rather than for any view on the companies: the strategy's edge
# is measured in tenths of a percent, so a wide spread erases it.
LIQUID_US_LARGE_CAP: tuple[str, ...] = (
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "INTC", "MU", "QCOM", "TXN",
    "ADBE", "CRM", "ORCL", "CSCO", "IBM", "NOW", "INTU", "PANW", "SNPS",
    "CDNS", "KLAC", "LRCX", "AMAT", "ADI", "MRVL", "ANET", "SMCI", "DELL",
    "HPQ", "WDC", "STX", "ON", "NXPI", "TER", "SWKS", "MCHP",
    # Communication services and internet
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "SPOT",
    "UBER", "ABNB", "DASH", "SHOP", "SQ", "PYPL", "SNAP", "PINS", "RBLX",
    # Consumer
    "AMZN", "TSLA", "HD", "LOW", "MCD", "SBUX", "NKE", "TGT", "COST",
    "WMT", "PG", "KO", "PEP", "MDLZ", "CL", "MO", "CMG", "LULU", "ROST",
    "TJX", "YUM", "DRI", "MAR", "HLT", "BKNG", "RCL", "CCL", "DAL", "UAL",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK", "AXP", "V", "MA",
    "COF", "USB", "PNC", "TFC", "BK", "SPGI", "CME", "ICE", "COIN", "HOOD",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "VRTX", "REGN", "ISRG", "SYK", "BSX", "MDT", "CVS",
    "MCK", "HCA", "ZTS", "BIIB", "MRNA",
    # Industrials and energy
    "CAT", "DE", "BA", "GE", "HON", "UNP", "UPS", "FDX", "LMT", "RTX",
    "NOC", "GD", "MMM", "EMR", "ETN", "PH", "CSX", "NSC", "WM",
    "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "MPC", "VLO", "OXY", "HAL",
    "DVN", "FANG", "KMI", "WMB", "OKE",
    # Materials, utilities, real estate
    "LIN", "APD", "SHW", "FCX", "NEM", "NUE", "DOW", "DD",
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE",
    "AMT", "PLD", "CCI", "EQIX", "SPG", "O",
)

# Higher-beta names. Larger moves mean an ATR-sized position is smaller, so
# these do not add dollar risk — they add opportunity and turnover, which is
# what a small account needs to compound at a meaningful rate.
HIGH_BETA_US: tuple[str, ...] = (
    "NVDA", "AMD", "TSLA", "SMCI", "COIN", "HOOD", "MSTR", "PLTR", "SOFI",
    "RIVN", "LCID", "MARA", "RIOT", "CLSK", "AFRM", "UPST", "ROKU", "SNAP",
    "DKNG", "CVNA", "CELH", "ENPH", "FSLR", "ARM", "IONQ", "RKLB", "ASTS",
    "APP", "NET", "DDOG", "SNOW", "CRWD", "ZS", "MDB", "TTD", "SHOP",
)

UNIVERSES: dict[str, tuple[str, ...]] = {
    "liquid_us_large_cap": LIQUID_US_LARGE_CAP,
    "high_beta_us": HIGH_BETA_US,
    # The default blend: large-cap breadth plus a high-beta tail, deduplicated.
    "blended": tuple(dict.fromkeys(LIQUID_US_LARGE_CAP + HIGH_BETA_US)),
}


def get_universe(name: str) -> tuple[str, ...]:
    """Look up a named universe, deduplicated and order-stable."""
    if name not in UNIVERSES:
        raise KeyError(f"Unknown universe {name!r}; available: {sorted(UNIVERSES)}")
    return tuple(dict.fromkeys(UNIVERSES[name]))


# Instruments that must never be sent as an order under PRIIPs, but which the
# strategy legitimately reads as signals. Listed explicitly so the execution
# layer can assert on it rather than relying on the universe being correct.
PRIIPS_BLOCKED_SIGNALS_ONLY: frozenset[str] = frozenset(
    {"SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "TQQQ", "SQQQ", "SOXL", "UVXY", "VXX"}
)
