# trading-bot

An automated swing-trading bot for an Interactive Brokers account, built to run
unattended on GitHub Actions.

**Status: paper trading. It is not connected to real money.**

---

## What it does

Once or twice a trading day it wakes up, reads the account from IBKR, and:

1. Places a stop on any open position that has none.
2. Decides whether the market regime permits new risk.
3. Exits positions that hit their target or have gone stale.
4. Ratchets trailing stops upward on positions that are working.
5. Opens new positions, if there is room and the setup is there.

Then it commits its state and exits. Everything it needs to know next time is
either in the repository or at the broker.

## The strategy

**Buy short-term weakness inside a long-term uptrend.**

A stock is a candidate when *all* of these hold:

| Filter | Rule | Why |
| --- | --- | --- |
| Market regime | SPY above its 200-day average, volatility not in its top decile | Buying dips works in uptrends and fails badly in sustained declines, where every dip keeps going. This one gate matters more than every entry rule combined. |
| Trend | Price above its own 200-day average | Trade pullbacks, not collapses. |
| Momentum | Positive 6-month return | Distinguishes a pullback from a topping pattern. |
| Trigger | 3-period RSI below 25 | Enter on weakness rather than chasing strength. Better fills, which matters when commission is ~0.3% a side. |
| Liquidity | Over $50m traded daily, $5–$400 | A wide spread costs more than the edge. |

Exits, whichever comes first: a close back above the 10-day average or RSI
above 70 (target), 12 days elapsed (time stop), or the stop being hit.

Positions are sized so that **being wrong costs a fixed, known fraction of the
account** — 2% by default. The stop sits 2.5 ATR below entry, and the share
count falls out of that distance, so a volatile stock gets fewer shares for the
same dollar risk.

SPY is read as a signal and **never traded**: on an IBIE (Ireland) account,
PRIIPs blocks US-domiciled ETFs. The tradable universe is single stocks only.

## Risk controls

Sizing is the minimum of five independent ceilings — risk budget, 30% of equity
per position, available settled cash, 1% of average daily volume, and a hard
$500 per-order cap. Adding a limit can only ever make a position smaller.

Three circuit breakers halt new positions: a 6% daily loss, a 25% drawdown from
the equity peak (which requires a manual override to clear), and price data
more than 5 days stale. **A halt never cancels protection on open positions** —
their stops stay live at the broker.

The most important property: **every stop is a real GTC order resting at IBKR**,
submitted as a child of the entry so it activates automatically on fill. If
this repository is deleted, GitHub Actions goes down, or the bot crashes
mid-run, open positions are still protected. Protection does not depend on the
software ever running again.

`config.yaml` refuses to load a configuration that would break these
guarantees — a stop of zero, a trailing stop tighter than the initial stop, or
a combination of position count and per-trade risk that could exceed the
drawdown halt in a single day.

## Running it

```bash
pip install -r requirements-dev.txt

python -m bot.cli validate-config   # check config.yaml
python -m bot.cli scan              # today's candidates, touches no broker
python -m bot.cli check-account     # what IBKR says about the account
python -m bot.cli run               # one full run
python -m bot.cli backtest --years 8
pytest -q
```

Execution modes: `dry_run` computes and logs orders without sending them,
`paper` trades an IBKR paper account, `live` uses real money and refuses to
start without an explicit confirmation flag.

Connecting to IBKR: **[docs/IBKR_SETUP.md](docs/IBKR_SETUP.md)**.

## Layout

```
bot/
  config.py        typed config + the safety validations
  indicators.py    RSI, ATR, moving averages — pure functions
  signals.py       the strategy: regime, entries, exits, stops
  risk.py          position sizing and circuit breakers
  data.py          free price data with fallback and caching
  state.py         what the broker cannot tell us, committed between runs
  engine.py        one run, start to finish
  broker/          ibkr.py (live), paper.py (simulated), base.py (interface)
backtest/          replays the strategy over history
.github/workflows/ ci.yml, backtest.yml, trade.yml
```

The backtester calls the same `signals.py` and `risk.py` the live engine does.
A backtest of a reimplementation measures the reimplementation.

## Measured results

2016–2026, S&P 500 (503 symbols), $1,400 start, commission and slippage
charged on every fill:

| | Strategy | Buy-and-hold |
| --- | ---: | ---: |
| Total return | +78.6% | +271.8% |
| CAGR | +6.5% | ~+14.0% |
| **Max drawdown** | **−21.0%** | **−33.7%** |
| Sharpe | 0.61 | — |
| Calmar | 0.31 | ~0.42 |

**It underperforms buy-and-hold on both raw and risk-adjusted return on this
universe**, though its drawdown is meaningfully shallower. The picture is
notably better on a broader universe that mixes in higher-beta names outside
the index (Calmar 0.60 there, against 0.31 restricted to the S&P 500) — see
`docs/STRATEGY.md` for that comparison. Run this strategy because you want a
shallower drawdown than the index, not because you expect to beat it on
return — and note that an IBIE account *can* buy a UCITS index fund (CSPX,
VUAA), so buy-and-hold is a real alternative, not a hypothetical one.

The original, untouched version of this strategy actually **loses money** on
the S&P 500 alone (Calmar −0.01) and fails an out-of-sample check split
across the ten-year history. What fixed it, what failed — including two
specifically requested changes that were tested and rejected — is written up
in **[docs/STRATEGY.md](docs/STRATEGY.md)**.

## What this cannot do

It cannot promise a profit. High risk means high variance in *both* directions:
the same settings that allow a good year allow losing a large part of the
account. The strategy is a well-documented, widely-used pattern with no secret
edge — its value is in being executed consistently and with fixed risk, not in
being clever.

Read the drawdown before the return. Backtests overstate live results: they
assume every intended fill happened, and the universe is today's liquid names,
which quietly excludes companies that were liquid then and are not now. Nine
years is also one sample — it contains a single crash, a single bear market,
and a long bull run.

## Disclaimer

Not financial advice. Trading involves substantial risk of loss. Use the paper
account until you have seen the behaviour yourself over a meaningful period.
