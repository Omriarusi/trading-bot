# Strategy: what it does, and what the evidence says

A record of what was measured and why the settings are what they are. Read
this before changing a number in `config.yaml`.

---

## The idea

Buy a stock that is in a long-term uptrend but is short-term oversold, while
the broad market is also trending up. Sell when it bounces, when the trailing
stop is hit, or when it has gone nowhere for long enough.

The pieces, in order of how much they matter:

**1. The market regime gate.** No new position is opened unless the benchmark
is above its 200-day average and its short-term volatility is outside its own
top decile. Buying dips works while the market trends up and fails badly in a
sustained decline, where every dip keeps going. This one gate does more for
the return profile than every entry rule combined.

**2. Position sizing from the stop.** The stop sits 2.5 ATR below the entry,
and the share count is whatever makes that distance cost a fixed fraction of
equity. A volatile stock gets fewer shares. That is what makes risk comparable
across a universe where a 1% move means very different things.

**3. The entry.** Price above its own 200-day average, positive 6-month
momentum, and a 3-period RSI below the entry threshold. The trend filters keep
us out of collapses; the RSI trigger means buying weakness rather than chasing
strength, which matters when commission is a meaningful share of the trade.

**4. The exit.** A trailing stop that only ever ratchets up, an RSI target, and
a time stop. Which of these fires most often is a design decision with large
consequences — see below.

---

## What the first backtest showed

2017-05-31 to 2026-08-14, 187 US single stocks, $1,400 starting equity,
IBKR tiered commission and 5bp slippage charged on every fill.

| | Baseline |
| --- | ---: |
| Total return | +45.4% |
| CAGR | +4.2% |
| Max drawdown | -23.4% |
| Sharpe | 0.39 |
| Trades | 1,078 |
| Win rate | 62% |
| Profit factor | 1.11 |
| Commission | $755 |
| **Commission as a share of net profit** | **119%** |

Benchmark buy-and-hold over the same period: **+272.8%**, max drawdown -33.7%.

The strategy was right 62% of the time and still barely made money. Two facts
explain it, and they turned out to be the whole problem:

**Costs swamped the edge.** $755 of commission against $634 of net profit. At
this account size a ~$340 position pays roughly 0.2% per round trip, and 1,078
trades of a thin edge cannot carry that.

**The exit cut winners and let losers run.** Average win +2.53%, average loss
-3.81%. The moving-average exit fires within a day or two of a bounce, while a
loss runs all the way to the 2.5 ATR stop. A high win rate with a
worse-than-1:1 payoff is a coin flip with a fee attached.

---

## What was changed, and what it bought

Six variants, each written down as a hypothesis about one of those two facts
before its result was known, then run over identical data. (These first
figures are from the universe traded at the time, a large-cap-plus-high-beta
blend. The universe was later restricted to the S&P 500 — see below — which
changes the numbers but not the ranking or the conclusion.)

| Variant | Return | Max DD | Sharpe | Calmar | Trades | PF | Comm/PnL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `combined_hot` | +158.5% | -15.9% | 0.88 | 0.68 | 488 | 1.37 | 15% |
| **`combined`** | **+116.6%** | **-14.5%** | **0.82** | **0.60** | 459 | 1.36 | 20% |
| `winners_run` | +90.1% | -20.4% | 0.64 | 0.35 | 725 | 1.22 | 40% |
| `selective` | +72.6% | -16.8% | 0.59 | 0.36 | 849 | 1.21 | 59% |
| `baseline` | +41.9% | -23.4% | 0.37 | 0.17 | 1,076 | 1.10 | 128% |
| `fewer_bigger` | -14.4% | -25.1% | -0.22 | -0.07 | 256 | 0.87 | 89% |

Both hypotheses were confirmed. Handing the exit to the trailing stop lifted
the profit factor from 1.11 to 1.23; demanding a deeper pullback lifted it to
1.21; together they reached 1.35. Commission fell from 119% of net profit to
15% — not by trading cheaper, but by trading less and holding longer, so a
fixed fee is spread over a larger move.

**The most instructive result is the one that failed.** `fewer_bigger` —
concentrating into three larger positions purely to cut commission — *lost
money*, and was the worst variant of the six. Cutting the number of positions
without fixing the exit removed diversification and kept every flaw that made
the baseline weak. The same change inside `combined`, where the exit had been
fixed first, helped. Cost reduction was not the fix; it was an amplifier of
whatever the strategy already did.

Reproduce any of this with:

```bash
python -m bot.cli sweep --years 10 --universe blended
```

---

## Restricting to the S&P 500, and two more requested changes

The universe was later narrowed to the S&P 500 (503 symbols) by request. On
this narrower universe the whole picture is harsher: the **original,
untouched strategy loses money** — total return -2.7% over ten years, Calmar
-0.01 — and fails the out-of-sample check (profitable in the first half,
losing in the second). Of the six historical variants, `combined` is the
*only* one that stays profitable in both halves:

| Variant | Return | Calmar | Consistent (both halves profitable) |
| --- | ---: | ---: | :---: |
| **`combined`** | **+78.6%** | **0.31** | **yes** |
| `combined_hot` | +54.1% | 0.19 | yes |
| `ma50` (alone, vs. the original baseline) | +16.1% | 0.06 | no |
| `baseline` (the original strategy) | -2.7% | -0.01 | no |
| `winners_run` | -3.5% | -0.01 | no |
| `fewer_bigger` | -12.4% | -0.06 | no |
| `selective` | -16.6% | -0.08 | no |

Two further changes were requested and tested against `combined` specifically
— a 50-day trend filter instead of 200-day, and volume confirmation in both
possible directions (heavy volume as capitulation, light volume as an orderly
pullback). All three passed the out-of-sample check, and all three made
`combined` worse:

| Variant | Return | Calmar |
| --- | ---: | ---: |
| **`combined` (no volume filter, 200-day trend)** | **+78.6%** | **0.31** |
| `combined_quiet` (volume ≤ 1.0x average) | +47.3% | 0.22 |
| `combined_ma50` (50-day trend filter) | +35.9% | 0.18 |
| `combined_capitulation` (volume ≥ 1.5x average) | +36.2% | 0.16 |

The faster trend filter let in more candidates, but the extra ones were lower
quality — names rolling over rather than pulling back inside an intact
uptrend. Both volume readings simply filtered out good trades without
improving what remained; "heavy volume" (capitulation) did more damage than
"light volume" (orderly pullback), but neither beat trading with no volume
filter at all. None of the three were adopted.

Reproduce this on the current universe with:

```bash
python -m bot.cli sweep --years 10
```

*(A methodological note for anyone re-running this: the first two attempts at
this specific comparison were wrong, for two different reasons, before the
numbers above. The sweep's six historical variants were originally built by
mutating whatever config was passed in — once `combined` was adopted into
`config.yaml`, that made them collapse into five identical copies of one
config, since re-applying an already-present change is a no-op. Separately,
the sweep workflow's own `workflow_dispatch` input declared a hardcoded
default universe, which silently overrode a config-driven fallback on manual
runs. Both are fixed: the historical variants are now pinned to a fixed
baseline configuration regardless of what is committed, and the workflow
carries no default universe of its own. If a sweep run's "combined" and
"baseline" rows ever look identical again, or a run's universe doesn't match
what you asked for, look here first.)*

---

## Honest limits

**It still underperforms buy-and-hold on raw return.** On the S&P 500,
`combined` returned +78.6% against the index's own +271.8% over the same ten
years. It wins on risk-adjusted terms — a max drawdown of -21.0% against the
index's -33.7%, and a positive Calmar where the untouched original strategy
has a negative one — but if the goal is the largest number after ten years,
holding an index fund beat this. Anyone running this strategy should be doing
it because they want the shallower drawdown, not because they expect to beat
the market on return.

Worth knowing: an IBIE account cannot buy SPY, but a UCITS equivalent
(CSPX, VUAA) is permitted and tracks the same index. Buy-and-hold is a real
option, not a hypothetical one.

**The backtest is optimistic.** It assumes every intended fill happened at the
next open, and the universe is today's liquid names — which quietly excludes
companies that were liquid in 2017 and are not now. Survivorship bias flatters
the result by an unknown amount. Treat the drawdown as the honest number and
the return as an upper bound.

**Nine years is one sample.** It contains exactly one crash (2020), one bear
market (2022), and a long bull run. A strategy gated on "benchmark above its
200-day average" has only been tested against a handful of regime changes.

**Nothing here is a secret edge.** Trend-plus-pullback is a widely published
pattern. Its value is in being executed consistently with fixed risk, not in
being clever.

---

## Changing the strategy

The sweep runs automatically on any push touching `bot/signals.py`,
`bot/risk.py`, `bot/indicators.py`, `config.yaml`, or `backtest/`, so the
effect of a change is visible where the change was made.

Two rules worth keeping:

**Write the hypothesis before the result.** The variants in
`backtest/sweep.py` each name the weakness they target. That is what makes a
good result evidence rather than a coincidence — searching a large parameter
space over ~1,000 trades will always find something that fits this history.

**Check both halves.** The sweep reports each variant over the first and
second half of the history separately and flags any that lost money in either.
A variant that wins the first half and loses the second was fitted to the
first half, whatever its full-period number says.
