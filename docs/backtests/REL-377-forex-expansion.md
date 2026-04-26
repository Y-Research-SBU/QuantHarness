# REL-377: Forex Universe Expansion — Backtest Results

**Issue:** REL-377 (Linear, QuantAgent project)
**Date:** 2026-04-25 / 2026-04-26
**Source data:** `backtest_results/backtest_REL377-forex_20260426_002956.json`
**Harness:** `python3 run_full_backtest.py --symbols ... --years 5 --interval 1d --tag REL377-forex`
**Universe:** 6 new forex symbols × 7 fast (non-Kronos) strategies = 42 cells
**Period:** 5 years of daily candles (yfinance)
**Whitelist filter:** Sharpe ≥ 0.30 AND total_trades ≥ 10

## Motivation

The 2026-04-25 full universe backtest identified forex as the highest-Sharpe asset class:

| Strategy | Forex Sharpe |
|---|---|
| momentum | +0.46 (best overall) |
| bb_squeeze | +0.38 |
| mean_reversion | +0.25 (only place it works) |
| ema_crossover | +0.15 |

At that time only 2 forex pairs (EURUSD=X, GBPUSD=X) were tradeable. This expansion adds 6 majors and JPY-crosses to broaden the highest-edge bucket.

## Per-symbol × per-strategy results (Sharpe / trades)

`breakout` returned 0 trades on every forex pair (no qualifying breakouts on daily candles), so it is filtered out of the whitelist by construction.

| Symbol | momentum | mean_reversion | breakout | multi_factor | vwap_reversion | bb_squeeze | ema_crossover |
|--------|------|------|------|------|------|------|------|
| USDJPY=X | -0.22 (44) | -0.57 (46) | +0.00 (0) | +0.29 (78) | -0.53 (34) | +0.06 (13) | +0.19 (26) |
| AUDUSD=X | -0.52 (46) | +0.01 (40) | +0.00 (0) | -0.95 (60) | -0.01 (19) | -0.51 (17) | **+0.58 (29)** |
| USDCAD=X | -0.80 (49) | +0.21 (53) | +0.00 (0) | -0.84 (67) | **+0.72 (15)** | +0.01 (14) | +0.07 (30) |
| USDCHF=X | -0.27 (51) | **+0.66 (44)** | +0.00 (0) | -0.14 (77) | -0.60 (16) | -0.29 (15) | -0.18 (38) |
| EURJPY=X | +0.05 (37) | **+0.55 (50)** | +0.00 (0) | -0.09 (60) | -0.25 (35) | -0.36 (17) | -0.43 (30) |
| GBPJPY=X | -0.86 (44) | **+0.52 (46)** | +0.00 (0) | -0.64 (57) | **+0.44 (24)** | -0.47 (17) | **+0.37 (30)** |

**Bold** = passes whitelist filter (Sharpe ≥ 0.30 AND trades ≥ 10).

## Cells admitted to `WHITELIST`

7 cells across 5 of the 6 new symbols:

| Symbol | Strategy | Sharpe | Trades |
|---|---|---|---|
| AUDUSD=X | ema_crossover | +0.58 | 29 |
| USDCAD=X | vwap_reversion | +0.72 | 15 |
| USDCHF=X | mean_reversion | +0.66 | 44 |
| EURJPY=X | mean_reversion | +0.55 | 50 |
| GBPJPY=X | mean_reversion | +0.52 | 46 |
| GBPJPY=X | vwap_reversion | +0.44 | 24 |
| GBPJPY=X | ema_crossover | +0.37 | 30 |

USDJPY=X is **not** added to the whitelist on this pass — its best cell (multi_factor, +0.29 over 78 trades) just misses the 0.30 threshold. It remains in `MARKETS` and can be revisited if the threshold is later relaxed.

## Whitelist totals after this change

- Symbols: 53 → **58**
- Cells: 102 → **109**
- Forex symbols in `MARKETS`: 2 → **8**

## Notes & caveats

- Daily candles only. The runtime config gives forex pairs `["4h"]` timeframes; the backtest was run on `1d` because that is what the rest of the universe is benchmarked on. Expect live behavior to diverge.
- `multi_factor` on USDJPY=X (+0.29 / 78 trades) is the closest miss. Worth re-checking if filter logic ever drops to 0.25.
- `mean_reversion` lives up to the original finding: it works on 4 of 8 forex pairs (EUR/USD, USD/CHF, EUR/JPY, GBP/JPY), nowhere else in the whitelist.
- `breakout` produced 0 trades across all 6 new symbols (consistent with the existing forex pairs at this timeframe).
- No slippage modeling beyond the harness's 0.05% commission; live results will be lower.
