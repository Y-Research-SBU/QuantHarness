# Kronos Horizon Sweep — REL-376 (2026-04-25)

## TL;DR

Tested Kronos at 14 (timeframe, horizon) cells across 5 symbols on yfinance
data over the last 90–180 days. Surface result: **most cells clear 45%
hit rate**. Real result: **almost no cell beats a trivial constant
baseline**. Kronos's apparent accuracy is a regime artifact — during
this window the market had a strong directional drift and Kronos
benefited from coincident bias, but a "always-predict-majority-class"
baseline beats Kronos in 12 of 14 cells.

**Recommendation: do NOT widen `KRONOS_TRUSTED_TIMEFRAMES`. Production
fix from 917e1f4 stays as-is (`{"1d"}` only).** Only one intraday cell
shows a real edge over baseline (5m h=24, +9.8% over majority class with
ε=0.25%, n=51) — too marginal to ship.

---

## Method

For each `(symbol ∈ {BTC-USD, ETH-USD, SOL-USD, SPY, AAPL}, timeframe,
horizon)` we:

1. Fetch up to 90–180 days of OHLCV via yfinance (per-tf lookback caps).
2. Pick 30 evenly-spaced evaluation points per (symbol, tf), each with a
   400-bar context window into Kronos.
3. Run Kronos at the **maximum horizon** for that timeframe. Read off
   intermediate horizons from the same predicted close path. (No
   look-ahead — context only contains bars at-or-before the eval index.)
4. Compare predicted close at t+h vs actual close at t+h.

Equity tickers (SPY/AAPL) were skipped at 4h (only 243 bars in 180 days
— under the 412-bar minimum) and got n=6/7 at 1h (insufficient bars for
a 400-bar context). Crypto contributed n≈90 per cell at 1h/4h.

Total prediction rows collected: **1,625**.

Script: `scripts/kronos_horizon_sweep.py`.
Raw data: `kronos_horizon_sweep_2026-04-25.json`.

---

## Headline heatmap (strict ε=0 directional accuracy)

| timeframe | horizon |   n  | hit_rate | mean_pred_% | mean_actual_% | signed_error | bearish_bias_% |
|-----------|---------|------|----------|-------------|----------------|---------------|-----------------|
| 5m  | 24 | 150 | **57.3%** | -0.15% | -0.16% | +0.01% |  +2.7% |
| 5m  | 48 | 150 | 55.3% | -0.67% | -0.16% | -0.52% |  +6.0% |
| 15m | 12 | 150 | 49.3% | -0.07% | +0.27% | -0.34% |  +9.3% |
| 15m | 24 | 150 | 48.0% | -0.41% | +0.38% | -0.79% | +16.0% |
| 15m | 48 | 150 | 54.0% | -0.59% | +0.20% | -0.79% |  +6.0% |
| 1h  |  4 | 103 | 49.5% | -0.09% | +0.13% | -0.22% |  +7.8% |
| 1h  |  6 | 103 | 49.5% | -0.11% | +0.22% | -0.33% | +15.5% |
| 1h  |  8 | 103 | 41.7% | -0.09% | +0.18% | -0.27% |  +5.8% |
| 1h  | 12 | 103 | 41.7% | -0.28% | +0.15% | -0.43% | +13.6% |
| 1h  | 24 | 103 | 52.4% | -0.49% | +0.10% | -0.59% | +14.6% |
| 4h  |  3 |  90 | **57.8%** | -0.81% | -1.25% | +0.44% |  -4.4% |
| 4h  |  6 |  90 | 56.7% | -1.15% | -1.16% | +0.01% | +10.0% |
| 4h  |  8 |  90 | 56.7% | -1.12% | -1.28% | +0.16% |  -1.1% |
| 4h  | 12 |  90 | 53.3% | -1.60% | -0.24% | -1.37% | +15.6% |

Sanity-check on the production-broken cell: **1h h=24** comes out at
**52.4%** here, not 27.9%. The discrepancy is explained by the audit
doc (`kronos_audit_2026-04-25.md` §1.5): production "actual" was the
trade-exit price (≈ entry price), not the price at t+h. This sweep
measures against the actual close at t+h, the methodology the audit
endorsed. So Kronos itself is roughly the same model as on 1d — the
production number was a measurement artifact.

---

## The killer chart: vs. constant-baseline edge

Hit rate alone is misleading because the actual UP/DOWN distribution in
this 90–180 day window was not balanced. Compare Kronos to "always
predict the majority class for this cell" (oracle-best constant, picked
post-hoc, so this is a **lenient** baseline for Kronos to beat):

| timeframe | h | n | always-UP | always-DOWN | Kronos | edge_vs_majority |
|-----------|---|----|-----------|-------------|--------|-------------------|
| 5m  | 24 | 150 | 45.3% | 54.7% | 57.3% | **+2.7%** |
| 5m  | 48 | 150 | 44.7% | 55.3% | 55.3% |  0.0%      |
| 15m | 12 | 150 | 55.3% | 44.7% | 49.3% | -6.0%      |
| 15m | 24 | 150 | 56.7% | 42.7% | 48.0% | -8.7%      |
| 15m | 48 | 150 | 50.0% | 50.0% | 54.0% | **+4.0%**  |
| 1h  |  4 | 103 | 54.4% | 45.6% | 49.5% | -4.9%      |
| 1h  |  6 | 103 | 60.2% | 39.8% | 49.5% | -10.7%     |
| 1h  |  8 | 103 | 49.5% | 50.5% | 41.7% | -8.7%      |
| 1h  | 12 | 103 | 53.4% | 46.6% | 41.7% | -11.6%     |
| 1h  | 24 | 103 | 53.4% | 46.6% | 52.4% | -1.0%      |
| 4h  |  3 |  90 | 26.7% | 73.3% | 57.8% | **-15.6%** |
| 4h  |  6 |  90 | 38.9% | 61.1% | 56.7% | -4.4%      |
| 4h  |  8 |  90 | 33.3% | 66.7% | 56.7% | -10.0%     |
| 4h  | 12 |  90 | 44.4% | 55.6% | 53.3% | -2.2%      |

**Kronos beats the best constant baseline in only 2 of 14 cells**, and in
both cases the margin is small (≤4%). At 4h h=3 — the cell that looked
most attractive at face value (57.8%) — a constant always-DOWN strategy
would have hit 73.3%. Kronos is decisively *worse* than a coin glued to
DOWN.

### With Kronos's own ±0.25% NEUTRAL band (crypto-only)

The audit doc notes Kronos uses ε=0.25% for its NEUTRAL band. Re-running
with directional calls only (drop FLAT predictions and FLAT actuals)
narrows the picture further:

| timeframe | h | n_called | Kronos | best_baseline | edge |
|-----------|---|----------|--------|----------------|------|
| 5m  | 24 | 51 | **70.6%** | 60.8% | **+9.8%** |
| 5m  | 48 | 66 | 60.6% | 71.2% | -10.6% |
| 15m | 12 | 45 | 53.3% | 53.3% |  0.0%  |
| 15m | 24 | 62 | 45.2% | 56.5% | -11.3% |
| 15m | 48 | 80 | 57.5% | 63.8% | -6.3%  |
| 1h  |  4 | 55 | 40.0% | 63.6% | -23.6% |
| 1h  |  6 | 56 | 42.9% | 58.9% | -16.1% |
| 1h  |  8 | 67 | 38.8% | 52.2% | -13.4% |
| 1h  | 12 | 70 | 41.4% | 52.9% | -11.4% |
| 1h  | 24 | 70 | 55.7% | 52.9% | **+2.9%** |
| 4h  |  3 | 72 | 55.6% | 79.2% | -23.6% |
| 4h  |  6 | 75 | 60.0% | 65.3% | -5.3%  |
| 4h  |  8 | 79 | 59.5% | 67.1% | -7.6%  |
| 4h  | 12 | 82 | 54.9% | 59.8% | -4.9%  |

Only **5m h=24** shows a meaningful edge (+9.8%). 1h h=24 squeaks
positive (+2.9%) but it's well within sample noise.

---

## Per-symbol breakdown (sanity check)

Per-symbol numbers are noisy but reassuringly consistent — no single
symbol is driving the aggregate (numbers from strict ε=0):

```
1h h=24:
  BTC=56.7%  ETH=63.3%  SOL=50.0%  SPY=0.0% (n=6)  AAPL=42.9% (n=7)
4h h=3:
  BTC=60.0%  ETH=46.7%  SOL=66.7%
5m h=24:
  BTC=66.7%  ETH=63.3%  SOL=50.0%  SPY=63.3%  AAPL=43.3%
```

The eq tickers at 1h/4h are too thin to read into. The 5m h=24 result is
the only one that holds up across both crypto and equities.

---

## Systematic bias

Across nearly every cell the `signed_error` and `bearish_bias_%` columns
show Kronos leans bearish — predicted means are negative, bearish bias
is +5% to +16%. This matches the production observation in commit
917e1f4. Kronos has a structural bearish prior on intraday horizons.

It happens to score okay on hit rate this window because:
- Crypto had a meaningfully bearish 4h sample (always-DOWN ≈ 73% on h=3)
- Kronos calls DOWN often, so it lines up with reality on those bars

In a bull regime this would invert and Kronos's hit rate would collapse.
The 1h h=24 production failure (Apr 24-25 BTC drawdown propagating
incorrect bearish calls to alts that didn't follow) is exactly that
mechanism in microcosm.

---

## Recommendations

### Per-timeframe verdict

- **1h: untrustworthy.** Best cell (h=24, 52.4%) is -1.0% vs majority
  baseline. h=8 and h=12 both miss the 45% threshold outright.
- **4h: untrustworthy.** Surface hit rates look great (53–58%) but every
  cell loses to always-DOWN by 2–16%.
- **15m: untrustworthy.** Mixed; h=48 marginal (+4% edge), other
  horizons negative.
- **5m: marginally interesting.** h=24 is the only cell with a real
  edge (+9.8% over baseline with ε=0.25%, n=51). Not enough alone to
  ship trust on a noisy timeframe.

### Trusted set: no change

```python
# Stays as commit 917e1f4 shipped:
KRONOS_TRUSTED_TIMEFRAMES = frozenset({"1d"})
```

The 1d horizon=5 case (validated at 54.55% in `kronos_audit_2026-04-25.md`)
remains the only place Kronos has a defensible track record. Daily bars
+ short horizon = the model's strength.

### Suggested follow-ups

1. **Re-run this sweep after the production measurement fix lands** so
   we have an apples-to-apples comparison with the (corrected) live
   evaluator. If 5m h=24 holds up there too, we can revisit.
2. **Hold the line on intraday Kronos.** The strategies that depend on
   it (`kronos_momentum_confirm`, `kronos_divergence`,
   `multi_timeframe_kronos`) should keep firing only on 1d for now.
3. **Consider de-biasing.** Kronos has a structural bearish prior on
   sub-daily horizons (mean predicted ≈ −0.1% to −1.6% across cells).
   A future REL could subtract the running bias from raw predictions
   before deriving direction, then re-evaluate. Out of scope for
   REL-376.
4. **Don't extend the trusted set without baseline-edge evidence.**
   The 45% threshold in the original task spec is too lenient — a
   constant always-DOWN model hits 73% on 4h h=3. Hit rate ≥ 45% with
   n ≥ 50 is necessary but not sufficient. Add an
   `edge_vs_majority_class ≥ +5%` filter for any future widening.

---

## Linear ticket REL-376 update

> **Horizon sweep complete. Kronos remains restricted to 1d.** All 4
> intraday timeframes (5m / 15m / 1h / 4h) tested across crypto + eq;
> 12 of 14 (tf, horizon) cells are beaten by a trivial majority-class
> baseline. Only 5m h=24 shows a meaningful edge (+9.8% with
> ε=0.25%, n=51) — too marginal alone. Kronos has a structural bearish
> prior intraday that produces flattering hit rates in bear-drift
> windows but not real predictive value.
>
> Production guard from 917e1f4 holds. No code change needed.
> Sweep harness committed to `scripts/kronos_horizon_sweep.py` so we
> can re-run after the measurement-fix lands in production.
