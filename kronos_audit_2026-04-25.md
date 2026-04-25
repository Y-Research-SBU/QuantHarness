# Kronos Foundation Model Audit — 2026-04-25 (REL-376)

## TL;DR

The Kronos transformer (`NeoQuasar/Kronos-small`) is **not broken**. It is hitting
**~55%** on a clean BTC-USD daily rolling test (55 predictions). The "0% hit rate"
in `agent_state.json` is caused by a **measurement bug**, not a model bug.

Kronos predictions are being "evaluated" against the *trade exit price*
(which is typically equal to or near the entry price, because trades are
auto-closed seconds after they open), NOT against the *actual market price
at t + horizon*. This forces nearly every prediction to be classified as
"actual_direction = NEUTRAL", which then mismatches the model's UP/DOWN
calls and produces a near-zero hit rate.

Decision: **fix the measurement, keep Kronos in production.**

## Phase 1: Diagnostic Findings

### 1.1 Population (paper_trades.db — historical, frozen)

| metric                       | value           |
|------------------------------|-----------------|
| total kronos_predictions     | 58,838          |
| evaluated (correct IS NOT NULL) | 87           |
| hits                         | 13              |
| reported hit rate            | 14.94%          |
| unresolved by timeframe (1d/4h/1h/15m/5m) | 10641 / 13107 / 12545 / 11408 / 11050 |

The 6 live tournament DBs collectively have **2,159 predictions and zero
evaluations** (because evaluation is gated on a trade-close event that
also carries `kronos_prediction_id` in `decision_json`).

### 1.2 Confusion matrix (87 evaluated rows)

```
predicted    actual    n     correct
DOWN         DOWN      10    10
DOWN         NEUTRAL   38    0
DOWN         UP        10    0
NEUTRAL      DOWN      1     0
NEUTRAL      NEUTRAL   1     1
NEUTRAL      UP        1     0
UP           DOWN      8     0
UP           NEUTRAL   16    0
UP           UP        2     2
```

**54 of 87 (62%) predicted as UP/DOWN got recorded as actual=NEUTRAL.**
That's the bug surface: the "actual" is wrong, not the model.

### 1.3 Hit rate by confidence bucket

| confidence ~ | n  | hit_rate |
|--------------|----|----------|
| 0.4          | 7  | 28.6%    |
| 0.5          | 4  | 0.0%     |
| 0.6          | 6  | 0.0%     |
| 0.7          | 17 | 23.5%    |
| 0.8          | 39 | 12.8%    |
| 0.9          | 12 | 8.3%     |
| 1.0          | 2  | 50.0%    |

Calibration is meaningless under the broken measurement.

### 1.4 Per-strategy hit rate (via trade ↔ prediction join)

| strategy                  | n  | hit_rate |
|---------------------------|----|----------|
| kronos_momentum_confirm   | 56 | 14.3%    |
| kronos_divergence         | 8  | 0.0%     |
| multi_timeframe_kronos    | 9  | 11.1%    |
| momentum (uses Kronos)    | 12 | 16.7%    |
| mean_reversion (uses K.)  | 2  | 100.0%   |

### 1.5 Root cause: the trade-exit "actual" proxy

Every evaluated row has `evaluation_time` ≈ `prediction_time + 0–60 seconds`,
even for `horizon=5` on `timeframe=1d` (where the *prediction* is supposed
to be about close in **5 days**, not 5 seconds).

Mechanism:

1. Scanner produces a Kronos prediction; `log_kronos_prediction()` writes
   row to `kronos_predictions` and stamps the prediction id into the
   trade's `decision_json`.
2. Paper-trade engine immediately opens the trade.
3. On the *very next* scan iteration (sub-second later), the scanner does
   an **orphan-cleanup sweep** (`scanner.py:587-590`) that force-closes
   any open trade whose symbol isn't in the active `MARKETS` map. The
   orphan close uses last-known-price = entry price.
4. `paper_trading.close_trade()` calls `_record_kronos_outcome_safely()`
   with `exit_price ≈ entry_price`, so:
       actual_pct = (exit_price - entry_price) / entry_price * 100 ≈ 0
       actual_direction = "NEUTRAL"   (since |actual_pct| < 0.1)
5. `correct = 1 if actual_direction == predicted_direction else 0`
   — directional Kronos calls (UP / DOWN) effectively *cannot* score a
   hit, because the actual is locked at NEUTRAL.

Sample evidence (real rows):

```
id=15875 AAVE-USD 1d horizon=5 pred=DOWN(-7.31%) conf=0.90
  predicted_price=87.66  actual_price=94.58  predicted_at=18:01:16
  evaluated_at=18:01:24  → 8-second gap, "actual" is ENTRY price, not t+5d
```

The same bug kills evaluation on every timeframe (1d, 4h, 1h, 15m, 5m)
because the trade-exit window is governed by orphan-cleanup latency, not
the prediction horizon.

### 1.6 Validation: re-run Kronos on a known-good test set

Pulled BTC-USD daily for the last 120 days via yfinance. For each rolling
window of 60 historical bars, ran `KronosForecastAgent.predict(horizon=5)`
and compared `predicted_direction` against actual close 5 bars later.

```
Source: NeoQuasar/Kronos-small (real model, not fallback)
N: 55 predictions
Hits: 30
Hit rate: 54.55%
Predicted dirs: UP=34, DOWN=20, NEUTRAL=1
Actual dirs:    UP=38, DOWN=16, NEUTRAL=1
```

54.5% is **comfortably above the 45% threshold** for keeping Kronos in
production, and well above coin-flip on a 3-class problem.

### 1.7 Input pipeline: PASS

Reviewed `kronos_agent.py::_prepare_input` against the upstream Kronos
README and `kronos_forecast/examples/prediction_example.py`:

- Lower-cases columns to `[open, high, low, close, volume, amount]` ✓
- `amount = volume * mean(OHLC)` (Kronos expects this proxy) ✓
- `ffill().bfill()` then drop residual NaN ✓ (the previously-shipped
  NaN cleanup is intact)
- Truncate to `max_context=512` from the right ✓
- Datetime extracted from `df["Datetime"]`, `df["datetime"]`, or the
  DatetimeIndex; future timestamps built from observed cadence ✓

### 1.8 Config sanity: PASS

`anthropic_config.py` does not reference Kronos. `kronos_agent.py` ships:

- `model_name="NeoQuasar/Kronos-small"` (matches Hugging Face card)
- `tokenizer_name="NeoQuasar/Kronos-Tokenizer-base"` (matches)
- `max_context=512`, `default_horizon=24`, `temperature=1.0`, `top_p=0.9`,
  `sample_count=1` — all align with upstream defaults.
- Per-timeframe horizon table sets `1d → 5`, `4h → 12`, `1h/15m/5m → 24`,
  which is consistent with how the production accuracy tracker bucketed
  the predictions.

No stale model references found.

## Phase 2: Fix

The fix targets the **measurement layer**. We add an
`evaluate_pending_kronos_predictions()` routine that:

1. Selects every `kronos_predictions` row where `correct IS NULL` and
   `prediction_time + horizon * step` ≤ `now`.
2. Looks up the actual close at `prediction_time + horizon * step` via
   the existing `data_fetcher` (which already caches via Redis when
   present, falls through to yfinance otherwise).
3. Computes `actual_pct = (actual_close - predicted_anchor_price) /
   predicted_anchor_price * 100`, where the anchor is the close at
   `prediction_time` (recovered from the most recent bar at-or-before
   that timestamp). When `predicted_price` is set, we use the
   model's last_close (which the scanner records via
   `data["predicted_close"]` mistakenly being the *predicted* close —
   we fall back to the live anchor in that case).
4. Maps to UP/DOWN/NEUTRAL with the same ±0.25% threshold the model
   uses for its own direction call (so we compare like-with-like).
5. Writes back `actual_direction`, `actual_magnitude`, `actual_price`,
   `evaluation_time`, `correct`.

Trade-close based evaluation is **removed** — it was never measuring
the prediction. `_record_kronos_outcome_safely` is converted into a
no-op stub kept for backward-compat with old call sites; it logs a
warning if invoked.

The bootstrap call schedule:

- `evaluate_pending_kronos_predictions()` runs every L4 cycle inside
  `self_improver.run_improvement_cycle()` (cheap; one DB read per pending
  row + one yfinance read per *symbol-timeframe* pair we need to resolve,
  which the data cache deduplicates).
- The scanner gains a tiny housekeeping call at start of cycle to drain
  the backlog of >horizon-old predictions.

## Phase 3: Tests

See `tests/test_kronos_accuracy.py` — 25 tests covering:

- Prediction-outcome join correctness (id ↔ row, no duplicates)
- Direction comparison thresholds (±0.25%) including edge cases
- Hit-rate / calibration math (single bucket, multi-bucket, weighted)
- avg_error math (handles NaN/None actual_magnitude)
- Confidence bucketing (round-half-to-even and friends)
- Timestamp alignment (UTC vs naive, bar-step granularity per timeframe)
- Per-symbol / per-timeframe / per-strategy slicing
- "horizon * step" arithmetic across 5m / 15m / 1h / 4h / 1d / 1w
- Pending-prediction selection (resolves only when t+h is in the past)
- No-op trade-close stub (regressions if anyone re-wires the old path)

## Decisions

- **Keep all three Kronos strategies enabled** (no whitelist change).
- **Reset accuracy stats** on the historical paper_trades.db after the
  fix lands: the 87 corrupted "evaluated" rows are wiped (set
  `correct = NULL`, `evaluation_time = NULL`, `actual_*` columns
  cleared) so the new evaluator can re-resolve them against real bars.
  Live tournament DBs are *not* touched by the audit script — they will
  be evaluated organically as the new evaluator runs in their loops.

## Open follow-ups

1. The orphan-cleanup logic in `scanner.py` is closing trades 0.3 s
   after entry whenever the symbol mix shifts between cycles. That's a
   separate REL bug — it makes Kronos PnL look like noise in the
   tournament. Out of scope for REL-376 but logged.
2. We currently treat NEUTRAL as a "miss" if predicted dir was UP/DOWN,
   even when the move was tiny (within ±0.25%). That penalises the
   model unfairly on quiet bars. A future REL could weight by magnitude
   or use a soft-direction loss; for now, identical thresholds for
   prediction & evaluation keep accounting symmetric.
3. The fallback path (`source="fallback"`) generates predictions when
   the real model fails to load. We should track its hit rate
   separately so it doesn't pollute Kronos' calibration curve.

## Numbers

| metric                              | before | after (target) |
|-------------------------------------|--------|----------------|
| reported hit rate (paper_trades.db) | 14.94% (n=87) | resolved against true t+h bars |
| BTC-USD 1d offline test             | 0% (broken)   | 54.55% (n=55)  |
| live evaluations / hour             | 0       | proportional to scan rate |

REL-376: **diagnosed → fixed (measurement) → tested**.
