# Brian 2026 Phase 2.7 Market Structure

Phase 2.7 is research-only. It adds a deterministic chart-structure layer
without changing the locked Phase 2.5 specialist set. The root `dip_tracker.py`
remains untouched and is used directly for the explicit `LEGACY_SIMPLE_DIP`
entry-eligibility baseline.

## Point-in-time contract

A swing at index `j` is emitted only when candle `j + right_bars` has closed.
HH/HL/LH/LL labels, structure states, BOS, CHOCH, zones, sweeps, failed breaks,
retests, and RSI divergence use confirmed information only. Every swing includes
pivot and confirmation timestamps plus a deterministic provenance ID.
Fifteen-minute and hourly structures are computed independently and joined to a
5m decision only when their close timestamp is not later than that decision.
Targets must resolve inside their train/validation/test partition.

Historical Spot data supplies OHLCV only. Historical order book and bid/ask are
`UNAVAILABLE`; spread and slippage are simulation assumptions. Exhaustion fields
are explicitly proxies, not observed order flow. `unavailable` structure values
remain missing and are handled by train-only median imputation plus missingness
indicators; they are not silently coerced to zero and rows are not discarded
merely because a structural fact is not yet available.

## Audit repairs

The first Phase 2.7 draft was audited before promotion. The audit found issues
that can materially change development results, so its experiment output is
**INVALIDATED_BY_AUDIT** and must not be used as evidence.

Repairs include:

- structure exits are synchronized with the real stateful portfolio lifecycle;
- long/short failed-break exit semantics are directionally correct;
- bullish and bearish momentum recovery are scored symmetrically;
- missing structural values use the already-locked train-only imputer rather
  than deleting all rows containing missing values;
- candidate gates count actual validation/calibration samples;
- target labels may not resolve beyond their partition boundary;
- `LEGACY_SIMPLE_DIP` uses the actual root `DipTracker` entry eligibility and
  does not invent a legacy sell rule;
- Phase 2.7 specialists receive completed 15m/1h context;
- yearly robustness is expanding walk-forward with past-only training;
- source-gap sensitivity excludes label paths that cross excluded months and
  still forces the portfolio flat at excluded boundaries.

## Research suite after repair

The preregistration keeps the locked Phase 2.5 folds and LOW/BASE/STRESS cost
assumptions, feature-group ablations, Phase 2.5 LR/GB/STATIC references,
`LEGACY_SIMPLE_DIP`, two Phase 2.7 specialists, fixed/structure/hybrid exits,
refitted source-gap sensitivity, causal expanding yearly walk-forward checks,
calibration metrics, feature-availability diagnostics, structure regimes, and
equity curves.

Structure-only exits retain the hard protective stop and max hold while
removing the fixed profit target. Hybrid exits retain fixed TP/SL/max-hold and
add causal structure exits. Lifecycle exits have priority on every bar.

## Evidence status

The pre-audit dataset/experiment identifiers and metrics are retained only in
Git history for traceability. They are not valid Phase 2.7 evidence after the
audit repairs.

A fresh 2020-2025 development run is required after the repaired code passes CI.
The result must be reported as-is even if it is negative. No retuning may use
2026.

The 2026 holdout remains permanently `INVALID_CONTAMINATED`, with
`evaluation_allowed = false` and
`NO PRISTINE FINAL HOLDOUT EVALUATED`.

No concept is claimed profitable and no live configuration is changed.
