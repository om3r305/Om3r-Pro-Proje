# Brian 2026 Phase 2.7 Market Structure

Phase 2.7 is research-only. It adds a deterministic chart-structure layer
without changing the Phase 2.5 specialist set or portfolio implementation.
The root `dip_tracker.py` remains untouched and is represented only by the
explicit `LEGACY_SIMPLE_DIP` comparison.

## Point-in-time contract

A swing at index `j` is emitted only when candle `j + right_bars` has closed.
HH/HL/LH/LL labels, structure states, BOS, CHOCH, zones, sweeps, retests, and
RSI divergence use confirmed swings only. Every swing includes pivot and
confirmation timestamps plus a deterministic provenance ID. Fifteen-minute
and hourly structures are computed independently and joined to a 5m decision
only when their close timestamp is not later than that decision.

Historical Spot data supplies OHLCV only. Order book and bid/ask are
`UNAVAILABLE`; spread and slippage are simulation assumptions. Exhaustion
fields are explicitly proxies, not observed order flow. Dip/rally scores are
bounded candidate-evidence features and include completed 15m/1h context.

## Research suite

The preregistration contains locked Phase 2.5 folds, LOW/BASE/STRESS costs,
15 feature groups and ablations, Phase 2.5 LR/GB/STATIC baselines,
`LEGACY_SIMPLE_DIP`, two Phase 2.7 specialists, fixed/structure/hybrid exits,
three correctly refitted source-gap sensitivity folds, six purged temporal
yearly robustness splits, calibration metrics, structure regimes, and equity
curves. Structure-only exits disable the fixed profit target but retain the
hard protective stop and max hold; hybrid exits retain fixed TP/SL as well.

## Final development evidence

Dataset ID:
`df0863e175d0600b200e5098131037040dd21e2b01826766638c0097a1470cc1`

Experiment ID:
`d1f9e2a74fb98b6fd83e8ec679a70111b1dc631f50d8d46ed1919d44391bb90b`

Maximum timestamp was `1767225599.999`, strictly before 2026. The all-feature
model produced 7 entries and -35.42 BASE PnL in Fold 1, 441 entries and
-958.74 in Fold 2, and zero entries in Fold 3. Five of six yearly robustness
splits lost money. The positive 2024 split made only 4.07 with approximately
0.014% coverage, so it is not meaningful champion evidence.

`LEGACY_SIMPLE_DIP`, STATIC Brian, market-structure specialist, and DIP
specialist were also negative. The final decision is
`INSUFFICIENT_EVIDENCE`: minimum trades per fold, coverage, multi-fold
expectancy, profit factor, and STRESS survivability gates failed. No concept
is claimed profitable and no live configuration is changed.

The 2026 holdout remains `INVALID_CONTAMINATED`, with
`evaluation_allowed = false` and
`NO PRISTINE FINAL HOLDOUT EVALUATED`.
