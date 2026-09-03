# Brian Phase 3.5 — 1,000-Life Training

## Status

Phase 3.5 is a preregistered **TRAINING_ONLY / SHADOW_ONLY** curriculum. It is not a profitability claim, not a final-holdout evaluation, and cannot promote Brian to live execution.

## Locked objective

Scale the Phase 3.3 causal counterfactual learner from a 100-life single-month smoke into a materially broader 1,000-life training curriculum while preserving the same causal decision boundary, virtual cost model, unlevered portfolio limits and post-episode-only learning contract.

## Locked training universe

- Symbols: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT
- Source: official Binance Vision monthly spot archives with checksum verification
- Raw source interval: 1m
- Canonical training interval: 5m
- Training months: 2023-01 through 2024-02 inclusive
- No 2026 data is permitted
- No historical orderbook, bid/ask or order-flow data is fabricated

## Locked curriculum

Exactly 1,000 lives:

- 250 REAL_REPLAY
- 500 BLOCK_BOOTSTRAP
- 250 STRESS_BOOTSTRAP
- 100 lives per sequential checkpoint shard
- fixed seed: 3501
- horizon: 128 steps
- block length: 32
- stress return scale: 1.75

This is the exact 25/50/25 Phase 3.3 curriculum ratio scaled by 10x. Thresholds and learner hyperparameters are not tuned from Phase 3.4 February outcomes.

## Preregistered next development exam

Before Phase 3.5 training evidence is reviewed, the following months are reserved:

- 2024-03
- 2024-04

Phase 3.5 training must not read, fetch, derive, train on or evaluate those months. They are reserved for a later frozen-policy development exam with learning disabled. They are development evidence only, not a pristine final holdout.

The previously contaminated 2026 holdout remains permanently `INVALID_CONTAMINATED` with `evaluation_allowed=false`.

## State reproducibility

Raw checkpoints retain full floating-point precision. Phase 3.5 additionally records a portable audit fingerprint that normalizes only insignificant last-bit floating-point tails. This addresses cross-run CPU/BLAS differences observed at approximately 1e-19 without modifying learned weights or accepting materially different states as identical.

Both identifiers are retained:

- raw training state ID: exact checkpoint identity inside one runtime/checkpoint chain
- portable training state fingerprint: cross-runtime scientific audit identity

## Evidence interpretation

Training episode returns, regret, turnover, costs and drawdown are diagnostics of the learning curriculum. Positive training returns do **not** establish profitability. Negative training evidence is retained unchanged.

There is no authenticated exchange execution, broker surface, credential requirement, leverage, automatic promotion, self-modifying code, post-hoc rescue tuning, or final-holdout evaluation in Phase 3.5.
