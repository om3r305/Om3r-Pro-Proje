# Brian Phase 3.4 — Profit-Seeking Shadow Mode

Status: preregistered development research only.

## Purpose

Phase 3.4 changes the decision objective, not the safety boundary. Brian remains SHADOW_RESEARCH_ONLY and has no broker/exchange execution surface. The objective is to seek positive net growth after explicit virtual trading costs and uncertainty, instead of equating more activity with better performance.

## Locked training/exam split

- Training: verified Binance Vision monthly archives, BTCUSDT/ETHUSDT/SOLUSDT, 2024-01, canonical completed 1m -> 5m.
- Training procedure: exactly the already-merged Phase 3.3 100-life curriculum and learner checkpoint.
- Exam: verified Binance Vision monthly archives, BTCUSDT/ETHUSDT/SOLUSDT, 2024-02, canonical completed 1m -> 5m.
- Learning is disabled for the entire 2024-02 exam.
- The exam month is excluded from Phase 3.3 learner training, but it is still development data and is not a pristine project-wide final holdout.
- 2026 remains INVALID_CONTAMINATED and is forbidden.

## Profit-seeking objective

The frozen learner provides causal per-asset return predictions and empirical prediction-error uncertainty. The profit policy:

1. computes long and short directional edge using the learner's already-locked risk-aversion rule;
2. charges a conservative two-way estimate of the configured virtual fee + half-spread + slippage cost for incremental turnover;
3. trades only if net edge remains above the learner's pre-existing minimum edge threshold;
4. sizes only inside the learner and Market Gym unlevered caps;
5. preserves the existing drawdown throttles and full flattening rule.

The policy never updates the learner checkpoint during the exam.

## Locked comparison

The same frozen Phase 3.3 checkpoint is evaluated twice on the same chronological February path:

- FROZEN_NATIVE_COUNTERFACTUAL — the existing learner allocation rule, read only.
- PROFIT_SEEKING_SHADOW — the explicit conservative net-edge rule.

No result-dependent tuning is allowed between these two evaluations.

## Development-candidate gate

`DEVELOPMENT_CANDIDATE` requires all of the following on the locked exam:

- positive net virtual return after costs;
- net step profit factor greater than 1 (or no losing steps with positive return);
- maximum drawdown no greater than 10%;
- at least two of four chronological quarters positive;
- profit-seeking return not worse than the frozen-native return;
- no virtual ruin.

Failure of any check is `INSUFFICIENT_EVIDENCE`.

Passing is not proof of profitability, not final-holdout evidence, and not permission to trade real funds. Automatic promotion is disabled.

## Safety invariants

- no credentials or authenticated exchange APIs;
- no live order methods;
- no leverage;
- no self-modifying code;
- no automatic promotion;
- no 2026 historical reuse;
- official archive checksum/provenance retained;
- exam learner state hash must be identical before and after evaluation.
