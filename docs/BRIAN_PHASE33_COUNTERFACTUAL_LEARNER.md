# Brian 2026 Phase 3.3 — Causal Counterfactual Learner

Status: **PREREGISTERED TRAINING FOUNDATION / SHADOW_RESEARCH_ONLY**

This learner is the first policy intended to accumulate experience inside the Market Gym.
It does not self-modify code, does not execute exchange orders and does not promote itself.

## Why this learner first

The first 100k-life learner should be auditable and deterministic before introducing a more complex RL stack.
Brian therefore starts with an online full-information learner:

- decision features are built only from frames already visible at decision time,
- the target decision executes at the next frame open,
- after an episode is over, the learner observes the full resolved outcome,
- every observed asset supplies a counterfactual return, even if Brian did not trade it,
- BUY, SELL and WAIT can therefore be compared without inventing a second market path.

This is closer to supervised/contextual full-information learning than opaque end-to-end reinforcement learning.
A later RL layer can be compared against it as a challenger rather than replacing it by assumption.

## Causal holding label

A target chosen at decision step `d` executes at `open(d+1)`.
Market Gym keeps that exposure through the next close-to-open gap and allows the next target to replace it at `open(d+2)`.

Therefore the training label for that decision is the real or synthetic-world return:

`open(d+2) / open(d+1) - 1`

for every tradable asset.

No label is released while the episode is alive. The fully resolved path is supplied only after termination.
The world seed and sampled source-block recipe are never supplied to policy code.

## Initial causal feature vector

Per asset:

- one-bar close return,
- three-bar return,
- short rolling trend,
- realized volatility,
- bar range,
- cross-sectional relative momentum,
- market-wide momentum,
- market dispersion,
- volume change when genuinely available,
- bias term.

Features are bounded before online optimization. Missing volume is not converted into fabricated volume history.

## Optimization

Each asset maintains a small deterministic linear return model.
After a resolved episode:

- label magnitude is clipped for numerical robustness,
- squared-error gradient updates are applied with L2 regularization,
- recent absolute prediction error is tracked as uncertainty,
- all assets update, not only the chosen asset.

Mode weights are locked before training:

- `REAL_REPLAY`: 1.00
- `BLOCK_BOOTSTRAP`: 0.50
- `STRESS_BOOTSTRAP`: 0.25

The purpose is to prevent the synthetic majority of the 100k curriculum from numerically overwhelming real historical replay.
These are training weights, not evidence weights or probabilities.

## Decision policy

The policy remains unlevered.

Default policy limits:

- max 3 simultaneous assets,
- max 25% target weight per asset,
- max 75% gross exposure,
- uncertainty penalty before an edge is actionable,
- turnover penalty to discourage needless churn,
- minimum weighted-sample warmup before trading.

### Drawdown survival throttle

Relative to each fresh $500 virtual life:

- below 15% drawdown: normal training budget,
- 15%–30%: gross budget halved,
- 30%–50%: gross budget quartered,
- at or above 50%: flat / WAIT.

This is a fixed preregistered survival rule, not a post-hoc rescue of bad results.

## Counterfactual regret diagnostic

For each resolved decision Brian records a training diagnostic comparing its chosen gross return with a feasible single-asset long/short counterfactual using the same per-asset cap.

This regret value is **training feedback only**. It is not a performance claim and does not replace stateful portfolio PnL, transaction costs, robustness testing or unseen real-data evaluation.

## Checkpoint chain

The learner exports a deterministic state dictionary containing:

- fixed configuration,
- feature contract,
- per-asset model weights,
- weighted sample counts,
- prediction-error uncertainty,
- episode/transition counters,
- counterfactual training diagnostics.

`training_state_id` is the content hash of that complete checkpoint.
A shard cannot be accepted as the continuation of another shard unless the required incoming checkpoint ID matches exactly.
A checkpoint cannot be created while an episode is half-resolved.

## Hard scientific boundaries

- any source timestamp at or after the contaminated 2026 cutoff is rejected,
- synthetic/replay learning remains `TRAINING_ONLY`,
- no automatic champion/model promotion,
- no live credentials or exchange execution,
- no free-form LLM output is an authoritative trading signal,
- no claim of profitability follows from curriculum performance.

## Activation sequence

1. CI/invariant tests,
2. deterministic micro worlds,
3. 100–1,000-life smoke curriculum,
4. inspect learning curves, turnover, drawdown, WAIT rate and checkpoint reproducibility,
5. only then consider the preregistered 100k curriculum on the verified broad dataset.

If the smoke curriculum is negative or unstable, preserve that evidence and fix methodology rather than retuning on a hidden test result.
