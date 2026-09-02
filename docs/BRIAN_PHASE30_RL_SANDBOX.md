# Brian Phase 3.0 — Causal RL Sandbox Challenger

Status: PREREGISTERED FOUNDATION

Execution: SHADOW_RESEARCH_ONLY

Final holdout: 2026 remains INVALID_CONTAMINATED and evaluation is forbidden.

## Goal

Phase 3.0 adds a reinforcement-learning challenger without giving an RL policy any live execution authority. The first objective is to prove that the sequential decision problem, reward accounting, timing and evaluation are causal and reproducible before adding heavier neural RL algorithms.

## Fixed causal timing

- Observation at bar `t` may use only information available after the completed `t` bar.
- The earliest execution price available to an action selected from that observation is the **open of bar `t+1`**.
- Same-close observation/execution is forbidden.
- Reward may use the subsequent realized market path because reward is an outcome, never an observation.
- Every observation and every execution timestamp must be strictly earlier than 2026-01-01T00:00:00Z.

## State and actions

State = PIT-safe market features plus explicit current position state.

Each market feature is encoded as `(value, missing_flag)`. An unavailable feature is therefore not silently treated as a genuine numeric zero. Position is then appended as a three-way one-hot state.

Target actions are fixed to:

- `-1`: target SHORT
- `0`: target FLAT / abstain
- `+1`: target LONG

The position is one-hot encoded into the learner state so switching costs and sequential exposure are observable to the policy.

## Reward

For a target exposure applied at the next bar open:

`reward = signed(open->close return) - transition costs - fixed adverse-excursion risk penalty`

Fixed defaults:

- fee: 10 bps per side
- assumed spread: 2 bps full spread
- slippage: 1 bp per side
- adverse-excursion risk penalty coefficient: 0.20

Changing from SHORT directly to LONG pays two one-way exposure changes. Costs are not hidden inside labels.

## First challenger

The first RL challenger is deterministic Conservative Fitted-Q using `ExtraTreesRegressor`, already available through the existing scikit-learn dependency.

Fixed defaults:

- gamma: 0.97
- fitted-Q iterations: 5
- trees per action: 96
- minimum samples per leaf: 25
- minimum action advantage over FLAT: 0.015 percentage points
- maximum training transitions: 250,000
- deterministic evenly-spaced reduction only when the cap is exceeded
- random seed: 30

No outcome-based sampling is allowed.

## Scientific restrictions

- Learner fit is TRAIN ONLY.
- Validation may later be used only for explicitly preregistered challenger gating; test outcomes may never alter model weights, hyperparameters or promotion rules.
- RL is a challenger, not a champion.
- No automatic promotion to Brian configuration.
- No authenticated exchange API, credentials, order submission, self-modifying code or live trading surface.
- No fabricated historical order book or order-flow features.
- Negative evidence must be retained.
- A neural PPO/SAC/TD3 agent must not be introduced merely because Fitted-Q underperforms. Any heavier agent requires its own preregistered phase or amendment committed before its development evidence is inspected.

## Required invariants before merge

1. next-open execution is enforced;
2. observation/execution at or beyond the 2026 cutoff hard-fails;
3. full counterfactual action coverage is deterministic;
4. position state is included in the Markov state;
5. unavailable evidence remains distinguishable from genuine numeric zero;
6. RL fitting is train-only;
7. repeated training with identical inputs is deterministic;
8. no exchange execution methods exist;
9. existing Brian CI, compile, import and hard-shadow checks pass.

## What this phase does not claim

This foundation does not claim that RL improves profitability, drawdown, hit rate or robustness. It only creates a causal, cost-aware and auditable RL research surface. A later Phase 3.0 development experiment must compare the frozen RL challenger against the frozen Phase 2.8/2.9 baselines under the same stateful portfolio and cost assumptions before any scientific claim is made.
