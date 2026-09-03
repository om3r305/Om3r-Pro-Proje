# Brian 2026 Phase 3.3 — 100k-Life Curriculum Runner

Status: **FOUNDATION / SHADOW_RESEARCH_ONLY**

The runner makes the Market Gym scalable without turning 100,000 episodes into one giant non-reproducible job or one giant trace file.

## Default curriculum

- 10,000 `REAL_REPLAY` lives
- 60,000 synchronized `BLOCK_BOOTSTRAP` lives
- 30,000 `STRESS_BOOTSTRAP` lives
- 100,000 total lives
- 5,000 lives per shard
- 20 deterministic shards

These counts are training curriculum choices, not evidence weights and not claims that 100,000 lives equal 100,000 years of independent market history.

## Causal policy boundary

While an episode is alive the policy receives only:

- market frames visible up to the current step
- current virtual equity
- current virtual portfolio weights
- current step index
- starting equity

It does **not** receive:

- future frames
- world seed
- sampled source-block recipe
- future outcome
- world receipt

A policy may update its internal training state only after an episode has resolved and the compact `EpisodeExperience` is released.

## Sharding and learner-state lineage

The global episode index is stable across shards. Sharding changes compute/storage boundaries, not the identity of a world.

Every shard receipt records:

- stable `policy_version`
- `policy_state_in`
- `policy_state_out`
- global first/last episode indices
- mode counts
- compact experience summary hash

For an adaptive learner, the next shard can require `policy_state_in == previous policy_state_out`. A mismatched checkpoint is rejected before any new life is run. This prevents 20 independent, forgetful learners from being mistaken for one continuous 100,000-life training history.

The same plan, source dataset, world-model config, episode index, policy version and initial policy state must reproduce the same shard receipt.

Each shard stores compact episode summaries. Full traces remain bounded by the experience-memory audit policy.

## Activation gate

The 100k curriculum must **not** be launched merely because the runner exists. Before the full run:

1. a broader verified multi-asset historical curriculum dataset must exist,
2. its provenance and calendars must be validated,
3. cross-market alignment must not fabricate unavailable prices,
4. the training policy/learner, its initial state and update rules must be preregistered,
5. policy checkpoints must be content-addressed/reproducible across shards,
6. synthetic/replay output must remain `TRAINING_ONLY`.

Until those gates are met, small smoke runs are allowed only to validate infrastructure.

## Why this ordering matters

Running the same narrow BTC history 100,000 times would mostly increase memorization and compute cost. The intended experience comes from diversity in assets, regimes, eras, synchronized source blocks, stresses and later event/social/on-chain contexts.

No curriculum output can automatically promote a model or replace genuinely unseen real point-in-time evaluation.
