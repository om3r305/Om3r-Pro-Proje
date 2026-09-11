# Brian Evolution OS — Layer Status

Status: **DRAFT / GitHub only / not deployed**

This file is the living implementation ledger for PR #92. It tracks what is actually implemented in code, not what is merely planned.

## Boundary

- MAIN / ALPHA only.
- `/dip` and all DIP treasury/runtime/data/control surfaces are excluded.
- SHADOW ONLY.
- No authenticated exchange execution, withdrawals, or live-money routes.
- Browser/mobile is observation/control only; cloud workers are designed to continue without an open page.
- Autonomous code candidates may not mutate DIP, CI/security, auth/secrets, migrations/control-plane, or protected production surfaces.

## Layer 0 — Baseline & protected scope

**Implemented in PR #92.**

- Pre-Evolution MAIN/ALPHA baseline captured in `config/brian-evolution-baseline-v1.json`.
- Evolution lifecycle and protected-path contract in `_shared/evolution_contract.ts`.
- Tests block autonomous DIP, CI, auth/secrets and DB-control-plane edits.
- Append-only/RLS Evolution ledger foundation added.
- Code-candidate persistence enforces `autonomous_apply_allowed=false`.

## Layer 1 — Evolution Core

**Implemented in GitHub; not deployed.**

- Capability Graph derives HEALTHY / DEGRADED / STALE / MISSING state from prospective runtime evidence.
- Gap Detector turns missing/stale capabilities into prioritized research gaps.
- World Explorer discovers source domains from prospective intel events.
- Source assessment scores authority, freshness, manipulation risk, corroboration and access.
- Discovered/VERIFYING sources cannot directly influence ALPHA.
- Cloud `brian-evolution-orchestrator` persists capability snapshots, gaps, source assessments and Evolution Journal events.
- `brian-evolution-status` provides authenticated read-only dashboard data.
- `/evolution.html` provides a mobile-readable Evolution dashboard.
- Browser-independent cron preparation exists; deploy-time Vault secrets are required before schedules activate.

## Layer 2 — World Brain

**Implemented as first complete World Brain slice in GitHub; not deployed.**

- Broad rotating World Discovery Eye covers technology/AI, macro/rates, commodities/energy, geopolitics, corporate/product events and crypto/regulation.
- GDELT remains discovery-only; it never casts a directional ALPHA vote.
- World Brain creates event frames, entity observations and low-confidence co-mention/supply-chain assertions.
- Narrative Radar recognizes AI compute, semiconductors, monetary policy, inflation, geopolitics, energy, crypto regulation, ETF flows, stablecoins, cybersecurity, product launches, earnings and token events.
- Future Calendar only accepts explicit future timestamps; it does not invent dates.
- Causal mechanisms are stored as research hypotheses with mandatory counter-evidence.
- Scenario engine creates `MECHANISM_HOLDS` and `MECHANISM_BREAKS` branches.
- Cross-asset impact candidates are conditional and have `direct_alpha_influence=false`.
- World Brain includes initial conditional transmission maps for monetary policy, geopolitics, energy supply, AI compute, crypto regulation and product launches.
- `brian-world-status` and `/world.html` expose narratives, future events, entity relations, causal mechanisms, scenario branches and runtime health.
- Cloud schedules for discovery and World Brain are prepared but not active before merge/deploy.

## Layer 3 — Self-Improvement Lab

**Foundation and autonomous researcher implemented; sandbox code generation remains in progress.**

Implemented:

- Hypothesis Engine converts measured gaps, challenger disagreement, negative after-cost outcomes, frozen reliability and cost burden into falsifiable research hypotheses.
- Experiment Factory creates prospective SHADOW experiment plans with minimum sample/regime gates.
- Promotion Council rejects leakage/data-quality failures and requires positive prospective net edge plus control improvement before `SHADOW_CANDIDATE` nomination.
- Drift detector measures baseline-vs-recent deterioration.
- Code-candidate planner enforces protected-path guards and can never autonomously apply a patch.
- Append-only experiment/result/drift/promotion/code-review persistence added.
- `brian-evolution-researcher` reads current gaps, calibration challenger, ALPHA outcomes and reliability snapshots and creates hypothesis/experiment candidates automatically.
- 30-minute cloud researcher schedule is prepared but not active before deployment.

Still required before Layer 3 is complete:

- sandbox code-generation provider/adapter,
- isolated candidate workspace/branch builder,
- automated replay/stress/prospective result ingestion,
- drift-driven DECAYING/RETIRED recommendations,
- human-review handoff for promotion-ready patches.

No code-generation worker will receive production/deploy credentials. Candidate patches must remain in isolated review branches/workspaces and pass the same protected-path, replay, stress and prospective gates before human-approved promotion.

## Layer 4 — ALPHA Intelligence

**Not implemented yet.**

Target: expected gross/net edge, cost/uncertainty/decay model, bounded prospective reliability feedback, opportunity ranking and canonical promotion gates.

## Layer 5 — Brian Treasury

**Not implemented yet.**

Target: unified `$10,000` SHADOW cash pool, allocation, opportunity replacement, concentration/correlation/liquidity/risk reasoning, exits and capital recycling.

## Layer 6 — Ocean Run

**Not implemented yet.**

Target: 24–48 hour cloud-only SHADOW run with complete evidence on what Brian observed, learned, coded, rejected, promoted and how the Treasury performed.

## CI

A dedicated `Brian Evolution OS CI` workflow type-checks/lints/tests Evolution and World Brain code, validates dashboard JavaScript, rejects DIP-path changes in this PR, and checks patch formatting. Existing Brian and ALPHA CI remain active as independent regression gates.

Latest checkpoint at branch head `221c91368d50089b1445c58dce43b55c228e6e68`: **Brian Evolution OS CI = success, Brian ALPHA v2 CI = success, Brian 2026 CI = success.**

## Activation rule

PR #92 remains a draft until all planned layers are complete, CI is green, DIP isolation is verified, and the user explicitly authorizes merge + production rollout.
