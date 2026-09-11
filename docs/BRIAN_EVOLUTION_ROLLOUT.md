# Brian Evolution OS — Guarded Rollout Runbook

Mode: **SHADOW ONLY / MAIN + ALPHA / DIP OUT OF SCOPE**

This runbook is intentionally activation-gated. Applying migrations must not start Evolution cron jobs. Runtime activation is a separate, explicit final action after migrations, Edge Functions, UI and smoke checks are in place.

## 0. Preconditions

Before merge or deployment:

- PR #92 remains draft until the owner explicitly approves merge/deploy.
- Brian Evolution OS CI, Brian ALPHA v2 CI and Brian 2026 CI must all be green on the exact PR head against the current `brian-2026` base.
- Synchronize the Evolution branch with the current `brian-2026` head before the final exact-head CI. Base-only DIP commits may be inherited by the merge commit, but Evolution must not author or modify DIP paths.
- `git diff origin/brian-2026...HEAD` must contain no DIP paths.
- Existing Brian Vault runtime secrets must be present and valid: `brian_project_url`, `brian_anon_jwt`, `brian_cron_key`.
- No live exchange credential, withdrawal route or authenticated order route is introduced by this rollout.
- After the final merge, set the Supabase Edge Function secret `BRIAN_EVOLUTION_PARENT_COMMIT` to the exact 40-character SHA at the head of `brian-2026`. The self-coding planner intentionally fails closed if this value is missing, malformed or stale relative to candidate materialization.

If canonical `brian-2026` changes later, update `BRIAN_EVOLUTION_PARENT_COMMIT` before expecting new self-code candidates. Existing evidence remains valid; only new candidate planning should pause until the exact parent is refreshed.

## 1. Migration order

Apply the Evolution migrations in filename order. The important dependency chain is:

1. `202609111330_brian_evolution_os_foundation.sql`
2. `202609111430_brian_evolution_os_layer1.sql`
3. `202609111500_brian_world_brain_layer2.sql`
4. `202609111510_brian_evolution_cloud_schedule.sql`
5. `202609111520_brian_world_discovery_schedule.sql`
6. `202609111600_brian_evolution_research_layer3.sql`
7. `202609111610_brian_evolution_research_schedule.sql`
8. `202609111620_brian_evolution_sandbox_layer3.sql`
9. `202609111630_brian_evolution_lab_schedule.sql`
10. `202609111640_brian_evolution_codegen_schedule.sql`
11. `202609111650_brian_alpha_intelligence_layer4.sql`
12. `202609111660_brian_alpha_intelligence_schedule.sql`
13. `202609111670_brian_treasury_layer5.sql`
14. `202609111675_brian_treasury_atomic_cas_guard.sql`
15. `202609111680_brian_treasury_schedule.sql`
16. `202609111690_brian_ocean_layer6.sql`
17. `202609111695_brian_ocean_atomic_control_guard.sql`
18. `202609111700_brian_ocean_schedule.sql`
19. `202609111710_brian_evolution_activation_control.sql`

The `1675` Treasury hardening migration makes the append-only cashbox a serialized compare-and-swap chain at the database boundary. Exact retries remain idempotent, while stale-parent or non-monotonic forks fail closed even if a caller races without respecting the worker lease.

The `1695` Ocean control hardening migration serializes START/STOP at the database boundary and revokes direct service-role INSERT access to the command log. Two concurrent dashboard START requests therefore cannot create two active Ocean runs; command writes must pass the guarded RPCs.

**Expected state after migrations:** tables/functions exist, but Evolution cron job count is **0**. Existing Brian/DIP schedules are untouched.

## 2. Edge Function deployment

Deploy the Evolution functions before activation:

- `brian-evolution-orchestrator`
- `brian-world-brain`
- `brian-world-discovery-eye`
- `brian-evolution-researcher`
- `brian-evolution-sandbox`
- `brian-evolution-template-generator`
- `brian-evolution-experiment-runner`
- `brian-evolution-promotion-council`
- `brian-evolution-alpha-edge-challenger`
- `brian-evolution-alpha-intelligence-status`
- `brian-evolution-treasury`
- `brian-evolution-treasury-status`
- `brian-evolution-ocean-control`
- `brian-evolution-ocean-worker`
- `brian-evolution-ocean-status`
- `brian-evolution-status`
- `brian-evolution-lab-status`
- `brian-world-status`

Before smoke-testing `brian-evolution-sandbox`, verify `BRIAN_EVOLUTION_PARENT_COMMIT` equals the exact current `brian-2026` SHA. Do not substitute a short SHA, old feature-branch parent or fallback value.

Do not invoke the cron scheduler functions yet.

## 3. UI deployment

Deploy the observation/control pages:

- `/evolution.html`
- `/world.html`
- `/treasury.html`
- `/ocean.html`

Browser/mobile remains an observation/control surface only. Closing the page must not be required for cloud workers to continue.

## 4. Pre-activation smoke checks

Before activating cron:

- authenticated status endpoints return JSON rather than 404/500;
- Evolution status can read the new persistence tables;
- Treasury status reports SHADOW mode and no live execution;
- Ocean status/control can read command/report tables;
- self-code sandbox `plan` either succeeds with the exact canonical parent or fails closed with an explicit parent-commit error; it must never silently fall back to an old commit;
- no unexpected Evolution jobs exist in `cron.job`;
- canonical ALPHA remains unchanged;
- expected-edge reliability is bound to the exact source observation / raw group / sensor family / horizon that actually voted in the decision; `intrabar_tape` must resolve back to its raw micro sensor lineage;
- Treasury starts at `$10,000` SHADOW cash and must remain 100% cash while the Layer-4 promotion gate is closed;
- Treasury cycle persistence rejects a stale `previous_snapshot_id` and accepts an exact completed-cycle retry idempotently;
- two concurrent Ocean START attempts serialize to exactly one active run, and direct service-role command inserts are denied.

Database check:

```sql
select brian_private.evolution_activation_status();
```

Expected before activation: `active_job_count = 0`, `fully_active = false`.

## 5. Explicit activation

Only after the owner explicitly approves activation, run exactly one database action:

```sql
select brian_private.activate_evolution_os();
```

The activation is designed as one transaction. If any scheduler fails because a required secret/function is missing, the activation should roll back rather than leave a half-enabled Evolution stack.

Then verify:

```sql
select brian_private.evolution_activation_status();
```

Expected: `active_job_count = 11`, `fully_active = true`.

## 6. First runtime verification

After activation, wait long enough for each cadence to fire and verify collector receipts. Required observations include:

- orchestrator and World Brain runs;
- World Discovery Eye runs;
- researcher, sandbox, experiment runner and Promotion Council receipts;
- expected-edge challenger receipts with exact decision evidence lineage;
- Treasury snapshots every minute with a single monotonic parent chain;
- Ocean worker receipts every five minutes;
- self-code requests carry the exact canonical parent SHA and remain isolated from canonical branches;
- no live execution anywhere;
- no DIP collector/job change caused by Evolution.

The expected Treasury behavior at the start is conservative: if Layer 4 has not earned prospective promotion, Treasury remains fully in cash. That is success, not a failure.

## 7. Ocean start gate

Do not start a 24h/48h Ocean run until its preflight sees healthy runtime evidence for the required Evolution/Treasury/Layer-4 workers.

Ocean observes the system; it does not bypass promotion gates and does not enable live execution. Ocean START/STOP commands must pass the serialized database control RPCs installed by migration `1695`.

## 8. Emergency rollback / pause

To immediately stop Evolution cron workers without touching data or DIP:

```sql
select brian_private.deactivate_evolution_os();
```

This removes only the 11 known Evolution job names. It does not delete Evolution evidence, Treasury history, Ocean reports, existing Brian schedules or any DIP schedule.

After deactivation, verify:

```sql
select brian_private.evolution_activation_status();
```

Expected: `active_job_count = 0`.

Do not delete evidence tables to roll back runtime behavior. Preserve the audit trail.

## 9. Promotion boundary

A green rollout does **not** mean Layer-4 expected edge is proven. Canonical ALPHA remains unchanged until prospective evidence satisfies the Promotion Council gates. Treasury allocation remains dependent on that promotion gate. Real-money execution is outside this rollout.
