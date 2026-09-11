# Brian Evolution OS — Guarded Rollout Runbook

Mode: **SHADOW ONLY / MAIN + ALPHA / DIP OUT OF SCOPE**

This runbook is intentionally activation-gated. Applying migrations must not start Evolution cron jobs. Runtime activation is a separate, explicit final action after migrations, Edge Functions, UI and smoke checks are in place.

## 0. Preconditions

Before merge or deployment:

- PR #92 remains draft until the owner explicitly approves merge/deploy.
- Brian Evolution OS CI, Brian ALPHA v2 CI and Brian 2026 CI must all be green on the exact PR head.
- `git diff origin/brian-2026...HEAD` must contain no DIP paths.
- Existing Brian Vault runtime secrets must be present and valid: `brian_project_url`, `brian_anon_jwt`, `brian_cron_key`.
- No live exchange credential, withdrawal route or authenticated order route is introduced by this rollout.

The feature branch may be behind `brian-2026` by unrelated DIP-only commits. Do not rebase merely to absorb DIP work. The PR synthetic merge CI is the relevant compatibility check, and Evolution's changed-file fence must remain DIP-clean.

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
14. `202609111680_brian_treasury_schedule.sql`
15. `202609111690_brian_ocean_layer6.sql`
16. `202609111700_brian_ocean_schedule.sql`
17. `202609111710_brian_evolution_activation_control.sql`

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
- no unexpected Evolution jobs exist in `cron.job`;
- canonical ALPHA remains unchanged;
- Treasury starts at `$10,000` SHADOW cash and must remain 100% cash while the Layer-4 promotion gate is closed.

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
- expected-edge challenger receipts;
- Treasury snapshots every minute;
- Ocean worker receipts every five minutes;
- no live execution anywhere;
- no DIP collector/job change caused by Evolution.

The expected Treasury behavior at the start is conservative: if Layer 4 has not earned prospective promotion, Treasury remains fully in cash. That is success, not a failure.

## 7. Ocean start gate

Do not start a 24h/48h Ocean run until its preflight sees healthy runtime evidence for the required Evolution/Treasury/Layer-4 workers.

Ocean observes the system; it does not bypass promotion gates and does not enable live execution.

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
