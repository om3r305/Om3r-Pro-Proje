# Brian Evolution OS — Layer Status

Branch: `feat/brian-evolution-os`
Integration PR: #92
Base: `brian-2026`
Mode: **GITHUB DRAFT / SHADOW ONLY / NOT DEPLOYED**

## Non-negotiable boundary

- MAIN / ALPHA Evolution work only.
- `/dip`, DIP runtime, DIP treasury, DIP data and DIP control-plane remain outside Evolution OS.
- No authenticated exchange orders, withdrawals or live-money execution.
- Browser/mobile is observation/control only; runtime workers are designed for cloud cron operation.
- No generated candidate may autonomously modify canonical ALPHA, CI/security, credentials/auth, migrations/control-plane or protected DIP paths.

## Layer 0 — Baseline + protected-scope fence

Status: **CODE COMPLETE**

Implemented:

- machine-readable pre-Evolution MAIN/ALPHA baseline;
- lifecycle contract from DISCOVERED through ARCHIVED;
- hard path fences for DIP, CI/security, auth/secrets and DB control-plane;
- stage-skip rejection and fail-closed behavioral tests;
- append-only Evolution event/capability/source/hypothesis/code-candidate foundations.

## Layer 1 — Capability Graph + World Explorer + Evolution Ledger

Status: **CODE COMPLETE / RUNTIME ROLLOUT PENDING**

Implemented:

- Capability Graph derived from actual collector evidence;
- capability-gap detector;
- public-source discovery + trust assessment with discovery != truth;
- Evolution orchestrator and dashboard status endpoint;
- append-only source assessments, gap snapshots and orchestrator run receipts;
- cloud schedules using existing Brian Vault auth boundaries.

No source discovered by World Explorer receives direct ALPHA influence in this layer.

## Layer 2 — World Brain

Status: **CODE COMPLETE / RUNTIME ROLLOUT PENDING**

Implemented:

- entity graph and typed relations;
- narrative clusters;
- future-event calendar that requires explicit future timestamps;
- conditional causal mechanisms with counter-evidence slots;
- scenario paths and cross-asset implications;
- broad World Discovery Eye for AI/technology, macro/rates, energy/commodities, geopolitics, corporate/product and crypto/regulation domains;
- World Intelligence status/UI surfaces;
- append-only persistence and cloud scheduling.

World Brain remains a research/context organ until prospective validation promotes a capability. It cannot directly vote BUY/SELL merely because a headline exists.

## Layer 3 — Self-Improvement Lab

Status: **BOUNDED LOOP CODE COMPLETE / RUNTIME EVIDENCE PENDING**

Implemented:

- Hypothesis Engine that converts observed capability gaps, calibration failures, negative after-cost outcomes, reliability gaps and drift into falsifiable research candidates;
- Experiment Factory with prospective control/challenger plans and contamination declarations;
- stable experiment identity so hourly research refreshes do not reset sample maturity;
- deterministic replay/stress/prospective metric contracts;
- Promotion Council with explicit leakage, sample, regime, edge, drawdown, stability and complexity gates;
- drift snapshots and decay-oriented research signals;
- Self-Coding Sandbox contract with a strict path allowlist, maximum file/patch budgets and mandatory provenance;
- exact 40-character canonical parent pinning with no stale fallback;
- parent rotation creates a fresh candidate identity while preserving old evidence;
- cloud sandbox broker that creates code-candidate manifests and records TYPECHECK / UNIT / REPLAY / STRESS / PROSPECTIVE receipts;
- built-in safe challenger generator for ACTION_GATE, EXPECTED_EDGE, RELIABILITY_FEEDBACK, COST_CONTROL and DRIFT hypotheses;
- automatic prospective measurement is intentionally limited to ACTION_GATE and EXPECTED_EDGE until dedicated reliability/cost challenger labels exist; unsupported experiment lineage fails closed rather than borrowing EXPECTED_EDGE evidence;
- general capability-gap code generation deliberately fails closed until a specialized generator exists;
- isolated `evolution-candidate/*` branch materializer with exact-parent SHA pinning, hash validation, protected-path checks, generated test execution and **no PR/merge/canonical write**;
- candidate materialization stages all generated files before the path fence, permits only additive isolated files, does not persist checkout credentials, executes generated tests without filesystem/network permissions, and exposes the write token only in the final non-code-execution push step;
- manual/human handoff remains mandatory for candidate branch materialization and any later canonical promotion;
- Evolution Lab status/dashboard sections for hypotheses, experiments, candidates, review verdicts and cloud runs.

Layer 3 exit condition is satisfied at code-contract level for supported hypothesis classes. Runtime activation and real prospective evidence are still pending deployment.

## Layer 4 — ALPHA Expected Edge + Reliability Feedback

Status: **CHALLENGER IMPLEMENTED / PROSPECTIVE VALIDATION PENDING**

Implemented so far:

- expected-edge decomposition: lagged prospective gross directional move − exact decision-time round-trip cost − uncertainty penalty − evidence-decay penalty;
- null/unknown cost fails closed and is never coerced to zero;
- reliability feedback uses only snapshots whose `window_end` and `generated_at` are at or before the ALPHA decision timestamp;
- reliability is bound to the exact decision source observation + original independent group + sensor family + horizon instead of a look-alike group row;
- compiler-collapsed `intrabar_tape` evidence resolves back to the actual raw micro sensor lineage that voted;
- historical `avg_signed_bps` is treated correctly as already sensor-direction aligned and is not flipped a second time for SHORT decisions;
- Bayesian/maturity shrinkage keeps measured sensor reliability bounded around the neutral 0.5 prior;
- at least two mature independent evidence groups are required;
- post-decision reliability or future source observations contaminate and block the challenger decision;
- append-only `brian_alpha_expected_edge_challenger` persistence with fail-closed outcomes;
- cloud expected-edge challenger schedule and authenticated status endpoint;
- Treasury/runtime prospective timing budget is aligned with the two-minute challenger cadence: expected-edge labels remain bounded but may arrive within three minutes;
- Experiment Runner measures challenger/control results prospectively once per evidence cycle and explicitly marks timing leakage as contaminated evidence;
- Promotion Council can evaluate those result pairs without mutating canonical ALPHA and may refresh/revoke a verdict on the next hourly evidence cycle rather than waiting three hours.

Not yet promoted:

- canonical ALPHA still uses its existing decision compiler;
- reliability weights are challenger outputs, not canonical weights;
- `gross_edge_bps` / `net_edge_bps` on canonical ALPHA are not being rewritten.

Layer 4 promotion requires enough clean prospective samples/regimes showing better after-cost edge and acceptable drawdown/stability. Until that evidence exists, this remains challenger-only.

## Layer 5 — $10,000 SHADOW Brian Treasury

Status: **CODE COMPLETE / FINAL REGRESSION GREEN / ROLLOUT VALIDATION PENDING**

Implemented:

- one unified `$10,000` Brian SHADOW cash pool rather than independent per-signal fake tickets;
- Layer-4 promotion gate: Treasury may deploy only while a current EXPECTED_EDGE prospective experiment retains a valid `PROMOTE_CANDIDATE` authority;
- newer KEEP_EXPERIMENTAL decisions on unrelated EXPECTED_EDGE experiments do not revoke another experiment's still-valid PROMOTE, while a later verdict on the same experiment does revoke its older authority;
- a closed/revoked Layer-4 gate blocks new allocation and fail-closes any existing SHADOW allocation back to cash;
- no fixed `$3/$5/$10/$20` Treasury ticket size, mandatory 30% reserve, arbitrary 70% deployment ceiling or 12% per-position ceiling;
- conviction sizing uses after-cost expected net edge, bounded reliability confidence and mature independent-group breadth;
- ordinary evidence receives partial capital while exceptionally strong validated evidence can use effectively all available SHADOW equity after reserving point-in-time entry cost;
- maximum eight simultaneous positions remains a book-complexity bound, not a forced diversification target;
- entry/exit costs are charged against the Treasury ledger rather than displayed only as diagnostics;
- exit brain supports direction flip, edge invalidation, stale edge, hard risk stop, profit-edge decay, time decay and opportunity replacement;
- opportunity replacement is not limited to a full book: a materially superior candidate can liquidate one or more weaker positions, including at a realized loss, to fund its higher conviction target;
- replacement compares SWITCH against the remaining HOLD value and explicitly charges the old position's incremental exit cost instead of using a raw edge-gap threshold alone;
- same-cycle re-open protection prevents a stopped/invalidated/replaced asset from immediately reopening from stale evidence;
- latest marks are timestamp ordered and sizing formulas reserve exact entry cost rather than allowing negative cash;
- full-conviction adverse MTM no longer creates an invalid `deployment_pct > 1` persistence state: utilization is normalized to deployed principal versus max(MTM equity, deployed principal), while leverage remains zero;
- missing fresh marks no longer abort the entire Treasury worker: bounded historical marks keep risk/gate exits alive, while stale evidence remains non-actionable for new deployment;
- append-only `brian_treasury_shadow_snapshots` and normalized `brian_treasury_shadow_actions` persistence;
- database-level advisory lock + parent compare-and-swap serialize Treasury history, reject stale forks/non-monotonic cycles and keep exact retries idempotent;
- real Postgres CI now applies the Treasury/Ocean hardening migrations and verifies stale-parent rejection, exact retry idempotence, full-conviction drawdown persistence and concurrent-child fork rejection;
- cloud Treasury worker, one-minute schedule preparation, authenticated Treasury status endpoint and `/treasury.html` mobile dashboard;
- Evolution UI links directly to Treasury while keeping browser/mobile observation-only.

The Treasury remains SHADOW ONLY. No exchange order, credential, withdrawal or live-money route exists. The worker and schedule are not active until the draft PR is explicitly merged/deployed. If Layer 4 has not earned prospective promotion at runtime, the expected steady state is **100% cash**.

## Layer 6 — Ocean Run

Status: **CODE COMPLETE / FINAL REGRESSION GREEN / ROLLOUT + 24–48H PROSPECTIVE RUN PENDING**

Implemented:

- append-only Ocean START/STOP command ledger with only 24h or 48h planned durations;
- database-serialized START/STOP RPCs; direct service-role command inserts are revoked so concurrent dashboard requests cannot create parallel active exams;
- Ocean command time is bound to the database clock with a narrow caller-clock skew check, so a privileged caller cannot backdate a START to bypass the active-run overlap window;
- real Postgres CI races two concurrent Ocean START calls and requires exactly one winner, rejects direct service-role command inserts and verifies wrong-run STOP rejection;
- dashboard-authenticated preflight that requires Treasury, Layer-4 expected-edge and key Evolution workers to have healthy runtime evidence before Ocean can start;
- browser-independent `brian-evolution-ocean-worker` with cloud lease protection and five-minute schedule preparation;
- active-run checkpoints capturing Treasury equity/cash/deployment/open positions plus recent collector health;
- deterministic run-state reconstruction from immutable commands instead of a mutable browser session flag;
- early STOP support without rewriting original planned duration;
- automatic final report generation after planned or early termination;
- final report covers beginning/ending Treasury, P&L/costs, actions/replacements, discoveries, hypotheses, code candidates, experiment outcomes, promotion/rejection, material drift, capability events, missed opportunities, ALPHA after-cost outcomes and collector health;
- authenticated Ocean status/control endpoints and `/ocean.html` mobile dashboard;
- Ocean behavioral tests cover lifecycle, 24/48h constraints, early stop and report reconciliation.

Ocean does not bypass Layer 4 or Treasury gates. It is an observation/exam envelope around the existing SHADOW stack. Starting an Ocean run does not enable live execution and does not create exchange credentials or orders.

## External red-team remediation checkpoint

Independent reviews raised concrete issues in candidate materialization, Treasury full-conviction persistence, replacement economics, mark-gap behavior, promotion semantics, prospective latency, Ocean caller-time trust and SQL/runtime CI coverage. The confirmed findings were reproduced against the actual branch and remediated before this checkpoint.

Current exact code checkpoint before any documentation-only refresh was `e253ff7c824f25b7eea35089f54009cb85ca31f5` against base `fb8412e9438828b635fd3c96b50f9b06925a760e` (187 commits ahead / 0 behind). On that exact code head:

- Brian Evolution OS CI #167: **SUCCESS**;
- Evolution behavioral suite: **87 passed / 0 failed**;
- real Postgres Evolution Treasury/Ocean concurrency job: **SUCCESS**;
- Brian ALPHA v2 CI #364: **SUCCESS**;
- Brian 2026 CI #652: **SUCCESS**;
- dormant-schedule guard: **SUCCESS**;
- DIP path + SQL-control-content guard: **SUCCESS**;
- type-check, lint, dashboard JS and patch checks: **SUCCESS**.

This documentation commit itself is non-runtime, but final merge readiness still requires the workflows on the resulting exact PR head to remain green.

## Completion state before rollout

Layers 0–6 exist at code-contract level in the draft integration branch. Remaining sequence: exact-head regression after this documentation refresh -> explicit merge/deploy approval -> cloud rollout with schedules dormant -> pre-activation smoke checks -> explicit activation approval -> actual 24–48h Ocean observation.

## Activation rule

PR #92 stays draft and unmerged while the stack is being assembled/reviewed. Migrations, Edge Functions, cron schedules and dashboard additions in this branch are preparation only until an explicit rollout. Final activation requires green CI, confirmed DIP isolation, migration/deployment review and explicit user approval.
