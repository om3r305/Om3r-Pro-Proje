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
- deterministic replay/stress/prospective metric contracts;
- Promotion Council with explicit leakage, sample, regime, edge, drawdown, stability and complexity gates;
- drift snapshots and decay-oriented research signals;
- Self-Coding Sandbox contract with a strict path allowlist, maximum file/patch budgets and mandatory provenance;
- cloud sandbox broker that creates code-candidate manifests and records TYPECHECK / UNIT / REPLAY / STRESS / PROSPECTIVE receipts;
- built-in safe challenger generator for ACTION_GATE, EXPECTED_EDGE, RELIABILITY_FEEDBACK, COST_CONTROL and DRIFT hypotheses;
- general capability-gap code generation deliberately fails closed until a specialized generator exists;
- isolated `evolution-candidate/*` branch materializer with exact-parent SHA pinning, hash validation, protected-path checks, generated test execution and **no PR/merge/canonical write**;
- manual/human handoff remains mandatory for candidate branch materialization and any later canonical promotion;
- Evolution Lab status/dashboard sections for hypotheses, experiments, candidates, review verdicts and cloud runs.

Layer 3 exit condition is satisfied at code-contract level for supported hypothesis classes: Brian can produce an isolated candidate artifact, test/evaluate it prospectively, nominate or reject it, and preserve a complete audit trail without mutating canonical behavior. Runtime activation and real prospective evidence are still pending deployment.

## Layer 4 — ALPHA Expected Edge + Reliability Feedback

Status: **CHALLENGER IMPLEMENTED / PROSPECTIVE VALIDATION PENDING**

Implemented so far:

- expected-edge decomposition: historical prospective gross directional move − exact decision-time round-trip cost − uncertainty penalty − evidence-decay penalty;
- null/unknown cost fails closed and is never coerced to zero;
- reliability feedback uses only snapshots whose `window_end` and `generated_at` are at or before the ALPHA decision timestamp;
- Bayesian/maturity shrinkage keeps measured sensor reliability bounded around the neutral 0.5 prior;
- at least two mature independent evidence groups are required;
- post-decision reliability or future source observations contaminate and block the challenger decision;
- append-only `brian_alpha_expected_edge_challenger` persistence with `ALLOW_EDGE`, `DOWNGRADE_TO_WAIT`, `COST_UNAVAILABLE`, `INSUFFICIENT_LAGGED_EVIDENCE` and `CONTAMINATED_EVIDENCE` outcomes;
- cloud expected-edge challenger schedule and authenticated status endpoint;
- Experiment Runner now measures ACTION_GATE plus EXPECTED_EDGE / RELIABILITY_FEEDBACK / COST_CONTROL challengers against 15-minute prospective after-cost outcomes;
- Promotion Council can evaluate those challenger/control result pairs without mutating canonical ALPHA.

Not yet promoted:

- canonical ALPHA still uses its existing decision compiler;
- reliability weights are challenger outputs, not canonical weights;
- `gross_edge_bps` / `net_edge_bps` on canonical ALPHA are not being rewritten.

Layer 4 promotion requires enough clean prospective samples/regimes showing better after-cost edge and acceptable drawdown/stability. Until that evidence exists, this remains challenger-only.

## Layer 5 — $10,000 SHADOW Brian Treasury

Status: **CODE COMPLETE / ROLLOUT + PROSPECTIVE VALIDATION PENDING**

Implemented:

- one unified `$10,000` Brian SHADOW cash pool rather than independent per-signal fake tickets;
- Layer-4 promotion gate: Treasury may deploy only when the latest EXPECTED_EDGE prospective experiment is `PROMOTE_CANDIDATE`;
- a closed/revoked Layer-4 gate blocks new allocation and fail-closes any existing SHADOW allocation back to cash;
- max 70% total deployment, minimum 30% cash reserve, max 12% per position, minimum 2.5% sizing band and max eight simultaneous positions;
- sizing uses expected net edge, bounded reliability confidence and mature independent-group breadth;
- entry/exit costs are charged against the Treasury ledger rather than displayed only as diagnostics;
- exit brain supports direction flip, edge invalidation, stale edge, hard risk stop, profit-edge decay, time decay and opportunity replacement;
- replacement logic can recycle capital from a weaker position when a materially better validated opportunity appears;
- same-cycle re-open protection prevents a stopped/invalidated asset from immediately reopening from stale evidence;
- latest marks are timestamp ordered and reserve/deployment formulas account for entry costs;
- append-only `brian_treasury_shadow_snapshots` and normalized `brian_treasury_shadow_actions` persistence;
- one atomic/idempotent DB commit function writes each Treasury snapshot and its actions in a single transaction;
- cloud Treasury worker, one-minute schedule preparation, authenticated Treasury status endpoint and `/treasury.html` mobile dashboard;
- Evolution UI links directly to Treasury while keeping browser/mobile observation-only.

The Treasury remains SHADOW ONLY. No exchange order, credential, withdrawal or live-money route exists. The worker and schedule are not active until the draft PR is explicitly merged/deployed. If Layer 4 has not earned prospective promotion at runtime, the expected steady state is **100% cash**.

## Layer 6 — Ocean Run

Status: **CODE COMPLETE / ROLLOUT + 24–48H PROSPECTIVE RUN PENDING**

Implemented:

- append-only Ocean START/STOP command ledger with only 24h or 48h planned durations;
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

## Regression checkpoint

The Layer 0–6 code path has passed Deno type-checking, all 59 Evolution behavioral tests, Deno lint, dashboard JavaScript syntax checks and the dedicated no-DIP-path guard. The existing Brian 2026 and ALPHA regression suites have also passed on the same implementation line. The branch remains draft until its final current-head CI pass and migration review are complete.

## Completion state before rollout

Layers 0–6 now exist at code-contract level in the draft integration branch. Remaining sequence: final CI -> migration/order review -> reconcile the moving `brian-2026` base if required -> explicit merge/deploy approval -> cloud rollout verification -> actual 24–48h Ocean observation.

## Activation rule

PR #92 stays draft and unmerged while the stack is being assembled. Migrations, Edge Functions, cron schedules and dashboard additions in this branch are preparation only until an explicit rollout. Final activation requires green CI, confirmed DIP isolation, migration/deployment review and explicit user approval.
