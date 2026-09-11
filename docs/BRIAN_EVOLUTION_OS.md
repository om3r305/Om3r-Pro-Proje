# Brian Evolution OS

Status: DRAFT / GITHUB-ONLY / NOT DEPLOYED

Target branch: `brian-2026`

## Product identity

Brian is the market-intelligence operating system. ALPHA is Brian's market-decision organ. Collectors, sensors, L2, derivatives, macro, news, social/on-chain adapters and future data providers are Brian's eyes and ears.

The goal of Evolution OS is not unrestricted self-modification. The goal is a continuously learning research-and-engineering loop that can discover new information, identify missing capabilities, propose code, test it, measure it prospectively, and only then make it eligible for promotion.

## Non-negotiable boundaries

1. `/dip` is a separate experiment with a separate treasury, runtime and decision system. Evolution OS must not modify, deploy, schedule, query-for-control, promote, pause, restart, or rebalance DIP.
2. Evolution OS remains `SHADOW_ONLY` until a separate explicit production-money project is authorized. No authenticated exchange order, withdrawal or transfer surface belongs in this project.
3. Brian may propose and write code in a sandbox branch, but it may not grant itself new secrets, disable evidence logging, delete adverse results, raise its own hard safety permissions, or self-promote directly to canonical production.
4. Every material self-change must be attributable: hypothesis -> source evidence -> code artifact -> tests -> experiment -> prospective result -> promotion decision.
5. Old knowledge is never silently overwritten. Drift can decay/retire a capability, but evidence and lineage remain auditable.
6. Browser/mobile is a control and observation surface only. Long-running Brian work must be cloud/server-side and continue when the phone or page is closed.

## Delivery strategy

This work is accumulated on one long-lived draft PR. Nothing in the PR is deployed merely because it exists on GitHub. The PR stays unmerged until all layers are complete, tested and reviewed. Final activation is a separate explicit merge/deploy decision.

### Layer 0 — Baseline and scope fence

- freeze a machine-readable pre-Evolution baseline
- define protected DIP path/resource patterns
- define MAIN/ALPHA capability inventory
- add CI guards proving Evolution code cannot target protected DIP surfaces

Exit gate: baseline reproducible; DIP exclusion tests pass.

### Layer 1 — Evolution Core

- Capability Graph
- World Source Registry / Source Discovery receipts
- Evolution Journal / append-only event ledger
- capability-gap detector
- cloud Evolution orchestrator
- dashboard Evolution status model

Lifecycle:

`DISCOVERED -> VERIFYING -> RESEARCHING -> EXPERIMENTAL -> SHADOW_CANDIDATE -> ACTIVE`

with terminal/decay paths:

`REJECTED`, `DECAYING`, `RETIRED`, `ARCHIVED`.

Exit gate: Brian can explain what capabilities it has, what is missing, what it discovered, and why a candidate is or is not trusted.

### Layer 2 — World Brain

- global asset/entity graph
- company/product/supply-chain relations
- macro/commodity/rates/equity/crypto context
- future event calendar
- narrative/theme tracking
- causal-mechanism records
- scenario tree and cross-asset impact paths

Exit gate: an event can be traced from source -> entities -> causal mechanisms -> exposed assets -> uncertainty/counter-evidence.

### Layer 3 — Self-Improvement Lab

- Hypothesis Engine
- Experiment Factory
- sandbox code artifact contract
- automated bounded challenger-code generation
- isolated candidate branch materialization
- replay/stress/prospective challenger contracts
- promotion council evidence bundle
- failure memory and concept-drift retirement

Exit gate: Brian can identify a measurable weakness, propose a bounded change, generate an isolated candidate artifact, test it, and produce an auditable accept/reject result without mutating canonical behavior.

### Layer 4 — ALPHA Intelligence Upgrade

Implemented as a challenger path pending prospective promotion:

- prospective reliability feedback into bounded challenger weights
- expected gross move estimate from lagged prospective evidence
- exact decision-time round-trip cost
- uncertainty/decay penalties
- expected net edge
- prospective control/challenger measurement
- promotion council gate

Canonical decision target:

`expected_net_edge = expected_gross_move - estimated_round_trip_cost - uncertainty_penalty - event_decay_penalty`

Evidence score alone must never be interpreted as expected return. Canonical ALPHA remains unchanged until the challenger earns promotion from clean prospective evidence.

### Layer 5 — Brian Treasury / Portfolio Brain

Implemented in the draft branch as a `$10,000` SHADOW treasury, pending rollout and prospective validation.

The treasury is not a fixed per-trade ticket schedule. It maintains one reconciled cash pool and decides:

- cash reserve
- per-asset allocation
- concentration
- position replacement
- realized-loss acceptance when a superior opportunity exists
- liquidity-aware sizing
- entry/exit/re-entry
- opportunity-cost based capital recycling

Current hard policy boundaries:

- maximum 70% deployment
- minimum 30% cash reserve
- maximum 12% per position
- maximum eight simultaneous positions
- no deployment unless the Layer-4 EXPECTED_EDGE prospective experiment has a current `PROMOTE_CANDIDATE` decision
- gate closure or revocation fail-closes the SHADOW portfolio back to cash

A portfolio change persists its source decision, expected edge, cost and reason. Entry and exit costs are charged to the ledger. Snapshot + actions are committed atomically and append-only.

Exit gate: treasury cash + positions + realized/unrealized P&L + costs reconcile exactly and survive browser closure/restart.

### Layer 6 — Ocean Run

Implemented at code-contract level in the draft branch; actual prospective run is pending deployment and explicit start.

Ocean is a 24–48h browser-independent prospective SHADOW exam. START/STOP commands are append-only. Preflight blocks start unless Treasury, Layer-4 expected edge and core Evolution workers have healthy runtime evidence. The cloud worker records periodic Treasury/system-health checkpoints and creates a final report after planned or early termination.

Required post-run report:

- beginning/ending treasury
- realized/unrealized P&L and costs
- allocation history and replacements
- discoveries and source trust decisions
- new hypotheses
- generated code candidates
- passed/failed experiments
- promoted/rejected candidates
- capability changes
- drift/retirement events
- missed opportunities
- ALPHA decision quality during the run
- system health and degraded periods

## World Explorer policy

The explorer may discover public/open sources and provider capabilities. Discovery is not truth. A source must receive provenance, freshness, authority class, corroboration state, manipulation risk and access/licensing semantics before its data can influence decisions.

A public headline, social post, whale transfer or popularity spike is never a direct BUY/SELL command.

## Self-coding policy

Allowed candidate work:

- new collectors/adapters
- new features/sensors
- research models
- dashboard/diagnostic surfaces
- tests
- challenger policies
- data-quality and observability improvements

Protected from autonomous promotion/mutation:

- secrets and credential policy
- evidence deletion/rewrites
- security/RLS weakening
- live-money order/withdrawal surfaces
- hard authorization boundaries
- DIP runtime/code/data/control surfaces

Every code candidate must carry:

- `hypothesis_id`
- parent canonical commit
- changed paths
- test plan/results
- data windows and provenance
- replay/stress/prospective results
- contamination declaration
- promotion status

## Dashboard target

Main dashboard surfaces:

1. Overview
2. Treasury / Portfolio
3. World Intelligence
4. ALPHA
5. Evolution
6. Laboratory
7. Sources
8. System Health
9. Audit / Memory
10. Ocean Run

DIP remains a separate laboratory surface and is not part of Evolution OS control.

## Cloud independence

Starting/pausing an Ocean observation session from mobile mutates only append-only server-side Ocean command state. Collectors, research jobs, ALPHA, portfolio accounting and Evolution workers continue in cloud infrastructure when the browser is hidden or closed.

## Completion definition

Layers 0–6 are now implemented at code-contract level on the draft integration branch. This does not mean the system is deployed or prospectively proven. The draft is eligible for final rollout review only when protected-scope tests prove DIP isolation, all regression CI is green, migrations and rollout order are reviewed, and no unauthorized live-execution surface exists. Remaining sequence is final CI, migration/order review, base reconciliation if required, explicit merge/deploy approval, cloud rollout verification, then the actual 24–48h Ocean observation. The Ocean exam cannot be simulated by declaring the draft complete.
