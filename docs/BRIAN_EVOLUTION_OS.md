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
- automated test plan generation
- replay/stress/prospective challenger contracts
- promotion council evidence bundle
- failure memory and concept-drift retirement

Exit gate: Brian can identify a measurable weakness, propose a bounded change, generate a candidate artifact, and produce an auditable accept/reject result without mutating canonical behavior.

### Layer 4 — ALPHA Intelligence Upgrade

- prospective reliability feedback into decision weights
- expected gross move model
- explicit cost model
- uncertainty/decay penalties
- expected net edge
- opportunity ranking
- decision explanations based on evidence lineage

Canonical decision target:

`expected_net_edge = expected_gross_move - estimated_round_trip_cost - uncertainty_penalty - event_decay_penalty`

Evidence score alone must never be interpreted as expected return.

Exit gate: every actionable ALPHA decision has an expected-edge decomposition and sufficient prospective reliability evidence.

### Layer 5 — Brian Treasury / Portfolio Brain

Brian receives a real shadow treasury (planned Ocean baseline: `$10,000`). The treasury is not a fixed per-trade ticket schedule.

Brian decides:

- cash reserve
- per-asset allocation
- concentration
- position replacement
- realized-loss acceptance when a superior opportunity exists
- correlation/exposure
- liquidity-aware sizing
- entry/exit/re-entry
- opportunity-cost based capital recycling

A portfolio change must persist the reason and competing alternatives considered.

Exit gate: treasury cash + positions + realized/unrealized P&L + costs reconcile exactly and survive browser closure/restart.

### Layer 6 — Ocean Run

24–48h cloud-only prospective SHADOW run with the full stack enabled.

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
- ALPHA decision quality before/after Evolution candidates
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

Main dashboard sections:

1. Overview
2. Treasury / Portfolio
3. World Intelligence
4. ALPHA
5. Evolution
6. Laboratory
7. Sources
8. System Health
9. Audit / Memory

DIP remains a separate laboratory surface and is not part of Evolution OS control.

## Cloud independence

Starting/pausing an Ocean observation session from mobile must only mutate server-side session state. Collectors, research jobs, ALPHA, portfolio accounting and Evolution workers continue in cloud infrastructure when the browser is hidden or closed.

## Completion definition

The draft PR is ready to merge only when all six layers are implemented, the protected-scope tests prove DIP isolation, CI is green, Ocean mode remains shadow-only, and the final diff has no unauthorized live-execution surface. Merge/deploy remains an explicit user action after final review.