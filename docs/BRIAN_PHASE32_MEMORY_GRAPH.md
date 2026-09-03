# Brian Phase 3.2 — Causal Intelligence Memory + Entity Graph

Status: SHADOW_RESEARCH_ONLY

This increment implements two planned intelligence-moat components without adding any exchange execution.

## Source Reputation Memory

A source can earn or lose reputation only after an event's evaluation horizon has resolved. Brian records source/event-type outcomes and tracks:

- resolved sample count
- directional sample count and accuracy
- truth-confirmation rate
- confirmed manipulation rate
- conservative reliability score
- sample confidence

A single lucky observation receives very little confidence. Queries use an explicit `as_of` timestamp, so outcomes resolved later cannot improve a source's historical reputation.

## Opportunity Experience Memory

Event contexts can accumulate empirical net return, win rate, adverse excursion and a conservative edge estimate. Outcomes cannot be learned before their resolution timestamp, and queries only see prior resolved outcomes.

This memory is evidence for future research prioritization; it is not a live promotion or execution rule.

## Entity / Wallet Graph

Wallet labels carry:

- address
- entity id
- entity role
- provider
- trust class
- confidence
- first-observed timestamp
- historical point-in-time verification status

Resolution is strictly `as_of` time. A label learned tomorrow cannot rewrite yesterday's whale interpretation.

High-authority conflicting labels resolve to `unknown` rather than false certainty. User-generated labels cannot override a stronger provider-verified label.

## Transfer semantics

The graph conservatively distinguishes:

- internal transfer when both addresses resolve to the same entity
- exchange deposit
- exchange withdrawal
- unresolved transfer

A DEX interaction is not called BUY or SELL unless the actual swap-leg semantics are available. Exchange deposit is likewise not automatically called a sale; it is only a directional-risk context.

## Invariants

- No future outcome learning.
- No hindsight wallet labels.
- No forced resolution of conflicting high-authority labels.
- No DEX transfer -> BUY/SELL shortcut.
- No source with tiny sample count becomes high-confidence merely from luck.
- No live exchange execution.
