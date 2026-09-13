# Capability-gap ALPHA compiler

This candidate is a pure, diagnostic-only compiler for capability-gap evidence.
It does not create actions, alter canonical policy, place orders, access
execution surfaces, or imply promotion. Every result is `shadow_only: true`,
`live_execution: false`, and `promotionReady: false`.

Evidence is evaluated at a caller-supplied `decisionAt`. Evidence after that
instant is retained only in bounded telemetry and cannot enter accepted
evidence, deduplication, or conflict analysis. `LEASE_SKIPPED`, `SKIPPED_LEASE`,
and `LEASE_UNAVAILABLE` are the canonical `LEASE_UNAVAILABLE` family before
identity and conflict analysis. An omitted `failures` field is an empty
compatible set; explicit `failures: null` is malformed and fails closed.

The compiler sorts before grouping, deduplicates equivalent observations,
reports conflicting same-instant outcomes as diagnostics, bounds collections,
and bounds diagnostic text. Replay and stress coverage is separate from the
focused unit tests. This candidate is isolated from protected control-plane,
authentication, migration, and execution surfaces. Independent review, guarded
preview, exact measurement, and human approval remain required; deployment and
live execution are not part of this candidate.
