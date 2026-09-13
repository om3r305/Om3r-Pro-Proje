# Capability-gap alpha compiler

This candidate is a diagnostic-only compiler for capability and collector
evidence. It is pure and import-safe: `shadowOnly` is always `true`,
`liveExecution` is always `false`, and `promotionReady` is always `false`. It
does not place orders, access credentials, or promote production behavior.

## Decision semantics

The compiler evaluates only evidence at or before `decisionAt`. Later snapshots,
collector diagnostics, and failures are retained as telemetry counts and produce
explicit `FUTURE_EVIDENCE_TELEMETRY_ONLY` diagnostics; they cannot affect gaps,
conflicts, severity, or readiness. Invalid timestamps, identifiers, health
values, lease values, and non-finite metrics are malformed and fail closed.

`LEASE_SKIPPED`, `SKIPPED_LEASE`, and `LEASE_UNAVAILABLE` normalize to the
canonical `LEASE_SKIPPED` family before same-instant fingerprinting and conflict
detection. Thus equivalent contention representations cannot amplify a gap or
create a false conflict.

The `failures` field is optional. Omission means compatible empty evidence;
explicit `null` is malformed. Failure rows are bounded to 32 per capability and
256 total. Snapshots and collector diagnostics are bounded to 512 each, and
returned diagnostics are bounded to 128. Every output collection is stably
sorted.

## Evidence and gates

The candidate hypothesis is that deterministic capability-gap evidence can
expose missing, stale, degraded, failed, or lease-blocked research capabilities
without contaminating point-in-time decisions. Replay coverage checks
input-order invariance, future-evidence isolation, and lease-alias equivalence.
Adversarial stress coverage checks duplicate/conflict handling and bounded
processing.

Prospective multi-window shadow A/B evidence remains a blocker for any promotion
decision. This document does not claim review approval, measurement, promotion,
or deployment.
