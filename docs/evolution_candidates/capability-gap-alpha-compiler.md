# Capability-gap alpha compiler

This artifact is a pure, shadow-only diagnostic for point-in-time collector
evidence. It does not import canonical execution, authentication, secrets,
migrations, or control-plane resources. Every result declares
`shadow_only: true`, `live_execution: false`, and `promotionReady: false`.

Rows are runtime-validated before use. Timestamps must be UTC ISO-8601 `Z`
values with zero through three fractional digits. Invalid evidence is
provider-scoped when its provider identifier is valid and blocks a healthy
report. A non-null `freshnessAt` is validated by the same strict contract;
future completion or freshness timestamps are observationally counted and
excluded from the decision projection. Equal-provider/equal-completion conflicts
are detected across the accepted input envelope before the decision row limit is
applied.

Collector status is an explicit allowlist: `COMPLETED`, `LEASE_SKIPPED`,
`SKIPPED_LEASE`, and `LEASE_UNAVAILABLE`. Other statuses are invalid evidence.
Failure identities include provider, row, failure ID, and message, so exact
duplicates collapse while different messages remain distinct in deterministic
order.

The input envelope, decision row count, and provider diagnostics are
independently bounded. Overflow is exposed as truncation telemetry, not invalid
evidence. Future rows are partitioned before the decision budget and are
represented only by `futureTelemetry`; they cannot change classification,
freshness, failures, lease accounting, invalid counts, blockers, provider
diagnostics, or truncation state. The report always includes an explicit missing
prospective multi-window shadow A/B blocker and remains non-promotable.

Replay and adversarial stress evidence are intentionally separate from
prospective shadow A/B evidence. This artifact is not promotable until
independent prospective evidence covers multiple cadence windows.
