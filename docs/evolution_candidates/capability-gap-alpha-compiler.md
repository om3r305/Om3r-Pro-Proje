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
are detected using the parsed completion instant across the accepted input
envelope before the decision row limit is applied. Equivalent UTC spellings
collapse deterministically; contradictory rows use a complete classification
fingerprint containing row identity, health, canonical freshness, normalized
status, and sorted failure identities including messages, and block independent
of input order.

Collector status is an explicit allowlist: `COMPLETED`, `LEASE_SKIPPED`,
`SKIPPED_LEASE`, and `LEASE_UNAVAILABLE`. Other statuses are invalid evidence.
Failure identities include provider, row, failure ID, and message, so exact
duplicates collapse while different messages remain distinct in deterministic
order.

The input envelope, decision row count, and provider diagnostics are
independently bounded. Overflow is exposed as truncation telemetry, not invalid
evidence. The complete recognized provider set is determined from the accepted
hard input envelope after runtime validation and future-row exclusion, before
`maxRows` slices the retained decision rows; valid providers beyond the row
limit still participate in deterministic provider-limit accounting, while
well-formed future-only providers remain telemetry-only. Future rows are
partitioned before the decision budget and are represented only by
`futureTelemetry`; malformed future rows remain invalid evidence, while
well-formed future rows cannot change classification, freshness, failures, lease
accounting, invalid counts, blockers, provider diagnostics, or truncation state.
The report always includes an explicit missing prospective multi-window shadow
A/B blocker and remains non-promotable.

Replay and adversarial stress evidence are intentionally separate from
prospective shadow A/B evidence. This artifact is not promotable until
independent prospective evidence covers multiple cadence windows.

Option limits accept only finite positive integers; non-finite, zero, negative,
and fractional values use bounded defaults. `maxRows` selects retained decision
rows, while `maxProviders` independently bounds diagnostics over all recognized
decision-time providers in the inspected envelope. Invalid-only providers may
receive `UNKNOWN` diagnostics and provider-scoped invalid counts; provider
overflow is truncation telemetry, never invalid evidence.

A decision-time completion with a future `freshnessAt` is excluded from provider
recognition and classification and appears only in future telemetry. A malformed
non-null freshness value is invalid evidence, including when the completion
itself is future-dated. The fixed input envelope is an observability boundary:
rows beyond it are not classified, and envelope truncation produces explicit
fail-closed blockers rather than an assertion about uninspected data.

The explicit missing prospective multi-window shadow A/B blocker is intentional.
Current diagnostics and replay/stress evidence do not constitute promotion
evidence; `promotionReady` remains false until that separate shadow A/B gate is
supplied.
