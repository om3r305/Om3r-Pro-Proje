# Cost-control expected-edge challenger

This artifact is a bounded, shadow-only compiler for ranking opportunities by
decision-time net edge. It does not modify canonical ALPHA behavior and has no
execution, persistence, authentication, order, or withdrawal surface.

For each opportunity, the projection is:

`weighted gross edge = gross edge * mature lagged reliability`

`round-trip cost = 2 * (spread + fee + depth/slippage) / fillability`

`net edge = weighted gross edge - round-trip cost`

The input cost snapshot must declare `costConvention:
"ONE_WAY_COMPONENTS_BPS"`.
Its spread, fee, and depth/slippage values are one-way basis-point components.
The report exposes each normalized component as round-trip basis points
(`2 * component / fillability`), and their sum is exactly `roundTripCostBps`;
all edge and margin calculations use that same round-trip value. Fillability is
a bounded fraction in `(0, 1]`; it increases the estimated cost rather than
creating an unbounded value. An opportunity is eligible only when net edge and
the explicit `net edge / round-trip cost` margin are both strictly above their
configured thresholds. Results are ranked by net edge, then stable opportunity
identity.

Only observations and cost snapshots at or before `decisionAt` are decision
features. Reliability must be mature, positive, bound to an observed
opportunity, and from a lagged independent group. Reliability is accepted only
when its provenance binds to an exact `sourceObservationId` in the same
decision-time envelope. The binding verifies the source and lineage IDs,
opportunity, raw independent group, sensor family, sensor horizon, direction,
snapshot window end, and snapshot generation time. `independent: true` is
diagnostic metadata and never establishes independence by itself. Raw micro
groups are canonicalized through the ALPHA `intrabar_tape` mapping before
independent-group counting.

Evidence is checked for conflicts at `opportunity/canonical-group/snapshot`
before provenance deduplication. Conflicting reliability, source, lineage, or
group bindings contaminate the report rather than allowing the highest-valued
variant to win. A single opportunity is emitted at most once; when several
verified groups support it, its reliability is the deterministic mean of the
selected group snapshots and the report retains the selected canonical groups
and source observation IDs.

Future rows are filtered before bounded selection, counted as telemetry, and
cannot consume decision capacity. Missing, stale, contradictory, negative,
non-finite, zero, or unverifiable evidence fails closed. Inputs are
canonicalized before bounded selection so equivalent permutations produce the
same report. `grossEdgeBps` is the sole opportunity gross-edge field.

Replay and adversarial stress tests are separate evidence classes. Neither
establishes profitability or prospective performance. Before any promotion
consideration, an external workflow must run a prospective shadow A/B against
the control using contemporaneous cost snapshots and separately assess positive
after-cost net edge, lower turnover when cost dominates, missed-opportunity
profitability exclusions, sample/regime coverage, data quality, leakage, and
stability across multiple windows. Shadow, replay, and stress results must not
be represented as live or production execution evidence.
