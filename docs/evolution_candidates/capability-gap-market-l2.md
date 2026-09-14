# Capability-gap market L2 compiler

This is a pure, bounded, shadow-only research artifact. It accepts an immutable
evidence envelope and derives a synchronized depth book and fillability-aware
cost projection without network, storage, authentication, execution, scheduling,
rebalancing, or control-plane access. It never reads or mutates DIP resources.
Every report has `shadow_only: true`, `live_execution: false`, and
`promotionReady: false`.

## Source-truth contract

Each row identifies a venue, symbol, collector session, connection generation,
sync generation, arrival sequence, source event ID, observed and received UTC
timestamps, non-empty source lineage, and exact decimal-string levels. Snapshots
provide `lastUpdateId`; diffs provide an inclusive update range. Rows must form
one contiguous arrival stream. A snapshot is required before diffs, and a gap,
crossed book, conflicting arrival, malformed level, mixed lineage, stale
evidence, or missing source truth fails closed. Future rows are excluded from
the decision projection and counted only in telemetry.

The compiler reconstructs the book in arrival order, applies overlapping diffs
using the sequence boundary, and derives best prices, spread, visible depth,
depth-walk VWAP/slippage, requested-notional fillability, and round-trip cost.
Insufficient visible depth is never converted into a healthy or fillable result.
Input rows and levels are bounded; truncation is observable and blocks a healthy
classification. The result retains source event IDs, arrival sequence, snapshot
lineage, and cadence metadata so raw capture and derived state remain separate.

## Evidence boundaries

Deterministic unit tests are regression/type-contract evidence. Immutable replay
tests append and reorder future rows to prove that decision-time features do not
change. Adversarial stress tests exercise bounded oversized envelopes and
pathological inputs. These are separate from prospective shadow A/B evidence:
replay and stress results cannot establish cadence health, net-edge impact,
multi-regime stability, or promotion readiness. A later reviewed integration
must collect post-decision labels and maintain dedicated challenger lineage
before any promotion assessment.
