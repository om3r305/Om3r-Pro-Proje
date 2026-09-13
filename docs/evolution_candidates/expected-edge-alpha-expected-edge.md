# Expected-edge alpha challenger

This is a pure, deterministic, shadow-only compiler. It does not import
canonical ALPHA behavior, authenticate, persist, schedule, place orders, or
access protected resources. Its output always has `shadow_only: true`,
`live_execution: false`, `canonical_mutation: false`, and
`promotionReady:
false`.

At a decision timestamp, the compiler binds lagged reliability snapshots to the
exact source observation, sensor family, horizon, direction, and mature
independent group. Horizons are explicit durations such as `300s`, `900s`,
`3600s`, or `24h`, bounded to one day. Each source declares its cadence and an
evaluation interval whose length exactly equals its horizon. A 24-hour
observation therefore retains a 24-hour evaluation window and is never reduced
to a five-minute label bucket.

Reliability, event, and cost evidence carries point-in-time provenance and
cadence contracts. Freshness is bounded from the declared cadence and horizon
plus a fixed operational allowance, rather than an arbitrary five-minute cap.
Missing, stale, malformed, contradictory, or out-of-bounds decision-time
evidence fails closed. Incomplete bounded envelopes cannot be eligible.

The estimate is:

`net = sum(expectedMoveBps * reliability) - roundTripCost / fillability -
uncertainty - eventDecay - freshnessDecay`.

Spread, fee, and slippage are explicit round-trip cost components and must be
finite, non-negative, and available at or before the decision. Inputs,
contribution counts, numeric values, and timestamps are bounded. Numeric
decision timestamps are integral epoch milliseconds and normalize to the same
canonical UTC instant as equivalent timestamp strings. Fillability must also
clear the safe lower bound implied by the bounded estimated round-trip cost;
otherwise cost arithmetic fails closed rather than producing an unbounded
estimate. Equal-time contradictory observations or reliability snapshots
contaminate the result; exact duplicates are canonicalized. Future source,
reliability, event, and cost evidence never enters the decision projection and
is exposed only as telemetry.

Replay compares the compiler with an independently hand-specified immutable
projection. Adversarial stress evidence separately covers malformed and missing
sources, provider and contribution bounds, cost and uncertainty extremes,
heterogeneous horizons, input-order invariance, duplicates, and future
contamination. Neither replay nor stress establishes prospective performance.
Promotion requires an independently run, multi-window prospective shadow A/B
showing positive net edge and improvement over control after cost, with adequate
regime coverage and no leakage. No consumer may connect this artifact to live
execution.
