# Expected-edge alpha challenger

This is a pure, deterministic, shadow-only compiler. It does not import
canonical ALPHA behavior, authenticate, persist, schedule, place orders, or
access protected resources. Its output always has `shadow_only: true`,
`live_execution: false`, `canonical_mutation: false`, and
`promotionReady:
false`.

At a decision timestamp, the compiler binds lagged reliability snapshots to the
exact source observation, sensor family, horizon, direction, and mature
independent group. The estimate is:

`net = sum(expectedMoveBps * reliability) - roundTripCost / fillability -
uncertainty - eventDecay - freshnessDecay`.

Spread, fee, and slippage are explicit round-trip cost components and must be
finite, non-negative, and available at or before the decision. Missing or
invalid cost fails closed. Inputs, contribution counts, numeric values, and
timestamps are bounded. Equal-time contradictory observations or reliability
snapshots contaminate the result; exact duplicates are canonicalized.

Future source, reliability, and cost evidence never enters the decision
projection and is exposed only as telemetry. Replay tests use immutable
point-in-time envelopes, while adversarial stress tests exercise bounds and
malformed data. Neither establishes prospective performance. Promotion requires
an independently run, multi-window prospective shadow A/B showing positive net
edge and improvement over control after cost, with adequate regime coverage and
no leakage. No consumer may connect this artifact to live execution.
