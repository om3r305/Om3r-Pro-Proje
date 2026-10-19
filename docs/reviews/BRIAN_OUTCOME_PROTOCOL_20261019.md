# ALPHA outcome coverage — prospective protocol (19 October 2026)

Ownership: this is product measurement. The other conversation owns the engineering queue and its schedule. No engineering control, queue, worker, timer or CODE provider changes belong to this package.

## Frozen prospective window

- Protocol: alpha-coverage-20261019-1300z-v1.
- Decisions observed from 2026-10-19 13:00:00 UTC inclusive to 14:00:00 UTC exclusive (Berlin 15:00–16:00).
- Every shadow-only/non-live ALPHA decision is eligible, including WAIT/VETO. Do not select on future outcome or recommendation success.
- Deterministic order observed_at, decision_id; first 1000 if capacity is exceeded, with explicit partial coverage flag.
- Horizons: 300, 900, 3600 seconds. Terminal tolerance remains target through target+120 seconds; no pre-target substitute.
- Last terminal window ends before/equal 2026-10-19 15:02 UTC (Berlin 17:02).
- Include late-arriving records only once actually available by the report snapshot. Observed and persisted/captured times must both be <= snapshot.
- Report compiler versions and their counts. Mixed versions are not a frozen-strategy experiment.
- Read-only repeated reports on this same fixed window. No reset of history, new strategy or automatic promotion.

## Acceptance and interpretation

Report sampled / mature / not mature / resolved / missing entry / invalid outcome / missing terminal / terminal present awaiting auditor / terminal window still open, by asset and horizon. The exclusive categories must conserve their totals. Missing cost is a subset of resolved directional decisions and is not double-counted. Cost null, malformed or negative is unknown, never zero. A valid outcome must match the immutable entry reference, have valid positive prices, finite directional return and a terminal timestamp inside the original tolerance.

A separate latest-300 / last-six-hours diagnostic remains explicitly nonrandom and rolling. It is not compared with another shifting sample as proof of improvement. Neither report proves profitable trading; overlapping decisions are not independent. This package reports observation gaps but does not collect prices or repair absent historical ticks. Independent price observation remains outstanding.

## Implementation and verification

Coverage SELECT runs within the existing authenticated shadow-scorecard repeatable-read/read-only transaction, with an eight-second statement timeout and bounded decision seeds. Snapshot timestamp is the transaction clock; no client-selected date window is accepted. Failure is isolated with a savepoint and reported UNAVAILABLE, preserving the existing scorecard without inventing empty successful coverage.

The main dashboard renders the new coverage module before existing scorecard content. DIP pages do not render the product coverage panel. Existing scorecard/dip calculations and policies are unchanged.

Local Node verification: 8 new UI/data-consistency tests and 9 unchanged scorecard tests passed. This environment has no pytest/PostgreSQL runtime; real PostgreSQL query tests are added to existing hosted PostgreSQL CI and must pass before release. They execute the actual production SELECT against isolated fixture tables, including all three price paths, missing categories, time/mode filtering, invalid receipts/costs, fixed boundaries and capacity truncation.

Live read-only query baseline at 2026-10-19 12:05:17 UTC:

| Horizon | Mature | Resolved | Missing terminal | Awaiting auditor | Window open |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5m | 284 | 149 | 120 | 15 | 0 |
| 15m | 260 | 135 | 109 | 16 | 0 |
| 60m | 156 | 85 | 53 | 15 | 3 |

Each horizon has 300 sampled decisions. No missing entry, invalid outcomes or resolved directional missing cost found in this diagnostic snapshot. This is not an estimate of the entire period, source completeness, realized P&L or causal impact. Prospective cohort has not started at baseline time. Authentication and browser end-to-end verification must be stated separately from query/renderer tests.
