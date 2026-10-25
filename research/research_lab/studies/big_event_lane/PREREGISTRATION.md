# Pre-registration — "big event" hold-to-close lane, prospective shadow tournament

- Study id: `big-event-lane-v1` · registered 2026-10-25 (Sunday, markets closed), before any
  evaluation-period data exists. Operator request 2026-10-24: important news is not abandoned after
  one hour; enter fast, exit slow.
- Shadow only and measure-only: nothing here sizes or opens treasury positions. A candidate can join
  the treasury only through a separate, later change after it passes the decision rule below.
- Data source: `public.brian_treasury_horizon_outcomes` (migration 20261025075229), horizon
  `SESSION_CLOSE`, lane `MULTIASSET_EVENT_REACTION`, `excluded_reason is null`. Every input there is
  point-in-time (breadth looks back 30 min only; roll artefacts excluded by a fixed rule).

## Honesty note (what was seen before freezing)

On 2026-10-19..23 (5 trading days, the only data that existed) the first signal per asset-day held to
session close gave, on $2,000 tickets: theme-match + breadth >= 3 −$261, theme-match only +$313, no
match −$228; Friday 23 Oct energy at the US open +$127. Candidates B and C below were therefore shaped
by data already seen and are marked so. **None of that window counts for the decision.**

## Common rules (frozen)

- Unit: one trade per asset per UTC day per candidate = the first qualifying signal of that day.
- Entry at the signal's observed reference price; exit at the last REGULAR observation of the same
  UTC day; return net of the signal's own `cost_bps` (round trip). No re-entry the same day.
- Evaluation period starts 2026-10-26 00:00 UTC.

## Candidates (frozen)

| id | rule | origin |
|---|---|---|
| A `MATCH_BREADTH3` | `theme_sector_match` and `breadth >= 3` | designed 2026-10-24 before data |
| B `MATCH_ONLY` | `theme_sector_match` (any breadth) | **data-suggested** |
| C `ENERGY_US_OPEN` | sector `ENERGY`, `theme_sector_match`, `breadth >= 3`, signal at or after 13:30 UTC | **data-suggested** (Friday 23 Oct) |
| D `ALL_FIRST` | every first signal per asset-day (control) | baseline |

## Decision (frozen)

- Evaluated after 20 trading days in the evaluation period (about 2026-11-20), or on 2026-12-31,
  whichever is first; at least 40 trades for the candidate, else `INSUFFICIENT_DATA`.
- Statistic: mean net bps per trade; 95 % interval from 2,000 bootstrap resamples **clustered by
  trading day** (signals on one day are not independent).
- PASS for a candidate only if (1) the interval's lower bound is > 0 after a Bonferroni correction
  for the three non-baseline candidates (i.e. a 98.33 % two-sided interval), and (2) its mean beats
  candidate D's mean on the same days.
- Otherwise `FAIL` (or `INSUFFICIENT_DATA`). A failed candidate is not re-tuned and re-run on the
  same data; a new idea needs a new registration and new data.
- On PASS the next step is a proposal to the operator, not an automatic switch.

## Evaluation query (frozen)

```sql
with o as (
  select *, (observed_at at time zone 'UTC')::date d
  from public.brian_treasury_horizon_outcomes
  where horizon = 'SESSION_CLOSE' and lane = 'MULTIASSET_EVENT_REACTION' and excluded_reason is null
    and observed_at >= '2026-10-26 00:00+00'),
c as (
  select 'A' k, * from o where theme_sector_match and breadth >= 3
  union all select 'B', * from o where theme_sector_match
  union all select 'C', * from o where sector = 'ENERGY' and theme_sector_match and breadth >= 3
                                  and (observed_at at time zone 'UTC')::time >= '13:30'
  union all select 'D', * from o),
f as (select distinct on (k, asset_id, d) * from c order by k, asset_id, d, observed_at)
select k, d, gross_bps - cost_bps net_bps from f;   -- bootstrap by day offline
```
