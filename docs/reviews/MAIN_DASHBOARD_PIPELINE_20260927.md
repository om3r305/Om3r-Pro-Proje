# Main dashboard news → multiasset shadow pipeline repair

Verified 2026-09-27 22:53 UTC (28 September 00:53 Berlin).

## Observed failures

- During 21–25 September, all 22,784 multiasset decisions waited. 18,737 carried a fresh-at-collection `OPEN` mark, but the decision engine required `REGULAR`. This was a collector/consumer contract mismatch, not evidence that no market opportunities existed. These counts are repeated decisions, not unique missed trades or missed profits.
- The engine trusted stored collection latency; a cached mark could remain apparently fresh indefinitely.
- Explicit gold/bullion headlines lacking a macro theme were not linked to gold. A single event matching several themes could inflate event strength.

## Changes

- Shared provider timestamp/session boundary assessment. Yahoo `currentTradingPeriod.regular` supplies actual start/end epochs. Missing, expired, malformed or future evidence fails closed; pre/post sessions do not qualify. No unconditional `OPEN` → `REGULAR` translation.
- Collector persists session boundaries and emits canonical session states. Engine recomputes provider age and session at decision time, recording a specific quality reason.
- Explicit gold/bullion → GOLD/GLD, silver → SILVER, and existing primary-asset links. Each event contributes once per asset; duplicate event frames do not inflate strength. Invalid/future publication times are excluded.
- Existing reaction, score, crowd, cost and bounded shadow execution requirements remain in place. No real-money execution or Telegram messages. No DIP changes.

## Verification

`node --test tests/main-multiasset-pipeline.test.cjs`: 13/13 passed. Includes mocked collector and authenticated engine invocation, stale cached marks, actual regular boundaries, duplicate headlines, gold-reserve linkage, and existing treasury planner contracts for bounded entries/costs/exits with the experimental promotion gate closed.

Deployed Edge Functions:

- REALTIME `brian-realtime-multiasset-market-eye`: deployment version 9; application `v5-verified-session`.
- CORE `brian-multiasset-opportunity-engine`: deployment version 5; application `v4-verified-session`.

Authenticated live collector verification returned HTTP 200 / SUCCESS, 32 returned marks and no degraded sources. At 22:53 UTC, the latest view contained 5 new-version REGULAR commodity marks and 23 new-version CLOSED marks. Four FX latest entries still selected earlier marks; they fail closed as SESSION_UNVERIFIED until newer provider marks replace them.

The live engine wrote 32 decisions using the new version. GOLD, SILVER, WTI, BRENT and NATGAS now pass the session gate. All five correctly waited on `REACTION_NOT_CONFIRMED` in this verification cycle. Closed exchanges remained blocked. No artificial opportunities or historical trades were inserted.

## Remaining limits

- This establishes that the session gate no longer blocks all instruments. It does not establish a profitable strategy or a completed live shadow trade from the new version.
- Last-24-hour news measurement: 5,060 frames with publication time no later than observation; median publication-to-observation delay 11 minutes. Frames are not unique stories. This is not an ultra-fast breaking-news feed.
- Public unofficial prices remain shadow-only and can be delayed. Publication freshness, source quality, causal asset mapping and prospective net-of-cost outcomes still need evaluation. Broad theme links alone do not establish causation.
- Canonical ALPHA micro-entry remains paused in BIG_MOVE validation; EXPECTED_EDGE promotion remains closed. The separate existing bounded multiasset shadow lane is enabled and was not made dependent on bypassing those gates.
