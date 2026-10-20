# BRIAN UI acceptance — 20 October 2026

Existing product extended in place. Backend/research belongs to Claude; frontend belongs to Codex. The operator authorized tested production releases, but explicitly requires a separate approval before starting any new experiment. No experiment start, scheduler change, treasury reset, or historical-data deletion is part of this patch.

## Implemented

- Existing rooms and listeners preserved; a navigation bar opens the same meeting, accounting, engineering, and anatomy functions. Anchors reveal collapsed ancestors before focusing a destination.
- Wide desktop has adjacent news, neural command, treasury/status, rooms, and performance. Mobile retains readable touch controls and horizontally scrollable navigation. Layout inspected at 390 × 844 and 2560 × 1440. A requested 5120 × 1440 browser override was capped by the browser to 4096 × 1440; no exact-5120 screenshot verification is claimed.
- All meeting entry points receive the real event-derived red/amber beacon (fresh critical within 2 hours, high within 1 hour); expired, future, and malformed timestamps cannot trigger it. Previously only the first matching shortcut was updated.
- Accounting/engineering modals now participate in mobile overlay locking. Existing mobile `overflow:auto!important` no longer overrides the lock. Background scroll is restored after closing.
- Missing system status no longer renders “system running” or enables control buttons. Existing known on/off behavior is preserved.
- Unlocking immediately refreshes an initially locked scorecard; timeout errors offer a readable retry. No test data is substituted for live reports.
- Engineering refreshes coalesce in-flight requests; periodic expensive requests occur only for a visible panel/open room. Public position-mark requests have a 10-second deadline.
- Lab readiness refreshes for every anatomy entry point and after unlock, rather than retaining an unauthorized startup error. Null percentages stay unknown and server-marked stale reports are visibly old.
- News items with an HTTP(S) provenance URL expose a source link. Missing URLs remain missing; script schemes and credential-bearing URLs are rejected.
- Frontier CI exposed three pre-existing explicit-any lint errors in the news endpoint. Query rows now have explicit `Frame`/`ScoutFrame` types; this is type-only and does not change the emitted JavaScript, queries, or permissions. Deno check and lint pass.
- Prediction/NIULAI4 read-only cards consume `brian-prediction-status.v1` from Claude's documented contract. Sections show independent server statuses, null values remain unknown, stale last-success data is labelled, HTTP 401 clears prior data, unsafe/incompatible response contracts fail closed. INTERIM reports never receive an advantage verdict. There is no ingest/start/key-management UI. Polling is at most once per 60 seconds while visible, except explicit refresh/unlock.

## Verification

- Node: interaction regressions, prediction presentation model, scorecard page isolation/auth retry, neural command, council, accounting, engineering, reliability, and lab readiness suites pass (63 tests, including the follow-up server-STALE and accounting-coverage contracts).
- Python: unified command, meeting, and dashboard mobile contracts pass (18 tests).
- Local browser: meeting open/close and evidence/update/result tabs; accounting open/close and ledger/position tabs; engineering open/close and tests/history/sources tabs; anatomy open/close; new navigation and research auth-required state; mobile background overflow hidden while modal open and auto after close.
- Local layout preview deliberately had no authentication and was labelled as a layout test. It was not used as proof of live data delivery, and its temporary HTML is not shipped.

## Live observations before release — backend follow-up required

Authenticated production session at approximately 09:32–09:36 Berlin: heartbeat/news/treasury summary data were observed; meeting and engineering rooms opened. Scorecard returned `signal timed out`; accounting remained loading during the observation; engineering showed intermittent `BRIAN_API_DIRECT_TIMEOUT`, with historical BLOCKED work correctly labelled as history. The backend owner was informed. Edge-function ACTIVE status alone is not acceptance.

No start/restart/stop, treasury amount mutation, or experiment-start button was clicked on production. Their contracts are tested separately. Full clean live-data acceptance remains dependent on backend responses and post-release verification; “all buttons and all data are working” must not be inferred from the UI regression results.

## Follow-up for backend PR #167

The engineering endpoint can now return HTTP 200 with `status: STALE` / `stale: true`. Both the console and room model treat that as last-known evidence, never live progress or animation, even if its timestamp is recent. Visible engineering polling is 60 seconds to match the 45-second shared backend cache.

## Live accounting follow-up

After backend read-load changes, authenticated accounting delivered the real history and ledger. Its payload caps are 300 actions and 360 snapshots, not the old UI's hardcoded 1,000. Labels now use `coverage.action_limit` and `coverage.history_limit`. The server's stale flag overrides a recent snapshot timestamp; missing initial data does not claim zero transactions/positions. No ledger rows or capital were changed.
