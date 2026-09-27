# DIP risk guard — 2026-09-27

## Observed baseline
The 18-hour session expired 2026-09-23 06:11 UTC. Main remained disabled at 996.245302463026 USDT (13 closes; 5 wins, 8 losses). Arena was incorrectly marked enabled/RUNNING with an expired window and 969.433984893621 USDT (25 closes; 14 wins, 11 losses). No open positions. All shadow_only=true, live_execution=false.

Arena wins total 36.2591; thesis-break losses total 66.8251. The deployed policy allowed 85–99.5% single-symbol exposure. A profitable edge is NOT established by these observations.

## Source provenance
The branch's worker source was older than production. Restore the actually deployed worker from fc963cd21846cd8b2dfe7edfc109fa4dfa572c97 before patching. This preserves the existing EARLY_SCOUT disable and V8.9.8 behavior. Guardian production source was pinned to 054e910c2f1fc25cafdf1d807a6f26ed08ae2c60; restore that source too. Status starts from deployed version 23.

## Changes
- Arena regular and monster entries share a sizing cap: 10% notional, 0.5% modeled stop risk including estimated costs, 30% aggregate notional. These are conservative controls, not optimized trading parameters; gap losses can exceed modeled risk.
- Arena session drawdown of 2.5% blocks new entries; existing 3.06% loss remains visible and frozen. Main cannot use emergency entries or scaling to bypass its 2.5% drawdown freeze.
- Armed trailing floors cannot be vetoed by a bullish forecast; fills still use current/worse price with modeled slippage and fees. Rising runners above the floor remain open.
- Expiry blocks buys/scaling but continues managing open positions; empty completed sessions stop both engines and retain prior scan evidence instead of overwriting it with a fake fresh heartbeat.
- Keep the existing 3-minute full scan. Confirmation memory 240s and forecast freshness 240s tolerate a normal cycle; 450s UI/API stale threshold prevents false outages. A missed confirmation cycle resets evidence.
- Restore conditional per-minute guardian dispatch; no-position calls exit immediately. Active bursts check every 5s for 35s after a 15s offset; this is not continuous 5s coverage. Increase guardian lease to 45s to cover market retries and both books; releases still occur after each tick.
- Preserve dispatch ACL: PUBLIC/anon/authenticated cannot execute it. No new credentials, external messages, or real orders.

## Verification
`node --test tests/dip-risk-guard.test.cjs`: 12 behavioral tests pass. Tests exercise actual worker/guardian functions with mocked market/database I/O, including ultra-monster sizing, regular sizing, drawdown freeze, expiry, strong-forecast floor breaches, above-floor continuation, and status cadence. UI JavaScript syntax checks pass.

## Rollback
Worker previous version 38: pinned fc963cd21846cd8b2dfe7edfc109fa4dfa572c97. Guardian previous version 11: pinned 054e910c2f1fc25cafdf1d807a6f26ed08ae2c60. Previous status source is available in brian-2026 parent. Disable the guardian job by its name if required. Do not erase events, reset balances, or enable real execution during rollback.

Deployment and fresh shadow-session observation are recorded after verification. No future profitability claim is made.

## Live verification and six-hour observation
- Code commit: 66f2a05e3a2d4672e05093675f20f1bbedd40f81; Vercel dpl_H7pDnNmFXWq3WiXUkjuTf9jwM8j1 READY on the requested brian-2026 alias.
- Worker version 39, Guardian version 12, status version 24 deployed. Main probe request 160802: HTTP 200 RUNNING, no market errors; 105 universe symbols and 32 deep scans.
- Guardian probe 160804: HTTP 200 COMPLETE, 8 successful idle ticks over 35.354s, no orders. Conditional cron 93 active every minute; first observed no-position run took ~8ms. Dispatch remains denied to anon/authenticated; security-advisor finding counts unchanged.
- Same session/cash/history resumed at 2026-09-27 07:38 UTC for 18 hours, until 2026-09-28 01:38 UTC (03:38 Berlin). Main 996.245302463026 USDT; arena 969.433984893621 USDT remains RISK_FROZEN under its session loss limit.
- At 13:38 UTC the automatic heartbeat was current with no market errors. Radar history contained 120 automatic scans from 07:41 through 13:38; adding the initial probe gives 121 evaluation minutes. 3,872 evaluations, 111 distinct symbols, 0 READY decisions and 0 new trades. 3,326 evaluations (85.90%) failed economics; median modeled net forecast -18.37 bps. These are model classifications, not proof that no profitable market opportunities occurred.
- 11 confirmation messages comprised 9 EARLY_SCOUT candidates in an already-disabled strategy and 2 isolated core/winner confirmations. Disabled scout messages now say WAIT_SCOUT_DISABLED rather than implying an imminent entry. No gates were loosened to force trades.
- Follow-up found the worker's legacy triggerFill could book stale stop/trail prices after a gap. Clamp exits to the worse of observed bid and trigger, matching guardian semantics. A regression test covers stop, ratchet, harvest and protective trail gaps. New worker policy .2; 13 behavioral tests pass. Earlier historical P&L is retained, not retrospectively rewritten.
- This observation establishes scheduler continuity, not profitable edge or live-money readiness. Full authenticated visual browser verification was not performed; frontend deployment and backend execution were verified.
