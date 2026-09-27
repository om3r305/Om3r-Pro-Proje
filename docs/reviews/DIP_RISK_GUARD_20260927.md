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
