# World evidence and prospective cost-adjusted validation

This release adds official document-body receipts, a bounded stored-headline counter-evidence search, BTC/ETH market-reaction probes, and a prospectively frozen ALPHA outcome evaluation. It changes no execution/risk settings, DIP engine, or engineering queue/worker/timer.

## Preregistered measurement

Protocol `alpha-net-probe-20261019-v1`: starts 2026-10-19 16:00 UTC; ends 2026-11-18 16:00 UTC; final terminal tolerance closes 16:17 UTC. First decision per crypto USDT asset in the first 60 seconds of fixed 1080-second slots anchored at the start; WAIT/VETO remain in the denominator. Horizon 900 seconds; terminal tolerance 120 seconds. No same-asset overlapping outcome windows. Different assets remain correlated; UTC daily blocks are reported, not assumed independent trades. Minimum 100 cost-known directional observations and 20 day blocks are review conditions, not a proof threshold. Directional decisions recorded more than 60 seconds late or more than five seconds ahead of database time are invalid for net measurement, and remain explicit in the denominator. Mixed/unknown policy versions, missing directional outcomes/costs, truncated bounded data and unfinished periods are explicit issues. Cash benchmark 0 bps. Prospective dates are server constants; requests cannot change them. Diagnostic last-six-hour measurements are labeled separately. No automatic promotion.

Net uses the recorded auditor outcome's estimated round-trip cost, never zero for unknown costs. These are modeled decision returns, not executed fills, actual slippage, realizable portfolio profit, or a controlled estimate of news contribution. Positive means remain `POSITIVE_SAMPLE_NOT_PROOF`; the system never manufactures a statistical/profitability conclusion. Independent replication and execution-cost evidence remain required.

## Evidence semantics

`READ` means a bounded complete HTTP response was received from a registered official hostname and meaningful document text extracted by the pinned real DOM parser. HTTPS, same-approved-host redirects only, fixed request timeout, content-type/one-megabyte limits, response SHA-256, received/assessment/persistence times. Scripts/navigation stripped. SEC index pages remain index-only unless a same-host filing document was actually read. HTTP/provider failures persist as failures. External text never becomes agent instructions.

Exact normalized claim-text presence does not establish truth or independent verification. Source registration is an origin prior, not calibrated accuracy. All `claim_verified_independently` values remain false. Lexical English opposite-direction/revision candidates from at most 100 relevant stored headlines are explicitly provisional: contradiction and independence are not verified, and no match is not confirmation. Search is not the whole web.

BTC/ETH probes measure available before/after publication/first-seen windows with strict point availability and two-minute tolerance. They are broad market proxies, not event-specific asset mapping or causal attribution. No priced-in probability or optimal trading horizon is invented. Late document reviews do not become evidence available to earlier decisions. No direct ALPHA influence or trading readiness is enabled.

The native existing data orchestrator calls this worker every five minutes, bounded to two events, with lease/cadence/internal authentication/operator-stop checks. Append-only service-only storage with RLS; no anonymous/authenticated read or write grants. Dashboard authentication remains required. Failures stay visible instead of producing empty success.
