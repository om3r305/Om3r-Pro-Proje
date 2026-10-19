# Brian evidence release — 19 October 2026

The evidence and measurement infrastructure is deployed. The four research goals are **not** all scientifically proven. No real-money execution or automatic promotion is enabled.

## Published behavior

- Official document-body read receipts with exact response SHA-256, source URI, response bytes and separate receive/assessment/persistence clocks; meaningful text extracted by the pinned DOM parser. Source origin and claim-text match remain separate from independent truth verification.
- Bounded counter-evidence candidate search over at most 100 relevant stored English headlines. Candidate contradiction/independence is unverified; no matches do not confirm the claim.
- BTC/ETH broad-market publication/first-seen price-window measurements with strict two-minute terminal tolerances and point availability. No event-specific causal attribution, optimal horizon or priced-in probability is claimed.
- Fixed prospective net-outcome protocol and rolling six-hour diagnostic. Unknown costs remain unknown; late directional decisions and unknown policy versions remain explicit. No positive sample becomes profitability proof.
- Native existing data orchestrator, every five minutes, at most two events per run, internal authentication, collector lease, cadence and operator-stop checks. World dashboard authentication retained. Append-only service-only evidence storage with RLS; no anonymous table/RPC access and no service-role UPDATE privilege.

Protocol: `alpha-net-probe-20261019-v1`, 19 Oct 16:00 UTC to 18 Nov 16:00 UTC (Berlin 19 Oct 18:00 to 18 Nov 17:00 because the daylight-saving offset changes). Final tolerance closes 18 Nov 16:17 UTC. The UI formats actual dates in Europe/Berlin. Frozen selection rules are documented in `BRIAN_EVIDENCE_PROTOCOL_20261019.md` and the source constants committed before the start.

## Actual first native run

19 Oct 14:13:01–14:13:03 UTC, `DEGRADED`, two reviewed/two persisted:

- ECB official speech: complete 137,148-byte response received, 28,902 extracted text characters, SHA-256 `1b6602fd85c5bd8a9915d7b88a5e66a0252d1470dab9d66fd5ef3335f3394ac0`. Exact complete headline not matched; independent claim truth remains unverified.
- EU Council official release: HTTP 403, stored `FAILED`. No content or provenance hash invented. No access-block circumvention attempted.
- Counter search: ECB 100 stored headlines; EU Council 78. No lexical candidates found, explicitly not confirmation.
- Available BTC/ETH windows persisted; unavailable windows null. Market movement remains a broad proxy and is not attributed to those events.

Actual receipts are in `BRIAN_EVIDENCE_LIVE_20261019.json`. Their rendering was replayed through the production view helper. This is not an authenticated visual browser test.

Authenticated live scorecard at 14:12:52 UTC returned HTTP 200, `forward-scorecard-v3-evidence`, outcome coverage and edge validation AVAILABLE, proof NOT_PROVEN, live_execution false. Diagnostic: 72 selected decisions, 55 cash/wait, 17 directional, 12 cost-known measured, 5 unresolved, average modeled net **−21.5446 bps**, one day block. The prospective period had not started; zero observations are not success.

Independent price collection continued after deployment: latest quote 14:15:04 UTC, 115 CORE quotes in the preceding five minutes. No engineering queue, worker, timer, DIP engine, risk settings or real-money flags were changed.

## Validation

PR #144 candidate `049cddbfa05cb978aa32ed993eed74ce4658c06c`; merge `a51b507ac346040a9ae551361005f9a24b0bd100`, identical Git tree. Main CI run 39266648841 passed all four jobs; ALPHA 39266648578 and Evolution 39266648561 passed. Behavioral/parser and authorization/STOP/failure checks, real PostgreSQL migration/grant/append-only/protocol checks, Deno type checks/lint and the existing Python suite ran. Preceding semantically identical source run 39266443180 recorded 144 required Node passes, 41 permanent product tests, 73 real PostgreSQL tests, 434 Python passes (47 skips) and 375 Deno passes. No failure was waived.

Applied migration history `20261019141022_world_evidence_reviews.sql` matches repository filename. Post-migration grants and RLS inspected. Security advisors added only the expected service-only table RLS-with-no-public-policy informational entry (115→116); the two pre-existing anonymous and two authenticated SECURITY DEFINER advisory findings were unchanged. Relevant advisory guidance: https://supabase.com/docs/guides/database/database-linter?lint=0008_rls_enabled_no_policy and https://supabase.com/docs/guides/database/database-linter?lint=0028_anon_security_definer_function_executable . No new public policies were added merely to silence the informational entry.

First production deployment `dpl_2LDJYBB6aZ3zYLTQVHrcDEUg1kFe` was READY and served all five changed JS/entry assets byte-for-byte from the tested source. Runtime logs contained existing `/api/scout-feed` Node DEP0169 deprecation warnings; do not interpret this as a clean log guarantee. An authenticated visual browser session was unavailable; the World endpoint's actual dashboard-key request was tested with behavioral mocks, while actual receipt persistence and rendered receipt content were separately verified.

## Still unproven / externally blocked

The EU Council 403 limits primary-source coverage. Independent semantic claim corroboration, verified counter-evidence, event-specific priced-in attribution and profitable executed net advantage remain unproven. This release supplies honest evidence collection and a future test, not those conclusions. More source reads or positive modeled averages alone cannot close these gates. No guaranteed profit or zero-error claim is made.

## Final runtime-health publication

PR #145 adds the evidence collector to World runtime health so partial document reads retain DEGRADED state. Candidate `ee35c343b61c54ff838be5564d66dd4af0471506`, merge `599cd1d5dabdf9462d71669953c8fde1eeca9509`, identical tested tree. Main CI 39267187943 passed all four jobs; ALPHA 39267187880 and Evolution 39267187856 passed both jobs each. The new behavioral test checks that a partial research collector remains visible as DEGRADED.

Final production deployment `dpl_3vvWJM11eRtBgUwuoTimFcXW8dGD` is READY on `monster-coins-pro-seven.vercel.app`, serving Git SHA `599cd1d5dabdf9462d71669953c8fde1eeca9509`. Final World HTML and JS were byte-identical to published source. Prior unchanged main entry/helper assets were already byte-verified. All four deployed Edge bundles were re-read and matched selected published source byte-for-byte: World research v1, World status v4, shadow scorecard v6, REALTIME data orchestrator v17. The final actual World endpoint rejected a keyless POST with HTTP 401. No authenticated browser visual test is claimed.
