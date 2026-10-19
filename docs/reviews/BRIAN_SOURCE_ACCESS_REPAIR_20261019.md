# Official source access repair — 19 October 2026

The current World evidence worker repeatedly consumes its two-document batches on new articles from hosts already denying access. Between approximately 15:00–18:45 UTC, recorded failures include SEC HTTP 403, EU Council HTTP 403 and rejected Federal Register redirects. These errors do not establish that a document was read.

The input function now pauses an official domain for 15 minutes after explicit HTTP 403/429, an unsafe redirect, or an unreadable/bot-block document. Related registered feeds share that pause; other official domains remain eligible. A later READ/INDEX_ONLY resets the pause. HTTP 5xx does not trigger the domain pause. Future-dated receipts do not affect source fairness or domain health. Existing event-level retry rules and the two-document/2,000-counter-candidate bounds remain.

The worker records source_access, next_retry_at and PARTIAL_ACCESS in its actual collector receipts. A batch with no fetches because providers are cooling down remains DEGRADED; it creates no invented document reviews and makes no independent-truth claim. This is an access scheduling repair, not an access bypass or a claim of universal source coverage.

Validation: permanent worker regression checks authentication/STOP, provider failures and a cooldown-only cycle; real PostgreSQL fixtures exercise shared domains, other-source progress, expiry, readable reset, future receipts and unchanged private execution grants. Local focused Node checks passed 22 tests; real PostgreSQL and type/lint checks must pass CI before production changes.

At 18:45 UTC the independent price collector was succeeding every minute, latest 25/25 assets with no missing quotes. Historical missing terminal observations cannot be reconstructed as contemporaneous evidence, and are not erased. Expected-edge profitability, semantic independent corroboration, verified contradiction and causal priced-in attribution remain unproven. Real money, DIP and the separate engineering automation remain outside this release.

## Published verification

PR #148 final candidate `ab65a964b4a4424a0fee7e00ad016ff482e9d216` passed all eight CI checks. Main checks include 135 required Node regressions, 59 product checks and 80 actual PostgreSQL tests; Deno type/lint and ALPHA/Evolution checks passed. No checks were waived. Merge `f74dae82a6e3ae0d1bfbdf8e2cb56224131a1024` has the identical tested tree. Applied migration filename is `20261019185251_world_research_domain_cooldown.sql`; the metadata alignment commit passed all four main CI jobs.

CORE World research v3 was read back byte-for-byte against the tested bundle. Its scheduled cycle at 18:58:04–18:58:09 UTC used `brian.world-evidence-review.v2-domain-cooldown`, recorded the cooldown source list and retry times, and correctly stayed DEGRADED. SEC became eligible after its earlier cooldown expired and still denied the document read; this is not presented as a successful verification. The 18:53 cycle had overlapped deployment and used the older worker; only later native receipts are used to verify the new code.

PR #149 candidate `2f2c354b724e0221405d20d97f2c820d093638ae` passed all six applicable CI checks and merged as `249668caf251cba121fb32c393071d32912c5ae8`, again with an identical tested tree. The unavailable general Yahoo Finance RSS URL was replaced in both the registry and actual direct-wire adapter with the provider's S&P 500/Nasdaq RSS feed. Applied filename: `20261019185946_repair_yahoo_benchmark_rss.sql`. Its scope is explicitly `SP500_NASDAQ_BENCHMARK_NEWS_NOT_ALL_YAHOO`; it does not claim all Yahoo news.

REALTIME Direct Wire v4 was read back exactly. Its scheduled 19:00:02 UTC cycle reported all four feeds healthy; Yahoo delivered 20 fresh items, two relevant discoveries, no fallback and no primary errors. Two actual new Yahoo event IDs were stored in REALTIME and reached CORE at 19:01:03 UTC with identical IDs, original observation times and the limited coverage label. Feed availability does not establish independent truth, timeliness advantage or profitable execution. Source registry health is earned through its unchanged quorum, not manually promoted.

The read-only last-hour price audit at 18:55 UTC found independent terminal observations for 128/128 matured five-minute windows and 104/104 matured fifteen-minute windows, with no truncation. No one-hour window from that last-hour cohort had matured plus its tolerance, so it was not measured. This is a finite price-availability check, not a universal 100% guarantee. Old absent observations remain absent. Latest quote checks still covered 25 assets.

Authenticated net scorecard remained AVAILABLE and NOT_PROVEN: at 18:55 UTC its fixed prospective sample had 24 selected decisions, three directional, two measured, one awaiting resolution, 21 cash decisions, one day block and mean modeled net return -65.659 basis points over the measured observations. Dates, policy, costs, denominator and promotion rules were not modified to improve these numbers.

Security advisor findings are unchanged: 116 service-only RLS informational items and two anonymous/two authenticated pre-existing SECURITY DEFINER findings. The repaired function remains invoker-only; anonymous/authenticated execution remains denied. No frontend source changed, so backend publication did not require a Vercel frontend redeploy.

## Readiness limits

| Capability | Verified state |
| --- | --- |
| Minute price capture and recent 5/15-minute terminal windows | Operational in the measured cohort |
| Multiasset shadow decisions and position continuity | Operational checks passed; simulated only |
| Yahoo benchmark news to REALTIME to CORE | Native end-to-end receipt verified |
| Source access failures and domain retry policy | Explicit, bounded, still partial |
| Historical missing observations | Cannot be recreated as contemporaneous evidence |
| Independent semantic corroboration and verified contradiction | Not established |
| Event-specific priced-in probability and causal attribution | Not established |
| Positive net advantage and live-money readiness | Not established; experimental gates remain closed |

These results justify continued measured shadow testing. They do not justify saying the full intended system is 100% ready or that it will make money. Full receipts and scope are in `BRIAN_SOURCE_ACCESS_LIVE_20261019.json`.

At 19:04 UTC the repaired Yahoo endpoint earned two healthy samples out of the existing three-sample quorum: HTTP 200, parseable=true, origin_match=true, composite 0.786, lifecycle ACTIVE, research_eligible=true. This was produced by the unchanged registry verifier after the normal 120-second polling interval; historical failures and direct-decision restrictions were preserved. The 19:03 scheduled research cycle independently confirmed a cooldown-only batch remained DEGRADED with zero fabricated reviews.

The broader registry check also exposed an origin mapping error for Bloomberg: its feeds.bloomberg.com endpoint redirects to the actual parsed RSS at https://www.bloomberg.com/feeds/markets/news.rss, but the registered canonical domain was the feeds-only subdomain. Direct checks returned HTTP 200 and RSS XML. Guarded configuration migration `20261019190608_repair_bloomberg_feed_origin.sql` records that observed canonical URL and publisher root `bloomberg.com`; it does not allow unrelated domains or raise trust scores. The normal health quorum must still be earned after this mapping repair.

Final verification at 19:09 UTC: Bloomberg earned two healthy samples of three through the normal verifier, HTTP 200, parseable/origin/reachable true, composite 0.7905, ACTIVE and research_eligible=true. Its direct-decision permission remains false. The final configuration/migration commit `a6dbf84021baa7f9697524bb5802a1b0c0bfd143` passed all four main CI jobs. The last native research run at 19:08 UTC continued with two persisted reviews and explicit DEGRADED status rather than masking unavailable document access. Remaining readiness limits above are unchanged.
