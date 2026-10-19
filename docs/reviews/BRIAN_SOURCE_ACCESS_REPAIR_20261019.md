# Official source access repair — 19 October 2026

The current World evidence worker repeatedly consumes its two-document batches on new articles from hosts already denying access. Between approximately 15:00–18:45 UTC, recorded failures include SEC HTTP 403, EU Council HTTP 403 and rejected Federal Register redirects. These errors do not establish that a document was read.

The input function now pauses an official domain for 15 minutes after explicit HTTP 403/429, an unsafe redirect, or an unreadable/bot-block document. Related registered feeds share that pause; other official domains remain eligible. A later READ/INDEX_ONLY resets the pause. HTTP 5xx does not trigger the domain pause. Future-dated receipts do not affect source fairness or domain health. Existing event-level retry rules and the two-document/2,000-counter-candidate bounds remain.

The worker records source_access, next_retry_at and PARTIAL_ACCESS in its actual collector receipts. A batch with no fetches because providers are cooling down remains DEGRADED; it creates no invented document reviews and makes no independent-truth claim. This is an access scheduling repair, not an access bypass or a claim of universal source coverage.

Validation: permanent worker regression checks authentication/STOP, provider failures and a cooldown-only cycle; real PostgreSQL fixtures exercise shared domains, other-source progress, expiry, readable reset, future receipts and unchanged private execution grants. Local focused Node checks passed 22 tests; real PostgreSQL and type/lint checks must pass CI before production changes.

At 18:45 UTC the independent price collector was succeeding every minute, latest 25/25 assets with no missing quotes. Historical missing terminal observations cannot be reconstructed as contemporaneous evidence, and are not erased. Expected-edge profitability, semantic independent corroboration, verified contradiction and causal priced-in attribution remain unproven. Real money, DIP and the separate engineering automation remain outside this release.
