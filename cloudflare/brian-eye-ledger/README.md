# Brian Cloudflare Eye + First-Seen Ledger (SHADOW)

This package is the first non-invasive multi-cloud layer for Brian.

## Safety contract

- Cloudflare is observer/shadow only.
- Existing Supabase collectors remain active as secondary/failover eyes.
- Supabase remains the canonical core.
- No live execution is enabled here.
- Treasury is not called.
- New source items are direction-neutral and locked from decision evidence.
- A Cloudflare event is forwarded to Supabase only through an idempotent canonical ingest endpoint.
- ALPHA event recheck is only called through the explicit authenticated route /route/alpha-recheck.

## Phase 1: deploy without R2

R2 is intentionally not bound in the initial configuration.

1. Open the Cloudflare account.
2. In this directory run: npm install
3. Authenticate with: npx wrangler login
4. Add the existing Brian cron key as a Worker secret:
   npx wrangler secret put BRIAN_CRON_KEY
5. Add the source manifest as a Worker secret:
   npx wrangler secret put SOURCE_MANIFEST_JSON
6. Keep ENABLE_SCHEDULED_EYE=false.
7. Deploy with: npx wrangler deploy
8. Verify /health.
9. Manually test one shadow scan with authenticated POST /run.
10. Compare Cloudflare ledger first-seen timestamps with the existing Supabase path before enabling scheduled polling.

## Source manifest shape

SOURCE_MANIFEST_JSON is a JSON array. Every source must be HTTPS and the final response host must match canonical_domain.

Example fields:
- endpoint_id
- source_id
- organization
- canonical_domain
- endpoint_url
- endpoint_kind
- tier
- category
- region
- priority

## Phase 2: R2 - only after explicit approval

After R2 is activated in the Cloudflare dashboard:

1. Create a Standard bucket named brian-intelligence-raw.
2. Add an R2 binding named RAW_BUCKET to wrangler.jsonc.
3. Set R2_ENABLED=true.
4. Deploy.
5. Verify /health reports r2_enabled=true.
6. Test one source and verify the raw XML object exists before broadening the source manifest.

Target R2 binding:

{
  "r2_buckets": [
    {
      "binding": "RAW_BUCKET",
      "bucket_name": "brian-intelligence-raw"
    }
  ]
}

## Durable Object

FirstSeenLedger uses SQLite-backed Durable Object storage and is sharded by the first two hex characters of event_id.

It records:
- immutable first_seen_at
- last_seen_at
- seen count
- source and payload hash
- canonical event envelope
- capture envelope
- Supabase forwarding state and errors

This keeps first-seen timing independent from Supabase database pressure.
