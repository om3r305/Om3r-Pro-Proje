# Brian Phase 3.2 — Supabase Prospective Intelligence Runtime

Status: **ACTIVE FOUNDATION / SHADOW_RESEARCH_ONLY**

## Purpose

This runtime gives Brian a durable point-in-time memory that begins collecting what the system actually observed from deployment time forward. It does not manufacture historical social/on-chain knowledge and it does not place exchange orders.

The production Brian Supabase project is intentionally isolated from Burger Brothers infrastructure.

## Data plane

The first live collector is `brian-universe-collector`.

It observes unauthenticated Binance Spot public endpoints:

- `exchangeInfo`
- 24h ticker statistics
- top-of-book ticker snapshots

The collector:

1. timestamps each provider response **after** it is received;
2. fails closed if required public sources are unavailable;
3. treats book-ticker as optional and marks it degraded instead of fabricating spread;
4. stores raw provider payloads in a **private** object bucket using SHA-256 content addressing;
5. gzip-compresses raw payloads before object storage;
6. stores only immutable capture metadata/pointers in Postgres;
7. applies the same initial Universe Radar thresholds used by `brian2026/universe_radar.py`;
8. writes an append-only universe snapshot;
9. treats the first snapshot as a baseline, never as a fake listing alert;
10. returns research priorities only — never BUY/SELL or an exchange order.

## Current cadence

The bootstrap collector is scheduled every **15 minutes** through Supabase `pg_cron` + `pg_net`.

The Edge Function also has a server-side minimum interval guard of 780 seconds to absorb duplicate/jittered invocations.

This cadence is deliberate. Supabase is the first durable prospective memory layer, not a low-latency HFT bus. Lower-latency WebSocket/social/on-chain collectors should use a dedicated streaming worker/object-store layer later while continuing to write the same PIT contracts.

## Storage model

### Private raw object storage

Bucket: `brian-intelligence-raw`

- private
- JSON source payloads stored as `.json.gz`
- object path is content-addressed by payload SHA-256
- repeated identical provider payloads reuse the same object path

### Append-only Postgres tables

The runtime includes:

- `brian_raw_captures`
- `brian_intel_events`
- `brian_entities`
- `brian_entity_labels`
- `brian_entity_edges`
- `brian_whale_flows`
- `brian_source_outcomes`
- `brian_universe_snapshots`
- `brian_opportunity_observations`
- `brian_opportunity_outcomes`

`anon` and `authenticated` have no table access. RLS is enabled with no client policies by design. `service_role` receives SELECT+INSERT only; UPDATE/DELETE/TRUNCATE are revoked and append-only mutation triggers provide a second barrier.

Outcome tables enforce causal timestamps: an outcome cannot resolve before its parent event/opportunity was observed.

## Runtime credentials

**No deployed project URL, JWT, provider token, service-role key, API key or secret may be committed to GitHub.**

The scheduler reads only these Supabase Vault secret names at runtime:

- `brian_project_url`
- `brian_anon_jwt`

Values are provisioned out-of-band in the target Supabase project.

The Edge Function itself reads Supabase-managed runtime variables `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`; these are never committed.

Deploy the Edge Function with JWT verification enabled.

## Scientific invariants

- SHADOW_RESEARCH_ONLY.
- No authenticated exchange API.
- No order submission surface.
- No social popularity -> truth shortcut.
- No whale transfer -> direction shortcut.
- No hindsight label backfill.
- Raw evidence is stored before derived ranking is treated as evidence.
- Missing optional data remains explicitly unavailable/degraded.
- First universe observation is baseline-only.
- 2026 historical contaminated-holdout status is unchanged; this new prospective memory is not a replacement pristine holdout by itself.

## Next intelligence increments

1. Official announcement/event collectors with Event Truth Engine normalization.
2. Social Forensics collectors with duplicate/bot/coordinated-campaign resistance.
3. On-chain entity/flow adapters and Smart-Money Consensus.
4. Derivatives/OI/funding/liquidation context where reliable PIT sources exist.
5. Cross-asset narrative and propagation memory.
6. Dedicated streaming runtime for sub-minute sources while Supabase remains the durable structured memory/evidence ledger.

None of these sources may bypass the downstream evidence, portfolio-risk and shadow-execution separation.
