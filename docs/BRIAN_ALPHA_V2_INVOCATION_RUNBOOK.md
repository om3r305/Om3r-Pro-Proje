# Brian ALPHA v2 — Production SHADOW Invocation Contract

This runbook is a deployment gate for PR #49 / Issue #50. It does **not** authorize a production rollout by itself.

## Functions covered

- `brian-alpha-decision-compiler`
- `brian-missed-opportunity-auditor`
- `brian-official-macro-eye`

All three functions use the Supabase service-role client internally, so an invocation must be authenticated **before** any collector lease or write path is entered.

## Required outer and inner gates

1. Keep Supabase Edge Function gateway JWT verification enabled (`verify_jwt=true`) at deployment.
2. Every scheduled invocation must also include `x-brian-cron-key`.
3. The cron key is verified against `public.brian_dashboard_auth.cron_key_sha256` for `auth_id='control-v2'` using SHA-256 plus constant-time comparison.
4. Missing or incorrect cron key must return HTTP 401 and must not create collector/decision/audit rows.
5. If the auth row itself cannot be read, fail closed (HTTP 503); do not treat auth infrastructure failure as authorization.

The custom cron key is an application-level gate in addition to the gateway JWT. An anon JWT by itself is insufficient to invoke these service-role writers.

## Scheduling requirements

When the production schedule migration is prepared:

- Store invocation credentials in Supabase Vault; never hard-code them into a migration.
- `pg_net` / `pg_cron` requests must send the normal Supabase function Authorization/JWT headers required by the gateway **and** `x-brian-cron-key`.
- Reuse the existing Brian Control Center cron credential contract rather than creating a second plaintext secret.
- Do not place a Binance API key, exchange secret, HMAC material, or any authenticated exchange credential in these schedules.

## Mandatory pre-rollout smoke checks

For each of the three functions:

1. No `x-brian-cron-key` → 401, no DB writes.
2. Wrong `x-brian-cron-key` → 401, no DB writes.
3. Valid gateway JWT but wrong cron key → 401, no DB writes.
4. Correct gateway JWT + correct cron key → function may run in SHADOW mode.
5. Response and written rows must still report `shadow_only=true` and `live_execution=false`.

## Non-negotiable boundaries

- Phase 3.7 remains frozen/read-only.
- No live order endpoint or authenticated exchange execution surface.
- GDELT remains discovery-only/no direction vote.
- Official macro remains context-only/no direction vote.
- No Sep-4 hindsight threshold tuning.
- Production deployment remains blocked until Issue #50 review/CI is complete.
