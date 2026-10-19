import { createClient } from 'npm:@supabase/supabase-js@2.116.0';
import { withCollectorLease } from '../_shared/collector_lease.ts';
import { requireRealtimeInternal } from '../_shared/realtime_internal_auth.ts';
import { BOOK_URI, PRICE_CAPTURE_VERSION, parseOutcomeQuotes, readBookSnapshot } from './logic.ts';

const db = createClient(Deno.env.get('SUPABASE_URL')!, Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!, { auth: { persistSession: false, autoRefreshToken: false } });
const ID = 'brian-outcome-price-eye-v1';
const out = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' } });
async function hash(value: string) { return [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value)))].map(b => b.toString(16).padStart(2, '0')).join(''); }
async function enabled() {
  const q = await db.from('brian_evolution_runtime_config').select('config_value').eq('config_key', 'brian_system_enabled').single().abortSignal(AbortSignal.timeout(4000));
  if (q.error) throw Error('OPERATOR_CONTROL_UNAVAILABLE');
  return q.data?.config_value === 'true';
}
async function record(startedAt: string, status: string, observed: number, stored: number, details: Record<string, unknown>, error: string | null = null) {
  const finishedAt = new Date().toISOString();
  const q = await db.from('brian_collector_runs').insert({ run_id: await hash(`${ID}|${startedAt}|${finishedAt}|${status}`), collector_id: ID,
    started_at: startedAt, finished_at: finishedAt, status, observed_records: observed, stored_records: stored,
    degraded_sources: status === 'DEGRADED' ? ['outcome_book_quotes_partial'] : [], error_class: error ? 'OUTCOME_PRICE_CAPTURE' : null, error_message: error,
    evidence_class: 'PROSPECTIVE_DEVELOPMENT_SHADOW', shadow_only: true, live_execution: false,
    metadata: { ...details, version: PRICE_CAPTURE_VERSION, independent_of_radar: true, price_only: true, provider_uri: BOOK_URI, quote_time_semantics: 'HTTP_RECEIPT_TIME_EXCHANGE_TIME_UNKNOWN' },
  }).abortSignal(AbortSignal.timeout(4000));
  if (q.error) { console.error('outcome-price receipt unavailable'); return false; }
  return true;
}

Deno.serve(async (req: Request) => {
  if (req.method !== 'POST') return out({ error: 'POST_REQUIRED' }, 405);
  try { await requireRealtimeInternal(req); } catch { return out({ status: 'UNAUTHORIZED' }, 401); }
  const startedAt = new Date().toISOString();
  try {
    if (!await enabled()) return out({ status: 'OPERATOR_STOPPED', shadow_only: true, live_execution: false });
    const lease = await withCollectorLease(db, ID, 45, async () => {
      const last = await db.from('brian_collector_runs').select('started_at,metadata').eq('collector_id', ID)
        .in('status', ['SUCCESS', 'DEGRADED', 'FAILED']).order('started_at', { ascending: false }).limit(1).maybeSingle().abortSignal(AbortSignal.timeout(4000));
      if (last.error) throw Error('PRICE_CADENCE_UNAVAILABLE');
      const cooldown = Date.parse(String(last.data?.metadata?.next_allowed_at ?? ''));
      if (Number.isFinite(cooldown) && cooldown > Date.now()) return out({ status: 'SKIPPED_PROVIDER_COOLDOWN', shadow_only: true, live_execution: false });
      if (last.data?.started_at && last.data.started_at.slice(0, 16) === startedAt.slice(0, 16)) return out({ status: 'SKIPPED_INTERVAL', shadow_only: true, live_execution: false });
      const watch = await db.rpc('brian_outcome_watch_assets_v1').abortSignal(AbortSignal.timeout(4000));
      if (watch.error) throw Error('WATCH_ASSETS_UNAVAILABLE');
      const assets = (watch.data ?? []).map((r: { asset_id: string }) => r.asset_id);
      if (assets.length > 500) throw Error('WATCH_CAPACITY_EXCEEDED');
      if (!assets.length) {
        const receipt = await record(startedAt, 'SUCCESS', 0, 0, { requested_assets: 0, provider_requests: 0 });
        return out({ status: receipt ? 'NO_PENDING_ASSETS' : 'DEGRADED', requested_assets: 0, stored: 0, shadow_only: true, live_execution: false });
      }
      const snapshot = await readBookSnapshot();
      const { quotes, missing } = parseOutcomeQuotes(snapshot.payload, assets);
      // Receipt time is honest local observation time; the API has no exchange timestamp.
      const rows = await Promise.all(quotes.map(async quote => ({ source_event_id: await hash(`${PRICE_CAPTURE_VERSION}|${quote.asset_id}|${snapshot.receivedAt.slice(0, 16)}`),
        asset_id: quote.asset_id, observed_at: snapshot.receivedAt, request_started_at: snapshot.requestStartedAt,
        bid_price: quote.bid_price, ask_price: quote.ask_price,
        source_metadata: { version: PRICE_CAPTURE_VERSION, provider_response_sha256: snapshot.responseHash,
          quote_time_semantics: 'HTTP_RECEIPT_TIME_EXCHANGE_TIME_UNKNOWN', bid_qty: quote.bid_qty, ask_qty: quote.ask_qty, exchange_timestamp: null },
      })));
      if (!await enabled()) return out({ status: 'OPERATOR_STOPPED', shadow_only: true, live_execution: false });
      const persisted = await db.rpc('brian_record_outcome_prices_v1', { p_rows: rows }).abortSignal(AbortSignal.timeout(5000));
      if (persisted.error) throw Error('OUTCOME_PRICE_PERSIST_FAILED');
      if (persisted.data?.status === 'OPERATOR_STOPPED') return out({ status: 'OPERATOR_STOPPED', shadow_only: true, live_execution: false });
      if (persisted.data?.status !== 'SUCCESS' || persisted.data?.received !== rows.length || !Number.isInteger(persisted.data?.stored)) throw Error('OUTCOME_PRICE_RECEIPT_MISMATCH');
      const status = missing.length ? 'DEGRADED' : 'SUCCESS';
      const details = { requested_assets: assets.length, available_quotes: rows.length, missing_assets: missing, provider_requests: 1, observed_at: snapshot.receivedAt };
      const receipt = await record(startedAt, status, assets.length, persisted.data.stored, details);
      return out({ status: receipt ? status : 'DEGRADED', ...details, stored: persisted.data.stored, receipt_stored: receipt, shadow_only: true, live_execution: false });
    });
    return lease.contended ? out({ status: 'SKIPPED_LEASE_CONTENDED', shadow_only: true, live_execution: false }) : lease.value!;
  } catch (error) {
    const message = error instanceof Error ? error.message : 'OUTCOME_PRICE_FAILED';
    const retry = error && typeof error === 'object' && 'retryAfterSeconds' in error ? Number(error.retryAfterSeconds) : 60;
    await record(startedAt, 'FAILED', 0, 0, { next_allowed_at: new Date(Date.now() + Math.min(3600, Math.max(60, Number.isFinite(retry) ? retry : 60)) * 1000).toISOString() }, message).catch(() => {});
    return out({ status: 'FAILED_CLOSED', error: message, shadow_only: true, live_execution: false }, 503);
  }
});
