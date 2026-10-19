import { createClient } from 'npm:@supabase/supabase-js@2.116.0';
import { withCollectorLease } from '../_shared/collector_lease.ts';
import { requireRealtimeInternal } from '../_shared/realtime_internal_auth.ts';
import { collectPredictionCycle, DEFAULT_COLLECTOR_CONFIG, PREDICTION_COLLECTOR_VERSION } from '../_shared/prediction_collector.ts';

// Prediction Intelligence eye: one bounded, read-only, shadow-only collection cycle per call.
// It never places orders and never touches a decision. It does nothing unless BOTH the global
// Brian switch and brian_prediction_control.module_enabled are on; wallet trades additionally
// require wallet_tracking_enabled. No schedule is created by this function or its migration.

const db = createClient(Deno.env.get('SUPABASE_URL')!, Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!, { auth: { persistSession: false, autoRefreshToken: false } });
const ID = 'brian-prediction-eye-v1';
const SAFE = { shadow_only: true, live_execution: false, read_only: true };
const out = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' } });

async function systemEnabled() {
  const q = await db.from('brian_evolution_runtime_config').select('config_value').eq('config_key', 'brian_system_enabled').abortSignal(AbortSignal.timeout(4000)).single();
  if (q.error) throw Error('OPERATOR_CONTROL_UNAVAILABLE');
  return q.data?.config_value === 'true';
}

async function readControl() {
  const q = await db.from('brian_prediction_control').select('module_enabled,wallet_tracking_enabled,max_markets,min_liquidity,tag_slugs')
    .eq('control_id', 'prediction-intelligence').abortSignal(AbortSignal.timeout(4000)).single();
  if (q.error || !q.data) throw Error('PREDICTION_CONTROL_UNAVAILABLE');
  return q.data as { module_enabled: boolean; wallet_tracking_enabled: boolean; max_markets: number; min_liquidity: number; tag_slugs: string[] };
}

Deno.serve(async (req: Request) => {
  if (req.method !== 'POST') return out({ error: 'POST_REQUIRED' }, 405);
  try { await requireRealtimeInternal(req); } catch { return out({ status: 'UNAUTHORIZED' }, 401); }
  try {
    if (!await systemEnabled()) return out({ status: 'OPERATOR_STOPPED', ...SAFE });
    const control = await readControl();
    if (!control.module_enabled) return out({ status: 'MODULE_DISABLED', ...SAFE });

    const lease = await withCollectorLease(db, ID, 90, async () => {
      const config = {
        ...DEFAULT_COLLECTOR_CONFIG,
        tagSlugs: Array.isArray(control.tag_slugs) ? control.tag_slugs : DEFAULT_COLLECTOR_CONFIG.tagSlugs,
        maxMarkets: Number(control.max_markets) || DEFAULT_COLLECTOR_CONFIG.maxMarkets,
        minLiquidity: Number.isFinite(Number(control.min_liquidity)) ? Number(control.min_liquidity) : DEFAULT_COLLECTOR_CONFIG.minLiquidity,
        walletTracking: control.wallet_tracking_enabled === true,
      };
      let known: { wallet: string; watermark: string | null } | null = null;
      if (config.walletTracking) {
        const w = await db.from('brian_prediction_wallets').select('wallet,trade_watermark')
          .eq('display_name', config.trackedName).order('updated_at', { ascending: false }).limit(1).abortSignal(AbortSignal.timeout(4000)).maybeSingle();
        if (w.error) throw Error('PREDICTION_WALLET_STATE_UNAVAILABLE');
        if (w.data) known = { wallet: w.data.wallet, watermark: w.data.trade_watermark };
      }

      const batch = await collectPredictionCycle(config, known);

      // Registry first WITHOUT advancing the watermark, so a newly verified wallet exists before
      // its trades are inserted (the RPC only stores trades of a VERIFIED registry wallet).
      if (batch.wallet) {
        const reg = await db.rpc('brian_upsert_prediction_wallet_v1', {
          p_wallet: batch.wallet.wallet, p_display_name: batch.wallet.displayName, p_status: batch.wallet.status,
          p_evidence: batch.wallet.evidence, p_watermark: null,
        }).abortSignal(AbortSignal.timeout(4000));
        if (reg.error) throw Error('PREDICTION_WALLET_REGISTRY_FAILED');
      }

      if (!await systemEnabled()) return out({ status: 'OPERATOR_STOPPED', ...SAFE });
      const persisted = await db.rpc('brian_record_prediction_batch_v1', {
        p_markets: batch.markets, p_snapshots: batch.snapshots, p_trades: batch.trades, p_health: batch.health,
      }).abortSignal(AbortSignal.timeout(8000));
      if (persisted.error) throw Error('PREDICTION_PERSIST_FAILED');
      const r = persisted.data as Record<string, any>;
      if (r?.status === 'OPERATOR_STOPPED') return out({ status: 'MODULE_DISABLED', ...SAFE });
      if (r?.status !== 'SUCCESS' || r.markets?.received !== batch.markets.length || r.snapshots?.received !== batch.snapshots.length ||
          r.trades?.received !== batch.trades.length) throw Error('PREDICTION_RECEIPT_MISMATCH');

      // The watermark advances only after the trades carrying it were accepted by the database.
      if (batch.wallet?.status === 'VERIFIED' && batch.wallet.watermark && r.trades?.tracking_enabled === true) {
        const mark = await db.rpc('brian_upsert_prediction_wallet_v1', {
          p_wallet: batch.wallet.wallet, p_display_name: batch.wallet.displayName, p_status: 'VERIFIED',
          p_evidence: batch.wallet.evidence, p_watermark: batch.wallet.watermark,
        }).abortSignal(AbortSignal.timeout(4000));
        if (mark.error) throw Error('PREDICTION_WATERMARK_FAILED');
      }

      const degraded = batch.health.some((h) => h.status !== 'OK');
      return out({
        status: degraded ? 'DEGRADED' : 'SUCCESS', version: PREDICTION_COLLECTOR_VERSION, summary: batch.summary,
        stored: { markets: r.markets.stored, snapshots: r.snapshots.stored, trades: r.trades.stored, health: r.health.stored },
        ...SAFE,
      });
    });
    return lease.contended ? out({ status: 'SKIPPED_LEASE_CONTENDED', ...SAFE }) : lease.value!;
  } catch (error) {
    const message = error instanceof Error ? error.message : 'PREDICTION_EYE_FAILED';
    return out({ status: 'FAILED_CLOSED', error: message, ...SAFE }, 503);
  }
});
