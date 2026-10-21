import { createClient } from 'npm:@supabase/supabase-js@2.116.0';
import { withCollectorLease } from '../_shared/collector_lease.ts';
import { requireCronAuth } from '../_shared/cron_auth.ts';
import { fetchPredictionJson, gammaMarketByIdUrl, normalizeGammaResolution } from '../_shared/prediction_market.ts';
import { classifyPolymarketDomain } from '../_shared/polymarket_domain.ts';
import { decideEntry, fromRawBook, marketFeeRate, PM_CALIBRATION_RULES, PM_CALIBRATION_RULES_VERSION, pmCalibrationRulesFingerprint, returnPerDollar } from '../_shared/pm_calibration_shadow.ts';

// Prospective shadow test pm-calibration-shadow.v1 (PREREGISTRATION.md, commit 08ea1bc): once per
// hour, capture political Yes/No markets 29-31 h before their scheduled end and record the frozen
// entry decision at the observed best ask; later record the UMA-resolved outcome. Read-only public
// Polymarket APIs; hypothetical entries only. Obeys the Brian master switch and the prediction module.

const db = createClient(Deno.env.get('SUPABASE_URL')!, Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!, { auth: { persistSession: false, autoRefreshToken: false } });
const ID = 'brian-pm-calibration-shadow-v1';
const SAFE = { shadow_only: true, live_execution: false, trade_signal: false };
const MAX_CAPTURES = 60, MAX_RESOLUTIONS = 50, PAGE = 500, MAX_PAGES = 4;
const out = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' } });
const FETCH = { maxAttempts: 2, timeoutMs: 8000 };

async function enabled() {
  const [sys, ctl] = await Promise.all([
    db.from('brian_evolution_runtime_config').select('config_value').eq('config_key', 'brian_system_enabled').abortSignal(AbortSignal.timeout(4000)).single(),
    db.from('brian_prediction_control').select('module_enabled').eq('control_id', 'prediction-intelligence').abortSignal(AbortSignal.timeout(4000)).single(),
  ]);
  if (sys.error || ctl.error) throw Error('OPERATOR_CONTROL_UNAVAILABLE');
  return sys.data?.config_value === 'true' && ctl.data?.module_enabled === true;
}

const parseArr = (v: unknown): string[] | null => { try { const a = typeof v === 'string' ? JSON.parse(v) : v; return Array.isArray(a) ? a.map(String) : null; } catch { return null; } };

async function capture(fp: string) {
  const now = Date.now(), [lo, hi] = PM_CALIBRATION_RULES.entryWindowHours;
  const from = new Date(now + lo * 3600_000).toISOString(), to = new Date(now + hi * 3600_000).toISOString();
  const markets: Record<string, unknown>[] = [];
  let truncated = false;
  for (let page = 0; page < MAX_PAGES; page++) {
    const url = `https://gamma-api.polymarket.com/markets?closed=false&end_date_min=${encodeURIComponent(from)}&end_date_max=${encodeURIComponent(to)}&limit=${PAGE}&offset=${page * PAGE}`;
    const r = await fetchPredictionJson<unknown[]>(url, FETCH);
    if (r.status !== 'OK' || !Array.isArray(r.body)) throw Error(`GAMMA_${r.status}_${r.httpStatus ?? ''}`);
    markets.push(...(r.body as Record<string, unknown>[]));
    if (r.body.length < PAGE) break;
    if (page === MAX_PAGES - 1) truncated = true;
  }
  const eligible = markets.filter((m) => classifyPolymarketDomain(String(m.question ?? '')) === 'Politics'
    && JSON.stringify(parseArr(m.outcomes)) === '["Yes","No"]' && (parseArr(m.clobTokenIds)?.length ?? 0) === 2 && m.id != null && m.endDate);
  const ids = eligible.map((m) => String(m.id));
  const seen = ids.length ? await db.from('brian_pm_calibration_entries').select('market_id').eq('rules_version', PM_CALIBRATION_RULES_VERSION).in('market_id', ids).abortSignal(AbortSignal.timeout(5000)) : { data: [], error: null };
  if (seen.error) throw Error('ENTRY_LOOKUP_FAILED');
  const already = new Set((seen.data ?? []).map((r: { market_id: string }) => r.market_id));
  const fresh = eligible.filter((m) => !already.has(String(m.id))).slice(0, MAX_CAPTURES);
  let stored = 0, trades = 0, bookErrors = 0;
  for (const m of fresh) {
    const [yesToken, noToken] = parseArr(m.clobTokenIds)!;
    const [yb, nb] = await Promise.all([
      fetchPredictionJson(`https://clob.polymarket.com/book?token_id=${encodeURIComponent(yesToken)}`, FETCH),
      fetchPredictionJson(`https://clob.polymarket.com/book?token_id=${encodeURIComponent(noToken)}`, FETCH),
    ]);
    if (yb.status !== 'OK' || nb.status !== 'OK') { bookErrors++; continue; } // retried next hour while still in the window
    const yes = fromRawBook(yb.body), no = fromRawBook(nb.body), feeRate = marketFeeRate(m);
    const d = decideEntry(yes, no, feeRate);
    const events = Array.isArray(m.events) ? m.events as Array<{ id?: unknown }> : [];
    const row = {
      entry_id: `${PM_CALIBRATION_RULES_VERSION}:${m.id}`, market_id: String(m.id), event_id: events[0]?.id != null ? String(events[0].id) : null,
      question: String(m.question).slice(0, 500), end_date: new Date(String(m.endDate)).toISOString(), captured_at: new Date().toISOString(),
      rules_version: PM_CALIBRATION_RULES_VERSION, rules_fingerprint: fp,
      yes_bid: yes.bid, yes_ask: yes.ask, yes_ask_shares: yes.askShares, no_bid: no.bid, no_ask: no.ask, no_ask_shares: no.askShares,
      fee_rate: feeRate, mid: d.mid, q: d.q, action: d.action, skip_reason: d.action === 'SKIP' ? d.reason : null,
      price: d.action === 'SKIP' ? null : d.price, fee: d.action === 'SKIP' ? null : d.fee, edge: d.action === 'SKIP' ? null : d.edge,
      shadow_only: true, live_execution: false,
    };
    const ins = await db.from('brian_pm_calibration_entries').upsert(row, { onConflict: 'entry_id', ignoreDuplicates: true }).abortSignal(AbortSignal.timeout(5000));
    if (ins.error) throw Error('ENTRY_PERSIST_FAILED');
    stored++; if (d.action !== 'SKIP') trades++;
  }
  return { window: [from, to], markets_in_window: markets.length, truncated, eligible: eligible.length, already_captured: already.size, stored, trades, book_errors: bookErrors };
}

async function resolve() {
  const pending = await db.rpc('brian_pm_calibration_pending_v1', { p_limit: MAX_RESOLUTIONS }).abortSignal(AbortSignal.timeout(5000));
  if (pending.error) throw Error('PENDING_LOOKUP_FAILED');
  let resolved = 0, open = 0, errors = 0;
  for (const p of (pending.data ?? []) as Array<{ entry_id: string; market_id: string; action: string; price: number | null; fee: number | null }>) {
    const r = await fetchPredictionJson(gammaMarketByIdUrl(p.market_id), FETCH);
    if (r.status !== 'OK') { errors++; continue; }
    const n = normalizeGammaResolution(r.body, r.receivedAt);
    if (!n.ok || n.value.yesOutcome === null) { open++; continue; } // not (cleanly) resolved yet
    const y = n.value.yesOutcome as 0 | 1;
    const ret = p.action === 'SKIP' || p.price == null ? null : returnPerDollar(p.action as 'BUY_YES' | 'BUY_NO', Number(p.price), Number(p.fee ?? 0), y);
    const ins = await db.from('brian_pm_calibration_outcomes').upsert({ entry_id: p.entry_id, resolved_at: r.receivedAt, yes_outcome: y, return_per_dollar: ret, uma_resolution_status: n.value.umaResolutionStatus, closed_time: n.value.closedTime, shadow_only: true, live_execution: false }, { onConflict: 'entry_id', ignoreDuplicates: true }).abortSignal(AbortSignal.timeout(5000));
    if (ins.error) throw Error('OUTCOME_PERSIST_FAILED');
    resolved++;
  }
  return { pending: (pending.data ?? []).length, resolved, still_open: open, fetch_errors: errors };
}

async function receipt(started: string, status: string, observed: number, stored: number, metadata: Record<string, unknown>, error: string | null) {
  const finished = new Date().toISOString();
  const runId = [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(`${ID}|${started}|${finished}`)))].map((b) => b.toString(16).padStart(2, '0')).join('');
  await db.from('brian_collector_runs').insert({ run_id: runId, collector_id: ID, started_at: started, finished_at: finished, status, observed_records: observed, stored_records: stored,
    degraded_sources: status === 'SUCCESS' ? [] : ['pm_calibration_partial'], error_class: error ? 'PM_CALIBRATION_FAILED' : null, error_message: error?.slice(0, 200) ?? null,
    evidence_class: 'PROSPECTIVE_EVOLUTION_SHADOW', shadow_only: true, live_execution: false, metadata: { rules_version: PM_CALIBRATION_RULES_VERSION, ...metadata } }).abortSignal(AbortSignal.timeout(4000));
}

Deno.serve(async (req: Request) => {
  if (req.method !== 'POST') return out({ error: 'POST_REQUIRED' }, 405);
  try { await requireCronAuth(req, db); } catch { return out({ status: 'UNAUTHORIZED' }, 401); }
  const started = new Date().toISOString();
  try {
    if (!await enabled()) return out({ status: 'OPERATOR_STOPPED', ...SAFE });
    const fp = await pmCalibrationRulesFingerprint();
    const lease = await withCollectorLease(db, ID, 110, async () => {
      const cap = await capture(fp), res = await resolve();
      const status = cap.truncated || cap.book_errors || res.fetch_errors ? 'DEGRADED' : 'SUCCESS';
      await receipt(started, status, cap.eligible + res.pending, cap.stored + res.resolved, { rules_fingerprint: fp, capture: cap, resolution: res }, null);
      return out({ status, rules_fingerprint: fp, capture: cap, resolution: res, ...SAFE });
    });
    return lease.contended ? out({ status: 'SKIPPED_LEASE_CONTENDED', ...SAFE }) : lease.value!;
  } catch (e) {
    const error = e instanceof Error ? e.message : 'PM_CALIBRATION_FAILED';
    try { await receipt(started, 'FAILED', 0, 0, {}, error); } catch { /* response still reports failure */ }
    return out({ status: 'FAILED_CLOSED', error, ...SAFE }, 503);
  }
});
