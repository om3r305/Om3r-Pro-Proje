import { createClient } from "npm:@supabase/supabase-js@2";
import { gzip } from "npm:pako@2.1.0";
import { withCollectorLease } from "../_shared/collector_lease.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, { auth: { persistSession: false, autoRefreshToken: false } });

const COLLECTOR_ID = "phase39-binance-usdm-derivatives";
const EVIDENCE = "PROSPECTIVE_DEVELOPMENT_SHADOW";
const BUCKET = "brian-intelligence-raw";
const TOP_N = 15;
const LEASE_SECONDS = 240;

type Candidate = { symbol?: string };
type Obs = Record<string, unknown>;

function out(body: unknown, status = 200) { return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", "cache-control": "no-store" } }); }
function finite(v: unknown, fallback = 0) { const n = Number(v); return Number.isFinite(n) ? n : fallback; }
function clip(v: number) { return Math.max(0, Math.min(1, v)); }
function sign(v: number) { return v > 0 ? 1 : v < 0 ? -1 : 0; }
function bytes(v: string) { return new TextEncoder().encode(v); }
async function sha(v: string | Uint8Array) { const b = typeof v === "string" ? bytes(v) : v; const d = new Uint8Array(await crypto.subtle.digest("SHA-256", b)); return [...d].map(x => x.toString(16).padStart(2,"0")).join(""); }
async function getJson(url: string) { const r = await fetch(url, { headers: { accept: "application/json", "user-agent": "Brian-2026-Derivatives-Eye/1.0" }, signal: AbortSignal.timeout(8000) }); if (!r.ok) throw new Error(`${r.status} ${url}`); return await r.json(); }
async function mapLimit<T,R>(items: T[], limit: number, fn: (v:T)=>Promise<R>): Promise<R[]> { const result = new Array<R>(items.length); let i=0; async function worker(){ while(true){ const j=i++; if(j>=items.length) return; result[j]=await fn(items[j]); }} await Promise.all(Array.from({length:Math.min(limit,items.length)},()=>worker())); return result; }

async function rawCapture(payload: unknown, observedAt: string) {
  const canonical = JSON.stringify(payload); const raw = bytes(canonical); const hash = await sha(raw); const zipped = gzip(raw, { level: 6 });
  const path = `binance_public/phase39_derivatives/${observedAt.slice(0,10)}/${hash}.json.gz`;
  const up = await supabase.storage.from(BUCKET).upload(path, zipped, { contentType: "application/gzip", upsert: false, cacheControl: "31536000" });
  if (up.error && !String(up.error.message).toLowerCase().match(/exist|duplicate/)) throw up.error;
  const id = await sha(`binance_public|phase39_derivatives|${observedAt}|${hash}`);
  const ins = await supabase.from("brian_raw_captures").insert({ capture_id:id, provider:"binance_usdm_public", record_type:"phase39_derivatives", observed_at:observedAt, captured_at:new Date().toISOString(), provenance_uri:"https://fapi.binance.com", payload_hash:hash, payload:{storage_bucket:BUCKET,storage_path:path,content_type:"application/json",content_encoding:"gzip",compressed_byte_length:zipped.byteLength,uncompressed_byte_length:raw.byteLength} });
  if (ins.error) throw ins.error; return id;
}

async function recordRun(startedAt:string, status:string, observed:number, stored:number, degraded:string[], error?:unknown) {
  const finished = new Date().toISOString(); const runId = await sha(`${COLLECTOR_ID}|${startedAt}|${finished}|${status}`);
  await supabase.from("brian_collector_runs").insert({ run_id:runId, collector_id:COLLECTOR_ID, started_at:startedAt, finished_at:finished, status, observed_records:observed, stored_records:stored, degraded_sources:degraded, error_class:error ? "COLLECTOR_ERROR" : null, error_message:error ? String(error).slice(0,1000) : null, evidence_class:EVIDENCE, shadow_only:true, live_execution:false });
}

Deno.serve(async (req) => {
  if (req.method !== "POST") return out({error:"POST required"},405);
  const startedAt = new Date().toISOString();
  try {
    const lease = await withCollectorLease(supabase, COLLECTOR_ID, LEASE_SECONDS, async () => {
    const radarResp = await supabase.from("brian_universe_snapshots").select("snapshot_id,candidates").order("observed_at",{ascending:false}).limit(1).maybeSingle();
    if (radarResp.error || !radarResp.data) throw radarResp.error ?? new Error("universe radar unavailable");
    const envelope = radarResp.data.candidates as Record<string,unknown>; const rows = Array.isArray(envelope?.candidates) ? envelope.candidates as Candidate[] : [];
    const symbols = rows.map(x=>String(x.symbol??"")).filter(x=>x.endsWith("USDT")).slice(0,TOP_N);
    const premiumPayload = await getJson("https://fapi.binance.com/fapi/v1/premiumIndex");
    const premiumMap = new Map<string,Record<string,unknown>>(); if (Array.isArray(premiumPayload)) for (const r of premiumPayload) if (r && typeof r === "object") premiumMap.set(String((r as Record<string,unknown>).symbol??""), r as Record<string,unknown>);
    const degraded:string[]=[];
    const market = await mapLimit(symbols,6,async symbol=>{
      try {
        const [oi, taker, klines] = await Promise.all([
          getJson(`https://fapi.binance.com/futures/data/openInterestHist?symbol=${encodeURIComponent(symbol)}&period=5m&limit=3`),
          getJson(`https://fapi.binance.com/futures/data/takerlongshortRatio?symbol=${encodeURIComponent(symbol)}&period=5m&limit=2`),
          getJson(`https://fapi.binance.com/fapi/v1/klines?symbol=${encodeURIComponent(symbol)}&interval=5m&limit=3`),
        ]);
        return {symbol,premium:premiumMap.get(symbol)??null,oi,taker,klines,error:null};
      } catch (e) { degraded.push(symbol); return {symbol,premium:premiumMap.get(symbol)??null,oi:null,taker:null,klines:null,error:String(e)}; }
    });
    const observedAt = new Date().toISOString(); const captureId = await rawCapture({schema_version:"brian.phase39.derivatives.v1",observed_at:observedAt,universe_snapshot_id:radarResp.data.snapshot_id,market}, observedAt);
    const observations:Obs[]=[];
    for (const row of market) {
      if (row.error) continue; const assetId=`crypto:${row.symbol}`;
      const premium = row.premium as Record<string,unknown>|null; const funding=finite(premium?.lastFundingRate); const basis = premium ? (finite(premium.markPrice)/Math.max(finite(premium.indexPrice),1e-12)-1) : 0;
      if (Math.abs(funding)>=0.0002) {
        const direction=-sign(funding); const strength=clip(Math.abs(funding)/0.001); const eye=await sha(`funding-crowding|${assetId}`); observations.push(await makeObs(eye,"funding-crowding",assetId,"funding_crowding","derivatives_funding",observedAt,direction,strength,0.58,captureId,`contrarian funding crowding rate=${funding}`,{funding_rate:funding,basis}));
      }
      const oi = Array.isArray(row.oi) ? row.oi as Record<string,unknown>[] : []; const k = Array.isArray(row.klines) ? row.klines as unknown[][] : [];
      if (oi.length>=2 && k.length>=2) {
        const a=finite(oi.at(-2)?.sumOpenInterestValue), b=finite(oi.at(-1)?.sumOpenInterestValue); const p0=finite(k.at(-2)?.[4]), p1=finite(k.at(-1)?.[4]);
        if (a>0 && b>0 && p0>0 && p1>0) { const oiCh=Math.log(b/a); const pr=Math.log(p1/p0); if(Math.abs(oiCh)>=0.003 && Math.abs(pr)>=0.001){ const direction=sign(pr); const strength=clip((Math.abs(oiCh)/0.02+Math.abs(pr)/0.01)/2); const eye=await sha(`oi-price-confirmation|${assetId}`); observations.push(await makeObs(eye,"oi-price-confirmation",assetId,"open_interest","derivatives_oi",observedAt,direction,strength,0.65,captureId,`OI and price expanded together`,{oi_change:oiCh,price_return:pr})); }}
      }
      const taker = Array.isArray(row.taker) ? row.taker as Record<string,unknown>[] : []; const ratio=finite(taker.at(-1)?.buySellRatio,1);
      if (ratio>=1.08 || ratio<=0.925) { const direction=ratio>1?1:-1; const strength=clip(Math.abs(Math.log(Math.max(ratio,1e-12)))/0.35); const eye=await sha(`taker-imbalance|${assetId}`); observations.push(await makeObs(eye,"taker-imbalance",assetId,"taker_flow","derivatives_taker",observedAt,direction,strength,0.62,captureId,`public taker buy/sell imbalance ratio=${ratio}`,{buy_sell_ratio:ratio})); }
    }
    if (observations.length) { const ins=await supabase.from("brian_sensor_observations").insert(observations); if(ins.error) throw ins.error; }
    await recordRun(startedAt,degraded.length?"DEGRADED":"SUCCESS",market.length,observations.length,degraded);
      return out({status:degraded.length?"DEGRADED":"CAPTURED",symbols:market.length,stored_observations:observations.length,degraded_sources:degraded,learning_enabled:false,live_execution:false,shadow_only:true});
    });
    // Contended: another invocation already owns this collector's lease. No collector work has
    // run and no data has been written -- see supabase/functions/_shared/collector_lease.ts.
    if (lease.contended) return out({status:"SKIPPED_LEASE_CONTENDED",collector_id:COLLECTOR_ID,shadow_only:true,live_execution:false});
    return lease.value!;
  } catch(e) { await recordRun(startedAt,"FAILED",0,0,[],e); return out({status:"FAILED",error:String(e),shadow_only:true},500); }
});

async function makeObs(eyeId:string, templateId:string, assetId:string, family:string, group:string, observedAt:string, direction:number, strength:number, confidence:number, captureId:string, reason:string, metadata:Record<string,unknown>) {
  const observationId=await sha(`${eyeId}|${observedAt}|${direction}|${strength.toFixed(12)}`);
  return {observation_id:observationId,eye_id:eyeId,template_id:templateId,asset_id:assetId,market_domain:"crypto",sensor_family:family,horizon:"FAST_5_30M",independent_group:group,observed_at:observedAt,direction,strength:clip(strength),confidence:clip(confidence),reliability:0.5,available:true,source_ids:[captureId],reason,evidence_class:EVIDENCE,shadow_only:true,live_execution:false,metadata};
}
