import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { LIVE_MAX_AGE_SECONDS, markFreshness } from "../_shared/market_mark_freshness.ts";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });

const ALLOWED_ORIGIN = /^https:\/\/monster-coins(?:-pro)?-[a-z0-9-]*oemer-yildirim\.vercel\.app$/i;
const ALLOWED_EXACT = new Set([
  "https://monster-coins-pro-seven.vercel.app",
  "https://monster-coins-pro-oemer-yildirim.vercel.app",
  "https://monster-coins-pro-git-brian-2026-oemer-yildirim.vercel.app",
  "http://localhost:3000",
  "http://127.0.0.1:3000",
]);

function cors(origin?: string | null): Record<string,string> {
  const allowed = origin && (ALLOWED_EXACT.has(origin) || ALLOWED_ORIGIN.test(origin)) ? origin : "*";
  return {
    "access-control-allow-origin": allowed,
    "access-control-allow-headers": "content-type",
    "access-control-allow-methods": "POST,OPTIONS",
    "cache-control": "no-store",
    "vary": "Origin",
  };
}
function out(body:unknown,status=200,origin?:string|null){
  return new Response(JSON.stringify(body),{
    status,
    headers:{"content-type":"application/json; charset=utf-8",...cors(origin)}
  });
}
function validAssetId(value: unknown): string | null {
  const id = String(value ?? "").trim();
  return /^(crypto:[A-Z0-9]{2,20}USDT|fx:[A-Z0-9]{6,12}|index:[A-Z0-9]{2,24}|commodity:[A-Z0-9]{2,24}|equity:[A-Z0-9.-]{1,16}|etf:[A-Z0-9.-]{1,16})$/.test(id) ? id : null;
}

Deno.serve(async(req:Request)=>{
  const origin=req.headers.get("origin");
  if(req.method==="OPTIONS")return new Response(null,{status:204,headers:cors(origin)});
  if(req.method!=="POST")return out({error:"POST required"},405,origin);

  const body=await req.json().catch(()=>({})) as Record<string,unknown>;
  const raw=Array.isArray(body.asset_ids)?body.asset_ids:[];
  const assetIds=[...new Set(raw.map(validAssetId).filter((v):v is string=>Boolean(v)))].slice(0,20);
  if(!assetIds.length)return out({status:"SUCCESS",observed_at:new Date().toISOString(),marks:[],count:0,source:"brian-realtime"},200,origin);

  const cryptoIds=assetIds.filter(id=>id.startsWith("crypto:"));
  const multiIds=assetIds.filter(id=>!id.startsWith("crypto:"));
  const latest=new Map<string,Record<string,unknown>>();

  if(cryptoIds.length){
    const since=new Date(Date.now()-10*60_000).toISOString();
    const q=await db.from("brian_micro_book_ticks")
      .select("asset_id,observed_at,observed_mid_price")
      .in("asset_id",cryptoIds)
      .gte("observed_at",since)
      .order("observed_at",{ascending:false})
      .limit(Math.min(400,Math.max(60,cryptoIds.length*30)));
    if(q.error)return out({status:"FAILED_CLOSED",error:"CRYPTO_MARK_READ_FAILED",marks:[]},503,origin);
    for(const row of q.data??[]){
      const id=String(row.asset_id??"");
      const price=Number(row.observed_mid_price);
      if(!latest.has(id)&&Number.isFinite(price)&&price>0){
        const fresh=markFreshness(row.observed_at,Date.now());
        latest.set(id,{asset_id:id,observed_at:String(row.observed_at),price,session_state:"CONTINUOUS",age_seconds:fresh.age_seconds,live:fresh.live,source:"brian_micro_book_ticks"});
      }
    }
  }

  if(multiIds.length){
    const q=await db.from("brian_multiasset_market_latest")
      .select("asset_id,provider_time,price,session_state,data_latency_seconds,provider_quality")
      .in("asset_id",multiIds);
    if(q.error)return out({status:"FAILED_CLOSED",error:"MULTIASSET_MARK_READ_FAILED",marks:[]},503,origin);
    for(const row of q.data??[]){
      const id=String(row.asset_id??"");
      const price=Number(row.price);
      const latency=Number(row.data_latency_seconds);
      const session=String(row.session_state??"UNKNOWN").toUpperCase();
      if(!latest.has(id)&&Number.isFinite(price)&&price>0){
        latest.set(id,{
          asset_id:id,observed_at:String(row.provider_time),price,session_state:session,
          data_latency_seconds:Number.isFinite(latency)?latency:null,
          provider_quality:row.provider_quality,
          // data_latency_seconds is the delay measured when the row was written; liveness uses the age now.
          ...markFreshness(row.provider_time,Date.now(),session==="REGULAR"),
          source:"brian_multiasset_market_latest"
        });
      }
    }
  }

  const marks=assetIds.map(id=>latest.get(id)).filter(Boolean);
  return out({
    status:"SUCCESS",
    observed_at:new Date().toISOString(),
    marks,
    count:marks.length,
    source:"brian-realtime",
    live_max_age_seconds:LIVE_MAX_AGE_SECONDS,
    read_only:true,
    shadow_only:true,
    live_execution:false,
  },200,origin);
});
