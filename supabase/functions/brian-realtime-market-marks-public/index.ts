import { createClient } from "npm:@supabase/supabase-js@2.116.0";

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
  return /^crypto:[A-Z0-9]{2,20}USDT$/.test(id) ? id : null;
}

Deno.serve(async(req:Request)=>{
  const origin=req.headers.get("origin");
  if(req.method==="OPTIONS")return new Response(null,{status:204,headers:cors(origin)});
  if(req.method!=="POST")return out({error:"POST required"},405,origin);

  const body=await req.json().catch(()=>({})) as Record<string,unknown>;
  const raw=Array.isArray(body.asset_ids)?body.asset_ids:[];
  const assetIds=[...new Set(raw.map(validAssetId).filter((v):v is string=>Boolean(v)))].slice(0,20);
  if(!assetIds.length)return out({status:"SUCCESS",observed_at:new Date().toISOString(),marks:[],count:0,source:"brian-realtime"},200,origin);

  const since=new Date(Date.now()-10*60_000).toISOString();
  const q=await db.from("brian_micro_book_ticks")
    .select("asset_id,observed_at,observed_mid_price")
    .in("asset_id",assetIds)
    .gte("observed_at",since)
    .order("observed_at",{ascending:false})
    .limit(Math.min(400,Math.max(60,assetIds.length*30)));

  if(q.error)return out({status:"FAILED_CLOSED",error:"MARK_READ_FAILED",marks:[]},503,origin);

  const latest=new Map<string,{asset_id:string;observed_at:string;price:number}>();
  for(const row of q.data??[]){
    const id=String(row.asset_id??"");
    const price=Number(row.observed_mid_price);
    if(!latest.has(id)&&Number.isFinite(price)&&price>0){
      latest.set(id,{asset_id:id,observed_at:String(row.observed_at),price});
    }
  }

  const marks=assetIds.map(id=>latest.get(id)).filter(Boolean);
  return out({
    status:"SUCCESS",
    observed_at:new Date().toISOString(),
    marks,
    count:marks.length,
    source:"brian-realtime",
    read_only:true,
    shadow_only:true,
    live_execution:false,
  },200,origin);
});
