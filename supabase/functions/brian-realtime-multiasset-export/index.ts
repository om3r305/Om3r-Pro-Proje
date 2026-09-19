import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const VERSION="brian.realtime-multiasset-export.v1";
const ALLOWED_SHA256=new Set([
  "b0549b2b41a5b832b37455389583e1d166d210490a8c6fe43cda2748aca7c38a",
  "814a5df4f8d6e3b15f1b9ac19a4ea823ad69eedc52caa6ad7573fde7aa96eaab"
]);
async function sha256Hex(v:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(v)));return [...d].map(b=>b.toString(16).padStart(2,"0")).join("")}
async function auth(req:Request){const k=(req.headers.get("x-brian-internal-key")??req.headers.get("x-brian-cron-key")??"").trim();if(!k)throw new Error("UNAUTHORIZED");if(!ALLOWED_SHA256.has(await sha256Hex(k)))throw new Error("UNAUTHORIZED")}
function out(b:unknown,s=200){return new Response(JSON.stringify(b),{status:s,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}})}
Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);
  try{await auth(req)}catch{return out({status:"UNAUTHORIZED"},401)}
  const marks=await db.from("brian_multiasset_market_latest")
    .select("asset_id,asset_class,provider_symbol,provider,provider_quality,observed_at,provider_time,price,open_price,high_price,low_price,previous_close,volume,return_5m,return_1h,session_state,data_latency_seconds,metadata")
    .order("asset_id",{ascending:true});
  if(marks.error)return out({status:"FAILED_CLOSED",error:marks.error.message},500);
  const crowd=await db.from("brian_crowd_behavior_frames")
    .select("asset_id,observed_at,state,direction,strength,confidence,reason,metadata")
    .gte("observed_at",new Date(Date.now()-15*60_000).toISOString())
    .order("observed_at",{ascending:false}).limit(200);
  if(crowd.error)return out({status:"FAILED_CLOSED",error:crowd.error.message},500);
  const latestCrowd=new Map<string,unknown>();
  for(const row of crowd.data??[])if(!latestCrowd.has(String(row.asset_id)))latestCrowd.set(String(row.asset_id),row);
  return out({
    status:"SUCCESS",version:VERSION,observed_at:new Date().toISOString(),
    marks:marks.data??[],crowd:[...latestCrowd.values()],
    shadow_only:true,live_execution:false
  });
});