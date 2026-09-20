import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const VERSION="brian.core-scheduler-bridge.v1";
const ALLOWED_SHA256=new Set([
  "b0549b2b41a5b832b37455389583e1d166d210490a8c6fe43cda2748aca7c38a",
  "814a5df4f8d6e3b15f1b9ac19a4ea823ad69eedc52caa6ad7573fde7aa96eaab"
]);
const ALLOWED_ACTIONS=new Set([
  "alpha_sync","world","treasury","discovery",
  "official_primary","source_observer","source_registry","meeting_sync",
  "recovery","watchdog","dip","multiasset",
  "universe_heartbeat","sensor_heartbeat","intrabar_heartbeat","derivatives_heartbeat","behavior_heartbeat","fx_heartbeat"
]);

function out(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}})}
async function sha256(v:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(v)));return [...d].map(b=>b.toString(16).padStart(2,"0")).join("")}
function err(e:unknown){return e instanceof Error?e.message:String(e)}
async function auth(req:Request){
  const key=(req.headers.get("x-brian-internal-key")??req.headers.get("x-brian-cron-key")??"").trim();
  if(!key)throw new Error("UNAUTHORIZED");
  const digest=await sha256(key);
  if(!ALLOWED_SHA256.has(digest))throw new Error("UNAUTHORIZED");
}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);
  try{await auth(req)}catch{return out({status:"UNAUTHORIZED"},401)}
  try{
    const body=await req.json().catch(()=>({}));
    const action=String(body?.action??"").trim().toLowerCase();
    if(!ALLOWED_ACTIONS.has(action))return out({status:"INVALID_ACTION",action},400);
    const heartbeatMap:Record<string,string>={
      universe_heartbeat:"market.universe",
      sensor_heartbeat:"market.sensor-mesh",
      intrabar_heartbeat:"market.intrabar",
      derivatives_heartbeat:"market.derivatives",
      behavior_heartbeat:"market.crowd-behavior",
      fx_heartbeat:"world.fx",
    };
    const q=heartbeatMap[action]
      ? await db.rpc("brian_external_capability_heartbeat_v1",{p_capability:heartbeatMap[action]})
      : await db.rpc("brian_scheduler_bridge_v1",{p_action:action});
    if(q.error)throw new Error(q.error.message);
    return out({status:"SUCCESS",version:VERSION,action,result:q.data,shadow_only:true,live_execution:false});
  }catch(e){
    return out({status:"FAILED_CLOSED",version:VERSION,error:err(e),shadow_only:true,live_execution:false},500);
  }
});