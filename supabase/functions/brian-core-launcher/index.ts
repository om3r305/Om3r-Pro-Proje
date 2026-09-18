import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { requireCronAuth } from "../_shared/cron_auth.ts";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const ANON = Deno.env.get("SUPABASE_ANON_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });

const SERVICES: Record<string,string> = {
  alpha: "brian-alpha-decision-compiler",
  world: "brian-world-brain",
  treasury: "brian-evolution-treasury",
  discovery: "brian-world-discovery-eye",
  evolution: "brian-evolution-orchestrator",
  researcher: "brian-evolution-researcher",
  sandbox: "brian-evolution-sandbox",
  ocean: "brian-evolution-ocean-worker",
};

function out(body: unknown, status=200){
  return new Response(JSON.stringify(body), {
    status,
    headers: {"content-type":"application/json; charset=utf-8","cache-control":"no-store"}
  });
}
function errText(error: unknown){
  if(error instanceof Error) return `${error.name}: ${error.message}`;
  try{return JSON.stringify(error)}catch{return String(error)}
}

async function logLaunch(row: Record<string,unknown>){
  try{ await db.from("brian_core_launch_log").insert(row); }catch{ /* observability must not break dispatch */ }
}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST") return out({error:"POST required"},405);
  try{ await requireCronAuth(req,db); }
  catch(error){ return out({status:"UNAUTHORIZED",error:errText(error)},401); }

  const body=await req.json().catch(()=>({}));
  const service=String(body?.service??"").trim();
  const endpoint=SERVICES[service];
  if(!endpoint) return out({status:"UNKNOWN_SERVICE",service},400);
  const shardCount=service==="alpha"?Math.max(1,Math.min(5,Math.floor(Number(body?.shard_count??1)))):1;
  const shardIndex=service==="alpha"?Math.max(0,Math.min(shardCount-1,Math.floor(Number(body?.shard_index??0)))):0;
  const targetBody=service==="alpha"
    ? {shard_index:shardIndex,shard_count:shardCount}
    : {};

  const downstreamKey=(req.headers.get("x-brian-downstream-key")??req.headers.get("x-brian-cron-key")??"").trim();
  if(!downstreamKey) return out({status:"DOWNSTREAM_KEY_MISSING"},500);

  const launchId=crypto.randomUUID();
  const launchedAt=new Date().toISOString();

  const task=(async()=>{
    let httpStatus:number|null=null;
    let targetStatus:string|null=null;
    let errorText:string|null=null;
    try{
      const response=await fetch(`${URL}/functions/v1/${endpoint}`,{
        method:"POST",
        headers:{
          "content-type":"application/json",
          "authorization":`Bearer ${ANON}`,
          "apikey":ANON,
          "x-brian-cron-key":downstreamKey,
        },
        body:JSON.stringify(targetBody),
      });
      httpStatus=response.status;
      const payload=await response.json().catch(()=>({}));
      targetStatus=String(payload?.status??payload?.run_quality??"").slice(0,120)||null;
      if(!response.ok) errorText=JSON.stringify(payload).slice(0,1800);
    }catch(error){
      errorText=errText(error).slice(0,1800);
    }
    await logLaunch({
      launch_id:launchId,service:service==="alpha"?`alpha:${shardIndex}/${shardCount}`:service,endpoint,launched_at:launchedAt,finished_at:new Date().toISOString(),
      http_status:httpStatus,target_status:targetStatus,error_text:errorText
    });
  })();

  EdgeRuntime.waitUntil(task);
  return out({
    status:"ACCEPTED",
    launch_id:launchId,
    service,
    endpoint,
    shard_index:service==="alpha"?shardIndex:null,
    shard_count:service==="alpha"?shardCount:null,
    launched_at:launchedAt,
    background:true,
    shadow_only:true,
    live_execution:false
  },202);
});
