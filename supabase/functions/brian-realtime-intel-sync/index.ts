import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { requireRealtimeInternal } from "../_shared/realtime_internal_auth.ts";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const VERSION="brian.realtime-intel-sync.v2-direct-wire";
const COLLECTOR_ID="brian-realtime-intel-sync-v1";
const STATE_ID="core-intel-sync";
const CORE_INGEST="https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-core-intel-ingest";
const EVENT_SELECT="event_id,asset,event_kind,source_kind,source_id,published_at,first_observed_at,captured_at,claim,direction,magnitude,trust_class,entity_confidence,content_fingerprint,corroboration_key,provenance_uri,pit_verified,raw_capture_id,metadata";

type Json=Record<string,unknown>;
function out(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}})}
function err(e:unknown){return e instanceof Error?`${e.name}: ${e.message}`:String(e)}
async function sha(v:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(v)));return [...d].map(b=>b.toString(16).padStart(2,"0")).join("")}

async function record(startedAt:string,status:string,observed:number,stored:number,errorMessage:string|null=null){
  const finishedAt=new Date().toISOString();
  const runId=await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  await db.from("brian_collector_runs").insert({
    run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:finishedAt,status,
    observed_records:observed,stored_records:stored,degraded_sources:[],
    error_class:errorMessage?"REALTIME_INTEL_SYNC_ERROR":null,error_message:errorMessage,
    metadata:{version:VERSION,target_project:"brian-market-intelligence",cross_project_sync:true},
    evidence_class:"PROSPECTIVE_DEVELOPMENT_SHADOW",shadow_only:true,live_execution:false
  });
}
async function state(){
  const q=await db.from("brian_realtime_source_state").select("last_success_at").eq("endpoint_id",STATE_ID).maybeSingle();
  if(q.error)throw q.error;
  return q.data?.last_success_at?Date.parse(String(q.data.last_success_at)):NaN;
}
async function setState(ok:boolean,status:number,errorMessage:string|null,changed:boolean){
  const now=new Date().toISOString();
  const row:Json={endpoint_id:STATE_ID,last_fetch_at:now,last_http_status:status||null,last_error:errorMessage,updated_at:now};
  if(ok)row.last_success_at=now;
  if(changed)row.last_change_at=now;
  const q=await db.from("brian_realtime_source_state").upsert(row,{onConflict:"endpoint_id"});
  if(q.error)throw q.error;
}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);
  const startedAt=new Date().toISOString();
  let key="";
  try{key=await requireRealtimeInternal(req)}catch{return out({status:"UNAUTHORIZED"},401)}
  try{
    const last=await state();
    const fallback=Date.now()-12*60*60_000;
    const fromMs=Number.isFinite(last)?Math.max(fallback,last-5*60_000):fallback;
    const q=await db.from("brian_intel_events")
      .select(EVENT_SELECT)
      .in("trust_class",["OFFICIAL_PRIMARY","INDEPENDENT_PROFESSIONAL"])
      .gte("first_observed_at",new Date(fromMs).toISOString())
      .order("first_observed_at",{ascending:true})
      .limit(250);
    if(q.error)throw q.error;
    const events=((q.data??[]) as Json[]).filter((row)=>{
      const trust=String(row.trust_class??"");
      const kind=String(row.event_kind??"");
      const meta=(row.metadata&&typeof row.metadata==="object"?row.metadata:{}) as Json;
      return trust==="OFFICIAL_PRIMARY" ||
        (trust==="INDEPENDENT_PROFESSIONAL" && kind==="DIRECT_WIRE_DISCOVERY" && meta.direct_wire===true);
    });

    const r=await fetch(CORE_INGEST,{
      method:"POST",
      headers:{"content-type":"application/json","x-brian-internal-key":key},
      body:JSON.stringify({source:"brian-realtime",events}),
      signal:AbortSignal.timeout(12000)
    });
    const text=await r.text();
    let body:Json={};
    try{body=JSON.parse(text) as Json}catch{body={raw:text.slice(0,800)}}
    if(!r.ok||String(body.status)!=="SUCCESS"){
      const message=`CORE_INGEST_${r.status}:${String(body.error??body.status??"UNKNOWN")}`;
      await setState(false,r.status,message,false);
      await record(startedAt,"FAILED",events.length,0,message);
      return out({status:"FAILED_CLOSED",version:VERSION,error:message,scanned:events.length,shadow_only:true,live_execution:false},502);
    }
    const stored=Number(body.stored??0);
    await setState(true,r.status,null,stored>0);
    await record(startedAt,"SUCCESS",events.length,stored);
    return out({status:"SUCCESS",version:VERSION,from:new Date(fromMs).toISOString(),scanned:events.length,core_accepted:Number(body.accepted??0),core_stored:stored,shadow_only:true,live_execution:false});
  }catch(e){
    const message=err(e).slice(0,1000);
    try{await setState(false,0,message,false)}catch{}
    await record(startedAt,"FAILED",0,0,message);
    return out({status:"FAILED_CLOSED",version:VERSION,error:message,shadow_only:true,live_execution:false},500);
  }
});