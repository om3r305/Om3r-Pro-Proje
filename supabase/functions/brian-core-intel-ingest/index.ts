import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { requireRealtimeInternal } from "../_shared/realtime_internal_auth.ts";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const VERSION="brian.core-intel-ingest.v1";
const COLLECTOR_ID="brian-core-intel-ingest-v1";

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
    error_class:errorMessage?"CORE_INTEL_INGEST_ERROR":null,error_message:errorMessage,
    metadata:{version:VERSION,source_project:"brian-realtime",cross_project_sync:true},
    evidence_class:"PROSPECTIVE_DEVELOPMENT_SHADOW",shadow_only:true,live_execution:false
  });
}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);
  const startedAt=new Date().toISOString();
  try{await requireRealtimeInternal(req)}catch{return out({status:"UNAUTHORIZED"},401)}
  try{
    const body=await req.json().catch(()=>({})) as Json;
    const raw=Array.isArray(body.events)?body.events.slice(0,250):[];
    const accepted:Json[]=[];
    for(const value of raw){
      if(!value||typeof value!=="object")continue;
      const row=value as Json;
      const eventId=String(row.event_id??"").trim();
      const eventKind=String(row.event_kind??"").trim();
      const trust=String(row.trust_class??"").trim();
      const claim=String(row.claim??"").trim();
      if(!eventId||!claim||trust!=="OFFICIAL_PRIMARY"||!/^OFFICIAL_/i.test(eventKind))continue;
      const meta=(row.metadata&&typeof row.metadata==="object"?row.metadata:{}) as Json;
      accepted.push({
        event_id:eventId,
        asset:row.asset??"GLOBAL",
        event_kind:eventKind,
        source_kind:row.source_kind??"REALTIME_OFFICIAL",
        source_id:row.source_id??"unknown",
        published_at:row.published_at??null,
        first_observed_at:row.first_observed_at??startedAt,
        captured_at:row.captured_at??startedAt,
        claim:claim.slice(0,2000),
        direction:Number(row.direction??0),
        magnitude:Number(row.magnitude??0),
        trust_class:"OFFICIAL_PRIMARY",
        entity_confidence:Number(row.entity_confidence??1),
        content_fingerprint:row.content_fingerprint??null,
        corroboration_key:row.corroboration_key??null,
        provenance_uri:row.provenance_uri??null,
        pit_verified:row.pit_verified!==false,
        raw_capture_id:null,
        metadata:{
          ...meta,
          realtime_raw_capture_id:row.raw_capture_id??null,
          cross_project_sync:true,
          synced_from:"brian-realtime",
          synced_at:startedAt,
          direct_alpha_influence:false
        }
      });
    }

    let stored=0;
    if(accepted.length){
      const q=await db.from("brian_intel_events")
        .upsert(accepted,{onConflict:"event_id",ignoreDuplicates:true})
        .select("event_id");
      if(q.error)throw q.error;
      stored=Array.isArray(q.data)?q.data.length:0;
    }
    await record(startedAt,"SUCCESS",accepted.length,stored);
    return out({status:"SUCCESS",version:VERSION,accepted:accepted.length,stored,shadow_only:true,live_execution:false});
  }catch(e){
    const message=err(e).slice(0,1000);
    await record(startedAt,"FAILED",0,0,message);
    return out({status:"FAILED_CLOSED",version:VERSION,error:message,shadow_only:true,live_execution:false},500);
  }
});