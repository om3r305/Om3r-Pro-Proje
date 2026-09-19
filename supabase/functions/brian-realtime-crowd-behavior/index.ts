import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { requireRealtimeInternal } from "../_shared/realtime_internal_auth.ts";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const VERSION="brian.realtime-crowd-behavior.v1";
const COLLECTOR_ID="brian-realtime-crowd-behavior-v1";
const SOURCE_GROUPS=new Set(["micro_velocity","micro_volume","micro_breakout","micro_reclaim","micro_taker_flow","derivatives_oi","derivatives_taker","derivatives_funding"]);

function out(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}})}
function num(v:unknown,d=0){const n=Number(v);return Number.isFinite(n)?n:d}
function clip(v:number,lo=0,hi=1){return Math.max(lo,Math.min(hi,v))}
async function sha(s:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(s)));return [...d].map(b=>b.toString(16).padStart(2,"0")).join("")}
function errorText(e:unknown){return e instanceof Error?`${e.name}: ${e.message}`:String(e)}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);
  try{await requireRealtimeInternal(req)}catch{return out({status:"UNAUTHORIZED"},401)}
  const startedAt=new Date().toISOString();
  try{
    const since=new Date(Date.now()-8*60_000).toISOString();
    const q=await db.from("brian_sensor_observations")
      .select("observation_id,asset_id,observed_at,direction,strength,confidence,reliability,independent_group,sensor_family,available")
      .like("asset_id","crypto:%")
      .gte("observed_at",since)
      .eq("available",true)
      .order("observed_at",{ascending:false})
      .limit(1600);
    if(q.error)throw new Error(`sensor_read:${q.error.message}`);

    const byAsset=new Map<string,any[]>();
    for(const row of q.data??[]){
      const group=String(row.independent_group??"");
      if(!SOURCE_GROUPS.has(group))continue;
      const a=String(row.asset_id??"");
      const arr=byAsset.get(a)??[];
      // newest row per group only
      if(!arr.some(x=>String(x.independent_group)===group))arr.push(row);
      byAsset.set(a,arr);
    }

    const frames:any[]=[];
    for(const [asset,rows] of byAsset){
      if(rows.length<2)continue;
      let score=0,weight=0;
      const pos:string[]=[],neg:string[]=[],ids:string[]=[];
      let velocity=0,taker=0,oi=0,volume=0;
      for(const r of rows){
        const dir=Math.sign(num(r.direction));
        const strength=clip(num(r.strength));
        const confidence=clip(num(r.confidence,.5));
        const reliability=clip(num(r.reliability,.5));
        const w=Math.max(.05,strength)*(.5+.5*confidence)*(.5+.5*reliability);
        score+=dir*w;weight+=w;
        const g=String(r.independent_group);
        if(dir>0)pos.push(g); else if(dir<0)neg.push(g);
        ids.push(String(r.observation_id));
        if(g==="micro_velocity")velocity=dir*strength;
        if(g==="micro_taker_flow")taker=dir*strength;
        if(g==="derivatives_oi")oi=dir*strength;
        if(g==="micro_volume")volume=strength;
      }
      const normalized=weight>0?score/weight:0;
      let state="BALANCED",direction=0;
      if(normalized>=.28){state="CROWD_BID";direction=1}
      if(normalized<=-.28){state="CROWD_OFFER";direction=-1}
      if(normalized>=.55&&velocity>.35&&taker>.25){state="FOMO_CHASE";direction=1}
      if(normalized<=-.55&&velocity<-.35&&taker<-.25){state="PANIC_SELL";direction=-1}
      if(velocity>.3&&taker>.2&&oi<-.15){state="SHORT_SQUEEZE_PROXY";direction=1}
      if(velocity<-.3&&taker<-.2&&oi>.15){state="LONG_SQUEEZE_PROXY";direction=-1}
      const strength=clip(Math.abs(normalized));
      const confidence=clip(.35+.08*Math.min(6,rows.length)+.2*Math.min(1,volume));
      const observedAt=new Date().toISOString();
      const frameId=await sha(`${VERSION}|${asset}|${observedAt.slice(0,16)}|${state}|${normalized.toFixed(4)}`);
      frames.push({
        frame_id:frameId,asset_id:asset,observed_at:observedAt,state,direction,strength,confidence,
        supporting_groups:direction>0?[...new Set(pos)]:direction<0?[...new Set(neg)]:[],
        conflict_groups:direction>0?[...new Set(neg)]:direction<0?[...new Set(pos)]:[...new Set([...pos,...neg])],
        source_observation_ids:[...new Set(ids)].slice(0,30),
        reason:`crowd-behavior proxy from price/taker/volume/derivatives telemetry; state=${state}; normalized=${normalized.toFixed(3)}`,
        metadata:{
          version:VERSION,role:"CONTEXT_AND_RISK_PROXY_NOT_INDEPENDENT_DIRECTION_VOTE",
          normalized_score:normalized,source_group_count:rows.length,
          psychology_semantics:"observable crowd-behavior proxy; no claim about individual mental state",
          velocity_proxy:velocity,taker_proxy:taker,oi_proxy:oi,volume_proxy:volume
        },
        evidence_class:"PROSPECTIVE_DEVELOPMENT_SHADOW",shadow_only:true,live_execution:false
      });
    }
    if(frames.length){
      const ins=await db.from("brian_crowd_behavior_frames").upsert(frames,{onConflict:"frame_id",ignoreDuplicates:true});
      if(ins.error)throw new Error(`persist_frames:${ins.error.message}`);
    }
    const finishedAt=new Date().toISOString();
    const runId=await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|SUCCESS`);
    await db.from("brian_collector_runs").insert({
      run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:finishedAt,status:"SUCCESS",
      observed_records:q.data?.length??0,stored_records:frames.length,degraded_sources:[],
      metadata:{version:VERSION,context_only:true,not_independent_direction_vote:true,shadow_only:true,live_execution:false},
      evidence_class:"PROSPECTIVE_DEVELOPMENT_SHADOW",shadow_only:true,live_execution:false
    });
    return out({status:"SUCCESS",version:VERSION,frames:frames.map(f=>({asset_id:f.asset_id,state:f.state,direction:f.direction,strength:f.strength,confidence:f.confidence})),shadow_only:true,live_execution:false});
  }catch(e){
    const finishedAt=new Date().toISOString();
    const runId=await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|FAILED`);
    await db.from("brian_collector_runs").insert({
      run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:finishedAt,status:"FAILED",
      observed_records:0,stored_records:0,degraded_sources:[],error_class:"CROWD_BEHAVIOR_ERROR",error_message:errorText(e).slice(0,1200),
      metadata:{version:VERSION,shadow_only:true,live_execution:false},evidence_class:"PROSPECTIVE_DEVELOPMENT_SHADOW",shadow_only:true,live_execution:false
    });
    return out({status:"FAILED_CLOSED",error:errorText(e),shadow_only:true,live_execution:false},500);
  }
});