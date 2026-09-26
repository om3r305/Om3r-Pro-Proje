import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { requireRealtimeInternal } from "../_shared/realtime_internal_auth.ts";

const VERSION="brian.realtime-core-scheduler.v12-async-core-dispatch";
const RT_URL=Deno.env.get("SUPABASE_URL")!;
const RT_SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const rtDb=createClient(RT_URL,RT_SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});

type Result={action:string;ok:boolean;http_status:number;target_status:string;elapsed_ms:number;body?:unknown;error?:string};

function out(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}})}
function err(e:unknown){return e instanceof Error?`${e.name}: ${e.message}`:String(e)}

async function enqueueCoreAction(action:string):Promise<Result>{
  const started=Date.now();
  try{
    const q=await rtDb.rpc("brian_realtime_enqueue_core_action_v1",{p_action:action});
    if(q.error){
      return {action,ok:false,http_status:0,target_status:"ENQUEUE_FAILED",elapsed_ms:Date.now()-started,error:q.error.message.slice(0,1000)};
    }
    const body=q.data&&typeof q.data==="object"&&!Array.isArray(q.data)
      ? q.data as Record<string,unknown>
      : {raw:q.data};
    const target=String(body.status??"");
    const ok=["QUEUED","SKIPPED_BUSY","SUCCESS"].includes(target);
    return {
      action,
      ok,
      http_status:target==="QUEUED"?202:200,
      target_status:target||"UNKNOWN",
      elapsed_ms:Date.now()-started,
      body
    };
  }catch(e){
    return {action,ok:false,http_status:0,target_status:"ENQUEUE_FAILED",elapsed_ms:Date.now()-started,error:err(e).slice(0,1000)};
  }
}

async function runAction(action:string,key:string,timeoutMs=12000):Promise<Result>{
  if(!isLocalAction(action)) return enqueueCoreAction(action);

  const started=Date.now();
  try{
    const targetUrl=action==="direct_wire"
      ? RT_URL+"/functions/v1/brian-direct-wire-eye"
      : action==="readiness_cost"
        ? RT_URL+"/functions/v1/brian-realtime-readiness-cost-sampler"
        : RT_URL+"/functions/v1/brian-realtime-archive";
    const r=await fetch(targetUrl,{
      method:"POST",
      headers:{"content-type":"application/json","x-brian-internal-key":key},
      body:JSON.stringify({action}),
      signal:AbortSignal.timeout(timeoutMs)
    });
    const text=await r.text();
    let body:Record<string,unknown>={};
    try{
      const parsed:unknown=JSON.parse(text);
      body=parsed&&typeof parsed==="object"&&!Array.isArray(parsed)
        ? parsed as Record<string,unknown>
        : {raw:text.slice(0,1000)};
    }catch{body={raw:text.slice(0,1000)}}
    const nestedBody=body.result&&typeof body.result==="object"&&!Array.isArray(body.result)
      ? body.result as Record<string,unknown>
      : {};
    const target=String(body.status??"");
    const nested=String(nestedBody.status??"");
    const ok=r.ok&&![target,nested].some(s=>["FAILED","FAILED_CLOSED","UNAUTHORIZED","INVALID_ACTION","DEGRADED"].includes(s));
    return {action,ok,http_status:r.status,target_status:target,elapsed_ms:Date.now()-started,body};
  }catch(e){
    return {action,ok:false,http_status:0,target_status:"FETCH_FAILED",elapsed_ms:Date.now()-started,error:err(e).slice(0,1000)};
  }
}

function includes(minute:number,values:number[]){return values.includes(minute)}
function isLocalAction(action:string){
  return action==="direct_wire" || action==="readiness_cost" || action==="archive";
}
async function tableFresh(table:string,maxAgeMs:number){
  const q=await rtDb.from(table).select("observed_at").order("observed_at",{ascending:false}).limit(1).maybeSingle();
  if(q.error||!q.data)return false;
  const stamp=Date.parse(String(q.data.observed_at??""));
  return Number.isFinite(stamp) && Date.now()-stamp<=maxAgeMs;
}
async function collectorFresh(collectorId:string,maxAgeMs:number){
  const q=await rtDb.from("brian_collector_runs")
    .select("status,finished_at,started_at")
    .eq("collector_id",collectorId)
    .order("started_at",{ascending:false})
    .limit(1)
    .maybeSingle();
  if(q.error||!q.data)return false;
  const stamp=Date.parse(String(q.data.finished_at??q.data.started_at??""));
  return String(q.data.status).toUpperCase()==="SUCCESS" && Number.isFinite(stamp) && Date.now()-stamp<=maxAgeMs;
}

function coreActionForMinute(minute:number){
  const m=((minute%60)+60)%60;
  if([1,6,11,16,21,26,31,36,41,46,51,56].includes(m)) return "alpha_sync";
  if([2,12,22,32,42,52].includes(m)) return "treasury";
  if([3,13,23,33,43,53].includes(m)) return "official_primary";
  if([4,14,24,34,44,54].includes(m)) return "world";
  if([5,15,25,35,45,55].includes(m)) return "multiasset";
  if([7,17,27,37,47,57].includes(m)) return "source_observer";
  if([8,18,28,38,48,58].includes(m)) return "recovery";
  if([9,29,49].includes(m)) return "missed_auditor";
  if([19,39,59].includes(m)) return "discovery";
  if([0,30].includes(m)) return "source_registry";
  if([10,40].includes(m)) return "watchdog";
  if([20,50].includes(m)) return "meeting_sync";
  return null;
}

function planned(minute:number){
  const actions:string[]=[];
  if(minute%2===0) actions.push("direct_wire");
  if(minute%3===0) actions.push("readiness_cost");
  if(minute%10===4) actions.push("archive");
  if(minute%3===2) actions.push("dip");

  // At most one stateful Core lane is launched per scheduler minute.
  // This prevents slow Core RPCs from piling up behind each other on small compute.
  const coreAction=coreActionForMinute(minute);
  if(coreAction) actions.push(coreAction);
  return [...new Set(actions)];
}

async function handle(req:Request){
  if(req.method==="GET")return out({status:"OK",version:VERSION,role:"REALTIME_SCHEDULER_FOR_CORE",shadow_only:true,live_execution:false});
  if(req.method!=="POST")return out({error:"POST required"},405);

  let key="";
  try{key=await requireRealtimeInternal(req)}catch{return out({status:"UNAUTHORIZED"},401)}

  const now=new Date();
  const minute=now.getUTCMinutes();
  const actions=planned(minute);
  if(minute%5===2){
    const [u,s,i,d,b]=await Promise.all([
      tableFresh("brian_universe_snapshots",12*60_000),
      tableFresh("brian_sensor_observations",12*60_000),
      tableFresh("brian_intrabar_reaction_events",6*60_000),
      collectorFresh("phase39-binance-usdm-derivatives",12*60_000),
      tableFresh("brian_crowd_behavior_frames",12*60_000),
    ]);
    if(u) actions.push("universe_heartbeat");
    if(s) actions.push("sensor_heartbeat");
    if(i) actions.push("intrabar_heartbeat");
    if(d) actions.push("derivatives_heartbeat");
    if(b) actions.push("behavior_heartbeat");
  }
  if(minute%15===7 && await collectorFresh("phase39-ecb-fx",90*60_000)){
    actions.push("fx_heartbeat");
  }
  if(minute%5===2 && await collectorFresh("brian-direct-wire-eye-v1",6*60_000)){
    actions.push("direct_wire_heartbeat");
  }
  const results:Result[]=[];
  let marketCircuitOpen=false;

  for(const action of [...new Set(actions)]){
    if(!isLocalAction(action) && marketCircuitOpen){
      results.push({
        action,
        ok:false,
        http_status:0,
        target_status:"SKIPPED_MARKET_CIRCUIT_OPEN",
        elapsed_ms:0
      });
      continue;
    }

    const result=await runAction(
      action,
      key,
      action==="archive" ? 20000
        : action==="recovery"||action==="watchdog" ? 12000
        : isLocalAction(action) ? 12000
        : 6000
    );
    results.push(result);

    if(!isLocalAction(action) && !result.ok){
      marketCircuitOpen=true;
    }
  }

  const failed=results.filter(r=>!r.ok);
  return out({
    status:failed.length?"DEGRADED":"SUCCESS",
    version:VERSION,
    observed_at:now.toISOString(),
    utc_minute:minute,
    actions,
    results,
    circuit_breaker:failed.length?{active:true,failed_actions:failed.map(x=>x.action)}:{active:false},
    scheduler_project:"brian-realtime",
    target_project:"brian-market-intelligence",
    shadow_only:true,
    live_execution:false
  },failed.length?207:200);
 }
Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return handle(req);
  try{await requireRealtimeInternal(req);}catch{return out({status:"UNAUTHORIZED"},401);}
  const token=crypto.randomUUID();
  const lock=await rtDb.rpc("brian_realtime_acquire_lease",{p_job:"core-scheduler",p_token:token});
  if(lock.error)return out({status:"FAILED_CLOSED",error:lock.error.message},503);
  if(!lock.data)return out({status:"SKIPPED_BUSY",version:VERSION});
  try{return await handle(req);}
  finally{await rtDb.from("brian_realtime_job_leases").update({expires_at:new Date().toISOString()}).eq("job","core-scheduler").eq("token",token);}
});
