import type { SupabaseClient } from "npm:@supabase/supabase-js@2.116.0";
import { evaluatePath, type J } from "../_shared/dip_v84_contract.ts";
import {
  CALIBRATION_FAMILY_ID,
  LOGIC_HASH,
  METRIC_VERSION,
  POLICY_VERSION,
  RELEASE_ID,
  RESOLVER_VERSION,
  STRATEGY_MANIFEST_HASH,
  SYMBOL,
} from "../_shared/dip_v84_authority_contract.ts";
import { pricePath } from "../brian-dip-v84-worker/market.ts";

async function assertAuthorityReleaseSealed(db:SupabaseClient):Promise<void>{
  const q=await db.from("brian_dip_v84_releases").select("status,logic_hash,strategy_manifest_hash,calibration_family_id,manifest").eq("release_id",RELEASE_ID).maybeSingle();
  if(q.error||!q.data)throw Error("V841_RELEASE_REGISTRY_UNAVAILABLE");
  if(q.data.status!=="SEALED"||q.data.logic_hash!==LOGIC_HASH||q.data.strategy_manifest_hash!==STRATEGY_MANIFEST_HASH||q.data.calibration_family_id!==CALIBRATION_FAMILY_ID)throw Error("V841_RELEASE_REGISTRY_MISMATCH");
  if(q.data.manifest?.shadow_only!==true||q.data.manifest?.live_execution!==false||q.data.manifest?.browser_execution!==false||Number(q.data.manifest?.max_shadow_leverage)!==1)throw Error("V841_RELEASE_SAFETY_MISMATCH");
}

export async function resolvePending(db:SupabaseClient,assertOwned:()=>void,fetcher:typeof fetch=fetch):Promise<J>{
  await assertAuthorityReleaseSealed(db);
  const began=Date.now();
  const q=await db.from("brian_dip_v84_decisions").select("occurrence_id,decision_at,signal_at,due_at,direction,target_price,invalidation_price,entry_price,checked_until,last_price").eq("release_id",RELEASE_ID).eq("policy_version",POLICY_VERSION).eq("metric_version",METRIC_VERSION).not("target_price","is",null).not("invalidation_price","is",null).is("resolved_at",null).order("checked_until",{ascending:true,nullsFirst:true}).order("decision_at",{ascending:true}).limit(24);
  if(q.error)throw Error("V841_FORECAST_READ_FAILED:"+q.error.message);
  let resolved=0,progressed=0,deferred=0;
  for(const row of q.data||[]){
    if(Date.now()-began>32_000){deferred++;break;}
    assertOwned();
    const start=Date.parse(row.checked_until||row.decision_at),due=Date.parse(row.due_at),entry=Number(row.last_price||row.entry_price);
    try{
      const path=await pricePath(start,due,Date.now(),entry,fetcher);
      if(path.end===start)continue;
      const result=evaluatePath({direction:row.direction,target:Number(row.target_price),stop:Number(row.invalidation_price),start,due,now:Date.now(),end:path.end,entry,segments:path.segments});
      if(result.reason==="INDETERMINATE"){deferred++;continue;}
      const done=result.reason!=="PENDING",recordedAt=new Date().toISOString(),values:J={checked_until:new Date(result.checkedUntil).toISOString(),last_price:result.price};
      if(done)Object.assign(values,{resolved_at:recordedAt,hit:result.hit,resolution:{...result,resolver_version:RESOLVER_VERSION,recorded_at:recordedAt,event_time_precision:result.eventRangeStart===result.eventAt?"TRADE_OR_DEADLINE":"CANDLE_RANGE"}});
      assertOwned();
      let update=db.from("brian_dip_v84_decisions").update(values).eq("occurrence_id",row.occurrence_id).eq("release_id",RELEASE_ID).is("resolved_at",null);
      update=row.checked_until?update.eq("checked_until",row.checked_until):update.is("checked_until",null);
      const saved=await update.select("occurrence_id");
      if(saved.error)throw Error(saved.error.message);
      if(saved.data?.length){progressed++;if(done)resolved++;}
    }catch(e){deferred++;console.error("dip-v841-resolver deferred",row.occurrence_id,e instanceof Error?e.message:String(e));}
  }
  return{status:"OK",resolved,progressed,deferred,resolver_version:RESOLVER_VERSION,policy_version:POLICY_VERSION,release_id:RELEASE_ID,decision_authority:"BRIAN",shadow_only:true,live_execution:false};
}

export async function readForesight(db:SupabaseClient,sessionId?:string):Promise<J>{
  await assertAuthorityReleaseSealed(db);
  const startQ=await db.from("brian_dip_v84_session_events").select("session_id,requested_at,config").eq("event_kind","START").contains("config",{release_id:RELEASE_ID}).order("requested_at",{ascending:false}).order("event_id",{ascending:false}).limit(1).maybeSingle();
  if(startQ.error)throw Error("V841_START_READ_FAILED:"+startQ.error.message);
  const base={forecasts:{},focus:[SYMBOL],shadow_only:true,live_execution:false,browser_execution:false,dual_direction:true,decision_authority:"BRIAN",release_id:RELEASE_ID,policy_version:POLICY_VERSION,metric_version:METRIC_VERSION};
  if(!startQ.data)return{...base,status:"NO_ACTIVE_SESSION"};
  const sid=String(startQ.data.session_id);
  if(sessionId&&sessionId!==sid)return{...base,status:"STALE_SESSION",session_id:sid};
  const eventQ=await db.from("brian_dip_v84_session_events").select("event_kind,config").eq("session_id",sid).order("requested_at",{ascending:false}).order("event_id",{ascending:false}).limit(1).maybeSingle();
  if(eventQ.error||!eventQ.data)throw Error("V841_SESSION_READ_FAILED");
  if(eventQ.data.config?.release_id&&eventQ.data.config.release_id!==RELEASE_ID)return{...base,status:"V841_RELEASE_MISMATCH",session_id:sid};
  const q=await db.from("brian_dip_v84_runtime").select("runtime,snapshot,updated_at,state_version").eq("session_id",sid).maybeSingle();
  if(q.error||!q.data)throw Error("V841_RUNTIME_READ_FAILED");
  const t=q.data.runtime?.latestThesis;
  if(!t)return{...base,status:"WAIT_STRUCTURE",session_id:sid,state_version:q.data.state_version,updated_at:q.data.updated_at};
  const f={...t,symbol:SYMBOL,target:t.target_price,invalidation:t.invalidation_price,confidence:t.authority_confidence??t.raw_conviction,forecast_accuracy:t.forecast_probability,execution_edge_probability:t.execution_calibration?.p??null,execution_cal_state:t.execution_calibration?.state??"UNAVAILABLE",horizon_min:90,metric_version:METRIC_VERSION,candles:[],position:q.data.runtime?.pos??null,updated_at:q.data.updated_at,state_version:q.data.state_version};
  return{...base,status:eventQ.data.event_kind==="START"?"OK":"PAUSED",session_id:sid,forecasts:{[SYMBOL]:f},meaning:{confidence:"Brian Authority chart confidence; guaranteed probability is not implied",forecast_accuracy:"Resolved forecast calibration; no execution veto authority",execution_edge_probability:"Actual V8.4.1 shadow fill outcomes only",leverage:"Authority release remains 1x SHADOW"}};
}
