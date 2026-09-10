import type { SupabaseClient } from "npm:@supabase/supabase-js@2.116.0";
import {
  evaluatePath,
  type J,
  METRIC_VERSION,
  POLICY_VERSION,
  RELEASE_ID,
  RESOLVER_VERSION,
  SYMBOL,
} from "../_shared/dip_v84_contract.ts";
import { pricePath } from "../brian-dip-v84-worker/market.ts";
import { assertReleaseSealed } from "../brian-dip-v84-worker/worker.ts";

export async function resolvePending(db:SupabaseClient,assertOwned:()=>void,fetcher:typeof fetch=fetch):Promise<J>{
  await assertReleaseSealed(db);
  const began=Date.now();
  const q=await db.from("brian_dip_v84_decisions").select("occurrence_id,decision_at,signal_at,due_at,direction,target_price,invalidation_price,entry_price,checked_until,last_price").eq("release_id",RELEASE_ID).eq("policy_version",POLICY_VERSION).eq("metric_version",METRIC_VERSION).not("target_price","is",null).not("invalidation_price","is",null).is("resolved_at",null).order("checked_until",{ascending:true,nullsFirst:true}).order("decision_at",{ascending:true}).limit(24);
  if(q.error)throw Error("V84_FORECAST_READ_FAILED:"+q.error.message);
  let resolved=0,progressed=0,deferred=0;
  for(const row of q.data||[]){
    if(Date.now()-began>32_000){deferred++;break;}assertOwned();
    const start=Date.parse(row.checked_until||row.decision_at),due=Date.parse(row.due_at),entry=Number(row.last_price||row.entry_price);
    try{
      const path=await pricePath(start,due,Date.now(),entry,fetcher);if(path.end===start)continue;
      const result=evaluatePath({direction:row.direction,target:Number(row.target_price),stop:Number(row.invalidation_price),start,due,now:Date.now(),end:path.end,entry,segments:path.segments});
      if(result.reason==="INDETERMINATE"){deferred++;continue;}
      const done=result.reason!=="PENDING",recordedAt=new Date().toISOString(),values:J={checked_until:new Date(result.checkedUntil).toISOString(),last_price:result.price};
      if(done)Object.assign(values,{resolved_at:recordedAt,hit:result.hit,resolution:{...result,resolver_version:RESOLVER_VERSION,recorded_at:recordedAt,event_time_precision:result.eventRangeStart===result.eventAt?"TRADE_OR_DEADLINE":"CANDLE_RANGE"}});
      assertOwned();let update=db.from("brian_dip_v84_decisions").update(values).eq("occurrence_id",row.occurrence_id).is("resolved_at",null);update=row.checked_until?update.eq("checked_until",row.checked_until):update.is("checked_until",null);const saved=await update.select("occurrence_id");if(saved.error)throw Error(saved.error.message);if(saved.data?.length){progressed++;if(done)resolved++;}
    }catch(e){deferred++;console.error("dip-v84-resolver deferred",row.occurrence_id,e instanceof Error?e.message:String(e));}
  }
  return{status:"OK",resolved,progressed,deferred,resolver_version:RESOLVER_VERSION,policy_version:POLICY_VERSION,release_id:RELEASE_ID};
}

export async function readForesight(db:SupabaseClient,sessionId?:string):Promise<J>{
  await assertReleaseSealed(db);
  const s=await db.from("brian_dip_v84_session_events").select("session_id,event_kind,config").order("requested_at",{ascending:false}).order("event_id",{ascending:false}).limit(1).maybeSingle();
  if(s.error)throw Error("V84_SESSION_READ_FAILED:"+s.error.message);const session=s.data;
  const base={forecasts:{},focus:[SYMBOL],shadow_only:true,live_execution:false,dual_direction:true,release_id:RELEASE_ID,policy_version:POLICY_VERSION,metric_version:METRIC_VERSION};
  if(!session)return{...base,status:"NO_ACTIVE_SESSION"};
  if(sessionId&&sessionId!==session.session_id)return{...base,status:"STALE_SESSION",session_id:session.session_id};
  if(session.config?.release_id!==RELEASE_ID)return{...base,status:"V84_RELEASE_MISMATCH",session_id:session.session_id};
  const q=await db.from("brian_dip_v84_runtime").select("runtime,snapshot,updated_at,state_version").eq("session_id",session.session_id).maybeSingle();if(q.error||!q.data)throw Error("V84_RUNTIME_READ_FAILED");
  const t=q.data.runtime.latestThesis;if(!t)return{...base,status:"WAIT_STRUCTURE",session_id:session.session_id};
  const f={...t,symbol:SYMBOL,target:t.target_price,invalidation:t.invalidation_price,confidence:t.raw_conviction,forecast_accuracy:t.forecast_probability,execution_edge_probability:t.execution_calibration?.p??null,execution_cal_state:t.execution_calibration?.state??"UNAVAILABLE",horizon_min:90,metric_version:METRIC_VERSION,candles:[],position:q.data.runtime.pos,updated_at:q.data.updated_at,state_version:q.data.state_version};
  return{...base,status:"OK",session_id:session.session_id,forecasts:{[SYMBOL]:f},meaning:{raw_conviction:"Yapısal puan; başarı olasılığı değildir",forecast_accuracy:"Forecast kalibrasyonu; execution yetkisi vermez",execution_edge_probability:"Yalnız gerçek V8.4 shadow execution outcomes; cold iken null",leverage:"Package 1 evidence release daima 1x"}};
}
