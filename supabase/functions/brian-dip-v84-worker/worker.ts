import type { SupabaseClient } from "npm:@supabase/supabase-js@2.116.0";
import {
  CALIBRATION_FAMILY_ID,
  type ExecutionCalibration,
  executionCalibration,
  hash,
  type J,
  LOGIC_HASH,
  MAX_HOLD_MS,
  MARKET_DATA_CONTRACT_VERSION,
  METRIC_VERSION,
  POLICY_VERSION,
  RELEASE_ID,
  STRATEGY_MANIFEST_HASH,
  type Runtime,
  SYMBOL,
  unavailableCalibration,
  validateSession,
} from "../_shared/dip_v84_contract.ts";
import { candidate, type CandidateResult } from "./decision.ts";
import { closePosition } from "./execution.ts";
import { getMarket, positionFunding, pricePath } from "./market.ts";

type Lease={owner:string;generation:number;assertOwned:()=>void};

export async function assertReleaseSealed(db:SupabaseClient):Promise<void>{
  if(LOGIC_HASH==="UNSEALED_GITHUB_ONLY")throw Error("V84_RELEASE_NOT_SEALED_IN_SOURCE");
  const q=await db.from("brian_dip_v84_releases").select("status,logic_hash,strategy_manifest_hash,calibration_family_id,db_contract_version").eq("release_id",RELEASE_ID).maybeSingle();
  if(q.error||!q.data)throw Error("V84_RELEASE_REGISTRY_UNAVAILABLE");
  if(q.data.status!=="SEALED"||q.data.logic_hash!==LOGIC_HASH||q.data.strategy_manifest_hash!==STRATEGY_MANIFEST_HASH||q.data.calibration_family_id!==CALIBRATION_FAMILY_ID)throw Error("V84_RELEASE_REGISTRY_MISMATCH");
}

export async function readRuntime(db:SupabaseClient,sessionId:string):Promise<{runtime:Runtime;state_version:number}>{
  const q=await db.from("brian_dip_v84_runtime").select("runtime,state_version").eq("session_id",sessionId).maybeSingle();
  if(q.error)throw Error("V84_RUNTIME_READ_FAILED:"+q.error.message);if(!q.data)throw Error("V84_STATE_MISSING_RECONCILE");
  const rt=q.data.runtime as Runtime;
  if(!rt||![rt.start,rt.cash,rt.realized,rt.trades,rt.wins,rt.losses,rt.marketCursor,Number(q.data.state_version)].every(x=>typeof x==="number"&&Number.isFinite(x))||rt.start<=0||rt.cash<0||!Object.hasOwn(rt,"pos"))throw Error("V84_INVALID_RUNTIME_RECONCILE");
  return{runtime:structuredClone(rt),state_version:Number(q.data.state_version)};
}

export async function episodeEntered(db:SupabaseClient,episode:string):Promise<boolean>{
  const q=await db.from("brian_dip_v84_ledger").select("transition_id").eq("episode_id",episode).in("event_kind",["BUY","SHORT_OPEN"]).limit(1);
  if(q.error)throw Error("V84_EPISODE_READ_FAILED:"+q.error.message);return !!q.data?.length;
}

export async function loadExecutionCalibration(db:SupabaseClient,setup:string,direction:string,regime:string):Promise<ExecutionCalibration>{
  if(!["SWEEP_RECLAIM","FAILED_BREAK","BOS_RETEST","EARLY_REVERSAL"].includes(setup)||!["UP","DOWN"].includes(direction))return executionCalibration({wins:0,losses:0,episodes:0,days:0,ambiguousLosses:0});
  const q=await db.rpc("brian_dip_v84_execution_calibration",{p_release_id:RELEASE_ID,p_calibration_family_id:CALIBRATION_FAMILY_ID,p_setup:setup,p_direction:direction,p_regime:regime});
  if(q.error)return unavailableCalibration("RPC_ERROR:"+q.error.message);
  const row=Array.isArray(q.data)?q.data[0]:q.data;if(!row||typeof row!=="object")return unavailableCalibration("RPC_EMPTY");
  const r=row as Record<string,unknown>,state=String(r.state||"");
  if(state==="UNAVAILABLE")return unavailableCalibration(String(r.unavailable_reason||"RPC_UNAVAILABLE"));
  const cal=executionCalibration({wins:Number(r.wins||0),losses:Number(r.losses||0),episodes:Number(r.episodes||0),days:Number(r.days||0),ambiguousLosses:Number(r.ambiguous_losses||0)});
  if(cal.state!==state)return unavailableCalibration("RPC_STATE_MISMATCH");return cal;
}

function paperMark(pos:NonNullable<Runtime["pos"]>,market:Awaited<ReturnType<typeof getMarket>>|null){
  if(!market)return pos.market_price;const friction=pos.expected_exit_slippage_bps/10000;return pos.side==="LONG"?market.book.bid*(1-friction):market.book.ask*(1+friction);
}
function collateralValue(pos:NonNullable<Runtime["pos"]>,mark:number){const gross=(pos.side==="LONG"?mark-pos.entry:pos.entry-mark)*pos.qty,closeFee=mark*pos.qty*pos.fee_bps/10000;return pos.margin+gross-closeFee+(pos.funding_accrued??0);}

async function recordCandidateTelemetry(db:SupabaseClient,sid:string,c:CandidateResult,at:number,market:Awaited<ReturnType<typeof getMarket>>){
  try{
    const s=c.thesis.structure as Record<string,any>,signalAt=Date.parse(String(c.thesis.signal_at||new Date(at).toISOString())),latestSeal=Number(s?.s1?.lastClose?market.bars["1m"].at(-1)?.ct??at:market.bars["1m"].at(-1)?.ct??at)+1;
    const evaluationId=await hash(["v84-eval",sid,c.occurrence,at].join("|"));
    await db.from("brian_dip_v84_candidate_evaluations").insert({evaluation_id:evaluationId,session_id:sid,occurrence_id:c.occurrence||null,episode_id:c.episode||null,evaluated_at:new Date(at).toISOString(),release_id:RELEASE_ID,strategy_manifest_hash:STRATEGY_MANIFEST_HASH,candidate_set:c.candidateEvaluations,first_blocking_veto:c.firstBlockingVeto,veto_stage:c.vetoStage,spread_bps:market.book.spreadBps,atr_1m:Number(s?.s1?.atr||0)||null,atr_5m:Number(s?.s5?.atr||0)||null,atr_15m:Number(s?.s15?.atr||0)||null,eval_offset_ms_from_seal:Math.max(0,at-latestSeal),payload:{signal_at:new Date(signalAt).toISOString(),ofi_at_first_eval:market.flowFast.ofi,book_pressure:market.book.pressure,market_available_at:new Date(market.availableAt).toISOString(),market_data_contract_version:MARKET_DATA_CONTRACT_VERSION}});
  }catch(e){console.error("v84 telemetry non-authoritative failure",e instanceof Error?e.message:String(e));}
}

export async function runWorker(db:SupabaseClient,lease:Lease,fetcher:typeof fetch=fetch):Promise<J>{
  await assertReleaseSealed(db);
  const q=await db.from("brian_dip_v84_session_events").select("*").order("requested_at",{ascending:false}).order("event_id",{ascending:false}).limit(1).maybeSingle();if(q.error)throw Error("V84_SESSION_READ_FAILED:"+q.error.message);const sess=q.data;if(!sess)return{status:"NO_ACTIVE_SESSION",release_id:RELEASE_ID,shadow_only:true,live_execution:false};
  let startSess=sess;if(sess.event_kind!=="START"){const z=await db.from("brian_dip_v84_session_events").select("*").eq("session_id",String(sess.session_id)).eq("event_kind","START").order("requested_at",{ascending:true}).order("event_id",{ascending:true}).limit(1).maybeSingle();if(z.error)throw Error("V84_START_READ_FAILED:"+z.error.message);if(!z.data)throw Error("V84_START_MISSING");startSess=z.data;}
  const cfg=(startSess.config??{}) as J;validateSession(cfg);
  const sid=String(startSess.session_id),startingEquity=Number(startSess.starting_equity||0),tradeNotional=Math.min(Number(startSess.trade_notional||startingEquity||0),startingEquity);if(!(tradeNotional>0))throw Error("V84_INVALID_TRADE_NOTIONAL");
  const loaded=await readRuntime(db,sid),rt=loaded.runtime;if(sess.event_kind!=="START"&&!rt.pos)return{status:"PAUSED",session_id:sid,release_id:RELEASE_ID,shadow_only:true,live_execution:false};
  const events:J[]=[];let pathError:string|null=null;
  if(rt.pos){
    const p=rt.pos,start=p.checked_until||Date.parse(p.opened_at),due=Date.parse(p.due_at);
    try{const path=await pricePath(start,due,Date.now(),p.market_price||p.entry,fetcher),resolution=(await import("../_shared/dip_v84_contract.ts")).evaluatePath({direction:p.side==="LONG"?"UP":"DOWN",target:p.target,stop:p.stop,start,due,now:Date.now(),end:path.end,entry:p.market_price||p.entry,segments:path.segments});if(resolution.reason==="INDETERMINATE")throw Error("V84_PRICE_PATH_INDETERMINATE");const funding=await positionFunding(p,resolution.eventAt??resolution.checkedUntil,fetcher);p.funding_accrued=funding.cashflow;p.funding_settlements=funding.settlements;p.checked_until=resolution.checkedUntil;p.market_price=resolution.price;const closed=closePosition(rt,resolution,Date.now());if(closed)events.push(closed);}catch(e){pathError=e instanceof Error?e.message:String(e);}
  }
  let market:Awaited<ReturnType<typeof getMarket>>|null=null,marketError:string|null=null;try{market=await getMarket(fetcher);}catch(e){marketError=e instanceof Error?e.message:String(e);}lease.assertOwned();
  const cold=executionCalibration({wins:0,losses:0,episodes:0,days:0,ambiguousLosses:0});
  let c:CandidateResult=market?await candidate(market,sid,rt,cfg,tradeNotional,Date.now(),cold):{thesis:{veto:["DATA_UNAVAILABLE:"+marketError]},decision:null,occurrence:"",episode:"",combinedFp:"",last5m:0,direction:"WAIT",entry:0,inv:null,target:null,targetRole:"NO_FORWARD_LEVEL",l1:null,size:null,fee:Number(cfg.fee_bps||10),slip:Number(cfg.slippage_bps||1),canEnter:false,candidateEvaluations:[],firstBlockingVeto:"DATA_UNAVAILABLE:"+marketError,vetoStage:"RUNTIME"};
  if(market&&c.decision){const setup=String(c.decision.setup),direction=String(c.decision.direction),regime=String(c.decision.regime),cal=await loadExecutionCalibration(db,setup,direction,regime);c=await candidate(market,sid,rt,cfg,tradeNotional,Date.now(),cal);}
  const at=Date.now();let decision=c.decision;
  if(decision){const existing=await db.from("brian_dip_v84_decisions").select("occurrence_id").eq("occurrence_id",c.occurrence).maybeSingle();if(existing.error)throw Error("V84_DECISION_READ_FAILED:"+existing.error.message);if(existing.data)decision=null;}
  if(c.canEnter&&await episodeEntered(db,c.episode)){c.canEnter=false;(c.thesis.veto as string[]).push("EPISODE_ALREADY_TRADED");}
  const stale=!market||Date.now()-market.book.receivedAt>15000||Date.now()-at>20000;if(stale)(c.thesis.veto as string[]).push("STALE_DATA");if(pathError)(c.thesis.veto as string[]).push("RECONCILIATION_REQUIRED:"+pathError);if(events.length)(c.thesis.veto as string[]).push("CLOSED_THIS_RUN");if(sess.event_kind!=="START")(c.thesis.veto as string[]).push("SESSION_PAUSED");
  if(market&&c.canEnter&&decision&&!rt.pos&&!events.length&&!pathError&&!stale&&sess.event_kind==="START"&&c.target&&c.inv&&c.size){
    const s=c.size,openedAt=new Date(at).toISOString(),side=c.direction==="DOWN"?"SHORT":"LONG",stopMeta=c.thesis.stop as Record<string,unknown>;
    rt.pos={side,position_id:`v84-${side.toLowerCase()}-${c.occurrence}`,thesis_id:c.occurrence,episode_id:c.episode,setup:decision.setup,regime:decision.regime,entry:c.entry,qty:s.qty,notional:s.notional,target:c.target,stop:c.inv,stop_source:String(stopMeta.source||"UNKNOWN"),stop_structure_id:typeof stopMeta.structure_id==="string"?stopMeta.structure_id:null,opened_at:openedAt,due_at:new Date(at+MAX_HOLD_MS).toISOString(),fees_open:s.fees_open,fee_bps:c.fee,slippage_bps:c.slip,expected_exit_spread_bps:market.book.spreadBps,expected_exit_slippage_bps:c.slip,venue:"SHADOW_PERP",policy_version:POLICY_VERSION,release_id:RELEASE_ID,calibration_family_id:CALIBRATION_FAMILY_ID,checked_until:at,market_price:c.entry,leverage:1,margin:s.margin,actual_fraction:s.actual_fraction,gross_fraction:s.gross_fraction};
    rt.cash-=s.margin+s.fees_open;const mark=paperMark(rt.pos,market),equityAfter=rt.cash+collateralValue(rt.pos,mark);
    events.push({event_kind:side==="LONG"?"BUY":"SHORT_OPEN",position_id:rt.pos.position_id,occurrence_id:c.occurrence,episode_id:c.episode,price:c.entry,entry_price:c.entry,quantity:s.qty,notional:s.notional,fees:s.fees_open,realized_pnl:0,cash_after:rt.cash,equity_after:equityAfter,metadata:{server_v84:true,release_id:RELEASE_ID,calibration_family_id:CALIBRATION_FAMILY_ID,side,setup:decision.setup,regime:decision.regime,target:c.target,stop:c.inv,target_role:c.targetRole,economic_rr:c.thesis.economic_rr,raw_conviction:c.thesis.raw_conviction,execution_calibration:c.thesis.execution_calibration,actual_fraction:s.actual_fraction,gross_fraction:s.gross_fraction,margin:s.margin,notional:s.notional,worst_loss:s.worst_loss,risk_fraction:s.risk_fraction,leverage:1,policy_version:POLICY_VERSION,metric_version:METRIC_VERSION,market_source:market.source,fee_bps:c.fee,slippage_bps:c.slip,expected_exit_spread_bps:market.book.spreadBps}});
  }
  if(decision)rt.lastOccurrence=c.occurrence;rt.latestThesis=c.thesis;if(market)rt.marketCursor=Math.max(rt.marketCursor,market.bars["1m"].at(-1)!.ct+1);
  const recordedAt=new Date().toISOString(),hour=recordedAt.slice(0,13);rt.lastSnapshotHour=hour;const mark=rt.pos?paperMark(rt.pos,market):0,positionValue=rt.pos?collateralValue(rt.pos,mark):0,equity=rt.cash+positionValue,unrealized=equity-rt.start-rt.realized;
  const state={start:rt.start,cfg:{...cfg,symbols:[SYMBOL],trade_notional:tradeNotional},symbols:{[SYMBOL]:{symbol:SYMBOL,last:market?.book.mid??rt.pos?.market_price??null,price:market?.book.mid??rt.pos?.market_price??null,pos:rt.pos,armed:false,lastAction:rt.pos?rt.pos.side:"WATCH",thesis:c.thesis}},v84:rt,serverRuntime:{authoritative:true,release_id:RELEASE_ID,logic_hash:LOGIC_HASH,strategy_manifest_hash:STRATEGY_MANIFEST_HASH,calibration_family_id:CALIBRATION_FAMILY_ID,policy_version:POLICY_VERSION,metric_version:METRIC_VERSION,generated_at:recordedAt,state_version:loaded.state_version+1,browser_executor_disabled:true,shadow_only:true,live_execution:false,dual_direction:true,max_shadow_leverage:1,risk_promotion_enabled:false,decision_cadence_seconds:15,market_source:market?.source||"BINANCE_USDM_PERP",status:pathError?"RECONCILIATION_REQUIRED":marketError?"DATA_UNAVAILABLE":"OK"}};
  const snapshot={snapshot_id:`v84-hour-${sid}-${hour}`,session_id:sid,observed_at:recordedAt,cash:rt.cash,equity,realized_pnl:rt.realized,unrealized_pnl:unrealized,trade_count:rt.trades,win_count:rt.wins,loss_count:rt.losses,state};
  for(const e of events)e.transition_id=`v84-${String(e.event_kind).toLowerCase()}-${e.occurrence_id}`;
  const commitId=await hash([sid,loaded.state_version,rt.marketCursor,RELEASE_ID,String(c.occurrence||"none")].join("|"));lease.assertOwned();
  const committed=await db.rpc("brian_dip_v84_commit",{p_session_id:sid,p_expected_version:loaded.state_version,p_owner_id:lease.owner,p_lease_generation:lease.generation,p_commit_id:commitId,p_runtime:rt,p_snapshot:snapshot,p_decision:decision,p_events:events});if(committed.error)throw Error("V84_COMMIT_FAILED:"+committed.error.message);
  if(market)await recordCandidateTelemetry(db,sid,c,at,market);
  return{status:pathError?"RECONCILIATION_REQUIRED":marketError?"DATA_UNAVAILABLE":"OK",market_error:marketError,session_id:sid,release_id:RELEASE_ID,state_version:loaded.state_version+1,symbol:SYMBOL,thesis:c.thesis,position:rt.pos,equity,realized:rt.realized,events:events.map(e=>e.event_kind),shadow_only:true,live_execution:false};
}
