import type { SupabaseClient } from "npm:@supabase/supabase-js@2.116.0";
import {
  CALIBRATION_FAMILY_ID,
  DECISION_CADENCE_SECONDS,
  type J,
  LOGIC_HASH,
  MAX_HOLD_MS,
  MARKET_DATA_CONTRACT_VERSION,
  METRIC_VERSION,
  POLICY_VERSION,
  RELEASE_ID,
  STRATEGY_MANIFEST_HASH,
  SYMBOL,
  validateSession,
} from "../_shared/dip_v84_authority_contract.ts";
import { executionCalibration, unavailableCalibration, hash, type ExecutionCalibration, type Runtime, type Segment } from "../_shared/dip_v84_contract.ts";
import { authorityCandidate } from "./authority.ts";
import type { CandidateResult, DecisionRecord } from "../brian-dip-v84-worker/decision.ts";
import { closeOnBrianSell, closeOnMaxHold, closeOnStop } from "./execution.ts";
import { getMarket, positionFunding, pricePath } from "../brian-dip-v84-worker/market.ts";

type Lease={owner:string;generation:number;assertOwned:()=>void};
type Memory={sellVotes:number;lastExitAt:number|null;lastExitPrice:number|null;lastExitReason:string|null;lastTargetRaiseAt:number|null};
type Runtime842=Runtime&{v842?:Memory};

function memory(rt:Runtime842):Memory{
  if(!rt.v842)rt.v842={sellVotes:0,lastExitAt:null,lastExitPrice:null,lastExitReason:null,lastTargetRaiseAt:null};
  return rt.v842;
}

export async function assertReleaseSealed(db:SupabaseClient):Promise<void>{
  if(String(LOGIC_HASH)==="UNSEALED_GITHUB_ONLY")throw Error("V842_RELEASE_NOT_SEALED_IN_SOURCE");
  const q=await db.from("brian_dip_v84_releases").select("status,logic_hash,strategy_manifest_hash,calibration_family_id,db_contract_version,manifest").eq("release_id",RELEASE_ID).maybeSingle();
  if(q.error||!q.data)throw Error("V842_RELEASE_REGISTRY_UNAVAILABLE");
  if(q.data.status!=="SEALED"||q.data.logic_hash!==LOGIC_HASH||q.data.strategy_manifest_hash!==STRATEGY_MANIFEST_HASH||q.data.calibration_family_id!==CALIBRATION_FAMILY_ID)throw Error("V842_RELEASE_REGISTRY_MISMATCH");
  if(q.data.manifest?.shadow_only!==true||q.data.manifest?.live_execution!==false||q.data.manifest?.browser_execution!==false||q.data.manifest?.short_entries!==false||Number(q.data.manifest?.max_shadow_leverage)!==1)throw Error("V842_RELEASE_SAFETY_MISMATCH");
}

async function readRuntime(db:SupabaseClient,sessionId:string):Promise<{runtime:Runtime842;state_version:number}>{
  const q=await db.from("brian_dip_v84_runtime").select("runtime,state_version").eq("session_id",sessionId).maybeSingle();
  if(q.error)throw Error("V842_RUNTIME_READ_FAILED:"+q.error.message);if(!q.data)throw Error("V842_STATE_MISSING_RECONCILE");
  const rt=q.data.runtime as Runtime842;
  if(!rt||![rt.start,rt.cash,rt.realized,rt.trades,rt.wins,rt.losses,rt.marketCursor,Number(q.data.state_version)].every(x=>typeof x==="number"&&Number.isFinite(x))||rt.start<=0||rt.cash<0||!Object.hasOwn(rt,"pos"))throw Error("V842_INVALID_RUNTIME_RECONCILE");
  if(rt.pos?.side==="SHORT")throw Error("V842_SHORT_POSITION_FORBIDDEN");memory(rt);
  return{runtime:structuredClone(rt),state_version:Number(q.data.state_version)};
}
async function episodeEntered(db:SupabaseClient,episode:string):Promise<boolean>{
  const q=await db.from("brian_dip_v84_ledger").select("transition_id").eq("episode_id",episode).eq("event_kind","BUY").limit(1);
  if(q.error)throw Error("V842_EPISODE_READ_FAILED:"+q.error.message);return!!q.data?.length;
}
async function loadExecutionCalibration(db:SupabaseClient,setup:string,regime:string):Promise<ExecutionCalibration>{
  if(!["SWEEP_RECLAIM","FAILED_BREAK","BOS_RETEST","EARLY_REVERSAL"].includes(setup))return executionCalibration({wins:0,losses:0,episodes:0,days:0,ambiguousLosses:0});
  const q=await db.rpc("brian_dip_v84_execution_calibration",{p_release_id:RELEASE_ID,p_calibration_family_id:CALIBRATION_FAMILY_ID,p_setup:setup,p_direction:"UP",p_regime:regime});
  if(q.error)return unavailableCalibration("RPC_ERROR:"+q.error.message);
  const row=Array.isArray(q.data)?q.data[0]:q.data;if(!row||typeof row!=="object")return unavailableCalibration("RPC_EMPTY");
  const r=row as Record<string,unknown>,state=String(r.state||"");if(state==="UNAVAILABLE")return unavailableCalibration(String(r.unavailable_reason||"RPC_UNAVAILABLE"));
  const cal=executionCalibration({wins:Number(r.wins||0),losses:Number(r.losses||0),episodes:Number(r.episodes||0),days:Number(r.days||0),ambiguousLosses:Number(r.ambiguous_losses||0)});
  return cal.state===state?cal:unavailableCalibration("RPC_STATE_MISMATCH");
}
function paperMark(pos:NonNullable<Runtime["pos"]>,market:Awaited<ReturnType<typeof getMarket>>|null){if(!market)return pos.market_price;const friction=pos.expected_exit_slippage_bps/10000;return market.book.bid*(1-friction);}
function collateralValue(pos:NonNullable<Runtime["pos"]>,mark:number){const gross=(mark-pos.entry)*pos.qty,closeFee=mark*pos.qty*pos.fee_bps/10000;return pos.margin+gross-closeFee+(pos.funding_accrued??0);}
function sameFrozenPrice(a:unknown,b:number|null,tickSize:number){if(b===null)return a===null;const x=Number(a);return Number.isFinite(x)&&Math.abs(x-b)<=Math.max(1e-9,tickSize/2);}
function stopTouch(segments:Segment[],stop:number):{hit:boolean;at:number|null;price:number|null}{
  for(const s of segments){
    if(s.points?.length){for(const p of s.points)if(p.p<=stop)return{hit:true,at:p.t,price:p.p};}
    else if(s.l<=stop)return{hit:true,at:s.start,price:stop};
  }
  return{hit:false,at:null,price:null};
}
function lastPathPrice(segments:Segment[],fallback:number){const s=segments.at(-1);return s?.points?.at(-1)?.p??s?.c??fallback;}
function rememberExit(rt:Runtime842,e:J|null){if(!e)return;const m=memory(rt);m.sellVotes=0;m.lastExitAt=Date.now();m.lastExitPrice=Number(e.exit_price||e.price||0)||null;m.lastExitReason=String((e.metadata as J|undefined)?.exit_reason||"SELL");}

async function recordCandidateTelemetry(db:SupabaseClient,sid:string,c:CandidateResult,at:number,market:Awaited<ReturnType<typeof getMarket>>){
  try{
    const s=c.thesis.structure as Record<string,any>,signalAt=Date.parse(String(c.thesis.signal_at||new Date(at).toISOString())),latestSeal=(market.bars["1m"].at(-1)?.ct??at)+1;
    const evaluationId=await hash(["v842-eval",sid,c.occurrence,at].join("|"));
    await db.from("brian_dip_v84_candidate_evaluations").insert({evaluation_id:evaluationId,session_id:sid,occurrence_id:c.occurrence||null,episode_id:c.episode||null,evaluated_at:new Date(at).toISOString(),release_id:RELEASE_ID,strategy_manifest_hash:STRATEGY_MANIFEST_HASH,candidate_set:c.candidateEvaluations,first_blocking_veto:c.firstBlockingVeto,veto_stage:c.vetoStage,spread_bps:market.book.spreadBps,atr_1m:Number(s?.s1?.atr||0)||null,atr_5m:Number(s?.s5?.atr||0)||null,atr_15m:Number(s?.s15?.atr||0)||null,eval_offset_ms_from_seal:Math.max(0,at-latestSeal),payload:{decision_authority:"BRIAN",long_only:true,authority_action:c.thesis.authority_action,authority_confidence:c.thesis.authority_confidence,authority_entry_quality:c.thesis.authority_entry_quality,authority_allocation:c.thesis.authority_allocation,authority_scores:c.thesis.authority_scores,soft_evidence:c.thesis.soft_evidence,signal_at:new Date(signalAt).toISOString(),decision_market_price:c.thesis.decision_market_price,ofi_at_first_eval:market.flowFast.ofi,book_pressure:market.book.pressure,market_available_at:new Date(market.availableAt).toISOString(),market_data_contract_version:MARKET_DATA_CONTRACT_VERSION}});
  }catch(e){console.error("v842 telemetry non-authoritative failure",e instanceof Error?e.message:String(e));}
}

async function latestAuthoritySession(db:SupabaseClient):Promise<{latest:any;start:any}|null>{
  const startQ=await db.from("brian_dip_v84_session_events").select("*").eq("event_kind","START").contains("config",{release_id:RELEASE_ID}).order("requested_at",{ascending:false}).order("event_id",{ascending:false}).limit(1).maybeSingle();
  if(startQ.error)throw Error("V842_START_READ_FAILED:"+startQ.error.message);if(!startQ.data)return null;
  const q=await db.from("brian_dip_v84_session_events").select("*").eq("session_id",String(startQ.data.session_id)).order("requested_at",{ascending:false}).order("event_id",{ascending:false}).limit(1).maybeSingle();
  if(q.error)throw Error("V842_SESSION_READ_FAILED:"+q.error.message);if(!q.data)throw Error("V842_SESSION_EVENT_MISSING");
  return{latest:q.data,start:startQ.data};
}

export async function runWorker(db:SupabaseClient,lease:Lease,fetcher:typeof fetch=fetch):Promise<J>{
  await assertReleaseSealed(db);
  const current=await latestAuthoritySession(db);if(!current)return{status:"NO_ACTIVE_SESSION",release_id:RELEASE_ID,shadow_only:true,live_execution:false,browser_execution:false};
  const sess=current.latest,startSess=current.start,cfg=(startSess.config??{}) as J;validateSession(cfg);
  const sid=String(startSess.session_id),startingEquity=Number(startSess.starting_equity||0),tradeNotional=Math.min(Number(startSess.trade_notional||startingEquity||0),startingEquity);if(!(tradeNotional>0))throw Error("V842_INVALID_TRADE_NOTIONAL");
  const loaded=await readRuntime(db,sid),rt=loaded.runtime,mem=memory(rt);if(sess.event_kind!=="START"&&!rt.pos)return{status:"PAUSED",session_id:sid,release_id:RELEASE_ID,shadow_only:true,live_execution:false,browser_execution:false};
  const events:J[]=[];let pathError:string|null=null;

  // Reconcile only the hard stop. Advisory target is intentionally NOT an automatic exit in V8.4.2.
  if(rt.pos){
    const p=rt.pos,start=p.checked_until||Date.parse(p.opened_at),due=Date.parse(p.due_at);
    try{
      const path=await pricePath(start,due,Date.now(),p.market_price||p.entry,fetcher),touch=stopTouch(path.segments,p.stop),fundingEnd=touch.at??path.end;
      const funding=await positionFunding(p,fundingEnd,fetcher);p.funding_accrued=funding.cashflow;p.funding_settlements=funding.settlements;p.checked_until=path.end;p.market_price=lastPathPrice(path.segments,p.market_price||p.entry);
      if(touch.hit){const closed=closeOnStop(rt,p.stop,Date.now(),{path_trigger_price:touch.price,checked_until:path.end});if(closed){events.push(closed);rememberExit(rt,closed);}}
    }catch(e){pathError=e instanceof Error?e.message:String(e);}
  }

  let market:Awaited<ReturnType<typeof getMarket>>|null=null,marketError:string|null=null;try{market=await getMarket(fetcher);}catch(e){marketError=e instanceof Error?e.message:String(e);}lease.assertOwned();
  const cold=executionCalibration({wins:0,losses:0,episodes:0,days:0,ambiguousLosses:0});
  let c:CandidateResult=market?await authorityCandidate(market,sid,rt,cfg,tradeNotional,Date.now(),cold):{thesis:{veto:["DATA_UNAVAILABLE:"+marketError],decision_authority:"BRIAN",authority_action:rt.pos?"HOLD":"WAIT"},decision:null,occurrence:"",episode:"",combinedFp:"",last5m:0,direction:"WAIT",entry:0,inv:null,target:null,targetRole:"NO_FORWARD_LEVEL",l1:null,size:null,fee:Number(cfg.fee_bps||10),slip:Number(cfg.slippage_bps||1),canEnter:false,candidateEvaluations:[],firstBlockingVeto:"DATA_UNAVAILABLE:"+marketError,vetoStage:"RUNTIME"};
  if(market&&c.decision){const cal=await loadExecutionCalibration(db,String(c.decision.setup),String(c.decision.regime));c=await authorityCandidate(market,sid,rt,cfg,tradeNotional,Date.now(),cal);}

  if(market&&rt.pos&&events.length===0&&!pathError){
    const action=String(c.thesis.authority_action||"HOLD");mem.sellVotes=action==="SELL"?Math.min(5,mem.sellVotes+1):0;
    if(c.target&&c.target>rt.pos.target+market.rules.tickSize&&action==="HOLD"){rt.pos.target=c.target;mem.lastTargetRaiseAt=Date.now();(c.thesis as J).position_advisory_target_raised_to=c.target;}
    let closed=closeOnBrianSell(rt,market,Date.now(),c.thesis,mem.sellVotes);
    if(!closed&&rt.pos&&Date.now()>=Date.parse(rt.pos.due_at))closed=closeOnMaxHold(rt,market,Date.now());
    if(closed){events.push(closed);rememberExit(rt,closed);}
  }

  const at=Date.now();let decision=c.decision,entryDecision:DecisionRecord|null=c.decision;
  if(decision){
    const existing=await db.from("brian_dip_v84_decisions").select("occurrence_id,episode_id,setup,direction,target_price,invalidation_price").eq("occurrence_id",c.occurrence).maybeSingle();if(existing.error)throw Error("V842_DECISION_READ_FAILED:"+existing.error.message);
    if(existing.data){decision=null;const f=existing.data as Record<string,unknown>,samePlan=f.episode_id===c.episode&&f.setup===c.decision?.setup&&f.direction==="UP"&&sameFrozenPrice(f.target_price,c.target,market?.rules.tickSize??0)&&sameFrozenPrice(f.invalidation_price,c.inv,market?.rules.tickSize??0);if(!samePlan){entryDecision=null;c.canEnter=false;(c.thesis.veto as string[]).push("THESIS_LOCKED_NO_NEW_STRUCTURE");}}
  }
  if(c.canEnter&&await episodeEntered(db,c.episode)){c.canEnter=false;(c.thesis.veto as string[]).push("EPISODE_ALREADY_TRADED");}
  // Stateful anti-FOMO memory: after selling, Brian needs either a real pullback or 90 seconds of new evidence before buying higher again.
  if(c.canEnter&&mem.lastExitAt&&mem.lastExitPrice&&at-mem.lastExitAt<90_000&&c.entry>=mem.lastExitPrice*.998){c.canEnter=false;entryDecision=null;(c.thesis.soft_evidence as string[]).push("BRIAN_WAIT_REBASE_AFTER_EXIT");c.thesis.authority_action="WAIT";}
  const stale=!market||Date.now()-market.book.receivedAt>15000||Date.now()-at>20000;if(stale){c.canEnter=false;(c.thesis.veto as string[]).push("STALE_DATA");}if(pathError){c.canEnter=false;(c.thesis.veto as string[]).push("RECONCILIATION_REQUIRED:"+pathError);}if(events.length){c.canEnter=false;(c.thesis.veto as string[]).push("CLOSED_THIS_RUN");}if(sess.event_kind!=="START"){c.canEnter=false;(c.thesis.veto as string[]).push("SESSION_PAUSED");}

  if(market&&c.canEnter&&entryDecision&&!rt.pos&&!events.length&&!pathError&&!stale&&sess.event_kind==="START"&&c.direction==="UP"&&c.target&&c.inv&&c.size){
    const s=c.size,openedAt=new Date(at).toISOString(),stopMeta=c.thesis.stop as Record<string,unknown>;
    rt.pos={side:"LONG",position_id:`v842-long-${c.occurrence}`,thesis_id:c.occurrence,episode_id:c.episode,setup:entryDecision.setup,regime:entryDecision.regime,entry:c.entry,qty:s.qty,notional:s.notional,target:c.target,stop:c.inv,stop_source:String(stopMeta.source||"UNKNOWN"),stop_structure_id:typeof stopMeta.structure_id==="string"?stopMeta.structure_id:null,opened_at:openedAt,due_at:new Date(at+MAX_HOLD_MS).toISOString(),fees_open:s.fees_open,fee_bps:c.fee,slippage_bps:c.slip,expected_exit_spread_bps:market.book.spreadBps,expected_exit_slippage_bps:c.slip,venue:"SHADOW_PERP",policy_version:POLICY_VERSION,release_id:RELEASE_ID,calibration_family_id:CALIBRATION_FAMILY_ID,checked_until:at,market_price:c.entry,leverage:1,margin:s.margin,actual_fraction:s.actual_fraction,gross_fraction:s.gross_fraction};
    mem.sellVotes=0;rt.cash-=s.margin+s.fees_open;const mark=paperMark(rt.pos,market),equityAfter=rt.cash+collateralValue(rt.pos,mark);
    events.push({event_kind:"BUY",position_id:rt.pos.position_id,occurrence_id:c.occurrence,episode_id:c.episode,price:c.entry,entry_price:c.entry,quantity:s.qty,notional:s.notional,fees:s.fees_open,realized_pnl:0,cash_after:rt.cash,equity_after:equityAfter,metadata:{server_v84:true,brian_authority:true,long_only:true,release_id:RELEASE_ID,calibration_family_id:CALIBRATION_FAMILY_ID,side:"LONG",setup:entryDecision.setup,regime:entryDecision.regime,target:c.target,target_is_advisory:true,stop:c.inv,target_rank:(c.thesis.target_plan as Record<string,unknown>)?.selected_rank,economic_rr:c.thesis.economic_rr,authority_confidence:c.thesis.authority_confidence,authority_entry_quality:c.thesis.authority_entry_quality,authority_allocation:c.thesis.authority_allocation,authority_reason:c.thesis.authority_reason,decision_market_price:c.thesis.decision_market_price,soft_evidence:c.thesis.soft_evidence,actual_fraction:s.actual_fraction,gross_fraction:s.gross_fraction,margin:s.margin,notional:s.notional,worst_loss:s.worst_loss,risk_fraction:s.risk_fraction,leverage:1,policy_version:POLICY_VERSION,metric_version:METRIC_VERSION,market_source:market.source,fee_bps:c.fee,slippage_bps:c.slip,expected_exit_spread_bps:market.book.spreadBps}});
  }

  if(decision)rt.lastOccurrence=c.occurrence;rt.latestThesis=c.thesis;if(market)rt.marketCursor=Math.max(rt.marketCursor,market.bars["1m"].at(-1)!.ct+1);
  const recordedAt=new Date().toISOString(),hour=recordedAt.slice(0,13);rt.lastSnapshotHour=hour;const mark=rt.pos?paperMark(rt.pos,market):0,positionValue=rt.pos?collateralValue(rt.pos,mark):0,equity=rt.cash+positionValue,unrealized=equity-rt.start-rt.realized;
  const state={start:rt.start,cfg:{...cfg,symbols:[SYMBOL],trade_notional:tradeNotional},symbols:{[SYMBOL]:{symbol:SYMBOL,last:market?.book.mid??rt.pos?.market_price??null,price:market?.book.mid??rt.pos?.market_price??null,pos:rt.pos,armed:false,lastAction:rt.pos?"LONG":"WATCH",thesis:c.thesis}},v84:rt,serverRuntime:{authoritative:true,decision_authority:"BRIAN",release_id:RELEASE_ID,logic_hash:LOGIC_HASH,strategy_manifest_hash:STRATEGY_MANIFEST_HASH,calibration_family_id:CALIBRATION_FAMILY_ID,policy_version:POLICY_VERSION,metric_version:METRIC_VERSION,generated_at:recordedAt,state_version:loaded.state_version+1,browser_executor_disabled:true,shadow_only:true,live_execution:false,dual_direction:false,long_only:true,short_entries:false,max_shadow_leverage:1,risk_promotion_enabled:false,decision_cadence_seconds:DECISION_CADENCE_SECONDS,market_source:market?.source||"BINANCE_USDM_PERP",market_received_at:market?new Date(market.book.receivedAt).toISOString():null,market_price:market?.book.mid??null,status:pathError?"RECONCILIATION_REQUIRED":marketError?"DATA_UNAVAILABLE":"OK"}};
  const snapshot={snapshot_id:`v842-hour-${sid}-${hour}`,session_id:sid,observed_at:recordedAt,cash:rt.cash,equity,realized_pnl:rt.realized,unrealized_pnl:unrealized,trade_count:rt.trades,win_count:rt.wins,loss_count:rt.losses,state};
  for(const e of events)e.transition_id=`v842-${String(e.event_kind).toLowerCase()}-${e.occurrence_id}-${Date.now()}`;
  const commitId=await hash([sid,loaded.state_version,rt.marketCursor,RELEASE_ID,String(c.occurrence||"none"),events.map(e=>e.event_kind).join(",")].join("|"));lease.assertOwned();
  const committed=await db.rpc("brian_dip_v84_commit",{p_session_id:sid,p_expected_version:loaded.state_version,p_owner_id:lease.owner,p_lease_generation:lease.generation,p_commit_id:commitId,p_runtime:rt,p_snapshot:snapshot,p_decision:decision,p_events:events});if(committed.error)throw Error("V842_COMMIT_FAILED:"+committed.error.message);
  if(market)await recordCandidateTelemetry(db,sid,c,at,market);
  return{status:pathError?"RECONCILIATION_REQUIRED":marketError?"DATA_UNAVAILABLE":"OK",market_error:marketError,session_id:sid,release_id:RELEASE_ID,state_version:loaded.state_version+1,symbol:SYMBOL,thesis:c.thesis,position:rt.pos,equity,realized:rt.realized,events:events.map(e=>e.event_kind),shadow_only:true,live_execution:false,browser_execution:false,long_only:true};
}