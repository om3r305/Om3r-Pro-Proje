import { entryGuard } from "./entry_guard.ts";
import { buildCostContract, executionEconomics, fillForwardCostBps, referenceRoundtripCostBps } from "./cost.ts";
import { firstForwardLevel, type StructuralLevel } from "./levels.ts";
import { sizePosition } from "./risk.ts";
import {
  calibrationNoEdge,
  CALIBRATION_FAMILY_ID,
  COST_MODEL_VERSION,
  DB_CONTRACT_VERSION,
  DECISION_REVISION,
  ENGINE_VERSION,
  ENTRY_GUARD_VERSION,
  type ExecutionCalibration,
  hash,
  LOGIC_HASH,
  MAX_HOLD_MS,
  METRIC_VERSION,
  MIN_ECONOMIC_RR,
  POLICY_VERSION,
  RELEASE_ID,
  RESOLVER_VERSION,
  type Runtime,
  STRATEGY_MANIFEST_HASH,
  SYMBOL,
  TARGET_COST_MULTIPLE,
  TARGET_PLANNER_VERSION,
  EXECUTION_MODEL_VERSION,
  MARKET_DATA_CONTRACT_VERSION,
  type J,
  type Setup,
  type Struct,
  clip,
  n,
  WARM_RISK_PROMOTION_ENABLED,
} from "../_shared/dip_v84_contract.ts";
import type { Market } from "./market.ts";
import { structure } from "./structure.ts";

export type Direction="UP"|"DOWN"|"WAIT";
type TradeDirection="UP"|"DOWN";
type SetupCandidate={setup:Setup;direction:TradeDirection;trigger:Struct;triggerIdentity:string;priority:number;provisionalStop:number|null;reason:string};
export type CandidateTelemetry={
  rank:number;selected:boolean;setup:Setup;direction:TradeDirection;trigger_tf:string;trigger_identity:string;reason:string;base_priority:number;
  structure_score:number;fast_ofi_component:number;slow_ofi_component:number;book_component:number;direction_score:number;final_priority:number;
};
export type DecisionRecord={
  occurrence_id:string;session_id:string;episode_id:string;parent_occurrence_id:string|null;symbol:string;decision_at:string;signal_at:string;due_at:string;
  setup:Setup;direction:TradeDirection;regime:string;venue:"SHADOW_PERP";entry_price:number;target_price:number|null;invalidation_price:number|null;
  target_role:"L1_EXECUTABLE"|"L1_BLOCKING"|"NO_FORWARD_LEVEL";l1_id:string|null;raw_conviction:number;forecast_probability:number|null;
  release_id:string;logic_hash:string;strategy_manifest_hash:string;calibration_family_id:string;db_contract_version:string;
  policy_version:string;decision_revision:string;metric_version:string;resolver_version:string;entry_guard_version:string;target_planner_version:string;execution_model_version:string;cost_model_version:string;market_data_contract_version:string;
  evidence:J;shadow_only:true;live_execution:false;
};
export type CandidateResult={
  thesis:J;decision:DecisionRecord|null;occurrence:string;episode:string;combinedFp:string;last5m:number;
  direction:Direction;entry:number;inv:number|null;target:number|null;targetRole:"L1_EXECUTABLE"|"L1_BLOCKING"|"NO_FORWARD_LEVEL";
  l1:StructuralLevel|null;size:ReturnType<typeof sizePosition>;fee:number;slip:number;canEnter:boolean;
  candidateEvaluations:CandidateTelemetry[];firstBlockingVeto:string|null;vetoStage:string|null;
};

const STAGE:Record<string,string>={
  NO_SIGNAL:"STRUCTURE",NO_FORWARD_LEVEL:"TARGET_GEOMETRY",STRUCTURE_LEVEL_INCOMPLETE:"TARGET_GEOMETRY",
  TARGET_BELOW_COST:"TARGET_ECONOMICS",ECONOMIC_RR_TOO_LOW:"TARGET_ECONOMICS",
  WAIT_RETEST:"ENTRY_GUARD",ENTRY_TOO_LATE:"ENTRY_GUARD",ENTRY_DATA_INCOMPLETE:"ENTRY_GUARD",
  CALIBRATION_UNAVAILABLE:"CALIBRATION",SHORT_DISABLED:"STRUCTURE",COUNTER_STRUCTURE:"STRUCTURE",OPPOSING_MULTI_FLOW:"STRUCTURE",DIRECTION_REFEREE_REJECT:"STRUCTURE",RAW_CONVICTION_LOW:"STRUCTURE",
  MIN_NOTIONAL_OR_RISK_CAP:"RISK_SIZING",OBSERVATION_ONLY:"RUNTIME",THESIS_LOCKED_NO_NEW_STRUCTURE:"RUNTIME",
};
const STAGE_ORDER:Record<string,number>={STRUCTURE:0,TARGET_GEOMETRY:1,TARGET_ECONOMICS:2,ENTRY_GUARD:3,CALIBRATION:4,RISK_SIZING:5,RUNTIME:6,OTHER:7};
const BASE_PRIORITY:Record<Setup,number>={SWEEP_RECLAIM:7,FAILED_BREAK:7,BOS_RETEST:7,EARLY_REVERSAL:6};
function isTradeDirection(direction:Direction):direction is TradeDirection{return direction==="UP"||direction==="DOWN";}
function validGeometry(direction:Direction,entry:number,inv:number|null,target:number|null):boolean{if(!isTradeDirection(direction)||!(entry>0)||!(inv&&target))return false;return direction==="UP"?inv<entry&&entry<target:target<entry&&entry<inv;}

function recentReversal(m:Market,s1:Struct,s5:Struct):SetupCandidate|null{
  const rows=m.bars["1m"].slice(-8);if(rows.length<8)return null;
  const last=rows.at(-1)!,prev=rows.at(-2)!,prev2=rows.at(-3)!,recent=rows.slice(-7);
  const lowBar=recent.reduce((a,b)=>b.l<a.l?b:a),highBar=recent.reduce((a,b)=>b.h>a.h?b:a),recentLow=lowBar.l,recentHigh=highBar.h,fast=m.flowFast.ofi,slow=m.flowSlow.ofi,improve=fast-slow;
  const upMomentum=(last.c>prev.c&&prev.c>=prev2.c)||(last.c>prev.h&&last.c>last.o),downMomentum=(last.c<prev.c&&prev.c<=prev2.c)||(last.c<prev.l&&last.c<last.o),upDistance=(last.c-recentLow)/last.c*10000,downDistance=(recentHigh-last.c)/last.c*10000,minMove=Math.max(10,s1.atr/last.c*10000*.45),upFlow=fast>=.06||improve>=.18,downFlow=fast<=-.06||improve<=-.18,upConflict=s5.bos==="DOWN"&&s1.bos==="DOWN",downConflict=s5.bos==="UP"&&s1.bos==="UP";
  if(upMomentum&&upFlow&&!upConflict&&upDistance>=minMove)return{setup:"EARLY_REVERSAL",direction:"UP",trigger:s1,triggerIdentity:`EARLY_REVERSAL:UP:1m:EXTREME:${lowBar.t}`,priority:6,provisionalStop:recentLow,reason:`1m momentum + flow recovery ${fast.toFixed(2)}/${slow.toFixed(2)}`};
  if(downMomentum&&downFlow&&!downConflict&&downDistance>=minMove)return{setup:"EARLY_REVERSAL",direction:"DOWN",trigger:s1,triggerIdentity:`EARLY_REVERSAL:DOWN:1m:EXTREME:${highBar.t}`,priority:6,provisionalStop:recentHigh,reason:`1m momentum + flow deterioration ${fast.toFixed(2)}/${slow.toFixed(2)}`};
  return null;
}

function directionComponents(direction:TradeDirection,m:Market,s1:Struct,s5:Struct,s15:Struct){
  const opp=direction==="UP"?"DOWN":"UP";let structure_score=0;
  for(const [s,w] of [[s1,1.2],[s5,2.2],[s15,1.0]] as const){if(s.bos===direction)structure_score+=2*w;if(s.bos===opp)structure_score-=2*w;if(s.choch===direction)structure_score+=1.5*w;if(s.choch===opp)structure_score-=1.5*w;if(s.trend===direction)structure_score+=w;if(s.trend===opp)structure_score-=w;}
  const sign=direction==="UP"?1:-1,fast_ofi_component=sign*m.flowFast.ofi*1.8,slow_ofi_component=sign*m.flowSlow.ofi*.9;
  let book_component=0;if(direction==="UP"){if(m.book.pressure>1.08)book_component+=.7;if(m.book.pressure<.82)book_component-=.7;}else{if(m.book.pressure<.92)book_component+=.7;if(m.book.pressure>1.18)book_component-=.7;}
  return{structure_score,fast_ofi_component,slow_ofi_component,book_component,direction_score:structure_score+fast_ofi_component+slow_ofi_component+book_component};
}
function directionScore(direction:TradeDirection,m:Market,s1:Struct,s5:Struct,s15:Struct):number{return directionComponents(direction,m,s1,s5,s15).direction_score;}
function hardCounterStructure(direction:TradeDirection,s1:Struct,s5:Struct):boolean{return direction==="DOWN"?((s5.bos==="UP"&&(s1.trend==="UP"||s1.bos==="UP"))||(s5.trend==="UP"&&s1.bos==="UP")):((s5.bos==="DOWN"&&(s1.trend==="DOWN"||s1.bos==="DOWN"))||(s5.trend==="DOWN"&&s1.bos==="DOWN"));}

function collectCandidates(m:Market,s1:Struct,s5:Struct,s15:Struct):{ranked:SetupCandidate[];telemetry:CandidateTelemetry[]}{
  const raw:SetupCandidate[]=[];
  for(const s of [s1,s5]){
    const seal=m.bars[s.tf].at(-1)?.ct??0;
    if(s.sweep){const direction:TradeDirection=s.sweep==="BULL"?"UP":"DOWN",anchor=direction==="UP"?s.lastLow:s.lastHigh;raw.push({setup:"SWEEP_RECLAIM",direction,trigger:s,triggerIdentity:`SWEEP_RECLAIM:${s.tf}:${seal}:${anchor?.t??"NO_ANCHOR"}`,priority:s.tf==="5m"?8:7,provisionalStop:null,reason:`${s.tf} ${s.sweep} sweep`});}
    if(s.failedBreak){const direction:TradeDirection=s.failedBreak==="BULL"?"UP":"DOWN",anchor=direction==="UP"?s.lastLow:s.lastHigh;raw.push({setup:"FAILED_BREAK",direction,trigger:s,triggerIdentity:`FAILED_BREAK:${s.tf}:${seal}:${anchor?.t??"NO_ANCHOR"}`,priority:s.tf==="5m"?8:7,provisionalStop:null,reason:`${s.tf} ${s.failedBreak} failed break`});}
  }
  if(s5.bos){const p=s5.bos==="UP"?s5.lastHigh:s5.lastLow;if(p&&Math.abs(m.book.mid-p.p)<=s5.atr*.45){const seal=m.bars["5m"].at(-1)?.ct??0;raw.push({setup:"BOS_RETEST",direction:s5.bos,trigger:s5,triggerIdentity:`BOS_RETEST:5m:${seal}:${p.t}`,priority:7,provisionalStop:null,reason:"5m BOS retest"});}}
  const early=recentReversal(m,s1,s5);if(early)raw.push(early);
  const ranked=raw.map(c=>({...c,priority:c.priority+directionScore(c.direction,m,s1,s5,s15)})).sort((a,b)=>b.priority-a.priority||a.setup.localeCompare(b.setup)||a.direction.localeCompare(b.direction));
  const telemetry=ranked.map((c,i)=>{const x=directionComponents(c.direction,m,s1,s5,s15);return{rank:i+1,selected:i===0,setup:c.setup,direction:c.direction,trigger_tf:c.trigger.tf,trigger_identity:c.triggerIdentity,reason:c.reason,base_priority:BASE_PRIORITY[c.setup]+((c.setup==="SWEEP_RECLAIM"||c.setup==="FAILED_BREAK")&&c.trigger.tf==="5m"?1:0),...x,final_priority:c.priority};});
  return{ranked,telemetry};
}

function stopForCandidate(chosen:SetupCandidate|null,direction:Direction){
  if(!chosen||!isTradeDirection(direction))return{price:null as number|null,source:"NONE",structureId:null as string|null};
  if(chosen.setup==="EARLY_REVERSAL")return{price:chosen.provisionalStop,source:"RECENT_7_BAR_EXTREME",structureId:chosen.triggerIdentity};
  if(chosen.setup==="BOS_RETEST"){
    const p=direction==="UP"?chosen.trigger.lastHigh:chosen.trigger.lastLow;
    return{price:p?.p??null,source:"BOS_BROKEN_PIVOT",structureId:p?`${chosen.trigger.tf}:${p.t}:${p.kind}`:null};
  }
  const p=direction==="UP"?chosen.trigger.lastLow:chosen.trigger.lastHigh;
  return{price:p?.p??null,source:chosen.setup==="SWEEP_RECLAIM"?"SWEEP_TRIGGER_PIVOT":"FAILED_BREAK_TRIGGER_PIVOT",structureId:p?`${chosen.trigger.tf}:${p.t}:${p.kind}`:null};
}

function firstVeto(veto:string[]):{first:string|null;stage:string|null}{
  let best:{first:string;stage:string;rank:number;index:number}|null=null;
  veto.forEach((v,index)=>{const root=v.split(":")[0],stage=STAGE[root]??"OTHER",rank=STAGE_ORDER[stage]??STAGE_ORDER.OTHER;if(!best||rank<best.rank||(rank===best.rank&&index<best.index))best={first:v,stage,rank,index};});
  return best?{first:best.first,stage:best.stage}:{first:null,stage:null};
}

export async function candidate(m:Market,sessionId:string,rt:Runtime,cfg:J,tradeNotional:number,at:number,cal:ExecutionCalibration):Promise<CandidateResult>{
  const [s1,s5,s15,s1h,s4h]=await Promise.all(["1m","5m","15m","1h","4h"].map(tf=>structure(tf,m.bars[tf]))),all=[s1,s5,s15,s1h,s4h],combinedFp=await hash(all.map(s=>s.fingerprint).join("|"));
  let regime="RANGE";if(s4h.trend===s1h.trend&&s1h.trend!=="RANGE")regime="TREND_"+s1h.trend;else if(s15.choch||s5.choch)regime="TRANSITION";
  const pool=rt.pos?{ranked:[] as SetupCandidate[],telemetry:[] as CandidateTelemetry[]}:collectCandidates(m,s1,s5,s15),chosen=pool.ranked[0]||null;
  const setup:Setup|null=chosen?.setup??null,direction:Direction=chosen?.direction??"WAIT",trigger=chosen?.trigger??null;
  const fee=n(cfg.fee_bps,10),slip=n(cfg.slippage_bps,1),entry=isTradeDirection(direction)?(direction==="DOWN"?m.book.bid*(1-slip/10000):m.book.ask*(1+slip/10000)):m.book.mid;
  const stop=stopForCandidate(chosen,direction),inv=stop.price;
  const signalAt=trigger?m.bars[trigger.tf].at(-1)!.ct+1:m.bars["1m"].at(-1)!.ct+1;
  const levelPlan=isTradeDirection(direction)?await firstForwardLevel({direction,fill:entry,signalAt,tickSize:m.rules.tickSize,structs:all}):{l1:null,levels:[]};
  const l1=levelPlan.l1,target=l1?.normalized_price??null;
  const cost=buildCostContract({feeOpenBps:fee,feeCloseBps:fee,openingSlippageBps:slip,expectedExitSpreadBps:m.book.spreadBps,expectedExitSlippageBps:slip,expectedFundingBps:m.fundingBpsHold});
  const geometry=validGeometry(direction,entry,inv,target),economics=geometry&&isTradeDirection(direction)?executionEconomics(direction,entry,target!,inv!,cost):null;
  const targetDistanceBps=target?Math.abs(target-entry)/entry*10000:0,stopDistanceBps=inv?Math.abs(entry-inv)/entry*10000:0,fillForwardCost=fillForwardCostBps(cost),referenceCost=referenceRoundtripCostBps(cost,m.book.spreadBps),economicRR=economics?.economic_rr??0;
  const l1Economic=!!(l1&&geometry&&targetDistanceBps>=TARGET_COST_MULTIPLE*fillForwardCost&&economicRR>=MIN_ECONOMIC_RR);
  const targetRole:"L1_EXECUTABLE"|"L1_BLOCKING"|"NO_FORWARD_LEVEL"=!l1?"NO_FORWARD_LEVEL":l1Economic?"L1_EXECUTABLE":"L1_BLOCKING";
  const materialIdentity=setup&&isTradeDirection(direction)&&chosen?[setup,direction,trigger?.tf,chosen.triggerIdentity,stop.structureId??"NO_STOP_ID",stop.source].join("|"):"NO_SIGNAL";
  const episode=await hash([SYMBOL,materialIdentity].join("|")),occurrence=await hash([sessionId,RELEASE_ID,DECISION_REVISION,episode].join("|"));
  const veto:string[]=[];
  if(!setup||!isTradeDirection(direction))veto.push("NO_SIGNAL");
  else{
    if(!l1)veto.push("NO_FORWARD_LEVEL");
    if(!geometry||!inv||Math.abs(entry-inv)<s1.atr*.04)veto.push("STRUCTURE_LEVEL_INCOMPLETE");
    if(l1&&targetDistanceBps<TARGET_COST_MULTIPLE*fillForwardCost)veto.push("TARGET_BELOW_COST");
    if(l1&&economicRR<MIN_ECONOMIC_RR)veto.push("ECONOMIC_RR_TOO_LOW");
    if(direction==="DOWN"&&cfg.allow_shadow_short!==true)veto.push("SHORT_DISABLED");
    if(hardCounterStructure(direction,s1,s5))veto.push("COUNTER_STRUCTURE");
    const opposing=direction==="UP"?(m.flowFast.ofi<=-.25&&m.flowSlow.ofi<=-.20):(m.flowFast.ofi>=.25&&m.flowSlow.ofi>=.20);if(opposing)veto.push("OPPOSING_MULTI_FLOW");
    if(directionScore(direction,m,s1,s5,s15)<-.5)veto.push("DIRECTION_REFEREE_REJECT");
    if(cal.state==="UNAVAILABLE"||!cal.validContract)veto.push("CALIBRATION_UNAVAILABLE");
  }
  const guard=isTradeDirection(direction)&&setup?entryGuard(direction,setup,m.book.mid,entry,m.bars["1m"],[s1,s5,s15,s1h],s1.atr,at):null;if(guard)veto.push(...guard.veto);
  if(cfg.execution_mode==="OBSERVE")veto.push("OBSERVATION_ONLY");
  let raw=.45+(setup==="SWEEP_RECLAIM"?.15:setup==="FAILED_BREAK"?.14:setup==="EARLY_REVERSAL"?.13:setup==="BOS_RETEST"?.10:0);
  if(setup&&isTradeDirection(direction)){if(s5.bos===direction)raw+=.07;if(s1.bos===direction)raw+=.05;if(s15.choch===direction||s5.choch===direction)raw+=.07;const sign=direction==="UP"?1:-1;if(sign*m.flowFast.ofi>.08)raw+=.06;if(sign*m.flowSlow.ofi>.05)raw+=.04;if(direction==="UP"?m.book.pressure>1.08:m.book.pressure<.92)raw+=.05;if(hardCounterStructure(direction,s1,s5))raw-=.15;}raw=clip(raw,0,.92);if(setup&&isTradeDirection(direction)&&raw<.60)veto.push("RAW_CONVICTION_LOW");
  const size=geometry&&l1Economic&&setup&&isTradeDirection(direction)&&inv&&economics?sizePosition({cash:rt.cash,tradeNotional,direction,entry,stop:inv,feeOpenBps:fee,economics,calibration:cal,rules:m.rules,rawConviction:raw}):null;
  if(l1Economic&&setup&&isTradeDirection(direction)&&geometry&&!size)veto.push("MIN_NOTIONAL_OR_RISK_CAP");
  const blocking=firstVeto(veto),calNoEdge=calibrationNoEdge(cal,economicRR);
  const targetPlan={policy:"NEAREST_STRUCTURAL_THEN_ECONOMIC_GATE",target_planner_version:TARGET_PLANNER_VERSION,role:targetRole,l1:l1?{...l1,distance_from_fill_bps:targetDistanceBps}:null,farther_levels:levelPlan.levels.slice(1,8).map(x=>({level_id:x.level_id,price:x.normalized_price,timeframe:x.timeframe,origin:x.origin,confirmed_at:x.confirmed_at})),farther_levels_execution_authority:false,skip_obstacle_live:false,l2_live:false};
  const stopTelemetry={price:inv,source:stop.source,structure_id:stop.structureId,distance_bps:stopDistanceBps,distance_atr_1m:inv&&s1.atr?Math.abs(entry-inv)/s1.atr:null,distance_atr_setup_tf:inv&&trigger?.atr?Math.abs(entry-inv)/trigger.atr:null};
  const calSnapshot={state:cal.state,samples:cal.samples,episodes:cal.episodes,days:cal.days,wins:cal.wins,losses:cal.losses,p:cal.p,wilson_lower:cal.lower,wilson_upper:cal.upper,ambiguous_losses:cal.ambiguousLosses,calibration_no_edge_diagnostic:calNoEdge,calibration_no_edge_authoritative:WARM_RISK_PROMOTION_ENABLED};
  const thesis:J={thesis_id:occurrence,occurrence_id:occurrence,episode_id:episode,material_identity:materialIdentity,trigger_identity:chosen?.triggerIdentity??null,generated_at:new Date(at).toISOString(),decision_time:new Date(at).toISOString(),signal_at:new Date(signalAt).toISOString(),forecast_due_at:new Date(signalAt+MAX_HOLD_MS).toISOString(),setup:setup??"NONE",direction,regime,venue:"SHADOW_PERP",structural_reference_price:m.book.mid,entry_price:entry,invalidation_price:inv,target_price:target,target_role:targetRole,target_plan:targetPlan,stop:stopTelemetry,entry_guard:guard,rr:economicRR,economic_rr:economicRR,target_distance_bps:targetDistanceBps,stop_distance_bps:stopDistanceBps,net_reward_bps:economics?.target_net_reward_bps??0,net_risk_bps:economics?.stop_net_loss_bps??0,fill_forward_cost_bps:fillForwardCost,reference_roundtrip_cost_bps:referenceCost,cost_contract:cost,raw_conviction:setup?raw:null,forecast_probability:null,execution_calibration:calSnapshot,leverage:1,leverage_policy:"PACKAGE1_EVIDENCE_1X_ONLY",why:all.map(s=>`${s.tf}:${s.trend}/${s.sweep||s.failedBreak||s.bos||"NONE"}`),veto:[...new Set(veto)],first_blocking_veto:blocking.first,veto_stage:blocking.stage,structure:{s1,s5,s15,s1h,s4h,fingerprint:combinedFp},flow:{fast:m.flowFast,slow:m.flowSlow,book_pressure:m.book.pressure,spread_bps:m.book.spreadBps,funding_rate:m.fundingRate,funding_bps_hold:m.fundingBpsHold,source:m.source},candidate_set:pool.telemetry,release_id:RELEASE_ID,logic_hash:LOGIC_HASH,strategy_manifest_hash:STRATEGY_MANIFEST_HASH,calibration_family_id:CALIBRATION_FAMILY_ID,db_contract_version:DB_CONTRACT_VERSION,decision_revision:DECISION_REVISION,policy_version:POLICY_VERSION,engine_version:ENGINE_VERSION,metric_version:METRIC_VERSION,resolver_version:RESOLVER_VERSION,entry_guard_version:ENTRY_GUARD_VERSION,target_planner_version:TARGET_PLANNER_VERSION,execution_model_version:EXECUTION_MODEL_VERSION,cost_model_version:COST_MODEL_VERSION,market_data_contract_version:MARKET_DATA_CONTRACT_VERSION,shadow_only:true,live_execution:false,browser_execution:false};
  const decision:DecisionRecord|null=setup&&isTradeDirection(direction)?{occurrence_id:occurrence,session_id:sessionId,episode_id:episode,parent_occurrence_id:null,symbol:SYMBOL,decision_at:new Date(at).toISOString(),signal_at:new Date(signalAt).toISOString(),due_at:new Date(signalAt+MAX_HOLD_MS).toISOString(),setup,direction,regime,venue:"SHADOW_PERP",entry_price:entry,target_price:target,invalidation_price:inv,target_role:targetRole,l1_id:l1?.level_id??null,raw_conviction:raw,forecast_probability:null,release_id:RELEASE_ID,logic_hash:LOGIC_HASH,strategy_manifest_hash:STRATEGY_MANIFEST_HASH,calibration_family_id:CALIBRATION_FAMILY_ID,db_contract_version:DB_CONTRACT_VERSION,policy_version:POLICY_VERSION,decision_revision:DECISION_REVISION,metric_version:METRIC_VERSION,resolver_version:RESOLVER_VERSION,entry_guard_version:ENTRY_GUARD_VERSION,target_planner_version:TARGET_PLANNER_VERSION,execution_model_version:EXECUTION_MODEL_VERSION,cost_model_version:COST_MODEL_VERSION,market_data_contract_version:MARKET_DATA_CONTRACT_VERSION,evidence:{target_plan:targetPlan,stop:stopTelemetry,entry_guard:guard,material_identity:materialIdentity,trigger_identity:chosen?.triggerIdentity??null,structure_fingerprint:combinedFp,execution_calibration:calSnapshot,candidate_set:pool.telemetry,fill_forward_cost_bps:fillForwardCost,reference_roundtrip_cost_bps:referenceCost,economic_rr:economicRR,veto:[...new Set(veto)],first_blocking_veto:blocking.first,veto_stage:blocking.stage},shadow_only:true,live_execution:false}:null;
  const canEnter=!!(setup&&isTradeDirection(direction)&&targetRole==="L1_EXECUTABLE"&&size&&veto.length===0);
  return{thesis,decision,occurrence,episode,combinedFp,last5m:m.bars["5m"].at(-1)!.ct+1,direction,entry,inv,target,targetRole,l1,size,fee,slip,canEnter,candidateEvaluations:pool.telemetry,firstBlockingVeto:blocking.first,vetoStage:blocking.stage};
}