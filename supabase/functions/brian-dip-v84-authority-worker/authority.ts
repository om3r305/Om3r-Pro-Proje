import { candidate as package1Candidate, type CandidateResult, type DecisionRecord, type Direction } from "../brian-dip-v84-worker/decision.ts";
import { buildCostContract, executionEconomics, fillForwardCostBps, referenceRoundtripCostBps } from "../brian-dip-v84-worker/cost.ts";
import { firstForwardLevel, type StructuralLevel } from "../brian-dip-v84-worker/levels.ts";
import type { Market } from "../brian-dip-v84-worker/market.ts";
import type { ExecutionCalibration, J, Runtime, Setup, Struct } from "../_shared/dip_v84_contract.ts";
import { clip, hash, n } from "../_shared/dip_v84_contract.ts";
import {
  CALIBRATION_FAMILY_ID,
  COST_MODEL_VERSION,
  DB_CONTRACT_VERSION,
  DECISION_REVISION,
  ENGINE_VERSION,
  ENTRY_GUARD_VERSION,
  EXECUTION_MODEL_VERSION,
  LOGIC_HASH,
  MARKET_DATA_CONTRACT_VERSION,
  MAX_HOLD_MS,
  METRIC_VERSION,
  POLICY_VERSION,
  RELEASE_ID,
  RESOLVER_VERSION,
  STRATEGY_MANIFEST_HASH,
  SYMBOL,
  TARGET_PLANNER_VERSION,
} from "../_shared/dip_v84_authority_contract.ts";

const SOFT = new Set([
  "TARGET_BELOW_COST",
  "ECONOMIC_RR_TOO_LOW",
  "WAIT_RETEST",
  "ENTRY_TOO_LATE",
  "DIRECTION_REFEREE_REJECT",
  "COUNTER_STRUCTURE",
  "OPPOSING_MULTI_FLOW",
  "RAW_CONVICTION_LOW",
  "CALIBRATION_UNAVAILABLE",
]);

type AuthoritySize={
  qty:number;notional:number;margin:number;leverage:1;actual_fraction:number;gross_fraction:number;
  fees_open:number;worst_loss:number;risk_fraction:number;allocation:number;risk_policy:string;max_notional_fraction:number;
};

type ScoreState={up:number;down:number;rangePos:number;momentumAtr:number;reason:string[]};

function tradeDir(x:unknown):x is "UP"|"DOWN"{return x==="UP"||x==="DOWN";}
function validGeometry(direction:"UP"|"DOWN",entry:number,stop:number,target:number){return direction==="UP"?stop<entry&&entry<target:target<entry&&entry<stop;}

function chartScores(m:Market, s1:Struct,s5:Struct,s15:Struct,s1h:Struct,base:CandidateResult):ScoreState{
  const rows=m.bars["1m"].slice(-60),last=rows.at(-1)!;
  const low=Math.min(...rows.map(x=>x.l)),high=Math.max(...rows.map(x=>x.h)),span=Math.max(high-low,s1.atr*.5,1e-9);
  const rangePos=clip((m.book.mid-low)/span,0,1);
  const c4=rows.at(-4)?.c??last.c,momentumAtr=s1.atr>0?(last.c-c4)/s1.atr:0;
  let up=0,down=0;const reason:string[]=[];
  const apply=(s:Struct,w:number)=>{
    if(s.trend==="UP")up+=.55*w; else if(s.trend==="DOWN")down+=.55*w;
    if(s.bos==="UP")up+=1.15*w; else if(s.bos==="DOWN")down+=1.15*w;
    if(s.choch==="UP")up+=.95*w; else if(s.choch==="DOWN")down+=.95*w;
    if(s.sweep==="BULL")up+=.85*w; else if(s.sweep==="BEAR")down+=.85*w;
    if(s.failedBreak==="BULL")up+=.70*w; else if(s.failedBreak==="BEAR")down+=.70*w;
  };
  apply(s1,1);apply(s5,1.45);apply(s15,.9);apply(s1h,.35);
  // Buying near the recent low and selling near the recent high is part of Brian's thesis, not an external veto.
  up+=(0.5-rangePos)*3.2;down+=(rangePos-0.5)*3.2;
  const mom=clip(momentumAtr,-2.5,2.5);up+=mom*.65;down-=mom*.65;
  up+=m.flowFast.ofi*.75+m.flowSlow.ofi*.35;down-=m.flowFast.ofi*.75+m.flowSlow.ofi*.35;
  const book=Math.log(Math.max(.2,Math.min(5,m.book.pressure)));up+=book*.45;down-=book*.45;
  if(base.direction==="UP")up+=.9;if(base.direction==="DOWN")down+=.9;
  if(rangePos>.82){up-=1.25;reason.push("UP_CHASE_PENALTY_NEAR_RANGE_HIGH");}
  if(rangePos<.18){down-=1.25;reason.push("DOWN_CHASE_PENALTY_NEAR_RANGE_LOW");}
  if(rangePos<.35)reason.push("PRICE_NEAR_RECENT_LOW");
  if(rangePos>.65)reason.push("PRICE_NEAR_RECENT_HIGH");
  if(momentumAtr>.15)reason.push("RECOVERY_MOMENTUM");
  if(momentumAtr<-.15)reason.push("DECLINE_MOMENTUM");
  return{up,down,rangePos,momentumAtr,reason};
}

function confidenceFor(best:number,other:number):number{
  const separation=Math.max(0,best-other),strength=Math.max(0,best);
  return clip(.50+separation*.055+strength*.025,.50,.92);
}

function authoritySize(input:{cash:number;tradeNotional:number;entry:number;feeBps:number;allocation:number;rules:Market["rules"];stopNetLossBps:number}):AuthoritySize|null{
  const {cash,tradeNotional,entry,feeBps,rules,stopNetLossBps}=input,allocation=clip(input.allocation,0,1);
  if(!(cash>0&&tradeNotional>0&&entry>0&&allocation>0&&feeBps>=0))return null;
  const feeRate=feeBps/10000,maxCashNotional=cash/(1+feeRate),budget=Math.min(tradeNotional,maxCashNotional)*allocation;
  const rawQty=Math.min(budget/entry,rules.maxQty),qty=Math.floor((rawQty+Number.EPSILON)/rules.stepSize)*rules.stepSize;
  const notional=qty*entry,margin=notional,fees_open=notional*feeRate;
  if(!Number.isFinite(qty)||qty<rules.minQty||notional<rules.minNotional||margin+fees_open>cash+1e-8)return null;
  const worst_loss=Math.max(0,notional*Math.max(0,stopNetLossBps)/10000);
  return{qty,notional,margin,leverage:1,actual_fraction:margin/cash,gross_fraction:notional/cash,fees_open,worst_loss,risk_fraction:cash?worst_loss/cash:0,allocation,risk_policy:"V841_BRIAN_AUTHORITY_1X",max_notional_fraction:1};
}

function structuralAnchor(direction:"UP"|"DOWN",s1:Struct,s5:Struct):string{
  const a=direction==="UP"?s1.lastLow:s1.lastHigh,b=direction==="UP"?s5.lastLow:s5.lastHigh;
  return `${direction}|1m:${a?.t??"none"}|5m:${b?.t??"none"}`;
}

export async function authorityCandidate(m:Market,sessionId:string,rt:Runtime,cfg:J,tradeNotional:number,at:number,cal:ExecutionCalibration):Promise<CandidateResult>{
  // Package-1 candidate generation remains useful as evidence. It no longer has independent strategy authority.
  const base=await package1Candidate(m,sessionId,rt,cfg,tradeNotional,at,cal);
  const S=base.thesis.structure as Record<string,Struct>;
  const s1=S.s1,s5=S.s5,s15=S.s15,s1h=S.s1h,s4h=S.s4h;
  if(!s1||!s5||!s15||!s1h||!s4h)return{...base,canEnter:false,firstBlockingVeto:"DATA_INCONSISTENT",vetoStage:"RUNTIME",thesis:{...base.thesis,veto:["DATA_INCONSISTENT"],decision_authority:"BRIAN"}};

  const scores=chartScores(m,s1,s5,s15,s1h,base),bestDir:scores.up>=scores.down?"UP":"DOWN",bestScore=Math.max(scores.up,scores.down),otherScore=Math.min(scores.up,scores.down),confidence=confidenceFor(bestScore,otherScore);
  // WAIT is Brian's own conclusion when the chart does not have enough directional separation yet.
  const direction:Direction=bestScore>=.75&&Math.abs(scores.up-scores.down)>=.35?bestDir:"WAIT";
  const baseSetup=String(base.thesis.setup||"NONE"),setup:Setup=(["SWEEP_RECLAIM","FAILED_BREAK","BOS_RETEST","EARLY_REVERSAL"].includes(baseSetup)&&base.direction===direction?baseSetup:"EARLY_REVERSAL") as Setup;
  const fee=n(cfg.fee_bps,10),slip=n(cfg.slippage_bps,1);
  const entry=direction==="UP"?m.book.ask*(1+slip/10000):direction==="DOWN"?m.book.bid*(1-slip/10000):m.book.mid;
  const recent=m.bars["1m"].slice(-12),recentLow=Math.min(...recent.map(x=>x.l)),recentHigh=Math.max(...recent.map(x=>x.h));
  let inv:number|null=null,stopSource="AUTHORITY_RECENT_EXTREME",stopStructureId:string|null=null;
  if(direction==="UP"){
    const pivot=Math.min(recentLow,s1.lastLow?.p??recentLow);inv=pivot-s1.atr*.12;stopStructureId=`UP:${s1.lastLow?.t??recent.at(-1)?.t??0}`;
  }else if(direction==="DOWN"){
    const pivot=Math.max(recentHigh,s1.lastHigh?.p??recentHigh);inv=pivot+s1.atr*.12;stopStructureId=`DOWN:${s1.lastHigh?.t??recent.at(-1)?.t??0}`;
  }
  if(direction!=="WAIT"&&base.direction===direction&&base.inv&&validGeometry(direction,entry,base.inv,direction==="UP"?entry+1:entry-1)){
    // Preserve a structurally tighter Package-1 stop when it is on the correct side; it is evidence selected by Brian, not a veto.
    inv=base.inv;stopSource=String((base.thesis.stop as Record<string,unknown>)?.source||"PACKAGE1_STRUCTURAL_STOP");stopStructureId=String((base.thesis.stop as Record<string,unknown>)?.structure_id||stopStructureId||"")||null;
  }

  const signalAt=m.bars["1m"].at(-1)!.ct+1,structs=[s1,s5,s15,s1h,s4h];
  const levelPlan=direction!=="WAIT"?await firstForwardLevel({direction,fill:entry,signalAt,tickSize:m.rules.tickSize,structs}):{l1:null,levels:[] as StructuralLevel[]};
  const cost=buildCostContract({feeOpenBps:fee,feeCloseBps:fee,openingSlippageBps:slip,expectedExitSpreadBps:m.book.spreadBps,expectedExitSlippageBps:slip,expectedFundingBps:m.fundingBpsHold});
  const viable=direction!=="WAIT"&&inv?levelPlan.levels.map((level,rank)=>{
    const target=level.normalized_price;if(!validGeometry(direction,entry,inv!,target))return null;
    const economics=executionEconomics(direction,entry,target,inv!,cost);return{level,rank:rank+1,target,economics};
  }).filter(Boolean) as {level:StructuralLevel;rank:number;target:number;economics:ReturnType<typeof executionEconomics>}[]:[];
  const positive=viable.filter(x=>x.economics.target_net_reward_bps>0&&x.economics.stop_net_loss_bps>0);
  const desiredIndex=confidence>=.80?2:confidence>=.68?1:0;
  const selected=positive[Math.min(desiredIndex,Math.max(0,positive.length-1))]??null;
  const target=selected?.target??null,economics=selected?.economics??null;
  const geometry=direction!=="WAIT"&&inv!==null&&target!==null&&validGeometry(direction,entry,inv,target);

  const allocation=direction==="WAIT"?0:clip(.20+Math.max(0,confidence-.55)/.35*.80,.20,1);
  const size=geometry&&economics?authoritySize({cash:rt.cash,tradeNotional,entry,feeBps:fee,allocation,rules:m.rules,stopNetLossBps:economics.stop_net_loss_bps}):null;

  const oldVeto=Array.isArray(base.thesis.veto)?(base.thesis.veto as unknown[]).map(String):[];
  const softEvidence=[...new Set(oldVeto.filter(v=>SOFT.has(v.split(":")[0])))];
  const hard:string[]=[];
  if(cfg.execution_mode==="OBSERVE")hard.push("OBSERVATION_ONLY");
  if(direction==="DOWN"&&cfg.allow_shadow_short!==true)hard.push("SHORT_DISABLED");
  if(direction!=="WAIT"&&(!inv||!target||!geometry))hard.push("INVALID_GEOMETRY_OR_TARGET");
  if(direction!=="WAIT"&&geometry&&!size)hard.push("MIN_NOTIONAL_OR_AVAILABLE_CASH");
  const action=direction==="UP"?"LONG":direction==="DOWN"?"SHORT":"WAIT";
  const brianWaitReason=direction==="WAIT"?"CHART_NOT_DECISIVE":!selected?"NO_POSITIVE_NET_STRUCTURAL_TARGET":null;
  if(brianWaitReason)softEvidence.push(brianWaitReason);

  const triggerIdentity=base.direction===direction&&typeof base.thesis.trigger_identity==="string"?String(base.thesis.trigger_identity):`AUTHORITY:${setup}:${structuralAnchor(bestDir,s1,s5)}`;
  const materialIdentity=direction!=="WAIT"?["AUTHORITY",setup,direction,structuralAnchor(direction,s1,s5),stopStructureId??"NO_STOP"].join("|"):"AUTHORITY_WAIT";
  const episode=direction!=="WAIT"?await hash([SYMBOL,materialIdentity].join("|")):await hash([SYMBOL,"AUTHORITY_WAIT",structuralAnchor(bestDir,s1,s5)].join("|"));
  const occurrence=await hash([sessionId,RELEASE_ID,DECISION_REVISION,episode].join("|"));
  const targetDistanceBps=target?Math.abs(target-entry)/entry*10000:0,stopDistanceBps=inv?Math.abs(entry-inv)/entry*10000:0;
  const economicRR=economics?.economic_rr??0;
  const targetPlan={policy:"BRIAN_STRUCTURAL_LADDER",target_planner_version:TARGET_PLANNER_VERSION,l1_mandatory:false,selected_rank:selected?.rank??null,selected_level:selected?.level??null,levels:levelPlan.levels.slice(0,8).map((x,i)=>({rank:i+1,level_id:x.level_id,price:x.normalized_price,timeframe:x.timeframe,origin:x.origin,confirmed_at:x.confirmed_at})),strategy_cost_gate:false};
  const stopTelemetry={price:inv,source:stopSource,structure_id:stopStructureId,distance_bps:stopDistanceBps,distance_atr_1m:inv&&s1.atr?Math.abs(entry-inv)/s1.atr:null,distance_atr_setup_tf:inv&&s1.atr?Math.abs(entry-inv)/s1.atr:null};
  const calSnapshot={state:cal.state,samples:cal.samples,episodes:cal.episodes,days:cal.days,wins:cal.wins,losses:cal.losses,p:cal.p,wilson_lower:cal.lower,wilson_upper:cal.upper,ambiguous_losses:cal.ambiguousLosses,entry_blocking:false};
  const reason=[...scores.reason,`score up=${scores.up.toFixed(2)} down=${scores.down.toFixed(2)}`,`range=${scores.rangePos.toFixed(2)}`,`mom=${scores.momentumAtr.toFixed(2)}ATR`];
  const targetRole="L1_EXECUTABLE" as const; // compatibility column; evidence.target_plan carries Brian-selected rank.
  const thesis:J={
    ...base.thesis,
    thesis_id:occurrence,occurrence_id:occurrence,episode_id:episode,material_identity:materialIdentity,trigger_identity:triggerIdentity,
    generated_at:new Date(at).toISOString(),decision_time:new Date(at).toISOString(),signal_at:new Date(signalAt).toISOString(),forecast_due_at:new Date(signalAt+MAX_HOLD_MS).toISOString(),
    setup:direction==="WAIT"?"NONE":setup,direction,regime:String(base.thesis.regime||"RANGE"),venue:"SHADOW_PERP",entry_price:entry,invalidation_price:inv,target_price:target,
    target_role:target?targetRole:"NO_FORWARD_LEVEL",target_plan:targetPlan,stop:stopTelemetry,entry_guard:{mode:"BRIAN_SETUP_SPECIFIC",package1_observation:base.thesis.entry_guard??null},
    rr:economicRR,economic_rr:economicRR,target_distance_bps:targetDistanceBps,stop_distance_bps:stopDistanceBps,net_reward_bps:economics?.target_net_reward_bps??0,net_risk_bps:economics?.stop_net_loss_bps??0,
    fill_forward_cost_bps:fillForwardCostBps(cost),reference_roundtrip_cost_bps:referenceRoundtripCostBps(cost,m.book.spreadBps),cost_contract:cost,
    raw_conviction:confidence,forecast_probability:null,execution_calibration:calSnapshot,leverage:1,leverage_policy:"V841_BRIAN_AUTHORITY_1X",
    decision_authority:"BRIAN",authority_action:action,authority_confidence:confidence,authority_allocation:size?.allocation??allocation,authority_scores:{up:scores.up,down:scores.down,range_position:scores.rangePos,momentum_atr:scores.momentumAtr},authority_reason:reason,
    soft_evidence:softEvidence,veto:hard,first_blocking_veto:hard[0]??null,veto_stage:hard.length?"TECHNICAL_RAIL":null,
    release_id:RELEASE_ID,logic_hash:LOGIC_HASH,strategy_manifest_hash:STRATEGY_MANIFEST_HASH,calibration_family_id:CALIBRATION_FAMILY_ID,db_contract_version:DB_CONTRACT_VERSION,decision_revision:DECISION_REVISION,policy_version:POLICY_VERSION,engine_version:ENGINE_VERSION,metric_version:METRIC_VERSION,resolver_version:RESOLVER_VERSION,entry_guard_version:ENTRY_GUARD_VERSION,target_planner_version:TARGET_PLANNER_VERSION,execution_model_version:EXECUTION_MODEL_VERSION,cost_model_version:COST_MODEL_VERSION,market_data_contract_version:MARKET_DATA_CONTRACT_VERSION,shadow_only:true,live_execution:false,browser_execution:false,
  };

  const decision:DecisionRecord|null=direction!=="WAIT"&&target&&inv?{
    occurrence_id:occurrence,session_id:sessionId,episode_id:episode,parent_occurrence_id:null,symbol:SYMBOL,decision_at:new Date(at).toISOString(),signal_at:new Date(signalAt).toISOString(),due_at:new Date(signalAt+MAX_HOLD_MS).toISOString(),setup,direction,regime:String(base.thesis.regime||"RANGE"),venue:"SHADOW_PERP",entry_price:entry,target_price:target,invalidation_price:inv,target_role:targetRole,l1_id:selected?.level.level_id??null,raw_conviction:confidence,forecast_probability:null,
    release_id:RELEASE_ID,logic_hash:LOGIC_HASH,strategy_manifest_hash:STRATEGY_MANIFEST_HASH,calibration_family_id:CALIBRATION_FAMILY_ID,db_contract_version:DB_CONTRACT_VERSION,policy_version:POLICY_VERSION,decision_revision:DECISION_REVISION,metric_version:METRIC_VERSION,resolver_version:RESOLVER_VERSION,entry_guard_version:ENTRY_GUARD_VERSION,target_planner_version:TARGET_PLANNER_VERSION,execution_model_version:EXECUTION_MODEL_VERSION,cost_model_version:COST_MODEL_VERSION,market_data_contract_version:MARKET_DATA_CONTRACT_VERSION,
    evidence:{decision_authority:"BRIAN",authority_action:action,authority_confidence:confidence,authority_allocation:size?.allocation??allocation,authority_reason:reason,authority_scores:{up:scores.up,down:scores.down,range_position:scores.rangePos,momentum_atr:scores.momentumAtr},target_plan:targetPlan,stop:stopTelemetry,soft_evidence:softEvidence,technical_rails:hard,execution_calibration:calSnapshot,fill_forward_cost_bps:fillForwardCostBps(cost),reference_roundtrip_cost_bps:referenceRoundtripCostBps(cost,m.book.spreadBps),economic_rr:economicRR},shadow_only:true,live_execution:false,
  }:null;

  const canEnter=!!(decision&&size&&hard.length===0&&selected);
  return{...base,thesis,decision,occurrence,episode,direction,entry,inv,target,targetRole:target?targetRole:"NO_FORWARD_LEVEL",l1:selected?.level??null,size:size as CandidateResult["size"],canEnter,firstBlockingVeto:hard[0]??null,vetoStage:hard.length?"TECHNICAL_RAIL":null};
}
