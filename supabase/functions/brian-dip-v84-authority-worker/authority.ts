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
  MIN_ENTRY_QUALITY,
  POLICY_VERSION,
  PROFIT_PROTECT_ARM_BPS,
  PROFIT_PROTECT_GIVEBACK_BPS,
  PROFIT_PROTECT_MOMENTUM_ATR_CEILING,
  RELEASE_ID,
  RESOLVER_VERSION,
  STRATEGY_MANIFEST_HASH,
  SYMBOL,
  TARGET_PLANNER_VERSION,
} from "../_shared/dip_v84_authority_contract.ts";

type AuthoritySize={
  qty:number;notional:number;margin:number;leverage:1;actual_fraction:number;gross_fraction:number;
  fees_open:number;worst_loss:number;risk_fraction:number;allocation:number;risk_policy:string;max_notional_fraction:number;
};
type ScoreState={up:number;down:number;rangePos:number;momentumAtr:number;entryQualityRaw:number;chase:boolean;reason:string[]};
type ProfitMemory={sellVotes?:number;peakBid?:number;peakProfitBps?:number;lastSellReason?:string|null};
type Runtime843=Runtime&{v842?:ProfitMemory};

function validLong(entry:number,stop:number,target:number){return stop<entry&&entry<target;}

function chartScores(m:Market,s1:Struct,s5:Struct,s15:Struct,s1h:Struct,base:CandidateResult):ScoreState{
  const rows=m.bars["1m"].slice(-60),last=rows.at(-1)!;
  const low=Math.min(...rows.map(x=>x.l)),high=Math.max(...rows.map(x=>x.h)),span=Math.max(high-low,s1.atr*.5,1e-9);
  const live=m.book.mid,rangePos=clip((live-low)/span,0,1);
  const c4=rows.at(-4)?.c??last.c,momentumAtr=s1.atr>0?(live-c4)/s1.atr:0;
  let up=0,down=0;const reason:string[]=[];
  const apply=(s:Struct,w:number)=>{
    if(s.trend==="UP")up+=.55*w;else if(s.trend==="DOWN")down+=.55*w;
    if(s.bos==="UP")up+=1.15*w;else if(s.bos==="DOWN")down+=1.15*w;
    if(s.choch==="UP")up+=.95*w;else if(s.choch==="DOWN")down+=.95*w;
    if(s.sweep==="BULL")up+=.85*w;else if(s.sweep==="BEAR")down+=.85*w;
    if(s.failedBreak==="BULL")up+=.70*w;else if(s.failedBreak==="BEAR")down+=.70*w;
  };
  apply(s1,1);apply(s5,1.35);apply(s15,.9);apply(s1h,.4);
  up+=(.5-rangePos)*2.5;down+=(rangePos-.5)*2.5;
  const mom=clip(momentumAtr,-1.5,1.5);up+=mom*.45;down-=mom*.45;
  up+=m.flowFast.ofi*.55+m.flowSlow.ofi*.30;down-=m.flowFast.ofi*.55+m.flowSlow.ofi*.30;
  const book=Math.log(Math.max(.2,Math.min(5,m.book.pressure)));up+=book*.30;down-=book*.30;
  if(base.direction==="UP")up+=.55;else if(base.direction==="DOWN")down+=.25;

  let q=rangePos<=.22?.96:rangePos<=.40?.82:rangePos<=.58?.62:rangePos<=.70?.42:rangePos<=.82?.24:.10;
  if(momentumAtr>1.25)q*=.72;
  if(momentumAtr>2.0)q*=.48;
  if(momentumAtr>3.0)q*=.28;
  if(momentumAtr>4.0)q*=.16;
  if(m.flowFast.ofi<-.18)q*=.82;
  if(s5.bos==="DOWN"&&s1.bos==="DOWN")q*=.72;
  const chase=(rangePos>.76&&momentumAtr>1.25)||(momentumAtr>3.0&&rangePos>.62);
  if(chase){q=Math.min(q,.18);reason.push("BRIAN_CHASE_FOMO_WAIT");}
  if(rangePos<.35)reason.push("PRICE_NEAR_RECENT_LOW");
  if(rangePos>.65)reason.push("PRICE_NEAR_RECENT_HIGH");
  if(momentumAtr>.15)reason.push("LIVE_RECOVERY_MOMENTUM");
  if(momentumAtr<-.15)reason.push("LIVE_DECLINE_MOMENTUM");
  return{up,down,rangePos,momentumAtr,entryQualityRaw:clip(q,0,1),chase,reason};
}

function confidenceFor(up:number,down:number):number{
  const separation=Math.max(0,up-down),strength=Math.max(0,up);
  return clip(.50+separation*.06+strength*.022,.50,.93);
}

function authoritySize(input:{cash:number;tradeNotional:number;entry:number;feeBps:number;allocation:number;rules:Market["rules"];stopNetLossBps:number}):AuthoritySize|null{
  const {cash,tradeNotional,entry,feeBps,rules,stopNetLossBps}=input,allocation=clip(input.allocation,0,1);
  if(!(cash>0&&tradeNotional>0&&entry>0&&allocation>0&&feeBps>=0))return null;
  const feeRate=feeBps/10000,maxCashNotional=cash/(1+feeRate),budget=Math.min(tradeNotional,maxCashNotional)*allocation;
  const rawQty=Math.min(budget/entry,rules.maxQty),qty=Math.floor((rawQty+Number.EPSILON)/rules.stepSize)*rules.stepSize;
  const notional=qty*entry,margin=notional,fees_open=notional*feeRate;
  if(!Number.isFinite(qty)||qty<rules.minQty||notional<rules.minNotional||margin+fees_open>cash+1e-8)return null;
  const worst_loss=Math.max(0,notional*Math.max(0,stopNetLossBps)/10000);
  return{qty,notional,margin,leverage:1,actual_fraction:margin/cash,gross_fraction:notional/cash,fees_open,worst_loss,risk_fraction:cash?worst_loss/cash:0,allocation,risk_policy:"V843_BRIAN_LONG_ONLY_1X",max_notional_fraction:1};
}

function structuralAnchor(s1:Struct,s5:Struct):string{return `UP|1m:${s1.lastLow?.t??"none"}|5m:${s5.lastLow?.t??"none"}`;}

function profitState(rt:Runtime843,m:Market,structs:Struct[]):{
  profitBps:number;peakBid:number;peakProfitBps:number;givebackBps:number;resistanceTouched:boolean;priorSellVotes:number;
}{
  if(!rt.pos||rt.pos.side!=="LONG")return{profitBps:0,peakBid:0,peakProfitBps:0,givebackBps:0,resistanceTouched:false,priorSellVotes:0};
  const mem=(rt.v842??={}) as ProfitMemory;
  const entry=rt.pos.entry,bid=m.book.bid,priorPeak=n(mem.peakBid,entry),peakBid=Math.max(priorPeak,bid);
  const profitBps=(bid-entry)/entry*10000,peakProfitBps=(peakBid-entry)/entry*10000,givebackBps=Math.max(0,(peakBid-bid)/peakBid*10000);
  const resistance=structs.flatMap(s=>[s.lastHigh?.p,s.equalHigh]).map(Number).filter(v=>Number.isFinite(v)&&v>entry);
  const resistanceTouched=resistance.some(v=>peakBid>=v);
  mem.peakBid=peakBid;mem.peakProfitBps=Math.max(n(mem.peakProfitBps),peakProfitBps);
  return{profitBps,peakBid,peakProfitBps:Math.max(n(mem.peakProfitBps),peakProfitBps),givebackBps,resistanceTouched,priorSellVotes:Math.max(0,Math.floor(n(mem.sellVotes)))};
}

export async function authorityCandidate(m:Market,sessionId:string,rt:Runtime,cfg:J,tradeNotional:number,at:number,cal:ExecutionCalibration):Promise<CandidateResult>{
  const base=await package1Candidate(m,sessionId,rt,cfg,tradeNotional,at,cal);
  const S=base.thesis.structure as Record<string,Struct>;
  const s1=S.s1,s5=S.s5,s15=S.s15,s1h=S.s1h,s4h=S.s4h;
  if(!s1||!s5||!s15||!s1h||!s4h)return{...base,canEnter:false,firstBlockingVeto:"DATA_INCONSISTENT",vetoStage:"RUNTIME",thesis:{...base.thesis,veto:["DATA_INCONSISTENT"],decision_authority:"BRIAN"}};

  const scores=chartScores(m,s1,s5,s15,s1h,base),confidence=confidenceFor(scores.up,scores.down);
  const bullish=scores.up>=1.0&&scores.up-scores.down>=.55;
  const rawBearish=scores.down>=1.0&&scores.down-scores.up>=.55;
  const hasLong=rt.pos?.side==="LONG";
  const direction:Direction=!hasLong&&bullish?"UP":"WAIT";
  const baseSetup=String(base.thesis.setup||"NONE"),setup:Setup=(["SWEEP_RECLAIM","FAILED_BREAK","BOS_RETEST","EARLY_REVERSAL"].includes(baseSetup)&&base.direction==="UP"?baseSetup:"EARLY_REVERSAL") as Setup;
  const fee=n(cfg.fee_bps,10),slip=n(cfg.slippage_bps,1),entry=direction==="UP"?m.book.ask*(1+slip/10000):m.book.mid;
  const recent=m.bars["1m"].slice(-12),recentLow=Math.min(...recent.map(x=>x.l));
  let inv:number|null=null,stopSource="AUTHORITY_RECENT_EXTREME",stopStructureId:string|null=null;
  if(direction==="UP"){
    const pivot=Math.min(recentLow,s1.lastLow?.p??recentLow);inv=pivot-s1.atr*.12;stopStructureId=`UP:${s1.lastLow?.t??recent.at(-1)?.t??0}`;
    if(base.direction==="UP"&&base.inv&&base.inv<entry){inv=base.inv;stopSource=String((base.thesis.stop as Record<string,unknown>)?.source||"PACKAGE1_STRUCTURAL_STOP");stopStructureId=String((base.thesis.stop as Record<string,unknown>)?.structure_id||stopStructureId||"")||null;}
  }

  const signalAt=m.bars["1m"].at(-1)!.ct+1,structs=[s1,s5,s15,s1h,s4h];
  const ladderFill=hasLong?m.book.bid:entry;
  const levelPlan=(direction==="UP"||hasLong)?await firstForwardLevel({direction:"UP",fill:ladderFill,signalAt,tickSize:m.rules.tickSize,structs}):{l1:null,levels:[] as StructuralLevel[]};
  const cost=buildCostContract({feeOpenBps:fee,feeCloseBps:fee,openingSlippageBps:slip,expectedExitSpreadBps:m.book.spreadBps,expectedExitSlippageBps:slip,expectedFundingBps:m.fundingBpsHold});
  const viable=direction==="UP"&&inv?levelPlan.levels.map((level,rank)=>{
    const target=level.normalized_price;if(!validLong(entry,inv!,target))return null;
    const economics=executionEconomics("UP",entry,target,inv!,cost);return{level,rank:rank+1,target,economics};
  }).filter(Boolean) as {level:StructuralLevel;rank:number;target:number;economics:ReturnType<typeof executionEconomics>}[]:[];
  const fillCost=fillForwardCostBps(cost);
  const selected=viable.find(x=>x.economics.economic_rr>=1.0&&x.economics.target_net_reward_bps>=fillCost*.80)
    ??viable.slice().sort((a,b)=>b.economics.economic_rr-a.economics.economic_rr)[0]
    ??null;
  const target=hasLong?(levelPlan.levels.find(x=>x.normalized_price>(rt.pos?.target??m.book.bid)+m.rules.tickSize)?.normalized_price??rt.pos?.target??null):(selected?.target??null);
  const economics=selected?.economics??null,geometry=direction==="UP"&&inv!==null&&target!==null&&validLong(entry,inv,target);
  const econQuality=economics?clip((economics.economic_rr/1.6)*.62+clip(economics.target_net_reward_bps/Math.max(fillCost*1.5,1e-9),0,1)*.38,0,1):0;
  const entryQuality=direction==="UP"?clip(scores.entryQualityRaw*econQuality,0,1):0;
  const normalizedConfidence=clip((confidence-.50)/.43,0,1);
  const allocation=direction==="UP"?clip(.12+.88*normalizedConfidence*entryQuality,.12,1):0;
  const size=geometry&&economics&&entryQuality>=MIN_ENTRY_QUALITY&&!scores.chase?authoritySize({cash:rt.cash,tradeNotional,entry,feeBps:fee,allocation,rules:m.rules,stopNetLossBps:economics.stop_net_loss_bps}):null;

  const ps=profitState(rt as Runtime843,m,structs);
  const targetTouched=hasLong&&rt.pos?m.book.bid>=rt.pos.target:false;
  const weakening=m.flowFast.ofi<-.12||scores.momentumAtr<-.20||rawBearish;
  const extended=scores.rangePos>.82||scores.momentumAtr>2.6;
  const profitProtectArmed=hasLong&&ps.peakProfitBps>=PROFIT_PROTECT_ARM_BPS&&(ps.resistanceTouched||scores.rangePos>.58||ps.peakProfitBps>=35);
  const profitGiveback=profitProtectArmed&&ps.givebackBps>=PROFIT_PROTECT_GIVEBACK_BPS&&scores.momentumAtr<=PROFIT_PROTECT_MOMENTUM_ATR_CEILING;
  const latchedSell=hasLong&&ps.priorSellVotes>0&&ps.profitBps>10&&(weakening||scores.momentumAtr<0);
  const strongSell=hasLong&&(rawBearish||(targetTouched&&weakening)||profitGiveback);
  const softSell=hasLong&&!strongSell&&((extended&&weakening&&ps.profitBps>0)||latchedSell);
  const sellVote=strongSell||softSell;
  const sellStrength=strongSell?"STRONG":softSell?"SOFT":"NONE";
  const sellReason=rawBearish?"BEARISH_PRESSURE":targetTouched&&weakening?"ADVISORY_TARGET_WEAKENING":profitGiveback?"PROFIT_GIVEBACK_AFTER_STRUCTURE":latchedSell?"LATCHED_SELL_CONFIRMATION":extended&&weakening?"EXTENDED_WEAKENING":"NONE";
  const mem=(rt as Runtime843).v842;if(mem)mem.lastSellReason=sellReason!=="NONE"?sellReason:mem.lastSellReason??null;
  const action=hasLong?(sellVote?"SELL":"HOLD"):(direction==="UP"&&entryQuality>=MIN_ENTRY_QUALITY&&!scores.chase&&size?"BUY":"WAIT");

  const softEvidence:string[]=[];
  if(rawBearish&&!hasLong)softEvidence.push("LONG_ONLY_BEARISH_WAIT");
  if(scores.chase)softEvidence.push("BRIAN_CHASE_FOMO_WAIT");
  if(direction==="UP"&&entryQuality<MIN_ENTRY_QUALITY)softEvidence.push("ENTRY_QUALITY_LOW");
  if(direction==="UP"&&economics&&economics.target_net_reward_bps<=0)softEvidence.push("TARGET_BELOW_COST");
  if(direction==="UP"&&economics&&economics.economic_rr<1)softEvidence.push("ECONOMIC_RR_LOW");
  if(profitProtectArmed)softEvidence.push("PROFIT_PROTECT_ARMED");
  if(ps.resistanceTouched)softEvidence.push("STRUCTURAL_RESISTANCE_TOUCHED");
  const hard:string[]=[];
  if(cfg.execution_mode==="OBSERVE")hard.push("OBSERVATION_ONLY");
  if(cfg.allow_shadow_short!==false)hard.push("LONG_ONLY_CONTRACT_MISMATCH");
  if(direction==="UP"&&(!inv||!target||!geometry))hard.push("INVALID_GEOMETRY_OR_TARGET");
  if(direction==="UP"&&entryQuality>=MIN_ENTRY_QUALITY&&!scores.chase&&geometry&&!size)hard.push("MIN_NOTIONAL_OR_AVAILABLE_CASH");

  const triggerIdentity=base.direction==="UP"&&typeof base.thesis.trigger_identity==="string"?String(base.thesis.trigger_identity):`AUTHORITY:${setup}:${structuralAnchor(s1,s5)}`;
  const materialIdentity=direction==="UP"?["AUTHORITY_LONG",setup,structuralAnchor(s1,s5),stopStructureId??"NO_STOP"].join("|"):"AUTHORITY_LONG_WAIT";
  const episode=direction==="UP"?await hash([SYMBOL,materialIdentity].join("|")):await hash([SYMBOL,"AUTHORITY_LONG_WAIT",structuralAnchor(s1,s5)].join("|"));
  const occurrence=await hash([sessionId,RELEASE_ID,DECISION_REVISION,episode].join("|"));
  const targetDistanceBps=target?Math.abs(target-entry)/Math.max(entry,1e-9)*10000:0,stopDistanceBps=inv?Math.abs(entry-inv)/Math.max(entry,1e-9)*10000:0,economicRR=economics?.economic_rr??0;
  const targetPlan={policy:"BRIAN_DYNAMIC_ADVISORY_TARGET",target_planner_version:TARGET_PLANNER_VERSION,hard_take_profit:false,trailing_thesis:true,selected_rank:selected?.rank??null,selected_level:selected?.level??null,levels:levelPlan.levels.slice(0,10).map((x,i)=>({rank:i+1,level_id:x.level_id,price:x.normalized_price,timeframe:x.timeframe,origin:x.origin,confirmed_at:x.confirmed_at}))};
  const stopTelemetry={price:inv,source:stopSource,structure_id:stopStructureId,distance_bps:stopDistanceBps,distance_atr_1m:inv&&s1.atr?Math.abs(entry-inv)/s1.atr:null,distance_atr_setup_tf:inv&&s1.atr?Math.abs(entry-inv)/s1.atr:null};
  const calSnapshot={state:cal.state,samples:cal.samples,episodes:cal.episodes,days:cal.days,wins:cal.wins,losses:cal.losses,p:cal.p,wilson_lower:cal.lower,wilson_upper:cal.upper,ambiguous_losses:cal.ambiguousLosses,entry_blocking:false};
  const reason=[...scores.reason,rawBearish?"BEARISH_PRESSURE_PRESENT":"",profitProtectArmed?"PROFIT_PROTECT_ARMED":"",profitGiveback?"PROFIT_GIVEBACK_TRIGGER":"",`score up=${scores.up.toFixed(2)} down=${scores.down.toFixed(2)}`,`range=${scores.rangePos.toFixed(2)}`,`live-mom=${scores.momentumAtr.toFixed(2)}ATR`,`entryQ=${entryQuality.toFixed(2)}`].filter(Boolean);
  const targetRole="L1_EXECUTABLE" as const;
  const thesis:J={
    ...base.thesis,
    thesis_id:occurrence,occurrence_id:occurrence,episode_id:episode,material_identity:materialIdentity,trigger_identity:triggerIdentity,
    generated_at:new Date(at).toISOString(),decision_time:new Date(at).toISOString(),signal_at:new Date(signalAt).toISOString(),forecast_due_at:new Date(signalAt+MAX_HOLD_MS).toISOString(),
    setup:direction==="UP"?setup:"NONE",direction,regime:String(base.thesis.regime||"RANGE"),venue:"SHADOW_PERP",entry_price:entry,invalidation_price:inv,target_price:target,
    target_role:target?targetRole:"NO_FORWARD_LEVEL",target_plan:targetPlan,stop:stopTelemetry,entry_guard:{mode:"BRIAN_LIVE_ENTRY_QUALITY",package1_observation:base.thesis.entry_guard??null},
    rr:economicRR,economic_rr:economicRR,target_distance_bps:targetDistanceBps,stop_distance_bps:stopDistanceBps,net_reward_bps:economics?.target_net_reward_bps??0,net_risk_bps:economics?.stop_net_loss_bps??0,
    fill_forward_cost_bps:fillCost,reference_roundtrip_cost_bps:referenceRoundtripCostBps(cost,m.book.spreadBps),cost_contract:cost,
    raw_conviction:confidence,forecast_probability:null,execution_calibration:calSnapshot,leverage:1,leverage_policy:"V843_BRIAN_LONG_ONLY_1X",
    decision_authority:"BRIAN",authority_action:action,authority_confidence:confidence,authority_entry_quality:entryQuality,authority_allocation:size?.allocation??allocation,
    authority_scores:{up:scores.up,down:scores.down,range_position:scores.rangePos,momentum_atr:scores.momentumAtr,entry_quality_raw:scores.entryQualityRaw,sell_vote:sellVote,sell_strength:sellStrength,sell_reason:sellReason,profit_bps:ps.profitBps,peak_bid:ps.peakBid,peak_profit_bps:ps.peakProfitBps,giveback_bps:ps.givebackBps,profit_protect_armed:profitProtectArmed,resistance_touched:ps.resistanceTouched,target_touched:targetTouched},authority_reason:reason,
    decision_market_price:m.book.mid,decision_market_received_at:new Date(m.book.receivedAt).toISOString(),soft_evidence:softEvidence,veto:hard,first_blocking_veto:hard[0]??null,veto_stage:hard.length?"TECHNICAL_RAIL":null,
    release_id:RELEASE_ID,logic_hash:LOGIC_HASH,strategy_manifest_hash:STRATEGY_MANIFEST_HASH,calibration_family_id:CALIBRATION_FAMILY_ID,db_contract_version:DB_CONTRACT_VERSION,decision_revision:DECISION_REVISION,policy_version:POLICY_VERSION,engine_version:ENGINE_VERSION,metric_version:METRIC_VERSION,resolver_version:RESOLVER_VERSION,entry_guard_version:ENTRY_GUARD_VERSION,target_planner_version:TARGET_PLANNER_VERSION,execution_model_version:EXECUTION_MODEL_VERSION,cost_model_version:COST_MODEL_VERSION,market_data_contract_version:MARKET_DATA_CONTRACT_VERSION,shadow_only:true,live_execution:false,browser_execution:false,
  };

  const decision:DecisionRecord|null=direction==="UP"&&target&&inv?{
    occurrence_id:occurrence,session_id:sessionId,episode_id:episode,parent_occurrence_id:null,symbol:SYMBOL,decision_at:new Date(at).toISOString(),signal_at:new Date(signalAt).toISOString(),due_at:new Date(signalAt+MAX_HOLD_MS).toISOString(),setup,direction:"UP",regime:String(base.thesis.regime||"RANGE"),venue:"SHADOW_PERP",entry_price:entry,target_price:target,invalidation_price:inv,target_role:targetRole,l1_id:selected?.level.level_id??null,raw_conviction:confidence,forecast_probability:null,
    release_id:RELEASE_ID,logic_hash:LOGIC_HASH,strategy_manifest_hash:STRATEGY_MANIFEST_HASH,calibration_family_id:CALIBRATION_FAMILY_ID,db_contract_version:DB_CONTRACT_VERSION,policy_version:POLICY_VERSION,decision_revision:DECISION_REVISION,metric_version:METRIC_VERSION,resolver_version:RESOLVER_VERSION,entry_guard_version:ENTRY_GUARD_VERSION,target_planner_version:TARGET_PLANNER_VERSION,execution_model_version:EXECUTION_MODEL_VERSION,cost_model_version:COST_MODEL_VERSION,market_data_contract_version:MARKET_DATA_CONTRACT_VERSION,
    evidence:{decision_authority:"BRIAN",authority_action:action,authority_confidence:confidence,authority_entry_quality:entryQuality,authority_allocation:size?.allocation??allocation,authority_reason:reason,authority_scores:(thesis.authority_scores as J),target_plan:targetPlan,stop:stopTelemetry,soft_evidence:softEvidence,technical_rails:hard,execution_calibration:calSnapshot,fill_forward_cost_bps:fillCost,reference_roundtrip_cost_bps:referenceRoundtripCostBps(cost,m.book.spreadBps),economic_rr:economicRR,decision_market_price:m.book.mid},shadow_only:true,live_execution:false,
  }:null;
  const canEnter=!!(decision&&size&&hard.length===0&&selected&&action==="BUY");
  return{...base,thesis,decision,occurrence,episode,direction,entry,inv,target,targetRole:target?targetRole:"NO_FORWARD_LEVEL",l1:selected?.level??levelPlan.l1??null,size:size as CandidateResult["size"],canEnter,firstBlockingVeto:hard[0]??null,vetoStage:hard.length?"TECHNICAL_RAIL":null};
}