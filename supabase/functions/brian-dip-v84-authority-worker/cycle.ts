import type { ExecutionCalibration, J, Runtime, Struct } from "../_shared/dip_v84_contract.ts";
import {
  CHASE_MOMENTUM_ATR,
  CHASE_RANGE_POSITION,
  HARVEST_BREAK_EVEN_BUFFER_BPS,
  HARVEST_GIVEBACK_BPS,
  MIN_ENTRY_QUALITY,
  MIN_FORECAST_NET_EDGE_BPS,
  MIN_THESIS_HOLD_MS,
  RANGE_PRIMARY_RANK_CAP,
  REBASE_PULLBACK_ATR,
  REBASE_PULLBACK_BPS,
  REBASE_RECLAIM_ATR,
  STRONG_PRIMARY_RANK_CAP,
  TARGET_PLANNER_VERSION,
} from "../_shared/dip_v84_authority_contract.ts";
import { buildCostContract, executionEconomics, fillForwardCostBps, referenceRoundtripCostBps } from "../brian-dip-v84-worker/cost.ts";
import type { CandidateResult } from "../brian-dip-v84-worker/decision.ts";
import type { Market } from "../brian-dip-v84-worker/market.ts";
import { authorityCandidate as legacyAuthorityCandidate } from "./authority.ts";

type Level={rank:number;level_id?:string;price:number;timeframe?:string;origin?:string;confirmed_at?:number};
type CycleMemory={
  sellVotes?:number;lastExitAt?:number|null;lastExitPrice?:number|null;lastExitReason?:string|null;lastTargetRaiseAt?:number|null;
  lastPeakPrice?:number|null;postExitLow?:number|null;rebaseReady?:boolean;
};
type Runtime844=Runtime&{v842?:CycleMemory};

function num(v:unknown,fallback=0){const x=Number(v);return Number.isFinite(x)?x:fallback;}
function obj(v:unknown):J{return v&&typeof v==="object"&&!Array.isArray(v)?v as J:{};}
function arr(v:unknown):unknown[]{return Array.isArray(v)?v:[];}
function pushUnique(a:string[],v:string){if(!a.includes(v))a.push(v);}

function normalizedLevels(thesis:J,entry:number):Level[]{
  const plan=obj(thesis.target_plan);
  return arr(plan.levels).map(x=>obj(x)).map(x=>({
    rank:Math.max(1,Math.floor(num(x.rank,999))),
    level_id:typeof x.level_id==="string"?x.level_id:undefined,
    price:num(x.price),
    timeframe:typeof x.timeframe==="string"?x.timeframe:undefined,
    origin:typeof x.origin==="string"?x.origin:undefined,
    confirmed_at:num(x.confirmed_at)||undefined,
  })).filter(x=>x.price>entry&&Number.isFinite(x.price)).sort((a,b)=>a.rank-b.rank||a.price-b.price);
}

function scoreState(thesis:J){
  const s=obj(thesis.authority_scores);
  return{
    up:num(s.up),down:num(s.down),rangePos:num(s.range_position,.5),momentumAtr:num(s.momentum_atr),entryQuality:num(thesis.authority_entry_quality),
    profitBps:num(s.profit_bps),peakProfitBps:num(s.peak_profit_bps),givebackBps:num(s.giveback_bps),peakBid:num(s.peak_bid),
    resistanceTouched:s.resistance_touched===true,
  };
}

function structureDamage(thesis:J):boolean{
  const S=obj(thesis.structure),s1=obj(S.s1),s5=obj(S.s5);
  return s1.bos==="DOWN"||s1.choch==="DOWN"||(s1.trend==="DOWN"&&s5.trend==="DOWN")||(s1.failedBreak==="BEAR"&&s5.trend!=="UP");
}

function cycleMemory(rt:Runtime844):CycleMemory{
  if(!rt.v842)rt.v842={sellVotes:0,lastExitAt:null,lastExitPrice:null,lastExitReason:null,lastTargetRaiseAt:null,lastPeakPrice:null,postExitLow:null,rebaseReady:false};
  return rt.v842;
}

function rebaseState(rt:Runtime844,m:Market,thesis:J):{required:boolean;ready:boolean;pullbackBps:number;reclaimAtr:number;reference:number|null;low:number|null}{
  const mem=cycleMemory(rt),s1=obj(obj(thesis.structure).s1),atr=Math.max(num(s1.atr),m.rules.tickSize);
  if(!mem.lastExitAt||!mem.lastExitPrice)return{required:false,ready:true,pullbackBps:0,reclaimAtr:0,reference:null,low:null};
  const reference=Math.max(num(mem.lastExitPrice),num(mem.lastPeakPrice),m.book.bid);
  mem.postExitLow=Math.min(num(mem.postExitLow,m.book.bid),m.book.bid);
  const low=num(mem.postExitLow,m.book.bid),pullback=reference-low,pullbackNeed=Math.max(reference*REBASE_PULLBACK_BPS/10000,atr*REBASE_PULLBACK_ATR);
  const reclaim=m.book.mid-low,reclaimNeed=atr*REBASE_RECLAIM_ATR;
  if(pullback>=pullbackNeed&&reclaim>=reclaimNeed)mem.rebaseReady=true;
  return{required:true,ready:mem.rebaseReady===true,pullbackBps:reference>0?pullback/reference*10000:0,reclaimAtr:atr>0?reclaim/atr:0,reference,low};
}

export async function authorityCandidate(m:Market,sessionId:string,rt:Runtime,cfg:J,tradeNotional:number,at:number,cal:ExecutionCalibration):Promise<CandidateResult>{
  const c=await legacyAuthorityCandidate(m,sessionId,rt,cfg,tradeNotional,at,cal);
  const thesis=c.thesis as J,scores=scoreState(thesis),hasLong=rt.pos?.side==="LONG";
  const reasons=arr(thesis.authority_reason).map(String),soft=arr(thesis.soft_evidence).map(String),veto=arr(thesis.veto).map(String);
  const scoreGap=scores.up-scores.down,confidence=num(thesis.authority_confidence,.5),regime=String(thesis.regime||"RANGE");

  // Forecast first: choose a near-term reachable destination before asking whether a trade is economic.
  // A distant historical pivot may be shown as stretch telemetry, but it cannot rescue an uneconomic primary forecast.
  let rankCap=RANGE_PRIMARY_RANK_CAP;
  const strongContinuation=confidence>=.72&&scoreGap>=1.5&&scores.momentumAtr>=0&&scores.momentumAtr<=1.15&&scores.rangePos<.52;
  if(strongContinuation)rankCap=STRONG_PRIMARY_RANK_CAP;
  if(scores.momentumAtr>1.35||scores.rangePos>.56)rankCap=Math.min(rankCap,3);
  if(scores.rangePos>.70)rankCap=Math.min(rankCap,2);
  const levels=normalizedLevels(thesis,c.entry),primaryLevels=levels.filter(x=>x.rank<=rankCap),primary=primaryLevels.at(-1)??null,stretch=levels.find(x=>!primary||x.rank>primary.rank)??null;

  const fee=num(cfg.fee_bps,10),slip=num(cfg.slippage_bps,1);
  const cost=buildCostContract({feeOpenBps:fee,feeCloseBps:fee,openingSlippageBps:slip,expectedExitSpreadBps:m.book.spreadBps,expectedExitSlippageBps:slip,expectedFundingBps:m.fundingBpsHold});
  const primaryEconomics=!hasLong&&primary&&c.inv&&c.entry>0&&c.inv<c.entry?executionEconomics("UP",c.entry,primary.price,c.inv,cost):null;
  const forecastNetEdge=primaryEconomics?.target_net_reward_bps??0;
  const forecastEconomic=!!primaryEconomics&&forecastNetEdge>=MIN_FORECAST_NET_EDGE_BPS&&primaryEconomics.economic_rr>=1;
  const chase=(scores.rangePos>=CHASE_RANGE_POSITION&&scores.momentumAtr>=CHASE_MOMENTUM_ATR)||(scores.rangePos>=.68&&scores.momentumAtr>.45)||scores.momentumAtr>1.75;
  const rebase=rebaseState(rt as Runtime844,m,thesis);

  let action=String(thesis.authority_action||"WAIT"),sellStrength="NONE",sellReason="NONE";
  let primaryTarget=primary?.price??null;

  if(hasLong&&rt.pos){
    // Never move the stored trade target farther away to justify staying in the trade.
    primaryTarget=rt.pos.target;
    const heldMs=Math.max(0,at-Date.parse(rt.pos.opened_at));
    const rawBearish=scores.down>=1&&scores.down-scores.up>=.55;
    const severeBearish=rawBearish&&(scores.down-scores.up)>=1.8&&scores.momentumAtr<=-.35&&structureDamage(thesis);
    const bearishConfirm=rawBearish&&scores.momentumAtr<=-.08;
    const breakEvenBps=fillForwardCostBps(cost),peakNetBps=scores.peakProfitBps-breakEvenBps,currentNetBps=scores.profitBps-breakEvenBps;
    const harvestArmed=peakNetBps>=HARVEST_BREAK_EVEN_BUFFER_BPS;
    const targetTouched=m.book.bid>=rt.pos.target;
    const weakening=rawBearish||scores.momentumAtr<.15||m.flowFast.ofi<-.12;
    const harvestStrong=harvestArmed&&scores.givebackBps>=HARVEST_GIVEBACK_BPS&&weakening;
    const harvestSoft=harvestArmed&&currentNetBps>=MIN_FORECAST_NET_EDGE_BPS&&scores.resistanceTouched&&weakening;
    const softBearish=heldMs>=MIN_THESIS_HOLD_MS&&bearishConfirm;
    if(severeBearish){action="SELL";sellStrength="STRONG";sellReason="SEVERE_BEARISH_STRUCTURE_BREAK";}
    else if(harvestStrong){action="SELL";sellStrength="STRONG";sellReason="NET_PROFIT_HARVEST_GIVEBACK";}
    else if(harvestSoft||softBearish){action="SELL";sellStrength="SOFT";sellReason=harvestSoft?"NET_PROFIT_LOCAL_TOP":"BEARISH_CONFIRMATION";}
    else {action="HOLD";sellStrength="NONE";sellReason="NONE";}
    pushUnique(reasons,`cycle hold=${Math.round(heldMs/1000)}s net=${currentNetBps.toFixed(1)}bps peakNet=${peakNetBps.toFixed(1)}bps`);
    if(rawBearish&&scores.momentumAtr>-.08)pushUnique(soft,"RAW_BEARISH_IGNORED_WITHOUT_NEGATIVE_MOMENTUM");
    if(rawBearish&&heldMs<MIN_THESIS_HOLD_MS&&!severeBearish)pushUnique(soft,"THESIS_MIN_HOLD_ACTIVE");
    if(harvestArmed)pushUnique(soft,"NET_PROFIT_HARVEST_ARMED");
    const a=obj(thesis.authority_scores);
    Object.assign(a,{sell_vote:action==="SELL",sell_strength:sellStrength,sell_reason:sellReason,profit_protect_armed:harvestArmed,target_touched:targetTouched,break_even_bps:breakEvenBps,current_net_bps:currentNetBps,peak_net_bps:peakNetBps});
    thesis.authority_scores=a;
    thesis.cycle_phase=action==="SELL"?"HARVEST":"RIDE";
  }else{
    const directionalUp=c.direction==="UP"&&scoreGap>=.55;
    const entryReady=directionalUp&&scores.entryQuality>=MIN_ENTRY_QUALITY&&!chase&&forecastEconomic&&rebase.ready&&!!primary;
    action=entryReady?"BUY":"WAIT";
    if(chase)pushUnique(soft,"CYCLE_WAIT_DONT_CHASE_RALLY");
    if(!primary)pushUnique(soft,"NO_NEAR_TERM_FORECAST_DESTINATION");
    if(primary&&!forecastEconomic)pushUnique(soft,"PRIMARY_FORECAST_BELOW_COST_WAIT");
    if(rebase.required&&!rebase.ready)pushUnique(soft,"CYCLE_WAIT_FOR_PULLBACK_REBASE");
    if(scores.entryQuality<MIN_ENTRY_QUALITY)pushUnique(soft,"CYCLE_ENTRY_QUALITY_LOW");
    thesis.cycle_phase=rebase.required&&!rebase.ready?"REBASE":entryReady?"ENTER_RECOVERY":"SEEK_DIP";
  }

  const forecastProbability=Math.max(.05,Math.min(.95,confidence*(1-Math.max(0,scores.rangePos-.5)*.5)*(1-Math.max(0,scores.momentumAtr-1)*.15)));
  const targetPlan={...obj(thesis.target_plan),policy:"BRIAN_FORECAST_PRIMARY_WITH_STRETCH_TELEMETRY",target_planner_version:TARGET_PLANNER_VERSION,selected_rank:primary?.rank??null,selected_level:primary,forecast_primary_rank_cap:rankCap,forecast_primary:primary,stretch_target:stretch,far_pivots:"STRETCH_ONLY"};
  thesis.target_plan=targetPlan;
  thesis.forecast_destination_price=primary?.price??null;
  thesis.forecast_probability=forecastProbability;
  thesis.forecast_primary_rank=primary?.rank??null;
  thesis.stretch_target_price=stretch?.price??null;
  thesis.forecast_net_edge_bps=forecastNetEdge;
  thesis.forecast_economic=forecastEconomic;
  thesis.entry_timing_state=chase?"WAIT_CHASE":rebase.required&&!rebase.ready?"WAIT_REBASE":action==="BUY"?"ENTRY_READY":"WAIT";
  thesis.rebase_required=rebase.required;
  thesis.rebase_ready=rebase.ready;
  thesis.rebase_pullback_bps=rebase.pullbackBps;
  thesis.rebase_reclaim_atr=rebase.reclaimAtr;
  thesis.authority_action=action;
  thesis.authority_reason=reasons;
  thesis.soft_evidence=soft;
  thesis.veto=veto;
  thesis.target_price=primaryTarget;
  thesis.target_role=primaryTarget?"L1_EXECUTABLE":"NO_FORWARD_LEVEL";
  thesis.fill_forward_cost_bps=fillForwardCostBps(cost);
  thesis.reference_roundtrip_cost_bps=referenceRoundtripCostBps(cost,m.book.spreadBps);
  if(primaryEconomics){thesis.economic_rr=primaryEconomics.economic_rr;thesis.rr=primaryEconomics.economic_rr;thesis.net_reward_bps=primaryEconomics.target_net_reward_bps;thesis.net_risk_bps=primaryEconomics.stop_net_loss_bps;}

  let decision=c.decision as any;
  const canEnter=!hasLong&&action==="BUY"&&!!primary&&!!primaryEconomics&&!!c.size&&veto.length===0;
  if(canEnter&&decision){
    decision={...decision,target_price:primary!.price,forecast_probability:forecastProbability,l1_id:primary!.level_id??null,evidence:{...obj(decision.evidence),authority_action:action,authority_reason:reasons,authority_scores:thesis.authority_scores,target_plan:targetPlan,soft_evidence:soft,fill_forward_cost_bps:fillForwardCostBps(cost),reference_roundtrip_cost_bps:referenceRoundtripCostBps(cost,m.book.spreadBps),economic_rr:primaryEconomics!.economic_rr,forecast_destination_price:primary!.price,forecast_probability:forecastProbability,forecast_net_edge_bps:forecastNetEdge,cycle_phase:thesis.cycle_phase,rebase_required:rebase.required,rebase_ready:rebase.ready}};
  }else if(!hasLong){
    // WAIT observations stay in candidate telemetry; do not freeze an occurrence/target in the decision ledger.
    decision=null;
  }

  return{
    ...c,
    thesis,
    decision,
    target:primaryTarget,
    targetRole:primaryTarget?"L1_EXECUTABLE":"NO_FORWARD_LEVEL",
    canEnter,
    firstBlockingVeto:veto[0]??null,
    vetoStage:veto.length?"TECHNICAL_RAIL":null,
  };
}
