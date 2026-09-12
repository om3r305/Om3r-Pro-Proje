import type { ExecutionCalibration, J, Runtime, Struct } from "../_shared/dip_v84_contract.ts";
import {
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
  breakoutBlockedLevel?:number|null;breakoutCrossAt?:number|null;breakoutSeedQuality?:number|null;
};
type Runtime844=Runtime&{v842?:CycleMemory};
type TacticalState={localLow:number;localHigh:number;localRangePos:number;reclaimAtr:number;microBull:boolean;microBear:boolean;ready:boolean;chase:boolean;quality:number};
type LevelEval={level:Level;economics:ReturnType<typeof executionEconomics>;distanceBps:number};
type BreakoutState={level:number|null;confirmed:boolean;holdMs:number;overshootBps:number;seedQuality:number};

const CYCLE_HOTFIX_VERSION="v844-breakout-rollover-20260912.3";
const MAX_PRIMARY_FORECAST_DISTANCE_BPS=60;
const BREAKOUT_TARGET_DISTANCE_BPS=45;
const RECOVERY_MIN_ECONOMIC_RR=.70;
const REBASE_COST_FRACTION=.85;
const SOFT_HARVEST_MIN_NET_BPS=12;
const BREAKOUT_CONFIRM_MS=20_000;
const BREAKOUT_MAX_OVERSHOOT_BPS=10;
const BREAKOUT_MIN_SCORE_GAP=.35;
const BREAKOUT_MIN_SEED_QUALITY=.72;
const BREAKOUT_MAX_MOMENTUM_ATR=1.40;

function num(v:unknown,fallback=0){const x=Number(v);return Number.isFinite(x)?x:fallback;}
function obj(v:unknown):J{return v&&typeof v==="object"&&!Array.isArray(v)?v as J:{};}
function arr(v:unknown):unknown[]{return Array.isArray(v)?v:[];}
function pushUnique(a:string[],v:string){if(!a.includes(v))a.push(v);}
function clip(v:number,lo=0,hi=1){return Math.max(lo,Math.min(hi,v));}

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

function tacticalState(m:Market,thesis:J):TacticalState{
  const rows=m.bars["1m"].slice(-12),s1=obj(obj(thesis.structure).s1),atr=Math.max(num(s1.atr),m.rules.tickSize),live=m.book.mid;
  const localLow=rows.length?Math.min(...rows.map(x=>x.l)):live,localHigh=rows.length?Math.max(...rows.map(x=>x.h)):live;
  const span=Math.max(localHigh-localLow,atr*.75,m.rules.tickSize*8),localRangePos=clip((live-localLow)/span);
  const reclaimAtr=Math.max(0,(live-localLow)/atr),microBull=m.flowFast.ofi>=.10||m.book.pressure>=1.35,microBear=m.flowFast.ofi<=-.22&&m.book.pressure<.85;
  const nearDip=localRangePos<=.48,reclaimReady=reclaimAtr>=.035&&(microBull||m.flowFast.ofi>=0);
  const ready=nearDip&&reclaimReady&&!microBear;
  let quality=localRangePos<=.14?.76:localRangePos<=.30?.94:localRangePos<=.48?.84:localRangePos<=.62?.60:localRangePos<=.74?.38:.16;
  if(!reclaimReady&&localRangePos<=.48)quality*=.55;
  if(microBull)quality*=1.08;
  if(microBear)quality*=.55;
  const macroRange=num(obj(thesis.authority_scores).range_position,.5);
  if(macroRange>.92)quality*=.86;
  if(macroRange>.97)quality*=.78;
  const momentum=num(obj(thesis.authority_scores).momentum_atr);
  const chase=(localRangePos>.76&&momentum>.55)||(localRangePos>.66&&momentum>1.15)||momentum>2.0;
  if(chase)quality=Math.min(quality,.24);
  return{localLow,localHigh,localRangePos,reclaimAtr,microBull,microBear,ready,chase,quality:clip(quality)};
}

function tacticalInvalidation(m:Market,thesis:J,t:TacticalState,legacyInv:number|null,entry:number):number|null{
  if(!(entry>0))return legacyInv;
  const s1=obj(obj(thesis.structure).s1),atr=Math.max(num(s1.atr),m.rules.tickSize);
  if(t.ready&&t.localLow>0&&t.localLow<entry){
    const buffer=Math.max(atr*.18,entry*.00012,m.rules.tickSize*4),stop=t.localLow-buffer;
    if(stop>0&&stop<entry)return stop;
  }
  return legacyInv;
}

function cycleSize(rt:Runtime,tradeNotional:number,entry:number,feeBps:number,allocation:number,rules:Market["rules"],stopNetLossBps:number):CandidateResult["size"]{
  const cash=rt.cash,feeRate=feeBps/10000,maxCashNotional=cash/(1+feeRate),budget=Math.min(tradeNotional,maxCashNotional)*clip(allocation,.12,1);
  const rawQty=Math.min(budget/entry,rules.maxQty),qty=Math.floor((rawQty+Number.EPSILON)/rules.stepSize)*rules.stepSize,notional=qty*entry,margin=notional,fees_open=notional*feeRate;
  if(!Number.isFinite(qty)||qty<rules.minQty||notional<rules.minNotional||margin+fees_open>cash+1e-8)return null;
  const worst_loss=Math.max(0,notional*Math.max(0,stopNetLossBps)/10000);
  return {qty,notional,margin,leverage:1,actual_fraction:margin/cash,gross_fraction:notional/cash,fees_open,worst_loss,risk_fraction:cash?worst_loss/cash:0,allocation:clip(allocation,.12,1),risk_policy:"V844_TACTICAL_DIP_RECLAIM_1X",max_notional_fraction:1} as CandidateResult["size"];
}

function structureDamage(thesis:J):boolean{
  const S=obj(thesis.structure),s1=obj(S.s1),s5=obj(S.s5);
  return s1.bos==="DOWN"||s1.choch==="DOWN"||(s1.trend==="DOWN"&&s5.trend==="DOWN")||(s1.failedBreak==="BEAR"&&s5.trend!=="UP");
}

function cycleMemory(rt:Runtime844):CycleMemory{
  if(!rt.v842)rt.v842={sellVotes:0,lastExitAt:null,lastExitPrice:null,lastExitReason:null,lastTargetRaiseAt:null,lastPeakPrice:null,postExitLow:null,rebaseReady:false,breakoutBlockedLevel:null,breakoutCrossAt:null,breakoutSeedQuality:null};
  return rt.v842;
}

function rebaseState(rt:Runtime844,m:Market,thesis:J,minPullbackBps=REBASE_PULLBACK_BPS):{required:boolean;ready:boolean;pullbackBps:number;reclaimAtr:number;reference:number|null;low:number|null}{
  const mem=cycleMemory(rt),s1=obj(obj(thesis.structure).s1),atr=Math.max(num(s1.atr),m.rules.tickSize);
  if(!mem.lastExitAt||!mem.lastExitPrice)return{required:false,ready:true,pullbackBps:0,reclaimAtr:0,reference:null,low:null};
  const reference=Math.max(num(mem.lastExitPrice),num(mem.lastPeakPrice),m.book.bid);
  const priorLow=mem.postExitLow!==null&&mem.postExitLow!==undefined&&num(mem.postExitLow)>0?num(mem.postExitLow):m.book.bid;
  mem.postExitLow=Math.min(priorLow,m.book.bid);
  const low=num(mem.postExitLow,m.book.bid),pullback=reference-low,pullbackNeed=Math.max(reference*Math.max(REBASE_PULLBACK_BPS,minPullbackBps)/10000,atr*REBASE_PULLBACK_ATR);
  const reclaim=m.book.mid-low,reclaimNeed=atr*REBASE_RECLAIM_ATR;
  if(pullback>=pullbackNeed&&reclaim>=reclaimNeed)mem.rebaseReady=true;
  return{required:true,ready:mem.rebaseReady===true,pullbackBps:reference>0?pullback/reference*10000:0,reclaimAtr:atr>0?reclaim/atr:0,reference,low};
}

function breakoutState(rt:Runtime844,m:Market,thesis:J,nearest:LevelEval|null,nearestEconomic:boolean,tactical:TacticalState,scores:ReturnType<typeof scoreState>,scoreGap:number,at:number,roundtripCostBps:number):BreakoutState{
  const mem=cycleMemory(rt),live=m.book.mid,s1=obj(obj(thesis.structure).s1),atr=Math.max(num(s1.atr),m.rules.tickSize),buffer=Math.max(m.rules.tickSize*4,atr*.06,live*.00004);
  const armDistance=Math.max(6,Math.min(14,roundtripCostBps*.65));
  if(!mem.breakoutBlockedLevel&&nearest&&!nearestEconomic&&nearest.distanceBps<=armDistance&&tactical.ready&&tactical.quality>=BREAKOUT_MIN_SEED_QUALITY&&!tactical.microBear){
    mem.breakoutBlockedLevel=nearest.level.price;mem.breakoutCrossAt=null;mem.breakoutSeedQuality=tactical.quality;
  }else if(mem.breakoutBlockedLevel&&nearest&&Math.abs(nearest.level.price-mem.breakoutBlockedLevel)<=buffer*2){
    mem.breakoutSeedQuality=Math.max(num(mem.breakoutSeedQuality),tactical.quality);
  }
  const level=num(mem.breakoutBlockedLevel);
  if(!(level>0))return{level:null,confirmed:false,holdMs:0,overshootBps:0,seedQuality:0};
  const overshootBps=(live-level)/level*10000;
  if(live>=level+buffer){if(!mem.breakoutCrossAt)mem.breakoutCrossAt=at;}
  else if(live<level-buffer){mem.breakoutCrossAt=null;if((level-live)/level*10000>Math.max(12,roundtripCostBps*.7)){mem.breakoutBlockedLevel=null;mem.breakoutSeedQuality=null;}}
  const holdMs=mem.breakoutCrossAt?Math.max(0,at-mem.breakoutCrossAt):0;
  const flowOk=m.flowFast.ofi>=.05||m.book.pressure>=1.20;
  const momentumOk=scores.momentumAtr>=-.10&&scores.momentumAtr<=BREAKOUT_MAX_MOMENTUM_ATR;
  const confirmed=holdMs>=BREAKOUT_CONFIRM_MS&&overshootBps>=0&&overshootBps<=BREAKOUT_MAX_OVERSHOOT_BPS&&flowOk&&momentumOk&&scoreGap>=BREAKOUT_MIN_SCORE_GAP&&!tactical.microBear&&num(mem.breakoutSeedQuality)>=BREAKOUT_MIN_SEED_QUALITY;
  return{level:mem.breakoutBlockedLevel??null,confirmed,holdMs,overshootBps,seedQuality:num(mem.breakoutSeedQuality)};
}

export async function authorityCandidate(m:Market,sessionId:string,rt:Runtime,cfg:J,tradeNotional:number,at:number,cal:ExecutionCalibration):Promise<CandidateResult>{
  const c=await legacyAuthorityCandidate(m,sessionId,rt,cfg,tradeNotional,at,cal);
  const thesis=c.thesis as J,scores=scoreState(thesis),hasLong=rt.pos?.side==="LONG",tactical=tacticalState(m,thesis),mem=cycleMemory(rt as Runtime844);
  if(hasLong){mem.breakoutBlockedLevel=null;mem.breakoutCrossAt=null;mem.breakoutSeedQuality=null;}
  const reasons=arr(thesis.authority_reason).map(String),soft=arr(thesis.soft_evidence).map(String),veto=arr(thesis.veto).map(String);
  const scoreGap=scores.up-scores.down,confidence=num(thesis.authority_confidence,.5);
  const effectiveEntryQuality=hasLong?scores.entryQuality:tactical.quality;
  const effectiveInv=hasLong?c.inv:tacticalInvalidation(m,thesis,tactical,c.inv,c.entry);

  let rankCap=RANGE_PRIMARY_RANK_CAP;
  const strongContinuation=confidence>=.72&&scoreGap>=1.5&&scores.momentumAtr>=0&&scores.momentumAtr<=1.15&&tactical.localRangePos<.58;
  const recoveryForecast=tactical.ready&&effectiveEntryQuality>=.62&&!tactical.chase&&scores.momentumAtr>=-1.25&&scores.momentumAtr<=1.25&&tactical.localRangePos<=.52;
  if(strongContinuation||recoveryForecast)rankCap=STRONG_PRIMARY_RANK_CAP;
  if(scores.momentumAtr>1.35||tactical.localRangePos>.72)rankCap=Math.min(rankCap,3);
  if(tactical.localRangePos>.84||(scores.rangePos>.96&&tactical.localRangePos>.68))rankCap=Math.min(rankCap,2);

  const fee=num(cfg.fee_bps,10),slip=num(cfg.slippage_bps,1);
  const cost=buildCostContract({feeOpenBps:fee,feeCloseBps:fee,openingSlippageBps:slip,expectedExitSpreadBps:m.book.spreadBps,expectedExitSlippageBps:slip,expectedFundingBps:m.fundingBpsHold});
  const roundtripCostBps=referenceRoundtripCostBps(cost,m.book.spreadBps);
  const levels=normalizedLevels(thesis,c.entry);
  const eligible=levels.filter(x=>x.rank<=rankCap&&((x.price-c.entry)/Math.max(c.entry,1e-9)*10000)<=MAX_PRIMARY_FORECAST_DISTANCE_BPS);
  const evalLevels=(rows:Level[]):LevelEval[]=>!hasLong&&effectiveInv&&c.entry>0&&effectiveInv<c.entry?rows.map(level=>({level,economics:executionEconomics("UP",c.entry,level.price,effectiveInv,cost),distanceBps:(level.price-c.entry)/Math.max(c.entry,1e-9)*10000})):[];
  const evaluated=evalLevels(eligible);
  const recoveryEconomics=recoveryForecast&&effectiveEntryQuality>=.80&&confidence>=.65&&scoreGap>=.55;
  const minEconomicRR=recoveryEconomics?RECOVERY_MIN_ECONOMIC_RR:1;
  const nearestPick=evaluated[0]??null;
  const nearestEconomic=!!(nearestPick&&nearestPick.economics.target_net_reward_bps>=MIN_FORECAST_NET_EDGE_BPS&&nearestPick.economics.economic_rr>=minEconomicRR);
  const breakout=hasLong?{level:null,confirmed:false,holdMs:0,overshootBps:0,seedQuality:0}:breakoutState(rt as Runtime844,m,thesis,nearestPick,nearestEconomic,tactical,scores,scoreGap,at,roundtripCostBps);
  const breakoutLevels=levels.filter(x=>x.rank<=STRONG_PRIMARY_RANK_CAP&&((x.price-c.entry)/Math.max(c.entry,1e-9)*10000)<=BREAKOUT_TARGET_DISTANCE_BPS);
  const breakoutEvaluated=evalLevels(breakoutLevels);
  const breakoutPick=breakout.confirmed?breakoutEvaluated.find(x=>x.economics.target_net_reward_bps>=MIN_FORECAST_NET_EDGE_BPS&&x.economics.economic_rr>=RECOVERY_MIN_ECONOMIC_RR)??null:null;
  const breakoutContinuation=!!breakoutPick&&breakout.confirmed&&c.direction==="UP";
  const economicPick=nearestEconomic?nearestPick:breakoutPick;
  const selectedPick=economicPick??nearestPick;
  const primary=selectedPick?.level??null;
  const primaryEconomics=selectedPick?.economics??null;
  const stretch=levels.find(x=>!primary||x.rank>primary.rank)??null;
  const forecastNetEdge=primaryEconomics?.target_net_reward_bps??0;
  const forecastEconomic=!!economicPick;
  const rawChase=tactical.chase||(scores.rangePos>.96&&tactical.localRangePos>.68&&scores.momentumAtr>.25)||scores.momentumAtr>2.0;
  const chase=rawChase&&!breakoutContinuation;
  const decisionEntryQuality=breakoutContinuation?Math.max(effectiveEntryQuality,breakout.seedQuality):effectiveEntryQuality;
  const rebaseMinPullbackBps=Math.max(REBASE_PULLBACK_BPS,roundtripCostBps*REBASE_COST_FRACTION);
  const rebase=hasLong?{required:false,ready:true,pullbackBps:0,reclaimAtr:0,reference:null,low:null}:rebaseState(rt as Runtime844,m,thesis,rebaseMinPullbackBps);
  const normalizedConfidence=clip((confidence-.50)/.43),allocation=clip(.12+.88*normalizedConfidence*decisionEntryQuality,.12,1);
  const effectiveSize=!hasLong&&primaryEconomics&&effectiveInv?cycleSize(rt,tradeNotional,c.entry,fee,allocation,m.rules,primaryEconomics.stop_net_loss_bps):c.size;

  let action=String(thesis.authority_action||"WAIT"),sellStrength="NONE",sellReason="NONE";
  let primaryTarget=primary?.price??null;

  if(hasLong&&rt.pos){
    primaryTarget=rt.pos.target;
    const heldMs=Math.max(0,at-Date.parse(rt.pos.opened_at));
    const rawBearish=scores.down>=1&&scores.down-scores.up>=.55;
    const severeBearish=rawBearish&&(scores.down-scores.up)>=1.8&&scores.momentumAtr<=-.35&&structureDamage(thesis);
    const bearishConfirm=rawBearish&&scores.momentumAtr<=-.08;
    const breakEvenBps=fillForwardCostBps(cost),peakNetBps=scores.peakProfitBps-breakEvenBps,currentNetBps=scores.profitBps-breakEvenBps;
    const softHarvestFloorBps=Math.max(SOFT_HARVEST_MIN_NET_BPS,breakEvenBps*.55);
    const harvestArmed=peakNetBps>=HARVEST_BREAK_EVEN_BUFFER_BPS;
    const targetTouched=m.book.bid>=rt.pos.target;
    const weakening=rawBearish||scores.momentumAtr<.15||m.flowFast.ofi<-.12;
    const harvestStrong=harvestArmed&&scores.givebackBps>=HARVEST_GIVEBACK_BPS&&weakening;
    const harvestSoft=harvestArmed&&currentNetBps>=softHarvestFloorBps&&scores.resistanceTouched&&weakening;
    const softBearish=heldMs>=MIN_THESIS_HOLD_MS&&bearishConfirm;
    if(severeBearish){action="SELL";sellStrength="STRONG";sellReason="SEVERE_BEARISH_STRUCTURE_BREAK";}
    else if(harvestStrong){action="SELL";sellStrength="STRONG";sellReason="NET_PROFIT_HARVEST_GIVEBACK";}
    else if(harvestSoft||softBearish){action="SELL";sellStrength="SOFT";sellReason=harvestSoft?"NET_PROFIT_LOCAL_TOP":"BEARISH_CONFIRMATION";}
    else {action="HOLD";sellStrength="NONE";sellReason="NONE";}
    pushUnique(reasons,`cycle hold=${Math.round(heldMs/1000)}s net=${currentNetBps.toFixed(1)}bps peakNet=${peakNetBps.toFixed(1)}bps`);
    if(rawBearish&&scores.momentumAtr>-.08)pushUnique(soft,"RAW_BEARISH_IGNORED_WITHOUT_NEGATIVE_MOMENTUM");
    if(rawBearish&&heldMs<MIN_THESIS_HOLD_MS&&!severeBearish)pushUnique(soft,"THESIS_MIN_HOLD_ACTIVE");
    if(harvestArmed)pushUnique(soft,"NET_PROFIT_HARVEST_ARMED");
    if(harvestArmed&&currentNetBps<softHarvestFloorBps)pushUnique(soft,"SOFT_HARVEST_WAIT_FOR_MEANINGFUL_NET");
    const a=obj(thesis.authority_scores);
    Object.assign(a,{sell_vote:action==="SELL",sell_strength:sellStrength,sell_reason:sellReason,profit_protect_armed:harvestArmed,target_touched:targetTouched,break_even_bps:breakEvenBps,current_net_bps:currentNetBps,peak_net_bps:peakNetBps,soft_harvest_min_net_bps:softHarvestFloorBps});
    thesis.authority_scores=a;thesis.cycle_phase=action==="SELL"?"HARVEST":"RIDE";
  }else{
    const localDirectionAssist=c.direction==="UP"&&scoreGap<.55&&scoreGap>=-.75&&tactical.ready&&decisionEntryQuality>=.72&&!tactical.microBear;
    const directionalUp=(c.direction==="UP"&&(scoreGap>=.55||localDirectionAssist))||breakoutContinuation;
    const recoveryReady=tactical.ready||breakoutContinuation;
    const entryReady=directionalUp&&decisionEntryQuality>=MIN_ENTRY_QUALITY&&recoveryReady&&!chase&&forecastEconomic&&rebase.ready&&!!primary&&!!effectiveSize;
    action=entryReady?"BUY":"WAIT";
    if(localDirectionAssist)pushUnique(soft,"TACTICAL_DIRECTION_LAG_ASSIST_ACTIVE");
    if(breakoutContinuation)pushUnique(soft,"CONFIRMED_BREAKOUT_TARGET_ROLLOVER");
    if(breakout.level&&!breakout.confirmed&&breakout.holdMs>0)pushUnique(soft,"BREAKOUT_CONFIRMATION_PENDING");
    if(chase)pushUnique(soft,"CYCLE_WAIT_DONT_CHASE_RALLY");
    if(!recoveryReady)pushUnique(soft,"CYCLE_WAIT_FOR_TACTICAL_DIP_RECLAIM");
    if(!primary)pushUnique(soft,"NO_NEAR_TERM_FORECAST_DESTINATION");
    if(primary&&!forecastEconomic)pushUnique(soft,"PRIMARY_FORECAST_BELOW_COST_WAIT");
    if(levels.length>0&&eligible.length===0&&!breakoutContinuation)pushUnique(soft,"PRIMARY_FORECAST_TOO_FAR_WAIT");
    if(rebase.required&&!rebase.ready)pushUnique(soft,"CYCLE_WAIT_FOR_PULLBACK_REBASE");
    if(rebase.required&&rebase.pullbackBps<rebaseMinPullbackBps)pushUnique(soft,"CYCLE_REBASE_COST_GUARD");
    if(decisionEntryQuality<MIN_ENTRY_QUALITY)pushUnique(soft,"CYCLE_ENTRY_QUALITY_LOW");
    if(tactical.ready)pushUnique(soft,"TACTICAL_DIP_RECLAIM_READY");
    thesis.cycle_phase=rebase.required&&!rebase.ready?"REBASE":entryReady?"ENTER_RECOVERY":"SEEK_DIP";
    if(entryReady){mem.breakoutBlockedLevel=null;mem.breakoutCrossAt=null;mem.breakoutSeedQuality=null;}
  }

  const macroPenalty=1-Math.max(0,scores.rangePos-.85)*.30;
  const recoveryBoost=!hasLong&&(tactical.ready||breakoutContinuation) ? .06 : 0;
  const forecastProbability=clip(confidence*macroPenalty*(1-Math.max(0,scores.momentumAtr-1)*.15)+recoveryBoost,.05,.95);
  const targetPlan={...obj(thesis.target_plan),policy:"BRIAN_FORECAST_PRIMARY_WITH_STRETCH_TELEMETRY",target_planner_version:TARGET_PLANNER_VERSION,selected_rank:primary?.rank??null,selected_level:primary,forecast_primary_rank_cap:rankCap,forecast_primary:primary,stretch_target:stretch,far_pivots:"STRETCH_ONLY",selection_mode:breakoutContinuation?"CONFIRMED_BREAKOUT_ROLLOVER":"NEAREST_STRUCTURAL_MUST_CLEAR_COST",max_primary_distance_bps:MAX_PRIMARY_FORECAST_DISTANCE_BPS,breakout_target_distance_bps:BREAKOUT_TARGET_DISTANCE_BPS,min_economic_rr:minEconomicRR};
  const a=obj(thesis.authority_scores);
  Object.assign(a,{range_position_macro:scores.rangePos,tactical_range_position:tactical.localRangePos,tactical_low:tactical.localLow,tactical_high:tactical.localHigh,tactical_reclaim_atr:tactical.reclaimAtr,tactical_recovery:tactical.ready,tactical_micro_bull:tactical.microBull,tactical_micro_bear:tactical.microBear,tactical_chase:tactical.chase,entry_quality_tactical:decisionEntryQuality,forecast_max_distance_bps:MAX_PRIMARY_FORECAST_DISTANCE_BPS,rebase_min_pullback_bps:rebaseMinPullbackBps,breakout_blocked_level:breakout.level,breakout_confirmed:breakout.confirmed,breakout_hold_ms:breakout.holdMs,breakout_overshoot_bps:breakout.overshootBps,breakout_seed_quality:breakout.seedQuality});
  thesis.authority_scores=a;
  thesis.target_plan=targetPlan;
  thesis.cycle_hotfix_version=CYCLE_HOTFIX_VERSION;
  thesis.forecast_destination_price=primary?.price??null;
  thesis.forecast_probability=forecastProbability;
  thesis.forecast_primary_rank=primary?.rank??null;
  thesis.stretch_target_price=stretch?.price??null;
  thesis.forecast_net_edge_bps=forecastNetEdge;
  thesis.forecast_economic=forecastEconomic;
  thesis.entry_timing_state=chase?"WAIT_CHASE":rebase.required&&!rebase.ready?"WAIT_REBASE":!hasLong&&!tactical.ready&&!breakoutContinuation?"WAIT_DIP_RECLAIM":action==="BUY"?"ENTRY_READY":"WAIT";
  thesis.rebase_required=rebase.required;thesis.rebase_ready=rebase.ready;thesis.rebase_pullback_bps=rebase.pullbackBps;thesis.rebase_reclaim_atr=rebase.reclaimAtr;thesis.rebase_min_pullback_bps=rebaseMinPullbackBps;
  thesis.authority_entry_quality=decisionEntryQuality;thesis.authority_allocation=effectiveSize?.allocation??allocation;thesis.authority_action=action;thesis.authority_reason=reasons;thesis.soft_evidence=soft;thesis.veto=veto;
  thesis.invalidation_price=effectiveInv;thesis.stop={...obj(thesis.stop),price:effectiveInv,source:!hasLong&&(tactical.ready||breakoutContinuation)?"TACTICAL_DIP_RECLAIM":obj(thesis.stop).source};
  thesis.target_price=primaryTarget;thesis.target_role=primaryTarget?"L1_EXECUTABLE":"NO_FORWARD_LEVEL";thesis.fill_forward_cost_bps=fillForwardCostBps(cost);thesis.reference_roundtrip_cost_bps=roundtripCostBps;
  if(primaryEconomics){thesis.economic_rr=primaryEconomics.economic_rr;thesis.rr=primaryEconomics.economic_rr;thesis.net_reward_bps=primaryEconomics.target_net_reward_bps;thesis.net_risk_bps=primaryEconomics.stop_net_loss_bps;}

  let decision=c.decision as any;
  const canEnter=!hasLong&&action==="BUY"&&!!primary&&!!primaryEconomics&&!!effectiveSize&&!!effectiveInv&&veto.length===0;
  if(canEnter&&decision){
    decision={...decision,target_price:primary!.price,invalidation_price:effectiveInv,forecast_probability:forecastProbability,l1_id:primary!.level_id??null,evidence:{...obj(decision.evidence),authority_action:action,authority_reason:reasons,authority_scores:thesis.authority_scores,authority_entry_quality:decisionEntryQuality,authority_allocation:effectiveSize?.allocation??allocation,target_plan:targetPlan,soft_evidence:soft,fill_forward_cost_bps:fillForwardCostBps(cost),reference_roundtrip_cost_bps:roundtripCostBps,economic_rr:primaryEconomics!.economic_rr,forecast_destination_price:primary!.price,forecast_probability:forecastProbability,forecast_net_edge_bps:forecastNetEdge,cycle_phase:thesis.cycle_phase,cycle_hotfix_version:CYCLE_HOTFIX_VERSION,rebase_required:rebase.required,rebase_ready:rebase.ready,rebase_min_pullback_bps:rebaseMinPullbackBps,tactical_dip_reclaim:tactical.ready,breakout_continuation:breakoutContinuation,breakout_blocked_level:breakout.level,breakout_hold_ms:breakout.holdMs,tactical_low:tactical.localLow,tactical_range_position:tactical.localRangePos,tactical_reclaim_atr:tactical.reclaimAtr}};
  }else if(!hasLong){decision=null;}

  return{...c,thesis,decision,inv:effectiveInv,size:effectiveSize,target:primaryTarget,targetRole:primaryTarget?"L1_EXECUTABLE":"NO_FORWARD_LEVEL",canEnter,firstBlockingVeto:veto[0]??null,vetoStage:veto.length?"TECHNICAL_RAIL":null};
}
