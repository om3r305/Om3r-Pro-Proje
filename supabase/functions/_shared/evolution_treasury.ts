export const BRIAN_TREASURY_VERSION="brian.treasury-shadow.v1";
export const BRIAN_TREASURY_STARTING_EQUITY_USD=10_000;

export interface TreasuryOpportunity{
  assetId:string;direction:-1|1;observedAt:string;referencePrice:number;expectedNetEdgeBps:number;
  roundTripCostBps:number;reliabilityConfidence:number;matureGroupCount:number;pitClear:boolean;
  recommendation:"ALLOW_EDGE"|string;sourceDecisionId:string;
}
export interface TreasuryPosition{
  positionId:string;assetId:string;direction:-1|1;openedAt:string;entryPrice:number;capitalUsd:number;
  entryExpectedNetEdgeBps:number;latestExpectedNetEdgeBps:number;roundTripCostBps:number;sourceDecisionId:string;
  highWaterPnlBps:number;
}
export interface TreasuryState{
  observedAt:string;startingEquityUsd:number;cashUsd:number;realizedPnlUsd:number;cumulativeCostsUsd:number;
  positions:TreasuryPosition[];
}
export type TreasuryExitReason="EDGE_INVALIDATED"|"DIRECTION_FLIP"|"STALE_EDGE"|"RISK_STOP"|"PROFIT_EDGE_DECAY"|"TIME_DECAY"|"OPPORTUNITY_REPLACEMENT";
export interface TreasuryAction{
  kind:"OPEN"|"EXIT";assetId:string;direction:-1|1;capitalUsd:number;referencePrice:number;costUsd:number;
  expectedNetEdgeBps:number;sourceDecisionId:string;reason:string;positionId:string|null;
}
export interface TreasuryCyclePlan{
  version:typeof BRIAN_TREASURY_VERSION;observedAt:string;beforeEquityUsd:number;afterEquityUsd:number;
  state:TreasuryState;actions:TreasuryAction[];deploymentUsd:number;deploymentPct:number;cashReservePct:number;
  blockedReasons:string[];
}

const MAX_DEPLOYMENT_PCT=.70;
const MIN_CASH_RESERVE_PCT=.30;
const MAX_POSITION_PCT=.12;
const MIN_POSITION_PCT=.025;
const MAX_POSITIONS=8;
const MIN_EDGE_BPS=3;
const REPLACEMENT_EDGE_ADVANTAGE_BPS=5;
const EDGE_STALE_SECONDS=20*60;
const RISK_STOP_BPS=-150;
const PROFIT_DECAY_ARM_BPS=50;
const MAX_HOLD_SECONDS=6*60*60;

function clamp(v:number,lo:number,hi:number){return Math.max(lo,Math.min(hi,v));}
function time(v:string){const t=Date.parse(v);return Number.isFinite(t)?t:null;}
function assertFinitePositive(v:number,label:string){if(!Number.isFinite(v)||v<=0)throw new Error(`${label} must be positive`);}
function halfCostUsd(capitalUsd:number,roundTripCostBps:number){return capitalUsd*Math.max(0,roundTripCostBps)/20_000;}

export function directionalReturnBps(position:TreasuryPosition,markPrice:number):number{
  assertFinitePositive(position.entryPrice,"entryPrice");assertFinitePositive(markPrice,"markPrice");
  return position.direction*(markPrice/position.entryPrice-1)*10_000;
}
export function positionGrossPnlUsd(position:TreasuryPosition,markPrice:number):number{return position.capitalUsd*directionalReturnBps(position,markPrice)/10_000;}
export function treasuryEquityUsd(state:TreasuryState,marks:Record<string,number>):number{
  let equity=state.cashUsd;
  for(const p of state.positions){const mark=marks[p.assetId];const gross=Number.isFinite(mark)&&mark>0?positionGrossPnlUsd(p,mark):0;equity+=p.capitalUsd+gross;}
  return equity;
}
function deploymentUsd(state:TreasuryState){return state.positions.reduce((s,p)=>s+p.capitalUsd,0);}
function opportunityQuality(o:TreasuryOpportunity){
  const reliability=clamp(o.reliabilityConfidence,.35,.65);const maturity=clamp(o.matureGroupCount/4,0,1);const edge=clamp(o.expectedNetEdgeBps/30,0,1);
  return edge*(.5+reliability)*(.5+.5*maturity);
}
export function opportunityScore(o:TreasuryOpportunity):number{return Math.max(0,o.expectedNetEdgeBps)*opportunityQuality(o);}
function validOpportunity(o:TreasuryOpportunity,nowMs:number){
  const at=time(o.observedAt);return Boolean(o.assetId)&&o.pitClear&&o.recommendation==="ALLOW_EDGE"&&o.direction!==0&&Number.isFinite(o.referencePrice)&&o.referencePrice>0&&Number.isFinite(o.expectedNetEdgeBps)&&o.expectedNetEdgeBps>=MIN_EDGE_BPS&&Number.isFinite(o.roundTripCostBps)&&o.roundTripCostBps>=0&&at!=null&&at<=nowMs+5_000&&(nowMs-at)/1000<=EDGE_STALE_SECONDS;
}
function targetCapitalUsd(o:TreasuryOpportunity,equity:number){
  const q=opportunityQuality(o);const pct=MIN_POSITION_PCT+(MAX_POSITION_PCT-MIN_POSITION_PCT)*q;return Math.max(0,equity*clamp(pct,MIN_POSITION_PCT,MAX_POSITION_PCT));
}
function latestByAsset(opportunities:TreasuryOpportunity[],nowMs:number){
  const map=new Map<string,TreasuryOpportunity>();for(const o of opportunities){if(!validOpportunity(o,nowMs))continue;const prev=map.get(o.assetId);if(!prev||Number(time(o.observedAt))>Number(time(prev.observedAt)))map.set(o.assetId,o);}return map;
}
function markMap(opportunities:TreasuryOpportunity[],state:TreasuryState){const marks:Record<string,number>={};for(const o of opportunities)if(Number.isFinite(o.referencePrice)&&o.referencePrice>0){const prev=marks[o.assetId];if(prev==null||Number(time(o.observedAt))>=0)marks[o.assetId]=o.referencePrice;}for(const p of state.positions)if(!(p.assetId in marks))marks[p.assetId]=p.entryPrice;return marks;}
function cloneState(state:TreasuryState):TreasuryState{return{...state,positions:state.positions.map(p=>({...p}))};}
function latestEdgeForPosition(p:TreasuryPosition,latest:Map<string,TreasuryOpportunity>){const o=latest.get(p.assetId);return o&&o.direction===p.direction?o.expectedNetEdgeBps:p.latestExpectedNetEdgeBps;}

function applyExit(state:TreasuryState,p:TreasuryPosition,markPrice:number,edgeBps:number,sourceDecisionId:string,reason:TreasuryExitReason,actions:TreasuryAction[]){
  const gross=positionGrossPnlUsd(p,markPrice),exitCost=halfCostUsd(p.capitalUsd,p.roundTripCostBps);state.cashUsd+=p.capitalUsd+gross-exitCost;state.realizedPnlUsd+=gross-exitCost;state.cumulativeCostsUsd+=exitCost;state.positions=state.positions.filter(x=>x.positionId!==p.positionId);
  actions.push({kind:"EXIT",assetId:p.assetId,direction:p.direction,capitalUsd:p.capitalUsd,referencePrice:markPrice,costUsd:exitCost,expectedNetEdgeBps:edgeBps,sourceDecisionId,reason,positionId:p.positionId});
}
function applyOpen(state:TreasuryState,o:TreasuryOpportunity,capitalUsd:number,positionId:string,actions:TreasuryAction[]){
  const entryCost=halfCostUsd(capitalUsd,o.roundTripCostBps);if(state.cashUsd+1e-9<capitalUsd+entryCost)return false;state.cashUsd-=capitalUsd+entryCost;state.realizedPnlUsd-=entryCost;state.cumulativeCostsUsd+=entryCost;
  state.positions.push({positionId,assetId:o.assetId,direction:o.direction,openedAt:o.observedAt,entryPrice:o.referencePrice,capitalUsd,entryExpectedNetEdgeBps:o.expectedNetEdgeBps,latestExpectedNetEdgeBps:o.expectedNetEdgeBps,roundTripCostBps:o.roundTripCostBps,sourceDecisionId:o.sourceDecisionId,highWaterPnlBps:0});
  actions.push({kind:"OPEN",assetId:o.assetId,direction:o.direction,capitalUsd,referencePrice:o.referencePrice,costUsd:entryCost,expectedNetEdgeBps:o.expectedNetEdgeBps,sourceDecisionId:o.sourceDecisionId,reason:"POSITIVE_VALIDATED_EDGE",positionId});return true;
}

export function planTreasuryCycle(input:{state:TreasuryState;opportunities:TreasuryOpportunity[];observedAt:string;positionIdFor:(o:TreasuryOpportunity)=>string;}):TreasuryCyclePlan{
  const nowMs=time(input.observedAt);if(nowMs==null)throw new Error("invalid cycle observedAt");const state=cloneState(input.state);const actions:TreasuryAction[]=[];const blockedReasons:string[]=[];const marks=markMap(input.opportunities,state);const beforeEquity=treasuryEquityUsd(state,marks);assertFinitePositive(beforeEquity,"treasury equity");const latest=latestByAsset(input.opportunities,nowMs);

  for(const p of [...state.positions]){
    const latestAny=input.opportunities.filter(o=>o.assetId===p.assetId&&time(o.observedAt)!=null&&Number(time(o.observedAt))<=nowMs+5_000).sort((a,b)=>Number(time(b.observedAt))-Number(time(a.observedAt)))[0];
    const mark=latestAny?.referencePrice??marks[p.assetId]??p.entryPrice;const pnlBps=directionalReturnBps(p,mark);p.highWaterPnlBps=Math.max(p.highWaterPnlBps,pnlBps);const ageSeconds=Math.max(0,(nowMs-Number(time(p.openedAt)??nowMs))/1000);const fresh=latest.get(p.assetId);const freshSame=fresh&&fresh.direction===p.direction?fresh:null;const currentEdge=freshSame?.expectedNetEdgeBps??p.latestExpectedNetEdgeBps;p.latestExpectedNetEdgeBps=currentEdge;
    let reason:TreasuryExitReason|null=null;
    if(latestAny&&validOpportunity(latestAny,nowMs)&&latestAny.direction!==p.direction)reason="DIRECTION_FLIP";
    else if(pnlBps<=RISK_STOP_BPS)reason="RISK_STOP";
    else if(freshSame&&freshSame.expectedNetEdgeBps<=0)reason="EDGE_INVALIDATED";
    else if(freshSame&&pnlBps>=PROFIT_DECAY_ARM_BPS&&freshSame.expectedNetEdgeBps<Math.max(MIN_EDGE_BPS,p.entryExpectedNetEdgeBps*.25))reason="PROFIT_EDGE_DECAY";
    else if(!freshSame&&ageSeconds>EDGE_STALE_SECONDS)reason="STALE_EDGE";
    else if(ageSeconds>MAX_HOLD_SECONDS&&pnlBps<=0)reason="TIME_DECAY";
    if(reason)applyExit(state,p,mark,currentEdge,freshSame?.sourceDecisionId??p.sourceDecisionId,reason,actions);
  }

  const candidates=[...latest.values()].sort((a,b)=>opportunityScore(b)-opportunityScore(a));
  for(const o of candidates){
    if(state.positions.some(p=>p.assetId===o.assetId))continue;
    const equity=treasuryEquityUsd(state,{...marks,[o.assetId]:o.referencePrice});const maxDeployment=equity*MAX_DEPLOYMENT_PCT;let availableDeployment=Math.max(0,maxDeployment-deploymentUsd(state));const minReserve=equity*MIN_CASH_RESERVE_PCT;let spendableCash=Math.max(0,state.cashUsd-minReserve);let desired=Math.min(targetCapitalUsd(o,equity),availableDeployment,spendableCash);
    const minimumTicket=Math.min(equity*MIN_POSITION_PCT,100);
    if((state.positions.length>=MAX_POSITIONS||desired<minimumTicket)&&state.positions.length){
      const weakest=[...state.positions].sort((a,b)=>latestEdgeForPosition(a,latest)-latestEdgeForPosition(b,latest))[0];const weakEdge=latestEdgeForPosition(weakest,latest);const weakScore=Math.max(0,weakEdge);if(o.expectedNetEdgeBps>=weakEdge+REPLACEMENT_EDGE_ADVANTAGE_BPS&&opportunityScore(o)>weakScore*1.25){const mark=marks[weakest.assetId]??weakest.entryPrice;applyExit(state,weakest,mark,weakEdge,o.sourceDecisionId,"OPPORTUNITY_REPLACEMENT",actions);const equity2=treasuryEquityUsd(state,{...marks,[o.assetId]:o.referencePrice});availableDeployment=Math.max(0,equity2*MAX_DEPLOYMENT_PCT-deploymentUsd(state));spendableCash=Math.max(0,state.cashUsd-equity2*MIN_CASH_RESERVE_PCT);desired=Math.min(targetCapitalUsd(o,equity2),availableDeployment,spendableCash);}
    }
    if(state.positions.length>=MAX_POSITIONS)continue;if(desired<minimumTicket)continue;applyOpen(state,o,desired,input.positionIdFor(o),actions);
  }

  const afterMarks={...marks};for(const o of candidates)afterMarks[o.assetId]=o.referencePrice;state.observedAt=input.observedAt;const afterEquity=treasuryEquityUsd(state,afterMarks);const deployed=deploymentUsd(state);const deploymentPct=afterEquity>0?deployed/afterEquity:0;const cashReservePct=afterEquity>0?state.cashUsd/afterEquity:0;
  if(deploymentPct>MAX_DEPLOYMENT_PCT+1e-6)blockedReasons.push("deployment cap exceeded");if(cashReservePct<MIN_CASH_RESERVE_PCT-0.02)blockedReasons.push("cash reserve below policy after costs");
  return{version:BRIAN_TREASURY_VERSION,observedAt:input.observedAt,beforeEquityUsd:beforeEquity,afterEquityUsd:afterEquity,state,actions,deploymentUsd:deployed,deploymentPct,cashReservePct,blockedReasons};
}

export function initialTreasuryState(observedAt:string,startingEquityUsd=BRIAN_TREASURY_STARTING_EQUITY_USD):TreasuryState{
  assertFinitePositive(startingEquityUsd,"startingEquityUsd");return{observedAt,startingEquityUsd,cashUsd:startingEquityUsd,realizedPnlUsd:0,cumulativeCostsUsd:0,positions:[]};
}
