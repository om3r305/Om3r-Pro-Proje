export const BRIAN_TREASURY_VERSION="brian.treasury-shadow.v3";
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

// SHADOW Treasury has no arbitrary 70/30 portfolio split. Brian may stay completely
// in cash or deploy effectively the whole cashbox when validated net edge, bounded
// reliability and independent evidence maturity jointly justify exceptional conviction.
// Point-in-time entry cost is still reserved, so "100%" never means negative cash.
const MAX_DEPLOYMENT_PCT=1;
const MIN_CASH_RESERVE_PCT=0;
const MAX_POSITION_PCT=1;
const MIN_POSITION_PCT=.005;
const MAX_POSITIONS=8;
const MIN_EDGE_BPS=3;
const FULL_CONVICTION_EDGE_BPS=60;
const REPLACEMENT_EDGE_ADVANTAGE_BPS=5;
const REPLACEMENT_SCORE_MULTIPLIER=1.25;
const EDGE_STALE_SECONDS=20*60;
const RISK_STOP_BPS=-150;
const PROFIT_DECAY_ARM_BPS=50;
const MAX_HOLD_SECONDS=6*60*60;

function clamp(v:number,lo:number,hi:number){return Math.max(lo,Math.min(hi,v));}
function time(v:string){const t=Date.parse(v);return Number.isFinite(t)?t:null;}
function assertFinitePositive(v:number,label:string){if(!Number.isFinite(v)||v<=0)throw new Error(`${label} must be positive`);}
function halfCostRate(roundTripCostBps:number){return Math.max(0,roundTripCostBps)/20_000;}
function halfCostUsd(capitalUsd:number,roundTripCostBps:number){return capitalUsd*halfCostRate(roundTripCostBps);}
function halfCostBps(roundTripCostBps:number){return Math.max(0,roundTripCostBps)/2;}

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

/**
 * Brian sizes by conviction rather than a fixed $3/$5 ticket or a fixed portfolio slice.
 * All dimensions are measured evidence: expected net edge after costs/uncertainty,
 * bounded prospective reliability, and independent-group maturity. A weak dimension
 * keeps size modest even if one headline metric is large.
 */
export function opportunityQuality(o:TreasuryOpportunity):number{
  const edge=clamp((o.expectedNetEdgeBps-MIN_EDGE_BPS)/(FULL_CONVICTION_EDGE_BPS-MIN_EDGE_BPS),0,1);
  const reliability=clamp((o.reliabilityConfidence-.5)/.15,0,1);
  const maturity=clamp((o.matureGroupCount-2)/4,0,1);
  return clamp(.60*edge+.25*reliability+.15*maturity,0,1);
}
export function opportunityScore(o:TreasuryOpportunity):number{return Math.max(0,o.expectedNetEdgeBps)*opportunityQuality(o);}
function rawOpportunityIsUsable(o:TreasuryOpportunity,nowMs:number){
  const at=time(o.observedAt);return Boolean(o.assetId)&&Number.isFinite(o.referencePrice)&&o.referencePrice>0&&Number.isFinite(o.expectedNetEdgeBps)&&Number.isFinite(o.roundTripCostBps)&&o.roundTripCostBps>=0&&at!=null&&at<=nowMs+5_000;
}
function validOpportunity(o:TreasuryOpportunity,nowMs:number){
  const at=time(o.observedAt);return rawOpportunityIsUsable(o,nowMs)&&o.pitClear&&o.recommendation==="ALLOW_EDGE"&&o.expectedNetEdgeBps>=MIN_EDGE_BPS&&at!=null&&(nowMs-at)/1000<=EDGE_STALE_SECONDS;
}
function targetCapitalUsd(o:TreasuryOpportunity,equity:number){
  const q=opportunityQuality(o);
  // Non-linear sizing makes ordinary edges use only part of the cashbox while allowing
  // truly exceptional evidence to earn essentially all available SHADOW capital.
  const pct=clamp(q*q,MIN_POSITION_PCT,MAX_POSITION_PCT);
  return Math.max(0,equity*pct);
}
function latestRawByAsset(opportunities:TreasuryOpportunity[],nowMs:number){
  const map=new Map<string,TreasuryOpportunity>();
  for(const o of opportunities){
    if(!rawOpportunityIsUsable(o,nowMs))continue;
    const prev=map.get(o.assetId);const at=time(o.observedAt)??0,prevAt=prev?time(prev.observedAt)??0:-1;
    if(!prev||at>prevAt)map.set(o.assetId,o);
  }
  return map;
}
function latestByAsset(opportunities:TreasuryOpportunity[],nowMs:number){
  const map=new Map<string,TreasuryOpportunity>();for(const o of opportunities){if(!validOpportunity(o,nowMs))continue;const prev=map.get(o.assetId);if(!prev||Number(time(o.observedAt))>Number(time(prev.observedAt)))map.set(o.assetId,o);}return map;
}
function markMap(opportunities:TreasuryOpportunity[],state:TreasuryState,nowMs:number){
  const marks:Record<string,number>={};const latestTs:Record<string,number>={};
  for(const o of opportunities){
    if(!rawOpportunityIsUsable(o,nowMs))continue;const at=time(o.observedAt)??-1;
    if(latestTs[o.assetId]==null||at>latestTs[o.assetId]){latestTs[o.assetId]=at;marks[o.assetId]=o.referencePrice;}
  }
  for(const p of state.positions)if(!(p.assetId in marks))marks[p.assetId]=p.entryPrice;
  return marks;
}
function cloneState(state:TreasuryState):TreasuryState{return{...state,positions:state.positions.map(p=>({...p}))};}
function latestEdgeForPosition(p:TreasuryPosition,latest:Map<string,TreasuryOpportunity>){const o=latest.get(p.assetId);return o&&o.direction===p.direction?o.expectedNetEdgeBps:p.latestExpectedNetEdgeBps;}
function remainingHoldEdgeBps(p:TreasuryPosition,latest:Map<string,TreasuryOpportunity>){
  // Expected-net edge is quoted as a full round trip. The entry half-cost is already sunk
  // for an open position, so add it back when comparing HOLD versus SWITCH from now.
  return latestEdgeForPosition(p,latest)+halfCostBps(p.roundTripCostBps);
}
function replacementAdvantageBps(candidate:TreasuryOpportunity,weakest:TreasuryPosition,latest:Map<string,TreasuryOpportunity>){
  // Switching must pay the old position's exit half-cost in addition to the candidate's
  // own full-round-trip cost already embedded in expectedNetEdgeBps.
  const switchEdgeAfterOldExit=candidate.expectedNetEdgeBps-halfCostBps(weakest.roundTripCostBps);
  return switchEdgeAfterOldExit-remainingHoldEdgeBps(weakest,latest);
}
function isFreshRaw(o:TreasuryOpportunity|undefined,nowMs:number){const at=o?time(o.observedAt):null;return at!=null&&nowMs>=at&&(nowMs-at)/1000<=EDGE_STALE_SECONDS;}
function maxCapitalPreservingReserve(cashUsd:number,equityUsd:number,roundTripCostBps:number){
  const c=halfCostRate(roundTripCostBps);const numerator=cashUsd-MIN_CASH_RESERVE_PCT*equityUsd;
  return Math.max(0,numerator/(1+c*(1-MIN_CASH_RESERVE_PCT)));
}
function maxCapitalPreservingDeployment(currentDeploymentUsd:number,equityUsd:number,roundTripCostBps:number){
  const c=halfCostRate(roundTripCostBps);const numerator=MAX_DEPLOYMENT_PCT*equityUsd-currentDeploymentUsd;
  return Math.max(0,numerator/(1+MAX_DEPLOYMENT_PCT*c));
}
function availableCapitalFor(o:TreasuryOpportunity,state:TreasuryState,equity:number){
  return Math.min(
    maxCapitalPreservingDeployment(deploymentUsd(state),equity,o.roundTripCostBps),
    maxCapitalPreservingReserve(state.cashUsd,equity,o.roundTripCostBps),
  );
}

function applyExit(state:TreasuryState,p:TreasuryPosition,markPrice:number,edgeBps:number,sourceDecisionId:string,reason:TreasuryExitReason,actions:TreasuryAction[]){
  const gross=positionGrossPnlUsd(p,markPrice),exitCost=halfCostUsd(p.capitalUsd,p.roundTripCostBps);state.cashUsd+=p.capitalUsd+gross-exitCost;state.realizedPnlUsd+=gross-exitCost;state.cumulativeCostsUsd+=exitCost;state.positions=state.positions.filter(x=>x.positionId!==p.positionId);
  actions.push({kind:"EXIT",assetId:p.assetId,direction:p.direction,capitalUsd:p.capitalUsd,referencePrice:markPrice,costUsd:exitCost,expectedNetEdgeBps:edgeBps,sourceDecisionId,reason,positionId:p.positionId});
}
function applyOpen(state:TreasuryState,o:TreasuryOpportunity,capitalUsd:number,positionId:string,openedAt:string,actions:TreasuryAction[]){
  const entryCost=halfCostUsd(capitalUsd,o.roundTripCostBps);if(state.cashUsd+1e-9<capitalUsd+entryCost)return false;state.cashUsd-=capitalUsd+entryCost;state.realizedPnlUsd-=entryCost;state.cumulativeCostsUsd+=entryCost;
  state.positions.push({positionId,assetId:o.assetId,direction:o.direction,openedAt,entryPrice:o.referencePrice,capitalUsd,entryExpectedNetEdgeBps:o.expectedNetEdgeBps,latestExpectedNetEdgeBps:o.expectedNetEdgeBps,roundTripCostBps:o.roundTripCostBps,sourceDecisionId:o.sourceDecisionId,highWaterPnlBps:0});
  actions.push({kind:"OPEN",assetId:o.assetId,direction:o.direction,capitalUsd,referencePrice:o.referencePrice,costUsd:entryCost,expectedNetEdgeBps:o.expectedNetEdgeBps,sourceDecisionId:o.sourceDecisionId,reason:"POSITIVE_VALIDATED_EDGE",positionId});return true;
}

export function planTreasuryCycle(input:{state:TreasuryState;opportunities:TreasuryOpportunity[];observedAt:string;positionIdFor:(o:TreasuryOpportunity)=>string;}):TreasuryCyclePlan{
  const nowMs=time(input.observedAt);if(nowMs==null)throw new Error("invalid cycle observedAt");const state=cloneState(input.state);const actions:TreasuryAction[]=[];const blockedReasons:string[]=[];const noReopenAssets=new Set<string>();const marks=markMap(input.opportunities,state,nowMs);const beforeEquity=treasuryEquityUsd(state,marks);assertFinitePositive(beforeEquity,"treasury equity");const latest=latestByAsset(input.opportunities,nowMs);const latestRaw=latestRawByAsset(input.opportunities,nowMs);

  for(const p of [...state.positions]){
    const latestAny=latestRaw.get(p.assetId);const mark=latestAny?.referencePrice??marks[p.assetId]??p.entryPrice;const pnlBps=directionalReturnBps(p,mark);p.highWaterPnlBps=Math.max(p.highWaterPnlBps,pnlBps);const ageSeconds=Math.max(0,(nowMs-Number(time(p.openedAt)??nowMs))/1000);const fresh=latest.get(p.assetId);const freshSame=fresh&&fresh.direction===p.direction?fresh:null;const currentEdge=latestAny&&latestAny.direction===p.direction?latestAny.expectedNetEdgeBps:(freshSame?.expectedNetEdgeBps??p.latestExpectedNetEdgeBps);p.latestExpectedNetEdgeBps=currentEdge;
    let reason:TreasuryExitReason|null=null;
    if(pnlBps<=RISK_STOP_BPS)reason="RISK_STOP";
    else if(isFreshRaw(latestAny,nowMs)&&latestAny&&latestAny.direction!==p.direction&&validOpportunity(latestAny,nowMs))reason="DIRECTION_FLIP";
    else if(isFreshRaw(latestAny,nowMs)&&latestAny&&latestAny.direction===p.direction&&(!latestAny.pitClear||latestAny.recommendation!=="ALLOW_EDGE"||latestAny.expectedNetEdgeBps<=0))reason="EDGE_INVALIDATED";
    else if(freshSame&&pnlBps>=PROFIT_DECAY_ARM_BPS&&freshSame.expectedNetEdgeBps<Math.max(MIN_EDGE_BPS,p.entryExpectedNetEdgeBps*.25))reason="PROFIT_EDGE_DECAY";
    else if(!freshSame&&ageSeconds>EDGE_STALE_SECONDS)reason="STALE_EDGE";
    else if(ageSeconds>MAX_HOLD_SECONDS&&pnlBps<=0)reason="TIME_DECAY";
    if(reason){applyExit(state,p,mark,currentEdge,latestAny?.sourceDecisionId??p.sourceDecisionId,reason,actions);if(reason!=="DIRECTION_FLIP")noReopenAssets.add(p.assetId);}
  }

  const candidates=[...latest.values()].sort((a,b)=>opportunityScore(b)-opportunityScore(a));
  for(const o of candidates){
    if(noReopenAssets.has(o.assetId)||state.positions.some(p=>p.assetId===o.assetId))continue;
    let equity=treasuryEquityUsd(state,{...marks,[o.assetId]:o.referencePrice});
    let target=targetCapitalUsd(o,equity);
    let desired=Math.min(target,availableCapitalFor(o,state,equity));
    let minimumTicket=Math.min(equity*MIN_POSITION_PCT,100);

    // A stronger opportunity is allowed to recycle weaker deployed capital even when
    // there is still some cash available. The switch must beat the remaining HOLD value
    // after accounting for the old position's incremental exit cost; the candidate's own
    // full round-trip cost is already included in expectedNetEdgeBps.
    while(state.positions.length&&(state.positions.length>=MAX_POSITIONS||desired+minimumTicket<target)){
      const weakest=[...state.positions].sort((a,b)=>latestEdgeForPosition(a,latest)-latestEdgeForPosition(b,latest))[0];
      const weakEdge=latestEdgeForPosition(weakest,latest);const weakOpp=latest.get(weakest.assetId);const weakScore=weakOpp?opportunityScore(weakOpp):Math.max(0,weakEdge)*.5;
      const netSwitchAdvantage=replacementAdvantageBps(o,weakest,latest);
      const dominates=netSwitchAdvantage>=REPLACEMENT_EDGE_ADVANTAGE_BPS&&opportunityScore(o)>weakScore*REPLACEMENT_SCORE_MULTIPLIER;
      if(!dominates)break;
      const mark=marks[weakest.assetId]??weakest.entryPrice;
      applyExit(state,weakest,mark,weakEdge,o.sourceDecisionId,"OPPORTUNITY_REPLACEMENT",actions);noReopenAssets.add(weakest.assetId);
      equity=treasuryEquityUsd(state,{...marks,[o.assetId]:o.referencePrice});target=targetCapitalUsd(o,equity);desired=Math.min(target,availableCapitalFor(o,state,equity));minimumTicket=Math.min(equity*MIN_POSITION_PCT,100);
    }

    if(state.positions.length>=MAX_POSITIONS)continue;if(desired<minimumTicket)continue;applyOpen(state,o,desired,input.positionIdFor(o),input.observedAt,actions);
  }

  const afterMarks={...marks};for(const o of candidates)afterMarks[o.assetId]=o.referencePrice;state.observedAt=input.observedAt;const afterEquity=treasuryEquityUsd(state,afterMarks);const deployed=deploymentUsd(state);
  // deploymentPct is an operational utilization ratio, not leverage. After an adverse mark,
  // fixed deployed principal can exceed current MTM equity even though Brian opened no new
  // leverage. Normalize by max(MTM equity, deployed principal) so a valid all-in shadow book
  // stays at 100% instead of becoming >100% and failing persistence before it can exit.
  const deploymentDenominator=Math.max(afterEquity,deployed);const deploymentPct=deploymentDenominator>0?deployed/deploymentDenominator:0;const cashReservePct=afterEquity>0?state.cashUsd/afterEquity:0;
  if(deploymentPct>MAX_DEPLOYMENT_PCT+1e-6)blockedReasons.push("deployment exceeds available SHADOW equity");if(cashReservePct<MIN_CASH_RESERVE_PCT-1e-6)blockedReasons.push("cash became negative after point-in-time costs");
  return{version:BRIAN_TREASURY_VERSION,observedAt:input.observedAt,beforeEquityUsd:beforeEquity,afterEquityUsd:afterEquity,state,actions,deploymentUsd:deployed,deploymentPct,cashReservePct,blockedReasons};
}

export function initialTreasuryState(observedAt:string,startingEquityUsd=BRIAN_TREASURY_STARTING_EQUITY_USD):TreasuryState{
  assertFinitePositive(startingEquityUsd,"startingEquityUsd");return{observedAt,startingEquityUsd,cashUsd:startingEquityUsd,realizedPnlUsd:0,cumulativeCostsUsd:0,positions:[]};
}