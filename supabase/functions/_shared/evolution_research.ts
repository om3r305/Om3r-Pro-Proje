import {
  assertAutonomousChangeSetAllowed,
  EVOLUTION_EVIDENCE_CLASS,
  type EvolutionStage,
} from "./evolution_contract.ts";

export const EVOLUTION_RESEARCH_VERSION = "brian.evolution-research.v2";

export interface GapSignal { gapId:string; capabilityId:string; severity:"LOW"|"MEDIUM"|"HIGH"|"CRITICAL"; reason:string; suggestedAction:string; evidenceRefs:string[]; }
export interface ChallengerSignal { allowAction:number; downgradeToWait:number; keepWait:number; allowAvgCostAdjustedBps:number|null; downgradeAvgCostAdjustedBps:number|null; observedAt:string; evidenceRefs:string[]; }
export interface OutcomeSignal { horizonSeconds:number; samples:number; grossPositiveRate:number|null; avgDirectionBps:number|null; avgAfterCostBps:number|null; favorableAfterCostRate:number|null; observedAt:string; evidenceRefs:string[]; }
export interface ReliabilitySignal { sensorFamily:string; samples:number; canonicalReliability:number|null; measuredScore:number|null; avgCostAdjustedBps:number|null; observedAt:string; evidenceRefs:string[]; }
export interface ResearchInputs { gaps:GapSignal[]; challenger:ChallengerSignal|null; outcomes:OutcomeSignal[]; reliability:ReliabilitySignal[]; observedAt:string; }
export interface HypothesisCandidate { hypothesisId:string; observedAt:string; problemStatement:string; proposedMechanism:string; targetCapabilities:string[]; evidenceRefs:string[]; counterEvidenceRefs:string[]; measurableSuccessCriteria:string[]; stage:EvolutionStage; uncertainty:number; priority:number; hypothesisKind:"CAPABILITY_GAP"|"ACTION_GATE"|"EXPECTED_EDGE"|"RELIABILITY_FEEDBACK"|"COST_CONTROL"|"DRIFT"; metadata:Record<string,unknown>; }
export interface ExperimentPlan { experimentId:string; hypothesisId:string; createdAt:string; controlVersion:string; challengerVersion:string; mode:"REPLAY"|"STRESS"|"PROSPECTIVE_SHADOW"; minimumSamples:number; minimumRegimes:number; successMetrics:string[]; hardFailConditions:string[]; contaminationRules:string[]; stage:EvolutionStage; }
export interface ExperimentMetrics { samples:number; regimes:number; netEdgeBps:number|null; grossEdgeBps:number|null; maxDrawdownPct:number|null; favorableAfterCostRate:number|null; turnover:number|null; costBps:number|null; leakageDetected:boolean; dataQualityOk:boolean; stabilityScore:number|null; complexityDelta:number; }
export interface PromotionDecision { decision:"PROMOTE_CANDIDATE"|"KEEP_EXPERIMENTAL"|"REJECT"; score:number; reasons:string[]; requiredNextStage:EvolutionStage; }
export interface DriftSnapshot { driftId:string; observedAt:string; metricId:string; baseline:number; recent:number; absoluteDelta:number; relativeDelta:number|null; severity:"NONE"|"WATCH"|"MATERIAL"|"SEVERE"; direction:"UP"|"DOWN"|"FLAT"; evidenceRefs:string[]; }
export interface CodeCandidatePlan { candidateId:string; hypothesisId:string; proposedAt:string; parentCommit:string; changedPaths:string[]; testPlan:string[]; contaminationDeclaration:string; stage:"EXPERIMENTAL"; autonomousApplyAllowed:false; metadata:Record<string,unknown>; }

const clamp=(n:number)=>Number.isFinite(n)?Math.max(0,Math.min(1,n)):0;
const id=(...parts:Array<string|number|null|undefined>)=>parts.map(v=>String(v??"").trim().toLowerCase().replace(/[^a-z0-9:_-]+/g,"-")).join("|");
const hypothesisIdentity=(kind:string,...parts:Array<string|number|null|undefined>)=>id("hypothesis",EVOLUTION_RESEARCH_VERSION,kind,...parts);

function outcomeEvidence(inputs:ResearchInputs):string[]{return inputs.outcomes.flatMap(x=>x.evidenceRefs).slice(0,80);}
function addUnique(out:HypothesisCandidate[], candidate:HypothesisCandidate){if(!out.some(x=>x.hypothesisId===candidate.hypothesisId))out.push(candidate);}

export function generateResearchHypotheses(inputs:ResearchInputs):HypothesisCandidate[]{
  const out:HypothesisCandidate[]=[];
  for(const gap of inputs.gaps.filter(g=>g.severity==="CRITICAL"||g.severity==="HIGH").slice(0,12)){
    addUnique(out,{hypothesisId:hypothesisIdentity("gap",gap.capabilityId),observedAt:inputs.observedAt,problemStatement:gap.reason,proposedMechanism:gap.suggestedAction,targetCapabilities:[gap.capabilityId],evidenceRefs:[...gap.evidenceRefs],counterEvidenceRefs:[],measurableSuccessCriteria:["capability produces fresh prospective evidence","collector remains healthy across multiple cadence windows","new capability does not bypass source-truth or execution boundaries"],stage:"RESEARCHING",uncertainty:gap.severity==="CRITICAL"?0.35:0.45,priority:gap.severity==="CRITICAL"?0.95:0.78,hypothesisKind:"CAPABILITY_GAP",metadata:{gap_id:gap.gapId,severity:gap.severity,identity_version:EVOLUTION_RESEARCH_VERSION}});
  }

  const challenger=inputs.challenger;
  if(challenger&&challenger.downgradeToWait>Math.max(25,challenger.allowAction*2)){
    const ratio=challenger.downgradeToWait/Math.max(1,challenger.allowAction);
    addUnique(out,{hypothesisId:hypothesisIdentity("action-gate","canonical-open-gate"),observedAt:inputs.observedAt,problemStatement:`Calibration challenger rejects substantially more canonical actions than it allows (${ratio.toFixed(1)}x downgrade/allow).`,proposedMechanism:"Create a challenger action gate that requires mature prospective calibration before canonical OPEN eligibility; keep canonical mutation disabled until prospective A/B gates pass.",targetCapabilities:["alpha.compiler","research.calibration"],evidenceRefs:[...challenger.evidenceRefs],counterEvidenceRefs:[],measurableSuccessCriteria:["prospective after-cost edge improves versus canonical control","favorable-after-cost rate improves without unacceptable opportunity loss","minimum sample and regime gates pass"],stage:"RESEARCHING",uncertainty:0.28,priority:0.92,hypothesisKind:"ACTION_GATE",metadata:{downgrade_to_wait:challenger.downgradeToWait,allow_action:challenger.allowAction,ratio,identity_version:EVOLUTION_RESEARCH_VERSION}});
  }

  const matureOutcomes=inputs.outcomes.filter(o=>o.samples>=100&&o.avgAfterCostBps!=null);
  const negative=matureOutcomes.filter(o=>Number(o.avgAfterCostBps)<0);
  if(negative.length){
    const worst=[...negative].sort((a,b)=>Number(a.avgAfterCostBps)-Number(b.avgAfterCostBps))[0];
    addUnique(out,{hypothesisId:hypothesisIdentity("expected-edge",worst.horizonSeconds),observedAt:inputs.observedAt,problemStatement:`Canonical actions show negative average after-cost outcome at ${worst.horizonSeconds}s (${Number(worst.avgAfterCostBps).toFixed(2)} bps across ${worst.samples} samples).`,proposedMechanism:"Estimate gross expected move and subtract dynamic cost, uncertainty and decay before action eligibility; abstain when expected net edge is not positive with margin.",targetCapabilities:["alpha.expected-edge","alpha.compiler"],evidenceRefs:outcomeEvidence(inputs),counterEvidenceRefs:[],measurableSuccessCriteria:["challenger expected-net-edge > 0 prospectively","after-cost outcome distribution improves versus control","cost model remains point-in-time and fillability-aware"],stage:"RESEARCHING",uncertainty:0.25,priority:0.98,hypothesisKind:"EXPECTED_EDGE",metadata:{horizon_seconds:worst.horizonSeconds,samples:worst.samples,avg_after_cost_bps:worst.avgAfterCostBps,identity_version:EVOLUTION_RESEARCH_VERSION}});
  }

  const frozen=inputs.reliability.filter(r=>r.samples>=100&&r.canonicalReliability!=null&&Math.abs(Number(r.canonicalReliability)-0.5)<1e-9&&r.measuredScore!=null&&Math.abs(Number(r.measuredScore)-0.5)>=0.05);
  if(frozen.length){
    addUnique(out,{hypothesisId:hypothesisIdentity("reliability-feedback","canonical-neutral-weight"),observedAt:inputs.observedAt,problemStatement:`${frozen.length} mature sensor families have measured reliability diverging from the canonical 0.5 weight.`,proposedMechanism:"Use bounded, lagged prospective reliability weights in a challenger compiler with shrinkage toward 0.5 and automatic decay under drift.",targetCapabilities:["research.calibration","alpha.compiler"],evidenceRefs:frozen.flatMap(r=>r.evidenceRefs).slice(0,100),counterEvidenceRefs:[],measurableSuccessCriteria:["weighted challenger improves after-cost edge out of sample","no sensor can dominate without mature independent samples","weights decay toward neutral when recent evidence weakens"],stage:"RESEARCHING",uncertainty:0.32,priority:0.88,hypothesisKind:"RELIABILITY_FEEDBACK",metadata:{families:frozen.map(r=>r.sensorFamily),identity_version:EVOLUTION_RESEARCH_VERSION}});
  }

  const highCost=inputs.outcomes.filter(o=>o.samples>=100&&o.avgDirectionBps!=null&&o.avgAfterCostBps!=null&&Number(o.avgDirectionBps)-Number(o.avgAfterCostBps)>=15);
  if(highCost.length){
    addUnique(out,{hypothesisId:hypothesisIdentity("cost-control","dynamic-round-trip-cost"),observedAt:inputs.observedAt,problemStatement:"Execution-cost burden materially exceeds observed directional edge in mature prospective outcomes.",proposedMechanism:"Rank opportunities by net edge after dynamic spread/fee/depth cost and reject trades whose edge-to-cost margin is insufficient.",targetCapabilities:["alpha.expected-edge","market.l2"],evidenceRefs:outcomeEvidence(inputs),counterEvidenceRefs:[],measurableSuccessCriteria:["net edge remains positive after contemporaneous cost","turnover falls when cost dominates","missed-opportunity audit does not show systematic profitable exclusions"],stage:"RESEARCHING",uncertainty:0.22,priority:0.94,hypothesisKind:"COST_CONTROL",metadata:{mature_horizons:highCost.map(o=>o.horizonSeconds),identity_version:EVOLUTION_RESEARCH_VERSION}});
  }

  return out.sort((a,b)=>b.priority-a.priority);
}

export function buildExperimentPlan(h:HypothesisCandidate,controlVersion:string,challengerVersion:string,mode:ExperimentPlan["mode"]="PROSPECTIVE_SHADOW"):ExperimentPlan{
  if(!controlVersion.trim())throw new Error("control version required");
  if(!challengerVersion.trim())throw new Error("challenger version required");
  return{experimentId:id("experiment",h.hypothesisId,controlVersion,challengerVersion,mode),hypothesisId:h.hypothesisId,createdAt:h.observedAt,controlVersion,challengerVersion,mode,minimumSamples:mode==="PROSPECTIVE_SHADOW"?300:150,minimumRegimes:3,successMetrics:["net_edge_bps","favorable_after_cost_rate","max_drawdown_pct","stability_score"],hardFailConditions:["point-in-time leakage detected","live execution surface enabled","protected scope modified","data quality below required threshold","challenger drawdown materially exceeds control without compensating edge"],contaminationRules:["no future data in features","decision-time inputs immutable","replay and prospective evidence labeled separately","DIP data and runtime excluded"],stage:"EXPERIMENTAL"};
}

export function evaluatePromotion(control:ExperimentMetrics,challenger:ExperimentMetrics):PromotionDecision{
  const reasons:string[]=[];
  if(challenger.leakageDetected){return{decision:"REJECT",score:0,reasons:["point-in-time leakage detected"],requiredNextStage:"REJECTED"};}
  if(!challenger.dataQualityOk){return{decision:"REJECT",score:0,reasons:["challenger data quality failed"],requiredNextStage:"REJECTED"};}
  if(challenger.samples<300||challenger.regimes<3){return{decision:"KEEP_EXPERIMENTAL",score:0.25,reasons:[`insufficient prospective coverage: ${challenger.samples} samples / ${challenger.regimes} regimes`],requiredNextStage:"EXPERIMENTAL"};}
  const net=Number(challenger.netEdgeBps??-Infinity),controlNet=Number(control.netEdgeBps??-Infinity);const favorable=Number(challenger.favorableAfterCostRate??0),controlFav=Number(control.favorableAfterCostRate??0);const dd=Number(challenger.maxDrawdownPct??Infinity),controlDd=Number(control.maxDrawdownPct??Infinity);const stability=Number(challenger.stabilityScore??0);const complexityPenalty=Math.min(0.25,Math.max(0,challenger.complexityDelta)*0.02);
  let score=0;if(net>0){score+=0.35;reasons.push("positive prospective net edge");}if(net>controlNet){score+=0.25;reasons.push("net edge beats control");}if(favorable>controlFav){score+=0.15;reasons.push("after-cost favorable rate beats control");}if(dd<=controlDd*1.15){score+=0.12;reasons.push("drawdown is not materially worse than control");}if(stability>=0.6){score+=0.13;reasons.push("multi-window stability gate passed");}score=clamp(score-complexityPenalty);if(complexityPenalty)reasons.push(`complexity penalty ${complexityPenalty.toFixed(2)}`);
  if(net<=0)reasons.push("prospective net edge is not positive");
  if(score>=0.72&&net>0&&net>controlNet&&favorable>=controlFav){return{decision:"PROMOTE_CANDIDATE",score,reasons,requiredNextStage:"SHADOW_CANDIDATE"};}
  return{decision:"KEEP_EXPERIMENTAL",score,reasons,requiredNextStage:"EXPERIMENTAL"};
}

export function detectDrift(metricId:string,baseline:number,recent:number,observedAt:string,evidenceRefs:string[]=[]):DriftSnapshot{
  const delta=recent-baseline;const denom=Math.abs(baseline);const relative=denom>1e-9?delta/denom:null;const scale=relative==null?Math.abs(delta):Math.abs(relative);const severity:DriftSnapshot["severity"]=scale>=0.5?"SEVERE":scale>=0.25?"MATERIAL":scale>=0.1?"WATCH":"NONE";return{driftId:id("drift",metricId,observedAt),observedAt,metricId,baseline,recent,absoluteDelta:delta,relativeDelta:relative,severity,direction:delta>1e-12?"UP":delta<-1e-12?"DOWN":"FLAT",evidenceRefs:[...evidenceRefs]};
}

export function buildCodeCandidatePlan(h:HypothesisCandidate,parentCommit:string,changedPaths:string[]):CodeCandidatePlan{
  assertAutonomousChangeSetAllowed(changedPaths);
  if(!parentCommit.trim())throw new Error("parent commit required");
  if(!changedPaths.length)throw new Error("at least one changed path required");
  return{candidateId:id("code",h.hypothesisId,parentCommit,...changedPaths),hypothesisId:h.hypothesisId,proposedAt:h.observedAt,parentCommit,changedPaths:[...changedPaths],testPlan:["unit/type tests","replay on immutable point-in-time evidence","stress/adversarial cases","prospective shadow A/B before promotion"],contaminationDeclaration:"Candidate must not use post-decision evidence during feature generation; replay and prospective evidence stay separately labeled.",stage:"EXPERIMENTAL",autonomousApplyAllowed:false,metadata:{research_version:EVOLUTION_RESEARCH_VERSION,evidence_class:EVOLUTION_EVIDENCE_CLASS,direct_canonical_apply:false}};
}
