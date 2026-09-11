import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import { EVOLUTION_EVIDENCE_CLASS } from "../_shared/evolution_contract.ts";
import {
  EVOLUTION_LAB_VERSION,
  measureActionGateExperiment,
  measureGateExperiment,
  type ChallengerDecisionLabel,
  type ProspectiveOutcomePoint,
} from "../_shared/evolution_lab.ts";
import type { ExperimentMetrics } from "../_shared/evolution_research.ts";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const COLLECTOR_ID="brian-evolution-experiment-runner-v1";
const LEASE_SECONDS=240;
const OUTCOME_HORIZON_SECONDS=900;
const LOOKBACK_MS=7*24*3600_000;
const MIN_REMEASURE_MS=3*3600_000;
const SUPPORTED_KINDS=new Set(["ACTION_GATE","EXPECTED_EDGE","RELIABILITY_FEEDBACK","COST_CONTROL"]);

function out(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}});}
async function sha(value:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(value)));return[...d].map(b=>b.toString(16).padStart(2,"0")).join("");}
function meta(v:unknown):Record<string,unknown>{return(v??{}) as Record<string,unknown>;}
function metricRow(metric:ExperimentMetrics){return{
  samples:metric.samples,regimes:metric.regimes,net_edge_bps:metric.netEdgeBps,gross_edge_bps:metric.grossEdgeBps,
  max_drawdown_pct:metric.maxDrawdownPct,favorable_after_cost_rate:metric.favorableAfterCostRate,turnover:metric.turnover,
  cost_bps:metric.costBps,leakage_detected:metric.leakageDetected,data_quality_ok:metric.dataQualityOk,
  stability_score:metric.stabilityScore,complexity_delta:metric.complexityDelta,
};}

async function latestHypothesisKinds(){
  const q=await db.from("brian_evolution_hypothesis_snapshots")
    .select("hypothesis_id,observed_at,metadata").order("observed_at",{ascending:false}).limit(500);
  if(q.error)throw new Error(`hypotheses:${q.error.message}`);
  const kinds=new Map<string,string>();
  for(const row of q.data??[]){const id=String(row.hypothesis_id);if(kinds.has(id))continue;kinds.set(id,String(meta(row.metadata).hypothesis_kind??""));}
  return kinds;
}

async function candidateExperiments(){
  const [expQ,kinds]=await Promise.all([
    db.from("brian_evolution_experiments")
      .select("experiment_id,hypothesis_id,created_at_source,control_version,challenger_version,mode,stage")
      .eq("mode","PROSPECTIVE_SHADOW").in("stage",["EXPERIMENTAL","SHADOW_CANDIDATE"]).order("created_at_source",{ascending:false}).limit(100),
    latestHypothesisKinds(),
  ]);
  if(expQ.error)throw new Error(`experiments:${expQ.error.message}`);
  return(expQ.data??[]).map(row=>({...row,hypothesis_kind:kinds.get(String(row.hypothesis_id))??""})).filter(row=>SUPPORTED_KINDS.has(row.hypothesis_kind)).slice(0,20);
}

async function recentlyMeasured(experimentId:string,nowMs:number){
  const q=await db.from("brian_evolution_experiment_results").select("measured_at").eq("experiment_id",experimentId).order("measured_at",{ascending:false}).limit(1).maybeSingle();
  if(q.error)throw new Error(`latest_result:${q.error.message}`);
  if(!q.data?.measured_at)return false;
  const t=Date.parse(String(q.data.measured_at));return Number.isFinite(t)&&nowMs-t<MIN_REMEASURE_MS;
}

async function loadOutcomes(observedAt:string,eligibleIds:Set<string>):Promise<ProspectiveOutcomePoint[]>{
  const since=new Date(Date.parse(observedAt)-LOOKBACK_MS).toISOString();
  const q=await db.from("brian_alpha_decision_outcomes")
    .select("decision_id,observed_at,horizon_seconds,direction_adjusted_return,classification,metadata,resolved_at")
    .eq("horizon_seconds",OUTCOME_HORIZON_SECONDS).gte("resolved_at",since).order("observed_at",{ascending:true}).limit(12000);
  if(q.error)throw new Error(`outcomes:${q.error.message}`);
  const outcomes:ProspectiveOutcomePoint[]=[];
  for(const row of q.data??[]){
    const decisionId=String(row.decision_id);if(!eligibleIds.has(decisionId))continue;
    const directionAdjustedReturn=Number(row.direction_adjusted_return),cost=Number(meta(row.metadata).estimated_round_trip_cost_bps);
    if(!Number.isFinite(directionAdjustedReturn)||!Number.isFinite(cost))continue;
    outcomes.push({decisionId,observedAt:String(row.observed_at),directionAdjustedReturn,estimatedRoundTripCostBps:cost,classification:String(row.classification??"")});
  }
  return outcomes;
}

async function loadActionGateEvidence(observedAt:string){
  const since=new Date(Date.parse(observedAt)-LOOKBACK_MS).toISOString();
  const labelsQ=await db.from("brian_alpha_calibration_challenger")
    .select("decision_id,canonical_action,challenger_action,observed_at,evaluated_at")
    .gte("observed_at",since).order("observed_at",{ascending:true}).limit(12000);
  if(labelsQ.error)throw new Error(`challenger_labels:${labelsQ.error.message}`);
  const labels:ChallengerDecisionLabel[]=[],openIds=new Set<string>();
  for(const row of labelsQ.data??[]){const canonical=String(row.canonical_action);if(canonical!=="OPEN_LONG"&&canonical!=="OPEN_SHORT")continue;const decisionId=String(row.decision_id);openIds.add(decisionId);labels.push({decisionId,challengerAction:String(row.challenger_action)});}
  return{labels,outcomes:await loadOutcomes(observedAt,openIds),source:"brian_alpha_calibration_challenger",allowedLabel:"ALLOW_ACTION",complexityDelta:1};
}

async function loadExpectedEdgeEvidence(observedAt:string){
  const since=new Date(Date.parse(observedAt)-LOOKBACK_MS).toISOString();
  const labelsQ=await db.from("brian_alpha_expected_edge_challenger")
    .select("decision_id,canonical_action,recommendation,observed_at,evaluated_at,pit_clear,model_version")
    .gte("observed_at",since).order("observed_at",{ascending:true}).limit(12000);
  if(labelsQ.error)throw new Error(`expected_edge_labels:${labelsQ.error.message}`);
  const labels:ChallengerDecisionLabel[]=[],openIds=new Set<string>();
  for(const row of labelsQ.data??[]){if(row.pit_clear!==true)continue;const canonical=String(row.canonical_action);if(canonical!=="OPEN_LONG"&&canonical!=="OPEN_SHORT")continue;const decisionId=String(row.decision_id);openIds.add(decisionId);labels.push({decisionId,challengerAction:String(row.recommendation)});}
  return{labels,outcomes:await loadOutcomes(observedAt,openIds),source:"brian_alpha_expected_edge_challenger",allowedLabel:"ALLOW_EDGE",complexityDelta:2};
}

async function persistMeasurement(experiment:Record<string,unknown>,observedAt:string){
  const kind=String(experiment.hypothesis_kind??"");
  const evidence=kind==="ACTION_GATE"?await loadActionGateEvidence(observedAt):await loadExpectedEdgeEvidence(observedAt);
  const measured=kind==="ACTION_GATE"?measureActionGateExperiment(evidence.outcomes,evidence.labels):measureGateExperiment(evidence.outcomes,evidence.labels,evidence.allowedLabel,evidence.complexityDelta);
  const rows:Record<string,unknown>[]=[];
  for(const [role,metric] of [["CONTROL",measured.control],["CHALLENGER",measured.challenger]] as const){
    const resultId=await sha(`evolution-result|${String(experiment.experiment_id)}|${role}|${observedAt}`);
    rows.push({
      result_id:resultId,experiment_id:String(experiment.experiment_id),measured_at:observedAt,role,...metricRow(metric),
      metric_payload:{lab_version:EVOLUTION_LAB_VERSION,horizon_seconds:OUTCOME_HORIZON_SECONDS,measurement_kind:`PROSPECTIVE_${kind}`,hypothesis_kind:kind,lineage:measured.lineage,control_version:experiment.control_version,challenger_version:experiment.challenger_version,label_source:evidence.source},
      evidence_refs:[`${evidence.source}:7d`,`brian_alpha_decision_outcomes:${OUTCOME_HORIZON_SECONDS}s:7d`],
      evidence_class:EVOLUTION_EVIDENCE_CLASS,shadow_only:true,live_execution:false,
    });
  }
  const q=await db.from("brian_evolution_experiment_results").insert(rows);if(q.error)throw new Error(`persist_results:${q.error.message}`);
  return{experiment_id:experiment.experiment_id,hypothesis_kind:kind,control:measured.control,challenger:measured.challenger,lineage:measured.lineage,stored:rows.length};
}

async function receipt(startedAt:string,status:"SUCCESS"|"FAILED"|"SKIPPED",observed:number,stored:number,error?:unknown){
  const finishedAt=new Date().toISOString();const runId=await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  const q=await db.from("brian_collector_runs").insert({run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:finishedAt,status,observed_records:observed,stored_records:stored,degraded_sources:[],error_class:error?"EVOLUTION_EXPERIMENT_RUNNER_ERROR":null,error_message:error?String(error).slice(0,1200):null,metadata:{lab_version:EVOLUTION_LAB_VERSION,canonical_mutation:false,supported_hypothesis_kinds:[...SUPPORTED_KINDS]},evidence_class:EVOLUTION_EVIDENCE_CLASS,shadow_only:true,live_execution:false});
  if(q.error)console.error("experiment-runner receipt",q.error.message);
}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);const startedAt=new Date().toISOString();
  try{await requireCronAuth(req,db);}catch(error){return out({status:"UNAUTHORIZED",error:String(error),shadow_only:true,live_execution:false},401);}
  try{
    const lease=await withCollectorLease(db,COLLECTOR_ID,LEASE_SECONDS,async()=>{
      const now=new Date().toISOString(),nowMs=Date.parse(now),experiments=await candidateExperiments();const results:unknown[]=[];let skippedRecent=0,stored=0;
      for(const experiment of experiments){if(await recentlyMeasured(String(experiment.experiment_id),nowMs)){skippedRecent++;continue;}const result=await persistMeasurement(experiment as Record<string,unknown>,now);results.push(result);stored+=2;}
      await receipt(startedAt,"SUCCESS",experiments.length,stored);
      return{status:"SUCCESS",collector_id:COLLECTOR_ID,lab_version:EVOLUTION_LAB_VERSION,experiments_considered:experiments.length,measured:results.length,skipped_recent:skippedRecent,results,canonical_mutation:false,autonomous_apply_allowed:false,shadow_only:true,live_execution:false};
    });
    if(lease.contended){await receipt(startedAt,"SKIPPED",0,0);return out({status:"SKIPPED_LEASE_CONTENDED",shadow_only:true,live_execution:false});}
    return out(lease.value);
  }catch(error){await receipt(startedAt,"FAILED",0,0,error);return out({status:"FAILED",error:String(error),canonical_mutation:false,shadow_only:true,live_execution:false},500);}
});
