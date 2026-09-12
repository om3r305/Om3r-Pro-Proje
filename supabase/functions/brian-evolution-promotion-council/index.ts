import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import { EVOLUTION_EVIDENCE_CLASS } from "../_shared/evolution_contract.ts";
import {
  detectDrift,
  evaluatePromotion,
  EVOLUTION_RESEARCH_VERSION,
  type ExperimentMetrics,
} from "../_shared/evolution_research.ts";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const COLLECTOR_ID="brian-evolution-promotion-council-v1";
const LEASE_SECONDS=180;
// Experiment measurements are refreshed hourly. Keep the anti-duplicate window slightly
// below an hour so fresh adverse evidence can revoke a promotion on the very next council run.
const MIN_REDECIDE_MS=55*60_000;

function out(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}});}
async function sha(value:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(value)));return[...d].map(b=>b.toString(16).padStart(2,"0")).join("");}
const n=(v:unknown):number|null=>Number.isFinite(Number(v))?Number(v):null;
function toMetrics(row:Record<string,unknown>):ExperimentMetrics{return{
  samples:Math.max(0,Math.trunc(Number(row.samples??0))),regimes:Math.max(0,Math.trunc(Number(row.regimes??0))),
  netEdgeBps:n(row.net_edge_bps),grossEdgeBps:n(row.gross_edge_bps),maxDrawdownPct:n(row.max_drawdown_pct),
  favorableAfterCostRate:n(row.favorable_after_cost_rate),turnover:n(row.turnover),costBps:n(row.cost_bps),
  leakageDetected:row.leakage_detected===true,dataQualityOk:row.data_quality_ok===true,stabilityScore:n(row.stability_score),
  complexityDelta:Math.trunc(Number(row.complexity_delta??0)),
};}

async function latestPair(experimentId:string){
  const q=await db.from("brian_evolution_experiment_results")
    .select("result_id,experiment_id,measured_at,role,samples,regimes,net_edge_bps,gross_edge_bps,max_drawdown_pct,favorable_after_cost_rate,turnover,cost_bps,leakage_detected,data_quality_ok,stability_score,complexity_delta,evidence_refs,metric_payload")
    .eq("experiment_id",experimentId).order("measured_at",{ascending:false}).limit(20);
  if(q.error)throw new Error(`experiment_results:${q.error.message}`);
  const control=(q.data??[]).find(row=>String(row.role)==="CONTROL") as Record<string,unknown>|undefined;
  const challenger=(q.data??[]).find(row=>String(row.role)==="CHALLENGER") as Record<string,unknown>|undefined;
  const challengers=(q.data??[]).filter(row=>String(row.role)==="CHALLENGER") as Record<string,unknown>[];
  return{control,challenger,previousChallenger:challengers[1]};
}

async function recentlyDecided(experimentId:string,nowMs:number){
  const q=await db.from("brian_evolution_promotion_decisions").select("decided_at").eq("experiment_id",experimentId).order("decided_at",{ascending:false}).limit(1).maybeSingle();
  if(q.error)throw new Error(`latest_promotion:${q.error.message}`);
  if(!q.data?.decided_at)return false;const t=Date.parse(String(q.data.decided_at));return Number.isFinite(t)&&nowMs-t<MIN_REDECIDE_MS;
}

async function persistDrift(experimentId:string,current:Record<string,unknown>,previous:Record<string,unknown>|undefined,observedAt:string){
  if(!previous)return 0;let stored=0;
  for(const [metricId,column] of [["net_edge_bps","net_edge_bps"],["favorable_after_cost_rate","favorable_after_cost_rate"]] as const){
    const baseline=n(previous[column]),recent=n(current[column]);if(baseline==null||recent==null)continue;
    const drift=detectDrift(`experiment:${experimentId}:${metricId}`,baseline,recent,observedAt,[String(previous.result_id),String(current.result_id)]);
    const q=await db.from("brian_evolution_drift_snapshots").insert({
      drift_id:await sha(`${drift.driftId}|${String(current.result_id)}`),observed_at:drift.observedAt,metric_id:drift.metricId,baseline:drift.baseline,recent:drift.recent,
      absolute_delta:drift.absoluteDelta,relative_delta:drift.relativeDelta,severity:drift.severity,direction:drift.direction,evidence_refs:drift.evidenceRefs,
      metadata:{research_version:EVOLUTION_RESEARCH_VERSION,experiment_id:experimentId,canonical_mutation:false},evidence_class:EVOLUTION_EVIDENCE_CLASS,shadow_only:true,live_execution:false,
    });
    if(q.error&&!String(q.error.message).toLowerCase().includes("duplicate"))throw new Error(`drift:${q.error.message}`);stored++;
  }
  return stored;
}

async function decideExperiment(experiment:Record<string,unknown>,observedAt:string){
  const experimentId=String(experiment.experiment_id);const pair=await latestPair(experimentId);
  if(!pair.control||!pair.challenger)return{experiment_id:experimentId,status:"WAITING_FOR_PAIRED_RESULTS",stored:0,drift:0};
  const control=toMetrics(pair.control),challenger=toMetrics(pair.challenger),decision=evaluatePromotion(control,challenger);
  const decisionId=await sha(`promotion|${experimentId}|${String(pair.control.result_id)}|${String(pair.challenger.result_id)}|${decision.decision}`);
  const q=await db.from("brian_evolution_promotion_decisions").upsert({
    decision_id:decisionId,experiment_id:experimentId,decided_at:observedAt,decision:decision.decision,score:decision.score,reasons:decision.reasons,
    required_next_stage:decision.requiredNextStage,control_result_id:String(pair.control.result_id),challenger_result_id:String(pair.challenger.result_id),
    metadata:{research_version:EVOLUTION_RESEARCH_VERSION,hypothesis_id:experiment.hypothesis_id,canonical_stage_mutation:false,human_promotion_required:true},
    evidence_class:EVOLUTION_EVIDENCE_CLASS,shadow_only:true,live_execution:false,autonomous_apply_allowed:false,
  },{onConflict:"decision_id",ignoreDuplicates:true});
  if(q.error)throw new Error(`promotion:${q.error.message}`);
  const drift=await persistDrift(experimentId,pair.challenger,pair.previousChallenger,observedAt);
  return{experiment_id:experimentId,status:"DECIDED",decision:decision.decision,score:decision.score,reasons:decision.reasons,required_next_stage:decision.requiredNextStage,stored:1,drift};
}

async function receipt(startedAt:string,status:"SUCCESS"|"FAILED"|"SKIPPED",observed:number,stored:number,error?:unknown){
  const finishedAt=new Date().toISOString(),runId=await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  const q=await db.from("brian_collector_runs").insert({run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:finishedAt,status,observed_records:observed,stored_records:stored,degraded_sources:[],error_class:error?"EVOLUTION_PROMOTION_COUNCIL_ERROR":null,error_message:error?String(error).slice(0,1200):null,metadata:{research_version:EVOLUTION_RESEARCH_VERSION,canonical_mutation:false,human_promotion_required:true,redecide_min_minutes:55},evidence_class:EVOLUTION_EVIDENCE_CLASS,shadow_only:true,live_execution:false});if(q.error)console.error("promotion council receipt",q.error.message);
}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);const startedAt=new Date().toISOString();
  try{await requireCronAuth(req,db);}catch(error){return out({status:"UNAUTHORIZED",error:String(error),shadow_only:true,live_execution:false},401);}
  try{
    const lease=await withCollectorLease(db,COLLECTOR_ID,LEASE_SECONDS,async()=>{
      const observedAt=new Date().toISOString(),nowMs=Date.parse(observedAt);
      const q=await db.from("brian_evolution_experiments").select("experiment_id,hypothesis_id,created_at_source,mode,stage").eq("mode","PROSPECTIVE_SHADOW").in("stage",["EXPERIMENTAL","SHADOW_CANDIDATE"]).order("created_at_source",{ascending:false}).limit(40);
      if(q.error)throw new Error(`experiments:${q.error.message}`);
      const results:unknown[]=[];let skippedRecent=0,stored=0;
      for(const raw of q.data??[]){const experiment=raw as Record<string,unknown>;if(await recentlyDecided(String(experiment.experiment_id),nowMs)){skippedRecent++;continue;}const result=await decideExperiment(experiment,observedAt);results.push(result);stored+=Number((result as {stored?:number}).stored??0)+Number((result as {drift?:number}).drift??0);}
      await receipt(startedAt,"SUCCESS",q.data?.length??0,stored);
      return{status:"SUCCESS",collector_id:COLLECTOR_ID,experiments_considered:q.data?.length??0,skipped_recent:skippedRecent,results,canonical_mutation:false,autonomous_apply_allowed:false,human_promotion_required:true,redecide_min_minutes:55,shadow_only:true,live_execution:false};
    });
    if(lease.contended){await receipt(startedAt,"SKIPPED",0,0);return out({status:"SKIPPED_LEASE_CONTENDED",shadow_only:true,live_execution:false});}
    return out(lease.value);
  }catch(error){await receipt(startedAt,"FAILED",0,0,error);return out({status:"FAILED",error:String(error),canonical_mutation:false,autonomous_apply_allowed:false,shadow_only:true,live_execution:false},500);}
});