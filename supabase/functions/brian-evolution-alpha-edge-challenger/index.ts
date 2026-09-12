import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import { EVOLUTION_EVIDENCE_CLASS } from "../_shared/evolution_contract.ts";
import {
  estimateExpectedNetEdge,
  EVOLUTION_ALPHA_INTELLIGENCE_VERSION,
  type EvidenceFreshness,
  type LaggedReliabilityEvidence,
} from "../_shared/evolution_alpha_intelligence.ts";
import {
  bindLaggedReliabilityToDecision,
  type DecisionSourceObservation,
  type ReliabilitySnapshotCandidate,
} from "../_shared/evolution_alpha_reliability_mapping.ts";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const COLLECTOR_ID="brian-evolution-alpha-edge-challenger-v1";
const LEASE_SECONDS=55;
const MIN_INTERVAL_SECONDS=45;
const DECISION_LOOKBACK_MS=6*60*60_000;
const MAX_DECISIONS=160;
const EDGE_HORIZON_SECONDS=900;

type DecisionRow={
  decision_id:string;observed_at:string;asset_id:string;action:string;direction:number;evidence_score:number;
  support_groups:string[]|null;source_observation_ids:string[]|null;estimated_round_trip_cost_bps:number|string|null;
};
type ReliabilityWindow={window_end:string;generated_at:string};

function out(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}});}
async function sha(value:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(value)));return[...d].map(b=>b.toString(16).padStart(2,"0")).join("");}
function finite(v:unknown):number|null{if(v==null||v==="")return null;const n=Number(v);return Number.isFinite(n)?n:null;}
function errorText(error:unknown){return error instanceof Error?`${error.name}: ${error.message}`:String(error);}

async function loadDecisions(){
  const since=new Date(Date.now()-DECISION_LOOKBACK_MS).toISOString();
  const q=await db.from("brian_alpha_decisions")
    .select("decision_id,observed_at,asset_id,action,direction,evidence_score,support_groups,source_observation_ids,estimated_round_trip_cost_bps")
    .in("action",["OPEN_LONG","OPEN_SHORT"]).gte("observed_at",since).order("observed_at",{ascending:false}).limit(MAX_DECISIONS);
  if(q.error)throw new Error(`decisions:${q.error.message}`);
  const rows=(q.data??[]) as DecisionRow[];
  if(!rows.length)return[];
  const ids=rows.map(row=>String(row.decision_id));
  const existing=await db.from("brian_alpha_expected_edge_challenger").select("decision_id").in("decision_id",ids).limit(MAX_DECISIONS*2);
  if(existing.error)throw new Error(`existing_edges:${existing.error.message}`);
  const seen=new Set((existing.data??[]).map(row=>String(row.decision_id)));
  return rows.filter(row=>!seen.has(String(row.decision_id))).sort((a,b)=>Date.parse(a.observed_at)-Date.parse(b.observed_at));
}

async function reliabilityWindowAsOf(decisionAt:string):Promise<ReliabilityWindow|null>{
  const q=await db.from("brian_sensor_reliability_shadow_snapshots")
    .select("window_end,generated_at")
    .eq("outcome_horizon_seconds",EDGE_HORIZON_SECONDS)
    .lte("window_end",decisionAt).lte("generated_at",decisionAt)
    .order("window_end",{ascending:false}).order("generated_at",{ascending:false}).limit(1).maybeSingle();
  if(q.error)throw new Error(`reliability_window:${q.error.message}`);
  if(!q.data?.window_end||!q.data?.generated_at)return null;
  return{window_end:String(q.data.window_end),generated_at:String(q.data.generated_at)};
}

async function sourceObservationsForDecision(decision:DecisionRow):Promise<DecisionSourceObservation[]>{
  const ids=[...new Set((decision.source_observation_ids??[]).map(String).filter(id=>id&&!id.startsWith("phase37:")&&!id.startsWith("dip:")))];
  if(!ids.length)return[];
  const q=await db.from("brian_sensor_observations")
    .select("observation_id,independent_group,sensor_family,horizon,direction,observed_at")
    .in("observation_id",ids.slice(0,100)).limit(200);
  if(q.error)throw new Error(`source_observations:${q.error.message}`);
  return(q.data??[]).flatMap(row=>{
    const direction=Number(row.direction);
    if(direction!==1&&direction!==-1)return[];
    return[{
      observationId:String(row.observation_id),independentGroup:String(row.independent_group),sensorFamily:String(row.sensor_family),
      sensorHorizon:String(row.horizon),direction:direction as -1|1,observedAt:String(row.observed_at),
    } satisfies DecisionSourceObservation];
  });
}

async function reliabilityForDecision(decision:DecisionRow,window:ReliabilityWindow|null,sources:DecisionSourceObservation[]):Promise<LaggedReliabilityEvidence[]>{
  const rawGroups=[...new Set(sources.map(row=>row.independentGroup).filter(Boolean))];
  if(!window||!rawGroups.length)return[];
  const q=await db.from("brian_sensor_reliability_shadow_snapshots")
    .select("independent_group,sensor_family,sensor_horizon,outcome_horizon_seconds,sample_count,bayesian_hit_rate_beta10_10,avg_cost_adjusted_signed_bps,avg_signed_bps,window_end,generated_at")
    .eq("window_end",window.window_end).eq("generated_at",window.generated_at).eq("outcome_horizon_seconds",EDGE_HORIZON_SECONDS)
    .in("independent_group",rawGroups).order("sample_count",{ascending:false}).limit(1000);
  if(q.error)throw new Error(`reliability_rows:${q.error.message}`);
  const snapshots:ReliabilitySnapshotCandidate[]=(q.data??[]).flatMap(row=>{
    const sampleCount=finite(row.sample_count),bayesianHitRate=finite(row.bayesian_hit_rate_beta10_10),avgSignedBps=finite(row.avg_signed_bps),avgCostAdjustedSignedBps=finite(row.avg_cost_adjusted_signed_bps);
    if(sampleCount==null||sampleCount<0||bayesianHitRate==null||bayesianHitRate<0||bayesianHitRate>1||avgSignedBps==null||avgCostAdjustedSignedBps==null)return[];
    return[{
      independentGroup:String(row.independent_group),sensorFamily:String(row.sensor_family),sensorHorizon:String(row.sensor_horizon),
      sampleCount:Math.max(0,Math.trunc(sampleCount)),bayesianHitRate,avgSignedBps,avgCostAdjustedSignedBps,
      outcomeHorizonSeconds:EDGE_HORIZON_SECONDS,snapshotWindowEnd:String(row.window_end),snapshotGeneratedAt:String(row.generated_at),
    } satisfies ReliabilitySnapshotCandidate];
  });
  const rawDirection=Number(decision.direction);
  if(rawDirection!==1&&rawDirection!==-1)throw new Error(`invalid canonical direction for ${decision.decision_id}`);
  return bindLaggedReliabilityToDecision({
    direction:rawDirection as -1|1,
    supportGroups:(decision.support_groups??[]).map(String),
    sourceObservations:sources,
    snapshotCandidates:snapshots,
  });
}

function freshnessForSources(sources:DecisionSourceObservation[]):EvidenceFreshness[]{
  return sources.map(row=>({group:row.independentGroup,observedAt:row.observedAt,horizon:row.sensorHorizon}));
}

async function persistDecision(decision:DecisionRow){
  const window=await reliabilityWindowAsOf(String(decision.observed_at));
  const sources=await sourceObservationsForDecision(decision);
  const reliability=await reliabilityForDecision(decision,window,sources);
  const freshness=freshnessForSources(sources);
  const rawDirection=Number(decision.direction);
  if(rawDirection!==1&&rawDirection!==-1)throw new Error(`invalid canonical direction for ${decision.decision_id}`);
  const direction=rawDirection as -1|1;
  const decomposition=estimateExpectedNetEdge({
    decisionObservedAt:String(decision.observed_at),direction,evidenceScore:Number(decision.evidence_score??0),
    roundTripCostBps:finite(decision.estimated_round_trip_cost_bps),reliability,freshness,minimumNetMarginBps:2,
  });
  const evaluatedAt=new Date().toISOString();
  const edgeId=await sha(`${EVOLUTION_ALPHA_INTELLIGENCE_VERSION}|${decision.decision_id}|${window?.window_end??"none"}|${window?.generated_at??"none"}|${EDGE_HORIZON_SECONDS}`);
  const row={
    edge_id:edgeId,decision_id:String(decision.decision_id),observed_at:String(decision.observed_at),evaluated_at:evaluatedAt,
    asset_id:String(decision.asset_id),canonical_action:String(decision.action),direction,canonical_evidence_score:Number(decision.evidence_score??0),
    support_groups:(decision.support_groups??[]).map(String),source_observation_ids:(decision.source_observation_ids??[]).map(String),
    reliability_window_end:window?.window_end??null,reliability_generated_at:window?.generated_at??null,
    expected_gross_move_bps:decomposition.expectedGrossMoveBps,estimated_round_trip_cost_bps:decomposition.estimatedRoundTripCostBps,
    uncertainty_penalty_bps:decomposition.uncertaintyPenaltyBps,event_decay_penalty_bps:decomposition.eventDecayPenaltyBps,
    expected_net_edge_bps:decomposition.expectedNetEdgeBps,minimum_net_margin_bps:decomposition.minimumNetMarginBps,
    recommendation:decomposition.recommendation,eligible:decomposition.eligible,mature_group_count:decomposition.matureGroupCount,
    group_contributions:decomposition.groupContributions,reliability_weights:decomposition.reliabilityWeights,pit_clear:decomposition.pitClear,
    reasons:decomposition.reasons,model_version:decomposition.version,
    metadata:{
      role:"SHADOW_CHALLENGER_ONLY",canonical_mutation:false,decision_time_reliability_only:true,decision_time_cost_only:true,
      source_freshness_rows:freshness.length,reliability_rows:reliability.length,newest_first_candidate_scan:true,
      reliability_horizon_seconds:EDGE_HORIZON_SECONDS,null_cost_fails_closed:true,
      reliability_lineage:"exact_source_observation_x_raw_group_x_sensor_family_x_horizon",
      compiler_group_canonicalization:true,avg_signed_bps_semantics:"sensor_direction_aligned_do_not_flip_again",
    },
    evidence_class:EVOLUTION_EVIDENCE_CLASS,shadow_only:true,live_execution:false,canonical_mutation:false,
  };
  const q=await db.from("brian_alpha_expected_edge_challenger").upsert(row,{onConflict:"decision_id",ignoreDuplicates:true});
  if(q.error)throw new Error(`persist_edge:${q.error.message}`);
  return{decision_id:decision.decision_id,recommendation:decomposition.recommendation,eligible:decomposition.eligible,expected_net_edge_bps:decomposition.expectedNetEdgeBps,pit_clear:decomposition.pitClear,mature_groups:decomposition.matureGroupCount};
}

async function recordRun(startedAt:string,status:"SUCCESS"|"FAILED"|"SKIPPED",observed:number,stored:number,metadata:Record<string,unknown>={},error?:unknown){
  const finishedAt=new Date().toISOString(),runId=await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  const q=await db.from("brian_collector_runs").insert({run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:finishedAt,status,observed_records:observed,stored_records:stored,degraded_sources:[],error_class:error?"EVOLUTION_ALPHA_EDGE_ERROR":null,error_message:error?errorText(error).slice(0,1200):null,metadata:{model_version:EVOLUTION_ALPHA_INTELLIGENCE_VERSION,canonical_mutation:false,reliability_horizon_seconds:EDGE_HORIZON_SECONDS,null_cost_fails_closed:true,exact_decision_lineage:true,...metadata},evidence_class:EVOLUTION_EVIDENCE_CLASS,shadow_only:true,live_execution:false});
  if(q.error)console.error("alpha edge run receipt",q.error.message);
}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);const startedAt=new Date().toISOString();
  try{await requireCronAuth(req,db);}catch(error){return out({status:"UNAUTHORIZED",error:errorText(error),shadow_only:true,live_execution:false},401);}
  try{
    const last=await db.from("brian_collector_runs").select("started_at").eq("collector_id",COLLECTOR_ID).in("status",["SUCCESS","DEGRADED"]).order("started_at",{ascending:false}).limit(1).maybeSingle();
    if(last.error)throw last.error;if(last.data?.started_at){const age=(Date.now()-Date.parse(String(last.data.started_at)))/1000;if(Number.isFinite(age)&&age<MIN_INTERVAL_SECONDS)return out({status:"SKIPPED_RATE_GUARD",age_seconds:age,shadow_only:true,live_execution:false});}
    const lease=await withCollectorLease(db,COLLECTOR_ID,LEASE_SECONDS,async()=>{
      const decisions=await loadDecisions(),results:unknown[]=[];
      for(const decision of decisions)results.push(await persistDecision(decision));
      const allow=results.filter(r=>(r as {recommendation?:string}).recommendation==="ALLOW_EDGE").length;
      const downgrade=results.filter(r=>(r as {recommendation?:string}).recommendation==="DOWNGRADE_TO_WAIT").length;
      const failClosed=results.length-allow-downgrade;
      await recordRun(startedAt,"SUCCESS",decisions.length,results.length,{allow_edge:allow,downgrade_to_wait:downgrade,fail_closed:failClosed,newest_first_candidate_scan:true,exact_decision_lineage:true});
      return{status:"SUCCESS",collector_id:COLLECTOR_ID,model_version:EVOLUTION_ALPHA_INTELLIGENCE_VERSION,evaluated:decisions.length,stored:results.length,allow_edge:allow,downgrade_to_wait:downgrade,fail_closed:failClosed,results,newest_first_candidate_scan:true,reliability_horizon_seconds:EDGE_HORIZON_SECONDS,null_cost_fails_closed:true,exact_decision_lineage:true,canonical_mutation:false,direct_alpha_influence:false,shadow_only:true,live_execution:false};
    });
    if(lease.contended){await recordRun(startedAt,"SKIPPED",0,0);return out({status:"SKIPPED_LEASE_CONTENDED",shadow_only:true,live_execution:false});}
    return out(lease.value);
  }catch(error){await recordRun(startedAt,"FAILED",0,0,{},error);return out({status:"FAILED",error:errorText(error),canonical_mutation:false,shadow_only:true,live_execution:false},500);}
});
