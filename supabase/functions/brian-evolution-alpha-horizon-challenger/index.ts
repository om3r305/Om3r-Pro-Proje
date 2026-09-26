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
const COLLECTOR_ID="brian-evolution-alpha-horizon-challenger-v1";
const VERSION="brian.phase127-horizon-challenger.v1";
const LEASE_SECONDS=70;
const MIN_INTERVAL_SECONDS=120;
const DECISION_LOOKBACK_MS=18*60*60_000;
const MAX_SCAN=160;
const MAX_DECISIONS_PER_RUN=12;
const HORIZONS=[900,3600] as const;

type DecisionRow={
  decision_id:string;observed_at:string;asset_id:string;action:string;direction:number;evidence_score:number;
  support_groups:string[]|null;source_observation_ids:string[]|null;estimated_round_trip_cost_bps:number|string|null;
};
type ReliabilityWindow={window_end:string;generated_at:string};
type HorizonResult={decision_id:string;asset_id:string;horizon:number;recommendation:string;eligible:boolean;expected_net_edge_bps:number|null;mature_groups:number;pit_clear:boolean};

function out(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}});}
async function sha(value:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(value)));return[...d].map(b=>b.toString(16).padStart(2,"0")).join("");}
function finite(v:unknown):number|null{if(v==null||v==="")return null;const n=Number(v);return Number.isFinite(n)?n:null;}
function errorText(error:unknown){return error instanceof Error?`${error.name}: ${error.message}`:String(error);}

async function loadPendingDecisions():Promise<DecisionRow[]>{
  const since=new Date(Date.now()-DECISION_LOOKBACK_MS).toISOString();
  const q=await db.from("brian_alpha_decisions")
    .select("decision_id,observed_at,asset_id,action,direction,evidence_score,support_groups,source_observation_ids,estimated_round_trip_cost_bps")
    .in("action",["OPEN_LONG","OPEN_SHORT"]).gte("observed_at",since).order("observed_at",{ascending:false}).limit(MAX_SCAN);
  if(q.error)throw new Error(`decisions:${q.error.message}`);
  const rows=(q.data??[]) as DecisionRow[];
  if(!rows.length)return[];
  const ids=rows.map(row=>String(row.decision_id));
  const existing=await db.from("brian_alpha_horizon_challenger")
    .select("decision_id,outcome_horizon_seconds").in("decision_id",ids).limit(MAX_SCAN*2+20);
  if(existing.error)throw new Error(`existing_horizons:${existing.error.message}`);
  const seen=new Set((existing.data??[]).map(row=>`${String(row.decision_id)}|${Number(row.outcome_horizon_seconds)}`));
  return rows.filter(row=>HORIZONS.some(h=>!seen.has(`${row.decision_id}|${h}`)))
    .sort((a,b)=>Date.parse(a.observed_at)-Date.parse(b.observed_at))
    .slice(0,MAX_DECISIONS_PER_RUN);
}

async function reliabilityWindowAsOf(decisionAt:string,horizon:number):Promise<ReliabilityWindow|null>{
  const q=await db.from("brian_sensor_reliability_shadow_snapshots")
    .select("window_end,generated_at").eq("outcome_horizon_seconds",horizon)
    .lte("window_end",decisionAt).lte("generated_at",decisionAt)
    .order("window_end",{ascending:false}).order("generated_at",{ascending:false}).limit(1).maybeSingle();
  if(q.error)throw new Error(`reliability_window_${horizon}:${q.error.message}`);
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
    return[{observationId:String(row.observation_id),independentGroup:String(row.independent_group),sensorFamily:String(row.sensor_family),sensorHorizon:String(row.horizon),direction:direction as -1|1,observedAt:String(row.observed_at)} satisfies DecisionSourceObservation];
  });
}

async function reliabilityForDecision(decision:DecisionRow,window:ReliabilityWindow|null,sources:DecisionSourceObservation[],horizon:number):Promise<LaggedReliabilityEvidence[]>{
  const rawGroups=[...new Set(sources.map(row=>row.independentGroup).filter(Boolean))];
  if(!window||!rawGroups.length)return[];
  const q=await db.from("brian_sensor_reliability_shadow_snapshots")
    .select("independent_group,sensor_family,sensor_horizon,outcome_horizon_seconds,sample_count,bayesian_hit_rate_beta10_10,avg_cost_adjusted_signed_bps,avg_signed_bps,window_end,generated_at")
    .eq("window_end",window.window_end).eq("generated_at",window.generated_at).eq("outcome_horizon_seconds",horizon)
    .in("independent_group",rawGroups).order("sample_count",{ascending:false}).limit(1000);
  if(q.error)throw new Error(`reliability_rows_${horizon}:${q.error.message}`);
  const snapshots:ReliabilitySnapshotCandidate[]=(q.data??[]).flatMap(row=>{
    const sampleCount=finite(row.sample_count),bayesianHitRate=finite(row.bayesian_hit_rate_beta10_10),avgSignedBps=finite(row.avg_signed_bps),avgCostAdjustedSignedBps=finite(row.avg_cost_adjusted_signed_bps);
    if(sampleCount==null||sampleCount<0||bayesianHitRate==null||bayesianHitRate<0||bayesianHitRate>1||avgSignedBps==null||avgCostAdjustedSignedBps==null)return[];
    return[{independentGroup:String(row.independent_group),sensorFamily:String(row.sensor_family),sensorHorizon:String(row.sensor_horizon),sampleCount:Math.max(0,Math.trunc(sampleCount)),bayesianHitRate,avgSignedBps,avgCostAdjustedSignedBps,outcomeHorizonSeconds:horizon,snapshotWindowEnd:String(row.window_end),snapshotGeneratedAt:String(row.generated_at)} satisfies ReliabilitySnapshotCandidate];
  });
  const rawDirection=Number(decision.direction);
  if(rawDirection!==1&&rawDirection!==-1)throw new Error(`invalid canonical direction for ${decision.decision_id}`);
  return bindLaggedReliabilityToDecision({direction:rawDirection as -1|1,supportGroups:(decision.support_groups??[]).map(String),sourceObservations:sources,snapshotCandidates:snapshots});
}

function freshnessForSources(sources:DecisionSourceObservation[]):EvidenceFreshness[]{
  return sources.map(row=>({group:row.independentGroup,observedAt:row.observedAt,horizon:row.sensorHorizon}));
}

async function persistHorizon(decision:DecisionRow,sources:DecisionSourceObservation[],horizon:number):Promise<HorizonResult>{
  const existing=await db.from("brian_alpha_horizon_challenger").select("comparison_id")
    .eq("decision_id",decision.decision_id).eq("outcome_horizon_seconds",horizon).limit(1).maybeSingle();
  if(existing.error)throw new Error(`existing_horizon_${horizon}:${existing.error.message}`);
  if(existing.data?.comparison_id){
    return{decision_id:decision.decision_id,asset_id:decision.asset_id,horizon,recommendation:"ALREADY_RECORDED",eligible:false,expected_net_edge_bps:null,mature_groups:0,pit_clear:true};
  }
  const window=await reliabilityWindowAsOf(String(decision.observed_at),horizon);
  const reliability=await reliabilityForDecision(decision,window,sources,horizon);
  const freshness=freshnessForSources(sources);
  const rawDirection=Number(decision.direction);
  if(rawDirection!==1&&rawDirection!==-1)throw new Error(`invalid canonical direction for ${decision.decision_id}`);
  const decomposition=estimateExpectedNetEdge({
    decisionObservedAt:String(decision.observed_at),direction:rawDirection as -1|1,evidenceScore:Number(decision.evidence_score??0),
    roundTripCostBps:finite(decision.estimated_round_trip_cost_bps),reliability,freshness,minimumNetMarginBps:2,
  });
  const evaluatedAt=new Date().toISOString();
  const comparisonId=await sha(`${VERSION}|${decision.decision_id}|${horizon}|${window?.window_end??"none"}|${window?.generated_at??"none"}`);
  const row={
    comparison_id:comparisonId,decision_id:String(decision.decision_id),outcome_horizon_seconds:horizon,
    observed_at:String(decision.observed_at),evaluated_at:evaluatedAt,asset_id:String(decision.asset_id),
    canonical_action:String(decision.action),direction:rawDirection,canonical_evidence_score:Number(decision.evidence_score??0),
    support_groups:(decision.support_groups??[]).map(String),source_observation_ids:(decision.source_observation_ids??[]).map(String),
    reliability_window_end:window?.window_end??null,reliability_generated_at:window?.generated_at??null,
    expected_gross_move_bps:decomposition.expectedGrossMoveBps,estimated_round_trip_cost_bps:decomposition.estimatedRoundTripCostBps,
    uncertainty_penalty_bps:decomposition.uncertaintyPenaltyBps,event_decay_penalty_bps:decomposition.eventDecayPenaltyBps,
    expected_net_edge_bps:decomposition.expectedNetEdgeBps,minimum_net_margin_bps:decomposition.minimumNetMarginBps,
    recommendation:decomposition.recommendation,eligible:decomposition.eligible,mature_group_count:decomposition.matureGroupCount,
    group_contributions:decomposition.groupContributions,reliability_weights:decomposition.reliabilityWeights,pit_clear:decomposition.pitClear,
    reasons:decomposition.reasons,model_version:decomposition.version,
    metadata:{phase:"PHASE127",role:horizon===900?"REFERENCE_15M":"CHALLENGER_60M",canonical_mutation:false,automatic_promotion:false,direct_alpha_influence:false,decision_time_reliability_only:true,decision_time_cost_only:true,reliability_horizon_seconds:horizon,reliability_rows:reliability.length,source_freshness_rows:freshness.length,comparison_version:VERSION},
    evidence_class:EVOLUTION_EVIDENCE_CLASS,shadow_only:true,live_execution:false,automatic_promotion:false,canonical_mutation:false,
  };
  const ins=await db.from("brian_alpha_horizon_challenger").insert(row);
  if(ins.error&&!String(ins.error.code??"").includes("23505"))throw new Error(`persist_horizon_${horizon}:${ins.error.message}`);
  return{decision_id:decision.decision_id,asset_id:decision.asset_id,horizon,recommendation:decomposition.recommendation,eligible:decomposition.eligible,expected_net_edge_bps:decomposition.expectedNetEdgeBps,mature_groups:decomposition.matureGroupCount,pit_clear:decomposition.pitClear};
}

async function recordRun(startedAt:string,status:"SUCCESS"|"FAILED"|"SKIPPED",observed:number,stored:number,metadata:Record<string,unknown>={},error?:unknown){
  const finishedAt=new Date().toISOString(),runId=await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  const q=await db.from("brian_collector_runs").insert({run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:finishedAt,status,observed_records:observed,stored_records:stored,degraded_sources:[],error_class:error?"PHASE127_HORIZON_ERROR":null,error_message:error?errorText(error).slice(0,1200):null,metadata:{comparison_version:VERSION,model_version:EVOLUTION_ALPHA_INTELLIGENCE_VERSION,horizons:[...HORIZONS],reference_horizon_seconds:900,challenger_horizon_seconds:3600,canonical_mutation:false,automatic_promotion:false,direct_alpha_influence:false,...metadata},evidence_class:EVOLUTION_EVIDENCE_CLASS,shadow_only:true,live_execution:false});
  if(q.error)console.error("phase127 run receipt",q.error.message);
}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);
  const startedAt=new Date().toISOString();
  try{await requireCronAuth(req,db);}catch(error){return out({status:"UNAUTHORIZED",error:errorText(error),shadow_only:true,live_execution:false},401);}
  try{
    const last=await db.from("brian_collector_runs").select("started_at").eq("collector_id",COLLECTOR_ID).in("status",["SUCCESS","DEGRADED"]).order("started_at",{ascending:false}).limit(1).maybeSingle();
    if(last.error)throw last.error;
    if(last.data?.started_at){const age=(Date.now()-Date.parse(String(last.data.started_at)))/1000;if(Number.isFinite(age)&&age<MIN_INTERVAL_SECONDS)return out({status:"SKIPPED_RATE_GUARD",age_seconds:age,shadow_only:true,live_execution:false});}
    const lease=await withCollectorLease(db,COLLECTOR_ID,LEASE_SECONDS,async()=>{
      const decisions=await loadPendingDecisions();
      const results:HorizonResult[]=[];
      for(const decision of decisions){
        const sources=await sourceObservationsForDecision(decision);
        for(const horizon of HORIZONS)results.push(await persistHorizon(decision,sources,horizon));
      }
      const fresh=results.filter(r=>r.recommendation!=="ALREADY_RECORDED");
      const ref=fresh.filter(r=>r.horizon===900),challenge=fresh.filter(r=>r.horizon===3600);
      const summary=(rows:HorizonResult[])=>({evaluated:rows.length,allow_edge:rows.filter(r=>r.recommendation==="ALLOW_EDGE").length,insufficient:rows.filter(r=>r.recommendation==="INSUFFICIENT_LAGGED_EVIDENCE").length,downgrade:rows.filter(r=>r.recommendation==="DOWNGRADE_TO_WAIT").length,other_fail_closed:rows.filter(r=>!["ALLOW_EDGE","INSUFFICIENT_LAGGED_EVIDENCE","DOWNGRADE_TO_WAIT"].includes(r.recommendation)).length});
      await recordRun(startedAt,"SUCCESS",decisions.length,fresh.length,{reference_900:summary(ref),challenger_3600:summary(challenge)});
      return{status:"SUCCESS",collector_id:COLLECTOR_ID,comparison_version:VERSION,decisions:decisions.length,stored:fresh.length,reference_900:summary(ref),challenger_3600:summary(challenge),results:fresh,canonical_mutation:false,automatic_promotion:false,direct_alpha_influence:false,shadow_only:true,live_execution:false};
    });
    if(lease.contended){await recordRun(startedAt,"SKIPPED",0,0,{reason:"LEASE_CONTENDED"});return out({status:"SKIPPED_LEASE_CONTENDED",shadow_only:true,live_execution:false});}
    return out(lease.value);
  }catch(error){await recordRun(startedAt,"FAILED",0,0,{},error);return out({status:"FAILED",error:errorText(error),canonical_mutation:false,automatic_promotion:false,shadow_only:true,live_execution:false},500);}
});
