import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import { EVOLUTION_EVIDENCE_CLASS } from "../_shared/evolution_contract.ts";
import { BRIAN_OCEAN_VERSION, buildOceanReport, deriveOceanRuns, type OceanCommand, type OceanRunState, type OceanTreasuryPoint } from "../_shared/evolution_ocean.ts";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const COLLECTOR_ID="brian-evolution-ocean-worker-v1";
const LEASE_SECONDS=240;
const CHECKPOINT_MIN_AGE_MS=4*60_000;

function out(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}});}
async function sha(value:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(value)));return[...d].map(b=>b.toString(16).padStart(2,"0")).join("");}
function finite(value:unknown):number|null{if(value==null||value==="")return null;const n=Number(value);return Number.isFinite(n)?n:null;}
function errorText(error:unknown){return error instanceof Error?`${error.name}: ${error.message}`:String(error);}

async function commands():Promise<OceanCommand[]>{
  const q=await db.from("brian_ocean_run_commands").select("command_id,run_id,command,requested_at,duration_hours,reason").order("requested_at",{ascending:false}).limit(500);
  if(q.error)throw new Error(`ocean_commands:${q.error.message}`);
  return(q.data??[]).map(row=>({commandId:String(row.command_id),runId:String(row.run_id),command:String(row.command) as OceanCommand["command"],requestedAt:String(row.requested_at),durationHours:row.duration_hours==null?null:Number(row.duration_hours) as 24|48,reason:row.reason==null?null:String(row.reason)}));
}

function treasuryPoint(row:Record<string,unknown>|null):OceanTreasuryPoint|null{
  if(!row)return null;const equity=finite(row.equity_usd),cash=finite(row.cash_usd),deployment=finite(row.deployment_usd),realized=finite(row.realized_pnl_usd),costs=finite(row.cumulative_costs_usd);
  if(equity==null||cash==null||deployment==null||realized==null||costs==null)return null;
  const positions=Array.isArray(row.positions)?row.positions.length:0;
  return{observedAt:String(row.observed_at),equityUsd:equity,cashUsd:cash,deploymentUsd:deployment,realizedPnlUsd:realized,cumulativeCostsUsd:costs,openPositions:positions};
}

async function treasuryAtOrBefore(at:string){
  const q=await db.from("brian_treasury_shadow_snapshots").select("snapshot_id,observed_at,equity_usd,cash_usd,deployment_usd,realized_pnl_usd,cumulative_costs_usd,positions").lte("observed_at",at).order("observed_at",{ascending:false}).limit(1).maybeSingle();
  if(q.error)throw new Error(`treasury_point:${q.error.message}`);return q.data as Record<string,unknown>|null;
}

async function countRows(table:string,timeColumn:string,start:string,end:string,extra?:(query:any)=>any):Promise<number>{
  let q:any=db.from(table).select("*",{count:"exact",head:true}).gte(timeColumn,start).lte(timeColumn,end);if(extra)q=extra(q);const result=await q;if(result.error)throw new Error(`${table}:${result.error.message}`);return Number(result.count??0);
}

async function uniqueCount(table:string,idColumn:string,timeColumn:string,start:string,end:string):Promise<number>{
  const q=await db.from(table).select(idColumn).gte(timeColumn,start).lte(timeColumn,end).limit(10000);if(q.error)throw new Error(`${table}:${q.error.message}`);return new Set((q.data??[]).map((row:any)=>String(row[idColumn]))).size;
}

async function latestCheckpointAge(runId:string,nowMs:number){
  const q=await db.from("brian_ocean_run_checkpoints").select("observed_at").eq("run_id",runId).order("observed_at",{ascending:false}).limit(1).maybeSingle();if(q.error)throw new Error(`ocean_checkpoint_age:${q.error.message}`);if(!q.data?.observed_at)return Infinity;const t=Date.parse(String(q.data.observed_at));return Number.isFinite(t)?nowMs-t:Infinity;
}

async function writeCheckpoint(run:OceanRunState,observedAt:string){
  const since=new Date(Date.parse(observedAt)-15*60_000).toISOString();
  const [treasuryQ,runsQ]=await Promise.all([
    db.from("brian_treasury_shadow_latest").select("snapshot_id,observed_at,equity_usd,cash_usd,deployment_usd,positions,promotion_gate_open,promotion_gate_reason").limit(1).maybeSingle(),
    db.from("brian_collector_runs").select("status").gte("started_at",since).lte("started_at",observedAt).limit(5000),
  ]);
  if(treasuryQ.error)throw new Error(`checkpoint_treasury:${treasuryQ.error.message}`);if(runsQ.error)throw new Error(`checkpoint_collectors:${runsQ.error.message}`);
  const rows=runsQ.data??[],failures=rows.filter(row=>String(row.status)==="FAILED").length,degraded=rows.filter(row=>String(row.status)==="DEGRADED").length;
  const treasury=treasuryQ.data as Record<string,unknown>|null,positions=treasury&&Array.isArray(treasury.positions)?treasury.positions.length:null;
  const checkpointId=await sha(`ocean-checkpoint|${run.runId}|${observedAt}`);
  const q=await db.from("brian_ocean_run_checkpoints").insert({checkpoint_id:checkpointId,run_id:run.runId,observed_at:observedAt,treasury_snapshot_id:treasury?.snapshot_id??null,treasury_equity_usd:treasury?.equity_usd??null,treasury_cash_usd:treasury?.cash_usd??null,treasury_deployment_usd:treasury?.deployment_usd??null,treasury_open_positions:positions,collector_runs_window:rows.length,collector_failures_window:failures,collector_degraded_window:degraded,payload:{ocean_version:BRIAN_OCEAN_VERSION,promotion_gate_open:treasury?.promotion_gate_open??false,promotion_gate_reason:treasury?.promotion_gate_reason??null},evidence_class:EVOLUTION_EVIDENCE_CLASS,shadow_only:true,live_execution:false});
  if(q.error)throw new Error(`ocean_checkpoint:${q.error.message}`);return checkpointId;
}

async function reportExists(runId:string){const q=await db.from("brian_ocean_run_reports").select("report_id").eq("run_id",runId).limit(1).maybeSingle();if(q.error)throw new Error(`ocean_report_exists:${q.error.message}`);return Boolean(q.data);}

async function buildAndPersistReport(run:OceanRunState){
  const start=run.startedAt,end=run.effectiveEndAt;
  const [startTreasuryRow,endTreasuryRow,treasuryActions,replacements,newSources,newHypotheses,newCodeCandidates,experimentResults,promotionCandidates,rejectedPromotions,driftEvents,capabilityEvents,missedOpportunities,alphaOutcomeSamples,alphaFavorableAfterCost,collectorRuns,collectorFailures,degradedRuns]=await Promise.all([
    treasuryAtOrBefore(start),treasuryAtOrBefore(end),
    countRows("brian_treasury_shadow_actions","observed_at",start,end),
    countRows("brian_treasury_shadow_actions","observed_at",start,end,q=>q.eq("reason","OPPORTUNITY_REPLACEMENT")),
    uniqueCount("brian_world_source_candidates","source_id","discovered_at",start,end),
    uniqueCount("brian_evolution_hypothesis_snapshots","hypothesis_id","observed_at",start,end),
    uniqueCount("brian_evolution_code_candidates","candidate_id","proposed_at",start,end),
    countRows("brian_evolution_experiment_results","measured_at",start,end),
    countRows("brian_evolution_promotion_decisions","decided_at",start,end,q=>q.eq("decision","PROMOTE_CANDIDATE")),
    countRows("brian_evolution_promotion_decisions","decided_at",start,end,q=>q.eq("decision","REJECT")),
    countRows("brian_evolution_drift_snapshots","observed_at",start,end,q=>q.in("severity",["MATERIAL","SEVERE"])),
    countRows("brian_evolution_events","occurred_at",start,end,q=>q.eq("entity_type","CAPABILITY")),
    countRows("brian_missed_opportunity_receipts","resolved_at",start,end),
    countRows("brian_alpha_decision_outcomes","resolved_at",start,end),
    countRows("brian_alpha_decision_outcomes","resolved_at",start,end,q=>q.eq("classification","ACTION_FAVORABLE_AFTER_COST")),
    countRows("brian_collector_runs","started_at",start,end),
    countRows("brian_collector_runs","started_at",start,end,q=>q.eq("status","FAILED")),
    countRows("brian_collector_runs","started_at",start,end,q=>q.eq("status","DEGRADED")),
  ]);
  const summary=buildOceanReport({run,treasuryStart:treasuryPoint(startTreasuryRow),treasuryEnd:treasuryPoint(endTreasuryRow),treasuryActions,replacements,newSources,newHypotheses,newCodeCandidates,experimentResults,promotionCandidates,rejectedPromotions,driftEvents,capabilityEvents,missedOpportunities,alphaOutcomeSamples,alphaFavorableAfterCost,collectorRuns,collectorFailures,degradedRuns});
  const reportId=await sha(`ocean-report|${run.runId}|${run.effectiveEndAt}|${BRIAN_OCEAN_VERSION}`);
  const q=await db.from("brian_ocean_run_reports").upsert({report_id:reportId,run_id:run.runId,started_at:run.startedAt,ended_at:run.effectiveEndAt,duration_hours:summary.durationHours,summary,evidence_refs:[`ocean:${run.runId}:commands`,`treasury:${run.startedAt}:${run.effectiveEndAt}`,`alpha-outcomes:${run.startedAt}:${run.effectiveEndAt}`,`collector-runs:${run.startedAt}:${run.effectiveEndAt}`],report_version:BRIAN_OCEAN_VERSION,metadata:{generated_at:new Date().toISOString(),planned_end_at:run.plannedEndAt,stopped_at:run.stoppedAt},evidence_class:EVOLUTION_EVIDENCE_CLASS,shadow_only:true,live_execution:false},{onConflict:"run_id",ignoreDuplicates:true});
  if(q.error)throw new Error(`ocean_report:${q.error.message}`);return{reportId,summary};
}

async function receipt(startedAt:string,status:"SUCCESS"|"FAILED"|"SKIPPED",observed:number,stored:number,metadata:Record<string,unknown>={},error?:unknown){const finishedAt=new Date().toISOString(),runId=await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);const q=await db.from("brian_collector_runs").insert({run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:finishedAt,status,observed_records:observed,stored_records:stored,degraded_sources:[],error_class:error?"EVOLUTION_OCEAN_ERROR":null,error_message:error?errorText(error).slice(0,1200):null,metadata:{ocean_version:BRIAN_OCEAN_VERSION,...metadata},evidence_class:EVOLUTION_EVIDENCE_CLASS,shadow_only:true,live_execution:false});if(q.error)console.error("ocean receipt",q.error.message);}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);const startedAt=new Date().toISOString();
  try{await requireCronAuth(req,db);}catch(error){return out({status:"UNAUTHORIZED",error:errorText(error),shadow_only:true,live_execution:false},401);}
  try{
    const lease=await withCollectorLease(db,COLLECTOR_ID,LEASE_SECONDS,async()=>{
      const observedAt=new Date().toISOString(),nowMs=Date.parse(observedAt),allCommands=await commands(),runs=deriveOceanRuns(allCommands,observedAt);let stored=0;const checkpoints:string[]=[],reports:unknown[]=[];
      for(const run of runs){
        if(run.status==="ACTIVE"&&await latestCheckpointAge(run.runId,nowMs)>=CHECKPOINT_MIN_AGE_MS){checkpoints.push(await writeCheckpoint(run,observedAt));stored++;}
        if(run.status==="ENDED"&&!(await reportExists(run.runId))){reports.push(await buildAndPersistReport(run));stored++;}
      }
      await receipt(startedAt,"SUCCESS",runs.length,stored,{active_runs:runs.filter(run=>run.status==="ACTIVE").length,checkpoints:checkpoints.length,reports:reports.length});
      return{status:"SUCCESS",collector_id:COLLECTOR_ID,observed_at:observedAt,runs_considered:runs.length,active:runs.filter(run=>run.status==="ACTIVE"),checkpoints_written:checkpoints,reports_written:reports,cloud_independent:true,shadow_only:true,live_execution:false};
    });
    if(lease.contended){await receipt(startedAt,"SKIPPED",0,0);return out({status:"SKIPPED_LEASE_CONTENDED",shadow_only:true,live_execution:false});}
    return out(lease.value);
  }catch(error){await receipt(startedAt,"FAILED",0,0,{},error);return out({status:"FAILED",error:errorText(error),shadow_only:true,live_execution:false},500);}
});
