import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import { EVOLUTION_EVIDENCE_CLASS } from "../_shared/evolution_contract.ts";
import { canGenerateBuiltIn, EVOLUTION_CODEGEN_VERSION, generateBuiltInArtifact } from "../_shared/evolution_codegen.ts";
import type { HypothesisCandidate } from "../_shared/evolution_research.ts";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const COLLECTOR_ID="brian-evolution-template-generator-v1";
const LEASE_SECONDS=180;

function out(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}});}
async function sha(value:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(value)));return[...d].map(b=>b.toString(16).padStart(2,"0")).join("");}
function metadata(row:Record<string,unknown>){return(row.metadata??{}) as Record<string,unknown>;}
function toHypothesis(row:Record<string,unknown>):HypothesisCandidate{const m=metadata(row);return{
  hypothesisId:String(row.hypothesis_id),observedAt:String(row.observed_at),problemStatement:String(row.problem_statement),proposedMechanism:String(row.proposed_mechanism),
  targetCapabilities:Array.isArray(row.target_capabilities)?row.target_capabilities.map(String):[],evidenceRefs:Array.isArray(row.evidence_refs)?row.evidence_refs.map(String):[],
  counterEvidenceRefs:Array.isArray(row.counter_evidence_refs)?row.counter_evidence_refs.map(String):[],measurableSuccessCriteria:Array.isArray(row.measurable_success_criteria)?row.measurable_success_criteria.map(String):[],
  stage:row.stage as HypothesisCandidate["stage"],uncertainty:Number(row.uncertainty??.5),priority:Number(m.priority??.5),hypothesisKind:String(m.hypothesis_kind??"CAPABILITY_GAP") as HypothesisCandidate["hypothesisKind"],metadata:m,
};}
async function loadHypothesis(hypothesisId:string){const q=await db.from("brian_evolution_hypothesis_snapshots").select("hypothesis_id,observed_at,problem_statement,proposed_mechanism,target_capabilities,evidence_refs,counter_evidence_refs,measurable_success_criteria,stage,uncertainty,metadata").eq("hypothesis_id",hypothesisId).order("observed_at",{ascending:false}).limit(1).maybeSingle();if(q.error)throw new Error(`hypothesis:${q.error.message}`);return q.data?toHypothesis(q.data as Record<string,unknown>):null;}
async function hasGenerated(candidateId:string){const q=await db.from("brian_evolution_code_artifact_receipts").select("receipt_id").eq("candidate_id",candidateId).eq("evidence_kind","GENERATED").limit(1).maybeSingle();if(q.error)throw new Error(`generated_receipt:${q.error.message}`);return Boolean(q.data);}

async function generatePending(){
  const q=await db.from("brian_evolution_codegen_requests").select("request_id,candidate_id,hypothesis_id,requested_at,parent_commit,branch_name,changed_paths,evidence_refs").order("requested_at",{ascending:false}).limit(100);
  if(q.error)throw new Error(`codegen_requests:${q.error.message}`);
  const results:Record<string,unknown>[]=[];let generated=0,unsupported=0,skippedExisting=0;
  for(const request of q.data??[]){
    const candidateId=String(request.candidate_id);if(await hasGenerated(candidateId)){skippedExisting++;continue;}
    const h=await loadHypothesis(String(request.hypothesis_id));if(!h||!canGenerateBuiltIn(h)){unsupported++;continue;}
    const artifact=generateBuiltInArtifact(h,String(request.parent_commit));
    const expectedPaths=Array.isArray(request.changed_paths)?request.changed_paths.map(String):[];
    if(artifact.brief.candidateId!==candidateId)throw new Error(`CANDIDATE_ID_MISMATCH:${candidateId}`);
    if(JSON.stringify(artifact.brief.changedPaths)!==JSON.stringify(expectedPaths))throw new Error(`CANDIDATE_PATH_MISMATCH:${candidateId}`);
    if(artifact.brief.branchName!==String(request.branch_name))throw new Error(`CANDIDATE_BRANCH_MISMATCH:${candidateId}`);
    const canonical=JSON.stringify(artifact.files);const artifactSha=await sha(canonical);const patchBytes=new TextEncoder().encode(canonical).length;
    if(patchBytes<=0||patchBytes>250000)throw new Error(`GENERATED_ARTIFACT_SIZE:${patchBytes}`);
    const observedAt=new Date().toISOString(),receiptId=await sha(`builtin-generated|${candidateId}|${artifactSha}`);
    const ins=await db.from("brian_evolution_code_artifact_receipts").upsert({
      receipt_id:receiptId,candidate_id:candidateId,hypothesis_id:String(request.hypothesis_id),evidence_kind:"GENERATED",observed_at:observedAt,passed:true,
      artifact_sha256:artifactSha,parent_commit:String(request.parent_commit),branch_name:String(request.branch_name),changed_paths:artifact.brief.changedPaths,patch_bytes:patchBytes,
      generated_by:artifact.generator,generator_run_id:receiptId,provenance_complete:true,protected_scope_clear:true,leakage_detected:false,
      evidence_refs:Array.isArray(request.evidence_refs)?request.evidence_refs.map(String):[],payload:{files:artifact.files,generator_version:EVOLUTION_CODEGEN_VERSION,branch_materialized:false,required_human_review:true,canonical_mutation:false},
      evidence_class:EVOLUTION_EVIDENCE_CLASS,shadow_only:true,live_execution:false,autonomous_apply_allowed:false,
    },{onConflict:"receipt_id",ignoreDuplicates:true});
    if(ins.error)throw new Error(`persist_generated:${ins.error.message}`);generated++;results.push({candidate_id:candidateId,hypothesis_kind:h.hypothesisKind,artifact_sha256:artifactSha,patch_bytes:patchBytes,files:artifact.files.map(f=>f.path)});
  }
  return{requests:q.data?.length??0,generated,unsupported,skipped_existing:skippedExisting,results};
}

async function receipt(startedAt:string,status:"SUCCESS"|"FAILED"|"SKIPPED",observed:number,stored:number,error?:unknown){const finishedAt=new Date().toISOString(),runId=await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);const q=await db.from("brian_collector_runs").insert({run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:finishedAt,status,observed_records:observed,stored_records:stored,degraded_sources:[],error_class:error?"EVOLUTION_TEMPLATE_GENERATOR_ERROR":null,error_message:error?String(error).slice(0,1200):null,metadata:{generator_version:EVOLUTION_CODEGEN_VERSION,branch_materialized:false,canonical_mutation:false,autonomous_apply_allowed:false},evidence_class:EVOLUTION_EVIDENCE_CLASS,shadow_only:true,live_execution:false});if(q.error)console.error("template generator receipt",q.error.message);}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);const startedAt=new Date().toISOString();
  try{await requireCronAuth(req,db);}catch(error){return out({status:"UNAUTHORIZED",error:String(error),shadow_only:true,live_execution:false},401);}
  try{const lease=await withCollectorLease(db,COLLECTOR_ID,LEASE_SECONDS,async()=>{const result=await generatePending();await receipt(startedAt,"SUCCESS",result.requests,result.generated);return{status:"SUCCESS",collector_id:COLLECTOR_ID,generator_version:EVOLUTION_CODEGEN_VERSION,...result,branch_materialized:false,required_human_review:true,canonical_mutation:false,autonomous_apply_allowed:false,shadow_only:true,live_execution:false};});if(lease.contended){await receipt(startedAt,"SKIPPED",0,0);return out({status:"SKIPPED_LEASE_CONTENDED",shadow_only:true,live_execution:false});}return out(lease.value);}catch(error){await receipt(startedAt,"FAILED",0,0,error);return out({status:"FAILED",error:String(error),canonical_mutation:false,autonomous_apply_allowed:false,shadow_only:true,live_execution:false},500);}
});
