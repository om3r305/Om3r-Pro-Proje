import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });
const AUTH_ID = "control-v3";
const ALLOWED_ORIGIN = /^https:\/\/monster-coins(?:-pro)?-[a-z0-9-]*oemer-yildirim\.vercel\.app$/i;
const ALLOWED_EXACT = new Set([
  "https://monster-coins-pro-seven.vercel.app",
  "https://monster-coins-pro-oemer-yildirim.vercel.app",
  "https://monster-coins-pro-git-brian-2026-oemer-yildirim.vercel.app",
  "http://localhost:3000",
  "http://127.0.0.1:3000",
]);
const PIPELINE = ["CLAIMED","UNDERSTAND","PLAN","CODE","COMPILE","TEST","REPLAY","STRESS","REVIEW","PR","PREVIEW","MEASURE","HUMAN_APPROVAL","DEPLOY","MONITOR","COMPLETE"];
const REVIEWED_PHASES = ["PR","PREVIEW","MEASURE","HUMAN_APPROVAL","DEPLOY","MONITOR","COMPLETE"];
const TERMINAL_SOURCE_STAGES = new Set(["REJECTED","RETIRED","ARCHIVED"]);

function cors(origin?: string | null): Record<string,string> {
  const allowed = origin && (ALLOWED_EXACT.has(origin) || ALLOWED_ORIGIN.test(origin)) ? origin : "https://monster-coins-pro-oemer-yildirim.vercel.app";
  return {"access-control-allow-origin":allowed,"access-control-allow-headers":"content-type,x-brian-dashboard-key","access-control-allow-methods":"POST,OPTIONS",vary:"Origin"};
}
function out(body:unknown,status=200,origin?:string|null){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store",...cors(origin)}});}
async function sha256Hex(value:string){const digest=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(value)));return[...digest].map(b=>b.toString(16).padStart(2,"0")).join("");}
function constantTimeEqual(left:string,right:string){if(left.length!==right.length)return false;let diff=0;for(let i=0;i<left.length;i++)diff|=left.charCodeAt(i)^right.charCodeAt(i);return diff===0;}
async function auth(req:Request){const supplied=(req.headers.get("x-brian-dashboard-key")??"").trim();if(!supplied)throw new Error("UNAUTHORIZED_DASHBOARD");const q=await db.from("brian_dashboard_auth").select("dashboard_key_sha256").eq("auth_id",AUTH_ID).single();if(q.error||!q.data)throw new Error("DASHBOARD_AUTH_UNAVAILABLE");if(!constantTimeEqual(await sha256Hex(supplied),String(q.data.dashboard_key_sha256??"")))throw new Error("UNAUTHORIZED_DASHBOARD");}
function priority(row:any){const n=Number(row?.metadata?.priority??0);return Number.isFinite(n)?n:0;}
function timeMs(value:unknown){const ms=Date.parse(String(value??""));return Number.isFinite(ms)?ms:0;}
function compareNewest(a:any,b:any){return timeMs(b.requested_at)-timeMs(a.requested_at)||timeMs(b.created_at)-timeMs(a.created_at)||String(b.request_id??"").localeCompare(String(a.request_id??""));}
function refs(value:unknown){return Array.isArray(value)?value.map(String).filter(Boolean):[];}
function eventEvidence(event:any){const p=event?.payload;if(!p)return null;if(typeof p==="string"){try{return JSON.parse(p);}catch{return {text:p};}}return p;}
function numberOr(value:unknown,fallback:number){const n=Number(value);return Number.isFinite(n)?n:fallback;}
function worldMeta(row:any){return (row?.metadata??{}) as Record<string,unknown>;}

Deno.serve(async(req:Request)=>{
  const origin=req.headers.get("origin");
  if(req.method==="OPTIONS")return new Response(null,{status:204,headers:cors(origin)});
  if(req.method!=="POST")return out({error:"POST required"},405,origin);
  try{await auth(req);}catch(error){return out({error:String(error)},401,origin);}
  try{
    const budgetWindowStart=new Date(Date.now()-24*3600_000).toISOString();
    const [controlQ,runsQ,claimedRequestQ,budgetRunsQ,reviewPassQ,reviewPhaseQ,eventsQ,requestsQ,hypQ,artifactQ,sourcesQ,assessQ]=await Promise.all([
      db.from("brian_evolution_engineering_control").select("control_id,autonomous_claim_enabled,base_branch,max_concurrent_runs,require_human_approval,monitor_minutes,updated_at,metadata").eq("control_id","default").single(),
      db.from("brian_evolution_engineering_runs").select("run_id,request_id,candidate_id,hypothesis_id,worker_id,phase,status,base_branch,base_sha,source_parent_sha,branch_name,commit_sha,pr_number,pr_url,preview_url,previous_good_sha,deployed_sha,rollback_sha,compile_passed,tests_passed,replay_passed,stress_passed,review_passed,preview_passed,measurement_passed,human_approval_status,human_approved_by,human_approved_at,monitor_status,failure_reason,metadata,shadow_only,live_execution,autonomous_apply_allowed,claimed_at,created_at,updated_at").order("created_at",{ascending:false}).limit(200),
      db.from("brian_evolution_engineering_runs").select("request_id").order("created_at",{ascending:false}).limit(5000),
      db.from("brian_evolution_engineering_runs").select("created_at,metadata").gte("created_at",budgetWindowStart).order("created_at",{ascending:false}).limit(1000),
      db.from("brian_evolution_engineering_runs").select("hypothesis_id").eq("review_passed",true).limit(4000),
      db.from("brian_evolution_engineering_runs").select("hypothesis_id,phase").in("phase",REVIEWED_PHASES).limit(4000),
      db.from("brian_evolution_engineering_events").select("event_id,run_id,observed_at,event_kind,phase,passed,commit_sha,evidence_class,shadow_only,live_execution,payload").order("observed_at",{ascending:false}).limit(1200),
      db.from("brian_evolution_codegen_requests").select("request_id,candidate_id,hypothesis_id,requested_at,created_at,parent_commit,branch_name,changed_paths,objective,constraints,success_criteria,evidence_refs,required_human_review,external_generator_required,shadow_only,live_execution,autonomous_apply_allowed,metadata").order("requested_at",{ascending:false}).limit(900),
      db.from("brian_evolution_hypothesis_snapshots").select("hypothesis_id,observed_at,title,evidence_refs,counter_evidence_refs,metadata").order("observed_at",{ascending:false}).limit(1200),
      db.from("brian_evolution_code_artifact_receipts").select("receipt_id,candidate_id,hypothesis_id,evidence_kind,observed_at,passed,artifact_sha256,parent_commit,branch_name,changed_paths,patch_bytes,generated_by,provenance_complete,protected_scope_clear,leakage_detected,evidence_refs,evidence_class,shadow_only,live_execution,autonomous_apply_allowed,payload").order("observed_at",{ascending:false}).limit(800),
      db.from("brian_world_source_candidates").select("candidate_id,source_id,discovered_at,canonical_uri,provider,authority_class,access_mode,stage").order("discovered_at",{ascending:false}).limit(2000),
      db.from("brian_world_source_assessments").select("source_id,assessed_at,trust_score,eligible_for_research").order("assessed_at",{ascending:false}).limit(2000),
    ]);
    if(controlQ.error||!controlQ.data)throw new Error(`control:${controlQ.error?.message??"missing"}`);
    for(const [name,q] of [["runs",runsQ],["claimed_requests",claimedRequestQ],["budget_runs",budgetRunsQ],["review_pass",reviewPassQ],["review_phase",reviewPhaseQ],["events",eventsQ],["requests",requestsQ],["hypotheses",hypQ],["artifacts",artifactQ],["sources",sourcesQ],["assessments",assessQ]] as const) if(q.error)throw new Error(`${name}:${q.error.message}`);

    const runs=runsQ.data??[],events=eventsQ.data??[],requests=requestsQ.data??[],hypotheses=hypQ.data??[],artifacts=artifactQ.data??[],sources=sourcesQ.data??[],assessments=assessQ.data??[];
    const control:any=controlQ.data,controlMeta:any=control.metadata??{};
    const trustFloor=Math.max(0,Math.min(1,numberOr(controlMeta.world_source_trust_floor,0.72)));
    const worldEnabled=controlMeta.world_to_engineering_enabled===true;
    const requestById=new Map(requests.map((r:any)=>[String(r.request_id),r]));
    const runRequestIds=new Set((claimedRequestQ.data??[]).map((r:any)=>String(r.request_id)));
    const reviewedHypothesisIds=new Set<string>();
    for(const row of reviewPassQ.data??[])reviewedHypothesisIds.add(String(row.hypothesis_id));
    for(const row of reviewPhaseQ.data??[])reviewedHypothesisIds.add(String(row.hypothesis_id));
    const latestHyp=new Map<string,any>();for(const h of hypotheses){const k=String(h.hypothesis_id);if(!latestHyp.has(k))latestHyp.set(k,h);}
    const latestAssess=new Map<string,any>();for(const a of assessments){const k=String(a.source_id);if(!latestAssess.has(k))latestAssess.set(k,a);}
    const latestSource=new Map<string,any>();for(const s of sources){const k=String(s.source_id);if(!latestSource.has(k))latestSource.set(k,s);}

    function sourceTier(s:any){
      if(!s)return "UNKNOWN";
      if(TERMINAL_SOURCE_STAGES.has(String(s.stage)))return "RETIRED";
      const assessment=s.assessment??null;
      const trust=numberOr(assessment?.trust_score,0);
      if(s.authority_class==="OFFICIAL_PRIMARY"&&s.access_mode==="PUBLIC_NO_KEY"&&assessment?.eligible_for_research===true&&trust>=trustFloor)return "VERIFIED_PRIMARY";
      if(assessment?.eligible_for_research===true&&trust>=trustFloor)return "VERIFIED_RESEARCH";
      if(s.access_mode==="PUBLIC_NO_KEY")return "OPEN_WEB_OBSERVATION";
      return "UNVERIFIED";
    }

    const sourceLibrary=[...latestSource.values()].map((s:any)=>{
      const assessment=latestAssess.get(String(s.source_id))??null;
      const row={...s,assessment};
      return {...row,tier:sourceTier(row),trust_score:numberOr(assessment?.trust_score,0)};
    }).sort((a:any,b:any)=>{
      const rank=(x:any)=>x.tier==="VERIFIED_PRIMARY"?0:x.tier==="VERIFIED_RESEARCH"?1:x.tier==="OPEN_WEB_OBSERVATION"?2:x.tier==="UNVERIFIED"?3:4;
      return rank(a)-rank(b)||numberOr(b.trust_score,0)-numberOr(a.trust_score,0)||timeMs(b.discovered_at)-timeMs(a.discovered_at);
    });
    const sourceMap=new Map(sourceLibrary.map((s:any)=>[String(s.source_id),s]));

    function currentWorldRequest(request:any){
      const meta=worldMeta(request);
      if(meta.world_engineering!==true)return true;
      if(!worldEnabled)return false;
      const sourceId=String(meta.world_source_id??""),candidateId=String(meta.world_source_candidate_id??"");
      const source=sourceMap.get(sourceId),assessment=source?.assessment;
      if(!source||!assessment||!sourceId||!candidateId)return false;
      return String(source.candidate_id??"")===candidateId
        && assessment.eligible_for_research===true
        && numberOr(assessment.trust_score,-1)>=trustFloor
        && source.authority_class==="OFFICIAL_PRIMARY"
        && source.access_mode==="PUBLIC_NO_KEY"
        && !TERMINAL_SOURCE_STAGES.has(String(source.stage))
        && timeMs(source.discovered_at)<=timeMs(assessment.assessed_at);
    }

    const newestByHypothesis=new Map<string,any>();for(const row of [...requests].sort(compareNewest)){const k=String(row.hypothesis_id??row.request_id??"");if(!newestByHypothesis.has(k))newestByHypothesis.set(k,row);}
    const eligibleQueue=[...newestByHypothesis.values()].filter((r:any)=>{
      if(runRequestIds.has(String(r.request_id))||r.required_human_review!==true||r.shadow_only!==true||r.live_execution!==false||r.autonomous_apply_allowed!==false)return false;
      const meta=worldMeta(r);
      if(meta.world_engineering===true){
        if(!currentWorldRequest(r))return false;
        if(meta.parent_rotation_policy==="STABLE_ONCE"&&reviewedHypothesisIds.has(String(r.hypothesis_id)))return false;
      }
      return true;
    }).sort((a:any,b:any)=>priority(b)-priority(a)||compareNewest(a,b));

    const eventsByRun=new Map<string,any[]>();for(const event of events){const k=String(event.run_id),list=eventsByRun.get(k)??[];if(list.length<60)list.push({...event,payload:eventEvidence(event)});eventsByRun.set(k,list);}
    const artifactsByCandidate=new Map<string,any[]>();for(const art of artifacts){const k=String(art.candidate_id),list=artifactsByCandidate.get(k)??[];if(list.length<30)list.push(art);artifactsByCandidate.set(k,list);}

    function lineage(request:any,hyp:any){
      const all=[...refs(request?.evidence_refs),...refs(hyp?.evidence_refs),...refs(hyp?.counter_evidence_refs)];
      const ids=new Set<string>();
      const requestSource=String(request?.metadata?.world_source_id??"");if(requestSource)ids.add(requestSource);
      const hypSource=String(hyp?.metadata?.world_source_id??"");if(hypSource)ids.add(hypSource);
      for(const ref of all)if(ref.startsWith("world_source:")){const sourceId=ref.slice("world_source:".length);if(sourceId)ids.add(sourceId);}
      const matched=[...ids].map(id=>sourceMap.get(id)).filter(Boolean).slice(0,30);
      const classes={collector:all.filter(r=>r.startsWith("collector:")),outcome:all.filter(r=>r.includes("outcome")||r.includes("decision_outcomes")),world_source:all.filter(r=>r.startsWith("world_source:")||r.startsWith("world_source_candidate:")||r.startsWith("world_source_assessment:")),engineering:all.filter(r=>r.includes("review")||r.includes("compile")||r.includes("candidate")||r.includes("engineering")),other:all.filter(r=>!r.startsWith("collector:")&&!r.includes("outcome")&&!r.startsWith("world_source:")&&!r.startsWith("world_source_candidate:")&&!r.startsWith("world_source_assessment:")&&!r.includes("review")&&!r.includes("compile")&&!r.includes("candidate")&&!r.includes("engineering"))};
      return {evidence_refs:all,direct_world_sources:matched,direct_world_source_count:matched.length,connection:matched.length?"DIRECT":"NO_DIRECT_SOURCE_REF",classes};
    }

    function classifyTask(request:any,hyp:any,lin:any){
      const meta=worldMeta(request);
      if(meta.world_engineering===true)return {scope:"WORLD",world_source_required:true,evidence_policy:"VERIFIED_WORLD_SOURCE_REQUIRED"};
      if((lin?.direct_world_source_count??0)>0||lin?.classes?.world_source?.length>0||typeof hyp?.metadata?.world_source_id==="string")return {scope:"MIXED",world_source_required:true,evidence_policy:"WORLD_PLUS_INTERNAL_EVIDENCE"};
      return {scope:"CORE_ALGORITHM",world_source_required:false,evidence_policy:"INTERNAL_TEST_REPLAY_OUTCOME_ALLOWED"};
    }

    const enrichedRuns=runs.slice(0,60).map((run:any)=>{
      const request=requestById.get(String(run.request_id));const hyp=latestHyp.get(String(run.hypothesis_id));const lin=lineage(request,hyp);const task=classifyTask(request,hyp,lin);
      return {...run,objective:request?.objective??null,requested_changed_paths:request?.changed_paths??[],request_priority:priority(request),request_evidence_refs:refs(request?.evidence_refs),request_constraints:request?.constraints??[],request_success_criteria:request?.success_criteria??[],request_metadata:request?.metadata??{},hypothesis:hyp??null,source_lineage:lin,task_scope:task.scope,world_source_required:task.world_source_required,evidence_policy:task.evidence_policy,artifact_receipts:artifactsByCandidate.get(String(run.candidate_id))??[],recent_events:eventsByRun.get(String(run.run_id))??[]};
    });
    const eligibleQueueView=eligibleQueue.map((request:any)=>{const hyp=latestHyp.get(String(request.hypothesis_id));const lin=lineage(request,hyp);const task=classifyTask(request,hyp,lin);return {...request,hypothesis:hyp??null,source_lineage:lin,task_scope:task.scope,world_source_required:task.world_source_required,evidence_policy:task.evidence_policy,claimable:true};});
    const approvals=enrichedRuns.filter((r:any)=>r.phase==="HUMAN_APPROVAL"&&r.status==="WAITING");
    const active=enrichedRuns.filter((r:any)=>["RUNNING","WAITING"].includes(String(r.status))&&!['HUMAN_APPROVAL','COMPLETE','BLOCKED','ROLLBACK'].includes(String(r.phase)));

    const worldReady=sourceLibrary.filter((s:any)=>s.tier==="VERIFIED_PRIMARY");
    const researchVerified=sourceLibrary.filter((s:any)=>s.tier==="VERIFIED_PRIMARY"||s.tier==="VERIFIED_RESEARCH");
    const observationPool=sourceLibrary.filter((s:any)=>s.tier==="OPEN_WEB_OBSERVATION");
    const worldHypotheses=[...latestHyp.values()].filter((h:any)=>typeof h?.metadata?.world_source_id==="string");
    const worldQueue=eligibleQueueView.filter((r:any)=>r?.metadata?.world_engineering===true);
    const worldRuns=enrichedRuns.filter((r:any)=>r?.request_metadata?.world_engineering===true||typeof r.hypothesis?.metadata?.world_source_id==="string");
    const autonomousClaims24h=(budgetRunsQ.data??[]).filter((r:any)=>String(r.metadata?.claim_mode??"")==="AUTONOMOUS").length;
    const autonomousLimit24h=Math.max(1,Math.min(24,Math.trunc(numberOr(controlMeta.autonomous_claim_limit_24h,4))));
    const autonomyBudget={limit_24h:autonomousLimit24h,used_24h:autonomousClaims24h,remaining_24h:Math.max(0,autonomousLimit24h-autonomousClaims24h),max_concurrent_runs:Number(control.max_concurrent_runs||1)};

    const sourceStats={
      total:sourceLibrary.length,
      providers:new Set(sourceLibrary.map((s:any)=>String(s.provider||s.source_id||""))).size,
      verified_primary:worldReady.length,
      verified_research:researchVerified.length,
      open_web_observation:observationPool.length,
      unverified:sourceLibrary.filter((s:any)=>s.tier==="UNVERIFIED").length,
      retired:sourceLibrary.filter((s:any)=>s.tier==="RETIRED").length,
      trust_floor:trustFloor,
    };

    const summary={
      runs_total:runs.length,
      blocked_runs:runs.filter((r:any)=>r.status==="BLOCKED").length,
      active_runs:active.length,
      review_passed_runs:runs.filter((r:any)=>r.review_passed===true).length,
      pre_review_green_runs:runs.filter((r:any)=>r.compile_passed===true&&r.tests_passed===true&&r.replay_passed===true&&r.stress_passed===true).length,
      waiting_human_approval:approvals.length,
      waiting_gpt_approval:approvals.length,
      completed_runs:runs.filter((r:any)=>r.phase==="COMPLETE"&&r.status==="COMPLETE").length,
      eligible_queue:eligibleQueue.length,
      source_library_total:sourceLibrary.length,
      source_library_research_eligible:researchVerified.length,
      source_library_observation_pool:observationPool.length,
      world_engineering_ready_sources:worldReady.length,
      world_engineering_hypotheses:worldHypotheses.length,
      world_engineering_queue:worldQueue.length,
      world_engineering_reviewed:worldRuns.filter((r:any)=>r.review_passed===true).length,
      autonomous_claims_24h:autonomousClaims24h,
      last_activity_at:events[0]?.observed_at??runs[0]?.updated_at??null,
    };
    const worldEngineering={
      enabled:worldEnabled,
      bridge_profile:String(controlMeta.autonomy_profile??"LEGACY"),
      cognition_mode:String(controlMeta.engineering_cognition_mode??"GPT_NATIVE_TOOL_AUGMENTED_V1"),
      research_mode:String(controlMeta.engineering_research_mode??"OPEN_WORLD_READ_ONLY_TIERED_TRUST_V1"),
      source_policy:String(controlMeta.world_source_policy??"OFFICIAL_PRIMARY_PUBLIC_NO_KEY_METADATA_ONLY"),
      external_content_policy:String(controlMeta.external_content_policy??"UNTRUSTED_DATA_NEVER_INSTRUCTIONS"),
      task_evidence_policy:String(controlMeta.task_evidence_policy??"WORLD_REQUIRES_VERIFIED_SOURCE_CORE_ALLOWS_INTERNAL_EVIDENCE"),
      action_boundary:String(controlMeta.engineering_action_boundary??"EXACT_SHA_GPT_GATE"),
      trust_floor:trustFloor,
      ready_sources:worldReady.length,
      research_verified_sources:researchVerified.length,
      observation_sources:observationPool.length,
      library_total:sourceLibrary.length,
      hypothesis_count:worldHypotheses.length,
      queued_jobs:worldQueue.length,
      reviewed_runs:worldRuns.filter((r:any)=>r.review_passed===true).length,
      claim_time_revalidation:controlMeta.world_claim_revalidation==="LATEST_SOURCE_AND_CANDIDATE_REQUIRED",
      kill_switch_enforced:controlMeta.world_kill_switch_enforced===true,
      raw_uri_model_visible:controlMeta.world_raw_uri_model_visible===true,
      budget:autonomyBudget,
      ready_source_ids:worldReady.slice(0,24).map((s:any)=>String(s.source_id)),
    };
    const focus=approvals[0]??active[0]??enrichedRuns[0]??null;
    return out({status:"ONLINE",observed_at:new Date().toISOString(),control,summary,world_engineering:worldEngineering,source_library_stats:sourceStats,pipeline:PIPELINE,current_run:active[0]??null,approval_run:approvals[0]??null,approval_runs:approvals,recent_runs:enrichedRuns,eligible_queue:eligibleQueueView.slice(0,30),source_library:sourceLibrary.slice(0,200),truth:{self_coding_enabled:control.autonomous_claim_enabled===true,gpt_approval_actor:String(controlMeta.gpt_approval_actor??"gpt-evidence-gate"),world_to_engineering_enabled:worldEngineering.enabled,world_claim_time_revalidation:worldEngineering.claim_time_revalidation,world_kill_switch_enforced:worldEngineering.kill_switch_enforced,external_content_never_instructions:worldEngineering.external_content_policy==="UNTRUSTED_DATA_NEVER_INSTRUCTIONS",autonomous_budget_remaining_24h:autonomyBudget.remaining_24h,end_to_end_proven:summary.completed_runs>0||summary.waiting_gpt_approval>0,gpt_approval_waiting:summary.waiting_gpt_approval,focus_task_scope:focus?.task_scope??null,focus_world_source_required:focus?.world_source_required??false,world_library_directly_linked_to_focus:(focus?.source_lineage?.direct_world_source_count??0)>0},shadow_only:true,live_execution:false,dip_isolated:true},200,origin);
  }catch(error){return out({status:"DEGRADED",error:String(error),shadow_only:true,live_execution:false,dip_isolated:true},500,origin);}
});
