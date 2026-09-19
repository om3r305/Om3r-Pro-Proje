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

function cors(origin?: string | null): Record<string,string> {
  const allowed = origin && (ALLOWED_EXACT.has(origin) || ALLOWED_ORIGIN.test(origin))
    ? origin
    : "https://monster-coins-pro-oemer-yildirim.vercel.app";
  return {
    "access-control-allow-origin": allowed,
    "access-control-allow-headers": "content-type,x-brian-dashboard-key",
    "access-control-allow-methods": "POST,OPTIONS",
    vary: "Origin",
  };
}
function out(body:unknown,status=200,origin?:string|null){
  return new Response(JSON.stringify(body),{
    status,
    headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store",...cors(origin)}
  });
}
async function sha256Hex(value:string){
  const digest=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(value)));
  return [...digest].map(b=>b.toString(16).padStart(2,"0")).join("");
}
function constantTimeEqual(left:string,right:string){
  if(left.length!==right.length)return false;
  let diff=0;
  for(let i=0;i<left.length;i++)diff|=left.charCodeAt(i)^right.charCodeAt(i);
  return diff===0;
}
async function auth(req:Request){
  const supplied=(req.headers.get("x-brian-dashboard-key")??"").trim();
  if(!supplied)throw new Error("UNAUTHORIZED_DASHBOARD");
  const q=await db.from("brian_dashboard_auth").select("dashboard_key_sha256").eq("auth_id",AUTH_ID).single();
  if(q.error||!q.data)throw new Error("DASHBOARD_AUTH_UNAVAILABLE");
  if(!constantTimeEqual(await sha256Hex(supplied),String(q.data.dashboard_key_sha256??"")))throw new Error("UNAUTHORIZED_DASHBOARD");
}
function num(v:unknown,f=0){const n=Number(v);return Number.isFinite(n)?n:f}
function refs(v:unknown){return Array.isArray(v)?v.map(String).filter(Boolean):[]}
function eventPayload(v:unknown){
  if(!v)return null;
  if(typeof v==="string"){try{return JSON.parse(v)}catch{return {text:v}}}
  return v;
}

Deno.serve(async(req:Request)=>{
  const origin=req.headers.get("origin");
  if(req.method==="OPTIONS")return new Response(null,{status:204,headers:cors(origin)});
  if(req.method!=="POST")return out({error:"POST required"},405,origin);
  try{await auth(req)}catch(error){return out({error:String(error)},401,origin)}

  try{
    const since24h=new Date(Date.now()-24*3600_000).toISOString();
    const [controlQ,runsQ,eventsQ,requestsQ,hypQ,sourcesQ,assessQ]=await Promise.all([
      db.from("brian_evolution_engineering_control")
        .select("control_id,autonomous_claim_enabled,base_branch,max_concurrent_runs,require_human_approval,monitor_minutes,updated_at,metadata")
        .eq("control_id","default").single(),
      db.from("brian_evolution_engineering_runs")
        .select("run_id,request_id,candidate_id,hypothesis_id,worker_id,phase,status,base_branch,base_sha,source_parent_sha,branch_name,commit_sha,pr_number,pr_url,preview_url,previous_good_sha,deployed_sha,rollback_sha,compile_passed,tests_passed,replay_passed,stress_passed,review_passed,preview_passed,measurement_passed,human_approval_status,human_approved_by,human_approved_at,monitor_status,failure_reason,metadata,shadow_only,live_execution,autonomous_apply_allowed,claimed_at,created_at,updated_at")
        .order("created_at",{ascending:false}).limit(120),
      db.from("brian_evolution_engineering_events")
        .select("event_id,run_id,observed_at,event_kind,phase,passed,commit_sha,evidence_class,shadow_only,live_execution,payload")
        .order("observed_at",{ascending:false}).limit(700),
      db.from("brian_evolution_codegen_requests")
        .select("request_id,candidate_id,hypothesis_id,requested_at,created_at,parent_commit,branch_name,changed_paths,objective,constraints,success_criteria,evidence_refs,required_human_review,external_generator_required,shadow_only,live_execution,autonomous_apply_allowed,metadata")
        .order("requested_at",{ascending:false}).limit(300),
      db.from("brian_evolution_hypothesis_snapshots")
        .select("hypothesis_id,observed_at,title,evidence_refs,counter_evidence_refs,metadata")
        .order("observed_at",{ascending:false}).limit(400),
      db.from("brian_world_source_latest")
        .select("candidate_id,source_id,discovered_at,canonical_uri,provider,authority_class,access_mode,stage")
        .order("discovered_at",{ascending:false}).limit(250),
      db.from("brian_world_source_assessment_latest")
        .select("source_id,assessed_at,trust_score,eligible_for_research")
        .order("assessed_at",{ascending:false}).limit(250),
    ]);

    if(controlQ.error||!controlQ.data)throw new Error(`control:${controlQ.error?.message??"missing"}`);
    for(const [name,q] of [["runs",runsQ],["events",eventsQ],["requests",requestsQ],["hypotheses",hypQ],["sources",sourcesQ],["assessments",assessQ]] as const){
      if(q.error)throw new Error(`${name}:${q.error.message}`);
    }

    const control:any=controlQ.data;
    const runs:any[]=runsQ.data??[];
    const events:any[]=eventsQ.data??[];
    const requests:any[]=requestsQ.data??[];
    const hypotheses:any[]=hypQ.data??[];
    const sources:any[]=sourcesQ.data??[];
    const assessments:any[]=assessQ.data??[];

    const requestById=new Map(requests.map(r=>[String(r.request_id),r]));
    const hypById=new Map<string,any>();
    for(const h of hypotheses){const k=String(h.hypothesis_id);if(!hypById.has(k))hypById.set(k,h)}
    const eventsByRun=new Map<string,any[]>();
    for(const ev of events){
      const k=String(ev.run_id),list=eventsByRun.get(k)??[];
      if(list.length<60)list.push({...ev,payload:eventPayload(ev.payload)});
      eventsByRun.set(k,list);
    }

    const enriched=runs.slice(0,60).map(run=>{
      const reqRow=requestById.get(String(run.request_id))??null;
      return {
        ...run,
        objective:reqRow?.objective??null,
        requested_changed_paths:reqRow?.changed_paths??[],
        request_constraints:reqRow?.constraints??[],
        request_success_criteria:reqRow?.success_criteria??[],
        request_evidence_refs:refs(reqRow?.evidence_refs),
        request_metadata:reqRow?.metadata??{},
        hypothesis:hypById.get(String(run.hypothesis_id))??null,
        recent_events:eventsByRun.get(String(run.run_id))??[],
      };
    });

    const active=enriched.filter(r=>["RUNNING","WAITING"].includes(String(r.status))&&!["HUMAN_APPROVAL","COMPLETE","BLOCKED","ROLLBACK"].includes(String(r.phase)));
    const approvals=enriched.filter(r=>r.phase==="HUMAN_APPROVAL"&&r.status==="WAITING");
    const claimedIds=new Set(runs.map(r=>String(r.request_id)));
    const eligible=requests.filter(r=>
      !claimedIds.has(String(r.request_id)) &&
      r.required_human_review===true &&
      r.shadow_only===true &&
      r.live_execution===false &&
      r.autonomous_apply_allowed===false
    );

    const assessmentBySource=new Map(assessments.map(a=>[String(a.source_id),a]));
    const sourceLibrary=sources.map(s=>({...s,assessment:assessmentBySource.get(String(s.source_id))??null}));

    const controlMeta:any=control.metadata??{};
    const budgetLimit=Math.max(1,Math.min(24,Math.trunc(num(controlMeta.autonomous_claim_limit_24h,4))));
    const autonomousClaims24h=runs.filter(r=>String(r?.metadata?.claim_mode??"")==="AUTONOMOUS" && String(r.created_at)>=since24h).length;
    const worldRequests=eligible.filter(r=>r?.metadata?.world_engineering===true);

    const summary={
      runs_total:runs.length,
      blocked_runs:runs.filter(r=>r.status==="BLOCKED").length,
      active_runs:active.length,
      review_passed_runs:runs.filter(r=>r.review_passed===true).length,
      pre_review_green_runs:runs.filter(r=>r.compile_passed===true&&r.tests_passed===true&&r.replay_passed===true&&r.stress_passed===true).length,
      waiting_human_approval:approvals.length,
      waiting_gpt_approval:approvals.length,
      completed_runs:runs.filter(r=>r.phase==="COMPLETE"&&r.status==="COMPLETE").length,
      eligible_queue:eligible.length,
      source_library_total:sourceLibrary.length,
      world_engineering_queue:worldRequests.length,
      autonomous_claims_24h:autonomousClaims24h,
      last_activity_at:events[0]?.observed_at??runs[0]?.updated_at??null,
    };

    const worldEngineering={
      enabled:controlMeta.world_to_engineering_enabled===true,
      ready_sources:sourceLibrary.filter(s=>s.authority_class==="OFFICIAL_PRIMARY"&&s.assessment?.eligible_for_research===true).length,
      research_verified_sources:sourceLibrary.filter(s=>s.assessment?.eligible_for_research===true).length,
      library_total:sourceLibrary.length,
      queued_jobs:worldRequests.length,
      hypothesis_count:hypotheses.filter(h=>typeof h?.metadata?.world_source_id==="string").length,
      reviewed_runs:enriched.filter(r=>r.review_passed===true).length,
      budget:{
        limit_24h:budgetLimit,
        used_24h:autonomousClaims24h,
        remaining_24h:Math.max(0,budgetLimit-autonomousClaims24h),
        max_concurrent_runs:Number(control.max_concurrent_runs||1),
      },
    };

    return out({
      status:"ONLINE",
      observed_at:new Date().toISOString(),
      control,
      summary,
      world_engineering:worldEngineering,
      pipeline:PIPELINE,
      current_run:active[0]??null,
      approval_run:approvals[0]??null,
      approval_runs:approvals,
      recent_runs:enriched,
      eligible_queue:eligible.slice(0,30).map(r=>({...r,hypothesis:hypById.get(String(r.hypothesis_id))??null,claimable:true})),
      source_library:sourceLibrary.slice(0,200),
      truth:{
        self_coding_enabled:control.autonomous_claim_enabled===true,
        gpt_approval_actor:String(controlMeta.gpt_approval_actor??"gpt-evidence-gate"),
        world_to_engineering_enabled:worldEngineering.enabled,
        autonomous_budget_remaining_24h:worldEngineering.budget.remaining_24h,
        end_to_end_proven:summary.completed_runs>0||summary.waiting_human_approval>0,
      },
      shadow_only:true,
      live_execution:false,
      dip_isolated:true,
    },200,origin);
  }catch(error){
    return out({status:"DEGRADED",error:String(error),shadow_only:true,live_execution:false,dip_isolated:true},500,origin);
  }
});
