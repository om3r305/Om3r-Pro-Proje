import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const AUTH_ID="control-v3";
const ALLOWED_ORIGIN=/^https:\/\/monster-coins(?:-pro)?-[a-z0-9-]*oemer-yildirim\.vercel\.app$/i;
const ALLOWED_EXACT=new Set([
  "https://monster-coins-pro-seven.vercel.app",
  "https://monster-coins-pro-oemer-yildirim.vercel.app",
  "https://monster-coins-pro-git-brian-2026-oemer-yildirim.vercel.app",
  "http://localhost:3000",
  "http://127.0.0.1:3000",
]);

function cors(origin?:string|null):Record<string,string>{const allowed=origin&&(ALLOWED_EXACT.has(origin)||ALLOWED_ORIGIN.test(origin))?origin:"https://monster-coins-pro-oemer-yildirim.vercel.app";return{"access-control-allow-origin":allowed,"access-control-allow-headers":"content-type,x-brian-dashboard-key","access-control-allow-methods":"POST,OPTIONS","vary":"Origin"};}
function out(body:unknown,status=200,origin?:string|null){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store",...cors(origin)}});}
async function sha256Hex(value:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(value)));return[...d].map(b=>b.toString(16).padStart(2,"0")).join("");}
function constantTimeEqual(left:string,right:string){if(left.length!==right.length)return false;let diff=0;for(let i=0;i<left.length;i++)diff|=left.charCodeAt(i)^right.charCodeAt(i);return diff===0;}
async function requireDashboardAuth(req:Request){const supplied=(req.headers.get("x-brian-dashboard-key")??"").trim();if(!supplied)throw new Error("UNAUTHORIZED_DASHBOARD");const q=await db.from("brian_dashboard_auth").select("dashboard_key_sha256").eq("auth_id",AUTH_ID).single();if(q.error||!q.data)throw new Error("DASHBOARD_AUTH_UNAVAILABLE");if(!constantTimeEqual(await sha256Hex(supplied),String(q.data.dashboard_key_sha256??"")))throw new Error("UNAUTHORIZED_DASHBOARD");}

function latestUnique(rows:Record<string,unknown>[],key:string){const seen=new Set<string>(),out:Record<string,unknown>[]=[];for(const row of rows){const id=String(row[key]??"");if(!id||seen.has(id))continue;seen.add(id);out.push(row);}return out;}

Deno.serve(async(req:Request)=>{
  const origin=req.headers.get("origin");
  if(req.method==="OPTIONS")return new Response(null,{status:204,headers:cors(origin)});
  if(req.method!=="POST")return out({error:"POST required"},405,origin);
  try{await requireDashboardAuth(req);}catch(error){return out({error:String(error)},401,origin);}

  const [hypQ,expQ,promoQ,codeQ,reviewQ,runsQ]=await Promise.all([
    db.from("brian_evolution_hypothesis_snapshots")
      .select("hypothesis_id,observed_at,title,problem_statement,proposed_mechanism,target_capabilities,stage,uncertainty,metadata,evidence_refs")
      .order("observed_at",{ascending:false}).limit(200),
    db.from("brian_evolution_experiments")
      .select("experiment_id,hypothesis_id,created_at_source,control_version,challenger_version,mode,minimum_samples,minimum_regimes,stage,metadata")
      .order("created_at_source",{ascending:false}).limit(100),
    db.from("brian_evolution_promotion_decisions")
      .select("decision_id,experiment_id,decided_at,decision,score,reasons,required_next_stage,control_result_id,challenger_result_id,metadata")
      .order("decided_at",{ascending:false}).limit(100),
    db.from("brian_evolution_codegen_requests")
      .select("request_id,candidate_id,hypothesis_id,requested_at,parent_commit,branch_name,changed_paths,objective,success_criteria,metadata")
      .order("requested_at",{ascending:false}).limit(100),
    db.from("brian_evolution_code_review_receipts")
      .select("receipt_id,candidate_id,reviewed_at,protected_scope_clear,tests_green,replay_green,stress_green,prospective_green,leakage_clear,reviewer_kind,verdict,reasons,metadata")
      .order("reviewed_at",{ascending:false}).limit(100),
    db.from("brian_collector_runs")
      .select("collector_id,status,started_at,finished_at,observed_records,stored_records,error_class,error_message,metadata")
      .in("collector_id",["brian-evolution-researcher-v1","brian-evolution-sandbox-v1","brian-evolution-experiment-runner-v1","brian-evolution-promotion-council-v1"])
      .order("started_at",{ascending:false}).limit(80),
  ]);
  const failures=[hypQ,expQ,promoQ,codeQ,reviewQ,runsQ].filter(q=>q.error).map(q=>q.error!.message);
  if(failures.length)return out({status:"DEGRADED",errors:failures,shadow_only:true,live_execution:false},500,origin);

  const hypotheses=latestUnique((hypQ.data??[]) as Record<string,unknown>[],"hypothesis_id").slice(0,30);
  const experiments=(expQ.data??[]).slice(0,30);
  const promotions=latestUnique((promoQ.data??[]) as Record<string,unknown>[],"experiment_id").slice(0,30);
  const codeRequests=latestUnique((codeQ.data??[]) as Record<string,unknown>[],"candidate_id").slice(0,30);
  const reviews=latestUnique((reviewQ.data??[]) as Record<string,unknown>[],"candidate_id").slice(0,30);
  const readyHuman=reviews.filter(r=>String(r.verdict)==="READY_FOR_HUMAN_REVIEW").length;
  const blocked=reviews.filter(r=>String(r.verdict)==="BLOCK").length;
  const promoted=promotions.filter(r=>String(r.decision)==="PROMOTE_CANDIDATE").length;
  const rejected=promotions.filter(r=>String(r.decision)==="REJECT").length;
  const now=new Date().toISOString();
  return out({
    status:"ONLINE",observed_at:now,
    summary:{hypotheses:hypotheses.length,experiments:experiments.length,code_candidates:codeRequests.length,ready_for_human_review:readyHuman,blocked_code_candidates:blocked,promotion_candidates:promoted,rejected_experiments:rejected},
    hypotheses,experiments,promotions,code_requests:codeRequests,code_reviews:reviews,runs:runsQ.data??[],
    external_generator_required:true,required_human_review:true,canonical_mutation:false,autonomous_apply_allowed:false,cloud_independent:true,shadow_only:true,live_execution:false,
  },200,origin);
});
