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
  "http://localhost:3000","http://127.0.0.1:3000",
]);
function cors(origin?:string|null){const a=origin&&(ALLOWED_EXACT.has(origin)||ALLOWED_ORIGIN.test(origin))?origin:"https://monster-coins-pro-oemer-yildirim.vercel.app";return{"access-control-allow-origin":a,"access-control-allow-headers":"content-type,x-brian-dashboard-key","access-control-allow-methods":"POST,OPTIONS",vary:"Origin"};}
function out(body:unknown,status=200,origin?:string|null){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store",...cors(origin)}})}
async function sha256Hex(v:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(v)));return[...d].map(b=>b.toString(16).padStart(2,"0")).join("")}
function ct(a:string,b:string){if(a.length!==b.length)return false;let d=0;for(let i=0;i<a.length;i++)d|=a.charCodeAt(i)^b.charCodeAt(i);return d===0}
async function auth(req:Request){const supplied=(req.headers.get("x-brian-dashboard-key")??"").trim();if(!supplied)throw new Error("UNAUTHORIZED_DASHBOARD");const q=await db.from("brian_dashboard_auth").select("dashboard_key_sha256").eq("auth_id",AUTH_ID).single();if(q.error||!q.data)throw new Error("DASHBOARD_AUTH_UNAVAILABLE");if(!ct(await sha256Hex(supplied),String(q.data.dashboard_key_sha256??"")))throw new Error("UNAUTHORIZED_DASHBOARD")}
const C=(n:number)=>Math.max(0,Math.min(100,Number.isFinite(n)?n:0));
const R=(n:number)=>Math.round(n*10)/10;
const ratio=(n:number,d:number)=>d>0?C(100*n/d):0;
const ageSec=(v:unknown)=>{const t=Date.parse(String(v??""));return Number.isFinite(t)?Math.max(0,(Date.now()-t)/1000):Infinity};

type Row=Record<string,any>;
function latestBy(rows:Row[],key:(r:Row)=>string){const m=new Map<string,Row>();for(const r of rows){const k=key(r);if(!k)continue;const old=m.get(k);if(!old||Date.parse(String(r.measured_at??r.decided_at??r.started_at))>Date.parse(String(old.measured_at??old.decided_at??old.started_at)))m.set(k,r)}return m}
function family(id:string){if(id.includes("action-gate"))return"ACTION_GATE";if(id.includes("expected-edge"))return"EXPECTED_EDGE";return"OTHER"}

Deno.serve(async(req:Request)=>{
  const origin=req.headers.get("origin");
  if(req.method==="OPTIONS")return new Response(null,{status:204,headers:cors(origin)});
  if(req.method!=="POST")return out({error:"POST required"},405,origin);
  try{await auth(req)}catch(e){return out({error:String(e)},401,origin)}
  try{
    const dayAgo=new Date(Date.now()-86400000).toISOString();
    const [eq,rq,pq,aq,cq]=await Promise.all([
      db.from("brian_evolution_experiments").select("experiment_id,hypothesis_id,minimum_samples,minimum_regimes,stage,mode,created_at_source").eq("mode","PROSPECTIVE_SHADOW").in("stage",["EXPERIMENTAL","CANARY","SHADOW_VALIDATED"]).order("created_at_source",{ascending:false}).limit(100),
      db.from("brian_evolution_experiment_results").select("result_id,experiment_id,measured_at,role,samples,regimes,net_edge_bps,favorable_after_cost_rate,data_quality_ok,leakage_detected,stability_score").order("measured_at",{ascending:false}).limit(500),
      db.from("brian_evolution_promotion_decisions").select("decision_id,experiment_id,decided_at,decision,score,reasons,required_next_stage,challenger_result_id").order("decided_at",{ascending:false}).limit(200),
      db.from("brian_evolution_code_artifact_receipts").select("observed_at,passed,protected_scope_clear,leakage_detected").gte("observed_at",dayAgo).order("observed_at",{ascending:false}).limit(200),
      db.from("brian_collector_runs").select("collector_id,status,started_at,finished_at").in("collector_id",["brian-evolution-experiment-runner-v1","brian-evolution-promotion-council-v1"]).gte("started_at",dayAgo).order("started_at",{ascending:false}).limit(100),
    ]);
    const errs=[eq,rq,pq,aq,cq].filter(x=>x.error).map(x=>x.error!.message);if(errs.length)return out({status:"DEGRADED",errors:errs,shadow_only:true,live_execution:false},500,origin);
    const experiments=(eq.data??[]) as Row[],results=(rq.data??[]) as Row[],promos=(pq.data??[]) as Row[],artifacts=(aq.data??[]) as Row[],runs=(cq.data??[]) as Row[];
    const latestResult=latestBy(results,r=>`${r.experiment_id}|${r.role}`),latestPromo=latestBy(promos,r=>String(r.experiment_id));
    const supported=experiments.filter(e=>family(String(e.experiment_id))!=="OTHER").map(e=>{
      const id=String(e.experiment_id),fam=family(id),control=latestResult.get(`${id}|CONTROL`)??null,challenger=latestResult.get(`${id}|CHALLENGER`)??null,promo=latestPromo.get(id)??null;
      const minSamples=Math.max(1,Number(e.minimum_samples??300)),minRegimes=Math.max(1,Number(e.minimum_regimes??3));
      const samples=Math.max(0,Number(challenger?.samples??0)),regimes=Math.max(0,Number(challenger?.regimes??0));
      const samplePct=C(100*samples/minSamples),regimePct=C(100*regimes/minRegimes),coverage=C(Math.min(samplePct,regimePct));
      const qualityOk=challenger?.data_quality_ok===true,leakage=challenger?.leakage_detected===true,net=Number(challenger?.net_edge_bps),freshCouncil=Boolean(promo&&challenger&&Date.parse(String(promo.decided_at))>=Date.parse(String(challenger.measured_at)));
      const thresholdsMet=Boolean(challenger&&samples>=minSamples&&regimes>=minRegimes&&qualityOk&&!leakage&&Number.isFinite(net)&&net>0);
      const promoted=freshCouncil&&String(promo?.decision)==="PROMOTE_CANDIDATE";
      return{family:fam,experiment_id:id,stage:e.stage,minimum_samples:minSamples,minimum_regimes:minRegimes,samples,regimes,sample_coverage_pct:R(samplePct),regime_coverage_pct:R(regimePct),evidence_maturity_pct:R(coverage),net_edge_bps:Number.isFinite(net)?net:null,favorable_after_cost_rate:challenger?.favorable_after_cost_rate??null,data_quality_ok:challenger?.data_quality_ok??null,leakage_detected:challenger?.leakage_detected??null,measured_at:challenger?.measured_at??null,control_samples:Number(control?.samples??0),control_regimes:Number(control?.regimes??0),council_decision:promo?.decision??"NO_DECISION",council_decided_at:promo?.decided_at??null,council_reasons:promo?.reasons??[],council_fresh:freshCouncil,thresholds_met:thresholdsMet,promotion_ready:promoted};
    });
    const clean=supported.filter(x=>x.data_quality_ok===true&&!x.leakage_detected).length;
    const runLatest=latestBy(runs,r=>String(r.collector_id));
    const runner=runLatest.get("brian-evolution-experiment-runner-v1"),council=runLatest.get("brian-evolution-promotion-council-v1");
    const runnerHealthy=Boolean(runner&&String(runner.status)==="SUCCESS"&&ageSec(runner.started_at)<7200),councilHealthy=Boolean(council&&String(council.status)==="SUCCESS"&&ageSec(council.started_at)<7200);
    const safeArtifacts=artifacts.filter(a=>a.passed===true&&a.protected_scope_clear===true&&!a.leakage_detected).length;
    const artifactPct=artifacts.length?ratio(safeArtifacts,artifacts.length):100;
    const cleanliness=supported.length?ratio(clean,supported.length):0;
    const pipeline=R(C((runnerHealthy?30:0)+(councilHealthy?30:0)+cleanliness*.2+artifactPct*.2));
    const evidence=R(supported.length?supported.reduce((s,x)=>s+x.evidence_maturity_pct,0)/supported.length:0);
    const ready=supported.filter(x=>x.promotion_ready).length;
    const promotionReadiness=R(supported.length?100*ready/supported.length:0);
    return out({status:"ONLINE",observed_at:new Date().toISOString(),metrics:{pipeline_health_pct:pipeline,evidence_maturity_pct:evidence,promotion_readiness_pct:promotionReadiness},candidates:supported,research_pockets:[],research_pocket_policy:{status:"NO_ROBUST_POCKET",minimum_days:2,max_single_day_share_pct:60,canonical_authority:false,note:"Thin asset/time pockets are not surfaced as edge until multi-day robustness passes."},governance:{human_promotion_required:true,canonical_mutation:false,thresholds_relaxed:false,shadow_only:true,live_execution:false},pipeline:{runner_healthy:runnerHealthy,council_healthy:councilHealthy,safe_artifact_pass_pct:R(artifactPct),clean_candidate_pct:R(cleanliness)}},200,origin);
  }catch(e){return out({status:"FAILED_CLOSED",error:e instanceof Error?e.message:String(e),shadow_only:true,live_execution:false},500,origin)}
});
