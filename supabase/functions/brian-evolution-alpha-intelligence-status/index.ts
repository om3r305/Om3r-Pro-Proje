import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const AUTH_ID="control-v3";
const ALLOWED_ORIGIN=/^https:\/\/monster-coins(?:-pro)?-[a-z0-9-]*oemer-yildirim\.vercel\.app$/i;
const ALLOWED_EXACT=new Set(["https://monster-coins-pro-seven.vercel.app","https://monster-coins-pro-oemer-yildirim.vercel.app","https://monster-coins-pro-git-brian-2026-oemer-yildirim.vercel.app","http://localhost:3000","http://127.0.0.1:3000"]);
function cors(origin?:string|null):Record<string,string>{const allowed=origin&&(ALLOWED_EXACT.has(origin)||ALLOWED_ORIGIN.test(origin))?origin:"https://monster-coins-pro-oemer-yildirim.vercel.app";return{"access-control-allow-origin":allowed,"access-control-allow-headers":"content-type,x-brian-dashboard-key","access-control-allow-methods":"POST,OPTIONS","vary":"Origin"};}
function out(body:unknown,status=200,origin?:string|null){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store",...cors(origin)}});}
async function sha256Hex(value:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(value)));return[...d].map(b=>b.toString(16).padStart(2,"0")).join("");}
function constantTimeEqual(left:string,right:string){if(left.length!==right.length)return false;let diff=0;for(let i=0;i<left.length;i++)diff|=left.charCodeAt(i)^right.charCodeAt(i);return diff===0;}
async function requireDashboardAuth(req:Request){const supplied=(req.headers.get("x-brian-dashboard-key")??"").trim();if(!supplied)throw new Error("UNAUTHORIZED_DASHBOARD");const q=await db.from("brian_dashboard_auth").select("dashboard_key_sha256").eq("auth_id",AUTH_ID).single();if(q.error||!q.data)throw new Error("DASHBOARD_AUTH_UNAVAILABLE");if(!constantTimeEqual(await sha256Hex(supplied),String(q.data.dashboard_key_sha256??"")))throw new Error("UNAUTHORIZED_DASHBOARD");}
function avg(values:number[]){return values.length?values.reduce((a,b)=>a+b,0)/values.length:null;}

Deno.serve(async(req:Request)=>{
  const origin=req.headers.get("origin");if(req.method==="OPTIONS")return new Response(null,{status:204,headers:cors(origin)});if(req.method!=="POST")return out({error:"POST required"},405,origin);
  try{await requireDashboardAuth(req);}catch(error){return out({error:String(error)},401,origin);}
  const [edgesQ,runsQ]=await Promise.all([
    db.from("brian_alpha_expected_edge_challenger")
      .select("edge_id,decision_id,observed_at,evaluated_at,asset_id,canonical_action,direction,canonical_evidence_score,support_groups,reliability_window_end,reliability_generated_at,expected_gross_move_bps,estimated_round_trip_cost_bps,uncertainty_penalty_bps,event_decay_penalty_bps,expected_net_edge_bps,minimum_net_margin_bps,recommendation,eligible,mature_group_count,reliability_weights,pit_clear,reasons,model_version")
      .order("observed_at",{ascending:false}).limit(120),
    db.from("brian_collector_runs")
      .select("collector_id,status,started_at,finished_at,observed_records,stored_records,error_class,error_message,metadata")
      .eq("collector_id","brian-evolution-alpha-edge-challenger-v1").order("started_at",{ascending:false}).limit(20),
  ]);
  if(edgesQ.error||runsQ.error)return out({status:"DEGRADED",errors:[edgesQ.error?.message,runsQ.error?.message].filter(Boolean),shadow_only:true,live_execution:false},500,origin);
  const edges=edgesQ.data??[],resolved=edges.filter(row=>Number.isFinite(Number(row.expected_net_edge_bps))),net=resolved.map(row=>Number(row.expected_net_edge_bps));
  const allow=edges.filter(row=>String(row.recommendation)==="ALLOW_EDGE").length,downgrade=edges.filter(row=>String(row.recommendation)==="DOWNGRADE_TO_WAIT").length;
  const failClosed=edges.length-allow-downgrade,pitClear=edges.filter(row=>row.pit_clear===true).length;
  return out({
    status:"ONLINE",observed_at:new Date().toISOString(),summary:{evaluated:edges.length,allow_edge:allow,downgrade_to_wait:downgrade,fail_closed:failClosed,pit_clear:pitClear,avg_expected_net_edge_bps:avg(net),positive_expected_net_edge:resolved.filter(row=>Number(row.expected_net_edge_bps)>0).length,resolved_edge:resolved.length},
    edges,runs:runsQ.data??[],canonical_mutation:false,direct_alpha_influence:false,bounded_reliability_feedback:true,shadow_only:true,live_execution:false,
  },200,origin);
});
