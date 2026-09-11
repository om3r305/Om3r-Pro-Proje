import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { deriveOceanRuns, type OceanCommand } from "../_shared/evolution_ocean.ts";

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

Deno.serve(async(req:Request)=>{
  const origin=req.headers.get("origin");if(req.method==="OPTIONS")return new Response(null,{status:204,headers:cors(origin)});if(req.method!=="POST")return out({error:"POST required"},405,origin);
  try{await requireDashboardAuth(req);}catch(error){return out({error:String(error)},401,origin);}
  const [commandsQ,checkpointsQ,reportsQ,runsQ,treasuryQ]=await Promise.all([
    db.from("brian_ocean_run_commands").select("command_id,run_id,command,requested_at,duration_hours,reason").order("requested_at",{ascending:false}).limit(300),
    db.from("brian_ocean_run_checkpoints").select("checkpoint_id,run_id,observed_at,treasury_snapshot_id,treasury_equity_usd,treasury_cash_usd,treasury_deployment_usd,treasury_open_positions,collector_runs_window,collector_failures_window,collector_degraded_window,payload").order("observed_at",{ascending:false}).limit(200),
    db.from("brian_ocean_run_reports").select("report_id,run_id,started_at,ended_at,duration_hours,summary,evidence_refs,report_version,metadata").order("ended_at",{ascending:false}).limit(20),
    db.from("brian_collector_runs").select("status,started_at,finished_at,observed_records,stored_records,error_class,error_message,metadata").eq("collector_id","brian-evolution-ocean-worker-v1").order("started_at",{ascending:false}).limit(30),
    db.from("brian_treasury_shadow_latest").select("snapshot_id,observed_at,equity_usd,cash_usd,deployment_usd,deployment_pct,cash_reserve_pct,positions,promotion_gate_open,promotion_gate_reason").limit(1).maybeSingle(),
  ]);
  const errors=[commandsQ.error,checkpointsQ.error,reportsQ.error,runsQ.error,treasuryQ.error].filter(Boolean).map(error=>error!.message);if(errors.length)return out({status:"DEGRADED",errors,shadow_only:true,live_execution:false},500,origin);
  const commands:OceanCommand[]=(commandsQ.data??[]).map(row=>({commandId:String(row.command_id),runId:String(row.run_id),command:String(row.command) as OceanCommand["command"],requestedAt:String(row.requested_at),durationHours:row.duration_hours==null?null:Number(row.duration_hours) as 24|48,reason:row.reason==null?null:String(row.reason)}));
  const now=new Date().toISOString(),derived=deriveOceanRuns(commands,now),active=derived.find(run=>run.status==="ACTIVE")??null;
  const latestCheckpointByRun=new Map<string,unknown>();for(const row of checkpointsQ.data??[]){const id=String(row.run_id);if(!latestCheckpointByRun.has(id))latestCheckpointByRun.set(id,row);}
  const reportByRun=new Map((reportsQ.data??[]).map(row=>[String(row.run_id),row]));
  const runViews=derived.slice(0,20).map(run=>({...run,latest_checkpoint:latestCheckpointByRun.get(run.runId)??null,report:reportByRun.get(run.runId)??null}));
  return out({status:active?"OCEAN_ACTIVE":"IDLE",observed_at:now,active_run:active,runs:runViews,reports:reportsQ.data??[],worker_runs:runsQ.data??[],treasury:treasuryQ.data??null,cloud_independent:true,shadow_only:true,live_execution:false},200,origin);
});
