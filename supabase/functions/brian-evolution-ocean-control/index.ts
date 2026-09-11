import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { activeOceanRun, BRIAN_OCEAN_VERSION, deriveOceanRuns, type OceanCommand } from "../_shared/evolution_ocean.ts";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const AUTH_ID="control-v3";
const ALLOWED_ORIGIN=/^https:\/\/monster-coins(?:-pro)?-[a-z0-9-]*oemer-yildirim\.vercel\.app$/i;
const ALLOWED_EXACT=new Set(["https://monster-coins-pro-seven.vercel.app","https://monster-coins-pro-oemer-yildirim.vercel.app","https://monster-coins-pro-git-brian-2026-oemer-yildirim.vercel.app","http://localhost:3000","http://127.0.0.1:3000"]);

function cors(origin?:string|null):Record<string,string>{const allowed=origin&&(ALLOWED_EXACT.has(origin)||ALLOWED_ORIGIN.test(origin))?origin:"https://monster-coins-pro-oemer-yildirim.vercel.app";return{"access-control-allow-origin":allowed,"access-control-allow-headers":"content-type,x-brian-dashboard-key","access-control-allow-methods":"POST,OPTIONS","vary":"Origin"};}
function out(body:unknown,status=200,origin?:string|null){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store",...cors(origin)}});}
async function sha(value:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(value)));return[...d].map(b=>b.toString(16).padStart(2,"0")).join("");}
async function sha256Hex(value:string){return await sha(value);}
function constantTimeEqual(left:string,right:string){if(left.length!==right.length)return false;let diff=0;for(let i=0;i<left.length;i++)diff|=left.charCodeAt(i)^right.charCodeAt(i);return diff===0;}
async function requireDashboardAuth(req:Request){const supplied=(req.headers.get("x-brian-dashboard-key")??"").trim();if(!supplied)throw new Error("UNAUTHORIZED_DASHBOARD");const q=await db.from("brian_dashboard_auth").select("dashboard_key_sha256").eq("auth_id",AUTH_ID).single();if(q.error||!q.data)throw new Error("DASHBOARD_AUTH_UNAVAILABLE");if(!constantTimeEqual(await sha256Hex(supplied),String(q.data.dashboard_key_sha256??"")))throw new Error("UNAUTHORIZED_DASHBOARD");}
function errorText(error:unknown){return error instanceof Error?`${error.name}: ${error.message}`:String(error);}

async function loadCommands():Promise<OceanCommand[]>{
  const q=await db.from("brian_ocean_run_commands").select("command_id,run_id,command,requested_at,duration_hours,reason").order("requested_at",{ascending:false}).limit(300);
  if(q.error)throw new Error(`ocean_commands:${q.error.message}`);
  return(q.data??[]).map(row=>({commandId:String(row.command_id),runId:String(row.run_id),command:String(row.command) as OceanCommand["command"],requestedAt:String(row.requested_at),durationHours:row.duration_hours==null?null:Number(row.duration_hours) as 24|48,reason:row.reason==null?null:String(row.reason)}));
}

async function preflight(){
  const [treasuryQ,edgeQ,collectorQ]=await Promise.all([
    db.from("brian_treasury_shadow_latest").select("snapshot_id,observed_at,equity_usd,promotion_gate_open").limit(1).maybeSingle(),
    db.from("brian_alpha_expected_edge_latest_by_asset").select("edge_id,observed_at").order("observed_at",{ascending:false}).limit(1).maybeSingle(),
    db.from("brian_collector_runs").select("collector_id,status,started_at").order("started_at",{ascending:false}).limit(300),
  ]);
  const errors=[treasuryQ.error,edgeQ.error,collectorQ.error].filter(Boolean).map(error=>error!.message);
  if(errors.length)return{ready:false,reasons:errors,treasury:null,edge:null};
  const latestByCollector=new Map<string,{status:string;started_at:string}>();
  for(const row of collectorQ.data??[]){const id=String(row.collector_id);if(!latestByCollector.has(id))latestByCollector.set(id,{status:String(row.status),started_at:String(row.started_at)});}
  const required=["brian-evolution-orchestrator-v1","brian-evolution-researcher-v1","brian-evolution-alpha-edge-challenger-v1","brian-evolution-treasury-v1"];
  const reasons:string[]=[];
  if(!treasuryQ.data)reasons.push("Treasury has no persisted cycle");
  if(!edgeQ.data)reasons.push("Layer-4 expected-edge has no persisted observation");
  for(const id of required){const latest=latestByCollector.get(id);if(!latest)reasons.push(`${id} has no collector receipt`);else if(latest.status!=="SUCCESS")reasons.push(`${id} latest status is ${latest.status}`);}
  return{ready:reasons.length===0,reasons,treasury:treasuryQ.data??null,edge:edgeQ.data??null};
}

Deno.serve(async(req:Request)=>{
  const origin=req.headers.get("origin");if(req.method==="OPTIONS")return new Response(null,{status:204,headers:cors(origin)});if(req.method!=="POST")return out({error:"POST required"},405,origin);
  try{await requireDashboardAuth(req);}catch(error){return out({error:errorText(error)},401,origin);}
  let body:Record<string,unknown>={};try{body=await req.json();}catch{return out({error:"invalid JSON"},400,origin);}
  const action=String(body.action??"").toUpperCase();
  try{
    const now=new Date().toISOString(),commands=await loadCommands(),active=activeOceanRun(commands,now);
    if(action==="START"){
      if(active)return out({status:"ALREADY_ACTIVE",active,shadow_only:true,live_execution:false},409,origin);
      const duration=Number(body.duration_hours);if(duration!==24&&duration!==48)return out({error:"duration_hours must be 24 or 48"},400,origin);
      if(body.confirm_shadow_only!==true)return out({error:"confirm_shadow_only=true required"},400,origin);
      const readiness=await preflight();if(!readiness.ready)return out({status:"PREFLIGHT_BLOCKED",preflight:readiness,shadow_only:true,live_execution:false},409,origin);
      const runId=await sha(`ocean|${now}|${duration}|${crypto.randomUUID()}`),commandId=await sha(`ocean-command|${runId}|START|${now}`);
      const q=await db.from("brian_ocean_run_commands").insert({command_id:commandId,run_id:runId,command:"START",requested_at:now,duration_hours:duration,reason:body.reason==null?null:String(body.reason).slice(0,500),requested_by:"dashboard",metadata:{ocean_version:BRIAN_OCEAN_VERSION,preflight:readiness},shadow_only:true,live_execution:false});
      if(q.error)throw new Error(`ocean_start:${q.error.message}`);
      const state=deriveOceanRuns([...(commands),{commandId,runId,command:"START",requestedAt:now,durationHours:duration as 24|48,reason:body.reason==null?null:String(body.reason)}],now).find(row=>row.runId===runId);
      return out({status:"STARTED",run:state,preflight:readiness,cloud_independent:true,shadow_only:true,live_execution:false},200,origin);
    }
    if(action==="STOP"){
      if(!active)return out({status:"NO_ACTIVE_RUN",shadow_only:true,live_execution:false},409,origin);
      const commandId=await sha(`ocean-command|${active.runId}|STOP|${now}`);
      const q=await db.from("brian_ocean_run_commands").insert({command_id:commandId,run_id:active.runId,command:"STOP",requested_at:now,duration_hours:null,reason:body.reason==null?"operator stop":String(body.reason).slice(0,500),requested_by:"dashboard",metadata:{ocean_version:BRIAN_OCEAN_VERSION},shadow_only:true,live_execution:false});
      if(q.error)throw new Error(`ocean_stop:${q.error.message}`);
      return out({status:"STOPPED",run_id:active.runId,stopped_at:now,shadow_only:true,live_execution:false},200,origin);
    }
    if(action==="PREFLIGHT")return out({status:"PREFLIGHT",preflight:await preflight(),active,shadow_only:true,live_execution:false},200,origin);
    return out({error:"action must be START, STOP or PREFLIGHT"},400,origin);
  }catch(error){return out({status:"FAILED",error:errorText(error),shadow_only:true,live_execution:false},500,origin);}
});
