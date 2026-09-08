import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import {
  ENGINE_VERSION,
  METRIC_VERSION,
  POLICY_VERSION,
  validateSession,
} from "../_shared/dip_v8_dual.ts";

const URL=Deno.env.get("SUPABASE_URL")!,KEY=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,KEY,{auth:{persistSession:false,autoRefreshToken:false}});
const EVIDENCE="AGGRESSIVE_DIP_SHADOW";
const EXACT=new Set([
  "https://monster-coins-pro-seven.vercel.app",
  "https://monster-coins-pro-oemer-yildirim.vercel.app",
  "http://localhost:3000","http://127.0.0.1:3000",
]);
const ORIGIN=/^https:\/\/monster-coins(?:-pro)?-[a-z0-9-]*oemer-yildirim\.vercel\.app$/i;

type Row={event_id:string;session_id:string;event_kind:"START"|"PAUSE";requested_at:string;starting_equity:number|string|null;trade_notional:number|string|null;config:Record<string,unknown>|null;engine_token_sha256:string|null};
type Session={start:Row;last:Row;active:boolean;ended_at:string|null};

function cors(o:string|null){const a=o&&(EXACT.has(o)||ORIGIN.test(o))?o:"https://monster-coins-pro-seven.vercel.app";return{"access-control-allow-origin":a,"access-control-allow-headers":"content-type,x-brian-dashboard-key,authorization,apikey","access-control-allow-methods":"POST,OPTIONS","vary":"Origin"};}
function out(x:unknown,status=200,o:string|null=null){return new Response(JSON.stringify(x),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store",...cors(o)}});}
async function sha(s:string){const b=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(s)));return[...b].map(x=>x.toString(16).padStart(2,"0")).join("");}
function same(a:string,b:string){if(a.length!==b.length)return false;let d=0;for(let i=0;i<a.length;i++)d|=a.charCodeAt(i)^b.charCodeAt(i);return d===0;}
async function auth(req:Request){const k=(req.headers.get("x-brian-dashboard-key")||"").trim();if(!k)throw Error("UNAUTHORIZED_DASHBOARD");const q=await db.from("brian_dashboard_auth").select("dashboard_key_sha256").order("created_at",{ascending:false}).limit(1).maybeSingle();if(q.error||!q.data||!same(await sha(k),String(q.data.dashboard_key_sha256||"")))throw Error("UNAUTHORIZED_DASHBOARD");}
function num(v:unknown,name:string){const x=Number(v);if(!Number.isFinite(x))throw Error("INVALID_"+name);return x;}
function object(v:unknown,name:string,max:number){if(!v||typeof v!=="object"||Array.isArray(v))throw Error("INVALID_"+name);const s=JSON.stringify(v);if(new TextEncoder().encode(s).byteLength>max)throw Error(name+"_TOO_LARGE");return v as Record<string,unknown>;}

async function latest():Promise<Session|null>{
  const q=await db.from("brian_dip_session_events").select("event_id,session_id,event_kind,requested_at,starting_equity,trade_notional,config,engine_token_sha256").order("requested_at",{ascending:false}).order("event_id",{ascending:false}).limit(1).maybeSingle();
  if(q.error)throw q.error;if(!q.data)return null;const last=q.data as Row;
  const z=await db.from("brian_dip_session_events").select("event_id,session_id,event_kind,requested_at,starting_equity,trade_notional,config,engine_token_sha256").eq("session_id",last.session_id).eq("event_kind","START").order("requested_at",{ascending:true}).limit(1).single();
  if(z.error||!z.data)throw Error("DIP_START_MISSING");return{start:z.data as Row,last,active:last.event_kind==="START",ended_at:last.event_kind==="PAUSE"?last.requested_at:null};
}

function settings(b:Record<string,unknown>){
  const starting=num(b.starting_equity??500,"STARTING_EQUITY"),trade=num(b.trade_notional??starting,"TRADE_NOTIONAL");
  if(starting<=0||starting>1e6)throw Error("STARTING_EQUITY_OUT_OF_RANGE");if(trade<=0||trade>starting)throw Error("TRADE_NOTIONAL_OUT_OF_RANGE");
  const i=object(b.config??{},"CONFIG",30000);
  const n=(k:string,f:number,min:number,max:number)=>{const v=num(i[k]??f,k.toUpperCase());if(v<min||v>max)throw Error(k.toUpperCase()+"_OUT_OF_RANGE");return v;};
  const mode=String(i.execution_mode??"SHADOW_PAPER");
  const config={
    symbols:["ETHUSDT"],interval:"1m",
    step_pct:n("step_pct",.15,.01,10),dip_trigger_steps:n("dip_trigger_steps",2,.25,100),rebound_steps:n("rebound_steps",1,.1,100),take_profit_steps:n("take_profit_steps",3,.1,200),max_chase_steps:n("max_chase_steps",5,.25,500),
    fee_bps:n("fee_bps",10,0,500),slippage_bps:n("slippage_bps",1,0,500),expert_mode:true,auto_universe:false,universe_size:1,
    engine_version:ENGINE_VERSION,policy_version:POLICY_VERSION,measurement:METRIC_VERSION,
    shadow_only:true,live_execution:false,server_authoritative:true,browser_execution:false,
    allow_shadow_short:true,max_shadow_leverage:2,execution_mode:mode,
    sizing_policy:"V83_RISK_BUDGETED_USABLE_CAPITAL",leverage_policy:"1X_BASE__2X_ONLY_AFTER_40_CALIBRATED_EDGE",chart_reader_version:"v8.3-dual",
    decision_cadence_seconds:60,market_source:"BINANCE_USDM_PERP",max_account_risk_fraction:.005,
  };
  validateSession(config);return{starting,trade,config};
}

async function status(){
  const s=await latest(),base={schema_version:"brian.dip.status.v8.3",generated_at:new Date().toISOString(),evidence_class:EVIDENCE,policy_version:POLICY_VERSION,shadow_only:true,live_execution:false,dual_direction:true,rootfix_version:"dip-v8.3-rootfix-20260908.3"};
  if(!s)return{...base,session:null,snapshot:null,events:[],engine_lease:null};
  const isDual=s.start.config?.policy_version===POLICY_VERSION;
  if(!isDual)return{...base,status:"WAIT_V8_DUAL_CLEAN_RESTART",session:{session_id:s.start.session_id,status:s.active?"RUNNING":"PAUSED",started_at:s.start.requested_at,ended_at:s.ended_at,starting_equity:Number(s.start.starting_equity),trade_notional:Number(s.start.trade_notional),config:s.start.config??{}},snapshot:null,events:[],engine_lease:null};
  const[z,e]=await Promise.all([
    db.from("brian_dip_v8_runtime").select("snapshot,runtime,state_version").eq("session_id",s.start.session_id).maybeSingle(),
    db.from("brian_dip_v8_ledger").select("transition_id,recorded_at,event_kind,payload").eq("session_id",s.start.session_id).in("event_kind",["BUY","SELL","SHORT_OPEN","SHORT_CLOSE"]).order("recorded_at",{ascending:false}).limit(160),
  ]);
  if(z.error)throw z.error;if(e.error)throw e.error;if(!z.data)throw Error("V8_STATE_MISSING_RECONCILE");
  const events=(e.data??[]).map(x=>({...x.payload,event_id:x.transition_id,session_id:s.start.session_id,observed_at:x.recorded_at,symbol:"ETHUSDT"}));
  return{...base,status:"OK",session:{session_id:s.start.session_id,status:s.active?"RUNNING":"PAUSED",started_at:s.start.requested_at,ended_at:s.ended_at,starting_equity:Number(s.start.starting_equity),trade_notional:Number(s.start.trade_notional),config:s.start.config??{}},snapshot:z.data.snapshot??null,events,engine_lease:null};
}

async function begin(b:Record<string,unknown>,restart=false){
  const{starting,trade,config}=settings(b);const previous=await latest();
  if(!restart&&previous?.active&&previous.start.config?.policy_version===POLICY_VERSION&&previous.start.config?.chart_reader_version==="v8.3-dual")return{status:"RESUMED",session_id:previous.start.session_id,started_at:previous.start.requested_at,starting_equity:Number(previous.start.starting_equity),trade_notional:Number(previous.start.trade_notional),config:previous.start.config,shadow_only:true,live_execution:false};
  const token=String(b.engine_token??"").trim();if(token.length<20||token.length>200)throw Error("INVALID_ENGINE_TOKEN");
  const id=`dip-v83-${new Date().toISOString().replace(/[-:.TZ]/g,"").slice(0,14)}-${crypto.randomUUID().slice(0,8)}`,h=await sha(token);
  const q=restart||previous?.active
    ?await db.rpc("brian_dip_restart_session",{p_pause_event_id:`dip-evt-${crypto.randomUUID()}`,p_start_event_id:`dip-evt-${crypto.randomUUID()}`,p_new_session_id:id,p_starting_equity:starting,p_trade_notional:trade,p_config:config,p_engine_token_sha256:h})
    :await db.rpc("brian_dip_start_session",{p_event_id:`dip-evt-${crypto.randomUUID()}`,p_session_id:id,p_starting_equity:starting,p_trade_notional:trade,p_config:config,p_engine_token_sha256:h});
  if(q.error)throw q.error;const row=Array.isArray(q.data)?q.data.at(-1):q.data;
  return{status:restart||previous?.active?"RESTARTED":"STARTED",session_id:String(row?.session_id??id),started_at:row?.requested_at??new Date().toISOString(),starting_equity:starting,trade_notional:trade,config,shadow_only:true,live_execution:false,dual_direction:true};
}
async function pause(){const s=await latest();if(!s||!s.active)return{status:"ALREADY_PAUSED",shadow_only:true,live_execution:false};const q=await db.rpc("brian_dip_pause_session",{p_event_id:`dip-evt-${crypto.randomUUID()}`,p_session_id:s.start.session_id});if(q.error)throw q.error;return{status:"PAUSED",session_id:s.start.session_id,shadow_only:true,live_execution:false};}

Deno.serve(async(req:Request)=>{
  const o=req.headers.get("origin");if(req.method==="OPTIONS")return new Response("ok",{headers:cors(o)});if(req.method!=="POST")return out({status:"METHOD_NOT_ALLOWED"},405,o);
  try{await auth(req);const b=await req.json().catch(()=>({})) as Record<string,unknown>,a=String(b.action??"status").toLowerCase();if(a==="status")return out(await status(),200,o);if(a==="start")return out(await begin(b,false),200,o);if(a==="restart")return out(await begin(b,true),200,o);if(a==="pause")return out(await pause(),200,o);if(a==="claim_engine"||a==="engine_check")return out({status:"SERVER_AUTHORITATIVE_VIEW_ONLY",browser_execution:false,shadow_only:true,live_execution:false},200,o);if(a==="event"||a==="snapshot")return out({status:"SERVER_AUTHORITATIVE_VIEW_ONLY",browser_execution:false,shadow_only:true,live_execution:false},200,o);return out({status:"UNKNOWN_ACTION"},400,o);
  }catch(e){const message=e instanceof Error?e.message:String(e);return out({status:"FAILED_CLOSED",error:message,policy_version:POLICY_VERSION,shadow_only:true,live_execution:false},message.includes("UNAUTHORIZED")?401:400,o);}
});
