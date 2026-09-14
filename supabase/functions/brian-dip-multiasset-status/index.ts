import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const ENGINE_ID="dip-multiasset-v1";
const RUN_HOURS=18;
const STATUS_VERSION="dip-v852-status-20260914.1";
const POLICY_VERSION="dip-v85-guardian-resilient-20260914.2";
const CORS={
  "access-control-allow-origin":"*",
  "access-control-allow-headers":"content-type,x-brian-dashboard-key",
  "access-control-allow-methods":"POST,OPTIONS",
  "cache-control":"no-store",
  "content-type":"application/json; charset=utf-8"
};

type J=Record<string,unknown>;
let cachedDashboardHash="",cachedDashboardHashAt=0;

async function sha256Hex(value:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(value)));return[...d].map(b=>b.toString(16).padStart(2,"0")).join("");}
function same(a:string,b:string){if(a.length!==b.length)return false;let d=0;for(let i=0;i<a.length;i++)d|=a.charCodeAt(i)^b.charCodeAt(i);return d===0;}
function num(v:unknown,f=0){const x=Number(v);return Number.isFinite(x)?x:f;}
function json(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:CORS});}
function sessionId(){const stamp=new Date().toISOString().replace(/[-:.TZ]/g,"").slice(0,14);return `dip-v852-${stamp}-${crypto.randomUUID().slice(0,8)}`;}

async function dashboardHash(){
  if(cachedDashboardHash&&Date.now()-cachedDashboardHashAt<5*60_000)return cachedDashboardHash;
  const q=await db.from("brian_dashboard_auth").select("dashboard_key_sha256").eq("auth_id","control-v3").single();
  if(q.error||!q.data)throw Error("AUTH_UNAVAILABLE");
  cachedDashboardHash=String(q.data.dashboard_key_sha256||"");cachedDashboardHashAt=Date.now();return cachedDashboardHash;
}
async function requireDashboard(req:Request){const supplied=(req.headers.get("x-brian-dashboard-key")||"").trim();if(!supplied)throw Error("UNAUTHORIZED_DASHBOARD");const expected=await dashboardHash();if(!same(await sha256Hex(supplied),expected))throw Error("UNAUTHORIZED_DASHBOARD");}

async function resetSession(startingEquity:unknown){
  const amount=Number(startingEquity);if(!Number.isFinite(amount)||amount<10||amount>1_000_000)throw Error("INVALID_TEST_CAPITAL");
  const existing=await db.from("brian_dip_multiasset_state").select("positions").eq("engine_id",ENGINE_ID).maybeSingle();if(existing.error)throw existing.error;
  const open=(existing.data?.positions??{}) as Record<string,unknown>;if(Object.keys(open).length)throw Error(`OPEN_POSITION_MULTI:${Object.keys(open).join(",")}`);
  const now=new Date(),source=sessionId(),runUntil=new Date(now.getTime()+RUN_HOURS*3600_000);
  const patch={source_session_id:source,started_at:now.toISOString(),run_until:runUntil.toISOString(),starting_equity:amount,cash:amount,realized_pnl:0,trade_count:0,win_count:0,loss_count:0,positions:{},cooldowns:{},last_scan:{status:"SESSION_RESET",source_session_id:source,policy_version:POLICY_VERSION,risk_engine:"V85_GUARDIAN",risk_mode:"COLD",risk_reason:"NEW_SESSION",standalone_guardian:true,status_version:STATUS_VERSION},last_eval_minute:null,updated_at:now.toISOString(),enabled:true,shadow_only:true,live_execution:false};
  const q=await db.from("brian_dip_multiasset_state").update(patch).eq("engine_id",ENGINE_ID).select("*").maybeSingle();if(q.error)throw q.error;
  if(q.data)return q.data;
  const ins=await db.from("brian_dip_multiasset_state").insert({engine_id:ENGINE_ID,...patch}).select("*").single();if(ins.error)throw ins.error;return ins.data;
}

async function safeEvents(started:string){
  try{const q=await db.from("brian_dip_multiasset_events").select("observed_at,symbol,action,price,qty,notional,pnl,reason,metadata").eq("engine_id",ENGINE_ID).gte("observed_at",started).order("observed_at",{ascending:false}).limit(80);if(q.error)throw q.error;return{rows:q.data??[],degraded:false};}catch{return{rows:[],degraded:true};}
}
async function safeEvaluations(started:string){
  try{const q=await db.from("brian_dip_multiasset_evaluations").select("observed_at,symbol,price,radar_score,signal_score,action,reason,metadata").eq("engine_id",ENGINE_ID).gte("observed_at",started).order("observed_at",{ascending:false}).limit(80);if(q.error)throw q.error;return{rows:q.data??[],degraded:false};}catch{return{rows:[],degraded:true};}
}

async function statusPayload(){
  const stateQ=await db.from("brian_dip_multiasset_state").select("engine_id,source_session_id,started_at,run_until,starting_equity,cash,realized_pnl,trade_count,win_count,loss_count,positions,last_scan,updated_at,enabled,shadow_only,live_execution").eq("engine_id",ENGINE_ID).maybeSingle();
  if(stateQ.error)throw stateQ.error;const s=(stateQ.data??null) as J|null;if(!s)return{status:"EMPTY",engine_id:ENGINE_ID,status_version:STATUS_VERSION,policy_version:POLICY_VERSION,shadow_only:true,live_execution:false};
  const started=String(s.started_at??"");
  const [events,evals]=await Promise.all([safeEvents(started),safeEvaluations(started)]);
  const lastScan=(s.last_scan??{}) as J,scanned=Array.isArray(lastScan.scanned)?lastScan.scanned as J[]:[];
  const markMap=new Map(scanned.map(r=>[String(r.symbol??""),num(r.price)]));
  const positions=(s.positions??{}) as Record<string,J>;
  const posRows=Object.values(positions).map(p=>{const symbol=String(p.symbol??""),entry=num(p.entry),qty=num(p.qty),mark=markMap.get(symbol)??entry;return{symbol,entry,mark,qty,opened_at:p.opened_at??null,stop:num(p.stop),target:num(p.target),trail:num(p.trail),max_price:num(p.max_price),radar_score:num(p.radar_score),unrealized_pnl:qty*(mark-entry),entry_reason:String(p.entry_reason??"")};});
  const cash=num(s.cash),starting=num(s.starting_equity,1000),openValue=posRows.reduce((a,p)=>a+p.qty*p.mark,0),equity=cash+openValue,realized=num(s.realized_pnl),wins=num(s.win_count),losses=num(s.loss_count),trades=num(s.trade_count),closed=wins+losses;
  const updated=String(s.updated_at??""),age=updated?Math.max(0,(Date.now()-Date.parse(updated))/1000):null;
  const runUntil=Date.parse(String(s.run_until??"")),windowDone=Number.isFinite(runUntil)&&Date.now()>=runUntil;
  const status=s.enabled===true&&!windowDone?(age!==null&&age<60?"RUNNING":"STALE"):"STOPPED";
  const fallbackEvaluations=scanned.map(r=>({observed_at:updated,symbol:r.symbol,price:r.price,radar_score:r.radar_score,signal_score:r.signal_score,action:r.ready===true?"READY":"WATCH",reason:r.reason,metadata:r}));
  const recentEvaluations=evals.rows.length?evals.rows:fallbackEvaluations;
  const degradedSources:string[]=[];if(events.degraded)degradedSources.push("event_history");if(evals.degraded)degradedSources.push("evaluation_history");
  return{
    status,engine_id:ENGINE_ID,status_version:STATUS_VERSION,source_session_id:s.source_session_id??null,started_at:s.started_at,run_until:s.run_until,updated_at:updated,age_seconds:age,
    starting_equity:starting,cash,equity,realized_pnl:realized,trade_count:trades,win_count:wins,loss_count:losses,win_rate:closed?wins/closed:null,
    positions:posRows,recent_events:events.rows,recent_evaluations:recentEvaluations,last_scan:lastScan,degraded_sources:degradedSources,
    policy_version:String(lastScan.policy_version??POLICY_VERSION),risk_engine:String(lastScan.risk_engine??"V85_GUARDIAN"),risk_mode:String(lastScan.risk_mode??"COLD"),risk_reason:String(lastScan.risk_reason??"CALIBRATING"),
    radar_age_seconds:num(lastScan.radar_age_seconds),radar_soft_stale:lastScan.radar_soft_stale===true,drawdown_pct:num(lastScan.drawdown_pct),gross_cap_pct:num(lastScan.gross_cap_pct),risk_cap_pct:num(lastScan.risk_cap_pct),max_positions:num(lastScan.max_positions,2),loss_streak:num(lastScan.loss_streak),
    shadow_only:s.shadow_only!==false,live_execution:s.live_execution===true,standalone_guardian:true
  };
}

Deno.serve(async(req:Request)=>{
  if(req.method==="OPTIONS")return new Response(null,{status:204,headers:CORS});if(req.method!=="POST")return json({error:"METHOD_NOT_ALLOWED"},405);
  try{
    await requireDashboard(req);let body:J={};try{body=await req.json();}catch{}
    const action=String(body.action??"status").toLowerCase();
    if(action==="restart"){const row=await resetSession(body.starting_equity);return json({status:"STARTED",engine_id:ENGINE_ID,status_version:STATUS_VERSION,source_session_id:row.source_session_id,started_at:row.started_at,run_until:row.run_until,starting_equity:num(row.starting_equity),policy_version:POLICY_VERSION,risk_engine:"V85_GUARDIAN",risk_mode:"COLD",shadow_only:true,live_execution:false,standalone_guardian:true});}
    if(action!=="status")return json({error:"ACTION_NOT_ALLOWED"},400);return json(await statusPayload());
  }catch(e){const message=e instanceof Error?e.message:String(e);const status=message.includes("UNAUTHORIZED")?401:message.includes("INVALID_")||message.includes("OPEN_POSITION_MULTI")?409:500;return json({status:"ERROR",error:message,engine_id:ENGINE_ID,status_version:STATUS_VERSION,shadow_only:true,live_execution:false},status);}
});
