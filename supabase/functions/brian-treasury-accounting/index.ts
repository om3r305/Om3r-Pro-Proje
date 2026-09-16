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

function cors(origin?: string | null): Record<string, string> {
  const allowed = origin && (ALLOWED_EXACT.has(origin) || ALLOWED_ORIGIN.test(origin))
    ? origin
    : "https://monster-coins-pro-oemer-yildirim.vercel.app";
  return {
    "access-control-allow-origin": allowed,
    "access-control-allow-headers": "content-type,x-brian-dashboard-key",
    "access-control-allow-methods": "POST,OPTIONS",
    "vary": "Origin",
  };
}
function out(body: unknown, status = 200, origin?: string | null) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...cors(origin) },
  });
}
async function sha256Hex(value: string) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)));
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}
function constantTimeEqual(left: string, right: string) {
  if (left.length !== right.length) return false;
  let diff = 0;
  for (let index = 0; index < left.length; index++) diff |= left.charCodeAt(index) ^ right.charCodeAt(index);
  return diff === 0;
}
async function requireDashboardAuth(req: Request) {
  const supplied = (req.headers.get("x-brian-dashboard-key") ?? "").trim();
  if (!supplied) throw new Error("UNAUTHORIZED_DASHBOARD");
  const q = await db.from("brian_dashboard_auth").select("dashboard_key_sha256").eq("auth_id", AUTH_ID).single();
  if (q.error || !q.data) throw new Error("DASHBOARD_AUTH_UNAVAILABLE");
  if (!constantTimeEqual(await sha256Hex(supplied), String(q.data.dashboard_key_sha256 ?? ""))) throw new Error("UNAUTHORIZED_DASHBOARD");
}

Deno.serve(async(req:Request)=>{
 const origin=req.headers.get("origin");
 if(req.method==="OPTIONS")return new Response(null,{status:204,headers:cors(origin)});
 if(req.method!=="POST")return out({error:"POST required"},405,origin);
 try{await requireDashboardAuth(req)}catch{return out({error:"UNAUTHORIZED_DASHBOARD"},401,origin)}
 try{
  const snapCols="snapshot_id,cycle_id,observed_at,starting_equity_usd,cash_usd,equity_usd,realized_pnl_usd,cumulative_costs_usd,deployment_usd,positions,treasury_version";
  const actionCols="action_id,position_id,cycle_id,observed_at,kind,asset_id,direction,capital_usd,reference_price,cost_usd,reason,source_decision_id";
  const latest=await db.from("brian_treasury_shadow_snapshots").select(snapCols).lte("observed_at",new Date().toISOString()).order("observed_at",{ascending:false}).limit(1).maybeSingle();
  if(latest.error)throw latest.error;
  if(!latest.data)return out({observed_at:new Date().toISOString(),snapshot:null,actions:[],opening_actions:[],history:[],coverage:{actions_truncated:false,history_truncated:false},shadow_only:true},200,origin);
  const cutoff=latest.data.observed_at;
  const [aq,hq]=await Promise.all([
   db.from("brian_treasury_shadow_actions").select(actionCols).lte("observed_at",cutoff).order("observed_at",{ascending:false}).order("action_id",{ascending:false}).limit(1000),
   db.from("brian_treasury_shadow_snapshots").select("snapshot_id,observed_at,starting_equity_usd,equity_usd,cash_usd,realized_pnl_usd,cumulative_costs_usd").lte("observed_at",cutoff).order("observed_at",{ascending:false}).limit(1000)
  ]);
  if(aq.error||hq.error)throw aq.error||hq.error;
  const actions=aq.data||[],history=hq.data||[],known=new Set(actions.filter(a=>a.kind==="OPEN").map(a=>a.position_id));
  const missing=[...new Set(actions.filter(a=>a.kind==="EXIT"&&a.position_id&&!known.has(a.position_id)).map(a=>a.position_id))];
  const openings:any[]=[];
  for(let i=0;i<missing.length;i+=100){const q=await db.from("brian_treasury_shadow_actions").select(actionCols).eq("kind","OPEN").in("position_id",missing.slice(i,i+100)).lte("observed_at",cutoff).limit(1000);if(q.error)throw q.error;openings.push(...(q.data||[]));}
  return out({observed_at:new Date().toISOString(),snapshot:latest.data,actions,opening_actions:openings,history:history.reverse(),coverage:{actions_truncated:actions.length===1000,history_truncated:history.length===1000,action_limit:1000,history_limit:1000,as_of:cutoff},shadow_only:true,live_execution:false},200,origin);
 }catch{return out({error:"ACCOUNTING_DATA_UNAVAILABLE"},503,origin)}
});
