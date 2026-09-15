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
function cors(origin: string | null) {
  const allowed = origin && (ALLOWED_EXACT.has(origin) || ALLOWED_ORIGIN.test(origin)) ? origin : "https://monster-coins-pro-oemer-yildirim.vercel.app";
  return {"access-control-allow-origin":allowed,"access-control-allow-headers":"content-type,x-brian-dashboard-key","access-control-allow-methods":"POST,OPTIONS","cache-control":"no-store","vary":"Origin"};
}
function json(body: unknown,status: number,origin: string|null){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8",...cors(origin)}})}
async function sha256Hex(value:string){const digest=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(value)));return [...digest].map(b=>b.toString(16).padStart(2,"0")).join("")}
function constantTimeEqual(a:string,b:string){if(a.length!==b.length)return false;let diff=0;for(let i=0;i<a.length;i++)diff|=a.charCodeAt(i)^b.charCodeAt(i);return diff===0}
async function requireDashboard(req:Request){const supplied=(req.headers.get("x-brian-dashboard-key")??"").trim();if(!supplied)throw new Error("UNAUTHORIZED_DASHBOARD");const q=await db.from("brian_dashboard_auth").select("dashboard_key_sha256").eq("auth_id",AUTH_ID).single();if(q.error||!q.data)throw new Error("AUTH_UNAVAILABLE");if(!constantTimeEqual(await sha256Hex(supplied),String(q.data.dashboard_key_sha256)))throw new Error("UNAUTHORIZED_DASHBOARD")}
const txt=(v:unknown,max=4000)=>{const s=String(v??"").trim();return s?s.slice(0,max):null};
const num=(v:unknown)=>{const n=Number(v);return Number.isFinite(n)?n:null};
const bool=(v:unknown)=>v===true?true:v===false?false:null;
const normAsset=(v:unknown)=>String(v??"").replace(/^crypto:/i,"").replace(/[^A-Z0-9]/gi,"").toUpperCase().slice(0,32)||null;
function iso(v:unknown){const t=typeof v==="number"?v:Date.parse(String(v??""));if(!Number.isFinite(t))return null;return new Date(t).toISOString()}
async function matchingTrade(asset:string|null,eventTime:string){if(!asset)return null;const from=new Date(Date.parse(eventTime)-5*60_000).toISOString();const to=new Date(Date.parse(eventTime)+12*60*60_000).toISOString();const q=await db.from("brian_treasury_shadow_actions").select("action_id,observed_at,kind,asset_id,direction,capital_usd,reference_price,reason,shadow_only,live_execution").eq("asset_id",asset).gte("observed_at",from).lte("observed_at",to).order("observed_at",{ascending:false}).limit(1).maybeSingle();if(q.error)throw q.error;return q.data??null}
async function upsertRecord(raw:Record<string,unknown>){
  const eventKey=txt(raw.event_key,500),eventTime=iso(raw.event_time),title=txt(raw.title,1000);if(!eventKey||!eventTime||!title)throw new Error("INVALID_EVENT_RECORD");
  const urgencyRaw=String(raw.urgency??"HIGH").toUpperCase();const urgency=["CRITICAL","HIGH","MEDIUM","LOW"].includes(urgencyRaw)?urgencyRaw:"HIGH";
  const alphaAsset=normAsset(raw.alpha_asset),asset=normAsset(raw.asset);const trade=await matchingTrade(alphaAsset||asset,eventTime);const statusInput=txt(raw.status,120)||"GÖZLEMLENİYOR";const status=trade?.kind==="EXIT"?"KAPANDI":trade?.kind==="OPEN"?"İŞLEM AÇILDI":statusInput;
  const row={event_key:eventKey,event_time:eventTime,first_seen_at:iso(raw.first_seen_at)||new Date().toISOString(),last_seen_at:new Date().toISOString(),urgency,importance:num(raw.importance),title,summary:txt(raw.summary),original_claim:txt(raw.original_claim),event_kind:txt(raw.event_kind,200),source:txt(raw.source,300),publisher:txt(raw.publisher,300),source_uri:txt(raw.source_uri,3000),source_trust:txt(raw.source_trust,300),source_verified:raw.source_verified===true,asset,alpha_asset:alphaAsset,alpha_action:txt(raw.alpha_action,120),alpha_evidence:num(raw.alpha_evidence),council_decision:txt(raw.council_decision),treasury_gate:bool(raw.treasury_gate),treasury_reason:txt(raw.treasury_reason),status,trade_kind:trade?txt(trade.kind,80):null,trade_action_id:trade?txt(trade.action_id,300):null,trade_observed_at:trade?iso(trade.observed_at):null,trade_reference_price:trade?num(trade.reference_price):null,trade_capital_usd:trade?num(trade.capital_usd):null,trade_reason:trade?txt(trade.reason):null,payload:raw.payload&&typeof raw.payload==="object"?raw.payload:{},evidence_class:"PROSPECTIVE_MEETING_SHADOW",shadow_only:true,live_execution:false,updated_at:new Date().toISOString()};
  const q=await db.from("brian_meeting_shadow_ledger").upsert(row,{onConflict:"event_key"}).select("*").single();if(q.error)throw q.error;return q.data;
}
Deno.serve(async(req:Request)=>{const origin=req.headers.get("origin");if(req.method==="OPTIONS")return new Response("ok",{headers:cors(origin)});if(req.method!=="POST")return json({status:"METHOD_NOT_ALLOWED"},405,origin);try{await requireDashboard(req);const body=await req.json().catch(()=>({})) as Record<string,unknown>;const action=String(body.action??"upsert").toLowerCase();if(action==="upsert"){const record=body.record&&typeof body.record==="object"?body.record as Record<string,unknown>:{};return json({status:"UPSERTED",record:await upsertRecord(record),shadow_only:true,live_execution:false},200,origin)}if(action==="list"){const limit=Math.min(80,Math.max(1,Number(body.limit??40)||40));const q=await db.from("brian_meeting_shadow_ledger").select("*").order("event_time",{ascending:false}).limit(limit);if(q.error)throw q.error;return json({status:"OK",records:q.data??[],shadow_only:true,live_execution:false},200,origin)}return json({status:"UNKNOWN_ACTION"},400,origin)}catch(e){const message=e instanceof Error?e.message:String(e);const unauthorized=message.includes("UNAUTHORIZED");console.error("brian-meeting-shadow-ledger",message);return json({status:unauthorized?"UNAUTHORIZED":"FAILED_CLOSED",error:message,shadow_only:true,live_execution:false},unauthorized?401:500,origin)}});