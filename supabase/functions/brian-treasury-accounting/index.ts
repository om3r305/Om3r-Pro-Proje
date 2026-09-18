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
    "access-control-allow-headers": "content-type,x-brian-dashboard-key,x-brian-cron-key",
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
async function requireAccountingAuth(req: Request): Promise<"dashboard"|"cron"> {
  const supplied = (req.headers.get("x-brian-dashboard-key") ?? req.headers.get("x-brian-cron-key") ?? "").trim();
  if (!supplied) throw new Error("UNAUTHORIZED_DASHBOARD");
  const q = await db.from("brian_dashboard_auth").select("dashboard_key_sha256,cron_key_sha256").eq("auth_id", AUTH_ID).single();
  if (q.error || !q.data) throw new Error("DASHBOARD_AUTH_UNAVAILABLE");
  const digest = await sha256Hex(supplied);
  if (constantTimeEqual(digest, String(q.data.dashboard_key_sha256 ?? ""))) return "dashboard";
  if (constantTimeEqual(digest, String(q.data.cron_key_sha256 ?? ""))) return "cron";
  throw new Error("UNAUTHORIZED_DASHBOARD");
}
function errorText(error: unknown) {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  try { return JSON.stringify(error); } catch { return String(error); }
}

Deno.serve(async(req:Request)=>{
 const origin=req.headers.get("origin");
 if(req.method==="OPTIONS")return new Response(null,{status:204,headers:cors(origin)});
 if(req.method!=="POST")return out({error:"POST required"},405,origin);
 let authKind:"dashboard"|"cron";
 try{authKind=await requireAccountingAuth(req)}catch{return out({error:"UNAUTHORIZED_DASHBOARD"},401,origin)}
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
  const positions=Array.isArray(latest.data.positions)?latest.data.positions:[];
  const missing=[...new Set(actions.filter(a=>a.kind==="EXIT"&&a.position_id&&!known.has(a.position_id)).map(a=>a.position_id))];
  const openings:any[]=[];
  for(let i=0;i<missing.length;i+=40){const ids=missing.slice(i,i+40);const q=await db.from("brian_treasury_shadow_actions").select(actionCols).eq("kind","OPEN").in("position_id",ids).lte("observed_at",cutoff).limit(ids.length);if(q.error)throw q.error;openings.push(...(q.data||[]));}

  const decisionIds=[...new Set([
    ...positions.map((p:any)=>String(p?.sourceDecisionId??p?.source_decision_id??"")),
    ...actions.slice(0,250).map((a:any)=>String(a?.source_decision_id??"")),
    ...openings.map((a:any)=>String(a?.source_decision_id??""))
  ].filter(Boolean))];
  let decisions:any[]=[];
  const decisionCols="decision_id,observed_at,asset_id,action,direction,evidence_score,independent_group_count,requested_virtual_notional_usd,estimated_round_trip_cost_bps,reason,veto_reason,support_groups,conflict_groups,metadata";
  for(let i=0;i<Math.min(decisionIds.length,500);i+=40){
    const ids=decisionIds.slice(i,Math.min(i+40,500));
    const q=await db.from("brian_alpha_decisions").select(decisionCols).in("decision_id",ids).limit(ids.length);
    if(q.error)throw q.error;
    decisions.push(...(q.data||[]));
  }
  const decisionById=new Map(decisions.map((d:any)=>[String(d.decision_id),d]));
  const evidenceIds=[...new Set(decisions.flatMap((d:any)=>Array.isArray(d?.metadata?.source_evidence_ids_all)?d.metadata.source_evidence_ids_all.map(String):[]).filter(Boolean))];
  let sensors:any[]=[];
  for(let i=0;i<evidenceIds.length;i+=40){
    const ids=evidenceIds.slice(i,i+40);
    const q=await db.from("brian_sensor_observations")
      .select("observation_id,asset_id,observed_at,direction,strength,confidence,reliability,independent_group,sensor_family,reason,source_ids,metadata")
      .in("observation_id",ids).limit(ids.length);
    if(q.error)throw q.error;
    sensors.push(...(q.data||[]));
  }
  const sensorById=new Map(sensors.map((s:any)=>[String(s.observation_id),s]));
  const assetIds=[...new Set(positions.map((p:any)=>String(p?.assetId??p?.asset_id??"")).filter(Boolean))];
  let ticks:any[]=[];
  if(assetIds.length){
    const since=new Date(Date.now()-30*60_000).toISOString();
    const q=await db.from("brian_micro_book_ticks")
      .select("asset_id,observed_at,observed_mid_price")
      .in("asset_id",assetIds).gte("observed_at",since)
      .order("observed_at",{ascending:false}).limit(1500);
    if(q.error)throw q.error;ticks=q.data||[];
  }
  const markByAsset=new Map<string,any>();
  for(const t of ticks){const a=String(t.asset_id);if(!markByAsset.has(a))markByAsset.set(a,t);}
  const openActionByPosition=new Map<string,any>();
  for(const a of [...actions,...openings])if(a.kind==="OPEN"&&a.position_id&&!openActionByPosition.has(String(a.position_id)))openActionByPosition.set(String(a.position_id),a);

  function decisionContext(decision:any){
    if(!decision)return null;
    const ids=Array.isArray(decision?.metadata?.source_evidence_ids_all)?decision.metadata.source_evidence_ids_all.map(String):[];
    const evidence=ids.map((id:string)=>sensorById.get(id)).filter(Boolean);
    const groups=[...new Set(evidence.map((s:any)=>String(s.independent_group||"")).filter(Boolean))];
    const eventDriven=evidence.some((s:any)=>String(s.independent_group)==="world_event_reaction"||String(s.sensor_family).toLowerCase().includes("event"));
    const macro=decision?.metadata?.official_macro_context??null;
    const macroEvents=Array.isArray(macro?.events)?macro.events:[];
    return {
      decision_id:String(decision.decision_id),
      observed_at:decision.observed_at,
      action:decision.action,
      direction:Number(decision.direction||0),
      evidence_score:Number(decision.evidence_score||0),
      independent_group_count:Number(decision.independent_group_count||0),
      requested_virtual_notional_usd:Number(decision.requested_virtual_notional_usd||0),
      estimated_round_trip_cost_bps:Number(decision.estimated_round_trip_cost_bps||0),
      reason:decision.reason||null,
      veto_reason:decision.veto_reason||null,
      support_groups:Array.isArray(decision.support_groups)?decision.support_groups:[],
      conflict_groups:Array.isArray(decision.conflict_groups)?decision.conflict_groups:[],
      source_type:eventDriven?"EVENT_CATALYST":"MARKET_STRUCTURE",
      source_label:eventDriven?"Haber / olay + piyasa doğrulaması":"Piyasa mikro-yapısı / akış",
      evidence_groups:groups,
      evidence:evidence.map((s:any)=>({
        observation_id:s.observation_id,observed_at:s.observed_at,group:s.independent_group,
        family:s.sensor_family,direction:Number(s.direction||0),strength:Number(s.strength||0),
        confidence:Number(s.confidence||0),reason:s.reason||null
      })),
      macro_context:macroEvents.length?{
        role:macro?.role||"context_only_no_direction_vote",
        direction_vote:Number(macro?.direction_vote||0),
        events:macroEvents.slice(0,3).map((e:any)=>({title:e.title||null,published_at:e.published_at||null,provenance_uri:e.provenance_uri||null,reason:e.reason||null}))
      }:null
    };
  }
  const decision_contexts=decisions.map(decisionContext).filter(Boolean);
  const contextById=new Map(decision_contexts.map((c:any)=>[String(c.decision_id),c]));

  const position_details=positions.map((p:any)=>{
    const assetId=String(p?.assetId??p?.asset_id??"");
    const positionId=String(p?.positionId??p?.position_id??"");
    const direction=Number(p?.direction||0);
    const capital=Number(p?.capitalUsd??p?.capital_usd??0);
    const entry=Number(p?.entryPrice??p?.entry_price??0);
    const roundTrip=Number(p?.roundTripCostBps??p?.round_trip_cost_bps??0);
    const decisionId=String(p?.sourceDecisionId??p?.source_decision_id??"");
    const mark=markByAsset.get(assetId);
    const markPrice=Number(mark?.observed_mid_price);
    const openAction=openActionByPosition.get(positionId);
    const entryCost=Number(openAction?.cost_usd||0);
    const exitCost=capital*Math.max(0,roundTrip)/20000;
    const gross=(Number.isFinite(markPrice)&&markPrice>0&&entry>0&&capital>0&&(direction===1||direction===-1))
      ?capital*direction*(markPrice/entry-1):null;
    const net=gross==null?null:gross-entryCost-exitCost;
    const rawClass=assetId.includes(":")?assetId.split(":")[0].toUpperCase():"UNKNOWN";
    return {
      position_id:positionId,asset_id:assetId,asset_class:rawClass,direction,capital_usd:capital,
      entry_price:entry,opened_at:p?.openedAt??p?.opened_at??null,round_trip_cost_bps:roundTrip,
      source_decision_id:decisionId,decision_context:contextById.get(decisionId)||null,
      latest_mark_price:Number.isFinite(markPrice)?markPrice:null,latest_mark_at:mark?.observed_at??null,
      gross_unrealized_pnl_usd:gross,entry_cost_usd:entryCost,estimated_exit_cost_usd:exitCost,
      estimated_net_if_closed_usd:net,shadow_only:true,live_execution:false
    };
  });

  return out({
    observed_at:new Date().toISOString(),snapshot:latest.data,actions,opening_actions:openings,
    history:history.reverse(),position_details,decision_contexts,
    coverage:{actions_truncated:actions.length===1000,history_truncated:history.length===1000,action_limit:1000,history_limit:1000,as_of:cutoff},
    shadow_only:true,live_execution:false
  },200,origin);
 }catch(error){
   const body=authKind==="cron"
     ? {error:"ACCOUNTING_DATA_UNAVAILABLE",detail:errorText(error)}
     : {error:"ACCOUNTING_DATA_UNAVAILABLE"};
   return out(body,503,origin);
 }
});
