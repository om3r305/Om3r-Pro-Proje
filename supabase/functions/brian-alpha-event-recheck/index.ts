import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, { auth: { persistSession: false, autoRefreshToken: false } });

const VERSION = "brian.alpha-catalyst-recheck.v1";
const COLLECTOR_ID = "brian-alpha-event-recheck-v1";
const EVIDENCE = "PROSPECTIVE_CATALYST_SENTINEL_SHADOW";
const AUTH_ID = "control-v3";
const MIN_SUPPORT_GROUPS = 2;
const MIN_CONSENSUS_SCORE = 0.18;
const FEE_BPS = 10;
const DEPTH_LIMIT = 100;
const MICRO_GROUPS = new Set(["micro_velocity","micro_volume","micro_breakout","micro_reclaim","micro_taker_flow"]);
const EVENT_GROUPS = new Set(["catalyst_sentinel_reaction","world_event_reaction"]);

type EvidenceRow = {
  observation_id: string; asset_id: string; sensor_family: string; horizon: string; independent_group: string;
  observed_at: string; direction: number; strength: number; confidence: number; reliability: number;
  reason: string | null; metadata: Record<string, unknown> | null;
};
type Vote = EvidenceRow & { group: string; quality: number };
type Decision = {
  action: "OPEN_LONG"|"OPEN_SHORT"|"WAIT"|"VETO";
  direction: -1|0|1; score: number; groups: number; support: string[]; conflict: string[];
  sourceIds: string[]; ignored: string[]; reason: string; vetoReason: string|null;
};

function out(body: unknown, status=200){ return new Response(JSON.stringify(body), {status, headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}}); }
function clip(n:number, lo=0, hi=1){ return Math.max(lo, Math.min(hi, Number.isFinite(n)?n:lo)); }
function finite(v:unknown, fallback=0){ const n=Number(v); return Number.isFinite(n)?n:fallback; }
function err(e:unknown){ return e instanceof Error ? `${e.name}: ${e.message}` : String(e); }
async function sha(s:string){ const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(s))); return [...d].map(b=>b.toString(16).padStart(2,"0")).join(""); }
async function requireCron(req:Request){
  const supplied=(req.headers.get("x-brian-cron-key")??"").trim(); if(!supplied) throw new Error("UNAUTHORIZED_CRON");
  const q=await db.from("brian_dashboard_auth").select("cron_key_sha256").eq("auth_id",AUTH_ID).single();
  if(q.error||!q.data) throw new Error(`CRON_AUTH_UNAVAILABLE:${q.error?.message??"missing"}`);
  const got=await sha(supplied), expected=String(q.data.cron_key_sha256??"");
  if(got.length!==expected.length) throw new Error("UNAUTHORIZED_CRON"); let diff=0; for(let i=0;i<got.length;i++) diff|=got.charCodeAt(i)^expected.charCodeAt(i); if(diff!==0) throw new Error("UNAUTHORIZED_CRON");
}
function normalizeAsset(v:unknown){ const raw=String(v??"").trim().toUpperCase().replace(/^CRYPTO:/,""); if(!/^[A-Z0-9]{2,20}USDT$/.test(raw)) throw new Error("INVALID_TRIGGER_ASSET"); return `crypto:${raw}`; }
function horizonMs(h:string){ if(h==="MICRO_1_5M")return 5*60_000; if(h==="FAST_5_30M")return 30*60_000; if(h==="EVENT_DRIVEN")return 2*60*60_000; if(h==="DAILY")return 36*60*60_000; return 0; }
function canonicalGroup(g:string){ if(MICRO_GROUPS.has(g)) return "intrabar_tape"; if(EVENT_GROUPS.has(g)) return "event_reaction"; return g; }
function direction(n:number):-1|0|1{return n>0?1:n<0?-1:0;}

async function loadEvidence(asset:string, nowMs:number){
  const since=new Date(nowMs-36*60*60_000).toISOString();
  const q=await db.from("brian_sensor_observations")
    .select("observation_id,asset_id,sensor_family,horizon,independent_group,observed_at,direction,strength,confidence,reliability,reason,metadata")
    .eq("asset_id",asset).eq("available",true).gte("observed_at",since).neq("independent_group","news_gdelt")
    .order("observed_at",{ascending:false}).limit(1500);
  if(q.error) throw q.error;
  return (q.data??[]) as EvidenceRow[];
}
function compile(rows:EvidenceRow[], nowMs:number):Decision{
  const byGroup=new Map<string,Vote>(); const ignored=new Set<string>();
  for(const r of rows){
    const h=horizonMs(String(r.horizon)); const t=Date.parse(String(r.observed_at)); const fresh=Number.isFinite(t)&&h>0&&t<=nowMs+5000&&nowMs-t<=h;
    const dir=direction(Number(r.direction)); const strength=finite(r.strength,-1), confidence=finite(r.confidence,-1), reliability=finite(r.reliability,-1);
    if(!fresh||dir===0||strength<0||strength>1||confidence<0||confidence>1||reliability<0||reliability>1){ignored.add(String(r.observation_id));continue;}
    const group=canonicalGroup(String(r.independent_group)); const quality=strength*confidence*reliability; const prior=byGroup.get(group);
    if(!prior||quality>prior.quality||(quality===prior.quality&&String(r.observed_at)>String(prior.observed_at))){ if(prior)ignored.add(prior.observation_id); byGroup.set(group,{...r,group,quality}); } else ignored.add(r.observation_id);
  }
  const votes=[...byGroup.values()].sort((a,b)=>a.group.localeCompare(b.group));
  if(!votes.length)return{action:"WAIT",direction:0,score:0,groups:0,support:[],conflict:[],sourceIds:[],ignored:[...ignored],reason:"no fresh directional independent evidence",vetoReason:null};
  const aggregate=votes.reduce((s,v)=>s+direction(v.direction)*v.quality,0)/votes.length; const dir=direction(aggregate);
  const support=votes.filter(v=>direction(v.direction)===dir), conflict=votes.filter(v=>direction(v.direction)!==dir);
  const supportRatio=support.length/votes.length; const breadth=.4+.6*Math.min(1,votes.length/3); const score=clip(Math.abs(aggregate)*(.5+.5*supportRatio)*breadth);
  const base={direction:dir,score,groups:votes.length,support:support.map(v=>v.group).sort(),conflict:conflict.map(v=>v.group).sort(),sourceIds:votes.map(v=>v.observation_id).sort(),ignored:[...ignored].sort()};
  if(dir===0||support.length<MIN_SUPPORT_GROUPS||score<MIN_CONSENSUS_SCORE)return{...base,action:"WAIT",reason:`consensus below preregistered gate: support=${support.length}, score=${score.toFixed(6)}`,vetoReason:null};
  return{...base,action:dir===1?"OPEN_LONG":"OPEN_SHORT",reason:"fresh independent evidence passed catalyst-priority preregistered consensus",vetoReason:null};
}

async function latestIntrabar(asset:string){
  const since=new Date(Date.now()-5*60_000).toISOString();
  const q=await db.from("brian_intrabar_reaction_events").select("event_id,observed_at,direction,status,late_chase,reason").eq("asset_id",asset).gte("observed_at",since).order("observed_at",{ascending:false}).limit(1).maybeSingle();
  if(q.error) throw q.error; return q.data??null;
}

type Cost={quoteId:string;mid:number;spreadBps:number;slippageBps:number;roundTripBps:number;fillable:boolean;notional:number;filled:number;sourceId:string;lastUpdateId:string};
async function observedDepthCost(asset:string, dir:-1|1, notional:number, observedAt:string):Promise<Cost>{
  const symbol=asset.replace(/^crypto:/,""); const url=`https://api.binance.com/api/v3/depth?symbol=${encodeURIComponent(symbol)}&limit=${DEPTH_LIMIT}`;
  const r=await fetch(url,{headers:{accept:"application/json","user-agent":"Brian-Catalyst-ALPHA/1.0"},signal:AbortSignal.timeout(6000)}); if(!r.ok)throw new Error(`BINANCE_DEPTH_HTTP_${r.status}`);
  const p=await r.json() as Record<string,unknown>; const bids=Array.isArray(p.bids)?p.bids as unknown[][]:[]; const asks=Array.isArray(p.asks)?p.asks as unknown[][]:[];
  const bid=finite(bids[0]?.[0]), ask=finite(asks[0]?.[0]); if(!(bid>0)||!(ask>=bid))throw new Error("INVALID_DEPTH_TOP"); const mid=(bid+ask)/2; const levels=dir===1?asks:bids;
  let remain=notional, filled=0, baseQty=0;
  for(const level of levels){const price=finite(level?.[0]),qty=finite(level?.[1]);if(!(price>0)||!(qty>0))continue;const cap=price*qty;const take=Math.min(remain,cap);filled+=take;baseQty+=take/price;remain-=take;if(remain<=1e-9)break;}
  const fillable=remain<=Math.max(.000001,notional*.001); const vwap=baseQty>0?filled/baseQty:mid; const slippageBps=fillable?Math.max(0,dir===1?(vwap/mid-1)*10000:(1-vwap/mid)*10000):9999;
  const spreadBps=10000*(ask-bid)/mid; const roundTripBps=2*FEE_BPS+spreadBps+2*slippageBps; const lastUpdateId=String(p.lastUpdateId??""); const sourceId=`binance_public_rest_depth:${symbol}:${lastUpdateId}`;
  return{quoteId:await sha(`${VERSION}|cost|${asset}|${observedAt}|${dir}|${notional}|${lastUpdateId}`),mid,spreadBps,slippageBps,roundTripBps,fillable,notional,filled,sourceId,lastUpdateId};
}
function ticket(score:number){return score>=.8?20:score>=.6?10:score>=.4?5:3;}
async function topOfBook(asset:string){
  const symbol=asset.replace(/^crypto:/,"");
  const r=await fetch(`https://api.binance.com/api/v3/ticker/bookTicker?symbol=${encodeURIComponent(symbol)}`,{headers:{accept:"application/json","user-agent":"Brian-Catalyst-ALPHA/1.0"},signal:AbortSignal.timeout(5000)});
  if(!r.ok)throw new Error(`BINANCE_BOOK_HTTP_${r.status}`); const p=await r.json() as Record<string,unknown>; const bid=finite(p.bidPrice),ask=finite(p.askPrice); if(!(bid>0)||!(ask>=bid))throw new Error("INVALID_BOOK_TOP"); const mid=(bid+ask)/2; return{mid,spreadBps:10000*(ask-bid)/mid};
}

async function macroContext(asOf:string){
  const since=new Date(Date.parse(asOf)-60*60_000).toISOString();
  const q=await db.from("brian_sensor_observations").select("observation_id,observed_at,source_ids,reason,metadata").eq("asset_id","global:MACRO").eq("sensor_family","official_macro_event").eq("available",true).gte("observed_at",since).lte("observed_at",asOf).order("observed_at",{ascending:false}).limit(24);
  if(q.error)return{role:"context_only_no_direction_vote",event_count:0,events:[],degraded_reason:q.error.message};
  const events=(q.data??[]).map((r:any)=>({observation_id:r.observation_id,observed_at:r.observed_at,source_ids:r.source_ids??[],reason:r.reason,event_id:r.metadata?.event_id??null,correlation_id:r.metadata?.correlation_id??r.metadata?.event_id??null,organization:r.metadata?.organization??null,title:r.metadata?.title??null,published_at:r.metadata?.published_at??null,provenance_uri:r.metadata?.provenance_uri??null}));
  return{role:"context_only_no_direction_vote",direction_vote:0,window_minutes:60,as_of:asOf,event_count:events.length,latest_observed_at:events[0]?.observed_at??null,events};
}
async function sourceEventIds(rows:EvidenceRow[], ids:string[]){
  const set=new Set(ids); const events=new Set<string>(); const alerts=new Set<string>(); const watches=new Set<string>();
  for(const r of rows){if(!set.has(r.observation_id))continue;const md=r.metadata??{};for(const key of ["event_id","trigger_event_id"]){const v=md[key];if(typeof v==="string"&&v)events.add(v);}const arr=md.source_event_ids;if(Array.isArray(arr))for(const v of arr)if(v)events.add(String(v));if(md.alert_id)alerts.add(String(md.alert_id));if(md.watch_id)watches.add(String(md.watch_id));}
  return{event_ids:[...events],alert_ids:[...alerts],watch_ids:[...watches]};
}
async function recordRun(startedAt:string,status:string,asset:string,eventId:string,stored:number,error?:unknown){
  const finishedAt=new Date().toISOString(); const runId=await sha(`${COLLECTOR_ID}|${startedAt}|${asset}|${eventId}|${status}`);
  await db.from("brian_collector_runs").insert({run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:finishedAt,status,observed_records:1,stored_records:stored,degraded_sources:[],error_class:error?"CATALYST_ALPHA_RECHECK_ERROR":null,error_message:error?err(error).slice(0,1200):null,evidence_class:EVIDENCE,shadow_only:true,live_execution:false,metadata:{version:VERSION,event_id:eventId,asset_id:asset}});
}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);
  try{await requireCron(req);}catch(e){return out({status:"UNAUTHORIZED",error:err(e),shadow_only:true,live_execution:false},401);}
  const startedAt=new Date().toISOString();
  try{
    const body=await req.json().catch(()=>({})) as Record<string,unknown>; const eventId=String(body.event_id??body.trigger_event_id??"").trim(); const alertId=String(body.alert_id??body.trigger_alert_id??"").trim(); const asset=normalizeAsset(body.asset_id??body.trigger_asset_id);
    if(!eventId)throw new Error("MISSING_EVENT_ID");
    const watchQ=await db.from("brian_catalyst_sentinel_watches").select("watch_id,status,reaction_score,direction,started_at").eq("event_id",eventId).eq("asset_id",asset).limit(1).maybeSingle();
    const nowMs=Date.now(), evidence=await loadEvidence(asset,nowMs); let decision=compile(evidence,nowMs); let intrabar:any=null; let cost:Cost|null=null; let book:{mid:number;spreadBps:number}|null=null;
    try{book=await topOfBook(asset);}catch{/* WAIT remains auditable even if top-of-book is temporarily unavailable */}
    if(decision.action==="OPEN_LONG"||decision.action==="OPEN_SHORT"){
      try{intrabar=await latestIntrabar(asset);}catch(e){decision={...decision,action:"VETO",reason:`actionable evidence cannot open because intrabar context is unavailable: ${err(e)}`,vetoReason:"CONTEXT_UNAVAILABLE"};}
      if((decision.action==="OPEN_LONG"||decision.action==="OPEN_SHORT")&&intrabar&&Boolean(intrabar.late_chase)&&String(intrabar.status)==="VETOED_LATE_CHASE"&&direction(Number(intrabar.direction))===decision.direction){decision={...decision,action:"VETO",reason:"independent evidence aligned but current intrabar state is a late-chase veto",vetoReason:"LATE_CHASE"};}
      if(decision.action==="OPEN_LONG"||decision.action==="OPEN_SHORT"){
        try{cost=await observedDepthCost(asset,decision.direction as -1|1,ticket(decision.score),new Date().toISOString());if(!cost.fillable)decision={...decision,action:"VETO",reason:"evidence is actionable but visible liquidity cannot fill requested shadow notional",vetoReason:"INSUFFICIENT_VISIBLE_DEPTH"};}
        catch(e){decision={...decision,action:"VETO",reason:`evidence is actionable but synchronized L2 cost is unavailable: ${err(e)}`,vetoReason:"COST_UNAVAILABLE"};}
      }
    }
    const observedAt=new Date().toISOString(); const linkage=await sourceEventIds(evidence,decision.sourceIds); if(!linkage.event_ids.includes(eventId))linkage.event_ids.push(eventId);
    const macro=await macroContext(observedAt); const referencePrice=cost?.mid??book?.mid??null; let costId:string|null=null;
    if(cost){costId=cost.quoteId;const ins=await db.from("brian_dynamic_cost_quotes").insert({quote_id:cost.quoteId,compiler_version:VERSION,asset_id:asset,observed_at:observedAt,side:decision.direction===1?"BUY":"SELL",requested_notional_usd:cost.notional,filled_notional_usd:cost.filled,fill_ratio:cost.notional>0?clip(cost.filled/cost.notional):0,fillable:cost.fillable,fee_bps:FEE_BPS,spread_bps:cost.spreadBps,depth_slippage_bps:cost.slippageBps,one_way_cost_bps:FEE_BPS+cost.spreadBps/2+cost.slippageBps,estimated_round_trip_cost_bps:cost.roundTripBps,quality:"L2_OBSERVED",source_ids:[cost.sourceId],reason:"catalyst-priority synchronized Binance L2 shadow cost",metadata:{version:VERSION,last_update_id:cost.lastUpdateId,depth_limit:DEPTH_LIMIT,trigger_event_id:eventId,trigger_alert_id:alertId},evidence_class:EVIDENCE,shadow_only:true,live_execution:false});if(ins.error)throw ins.error;}
    const decisionId=await sha(`${VERSION}|${eventId}|${asset}|${observedAt}|${decision.action}|${decision.direction}|${decision.score.toFixed(12)}`);
    const row={decision_id:decisionId,compiler_version:VERSION,observed_at:observedAt,asset_id:asset,observed_reference_price:referencePrice,action:decision.action,direction:decision.direction,evidence_score:decision.score,independent_group_count:decision.groups,support_groups:decision.support,conflict_groups:decision.conflict,source_observation_ids:decision.sourceIds,source_intrabar_event_ids:intrabar?.event_id?[String(intrabar.event_id)]:[],source_cost_quote_id:costId,requested_virtual_notional_usd:cost?.notional??0,gross_edge_bps:null,estimated_round_trip_cost_bps:cost?.roundTripBps??null,net_edge_bps:null,veto_reason:decision.vetoReason,reason:decision.reason,metadata:{ignored_observation_ids:decision.ignored,trigger_source:"catalyst_sentinel",trigger_event_id:eventId,correlation_id:eventId,trigger_alert_id:alertId||null,trigger_watch_id:watchQ.data?.watch_id??null,catalyst_watch_status:watchQ.data?.status??null,catalyst_reaction_score:watchQ.data?.reaction_score??null,catalyst_direction:watchQ.data?.direction??null,event_linkage:{role:"event_recheck_context",event_id:eventId,correlation_id:eventId,asset_id:asset,directional_evidence_event_ids:linkage.event_ids,source_alert_ids:linkage.alert_ids,source_watch_ids:linkage.watch_ids},reference_price:{value:referencePrice,source:cost?.mid?"binance_public_rest_depth_mid":book?.mid?"binance_public_book_ticker_mid":"unavailable",captured_at:observedAt},official_macro_context:macro,score_is_not_expected_return_bps:true,hard_risk_gates_bypassed:false,scheduling_priority_bypass_only:true},evidence_class:EVIDENCE,shadow_only:true,live_execution:false};
    const ins=await db.from("brian_alpha_decisions").insert(row); if(ins.error)throw ins.error; await recordRun(startedAt,"SUCCESS",asset,eventId,1);
    return out({status:"CAPTURED",version:VERSION,event_id:eventId,correlation_id:eventId,asset_id:asset,decision_id:decisionId,action:decision.action,direction:decision.direction,evidence_score:decision.score,hard_risk_gates_bypassed:false,scheduling_priority_bypass_only:true,shadow_only:true,live_execution:false});
  }catch(e){return out({status:"FAILED_CLOSED",error:err(e),version:VERSION,shadow_only:true,live_execution:false},500);}
});
