import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { requireRealtimeInternal } from "../_shared/realtime_internal_auth.ts";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });
const VERSION = "brian.realtime-big-move-hunter.v1";
const COLLECTOR_ID = "brian-big-move-hunter-v1";
const EVIDENCE = "PROSPECTIVE_BIG_MOVE_SHADOW";
const CORE = ["BTC","ETH","SOL","BNB","XRP"];
const STOP = new Set(["USD","USDT","USDC","API","SEC","ETF","UAE","THE","NEW","CEO","CPI","PCE","FED","ECB","HTTP"]);

type Sensor = { observation_id:string; asset_id:string; observed_at:string; direction:number; strength:number; confidence:number; reliability:number; independent_group:string; sensor_family:string };
type EventRow = { event_id:string; asset:string|null; event_kind:string; source_id:string; first_observed_at:string; published_at:string|null; claim:string; trust_class:string; content_fingerprint:string|null; provenance_uri:string|null };

function out(body: unknown, status=200){ return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}}); }
function errorText(e:unknown){ return e instanceof Error ? `${e.name}: ${e.message}` : String(e); }
function clip(n:number){ return Math.max(0,Math.min(1,Number.isFinite(n)?n:0)); }
async function sha(s:string){ const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(s))); return [...d].map(b=>b.toString(16).padStart(2,"0")).join(""); }
function assetId(base:string){ return `crypto:${base.toUpperCase()}USDT`; }
function eventDirection(claim:string): -1|0|1 {
  const t=claim.toLowerCase();
  if(/delist|trading suspension|suspend(?:ed)? trading|exploit|hack|security breach|insolv|bankrupt|withdrawals? (?:halt|suspend|disable)|enforcement action|charges? against/.test(t)) return -1;
  if(/will list|new listing|listing of|adds? support for|trading (?:will )?(?:begin|launch)|launchpool|launchpad|airdrop|mainnet launch|approval|approved/.test(t)) return 1;
  return 0;
}
function symbolsFromEvent(e:EventRow, eligible:Set<string>): string[] {
  const found=new Set<string>();
  const explicit=String(e.asset??"").toUpperCase().replace(/^CRYPTO:/,"").replace(/USDT$/,'');
  if(explicit && explicit!=="GLOBAL" && explicit!=="GLOBAL_WORLD" && explicit!=="CRYPTO" && eligible.has(`${explicit}USDT`)) found.add(explicit);
  for(const m of e.claim.matchAll(/\(([A-Z0-9]{2,12})\)/g)){ const s=m[1]; if(!STOP.has(s)&&eligible.has(`${s}USDT`)) found.add(s); }
  const upper=` ${e.claim.toUpperCase()} `;
  for(const sym of eligible){ const base=sym.replace(/USDT$/,''); if(base.length<2||STOP.has(base)) continue; if(upper.includes(` ${base} `)||upper.includes(` ${base})`)||upper.includes(`(${base} `)) found.add(base); }
  return [...found].slice(0,5);
}
function weighted(s:Sensor){ return clip(Number(s.strength))*clip(Number(s.confidence))*clip(Number(s.reliability)); }
function marketScore(rows:Sensor[], dir:number){
  const latestByGroup=new Map<string,Sensor>();
  for(const r of rows){ if(!latestByGroup.has(r.independent_group)) latestByGroup.set(r.independent_group,r); }
  let support=0, conflict=0, supportGroups=0, conflictGroups=0;
  for(const r of latestByGroup.values()){ const w=weighted(r); if(Number(r.direction)===dir){support+=w;supportGroups++;} else if(Number(r.direction)===-dir){conflict+=w;conflictGroups++;} }
  const denom=Math.max(0.01,support+conflict); const confirmation=clip((support/denom)*clip(support/0.65));
  return {confirmation,support,conflict,supportGroups,conflictGroups,groups:[...latestByGroup.keys()]};
}
async function latestUniverse(){
  const q=await db.from("brian_universe_snapshots").select("observed_at,candidates").order("observed_at",{ascending:false}).limit(1).maybeSingle();
  const env=(q.data?.candidates??{}) as Record<string,unknown>; const eligibleRaw=Array.isArray(env.eligible_symbols)?env.eligible_symbols:[];
  const eligible=new Set<string>(eligibleRaw.map(String).filter(s=>/^[A-Z0-9]+USDT$/.test(s)));
  for(const c of CORE) eligible.add(`${c}USDT`);
  const candidates=Array.isArray(env.candidates)?env.candidates as Record<string,unknown>[]:[];
  const radar=new Map<string,Record<string,unknown>>(); for(const c of candidates){ const s=String(c.symbol??""); if(s) radar.set(s,c); }
  return {eligible,radar,observedAt:String(q.data?.observed_at??"")};
}
async function recentSensors(since:string){
  const q=await db.from("brian_sensor_observations").select("observation_id,asset_id,observed_at,direction,strength,confidence,reliability,independent_group,sensor_family").gte("observed_at",since).eq("available",true).neq("independent_group","world_event_reaction").order("observed_at",{ascending:false}).limit(5000);
  if(q.error) throw q.error; const byAsset=new Map<string,Sensor[]>();
  for(const r of (q.data??[]) as Sensor[]){ const a=String(r.asset_id); const arr=byAsset.get(a)??[]; arr.push(r); byAsset.set(a,arr); }
  return byAsset;
}
async function latestPrice(asset:string, at:string){
  const q=await db.from("brian_micro_book_ticks").select("observed_at,observed_mid_price").eq("asset_id",asset).lte("observed_at",at).order("observed_at",{ascending:false}).limit(1).maybeSingle();
  const p=Number(q.data?.observed_mid_price); return Number.isFinite(p)&&p>0?p:null;
}
async function persistOpportunity(row:Record<string,unknown>){ const q=await db.from("brian_opportunity_observations").upsert(row,{onConflict:"observation_id",ignoreDuplicates:true}); if(q.error) throw q.error; }
async function persistSensor(row:Record<string,unknown>){ const q=await db.from("brian_sensor_observations").upsert(row,{onConflict:"observation_id",ignoreDuplicates:true}); if(q.error) throw q.error; }

async function eventLane(now:Date, eligible:Set<string>, radar:Map<string,Record<string,unknown>>, sensors:Map<string,Sensor[]>){
  const since=new Date(now.getTime()-45*60_000).toISOString();
  const q=await db.from("brian_intel_events").select("event_id,asset,event_kind,source_id,first_observed_at,published_at,claim,trust_class,content_fingerprint,provenance_uri").gte("first_observed_at",since).in("trust_class",["OFFICIAL_PRIMARY","INSTITUTIONAL"]).order("first_observed_at",{ascending:false}).limit(1000);
  if(q.error) throw q.error; let observed=0,confirmed=0;
  for(const e of (q.data??[]) as EventRow[]){ const dir=eventDirection(e.claim); if(!dir) continue; const bases=symbolsFromEvent(e,eligible); if(!bases.length) continue;
    for(const base of bases){ const asset=assetId(base); const eventMs=Date.parse(e.first_observed_at); const rows=(sensors.get(asset)??[]).filter(r=>Date.parse(r.observed_at)>=eventMs-15_000); const m=marketScore(rows,dir);
      const truth=e.trust_class==="OFFICIAL_PRIMARY"?0.98:0.86; const radarRow=radar.get(`${base}USDT`)??{}; const radarScore=clip(Number(radarRow.radar_score??0.35));
      const ageMin=Math.max(0,(now.getTime()-eventMs)/60_000); const freshness=clip(1-ageMin/60); const novelty=1; const priority=clip(0.30*truth+0.25*novelty+0.30*m.confirmation+0.15*radarScore)*freshness;
      const refPrice=await latestPrice(asset,e.first_observed_at); const id=await sha(`${VERSION}|event|${e.event_id}|${asset}`); const veto=m.supportGroups<2||m.confirmation<0.34?"WAIT_MARKET_CONFIRMATION":null;
      await persistOpportunity({observation_id:id,asset,observed_at:e.first_observed_at,event_truth_score:truth,manipulation_risk:e.trust_class==="OFFICIAL_PRIMARY"?0.02:0.12,social_authenticity:1,smart_money_score:clip(m.support),market_confirmation:m.confirmation,priority_score:priority,veto_reason:veto,source_event_ids:[e.event_id],metadata:{version:VERSION,lane:"EVENT_CATALYST",claim:e.claim,source_id:e.source_id,provenance_uri:e.provenance_uri,event_direction:dir,novelty_score:novelty,freshness_score:freshness,radar_score:radarScore,support_groups:m.supportGroups,conflict_groups:m.conflictGroups,support:m.support,conflict:m.conflict,reference_price:refPrice,big_move_target_bps:50,dip_dependency:false,shadow_only:true,live_execution:false}}); observed++;
      if(!veto && priority>=0.55){ const sid=await sha(`${VERSION}|sensor|${e.event_id}|${asset}`); await persistSensor({observation_id:sid,eye_id:"brian-big-move-hunter",template_id:"event-catalyst-reaction-v1",asset_id:asset,market_domain:"CRYPTO",sensor_family:"world_event_reaction",horizon:"EVENT_DRIVEN",independent_group:"world_event_reaction",observed_at:new Date().toISOString(),direction:dir,strength:clip(priority),confidence:clip(truth*m.confirmation),reliability:0.60,available:true,source_ids:[e.source_id],reason:`Official/institutional catalyst + independent market reaction: ${e.claim}`.slice(0,500),evidence_class:EVIDENCE,shadow_only:true,live_execution:false,metadata:{version:VERSION,opportunity_id:id,event_id:e.event_id,priority_score:priority,market_confirmation:m.confirmation,dip_dependency:false}}); confirmed++; }
    }
  }
  return {observed,confirmed};
}
async function precursorLane(now:Date, radar:Map<string,Record<string,unknown>>, sensors:Map<string,Sensor[]>){
  let observed=0; for(const [symbol,r] of radar){ const asset=`crypto:${symbol}`; const rows=sensors.get(asset)??[]; if(!rows.length) continue;
    const latestByGroup=new Map<string,Sensor>(); for(const x of rows){ if(!latestByGroup.has(x.independent_group)) latestByGroup.set(x.independent_group,x); }
    let pos=0,neg=0,pg=0,ng=0; for(const x of latestByGroup.values()){const w=weighted(x);if(x.direction>0){pos+=w;pg++;}else if(x.direction<0){neg+=w;ng++;}}
    const dir: -1|0|1=pos>neg?1:neg>pos?-1:0; const support=Math.max(pos,neg),conflict=Math.min(pos,neg),groups=dir>0?pg:ng; const consensus=clip((support-conflict)/Math.max(.01,support+conflict));
    const radarScore=clip(Number(r.radar_score??0)); const change=Math.abs(Number(r.price_change_pct??0)); const latePenalty=change>=12?0.25:change>=8?0.55:1; const priority=clip((0.45*clip(support/0.8)+0.30*consensus+0.25*radarScore)*latePenalty);
    if(!dir||groups<3||priority<0.55) continue; const at=new Date().toISOString(); const id=await sha(`${VERSION}|precursor|${symbol}|${Math.floor(now.getTime()/300000)}`); const refPrice=await latestPrice(asset,at);
    await persistOpportunity({observation_id:id,asset,observed_at:at,event_truth_score:0.70,manipulation_risk:0.15,social_authenticity:0.5,smart_money_score:clip(support),market_confirmation:consensus,priority_score:priority,veto_reason:change>=12?"LATE_MOVE_ALREADY_EXTENDED":null,source_event_ids:[],metadata:{version:VERSION,lane:"PRECURSOR_CONVERGENCE",direction:dir,support_groups:groups,support,conflict,consensus,radar_score:radarScore,price_change_pct:Number(r.price_change_pct??0),reference_price:refPrice,big_move_target_bps:50,dip_dependency:false,no_alpha_double_count:true,shadow_only:true,live_execution:false}}); observed++;
  } return {observed};
}
async function resolveOutcomes(now:Date){
  const since=new Date(now.getTime()-3*60*60_000).toISOString(); const q=await db.from("brian_opportunity_observations").select("observation_id,asset,observed_at,metadata").gte("observed_at",since).order("observed_at",{ascending:false}).limit(500); if(q.error) return 0; let stored=0;
  for(const o of q.data??[]){ const md=(o.metadata??{}) as Record<string,unknown>; const dir=Number(md.event_direction??md.direction??0); const p0=Number(md.reference_price); if(!dir||!Number.isFinite(p0)||p0<=0) continue;
    for(const sec of [300,900,3600]){ const target=new Date(Date.parse(String(o.observed_at))+sec*1000); if(now.getTime()<target.getTime()+30_000) continue; const oid=await sha(`${VERSION}|outcome|${o.observation_id}|${sec}`); const exists=await db.from("brian_opportunity_outcomes").select("outcome_id").eq("outcome_id",oid).maybeSingle(); if(exists.data) continue;
      const ticks=await db.from("brian_micro_book_ticks").select("observed_at,observed_mid_price").eq("asset_id",String(o.asset)).gte("observed_at",String(o.observed_at)).lte("observed_at",target.toISOString()).order("observed_at",{ascending:true}).limit(1000); if(ticks.error||!(ticks.data??[]).length) continue;
      const prices=(ticks.data??[]).map(x=>Number(x.observed_mid_price)).filter(x=>Number.isFinite(x)&&x>0); if(!prices.length) continue; const p1=prices[prices.length-1]; const rets=prices.map(p=>dir*(p/p0-1)); const gross=dir*(p1/p0-1); const row={outcome_id:oid,observation_id:o.observation_id,resolved_at:new Date().toISOString(),horizon_seconds:sec,gross_return:gross,net_return:gross,mae:Math.min(...rets),mfe:Math.max(...rets),metadata:{version:VERSION,price_start:p0,price_end:p1,direction:dir,cost_not_applied:true,shadow_only:true,live_execution:false}}; const ins=await db.from("brian_opportunity_outcomes").insert(row); if(!ins.error) stored++;
    }
  } return stored;
}

async function recordRun(startedAt:string,status:"SUCCESS"|"FAILED",observed:number,stored:number,error?:unknown){
  const finishedAt=new Date().toISOString();
  const runId=await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  const q=await db.from("brian_collector_runs").insert({
    run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:finishedAt,status,
    observed_records:observed,stored_records:stored,degraded_sources:[],
    error_class:error?"BIG_MOVE_HUNTER_ERROR":null,error_message:error?errorText(error).slice(0,1200):null,
    evidence_class:EVIDENCE,shadow_only:true,live_execution:false,
    metadata:{version:VERSION,dip_dependency:false,shadow_only:true,live_execution:false}
  });
  if(q.error) console.error("big-move run receipt",q.error.message);
}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST") return out({error:"POST required"},405);
  const startedAt=new Date().toISOString();
  try{await requireRealtimeInternal(req);}catch(e){return out({error:errorText(e),shadow_only:true,live_execution:false},401);}
  try{
    const now=new Date(); const {eligible,radar,observedAt}=await latestUniverse(); const sensors=await recentSensors(new Date(now.getTime()-35*60_000).toISOString()); const event=await eventLane(now,eligible,radar,sensors); const precursor=await precursorLane(now,radar,sensors); const outcomes=await resolveOutcomes(now);
    const observed=event.observed+precursor.observed;
    const stored=event.confirmed+outcomes;
    await recordRun(startedAt,"SUCCESS",observed,stored);
    return out({status:"SUCCESS",collector_id:COLLECTOR_ID,version:VERSION,universe_observed_at:observedAt,event_opportunities:event.observed,event_confirmed_alpha_evidence:event.confirmed,precursor_opportunities:precursor.observed,outcomes_resolved:outcomes,dip_dependency:false,shadow_only:true,live_execution:false});
  }catch(e){
    await recordRun(startedAt,"FAILED",0,0,e);
    return out({status:"FAILED",collector_id:COLLECTOR_ID,error:errorText(e),version:VERSION,dip_dependency:false,shadow_only:true,live_execution:false},500);
  }
});
