import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { XMLParser } from "npm:fast-xml-parser@4.5.0";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const ANON = Deno.env.get("SUPABASE_ANON_KEY") ?? "";
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });

const V = "brian.catalyst-sentinel.v2.1";
const E = "PROSPECTIVE_CATALYST_SENTINEL_SHADOW";
const AUTH = "control-v3";
const FAST_IDS = ["fed_monetary", "ecb_press", "bls_cpi", "bls_employment", "sec_press_rss"];
const CORE = ["crypto:BTCUSDT", "crypto:ETHUSDT", "crypto:SOLUSDT", "crypto:BNBUSDT", "crypto:XRPUSDT", "crypto:ADAUSDT"];
const RECHECK = [5, 10, 15, 30];
const ACTIVE = ["WATCHING", "BUILDING", "BREAKOUT_CANDIDATE", "CONFIRMED"];
const WATCH_TTL_MS = 120 * 60_000;
const FEED_RECENCY_MS = 20 * 60_000;
const BASELINE_RECENCY_MS = 2 * 60_000;
const ALERT_BUCKET_MS = 60_000;
const ALERT_COOLDOWN_MS = 60_000;
const DISPATCH_TIMEOUT_MS = 4_500;
const RUN_LEASE_SEC = 12;

type EP = { endpoint_id:string; source_id:string; organization:string; canonical_domain:string; endpoint_url:string; category:string; tier:string };
type Item = { title:string; link:string; guid:string; publishedAt:string|null };
type Book = { mid:number; spread:number };
type Reaction = { ret:number; spread:number|null; imb?:number; range?:number; cross?:number; vol?:number };

const FALLBACK_ENDPOINTS: EP[] = [
  {endpoint_id:"fed_monetary",source_id:"official:fed:monetary",organization:"Federal Reserve Board",canonical_domain:"federalreserve.gov",endpoint_url:"https://www.federalreserve.gov/feeds/press_monetary.xml",category:"CENTRAL_BANK",tier:"T1_OFFICIAL_PRIMARY"},
  {endpoint_id:"ecb_press",source_id:"official:ecb:press",organization:"ECB",canonical_domain:"ecb.europa.eu",endpoint_url:"https://www.ecb.europa.eu/rss/press.html",category:"CENTRAL_BANK",tier:"T1_OFFICIAL_PRIMARY"},
  {endpoint_id:"bls_cpi",source_id:"official:bls:cpi",organization:"BLS CPI",canonical_domain:"bls.gov",endpoint_url:"https://www.bls.gov/feed/cpi.rss",category:"MACRO_INFLATION",tier:"T1_OFFICIAL_PRIMARY"},
  {endpoint_id:"bls_employment",source_id:"official:bls:employment",organization:"BLS Employment Situation",canonical_domain:"bls.gov",endpoint_url:"https://www.bls.gov/feed/empsit.rss",category:"MACRO_EMPLOYMENT",tier:"T1_OFFICIAL_PRIMARY"},
  {endpoint_id:"sec_press_rss",source_id:"official:sec:press",organization:"SEC Newsroom",canonical_domain:"sec.gov",endpoint_url:"https://www.sec.gov/news/pressreleases.rss",category:"FINANCIAL_REGULATION",tier:"T1_OFFICIAL_PRIMARY"},
];

const clip=(n:number)=>Math.max(0,Math.min(1,Number.isFinite(n)?n:0));
const finite=(x:unknown,d=0)=>Number.isFinite(Number(x))?Number(x):d;
const arr=<T>(x:T|T[]|null|undefined):T[]=>x==null?[]:Array.isArray(x)?x:[x];
const err=(e:unknown)=>{
  if(e instanceof Error) return `${e.name}: ${e.message}`;
  if(e && typeof e === "object") {
    const x=e as Record<string,unknown>;
    try { return JSON.stringify({code:x.code??null,message:x.message??null,details:x.details??null,hint:x.hint??null}); }
    catch { return String(e); }
  }
  return String(e);
};

async function sha(s:string){
  const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(s)));
  return [...d].map(x=>x.toString(16).padStart(2,"0")).join("");
}

async function auth(req:Request){
  const k=(req.headers.get("x-brian-cron-key")??"").trim();
  if(!k) throw Error("UNAUTHORIZED_CRON");
  const q=await db.from("brian_dashboard_auth").select("cron_key_sha256").eq("auth_id",AUTH).single();
  if(q.error||!q.data) throw Error(`CRON_AUTH_UNAVAILABLE:${q.error?.message??"missing"}`);
  const a=await sha(k), b=String(q.data.cron_key_sha256??"");
  if(a.length!==b.length) throw Error("UNAUTHORIZED_CRON");
  let z=0; for(let i=0;i<a.length;i++) z|=a.charCodeAt(i)^b.charCodeAt(i);
  if(z) throw Error("UNAUTHORIZED_CRON");
}

async function acquireLease(token:string){
  const q=await db.rpc("brian_catalyst_sentinel_try_lease",{p_token:token,p_ttl_seconds:RUN_LEASE_SEC});
  if(q.error) throw Error(`LEASE_ACQUIRE:${err(q.error)}`);
  return q.data===true;
}
async function releaseLease(token:string,status:string,error:string|null){
  try{ await db.rpc("brian_catalyst_sentinel_release_lease",{p_token:token,p_status:status,p_error:error}); }catch{/* fail safe */}
}

function txt(x:unknown){
  if(typeof x==="string") return x.trim();
  if(x&&typeof x==="object") { const y=x as Record<string,unknown>; return String(y["#text"]??y["@_href"]??y.href??"").trim(); }
  return x==null?"":String(x).trim();
}
function iso(x:unknown){ const t=Date.parse(txt(x)); return Number.isFinite(t)?new Date(t).toISOString():null; }
function parse(xml:string):Item[]{
  const d=new XMLParser({ignoreAttributes:false,trimValues:true}).parse(xml);
  const r=d?.rss?.channel?.item??d?.feed?.entry??[];
  return arr<any>(r).slice(0,50).map(x=>{
    let link=""; for(const y of arr<any>(x.link)){ link=txt(y); if(link) break; }
    return {title:txt(x.title).replace(/\s+/g," "),link,guid:txt(x.guid??x.id)||link||txt(x.title),publishedAt:iso(x.pubDate??x.published??x.updated)};
  }).filter(x=>x.title&&x.guid);
}
function relevant(e:EP,t:string){
  if(e.endpoint_id==="fed_monetary") return /\bfomc\b|federal reserve|monetary policy|interest rate|federal funds|economic projections|minutes|statement/i.test(t);
  if(e.endpoint_id==="ecb_press") return /monetary policy|interest rate|governing council|policy decision|inflation|\brates?\b/i.test(t);
  if(e.endpoint_id==="bls_cpi"||e.endpoint_id==="bls_employment") return true;
  return /crypto|digital asset|bitcoin|ethereum|ether|stablecoin|blockchain|exchange-traded|\betf\b|token/i.test(t);
}
const macro=(c:string)=>/CENTRAL_BANK|MACRO_|FUNDING_LIQUIDITY/i.test(c);

function explicit(asset:unknown,title:string,eligible:Set<string>){
  const out=new Set<string>();
  const raw=String(asset??"").toUpperCase().replace(/^CRYPTO:/,"").replace(/[^A-Z0-9]/g,"");
  if(raw&&!/^(GLOBAL|GLOBALWORLD|MACRO|CRYPTO)$/.test(raw)){
    const s=raw.endsWith("USDT")?raw:`${raw}USDT`; if(eligible.has(s)) out.add(`crypto:${s}`);
  }
  const m:[RegExp,string][]=[[/\bbitcoin\b|\bBTC\b/i,"BTCUSDT"],[/\bethereum\b|\bether\b|\bETH\b/i,"ETHUSDT"],[/\bsolana\b|\bSOL\b/i,"SOLUSDT"],[/\bcardano\b|\bADA\b/i,"ADAUSDT"],[/\bripple\b|\bXRP\b/i,"XRPUSDT"],[/\bbnb\b|binance coin/i,"BNBUSDT"]];
  for(const [r,s] of m) if(r.test(title)&&eligible.has(s)) out.add(`crypto:${s}`);
  return [...out];
}

async function getUniverse(degraded:string[]){
  const s=new Set(CORE.map(x=>x.slice(7)));
  try{
    const q=await db.from("brian_universe_snapshots").select("candidates").order("observed_at",{ascending:false}).limit(1).maybeSingle();
    if(q.error) throw q.error;
    const c=(q.data?.candidates??{}) as any;
    for(const x of Array.isArray(c.eligible_symbols)?c.eligible_symbols:[]) if(/^[A-Z0-9]+USDT$/.test(String(x))) s.add(String(x));
  }catch(e){ degraded.push(`universe:${err(e)}`); }
  return s;
}

async function getEndpoints(degraded:string[]){
  try{
    const q=await db.from("brian_source_registry_status_v2").select("endpoint_id,source_id,organization,canonical_domain,endpoint_url,category,tier")
      .in("endpoint_id",FAST_IDS).eq("lifecycle_state","ACTIVE").eq("eligible_for_research",true);
    if(q.error) throw q.error;
    if((q.data??[]).length) return q.data as EP[];
    degraded.push("endpoints:registry_empty_using_fallback");
  }catch(e){ degraded.push(`endpoints:${err(e)}:using_fallback`); }
  return FALLBACK_ENDPOINTS;
}

async function getBooks(assets:string[],degraded:string[]){
  const m=new Map<string,Book>();
  try{
    const r=await fetch("https://api.binance.com/api/v3/ticker/bookTicker",{signal:AbortSignal.timeout(3000),headers:{"user-agent":"Brian-Catalyst-Sentinel/2.1"}});
    if(!r.ok) throw Error(`BINANCE_BOOK_${r.status}`);
    const p=await r.json(); if(!Array.isArray(p)) throw Error("BINANCE_BOOK_BAD_PAYLOAD");
    for(const x of p){ const b=finite(x.bidPrice),a=finite(x.askPrice); if(b>0&&a>=b){ const mid=(a+b)/2; m.set(`crypto:${x.symbol}`,{mid,spread:10000*(a-b)/mid}); } }
    return m;
  }catch(e){ degraded.push(`books_live:${err(e)}`); }
  try{
    const q=await db.from("brian_micro_book_ticks").select("asset_id,observed_mid_price,observed_spread_bps,observed_at")
      .in("asset_id",assets).gte("observed_at",new Date(Date.now()-2*60_000).toISOString()).order("observed_at",{ascending:false}).limit(1500);
    if(q.error) throw q.error;
    for(const r of q.data??[]){ const a=String(r.asset_id); if(m.has(a)) continue; const mid=finite(r.observed_mid_price); if(mid>0)m.set(a,{mid,spread:finite(r.observed_spread_bps,50)}); }
  }catch(e){ degraded.push(`books_fallback:${err(e)}`); }
  return m;
}

async function getDepth(asset:string){
  try{
    const r=await fetch(`https://api.binance.com/api/v3/depth?symbol=${asset.slice(7)}&limit=20`,{signal:AbortSignal.timeout(1800),headers:{"user-agent":"Brian-Catalyst-Sentinel/2.1"}});
    if(!r.ok) return 0; const p=await r.json();
    const sum=(x:any)=>arr<any>(x).reduce((n,y)=>n+finite(y?.[0])*finite(y?.[1]),0); const b=sum(p.bids),a=sum(p.asks); return b+a?(b-a)/(b+a):0;
  }catch{return 0;}
}

async function feed(e:EP,at:string,degraded:string[]){
  let prior:any=null,failures=0;
  try{
    const q=await db.from("brian_catalyst_sentinel_feed_state").select("last_content_hash,consecutive_failures,last_changed_at").eq("endpoint_id",e.endpoint_id).maybeSingle();
    if(q.error) throw q.error; prior=q.data; failures=finite(q.data?.consecutive_failures);
    const r=await fetch(e.endpoint_url,{redirect:"follow",signal:AbortSignal.timeout(2600),headers:{accept:"application/rss+xml,application/atom+xml,application/xml,text/xml","user-agent":"Brian-Catalyst-Sentinel/2.1","cache-control":"no-cache"}});
    if(!r.ok) throw Error(`HTTP_${r.status}`);
    const xml=await r.text(), h=await sha(xml), baseline=!prior?.last_content_hash, changed=!baseline&&h!==prior.last_content_hash, nowMs=Date.parse(at);
    const parsed=parse(xml);
    const items=parsed.filter(i=>{
      if(!i.publishedAt) return changed&&!baseline;
      const t=Date.parse(i.publishedAt),max=baseline?BASELINE_RECENCY_MS:FEED_RECENCY_MS;
      return Number.isFinite(t)&&t<=nowMs+60_000&&nowMs-t<=max&&(baseline||changed);
    });
    const up=await db.from("brian_catalyst_sentinel_feed_state").upsert({endpoint_id:e.endpoint_id,last_checked_at:at,last_changed_at:changed||(baseline&&items.length)?at:prior?.last_changed_at??null,last_content_hash:h,consecutive_failures:0,last_error:null,updated_at:at},{onConflict:"endpoint_id"});
    if(up.error) throw up.error;
    return {e,items,baseline,changed};
  }catch(x){
    degraded.push(`feed:${e.endpoint_id}:${err(x)}`);
    try{ await db.from("brian_catalyst_sentinel_feed_state").upsert({endpoint_id:e.endpoint_id,last_checked_at:at,consecutive_failures:failures+1,last_error:err(x).slice(0,600),updated_at:at},{onConflict:"endpoint_id"}); }catch{}
    return {e,items:[] as Item[],baseline:false,changed:false};
  }
}

async function upsertEvent(e:EP,i:Item,at:string){
  const id=await sha(`source-arch-v2|${e.endpoint_id}|${i.guid}`), fp=await sha(`${e.endpoint_id}|${i.title}|${i.link}`);
  const latency=i.publishedAt?Math.max(0,Date.parse(at)-Date.parse(i.publishedAt)):null;
  const row:any={event_id:id,asset:macro(e.category)?"MACRO":"GLOBAL_WORLD",event_kind:macro(e.category)?"OFFICIAL_MACRO_RELEASE":"OFFICIAL_SOURCE_ITEM",source_kind:`CATALYST_FAST_${e.tier}`,source_id:e.source_id,published_at:i.publishedAt,first_observed_at:at,captured_at:new Date().toISOString(),claim:i.title,direction:0,magnitude:1,trust_class:"OFFICIAL_PRIMARY",entity_confidence:1,content_fingerprint:fp,corroboration_key:await sha(`${e.category}|${i.title.toLowerCase().replace(/[^a-z0-9]+/g," ")}`),provenance_uri:i.link||e.endpoint_url,pit_verified:true,raw_capture_id:null,metadata:{runtime:"CATALYST_SENTINEL_FAST_LANE",version:V,endpoint_id:e.endpoint_id,organization:e.organization,category:e.category,first_http_seen_at:at,stub_inserted_at:new Date().toISOString(),detection_latency_ms:latency,canonical_event_id:id,correlation_id:id,direction_not_inferred:true,asset_mapping_policy:"macro_core_or_explicit_only",random_asset_mapping:false}};
  const q=await db.from("brian_intel_events").upsert(row,{onConflict:"event_id",ignoreDuplicates:true}); if(q.error) throw q.error;
  if(macro(e.category)){
    const oid=await sha(`${V}|macro|${id}`);
    const s=await db.from("brian_sensor_observations").upsert({observation_id:oid,eye_id:"brian-catalyst-sentinel",template_id:"official-macro-fast-lane-v21",asset_id:"global:MACRO",market_domain:"macro",sensor_family:"official_macro_event",horizon:"EVENT_DRIVEN",independent_group:`official_macro_${e.endpoint_id}`,observed_at:at,direction:0,strength:1,confidence:1,reliability:1,available:true,source_ids:[e.source_id],reason:`${e.organization} official release observed; direction delegated to market reaction`,evidence_class:E,shadow_only:true,live_execution:false,metadata:{event_id:id,correlation_id:id,title:i.title,published_at:i.publishedAt,provenance_uri:i.link||e.endpoint_url,detection_latency_ms:latency}},{onConflict:"observation_id",ignoreDuplicates:true});
    if(s.error) throw s.error;
  }
  return row;
}

async function createWatch(ev:any,asset:string,b?:Book){
  const id=await sha(`${V}|watch|${ev.event_id}|${asset}`), now=new Date().toISOString();
  const existing=await db.from("brian_catalyst_sentinel_watches").select("watch_id").eq("event_id",ev.event_id).eq("asset_id",asset).maybeSingle();
  if(existing.error) throw existing.error; if(existing.data?.watch_id) return null;
  const started=String(ev.first_observed_at||now), row:any={watch_id:id,event_id:ev.event_id,asset_id:asset,source_id:String(ev.source_id||"unknown"),event_kind:String(ev.event_kind||"UNKNOWN"),event_title:String(ev.claim||""),event_published_at:ev.published_at??null,started_at:started,expires_at:new Date(Date.parse(started)+WATCH_TTL_MS).toISOString(),status:"WATCHING",direction:0,reference_price:b?.mid??null,last_price:b?.mid??null,last_return:0,reaction_score:0,mfe:0,mae:0,last_evaluated_at:now,last_state_change_at:now,recheck_count:0,next_recheck_at:new Date(Date.parse(started)+RECHECK[0]*60_000).toISOString(),metadata:{version:V,correlation_id:ev.event_id,asset_binding:"macro_core_or_explicit_only",random_asset_mapping:false},evidence_class:E,shadow_only:true,live_execution:false,updated_at:now};
  const q=await db.from("brian_catalyst_sentinel_watches").insert(row).select("watch_id").maybeSingle();
  if(q.error){ if((q.error as any).code==="23505")return null; throw q.error; }
  return q.data?.watch_id?row:null;
}

async function updateAlert(id:string,patch:Record<string,unknown>,meta:Record<string,unknown>){
  const q=await db.from("brian_catalyst_sentinel_alerts").select("metadata").eq("alert_id",id).maybeSingle(); if(q.error)throw q.error;
  const u=await db.from("brian_catalyst_sentinel_alerts").update({...patch,metadata:{...((q.data?.metadata??{}) as any),...meta}}).eq("alert_id",id); if(u.error)throw u.error;
}
async function dispatch(eventId:string,assetId:string,alertId:string,key:string){
  const token=ANON||SERVICE, at=new Date().toISOString();
  try{
    const r=await fetch(`${URL}/functions/v1/brian-alpha-event-recheck`,{method:"POST",headers:{"content-type":"application/json",Authorization:`Bearer ${token}`,apikey:token,"x-brian-cron-key":key},body:JSON.stringify({event_id:eventId,asset_id:assetId,alert_id:alertId}),signal:AbortSignal.timeout(DISPATCH_TIMEOUT_MS)});
    const raw=await r.text(); let j:any={}; try{j=raw?JSON.parse(raw):{}}catch{}
    await updateAlert(alertId,{alpha_dispatched:r.ok,alpha_dispatch_request_id:String(j?.decision_id??`HTTP_${r.status}`),alpha_dispatched_at:at},{last_dispatch_attempt_at:at,alpha_dispatch_http_status:r.status,alpha_dispatch_error:r.ok?null:String(j?.error??raw??`HTTP_${r.status}`).slice(0,600)});
    return r.ok;
  }catch(x){ try{await updateAlert(alertId,{alpha_dispatched:false,alpha_dispatch_request_id:"NETWORK_ERROR",alpha_dispatched_at:at},{last_dispatch_attempt_at:at,alpha_dispatch_error:err(x).slice(0,600)});}catch{} return false; }
}
async function makeAlert(w:any,type:string,dir:number,score:number,m:Reaction,key:string){
  const id=await sha(`${V}|alert|${w.watch_id}|${type}|${Math.floor(Date.now()/ALERT_BUCKET_MS)}`);
  const q=await db.from("brian_catalyst_sentinel_alerts").insert({alert_id:id,watch_id:w.watch_id,event_id:w.event_id,asset_id:w.asset_id,observed_at:new Date().toISOString(),alert_type:type,direction:dir,reaction_score:score,price_return:m.ret??null,spread_bps:m.spread??null,orderbook_imbalance:m.imb??null,realized_range_bps:m.range??null,cross_market_confirmation:m.cross??null,metadata:{version:V,volume_confirmation:m.vol??null,hard_risk_gates_bypassed:false,scheduling_priority_bypass_only:true},evidence_class:E,shadow_only:true,live_execution:false}).select("alert_id").maybeSingle();
  if(q.error){if((q.error as any).code==="23505")return {inserted:false,dispatched:false,id};throw q.error;}
  if(!q.data?.alert_id)return {inserted:false,dispatched:false,id};
  return {inserted:true,dispatched:await dispatch(w.event_id,w.asset_id,id,key),id};
}
async function reactionSensor(w:any,dir:number,score:number,m:Reaction){
  if(!dir||score<.15)return;
  const id=await sha(`${V}|sensor|${w.event_id}|${w.asset_id}|${Math.floor(Date.now()/30_000)}`);
  const q=await db.from("brian_sensor_observations").upsert({observation_id:id,eye_id:"brian-catalyst-sentinel",template_id:"catalyst-market-reaction-v21",asset_id:w.asset_id,market_domain:"CRYPTO",sensor_family:"catalyst_sentinel_reaction",horizon:"EVENT_DRIVEN",independent_group:"catalyst_sentinel_reaction",observed_at:new Date().toISOString(),direction:dir,strength:clip(score),confidence:clip(.6+.3*score),reliability:.65,available:true,source_ids:[w.source_id],reason:`Catalyst Sentinel reaction for ${w.event_id}`,evidence_class:E,shadow_only:true,live_execution:false,metadata:{event_id:w.event_id,correlation_id:w.event_id,watch_id:w.watch_id,reaction_score:score,price_return:m.ret,spread_bps:m.spread,orderbook_imbalance:m.imb,realized_range_bps:m.range,cross_market_confirmation:m.cross,volume_confirmation:m.vol,news_direction_inferred:false,market_reaction_direction:true}},{onConflict:"observation_id",ignoreDuplicates:true});
  if(q.error)throw q.error;
}

async function supportData(assets:string[],degraded:string[]){
  const ranges=new Map<string,number>(),volumes=new Map<string,{d:number;s:number}>();
  try{
    const q=await db.from("brian_micro_book_ticks").select("asset_id,observed_mid_price").in("asset_id",assets).gte("observed_at",new Date(Date.now()-5*60_000).toISOString()).limit(2000); if(q.error)throw q.error;
    const v=new Map<string,number[]>(); for(const r of q.data??[]){const a=String(r.asset_id),p=finite(r.observed_mid_price);if(p>0){const z=v.get(a)??[];z.push(p);v.set(a,z);}}
    for(const [a,z] of v)if(z.length>1)ranges.set(a,10000*(Math.max(...z)-Math.min(...z))/z[0]);
  }catch(e){degraded.push(`ranges:${err(e)}`);}
  try{
    const q=await db.from("brian_sensor_observations").select("asset_id,direction,strength,confidence,reliability,observed_at").in("asset_id",assets).eq("independent_group","micro_volume").eq("available",true).gte("observed_at",new Date(Date.now()-5*60_000).toISOString()).order("observed_at",{ascending:false}).limit(300); if(q.error)throw q.error;
    for(const r of q.data??[]){const a=String(r.asset_id);if(!volumes.has(a))volumes.set(a,{d:Number(r.direction),s:clip(finite(r.strength)*finite(r.confidence)*finite(r.reliability))});}
  }catch(e){degraded.push(`volumes:${err(e)}`);}
  return {ranges,volumes};
}

async function monitor(ws:any[],bm:Map<string,Book>,key:string,degraded:string[]){
  if(!ws.length)return {alerts:0,dispatches:0,evaluated:0};
  const assets=[...new Set(ws.map(w=>String(w.asset_id)))], support=await supportData(assets,degraded);
  const current=new Map<string,{ret:number;dir:number}>(), needsDepth:string[]=[];
  for(const w of ws){const b=bm.get(w.asset_id),ref=finite(w.reference_price);if(b&&ref>0){const r=b.mid/ref-1,d=r>.00005?1:r<-.00005?-1:0;current.set(w.asset_id,{ret:r,dir:d});if(Math.abs(r)>=.0015)needsDepth.push(w.asset_id);}}
  const depthPairs=await Promise.all([...new Set(needsDepth)].slice(0,24).map(async a=>[a,await getDepth(a)] as const)),depthMap=new Map(depthPairs);
  let alerts=0,dispatches=0,evaluated=0;
  for(const w of ws){
    try{
      const b=bm.get(w.asset_id); if(!b)continue;
      let ref=finite(w.reference_price); if(ref<=0){ref=b.mid;const u=await db.from("brian_catalyst_sentinel_watches").update({reference_price:ref,last_price:b.mid,updated_at:new Date().toISOString()}).eq("watch_id",w.watch_id);if(u.error)throw u.error;}
      const ret=b.mid/ref-1,dir=ret>.00005?1:ret<-.00005?-1:0,ab=Math.abs(ret)*10000,imb=depthMap.get(w.asset_id)??0,ran=support.ranges.get(w.asset_id)??0,vs=support.volumes.get(w.asset_id),vol=dir&&vs?.d===dir?vs.s:0;
      const peers=CORE.filter(a=>a!==w.asset_id&&current.has(a));let cross=.5,use=0,ok=0;for(const p of peers){const x=current.get(p)!;if(Math.abs(x.ret)>=.0005){use++;if(x.dir===dir)ok++;}}if(use)cross=ok/use;
      const score=clip(.38*clip(ab/80)+.15*clip((dir*imb+.05)/.35)+.12*clip(ran/120)+.15*cross+.20*vol-.12*clip(b.spread/30));
      const state=ab>=60&&score>=.68&&b.spread<=25?"CONFIRMED":ab>=35&&score>=.55&&b.spread<=30?"BREAKOUT_CANDIDATE":ab>=15||score>=.34?"BUILDING":"WATCHING";
      const now=new Date().toISOString(),changed=state!==w.status,flip=Boolean(w.direction&&dir&&w.direction!==dir&&ab>=20),mins=(Date.now()-Date.parse(w.started_at))/60000;
      let rc=Number(w.recheck_count||0),timed:string|null=null;if(rc<RECHECK.length&&mins>=RECHECK[rc]){timed=`TIMED_RECHECK_${RECHECK[rc]}M`;rc++;}
      const met:Reaction={ret,spread:b.spread,imb,range:ran,cross,vol};
      const patch:any={status:state,direction:dir,last_price:b.mid,last_return:ret,reaction_score:score,mfe:Math.max(finite(w.mfe),ret),mae:Math.min(finite(w.mae),ret),spread_bps:b.spread,orderbook_imbalance:imb,realized_range_bps:ran,cross_market_confirmation:cross,last_evaluated_at:now,last_state_change_at:changed?now:w.last_state_change_at,recheck_count:rc,next_recheck_at:rc<RECHECK.length?new Date(Date.parse(w.started_at)+RECHECK[rc]*60000).toISOString():null,updated_at:now};
      const uq=await db.from("brian_catalyst_sentinel_watches").update(patch).eq("watch_id",w.watch_id);if(uq.error)throw uq.error; evaluated++;
      const live={...w,...patch}; await reactionSensor(live,dir,score,met);
      const cooldown=!w.last_alpha_trigger_at||Date.now()-Date.parse(w.last_alpha_trigger_at)>=ALERT_COOLDOWN_MS;
      const type=flip?"DIRECTION_FLIP":changed&&state!=="WATCHING"?state:timed;if(!type||!cooldown)continue;
      const a=await makeAlert(live,type,dir,score,met,key);if(a.inserted)alerts++;if(a.dispatched){dispatches++;const u=await db.from("brian_catalyst_sentinel_watches").update({last_alpha_trigger_at:now,updated_at:now}).eq("watch_id",w.watch_id);if(u.error)throw u.error;}
    }catch(e){degraded.push(`monitor:${w.watch_id}:${err(e)}`);}
  }
  return {alerts,dispatches,evaluated};
}

async function discover(eligible:Set<string>,bm:Map<string,Book>,degraded:string[]){
  const out:any[]=[];
  try{
    const q=await db.from("brian_intel_events").select("event_id,asset,event_kind,source_id,published_at,first_observed_at,claim,trust_class,metadata").gte("first_observed_at",new Date(Date.now()-20*60_000).toISOString()).in("trust_class",["OFFICIAL_PRIMARY","INSTITUTIONAL","INDEPENDENT_PROFESSIONAL"]).order("first_observed_at",{ascending:false}).limit(150); if(q.error)throw q.error;
    for(const e of q.data??[]){const md=(e.metadata??{}) as any,cat=String(md.category??""),filing=/sec_edgar_current/i.test(String(e.source_id))||/CORPORATE_FILINGS/i.test(cat);if(filing&&!/crypto|digital asset|bitcoin|ethereum|stablecoin|blockchain|\betf\b|token/i.test(String(e.claim)))continue;let aa=explicit(e.asset,e.claim,eligible);if(macro(cat)||String(e.asset).toUpperCase()==="MACRO"||/OFFICIAL_MACRO/i.test(String(e.event_kind)))aa=CORE.filter(x=>eligible.has(x.slice(7)));for(const a of aa){const w=await createWatch(e,a,bm.get(a));if(w)out.push(w);}}
  }catch(e){degraded.push(`discover:${err(e)}`);}
  return out;
}

async function expireOld(at:string,degraded:string[]){
  try{const q=await db.from("brian_catalyst_sentinel_watches").update({status:"EXPIRED",updated_at:at,last_state_change_at:at}).in("status",ACTIVE).lte("expires_at",at).select("watch_id");if(q.error)throw q.error;return(q.data??[]).length;}catch(e){degraded.push(`expire:${err(e)}`);return 0;}
}
async function retryPending(key:string,degraded:string[]){
  try{const q=await db.from("brian_catalyst_sentinel_alerts").select("alert_id,event_id,asset_id,metadata,observed_at").eq("alpha_dispatched",false).gte("observed_at",new Date(Date.now()-10*60_000).toISOString()).order("observed_at",{ascending:true}).limit(3);if(q.error)throw q.error;let n=0;for(const a of q.data??[]){const last=Date.parse(String((a.metadata as any)?.last_dispatch_attempt_at??""));if(Number.isFinite(last)&&Date.now()-last<30_000)continue;if(await dispatch(String(a.event_id),String(a.asset_id),String(a.alert_id),key))n++;}return n;}catch(e){degraded.push(`retry:${err(e)}`);return 0;}
}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return new Response("POST",{status:405});
  try{await auth(req);}catch(e){return new Response(JSON.stringify({status:"UNAUTHORIZED",error:err(e),version:V}),{status:401,headers:{"content-type":"application/json"}});}
  const key=(req.headers.get("x-brian-cron-key")??"").trim(),token=crypto.randomUUID();
  let leased=false,finalStatus="FAILED_CLOSED",finalError:string|null=null,stage="lease";
  try{
    leased=await acquireLease(token);if(!leased)return new Response(JSON.stringify({status:"WAIT_LEASE",version:V,shadow_only:true,live_execution:false}),{status:202,headers:{"content-type":"application/json"}});
    const degraded:string[]=[],at=new Date().toISOString();
    stage="expire";const expired=await expireOld(at,degraded);
    stage="core_inputs";const eligible=await getUniverse(degraded);const eps=await getEndpoints(degraded);const activeAssets=[...new Set([...CORE])];const bm=await getBooks(activeAssets,degraded);
    stage="feeds";const feeds=await Promise.all(eps.map(e=>feed(e,at,degraded)));
    stage="events";const events:any[]=[];for(const f of feeds)for(const i of f.items)if(relevant(f.e,i.title)){try{events.push(await upsertEvent(f.e,i,at));}catch(e){degraded.push(`event:${f.e.endpoint_id}:${err(e)}`);}}
    stage="watch_create";const created:any[]=[];for(const e of events){let aa=explicit(e.asset,e.claim,eligible);if(macro(String(e.metadata?.category??"")))aa=CORE.filter(x=>eligible.has(x.slice(7)));for(const a of aa){try{const w=await createWatch(e,a,bm.get(a));if(w)created.push(w);}catch(x){degraded.push(`watch:${e.event_id}:${a}:${err(x)}`);}}}
    created.push(...await discover(eligible,bm,degraded));
    stage="event_alerts";let detectedAlerts=0,detectedDispatches=0;for(const w of created){try{const a=await makeAlert(w,"EVENT_DETECTED",0,0,{ret:0,spread:bm.get(w.asset_id)?.spread??null},key);if(a.inserted)detectedAlerts++;if(a.dispatched){detectedDispatches++;const now=new Date().toISOString();await db.from("brian_catalyst_sentinel_watches").update({last_alpha_trigger_at:now,updated_at:now}).eq("watch_id",w.watch_id);}}catch(e){degraded.push(`event_alert:${w.watch_id}:${err(e)}`);}}
    stage="active_watches";let active:any[]=[];try{const q=await db.from("brian_catalyst_sentinel_watches").select("*").in("status",ACTIVE).gt("expires_at",new Date().toISOString()).order("started_at",{ascending:false}).limit(36);if(q.error)throw q.error;active=q.data??[];}catch(e){degraded.push(`active_watches:${err(e)}`);}
    const allAssets=[...new Set([...activeAssets,...active.map(w=>String(w.asset_id))])];if(allAssets.some(a=>!bm.has(a))){const b2=await getBooks(allAssets,degraded);for(const[a,b]of b2)bm.set(a,b);}
    stage="monitor";const mon=await monitor(active,bm,key,degraded);
    stage="retry";const retried=await retryPending(key,degraded);
    finalStatus=degraded.length?"DEGRADED":"SUCCESS";
    const body={status:finalStatus,version:V,feeds_checked:eps.length,feed_baselines:feeds.filter(f=>f.baseline).length,feed_changes:feeds.filter(f=>f.changed).length,events_detected:events.length,watches_created:created.length,watches_expired:expired,watches_evaluated:mon.evaluated,alerts_created:detectedAlerts+mon.alerts,alpha_dispatches:detectedDispatches+mon.dispatches+retried,pending_dispatches_retried:retried,degraded:degraded.slice(0,20),fast_lane_seconds:10,persistent_rechecks_minutes:RECHECK,watch_ttl_minutes:120,asset_binding:"macro_core_or_explicit_only",random_asset_mapping:false,hard_risk_gates_bypassed:false,scheduling_priority_bypass_only:true,shadow_only:true,live_execution:false};
    return new Response(JSON.stringify(body),{status:200,headers:{"content-type":"application/json","cache-control":"no-store"}});
  }catch(e){finalError=`${stage}:${err(e)}`;return new Response(JSON.stringify({status:"FAILED_CLOSED",stage,error:finalError,version:V,shadow_only:true,live_execution:false}),{status:500,headers:{"content-type":"application/json","cache-control":"no-store"}});}
  finally{if(leased)await releaseLease(token,finalStatus,finalError);}
});
