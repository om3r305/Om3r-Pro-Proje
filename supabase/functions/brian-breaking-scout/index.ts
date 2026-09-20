import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { requireCronAuth } from "../_shared/cron_auth.ts";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const VERSION="brian.breaking-scout.v9-or-web-fallback";
const COLLECTOR_ID="brian-breaking-scout-v1";

type Feed={id:string;url:string;theme:string;trust:"OFFICIAL_PRIMARY"|"UNVERIFIED_DISCOVERY";kind:"RSS"|"ATOM"|"GOOGLE"|"BING"|"BING_WEB"};
type Article={title:string;url:string;publishedAt:string|null;sourceId:string;theme:string;trust:string;provider:string;laneId?:string;critical?:boolean};

const FEEDS:Feed[]=[
  {id:"official:fed:monetary",url:"https://www.federalreserve.gov/feeds/press_monetary.xml",theme:"macro_rates",trust:"OFFICIAL_PRIMARY",kind:"RSS"},
  {id:"official:ecb:press",url:"https://www.ecb.europa.eu/rss/press.html",theme:"macro_rates",trust:"OFFICIAL_PRIMARY",kind:"RSS"},
  {id:"official:bls:cpi",url:"https://www.bls.gov/feed/cpi.rss",theme:"macro_rates",trust:"OFFICIAL_PRIMARY",kind:"RSS"},
  {id:"official:bls:employment",url:"https://www.bls.gov/feed/empsit.rss",theme:"macro_rates",trust:"OFFICIAL_PRIMARY",kind:"RSS"},
  {id:"official:boj:whatsnew",url:"https://www.boj.or.jp/en/rss/whatsnew.xml",theme:"macro_rates",trust:"OFFICIAL_PRIMARY",kind:"RSS"},
  {id:"official:eucouncil:press",url:"https://www.consilium.europa.eu/en/rss/pressreleases.ashx",theme:"geopolitics",trust:"OFFICIAL_PRIMARY",kind:"RSS"},
  {id:"official:sec:press",url:"https://www.sec.gov/news/pressreleases.rss",theme:"regulation",trust:"OFFICIAL_PRIMARY",kind:"RSS"},
  {id:"official:bundesbank:general",url:"https://www.bundesbank.de/service/rss/de/633290/feed.rss",theme:"macro_rates",trust:"OFFICIAL_PRIMARY",kind:"RSS"},
];

const DISCOVERY=[
  {id:"discovery:geopolitics",theme:"geopolitics",priority:"CRITICAL",googleQ:'(war OR missile OR drone OR sanctions OR ceasefire OR invasion OR escalation OR retaliation OR "Red Sea" OR Taiwan OR shipping OR blockade OR attack) when:1h',fallbackQ:'(war OR missile OR drone OR sanctions OR ceasefire OR invasion OR escalation OR retaliation OR "Red Sea" OR Taiwan OR shipping OR blockade OR attack)',compactQ:'(war OR missile OR drone OR sanctions OR escalation OR retaliation OR attack)'},
  {id:"discovery:energy",theme:"commodities_energy",priority:"CRITICAL",googleQ:'(oil OR Brent OR WTI OR OPEC OR "natural gas" OR LNG OR refinery OR pipeline OR tanker OR terminal OR "energy supply" OR Hormuz OR "Bab el-Mandeb") when:1h',fallbackQ:'(oil OR Brent OR WTI OR OPEC OR "natural gas" OR LNG OR refinery OR pipeline OR tanker OR terminal OR "energy supply" OR Hormuz OR "Bab el-Mandeb")',compactQ:'(oil OR OPEC OR LNG OR refinery OR tanker OR Hormuz)'},
  {id:"discovery:middle-east-risk",theme:"geopolitics",priority:"CRITICAL",googleQ:'(Iran OR Houthi OR Saudi OR Riyadh OR Yanbu OR Hormuz OR "Bab el-Mandeb" OR "Red Sea") (attack OR missile OR drone OR escalation OR retaliation OR tanker OR port OR refinery OR shipping) when:1h',fallbackQ:'(Iran OR Houthi OR Saudi OR Riyadh OR Yanbu OR Hormuz OR "Bab el-Mandeb" OR "Red Sea") (attack OR missile OR drone OR escalation OR retaliation OR tanker OR port OR refinery OR shipping)',compactQ:'(Iran OR Houthi OR Saudi OR Riyadh OR Hormuz) (missile OR drone OR attack OR escalation)'},
  {id:"discovery:reuters-market-wire",theme:"geopolitics",priority:"CRITICAL",googleQ:'site:reuters.com (Iran OR Houthi OR Saudi OR Riyadh OR Hormuz OR oil OR refinery OR missile OR drone OR sanctions OR "Federal Reserve" OR SEC OR NVIDIA) when:1h',fallbackQ:'site:reuters.com (Iran OR Houthi OR Saudi OR Riyadh OR Hormuz OR oil OR refinery OR missile OR drone OR sanctions OR "Federal Reserve" OR SEC OR NVIDIA)',compactQ:'site:reuters.com (Iran OR Houthi OR Saudi OR Hormuz OR oil OR attack)'},
  {id:"discovery:macro",theme:"macro_rates",priority:"NORMAL",googleQ:'("Federal Reserve" OR ECB OR inflation OR CPI OR PCE OR payrolls OR unemployment OR "interest rate" OR yields OR GDP) when:1h',fallbackQ:'("Federal Reserve" OR ECB OR inflation OR CPI OR PCE OR payrolls OR unemployment OR "interest rate" OR yields OR GDP)',compactQ:'("Federal Reserve" OR ECB OR inflation OR CPI OR PCE)'},
  {id:"discovery:ai",theme:"technology_ai",priority:"NORMAL",googleQ:'(NVIDIA OR OpenAI OR semiconductor OR GPU OR TSMC OR HBM OR "memory chip" OR "AI datacenter" OR "export control") when:1h',fallbackQ:'(NVIDIA OR OpenAI OR semiconductor OR GPU OR TSMC OR HBM OR "memory chip" OR "AI datacenter" OR "export control")',compactQ:'(NVIDIA OR OpenAI OR semiconductor OR TSMC OR HBM)'},
] as const;

function out(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}})}
function decode(v:string){return v.replace(/^<!\[CDATA\[|\]\]>$/g,"").replace(/&amp;/g,"&").replace(/&quot;/g,'"').replace(/&#39;|&apos;/g,"'").replace(/&lt;/g,"<").replace(/&gt;/g,">")}
function tag(block:string,name:string){const m=block.match(new RegExp(`<${name}(?:\\s[^>]*)?>([\\s\\S]*?)<\\/${name}>`,"i"));return m?decode(m[1].trim()).replace(/<[^>]+>/g,"").trim():""}
function attr(block:string,name:string,attrName:string){const m=block.match(new RegExp(`<${name}[^>]*${attrName}=["']([^"']+)["'][^>]*>`,"i"));return m?decode(m[1]):""}
function iso(v:string){if(!v)return null;const d=new Date(v);return Number.isFinite(d.getTime())?d.toISOString():null}
async function sha(v:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(v)));return [...d].map(b=>b.toString(16).padStart(2,"0")).join("")}
function errorText(e:unknown){return e instanceof Error?`${e.name}: ${e.message}`:String(e)}
const REALTIME_INTERNAL_KEY_SHA256="b0549b2b41a5b832b37455389583e1d166d210490a8c6fe43cda2748aca7c38a";
function ct(a:string,b:string){if(a.length!==b.length)return false;let x=0;for(let i=0;i<a.length;i++)x|=a.charCodeAt(i)^b.charCodeAt(i);return x===0}
async function authorize(req:Request):Promise<"REALTIME_ORCHESTRATOR"|"LEGACY_CORE_CRON">{
  const internal=(req.headers.get("x-brian-internal-key")??"").trim();
  if(internal){
    if(!ct(await sha(internal),REALTIME_INTERNAL_KEY_SHA256))throw new Error("UNAUTHORIZED_INTERNAL");
    return "REALTIME_ORCHESTRATOR";
  }
  await requireCronAuth(req,db);
  return "LEGACY_CORE_CRON";
}
function selectedFeeds(){
  const slot=Math.floor(Date.now()/120000);
  const critical=DISCOVERY.filter((row)=>row.priority==="CRITICAL");
  const rotating=DISCOVERY.filter((row)=>row.priority!=="CRITICAL").filter((_,i)=>i%2===slot%2);
  return {
    feeds:FEEDS.filter((_,i)=>i%2===slot%2),
    discovery:[...critical,...rotating],
    slot
  };
}

function parseXml(xml:string,feed:Feed):Article[]{
  const out:Article[]=[];
  for(const m of xml.matchAll(/<item(?:\s[^>]*)?>([\s\S]*?)<\/item>/gi)){
    const b=m[1],title=tag(b,"title"),url=tag(b,"link")||tag(b,"guid"),pub=iso(tag(b,"pubDate")||tag(b,"dc:date"));
    const sourceName=tag(b,"source")||tag(b,"News:Source")||tag(b,"news:source");
    const sourceUrl=attr(b,"source","url");
    if(!title||!url)continue;
    let sourceId=feed.id;
    if(feed.kind==="GOOGLE"||feed.kind==="BING"||feed.kind==="BING_WEB"){
      sourceId=sourceName||(feed.kind==="GOOGLE"?"google-news":feed.kind==="BING_WEB"?"bing-web":"bing-news");
      try{
        const host=new URL(sourceUrl||url).hostname.replace(/^www\./,"");
        if(host&&!/^(news\.google\.com|www\.bing\.com|bing\.com)$/i.test(host))sourceId=host;
      }catch{}
    }
    out.push({title,url,publishedAt:pub,sourceId,theme:feed.theme,trust:feed.trust,provider:feed.kind==="GOOGLE"?"google_news_rss":feed.kind==="BING"?"bing_news_rss":feed.kind==="BING_WEB"?"bing_web_rss":"official_rss"});
    if(out.length>=30)break;
  }
  for(const m of xml.matchAll(/<entry(?:\s[^>]*)?>([\s\S]*?)<\/entry>/gi)){
    const b=m[1],title=tag(b,"title");
    const url=attr(b,"link","href")||tag(b,"link");
    const pub=iso(tag(b,"published")||tag(b,"updated"));
    if(!title||!url)continue;
    out.push({title,url,publishedAt:pub,sourceId:feed.id,theme:feed.theme,trust:feed.trust,provider:"official_atom"});
    if(out.length>=30)break;
  }
  return out;
}

async function fetchFeed(feed:Feed,timeoutMs=5500,allowEmpty=false):Promise<Article[]>{
  const r=await fetch(feed.url,{
    headers:{
      accept:"application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.1",
      "user-agent":"Mozilla/5.0 (compatible; BrianBreakingScout/3.0; +market-intelligence)"
    },
    signal:AbortSignal.timeout(timeoutMs)
  });
  if(!r.ok)throw new Error(`${feed.id}:HTTP_${r.status}`);
  const body=await r.text();
  const rows=parseXml(body,feed);
  if(!rows.length){
    const looksLikeFeed=/<rss\b|<feed\b|<channel\b/i.test(body);
    if(allowEmpty&&looksLikeFeed)return [];
    throw new Error(`${feed.id}:EMPTY_OR_UNPARSEABLE`);
  }
  return rows;
}
function freshDiscoveryArticles(rows:Article[],maxAgeMs=2*60*60_000){
  const now=Date.now();
  return rows.filter((a)=>{
    if(!a.publishedAt)return true;
    const t=Date.parse(a.publishedAt);
    return !Number.isFinite(t)||now-t<=maxAgeMs;
  });
}
function uniqueArticles(rows:Article[]){
  const seen=new Set<string>();
  return rows.filter((a)=>{
    const k=(a.url||"")+"|"+a.title.toLowerCase();
    if(seen.has(k))return false;
    seen.add(k);
    return true;
  });
}
const TRUSTED_WEB_DISCOVERY_DOMAINS=[
  "reuters.com","apnews.com","bbc.com","bbc.co.uk","bloomberg.com","ft.com","wsj.com",
  "cnbc.com","cnn.com","theguardian.com","aljazeera.com"
];
function trustedWebDiscovery(a:Article){
  const s=String(a.sourceId||"").toLowerCase().replace(/^www\./,"");
  return TRUSTED_WEB_DISCOVERY_DOMAINS.some((d)=>s===d||s.endsWith("."+d));
}
async function fetchDiscovery(row:{id:string;theme:string;priority:string;googleQ:string;fallbackQ:string;compactQ:string}):Promise<{articles:Article[];provider:string;providerErrors:string[]}>{
  const providerErrors:string[]=[];
  const tagRows=(rows:Article[])=>freshDiscoveryArticles(rows).map((a)=>({...a,laneId:row.id,critical:row.priority==="CRITICAL"}));

  // Critical lanes use two independent discovery paths in parallel. A provider
  // returning only stale rows no longer counts as useful coverage.
  if(row.priority==="CRITICAL"){
    const bingParams=new URLSearchParams({q:row.fallbackQ,format:"rss",setlang:"en-us",mkt:"en-US",qft:'sortbydate="1" interval="4"',form:"YFNR"});
    const googleParams=new URLSearchParams({q:row.googleQ,hl:"en-US",gl:"US",ceid:"US:en"});
    const [bingResult,googleResult]=await Promise.allSettled([
      fetchFeed({
        id:row.id+":bing",
        url:`https://www.bing.com/news/search?${bingParams}`,
        theme:row.theme,trust:"UNVERIFIED_DISCOVERY",kind:"BING"
      },2200,true),
      fetchFeed({
        id:row.id+":google",
        url:`https://news.google.com/rss/search?${googleParams}`,
        theme:row.theme,trust:"UNVERIFIED_DISCOVERY",kind:"GOOGLE"
      },1800,true)
    ]);
    const merged:Article[]=[];
    if(bingResult.status==="fulfilled"){
      const fresh=tagRows(bingResult.value);
      if(fresh.length)merged.push(...fresh);
      else providerErrors.push("bing-full:STALE_ONLY");
    }else providerErrors.push("bing-full:"+errorText(bingResult.reason).slice(0,180));
    if(googleResult.status==="fulfilled"){
      const fresh=tagRows(googleResult.value);
      if(fresh.length)merged.push(...fresh);
      else providerErrors.push("google:STALE_ONLY");
    }else providerErrors.push("google:"+errorText(googleResult.reason).slice(0,180));
    if(merged.length){
      return {articles:uniqueArticles(merged),provider:"bing+google_critical",providerErrors};
    }
    let compactAvailable=false;
    try{
      const compactParams=new URLSearchParams({q:row.compactQ,format:"rss",setlang:"en-us",mkt:"en-US",qft:'sortbydate="1" interval="4"',form:"YFNR"});
      const raw=await fetchFeed({
        id:row.id+":bing-compact",
        url:`https://www.bing.com/news/search?${compactParams}`,
        theme:row.theme,trust:"UNVERIFIED_DISCOVERY",kind:"BING"
      },1800,true);
      compactAvailable=true;
      const fresh=tagRows(raw);
      if(fresh.length)return {articles:uniqueArticles(fresh),provider:"bing_news_rss_compact",providerErrors};
    }catch(e){providerErrors.push("bing-compact:"+errorText(e).slice(0,180))}
    let webAvailable=false;
    try{
      const webParams=new URLSearchParams({q:row.fallbackQ,format:"rss",setlang:"en-us",mkt:"en-US"});
      const raw=await fetchFeed({
        id:row.id+":bing-web",
        url:`https://www.bing.com/search?${webParams}`,
        theme:row.theme,trust:"UNVERIFIED_DISCOVERY",kind:"BING_WEB"
      },1800,true);
      webAvailable=true;
      const fresh=tagRows(raw).filter(trustedWebDiscovery);
      if(fresh.length)return {articles:uniqueArticles(fresh),provider:"bing_web_rss",providerErrors};
    }catch(e){providerErrors.push("bing-web:"+errorText(e).slice(0,180))}
    const primaryAvailable=bingResult.status==="fulfilled"||googleResult.status==="fulfilled";
    if(primaryAvailable||compactAvailable||webAvailable){
      return {articles:[],provider:webAvailable?"critical_scan_with_web":bingResult.status==="fulfilled"&&googleResult.status==="fulfilled"?"bing+google_critical":"critical_partial_provider",providerErrors};
    }
    throw new Error(`${row.id}:ALL_CRITICAL_PROVIDERS_FAILED:${providerErrors.join("|")}`);
  }

  // Normal lanes stay cheap: Bing first, Google only when Bing is unusable.
  try{
    const params=new URLSearchParams({q:row.fallbackQ,format:"rss",setlang:"en-us",mkt:"en-US",qft:'sortbydate="1" interval="4"',form:"YFNR"});
    const articles=tagRows(await fetchFeed({
      id:row.id+":bing",
      url:`https://www.bing.com/news/search?${params}`,
      theme:row.theme,trust:"UNVERIFIED_DISCOVERY",kind:"BING"
    },3000,true));
    return {articles:uniqueArticles(articles),provider:"bing_news_rss",providerErrors};
  }catch(e){providerErrors.push("bing-full:"+errorText(e).slice(0,180))}
  try{
    const params=new URLSearchParams({q:row.googleQ,hl:"en-US",gl:"US",ceid:"US:en"});
    const articles=tagRows(await fetchFeed({
      id:row.id+":google",
      url:`https://news.google.com/rss/search?${params}`,
      theme:row.theme,trust:"UNVERIFIED_DISCOVERY",kind:"GOOGLE"
    },2200,true));
    return {articles:uniqueArticles(articles),provider:"google_news_rss",providerErrors};
  }catch(e){providerErrors.push("google:"+errorText(e).slice(0,180))}
  throw new Error(`${row.id}:ALL_DISCOVERY_PROVIDERS_FAILED:${providerErrors.join("|")}`);
}
async function recordRun(startedAt:string,status:string,observed:number,stored:number,degraded:string[],metadata:Record<string,unknown>={}){
  const finishedAt=new Date().toISOString();
  const runId=await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  await db.from("brian_collector_runs").insert({
    run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:finishedAt,status,
    observed_records:observed,stored_records:stored,degraded_sources:degraded,
    error_class:status==="FAILED"?"BREAKING_SCOUT_ERROR":null,error_message:null,
    metadata:{version:VERSION,fast_lane:true,discovery_only:true,directional_vote:false,before_bbc_goal:true,...metadata},
    evidence_class:"PROSPECTIVE_DEVELOPMENT_SHADOW",shadow_only:true,live_execution:false
  });
}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);
  const startedAt=new Date().toISOString();
  let scheduler:"REALTIME_ORCHESTRATOR"|"LEGACY_CORE_CRON";
  try{scheduler=await authorize(req)}catch(e){return out({status:"UNAUTHORIZED",error:errorText(e)},401)}
  if(scheduler==="LEGACY_CORE_CRON"){
    return out({status:"MOVED_TO_REALTIME_ORCHESTRATOR",version:VERSION,shadow_only:true,live_execution:false});
  }
  try{
    const selected=selectedFeeds();
    const [officialSettled,discoverySettled]=await Promise.all([
      Promise.allSettled(selected.feeds.map((feed)=>fetchFeed(feed))),
      Promise.allSettled(selected.discovery.map((row)=>fetchDiscovery(row))),
    ]);
    const degraded:string[]=[];
    const articles:Article[]=[];
    const providerErrors:Record<string,string[]|string>={};
    const discoveryProviders:Record<string,string>={};

    officialSettled.forEach((s,i)=>{
      const id=selected.feeds[i].id;
      if(s.status==="fulfilled")articles.push(...s.value);
      else{
        degraded.push(id);
        providerErrors[id]=errorText(s.reason).slice(0,240);
      }
    });
    discoverySettled.forEach((s,i)=>{
      const id=selected.discovery[i].id;
      if(s.status==="fulfilled"){
        articles.push(...s.value.articles);
        discoveryProviders[id]=s.value.provider;
        if(s.value.providerErrors.length)providerErrors[id]=s.value.providerErrors;
      }else{
        degraded.push(id);
        providerErrors[id]=errorText(s.reason).slice(0,320);
      }
    });

    const now=Date.now();
    const events:any[]=[];
    for(const a of articles){
      const pubMs=a.publishedAt?Date.parse(a.publishedAt):NaN;
      if(Number.isFinite(pubMs)&&now-pubMs>2*60*60_000)continue;
      const observedAt=new Date().toISOString();
      const fingerprint=await sha(`${a.sourceId}|${a.url}|${a.title}`);
      const eventId=await sha(`${COLLECTOR_ID}|${fingerprint}`);
      const normalized=a.title.toLowerCase().replace(/[^a-z0-9]+/g," ").trim();
      const latency=Number.isFinite(pubMs)?Math.max(0,Math.round((Date.parse(observedAt)-pubMs)/1000)):null;
      events.push({
        event_id:eventId,asset:"GLOBAL_WORLD",event_kind:"BREAKING_SCOUT",
        source_kind:a.trust==="OFFICIAL_PRIMARY"?"BREAKING_FAST_OFFICIAL":"BREAKING_FAST_DISCOVERY",
        source_id:a.sourceId,published_at:a.publishedAt,first_observed_at:observedAt,captured_at:observedAt,
        claim:a.title,direction:0,magnitude:a.trust==="OFFICIAL_PRIMARY"?0.45:0.30,
        trust_class:a.trust,entity_confidence:a.trust==="OFFICIAL_PRIMARY"?0.70:0.35,
        content_fingerprint:fingerprint,corroboration_key:await sha(`${a.theme}|${normalized}`),
        provenance_uri:a.url,pit_verified:true,raw_capture_id:null,
        metadata:{
          breaking_fast_lane:true,world_theme:a.theme,provider:a.provider,discovery_only:true,
          directional_vote:false,requires_truth_engine:true,before_bbc_goal:true,
          discovery_lane:a.laneId??null,critical_watch:a.critical===true,
          source_latency_seconds:latency,fast_lane_version:VERSION
        }
      });
    }
    let stored=0;
    if(events.length){
      const q=await db.from("brian_intel_events")
        .upsert(events,{onConflict:"event_id",ignoreDuplicates:true})
        .select("event_id");
      if(q.error)throw q.error;
      stored=Array.isArray(q.data)?q.data.length:0;
    }
    const totalSources=officialSettled.length+discoverySettled.length;
    const selectedCritical=selected.discovery.filter((row)=>row.priority==="CRITICAL").map((row)=>row.id);
    const failedCritical=selectedCritical.filter((id)=>degraded.includes(id));
    const criticalCandidates=events.filter((row:any)=>row.metadata?.critical_watch===true).length;
    const criticalSources=[...new Set(events.filter((row:any)=>row.metadata?.critical_watch===true).map((row:any)=>String(row.source_id||"")).filter(Boolean))];
    const coverageState=failedCritical.length?"DEGRADED":criticalCandidates>0?"ACTIVE":"SCANNING_NO_CRITICAL_NEW_DATA";
    const status=degraded.length===totalSources?"FAILED":degraded.length||failedCritical.length?"DEGRADED":"SUCCESS";
    await recordRun(startedAt,status,articles.length,stored,degraded,{
      provider_errors:providerErrors,
      discovery_providers:discoveryProviders,
      candidate_records:events.length,
      critical_lanes_expected:selectedCritical.length,
      critical_lanes_failed:failedCritical,
      critical_candidates:criticalCandidates,
      critical_sources:criticalSources.slice(0,12),
      coverage_state:coverageState
    });
    return out({
      status,version:VERSION,scheduler,slot:selected.slot,feeds:totalSources,
      observed:articles.length,candidates:events.length,stored,degraded_sources:degraded,
      discovery_providers:discoveryProviders,provider_errors:providerErrors,
      critical_lanes:selected.discovery.filter((row)=>row.priority==="CRITICAL").map((row)=>row.id),
      critical_candidates:events.filter((row:any)=>row.metadata?.critical_watch===true).length,
      coverage_state:selected.discovery.some((row)=>row.priority==="CRITICAL"&&degraded.includes(row.id))?"DEGRADED":events.some((row:any)=>row.metadata?.critical_watch===true)?"ACTIVE":"SCANNING_NO_CRITICAL_NEW_DATA",
      fast_lane:true,direct_alpha_influence:false,shadow_only:true,live_execution:false
    },status==="FAILED"?503:200);
  }catch(e){
    await recordRun(startedAt,"FAILED",0,0,[],{fatal_error:errorText(e).slice(0,500)});
    return out({status:"FAILED",error:errorText(e),shadow_only:true,live_execution:false},500);
  }
});