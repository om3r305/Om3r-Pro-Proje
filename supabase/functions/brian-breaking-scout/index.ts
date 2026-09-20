import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { requireCronAuth } from "../_shared/cron_auth.ts";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const VERSION="brian.breaking-scout.v2-realtime-scheduled";
const COLLECTOR_ID="brian-breaking-scout-v1";

type Feed={id:string;url:string;theme:string;trust:"OFFICIAL_PRIMARY"|"UNVERIFIED_DISCOVERY";kind:"RSS"|"ATOM"|"GOOGLE"};
type Article={title:string;url:string;publishedAt:string|null;sourceId:string;theme:string;trust:string;provider:string};

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

const GOOGLE=[
  {id:"google:geopolitics",theme:"geopolitics",q:'(war OR missile OR sanctions OR ceasefire OR invasion OR Red Sea OR Taiwan OR shipping OR attack) when:1h'},
  {id:"google:macro",theme:"macro_rates",q:'("Federal Reserve" OR ECB OR inflation OR CPI OR payrolls OR "interest rate" OR yields) when:1h'},
  {id:"google:energy",theme:"commodities_energy",q:'(oil OR Brent OR WTI OR OPEC OR natural gas OR refinery OR pipeline OR energy supply) when:1h'},
  {id:"google:ai",theme:"technology_ai",q:'(NVIDIA OR OpenAI OR semiconductor OR GPU OR TSMC OR AI datacenter) when:1h'},
];

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
  return {
    feeds:FEEDS.filter((_,i)=>i%2===slot%2),
    google:GOOGLE.filter((_,i)=>i%2===slot%2),
    slot
  };
}

function parseXml(xml:string,feed:Feed):Article[]{
  const out:Article[]=[];
  for(const m of xml.matchAll(/<item(?:\s[^>]*)?>([\s\S]*?)<\/item>/gi)){
    const b=m[1],title=tag(b,"title"),url=tag(b,"link")||tag(b,"guid"),pub=iso(tag(b,"pubDate")||tag(b,"dc:date"));
    const sourceName=tag(b,"source");
    const sourceUrl=attr(b,"source","url");
    if(!title||!url)continue;
    let sourceId=feed.id;
    if(feed.kind==="GOOGLE"){
      sourceId=sourceName||"google-news";
      try{sourceId=new URL(sourceUrl||url).hostname.replace(/^www\./,"")||sourceId}catch{}
    }
    out.push({title,url,publishedAt:pub,sourceId,theme:feed.theme,trust:feed.trust,provider:feed.kind==="GOOGLE"?"google_news_rss":"official_rss"});
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

async function fetchFeed(feed:Feed):Promise<Article[]>{
  const r=await fetch(feed.url,{headers:{accept:"application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.1","user-agent":"Mozilla/5.0 BrianBreakingScout/1.0"},signal:AbortSignal.timeout(7000)});
  if(!r.ok)throw new Error(`${feed.id}:HTTP_${r.status}`);
  return parseXml(await r.text(),feed);
}
async function fetchGoogle(row:{id:string;theme:string;q:string}):Promise<Article[]>{
  const params=new URLSearchParams({q:row.q,hl:"en-US",gl:"US",ceid:"US:en"});
  return fetchFeed({id:row.id,url:`https://news.google.com/rss/search?${params}`,theme:row.theme,trust:"UNVERIFIED_DISCOVERY",kind:"GOOGLE"});
}
async function recordRun(startedAt:string,status:string,observed:number,stored:number,degraded:string[]){
  const finishedAt=new Date().toISOString();
  const runId=await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  await db.from("brian_collector_runs").insert({
    run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:finishedAt,status,
    observed_records:observed,stored_records:stored,degraded_sources:degraded,
    error_class:status==="FAILED"?"BREAKING_SCOUT_ERROR":null,error_message:null,
    metadata:{version:VERSION,fast_lane:true,discovery_only:true,directional_vote:false,before_bbc_goal:true},
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
    const settled=await Promise.allSettled([
      ...selected.feeds.map(fetchFeed),
      ...selected.google.map(fetchGoogle),
    ]);
    const degraded:string[]=[];
    const articles:Article[]=[];
    settled.forEach((s,i)=>{
      if(s.status==="fulfilled")articles.push(...s.value);
      else degraded.push(i<selected.feeds.length?selected.feeds[i].id:selected.google[i-selected.feeds.length].id);
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
          source_latency_seconds:latency,fast_lane_version:VERSION
        }
      });
    }
    if(events.length){
      const q=await db.from("brian_intel_events").upsert(events,{onConflict:"event_id",ignoreDuplicates:true});
      if(q.error)throw q.error;
    }
    const status=degraded.length===settled.length?"FAILED":degraded.length?"DEGRADED":"SUCCESS";
    await recordRun(startedAt,status,articles.length,events.length,degraded);
    return out({status,version:VERSION,scheduler,slot:selected.slot,feeds:settled.length,observed:articles.length,candidates:events.length,degraded_sources:degraded,fast_lane:true,direct_alpha_influence:false,shadow_only:true,live_execution:false},status==="FAILED"?503:200);
  }catch(e){
    await recordRun(startedAt,"FAILED",0,0,[]);
    return out({status:"FAILED",error:errorText(e),shadow_only:true,live_execution:false},500);
  }
});