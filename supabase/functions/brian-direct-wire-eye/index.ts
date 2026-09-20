import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { XMLParser } from "npm:fast-xml-parser@4.5.0";
import { requireRealtimeInternal } from "../_shared/realtime_internal_auth.ts";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const VERSION="brian.direct-wire-eye.v1";
const COLLECTOR_ID="brian-direct-wire-eye-v1";
const MAX_AGE_MS=100*60*1000;
const MAX_ITEMS=60;

type Json=Record<string,unknown>;
type FeedSpec={id:string;url:string;theme:string;publisher:string;mode:"MARKETS"|"GEOPOLITICS"};
type Article={title:string;description:string;url:string;publishedAt:string|null;publisher:string;feedId:string;theme:string};

const FEEDS:FeedSpec[]=[
  {id:"wire:yahoo-finance",url:"https://finance.yahoo.com/news/rssindex",theme:"markets_macro",publisher:"Yahoo Finance",mode:"MARKETS"},
  {id:"wire:bloomberg-markets",url:"https://feeds.bloomberg.com/markets/news.rss",theme:"markets_macro",publisher:"Bloomberg",mode:"MARKETS"},
  {id:"wire:cnbc-us",url:"https://www.cnbc.com/id/100003114/device/rss/rss.html",theme:"markets_macro",publisher:"CNBC",mode:"MARKETS"},
  {id:"wire:aljazeera",url:"https://www.aljazeera.com/xml/rss/all.xml",theme:"geopolitics",publisher:"Al Jazeera",mode:"GEOPOLITICS"},
];

const MARKET_RE=/\b(fed|federal reserve|fomc|kashkari|powell|warsh|waller|jefferson|bowman|cook|miran|inflation|cpi|pce|payroll|jobs report|unemployment|interest rate|rate hike|rate cut|treasury yield|bond yield|dollar|oil|brent|wti|opec|lng|iran|houthi|saudi|riyadh|yanbu|hormuz|bab el[- ]mandeb|missile|drone|sanction|ceasefire|blockade|refinery|tanker|shipping|tariff|nvidia|openai|tsmc|semiconductor|sec|crypto|bitcoin|ethereum|coinbase|stablecoin|etf|hack|exploit)\b/i;
const GEO_RE=/\b(iran|houthi|saudi|riyadh|yanbu|hormuz|bab el[- ]mandeb|red sea|israel|gaza|lebanon|russia|ukraine|china|taiwan|missile|drone|attack|war|sanction|ceasefire|blockade|refinery|tanker|shipping|pipeline|oil|lng)\b/i;
const HIGH_SIGNAL_PUBLISHER=/^(Reuters|Associated Press|AP|Bloomberg|CNBC|Financial Times|The Wall Street Journal|WSJ|Yahoo Finance|Al Jazeera|BBC)$/i;

function out(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}})}
function arr(v:unknown){return Array.isArray(v)?v:v==null?[]:[v]}
function txt(v:unknown):string{
  if(v==null)return "";
  if(typeof v==="string"||typeof v==="number"||typeof v==="boolean")return String(v);
  if(typeof v==="object"){
    const o=v as Json;
    return txt(o["#text"]??o["_text"]??o["__cdata"]??o["#cdata"]??"");
  }
  return "";
}
function link(v:unknown):string{
  if(typeof v==="string")return v.trim();
  if(v&&typeof v==="object"){
    const o=v as Json;
    return String(o["@_href"]??o["href"]??o["#text"]??"").trim();
  }
  return "";
}
function iso(v:unknown):string|null{
  const s=txt(v).trim();
  if(!s)return null;
  const d=new Date(s);
  return Number.isFinite(d.getTime())?d.toISOString():null;
}
function norm(v:string){return v.toLowerCase().replace(/[^a-z0-9]+/g," ").trim()}
async function sha(v:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(v)));return [...d].map(b=>b.toString(16).padStart(2,"0")).join("")}
function err(e:unknown){return e instanceof Error?`${e.name}: ${e.message}`:String(e)}
function publisherFrom(row:Json,fallback:string){
  const raw=row.source;
  const s=txt(raw).replace(/\s+/g," ").trim();
  return s||fallback;
}
function parseFeed(xml:string,spec:FeedSpec):Article[]{
  const parser=new XMLParser({ignoreAttributes:false,trimValues:true,processEntities:true});
  const doc=parser.parse(xml) as Json;
  const channel=(doc.rss as Json|undefined)?.channel as Json|undefined;
  const atom=doc.feed as Json|undefined;
  const rows=arr((channel?.item??atom?.entry) as unknown);
  const items:Article[]=[];
  for(const raw of rows.slice(0,MAX_ITEMS)){
    if(!raw||typeof raw!=="object")continue;
    const row=raw as Json;
    const title=txt(row.title).replace(/\s+/g," ").trim();
    const description=txt(row.description??row.summary??row["content:encoded"]).replace(/\s+/g," ").trim().slice(0,1800);
    let href="";
    for(const x of arr(row.link)){href=link(x);if(href)break}
    const publishedAt=iso(row.pubDate??row.published??row.updated??row["dc:date"]);
    const publisher=publisherFrom(row,spec.publisher);
    if(!title||!href)continue;
    items.push({title:title.slice(0,1500),description,url:href,publishedAt,publisher,feedId:spec.id,theme:spec.theme});
  }
  return items;
}
function isFresh(a:Article,now:number){
  if(!a.publishedAt)return true;
  const t=Date.parse(a.publishedAt);
  return Number.isFinite(t)&&t<=now+5*60_000&&now-t<=MAX_AGE_MS;
}
function relevant(a:Article,spec:FeedSpec){
  const text=`${a.title} ${a.description}`;
  if(spec.mode==="GEOPOLITICS")return GEO_RE.test(text);
  return MARKET_RE.test(text);
}
function highSignal(a:Article){
  return HIGH_SIGNAL_PUBLISHER.test(a.publisher)||/Reuters|Associated Press|Bloomberg|CNBC|Federal Reserve/i.test(a.description);
}
function sourceId(a:Article,spec:FeedSpec){
  const p=a.publisher.trim();
  if(p&&p.toLowerCase()!==spec.publisher.toLowerCase())return p;
  return spec.publisher;
}
async function fetchFeed(spec:FeedSpec){
  const r=await fetch(spec.url,{
    headers:{accept:"application/rss+xml,application/atom+xml,application/xml,text/xml;q=0.9,*/*;q=0.1","user-agent":"Mozilla/5.0 (compatible; BrianDirectWireEye/1.0; +market-intelligence)"},
    signal:AbortSignal.timeout(6500)
  });
  if(!r.ok)throw new Error(`${spec.id}:HTTP_${r.status}`);
  const body=await r.text();
  if(!body.trim())throw new Error(`${spec.id}:EMPTY`);
  return parseFeed(body,spec);
}
async function recordRun(startedAt:string,status:string,observed:number,stored:number,degraded:string[],metadata:Json){
  const finishedAt=new Date().toISOString();
  const runId=await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  const q=await db.from("brian_collector_runs").insert({
    run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:finishedAt,status,
    observed_records:observed,stored_records:stored,degraded_sources:degraded.slice(0,16),
    error_class:status==="FAILED"?"DIRECT_WIRE_EYE_ERROR":null,error_message:null,
    evidence_class:"PROSPECTIVE_DEVELOPMENT_SHADOW",shadow_only:true,live_execution:false,
    metadata:{version:VERSION,direct_wire:true,discovery_only:true,directional_vote:false,direct_alpha_influence:false,...metadata}
  });
  if(q.error)console.error("collector receipt",q.error.message);
}

Deno.serve(async(req:Request)=>{
  if(req.method==="GET")return out({status:"OK",version:VERSION,shadow_only:true,live_execution:false});
  if(req.method!=="POST")return out({error:"POST required"},405);
  try{await requireRealtimeInternal(req)}catch{return out({status:"UNAUTHORIZED"},401)}
  const startedAt=new Date().toISOString();
  try{
    const settled=await Promise.allSettled(FEEDS.map(fetchFeed));
    const now=Date.now();
    const degraded:string[]=[];
    const providerStats:Record<string,unknown>={};
    const candidates:Article[]=[];
    let observed=0;
    settled.forEach((s,i)=>{
      const spec=FEEDS[i];
      if(s.status==="rejected"){
        degraded.push(spec.id);
        providerStats[spec.id]={status:"FAILED",error:err(s.reason).slice(0,220)};
        return;
      }
      observed+=s.value.length;
      const fresh=s.value.filter(a=>isFresh(a,now));
      const rel=fresh.filter(a=>relevant(a,spec));
      const selected=rel.filter(a=>spec.id!=="wire:yahoo-finance"||highSignal(a));
      candidates.push(...selected);
      providerStats[spec.id]={status:"SUCCESS",observed:s.value.length,fresh:fresh.length,relevant:selected.length};
    });

    const unique=new Map<string,Article>();
    for(const a of candidates){
      const k=`${sourceId(a,FEEDS.find(f=>f.id===a.feedId)!)}|${norm(a.title)}`;
      if(!unique.has(k))unique.set(k,a);
    }
    const events:Json[]=[];
    for(const a of unique.values()){
      const spec=FEEDS.find(f=>f.id===a.feedId)!;
      const observedAt=new Date().toISOString();
      const publisher=sourceId(a,spec);
      const fingerprint=await sha(`${publisher}|${a.url}|${a.title}`);
      const eventId=await sha(`${COLLECTOR_ID}|${fingerprint}`);
      const publishedMs=a.publishedAt?Date.parse(a.publishedAt):NaN;
      const latency=Number.isFinite(publishedMs)?Math.max(0,Math.round((Date.parse(observedAt)-publishedMs)/1000)):null;
      const macro=/\b(fed|federal reserve|fomc|kashkari|powell|warsh|waller|jefferson|bowman|cook|miran|inflation|cpi|pce|interest rate|rate hike|rate cut|yield|payroll|unemployment)\b/i.test(a.title+" "+a.description);
      const geopolitical=GEO_RE.test(a.title+" "+a.description);
      events.push({
        event_id:eventId,asset:"GLOBAL_WORLD",event_kind:"DIRECT_WIRE_DISCOVERY",
        source_kind:"DIRECT_WIRE_INDEPENDENT",source_id:publisher,
        published_at:a.publishedAt,first_observed_at:observedAt,captured_at:observedAt,
        claim:a.title,direction:0,magnitude:macro||geopolitical?0.62:0.48,
        trust_class:"INDEPENDENT_PROFESSIONAL",entity_confidence:0.72,
        content_fingerprint:fingerprint,corroboration_key:await sha(`${a.theme}|${norm(a.title)}`),
        provenance_uri:a.url,pit_verified:true,raw_capture_id:null,
        metadata:{
          direct_wire:true,feed_id:a.feedId,publisher:a.publisher,world_theme:a.theme,
          discovery_only:true,requires_truth_engine:true,directional_vote:false,direct_alpha_influence:false,
          macro_watch:macro,geopolitical_watch:geopolitical,source_latency_seconds:latency,
          summary_hint:a.description.slice(0,700),runtime_version:VERSION
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
    const successFeeds=FEEDS.length-degraded.length;
    const status=successFeeds===0?"FAILED":degraded.length?"DEGRADED":"SUCCESS";
    const coverageState=successFeeds<2?"PARTIAL_PROVIDER":events.length?"ACTIVE":"SCANNING_NO_NEW_CRITICAL_DATA";
    await recordRun(startedAt,status,observed,stored,degraded,{
      feeds_expected:FEEDS.length,feeds_healthy:successFeeds,coverage_state:coverageState,
      candidate_records:events.length,provider_stats:providerStats
    });
    return out({
      status,version:VERSION,observed,candidates:events.length,stored,
      feeds_expected:FEEDS.length,feeds_healthy:successFeeds,degraded_sources:degraded,
      coverage_state:coverageState,provider_stats:providerStats,
      shadow_only:true,live_execution:false
    },status==="FAILED"?503:200);
  }catch(e){
    await recordRun(startedAt,"FAILED",0,0,[],{fatal_error:err(e).slice(0,500)});
    return out({status:"FAILED",version:VERSION,error:err(e),shadow_only:true,live_execution:false},500);
  }
});