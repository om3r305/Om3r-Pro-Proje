import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { requireRealtimeInternal } from "../_shared/realtime_internal_auth.ts";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const VERSION="brian.realtime-multiasset-market-eye.v4-meta-price-fallback";
const COLLECTOR_ID="brian-realtime-multiasset-market-eye-v1";

type Spec={asset_id:string;asset_class:string;symbol:string;themes:string[];priority:number};
const FALLBACK_SPECS:Spec[]=[
  {asset_id:"fx:EURUSD",asset_class:"fx",symbol:"EURUSD=X",themes:["MONETARY","FX"],priority:100},
  {asset_id:"fx:GBPUSD",asset_class:"fx",symbol:"GBPUSD=X",themes:["MONETARY","FX"],priority:95},
  {asset_id:"fx:USDJPY",asset_class:"fx",symbol:"JPY=X",themes:["MONETARY","FX","GEOPOLITICAL"],priority:100},
  {asset_id:"index:SP500",asset_class:"index",symbol:"^GSPC",themes:["MONETARY","FINANCIAL","GEOPOLITICAL"],priority:100},
  {asset_id:"index:NASDAQ100",asset_class:"index",symbol:"^NDX",themes:["MONETARY","FINANCIAL","AI_TECH"],priority:100},
  {asset_id:"index:DAX",asset_class:"index",symbol:"^GDAXI",themes:["MONETARY","FINANCIAL","GEOPOLITICAL"],priority:95},
  {asset_id:"commodity:GOLD",asset_class:"commodity",symbol:"GC=F",themes:["MONETARY","FINANCIAL","FX","GEOPOLITICAL"],priority:100},
  {asset_id:"commodity:WTI",asset_class:"commodity",symbol:"CL=F",themes:["ENERGY","GEOPOLITICAL"],priority:100},
  {asset_id:"commodity:BRENT",asset_class:"commodity",symbol:"BZ=F",themes:["ENERGY","GEOPOLITICAL"],priority:100},
  {asset_id:"equity:NVDA",asset_class:"equity",symbol:"NVDA",themes:["AI_TECH"],priority:100},
];

function out(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}})}
function n(v:unknown):number|null{if(v==null||v==="")return null;const x=Number(v);return Number.isFinite(x)?x:null}
function clip(v:number,lo=-1,hi=1){return Math.max(lo,Math.min(hi,v))}
async function sha(s:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(s)));return [...d].map(b=>b.toString(16).padStart(2,"0")).join("")}
function errorText(e:unknown){return e instanceof Error?`${e.name}: ${e.message}`:String(e)}
function expectedMarketClosed(spec:Spec,at=new Date()){
  if(spec.asset_class==="crypto")return false;
  const day=at.getUTCDay(),hour=at.getUTCHours();
  return day===6 || (day===0 && hour<21);
}

async function loadSpecs():Promise<Spec[]>{
  const q=await db.from("brian_multiasset_instruments")
    .select("asset_id,asset_class,provider_symbol,themes,priority,active")
    .eq("active",true)
    .order("priority",{ascending:false})
    .order("asset_id",{ascending:true})
    .limit(36);
  if(q.error||!(q.data??[]).length)return FALLBACK_SPECS;
  return (q.data??[]).map((row:any)=>({
    asset_id:String(row.asset_id),
    asset_class:String(row.asset_class),
    symbol:String(row.provider_symbol),
    themes:Array.isArray(row.themes)?row.themes.map(String):[],
    priority:Number(row.priority??50),
  }));
}

async function fetchOne(spec:Spec){
  const endpoint=`https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(spec.symbol)}?interval=5m&range=1d&includePrePost=true&events=div%2Csplits`;
  const r=await fetch(endpoint,{
    headers:{accept:"application/json","user-agent":"Mozilla/5.0 Brian-Shadow-MultiAsset/1.0"},
    signal:AbortSignal.timeout(9000)
  });
  if(!r.ok)throw new Error(`${spec.asset_id}:HTTP_${r.status}`);
  const payload=await r.json();
  const result=payload?.chart?.result?.[0];
  if(!result)throw new Error(`${spec.asset_id}:NO_RESULT`);
  const ts:Array<number>=Array.isArray(result.timestamp)?result.timestamp:[];
  const q=result?.indicators?.quote?.[0]??{};
  const closes:Array<number|null>=Array.isArray(q.close)?q.close:[];
  const opens:Array<number|null>=Array.isArray(q.open)?q.open:[];
  const highs:Array<number|null>=Array.isArray(q.high)?q.high:[];
  const lows:Array<number|null>=Array.isArray(q.low)?q.low:[];
  const vols:Array<number|null>=Array.isArray(q.volume)?q.volume:[];
  const meta=result?.meta??{};
  let idx=-1;
  for(let i=Math.min(ts.length,closes.length)-1;i>=0;i--){if(n(closes[i])!==null){idx=i;break}}

  let price:number;
  let providerMs:number;
  let openPrice:number|null=null,highPrice:number|null=null,lowPrice:number|null=null,volume:number|null=null;
  let ret5:number|null=null,ret1h:number|null=null;
  let priceFallback:string|null=null;

  if(idx>=0){
    price=n(closes[idx])!;
    providerMs=Number(ts[idx])*1000;
    openPrice=n(opens[idx]);highPrice=n(highs[idx]);lowPrice=n(lows[idx]);volume=n(vols[idx]);
    const at=(offset:number)=>{const i=idx-offset;return i>=0?n(closes[i]):null};
    const p5=at(1),p1h=at(12);
    ret5=p5&&p5>0?price/p5-1:null;
    ret1h=p1h&&p1h>0?price/p1h-1:null;
  }else{
    const metaPrice=n(meta.regularMarketPrice);
    const metaTime=Number(meta.regularMarketTime)*1000;
    if(metaPrice===null)throw new Error(`${spec.asset_id}:NO_PRICE`);
    if(!Number.isFinite(metaTime)||metaTime<=0)throw new Error(`${spec.asset_id}:BAD_TIMESTAMP`);
    price=metaPrice;
    providerMs=metaTime;
    priceFallback="META_REGULAR_MARKET_PRICE";
  }

  if(!Number.isFinite(providerMs)||providerMs<=0)throw new Error(`${spec.asset_id}:BAD_TIMESTAMP`);
  const prev=n(meta.chartPreviousClose??meta.previousClose);
  const observed=new Date().toISOString();
  const providerTime=new Date(providerMs).toISOString();
  const latency=Math.max(0,(Date.now()-providerMs)/1000);
  const state=latency<=15*60?String(meta.marketState??"OPEN"):"STALE_OR_CLOSED";
  const markId=await sha(`${VERSION}|${spec.asset_id}|${providerTime}|${price}`);
  return {
    mark_id:markId,asset_id:spec.asset_id,asset_class:spec.asset_class,provider_symbol:spec.symbol,
    provider:"yahoo_chart_public",provider_quality:"PUBLIC_UNOFFICIAL_SHADOW_ONLY",
    observed_at:observed,provider_time:providerTime,price,
    open_price:openPrice,high_price:highPrice,low_price:lowPrice,
    previous_close:prev,volume,return_5m:ret5,return_1h:ret1h,
    session_state:state,data_latency_seconds:latency,
    metadata:{
      version:VERSION,currency:meta.currency??null,exchange_name:meta.exchangeName??null,
      instrument_type:meta.instrumentType??null,regular_market_time:meta.regularMarketTime??null,
      provider_endpoint:"query1.finance.yahoo.com/v8/finance/chart",price_fallback:priceFallback,
      execution_grade:false,public_unofficial_source:true,
      themes:spec.themes,priority:spec.priority
    },
    shadow_only:true,live_execution:false
  };
}
async function recordRun(startedAt:string,status:"SUCCESS"|"DEGRADED"|"FAILED",observed:number,stored:number,degraded:string[],marketClosed:string[],error?:unknown){
  const finishedAt=new Date().toISOString();
  const runId=await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  await db.from("brian_collector_runs").insert({
    run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:finishedAt,status,
    observed_records:observed,stored_records:stored,degraded_sources:degraded,
    error_class:error?"MULTIASSET_MARKET_EYE_ERROR":null,error_message:error?errorText(error).slice(0,1200):null,
    metadata:{version:VERSION,execution_grade:false,provider:"yahoo_chart_public",market_closed_sources:marketClosed,shadow_only:true,live_execution:false},
    evidence_class:"PROSPECTIVE_DEVELOPMENT_SHADOW",shadow_only:true,live_execution:false
  });
}
Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);
  try{await requireRealtimeInternal(req)}catch{return out({status:"UNAUTHORIZED"},401)}
  const startedAt=new Date().toISOString();
  try{
    const specs=await loadSpecs();
    const rows:any[]=[];const degraded:string[]=[];const marketClosed:string[]=[];
    for(let i=0;i<specs.length;i+=4){
      const batch=specs.slice(i,i+4);
      const settled=await Promise.allSettled(batch.map(fetchOne));
      settled.forEach((s,j)=>{
        if(s.status==="fulfilled"){rows.push(s.value);return}
        const spec=batch[j],message=errorText(s.reason);
        if(expectedMarketClosed(spec)&&message.includes(":NO_PRICE"))marketClosed.push(`${spec.asset_id}:MARKET_CLOSED`);
        else degraded.push(`${spec.asset_id}:${message}`);
      });
    }
    if(rows.length){
      const q=await db.from("brian_multiasset_market_marks").upsert(rows,{onConflict:"mark_id",ignoreDuplicates:true});
      if(q.error)throw new Error(`persist:${q.error.message}`);
    }
    const status=rows.length===0?"FAILED":degraded.length?"DEGRADED":"SUCCESS";
    await recordRun(startedAt,status,specs.length,rows.length,degraded,marketClosed,status==="FAILED"?new Error("no multiasset marks"):undefined);
    return out({
      status,version:VERSION,requested:specs.length,stored:rows.length,degraded_sources:degraded,market_closed_sources:marketClosed,
      marks:rows.map(r=>({asset_id:r.asset_id,price:r.price,provider_time:r.provider_time,session_state:r.session_state,data_latency_seconds:r.data_latency_seconds})),
      execution_grade:false,shadow_only:true,live_execution:false
    },status==="FAILED"?503:200);
  }catch(e){
    await recordRun(startedAt,"FAILED",0,0,[],[],e);
    return out({status:"FAILED_CLOSED",error:errorText(e),shadow_only:true,live_execution:false},500);
  }
});