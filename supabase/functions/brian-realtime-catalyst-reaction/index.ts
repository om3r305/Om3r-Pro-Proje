import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { requireRealtimeInternal } from "../_shared/realtime_internal_auth.ts";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });

const VERSION = "brian.realtime-catalyst-reaction.v5";
const ALPHA_RECHECK_URL = "https://dliediwlldojkfjzlznm.supabase.co/functions/v1/brian-realtime-alpha-recheck";
const RECHECK_MINUTES = [5,10,15,30] as const;

type Json = Record<string, unknown>;
type Watch = {
  watch_id:string; event_id:string; asset_id:string; started_at:string; status:string;
  direction:number; reference_price:number|null; recheck_count:number;
};

function out(body:unknown,status=200){
  return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}});
}
function finite(v:unknown,d=0){const n=Number(v);return Number.isFinite(n)?n:d}
function clip(v:number){return Math.max(0,Math.min(1,v))}
function errText(e:unknown){
  if(e instanceof Error)return `${e.name}: ${e.message}`;
  if(e&&typeof e==="object"){
    try{return JSON.stringify(e)}catch{}
  }
  return String(e);
}

async function loadWatches():Promise<Watch[]>{
  const q=await db.from("brian_catalyst_sentinel_watches")
    .select("watch_id,event_id,asset_id,started_at,status,direction,reference_price,recheck_count")
    .in("status",["WATCHING","BUILDING","BREAKOUT_CANDIDATE","CONFIRMED"])
    .gt("expires_at",new Date().toISOString())
    .order("next_recheck_at",{ascending:true,nullsFirst:false})
    .limit(100);
  if(q.error)throw q.error;
  return (q.data??[]) as Watch[];
}
async function loadPendingAlerts(){
  const since=new Date(Date.now()-30*60*1000).toISOString();
  const q=await db.from("brian_catalyst_sentinel_alerts")
    .select("alert_id,event_id,asset_id,observed_at,alpha_dispatched")
    .eq("alpha_dispatched",false)
    .gt("observed_at",since)
    .order("observed_at",{ascending:true})
    .limit(20);
  if(q.error)throw q.error;
  return Array.isArray(q.data)?q.data as Json[]:[];
}

async function books(){
  const r=await fetch("https://api.binance.com/api/v3/ticker/bookTicker",{
    headers:{"user-agent":"Brian-Realtime-Catalyst/2.0"},
    signal:AbortSignal.timeout(3500)
  });
  if(!r.ok)throw new Error("BINANCE_BOOK_"+r.status);
  const p=await r.json();
  const map=new Map<string,{mid:number;spread:number}>();
  for(const value of Array.isArray(p)?p:[]){
    if(!value||typeof value!=="object")continue;
    const row=value as Json,bid=finite(row.bidPrice),ask=finite(row.askPrice),symbol=String(row.symbol??"");
    if(!(bid>0)||ask<bid||!symbol)continue;
    const mid=(bid+ask)/2;
    map.set("crypto:"+symbol,{mid,spread:10000*(ask-bid)/Math.max(mid,1e-12)});
  }
  return map;
}
async function imbalance(assetId:string){
  try{
    const symbol=assetId.replace(/^crypto:/,"");
    const r=await fetch("https://api.binance.com/api/v3/depth?symbol="+encodeURIComponent(symbol)+"&limit=20",{
      headers:{"user-agent":"Brian-Realtime-Catalyst/2.0"},
      signal:AbortSignal.timeout(1800)
    });
    if(!r.ok)return 0;
    const p=await r.json() as Json;
    const sum=(rows:unknown)=>{
      if(!Array.isArray(rows))return 0;
      return rows.reduce((t,v)=>Array.isArray(v)?t+finite(v[0])*finite(v[1]):t,0);
    };
    const bids=sum(p.bids),asks=sum(p.asks);
    return bids+asks?(bids-asks)/(bids+asks):0;
  }catch{return 0}
}
async function dispatchAlpha(internalKey:string,eventId:string,assetId:string,alertId:string){
  const r=await fetch(ALPHA_RECHECK_URL,{
    method:"POST",
    headers:{"content-type":"application/json","x-brian-internal-key":internalKey},
    body:JSON.stringify({event_id:eventId,asset_id:assetId,alert_id:alertId,request_id:alertId}),
    signal:AbortSignal.timeout(12000)
  });
  const body=await r.text();
  if(!r.ok)throw new Error("ALPHA_RECHECK_"+r.status+":"+body.slice(0,500));
  return r.status;
}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);
  let internalKey="";
  try{internalKey=await requireRealtimeInternal(req)}catch{return out({status:"UNAUTHORIZED"},401)}
  const started=Date.now();
  let stage="load_watches";

  try{
    const watches=await loadWatches();
    if(!watches.length)return out({status:"SUCCESS",version:VERSION,watches:0,updates:0,alerts:0,alpha_dispatches:0,elapsed_ms:Date.now()-started,shadow_only:true,live_execution:false});

    stage="books";
    const bookMap=await books();
    const baseline=new Map<string,{ret:number;dir:number}>();
    for(const w of watches){
      const b=bookMap.get(w.asset_id),ref=finite(w.reference_price);
      if(!b||ref<=0)continue;
      const ret=b.mid/ref-1;
      baseline.set(w.asset_id,{ret,dir:ret>.00005?1:ret<-.00005?-1:0});
    }

    stage="compute_reactions";
    const depthCandidates=watches.filter(w=>{
      const b=bookMap.get(w.asset_id),ref=finite(w.reference_price);
      return Boolean(b&&ref>0&&Math.abs(b.mid/ref-1)*10000>=20);
    }).slice(0,16);
    const imbalances=new Map<string,number>(await Promise.all(depthCandidates.map(async w=>[w.asset_id,await imbalance(w.asset_id)] as const)));

    const now=Date.now();
    const updates:Json[]=[];
    for(const w of watches){
      const b=bookMap.get(w.asset_id); if(!b)continue;
      let ref=finite(w.reference_price); if(ref<=0)ref=b.mid;
      const ret=b.mid/ref-1,direction=ret>.00005?1:ret<-.00005?-1:0,absBps=Math.abs(ret)*10000;
      const obi=imbalances.get(w.asset_id)??0;
      let comparable=0,aligned=0;
      for(const [assetId,v] of baseline.entries()){
        if(assetId===w.asset_id||Math.abs(v.ret)<.0005)continue;
        comparable++; if(v.dir===direction)aligned++;
      }
      const cross=comparable?aligned/comparable:.5;
      const score=clip(.55*clip(absBps/80)+.18*clip((direction*obi+.05)/.35)+.17*cross-.10*clip(b.spread/30));
      const state=absBps>=60&&score>=.58&&b.spread<=25?"CONFIRMED":absBps>=35&&score>=.46&&b.spread<=30?"BREAKOUT_CANDIDATE":absBps>=15?"BUILDING":"WATCHING";
      const elapsed=(now-Date.parse(w.started_at))/60000;
      let rc=Number(w.recheck_count||0),timed:string|null=null;
      if(rc<RECHECK_MINUTES.length&&elapsed>=RECHECK_MINUTES[rc]){timed="TIMED_RECHECK_"+RECHECK_MINUTES[rc]+"M";rc++}
      const flipped=Number(w.direction)!==0&&direction!==0&&Number(w.direction)!==direction&&absBps>=20;
      const changed=state!==w.status;
      const alertType=flipped?"DIRECTION_FLIP":changed&&state!=="WATCHING"?state:timed;
      updates.push({
        watch_id:w.watch_id,direction,status:state,reaction_score:score,price_return:ret,last_price:b.mid,
        reference_price:ref,spread_bps:b.spread,orderbook_imbalance:obi,cross_market_confirmation:cross,
        recheck_count:rc,next_recheck_at:rc<RECHECK_MINUTES.length?new Date(Date.parse(w.started_at)+RECHECK_MINUTES[rc]*60000).toISOString():null,
        alert_type:alertType
      });
    }

    stage="apply_reactions_rpc";
    const rpc=updates.length?await db.rpc("brian_catalyst_apply_reactions_v23",{p_updates:updates}):{data:[],error:null};
    if(rpc.error)throw rpc.error;
    const alerts=Array.isArray(rpc.data)?rpc.data as Json[]:[];
    stage="load_pending_alerts";
    const pending=await loadPendingAlerts();
    const dispatchMap=new Map<string,Json>();
    for(const row of [...alerts,...pending]){
      const id=String(row.alert_id??"");
      if(id&&!dispatchMap.has(id))dispatchMap.set(id,row);
    }
    const dispatchRows=[...dispatchMap.values()];

    stage="alpha_dispatch";
    let dispatches=0;
    for(const row of dispatchRows){
      const alertId=String(row.alert_id??""),eventId=String(row.event_id??""),assetId=String(row.asset_id??"");
      if(!alertId||!eventId||!assetId)continue;
      try{
        const status=await dispatchAlpha(internalKey,eventId,assetId,alertId);
        dispatches++;
        await db.from("brian_catalyst_sentinel_alerts").update({
          alpha_dispatched:true,alpha_dispatch_request_id:"HTTP_"+status,alpha_dispatched_at:new Date().toISOString()
        }).eq("alert_id",alertId);
      }catch(e){
        await db.from("brian_catalyst_sentinel_alerts").update({
          alpha_dispatched:false,alpha_dispatch_request_id:"ERROR:"+errText(e).slice(0,180)
        }).eq("alert_id",alertId);
      }
    }

    return out({status:"SUCCESS",version:VERSION,watches:watches.length,updates:updates.length,alerts:alerts.length,pending_retries:pending.length,dispatch_candidates:dispatchRows.length,alpha_dispatches:dispatches,elapsed_ms:Date.now()-started,shadow_only:true,live_execution:false});
  }catch(e){
    return out({status:"FAILED_CLOSED",version:VERSION,stage,error:errText(e).slice(0,1200),elapsed_ms:Date.now()-started,shadow_only:true,live_execution:false},500);
  }
});
