import { requireRealtimeInternal } from "../_shared/realtime_internal_auth.ts";

const VERSION = "brian.realtime-orchestrator.v7-heartbeat-offload";
const BASE = "https://dliediwlldojkfjzlznm.supabase.co/functions/v1";
const ENDPOINTS = {
  eye: BASE + "/brian-realtime-official-eye",
  universe: BASE + "/brian-universe-collector",
  sensor: BASE + "/brian-sensor-mesh",
  derivatives: BASE + "/brian-realtime-derivatives-eye",
  fx: BASE + "/brian-realtime-fx-eye",
  macro: BASE + "/brian-realtime-official-macro-eye",
  bigMove: BASE + "/brian-realtime-big-move-hunter",
  intrabar: BASE + "/brian-intrabar-eye",
  alpha: BASE + "/brian-alpha-decision-compiler",
  catalyst: BASE + "/brian-realtime-catalyst-reaction",
  crowd: BASE + "/brian-realtime-crowd-behavior",
  multiasset: BASE + "/brian-realtime-multiasset-market-eye",
  breakingScout: "https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-breaking-scout",
  heartbeatRefresh: "https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-frontier-heartbeat-refresh",
};

type Json = Record<string, unknown>;

function out(body:unknown,status=200){
  return new Response(JSON.stringify(body),{
    status,
    headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}
  });
}
function errText(e:unknown){return e instanceof Error?`${e.name}: ${e.message}`:String(e)}

async function call(name:string,url:string,key:string,timeoutMs:number){
  const started=Date.now();
  try{
    const r=await fetch(url,{
      method:"POST",
      headers:{"content-type":"application/json","x-brian-internal-key":key},
      body:"{}",
      signal:AbortSignal.timeout(timeoutMs)
    });
    const text=await r.text();
    let body:Json={};
    try{body=JSON.parse(text) as Json}catch{body={raw:text.slice(0,600)}}
    const targetStatus=String(body.status??"");
    const logicalOk=r.ok && !["FAILED","FAILED_CLOSED","DEGRADED","UNAUTHORIZED"].includes(targetStatus);
    return {
      name,ok:logicalOk,http_status:r.status,target_status:targetStatus,
      elapsed_ms:Date.now()-started,body
    };
  }catch(e){
    return {name,ok:false,http_status:0,target_status:"FETCH_FAILED",elapsed_ms:Date.now()-started,error:errText(e).slice(0,800)};
  }
}

async function marketLane(key:string,minute:number){
  const results:Json[]=[];

  if(minute%5===0){
    const universe=await call("universe",ENDPOINTS.universe,key,50000);
    results.push(universe);

    if(universe.ok){
      const sensor=await call("sensor",ENDPOINTS.sensor,key,50000);
      results.push(sensor);

      if(sensor.ok){
        if(minute%10===0){
          results.push(await call("derivatives",ENDPOINTS.derivatives,key,50000));
        }
        results.push(await call("crowd_behavior",ENDPOINTS.crowd,key,50000));
        results.push(await call("big_move",ENDPOINTS.bigMove,key,50000));
      }else{
        if(minute%10===0) results.push({name:"derivatives",ok:false,target_status:"SKIPPED_SENSOR_FAILED"});
        results.push({name:"big_move",ok:false,target_status:"SKIPPED_SENSOR_FAILED"});
      }
    }else{
      results.push({name:"sensor",ok:false,target_status:"SKIPPED_UNIVERSE_FAILED"});
      if(minute%10===0) results.push({name:"derivatives",ok:false,target_status:"SKIPPED_UNIVERSE_FAILED"});
      results.push({name:"big_move",ok:false,target_status:"SKIPPED_UNIVERSE_FAILED"});
    }
  }

  if(minute%10===0){
    results.push(await call("multiasset_market",ENDPOINTS.multiasset,key,50000));
  }

  if(minute%60===7){
    results.push(await call("fx",ENDPOINTS.fx,key,50000));
  }

  if(minute%20===9){
    results.push(await call("macro",ENDPOINTS.macro,key,50000));
  }

  results.push(await call("intrabar",ENDPOINTS.intrabar,key,50000));

  if(minute%2===0){
    results.push(await call("alpha",ENDPOINTS.alpha,key,50000));
  }
  return results;
}

Deno.serve(async(req:Request)=>{
  if(req.method==="GET")return out({
    status:"OK",version:VERSION,mode:"SUPABASE_INTERNAL",
    cloudflare_required:false,shadow_only:true,live_execution:false
  });
  if(req.method!=="POST")return out({error:"POST required"},405);

  let key="";
  try{key=await requireRealtimeInternal(req)}catch{return out({status:"UNAUTHORIZED"},401)}

  const started=Date.now();
  const minute=Math.floor(Date.now()/60000);

  const eyePromise=call("official_eye",ENDPOINTS.eye,key,50000);
  const marketPromise=marketLane(key,minute);
  const scoutPromise=minute%2===0
    ? call("breaking_scout",ENDPOINTS.breakingScout,key,30000)
    : Promise.resolve(null);
  const heartbeatPromise=call("frontier_heartbeat_refresh",ENDPOINTS.heartbeatRefresh,key,9000);

  const [eye,market,scout,heartbeat]=await Promise.all([eyePromise,marketPromise,scoutPromise,heartbeatPromise]);
  const catalyst=await call("catalyst_reaction",ENDPOINTS.catalyst,key,50000);
  const results=[eye,...market,...(scout?[scout]:[]),heartbeat,catalyst];

  const failed=results.filter((r:any)=>r.ok===false);
  return out({
    status:failed.length?"DEGRADED":"SUCCESS",
    version:VERSION,
    minute,
    results,
    elapsed_ms:Date.now()-started,
    cloudflare_required:false,
    shadow_only:true,
    live_execution:false
  },failed.length?207:200);
});
