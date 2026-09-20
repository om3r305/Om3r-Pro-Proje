import { requireRealtimeInternal } from "../_shared/realtime_internal_auth.ts";

const VERSION="brian.realtime-core-scheduler.v2-balanced";
const CORE_BRIDGE="https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-core-scheduler-bridge";

type Result={action:string;ok:boolean;http_status:number;target_status:string;elapsed_ms:number;body?:unknown;error?:string};

function out(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}})}
function err(e:unknown){return e instanceof Error?`${e.name}: ${e.message}`:String(e)}

async function runAction(action:string,key:string,timeoutMs=12000):Promise<Result>{
  const started=Date.now();
  try{
    const r=await fetch(CORE_BRIDGE,{
      method:"POST",
      headers:{"content-type":"application/json","x-brian-internal-key":key},
      body:JSON.stringify({action}),
      signal:AbortSignal.timeout(timeoutMs)
    });
    const text=await r.text();
    let body:any={};
    try{body=JSON.parse(text)}catch{body={raw:text.slice(0,1000)}}
    const target=String(body?.status??"");
    const ok=r.ok&&!["FAILED","FAILED_CLOSED","UNAUTHORIZED","INVALID_ACTION"].includes(target);
    return {action,ok,http_status:r.status,target_status:target,elapsed_ms:Date.now()-started,body};
  }catch(e){
    return {action,ok:false,http_status:0,target_status:"FETCH_FAILED",elapsed_ms:Date.now()-started,error:err(e).slice(0,1000)};
  }
}

function includes(minute:number,values:number[]){return values.includes(minute)}
function planned(minute:number){
  const actions:string[]=["dip"];

  if(minute%3===1) actions.push("alpha_sync");
  if(includes(minute,[2,7,12,17,22,27,32,37,42,47,52,57])) actions.push("treasury");
  if(includes(minute,[5,11,17,23,29,35,41,47,53,59])) actions.push("world");
  if(includes(minute,[3,13,23,33,43,53])) actions.push("multiasset");
  if(includes(minute,[6,16,26,36,46,56])) actions.push("official_primary");
  if(includes(minute,[8,18,28,38,48,58])) actions.push("source_observer");
  if(includes(minute,[9,24,39,54])) actions.push("discovery");
  if(includes(minute,[15,45])) actions.push("source_registry");
  if(includes(minute,[14,44])) actions.push("meeting_sync");
  if(includes(minute,[0,10,20,30,40,50])) actions.push("recovery");
  if(includes(minute,[5,20,35,50])) actions.push("watchdog");

  return [...new Set(actions)];
}

Deno.serve(async(req:Request)=>{
  if(req.method==="GET")return out({status:"OK",version:VERSION,role:"REALTIME_SCHEDULER_FOR_CORE",shadow_only:true,live_execution:false});
  if(req.method!=="POST")return out({error:"POST required"},405);

  let key="";
  try{key=await requireRealtimeInternal(req)}catch{return out({status:"UNAUTHORIZED"},401)}

  const now=new Date();
  const minute=now.getUTCMinutes();
  const actions=planned(minute);
  const results:Result[]=[];

  for(const action of actions){
    results.push(await runAction(action,key,action==="recovery"||action==="watchdog"?15000:12000));
    if(results[results.length-1].ok===false && action==="dip"){
      // A Core transport failure should not create a retry storm in the same minute.
      break;
    }
  }

  const failed=results.filter(r=>!r.ok);
  return out({
    status:failed.length?"DEGRADED":"SUCCESS",
    version:VERSION,
    observed_at:now.toISOString(),
    utc_minute:minute,
    actions,
    results,
    circuit_breaker:failed.length?{active:true,failed_actions:failed.map(x=>x.action)}:{active:false},
    scheduler_project:"brian-realtime",
    target_project:"brian-market-intelligence",
    shadow_only:true,
    live_execution:false
  },failed.length?207:200);
});