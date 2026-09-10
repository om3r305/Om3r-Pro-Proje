import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { POLICY_VERSION } from "../_shared/dip_v8_dual.ts";
import { requireCronAuth, withCollectorLease } from "./auth.ts";
import { runWorker } from "./worker.ts";

const db=createClient(Deno.env.get("SUPABASE_URL")!,Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,{auth:{persistSession:false,autoRefreshToken:false}});

async function workerDb(){
  const q=await db.from("brian_dip_session_events").select("*").order("requested_at",{ascending:false}).order("event_id",{ascending:false}).limit(1).maybeSingle();
  if(q.error||!q.data||q.data.event_kind!=="PAUSE")return db;
  const start=await db.from("brian_dip_session_events").select("*").eq("session_id",q.data.session_id).eq("event_kind","START").order("requested_at",{ascending:true}).limit(1).maybeSingle();
  if(start.error||!start.data)return db;
  const synthetic={...q.data,starting_equity:start.data.starting_equity,trade_notional:start.data.trade_notional,config:start.data.config,engine_token_sha256:start.data.engine_token_sha256};
  return new Proxy(db as any,{
    get(target,prop){
      if(prop==="from")return(table:string)=>{
        if(table!=="brian_dip_session_events")return target.from(table);
        const chain:any={select:()=>chain,order:()=>chain,limit:()=>chain,maybeSingle:async()=>({data:synthetic,error:null})};
        return chain;
      };
      const value=target[prop];return typeof value==="function"?value.bind(target):value;
    }
  }) as any;
}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return new Response("method",{status:405});
  try{
    await requireCronAuth(db,req);
    const lease=await withCollectorLease(db,"brian-dip-dual-shadow-worker-v82",async(owner,assertOwned)=>runWorker(await workerDb(),owner,assertOwned));
    return Response.json(lease.contended?{status:"WAIT_LEASE",worker_version:"dip-v8.3-entry-zone-20260910.1"}:lease.value,{headers:{"cache-control":"no-store"}});
  }catch(e){const message=e instanceof Error?e.message:String(e);console.error("dip-v83-dual",message);return Response.json({status:"FAILED_CLOSED",error:message,worker_version:"dip-v8.3-entry-zone-20260910.1",policy_version:POLICY_VERSION,shadow_only:true,live_execution:false},{status:message.includes("UNAUTHORIZED")?401:500,headers:{"cache-control":"no-store"}});}
});
