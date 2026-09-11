import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { POLICY_VERSION, RELEASE_ID, STRATEGY_MANIFEST_HASH } from "../_shared/dip_v84_authority_contract.ts";
import { requireCronAuth, withAuthorityLease } from "./auth.ts";
import { runWorker } from "./worker.ts";

const db=createClient(Deno.env.get("SUPABASE_URL")!,Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,{auth:{persistSession:false,autoRefreshToken:false}});
Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return new Response("method",{status:405});
  try{
    await requireCronAuth(db,req);
    const lease=await withAuthorityLease(db,l=>runWorker(db,l));
    return Response.json(lease.contended?{status:"WAIT_LEASE",release_id:RELEASE_ID,long_only:true,shadow_only:true,live_execution:false,browser_execution:false}:lease.value,{headers:{"cache-control":"no-store"}});
  }catch(e){
    const message=e instanceof Error?e.message:String(e);console.error("dip-v842-long-stateful",message);
    return Response.json({status:"FAILED_CLOSED",error:message,release_id:RELEASE_ID,strategy_manifest_hash:STRATEGY_MANIFEST_HASH,policy_version:POLICY_VERSION,decision_authority:"BRIAN",long_only:true,shadow_only:true,live_execution:false,browser_execution:false},{status:message.includes("UNAUTHORIZED")?401:500,headers:{"cache-control":"no-store"}});
  }
});