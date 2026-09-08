import type { SupabaseClient } from "npm:@supabase/supabase-js@2.116.0";
import { hash, same } from "../_shared/dip_v8_dual.ts";
export async function requireCronAuth(db:SupabaseClient,req:Request):Promise<void>{
  const supplied=(req.headers.get("x-brian-cron-key")||"").trim();if(!supplied)throw Error("UNAUTHORIZED_CRON");
  const q=await db.from("brian_dashboard_auth").select("cron_key_sha256").eq("auth_id","control-v3").single();
  if(q.error||!q.data)throw Error("CRON_AUTH_UNAVAILABLE");if(!same(await hash(supplied),String(q.data.cron_key_sha256||"")))throw Error("UNAUTHORIZED_CRON");
}
export async function withCollectorLease<T>(db:SupabaseClient,collector:string,work:(owner:string,assertOwned:()=>void)=>Promise<T>):Promise<{contended:boolean;value?:T}>{
  const owner=crypto.randomUUID(),seconds=55,a=await db.rpc("brian_acquire_collector_lease",{p_collector_id:collector,p_owner_token:owner,p_lease_seconds:seconds});
  if(a.error)throw Error("LEASE_READ_FAILED:"+a.error.message);if(a.data!==true)return{contended:true};
  let lost=false,finished=false;const heartbeat=setInterval(async()=>{try{const r=await db.rpc("brian_renew_collector_lease",{p_collector_id:collector,p_owner_token:owner,p_lease_seconds:seconds});if(r.error||r.data!==true)lost=true;}catch{lost=true;}if(lost||finished)clearInterval(heartbeat);},18000);
  try{return{contended:false,value:await work(owner,()=>{if(lost)throw Error("V8_DUAL_LEASE_LOST");})};}
  finally{finished=true;clearInterval(heartbeat);await db.rpc("brian_release_collector_lease",{p_collector_id:collector,p_owner_token:owner});}
}
