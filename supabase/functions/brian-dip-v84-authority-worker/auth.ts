import type { SupabaseClient } from "npm:@supabase/supabase-js@2.116.0";
import { hash, same } from "../_shared/dip_v84_authority_contract.ts";

export async function requireCronAuth(db:SupabaseClient,req:Request):Promise<void>{
  const supplied=(req.headers.get("x-brian-cron-key")||"").trim();if(!supplied)throw Error("UNAUTHORIZED_CRON");
  const q=await db.from("brian_dashboard_auth").select("cron_key_sha256").eq("auth_id","control-v3").single();
  if(q.error||!q.data)throw Error("CRON_AUTH_UNAVAILABLE");
  if(!same(await hash(supplied),String(q.data.cron_key_sha256||"")))throw Error("UNAUTHORIZED_CRON");
}
export type Lease={owner:string;generation:number;assertOwned:()=>void};
export async function withAuthorityLease<T>(db:SupabaseClient,work:(lease:Lease)=>Promise<T>):Promise<{contended:boolean;value?:T}>{
  const leaseKey="brian-dip-v843-profit-protect-worker",owner=crypto.randomUUID(),seconds=45;
  const a=await db.rpc("brian_dip_v84_acquire_lease",{p_lease_key:leaseKey,p_owner_id:owner,p_lease_seconds:seconds});
  if(a.error)throw Error("V843_LEASE_READ_FAILED:"+a.error.message);
  const row=Array.isArray(a.data)?a.data[0]:a.data;if(!row||row.acquired!==true)return{contended:true};
  const generation=Number(row.lease_generation);if(!Number.isSafeInteger(generation)||generation<=0)throw Error("V843_INVALID_LEASE_GENERATION");
  let lost=false,finished=false;
  const timer=setInterval(async()=>{try{const r=await db.rpc("brian_dip_v84_renew_lease",{p_lease_key:leaseKey,p_owner_id:owner,p_lease_generation:generation,p_lease_seconds:seconds});if(r.error||r.data!==true)lost=true;}catch{lost=true;}if(lost||finished)clearInterval(timer);},15000);
  const assertOwned=()=>{if(lost)throw Error("V843_LEASE_LOST");};
  try{return{contended:false,value:await work({owner,generation,assertOwned})};}
  finally{finished=true;clearInterval(timer);await db.rpc("brian_dip_v84_release_lease",{p_lease_key:leaseKey,p_owner_id:owner,p_lease_generation:generation});}
}