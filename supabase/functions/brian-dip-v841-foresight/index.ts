import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import type { SupabaseClient } from "npm:@supabase/supabase-js@2.116.0";
import { hash, same } from "../_shared/dip_v84_contract.ts";
import { POLICY_VERSION, RELEASE_ID } from "../_shared/dip_v84_authority_contract.ts";
import { readForesight, resolvePending } from "./resolver.ts";

const db=createClient(Deno.env.get("SUPABASE_URL")!,Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,{auth:{persistSession:false,autoRefreshToken:false}});
const LEASE_KEY="brian-dip-v84-foresight";
const EXACT=new Set(["https://monster-coins-pro-seven.vercel.app","https://monster-coins-pro-oemer-yildirim.vercel.app","http://localhost:3000","http://127.0.0.1:3000"]);
const ORIGIN=/^https:\/\/monster-coins(?:-pro)?-[a-z0-9-]*oemer-yildirim\.vercel\.app$/i;
function headers(origin:string|null){return{"content-type":"application/json","cache-control":"no-store","access-control-allow-origin":origin&&(EXACT.has(origin)||ORIGIN.test(origin))?origin:"https://monster-coins-pro-seven.vercel.app","access-control-allow-headers":"content-type,x-brian-dashboard-key,x-brian-cron-key,authorization,apikey","access-control-allow-methods":"POST,OPTIONS","vary":"Origin"};}
async function requireCronAuth(db:SupabaseClient,req:Request){const supplied=(req.headers.get("x-brian-cron-key")||"").trim();if(!supplied)throw Error("UNAUTHORIZED_CRON");const q=await db.from("brian_dashboard_auth").select("cron_key_sha256").eq("auth_id","control-v3").single();if(q.error||!q.data)throw Error("CRON_AUTH_UNAVAILABLE");if(!same(await hash(supplied),String(q.data.cron_key_sha256||"")))throw Error("UNAUTHORIZED_CRON");}
async function authorize(req:Request):Promise<"reader"|"cron">{const key=(req.headers.get("x-brian-dashboard-key")||"").trim();if(!key){await requireCronAuth(db,req);return"cron";}const q=await db.from("brian_dashboard_auth").select("dashboard_key_sha256").eq("auth_id","control-v3").single();if(q.error||!q.data||!same(await hash(key),String(q.data.dashboard_key_sha256||"")))throw Error("UNAUTHORIZED_DASHBOARD");return"reader";}

async function withForesightLease<T>(work:(assertOwned:()=>void)=>Promise<T>):Promise<{contended:boolean;value?:T}>{
  const owner=crypto.randomUUID(),seconds=55;
  const a=await db.rpc("brian_dip_v84_acquire_lease",{p_lease_key:LEASE_KEY,p_owner_id:owner,p_lease_seconds:seconds});
  if(a.error)throw Error("V841_FORECAST_LEASE_READ_FAILED:"+a.error.message);
  const row=Array.isArray(a.data)?a.data[0]:a.data;if(!row||row.acquired!==true)return{contended:true};
  const generation=Number(row.lease_generation);if(!Number.isSafeInteger(generation)||generation<=0)throw Error("V841_FORECAST_INVALID_LEASE_GENERATION");
  let lost=false,finished=false;
  const timer=setInterval(async()=>{try{const r=await db.rpc("brian_dip_v84_renew_lease",{p_lease_key:LEASE_KEY,p_owner_id:owner,p_lease_generation:generation,p_lease_seconds:seconds});if(r.error||r.data!==true)lost=true;}catch{lost=true;}if(lost||finished)clearInterval(timer);},18000);
  const assertOwned=()=>{if(lost)throw Error("V841_FORECAST_LEASE_LOST");};
  try{return{contended:false,value:await work(assertOwned)};}finally{finished=true;clearInterval(timer);await db.rpc("brian_dip_v84_release_lease",{p_lease_key:LEASE_KEY,p_owner_id:owner,p_lease_generation:generation});}
}

Deno.serve(async(req:Request)=>{
  const h=headers(req.headers.get("origin"));if(req.method==="OPTIONS")return new Response("ok",{headers:h});if(req.method!=="POST")return new Response("method",{status:405,headers:h});
  try{
    const mode=await authorize(req);
    if(mode==="cron"){
      const r=await withForesightLease(assertOwned=>resolvePending(db,assertOwned));
      return Response.json(r.contended?{status:"WAIT_LEASE",release_id:RELEASE_ID,policy_version:POLICY_VERSION,decision_authority:"BRIAN",shadow_only:true,live_execution:false}:r.value,{headers:h});
    }
    const body=await req.json().catch(()=>({}));
    return Response.json(await readForesight(db,typeof body.session_id==="string"?body.session_id:undefined),{headers:h});
  }catch(e){const message=e instanceof Error?e.message:String(e);return Response.json({status:"FAILED_CLOSED",error:message,release_id:RELEASE_ID,policy_version:POLICY_VERSION,decision_authority:"BRIAN",shadow_only:true,live_execution:false,browser_execution:false},{status:message.includes("UNAUTHORIZED")?401:500,headers:h});}
});
