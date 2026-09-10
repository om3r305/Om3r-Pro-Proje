import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import type { SupabaseClient } from "npm:@supabase/supabase-js@2.116.0";
import { hash, POLICY_VERSION, RELEASE_ID, same } from "../_shared/dip_v84_contract.ts";
import { withV84Lease } from "../brian-dip-v84-worker/auth.ts";
import { readForesight, resolvePending } from "./resolver.ts";

const db=createClient(Deno.env.get("SUPABASE_URL")!,Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,{auth:{persistSession:false,autoRefreshToken:false}});
const EXACT=new Set(["https://monster-coins-pro-seven.vercel.app","https://monster-coins-pro-oemer-yildirim.vercel.app","http://localhost:3000","http://127.0.0.1:3000"]);
const ORIGIN=/^https:\/\/monster-coins(?:-pro)?-[a-z0-9-]*oemer-yildirim\.vercel\.app$/i;
function headers(origin:string|null){return{"content-type":"application/json","cache-control":"no-store","access-control-allow-origin":origin&&(EXACT.has(origin)||ORIGIN.test(origin))?origin:"https://monster-coins-pro-seven.vercel.app","access-control-allow-headers":"content-type,x-brian-dashboard-key,x-brian-cron-key,authorization,apikey","access-control-allow-methods":"POST,OPTIONS","vary":"Origin"};}
async function requireCronAuth(db:SupabaseClient,req:Request){const supplied=(req.headers.get("x-brian-cron-key")||"").trim();if(!supplied)throw Error("UNAUTHORIZED_CRON");const q=await db.from("brian_dashboard_auth").select("cron_key_sha256").eq("auth_id","control-v3").single();if(q.error||!q.data)throw Error("CRON_AUTH_UNAVAILABLE");if(!same(await hash(supplied),String(q.data.cron_key_sha256||"")))throw Error("UNAUTHORIZED_CRON");}
async function authorize(req:Request):Promise<"reader"|"cron">{const key=(req.headers.get("x-brian-dashboard-key")||"").trim();if(!key){await requireCronAuth(db,req);return"cron";}const q=await db.from("brian_dashboard_auth").select("dashboard_key_sha256").order("created_at",{ascending:false}).limit(1).maybeSingle();if(q.error||!q.data||!same(await hash(key),String(q.data.dashboard_key_sha256||"")))throw Error("UNAUTHORIZED_DASHBOARD");return"reader";}

Deno.serve(async(req:Request)=>{
  const h=headers(req.headers.get("origin"));if(req.method==="OPTIONS")return new Response("ok",{headers:h});if(req.method!=="POST")return new Response("method",{status:405,headers:h});
  try{const mode=await authorize(req);if(mode==="cron"){const r=await withV84Lease(db,"brian-dip-v84-foresight",l=>resolvePending(db,l.assertOwned));return Response.json(r.contended?{status:"WAIT_LEASE",release_id:RELEASE_ID,policy_version:POLICY_VERSION}:r.value,{headers:h});}const body=await req.json().catch(()=>({}));return Response.json(await readForesight(db,typeof body.session_id==="string"?body.session_id:undefined),{headers:h});}
  catch(e){const message=e instanceof Error?e.message:String(e);return Response.json({status:"FAILED_CLOSED",error:message,release_id:RELEASE_ID,policy_version:POLICY_VERSION,shadow_only:true,live_execution:false},{status:message.includes("UNAUTHORIZED")?401:500,headers:h});}
});
