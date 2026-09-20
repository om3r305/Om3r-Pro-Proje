import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const VERSION="brian.frontier-heartbeat-refresh.v1";
const REALTIME_INTERNAL_KEY_SHA256="b0549b2b41a5b832b37455389583e1d166d210490a8c6fe43cda2748aca7c38a";

function out(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}})}
async function sha(v:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(v)));return [...d].map(b=>b.toString(16).padStart(2,"0")).join("")}
function ct(a:string,b:string){if(a.length!==b.length)return false;let x=0;for(let i=0;i<a.length;i++)x|=a.charCodeAt(i)^b.charCodeAt(i);return x===0}
async function auth(req:Request){
  const supplied=(req.headers.get("x-brian-internal-key")??"").trim();
  if(!supplied||!ct(await sha(supplied),REALTIME_INTERNAL_KEY_SHA256))throw new Error("UNAUTHORIZED_INTERNAL");
}
function errorText(e:unknown){return e instanceof Error?e.message:String(e)}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);
  try{await auth(req)}catch{return out({status:"UNAUTHORIZED"},401)}
  try{
    const q:any=db.rpc("brian_refresh_frontier_heartbeat_cache");
    const r=typeof q.abortSignal==="function"
      ? await q.abortSignal(AbortSignal.timeout(6500))
      : await Promise.race([
          Promise.resolve(q),
          new Promise((_,reject)=>setTimeout(()=>reject(new Error("HEARTBEAT_REFRESH_TIMEOUT")),6500))
        ]);
    if(r?.error)throw new Error(String(r.error.message??r.error));
    return out({status:"SUCCESS",version:VERSION,payload:r?.data??null,shadow_only:true,live_execution:false});
  }catch(e){
    return out({status:"DEGRADED",version:VERSION,error:errorText(e),shadow_only:true,live_execution:false},207);
  }
});