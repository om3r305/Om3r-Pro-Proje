import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { gzip, ungzip } from "npm:pako@2.1.0";
import { requireRealtimeInternal } from "../_shared/realtime_internal_auth.ts";
const db=createClient(Deno.env.get("SUPABASE_URL")!,Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,{auth:{persistSession:false}});
const BUCKET="brian-realtime-archive";
const TABLES:Record<string,string>={brian_sensor_observations:"observation_id",brian_micro_book_ticks:"tick_id"};
const out=(body:unknown,status=200)=>new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json"}});
async function sha(bytes:Uint8Array){return [...new Uint8Array(await crypto.subtle.digest("SHA-256",new Uint8Array(bytes)))].map(x=>x.toString(16).padStart(2,"0")).join("");}
async function checkedRows(a:any){
  const d=await db.storage.from(BUCKET).download(a.storage_path);if(d.error)throw d.error;
  const bytes=new Uint8Array(await d.data.arrayBuffer());
  if(await sha(bytes)!==a.sha256)throw new Error("ARCHIVE_CHECKSUM_MISMATCH");
  const rows=new TextDecoder().decode(ungzip(bytes)).trim().split("\n").filter(Boolean).map(x=>JSON.parse(x));
  if(rows.length!==a.row_count)throw new Error("ARCHIVE_ROW_COUNT_MISMATCH");
  return rows;
}
Deno.serve(async req=>{
  if(req.method!=="POST")return out({error:"POST required"},405);
  try{await requireRealtimeInternal(req);}catch{return out({status:"UNAUTHORIZED"},401);}
  const token=crypto.randomUUID();let acquired=false;
  try{
    const lock=await db.rpc("brian_realtime_acquire_lease",{p_job:"archive",p_token:token});
    if(lock.error)throw lock.error;
    if(!lock.data)return out({status:"SKIPPED_BUSY"});acquired=true;
    const body=await req.json().catch(()=>({}));
    if(body.action==="restore"){
      const q=await db.from("brian_realtime_archives").select("*").eq("archive_id",String(body.archive_id)).single();if(q.error)throw q.error;
      const pk=TABLES[q.data.table_name];if(!pk)throw new Error("TABLE_NOT_ALLOWED");
      const rows=await checkedRows(q.data);
      for(let i=0;i<rows.length;i+=200){const up=await db.from(q.data.table_name).upsert(rows.slice(i,i+200),{onConflict:pk,ignoreDuplicates:true});if(up.error)throw up.error;}
      const saved=await db.from("brian_realtime_archives").update({restored_at:new Date().toISOString()}).eq("archive_id",q.data.archive_id);if(saved.error)throw saved.error;
      return out({status:"RESTORED",rows:rows.length,archive_retained:true});
    }
    const bucket=await db.storage.getBucket(BUCKET);
    if(bucket.error){const c=await db.storage.createBucket(BUCKET,{public:false,fileSizeLimit:52428800});if(c.error&&!/exist/i.test(c.error.message))throw c.error;}
    const completed=[];
    for(const [table,pk] of Object.entries(TABLES)){
      const pending=await db.from("brian_realtime_archives").select("*").eq("table_name",table).is("compacted_at",null).order("verified_at").limit(1).maybeSingle();if(pending.error)throw pending.error;
      if(pending.data && Date.parse(pending.data.verified_at)<=Date.now()-86400000){
        const rows=await checkedRows(pending.data);
        const compact=await db.rpc("brian_realtime_compact_archive",{p_archive:pending.data.archive_id,p_rows:rows});if(compact.error)throw compact.error;
        completed.push({table,status:"ARCHIVED",rows:compact.data});
      }
      const latest=await db.from("brian_realtime_archives").select("last_observed_at,last_pk").eq("table_name",table).order("last_observed_at",{ascending:false}).order("last_pk",{ascending:false}).limit(1).maybeSingle();if(latest.error)throw latest.error;
      let query=db.from(table).select("*").lt("observed_at",new Date(Date.now()-7*86400000).toISOString()).order("observed_at").order(pk).limit(2000);
      if(latest.data)query=query.or(`observed_at.gt.${latest.data.last_observed_at},and(observed_at.eq.${latest.data.last_observed_at},${pk}.gt.${latest.data.last_pk})`);
      const q=await query;if(q.error)throw q.error;
      if(!q.data?.length){completed.push({table,status:"NO_OLD_DATA"});continue;}
      const bytes=gzip(new TextEncoder().encode(q.data.map(r=>JSON.stringify(r)).join("\n")+"\n"));
      const digest=await sha(bytes),path=table+"/"+digest+".jsonl.gz";
      // Idempotent content-addressed object; old archives remain private and restorable.
      const uploaded=await db.storage.from(BUCKET).upload(path,bytes,{contentType:"application/gzip",upsert:false});
      if(uploaded.error&&!/exist|duplicate/i.test(uploaded.error.message))throw uploaded.error;
      const archive={archive_id:digest,table_name:table,storage_path:path,sha256:digest,row_count:q.data.length,verified_at:new Date().toISOString(),last_observed_at:q.data.at(-1)!.observed_at,last_pk:String(q.data.at(-1)![pk])};
      await checkedRows(archive);
      const saved=await db.from("brian_realtime_archives").upsert(archive,{onConflict:"archive_id",ignoreDuplicates:true});if(saved.error)throw saved.error;
      completed.push({table,status:"VERIFIED",rows:q.data.length});
    }
    return out({status:"SUCCESS",results:completed,hot_days:7,grace_hours:24,restore_supported:true,shadow_only:true,live_execution:false});
  }catch(e){return out({status:"FAILED_CLOSED",error:e instanceof Error?e.message:String(e)},500);}
  finally{if(acquired)await db.from("brian_realtime_job_leases").update({expires_at:new Date().toISOString()}).eq("job","archive").eq("token",token);}
});
