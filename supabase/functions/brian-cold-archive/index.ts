import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { gzip } from "npm:pako@2.1.0";
import { requireCronAuth } from "../_shared/cron_auth.ts";

const URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const BUCKET="brian-cold-archive";
const GRACE_HOURS=24;
const MAX_ARCHIVES_PER_RUN=3;
const MAX_PURGES_PER_RUN=3;
const ALLOWED=new Set([
  "brian_treasury_shadow_snapshots",
  "brian_treasury_shadow_actions",
  "brian_collector_runs",
  "brian_live_shadow_ticks",
  "brian_micro_book_ticks",
  "brian_alpha_decisions",
  "brian_sensor_observations",
  "brian_intel_events",
]);

function out(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}})}
function errText(e:unknown){if(e instanceof Error)return `${e.name}: ${e.message}`;try{return JSON.stringify(e)}catch{return String(e)}}
function bytes(v:string){return new TextEncoder().encode(v)}
async function sha(v:Uint8Array|string){const b=typeof v==="string"?bytes(v):v;const d=new Uint8Array(await crypto.subtle.digest("SHA-256",b));return [...d].map(x=>x.toString(16).padStart(2,"0")).join("")}
function iso(v:unknown){const ms=Date.parse(String(v??""));return Number.isFinite(ms)?new Date(ms).toISOString():null}
function chunks<T>(a:T[],n:number){const out:T[][]=[];for(let i=0;i<a.length;i+=n)out.push(a.slice(i,i+n));return out}

async function ensureBucket(){
  const listed=await db.storage.listBuckets();
  if(listed.error)throw listed.error;
  if(!(listed.data??[]).some(b=>b.name===BUCKET)){
    const created=await db.storage.createBucket(BUCKET,{public:false,fileSizeLimit:52428800,allowedMimeTypes:["application/gzip"]});
    if(created.error && !String(created.error.message??"").toLowerCase().includes("exist"))throw created.error;
  }
}

async function healthyEnough(){
  const q=await db.rpc("brian_archive_runtime_health");
  if(q.error)throw q.error;
  const h=q.data&&typeof q.data==="object"?q.data as Record<string,unknown>:{};
  const queueDepth=Number(h.queue_depth??999);
  const alphaAge=Number(h.alpha_age_seconds??1e12);
  const treasuryAge=Number(h.treasury_age_seconds??1e12);
  const worldAge=Number(h.world_age_seconds??1e12);
  return {ok:queueDepth<=3&&alphaAge<1800&&treasuryAge<1800&&worldAge<1800,queueDepth,alphaAge,treasuryAge,worldAge};
}

async function latestManifest(tableName:string){
  const q=await db.from("brian_archive_manifests")
    .select("archive_id,last_observed_at,metadata,created_at")
    .eq("table_name",tableName)
    .order("created_at",{ascending:false}).limit(1).maybeSingle();
  if(q.error)throw q.error;
  return q.data;
}

async function fetchArchiveRows(policy:any){
  if(!ALLOWED.has(String(policy.table_name)))throw new Error(`archive table not allowed: ${policy.table_name}`);
  const table=String(policy.table_name),timeCol=String(policy.time_column),pk=String(policy.pk_column);
  const batch=Math.max(50,Math.min(2000,Number(policy.batch_size||500)));
  const cutoff=new Date(Date.now()-Number(policy.hot_retention_days)*86400_000).toISOString();
  const last=await latestManifest(table);
  const lastTime=iso(last?.last_observed_at);
  const lastPk=last?.metadata?.last_pk==null?null:String(last.metadata.last_pk);

  if(lastTime&&lastPk){
    const same=await db.from(table).select("*").eq(timeCol,lastTime).gt(pk,lastPk).lt(timeCol,cutoff).order(pk,{ascending:true}).limit(batch);
    if(same.error)throw same.error;
    if((same.data??[]).length)return {rows:same.data??[],cutoff};
  }
  let q=db.from(table).select("*").lt(timeCol,cutoff).order(timeCol,{ascending:true}).order(pk,{ascending:true}).limit(batch);
  if(lastTime)q=q.gt(timeCol,lastTime);
  const res=await q;
  if(res.error)throw res.error;
  return {rows:res.data??[],cutoff};
}

async function archivePolicy(policy:any){
  const table=String(policy.table_name),timeCol=String(policy.time_column),pk=String(policy.pk_column);
  const fetched=await fetchArchiveRows(policy);
  const rows=fetched.rows as Record<string,unknown>[];
  if(!rows.length)return {table,status:"NO_CANDIDATES",cutoff:fetched.cutoff};

  const jsonl=rows.map(r=>JSON.stringify(r)).join("\n")+"\n";
  const compressed=gzip(bytes(jsonl),{level:6});
  const digest=await sha(compressed);
  const first=iso(rows[0]?.[timeCol]);
  const last=iso(rows.at(-1)?.[timeCol]);
  const lastPk=String(rows.at(-1)?.[pk]??"");
  const archiveId=await sha(`${table}|${first}|${last}|${lastPk}|${digest}`);
  const path=`${table}/${(first??new Date().toISOString()).slice(0,10)}/${archiveId}.jsonl.gz`;

  const up=await db.storage.from(BUCKET).upload(path,compressed,{contentType:"application/gzip",upsert:false,cacheControl:"31536000"});
  if(up.error&&!/exist|duplicate/i.test(String(up.error.message??"")))throw up.error;

  const dl=await db.storage.from(BUCKET).download(path);
  if(dl.error)throw dl.error;
  const verifyBytes=new Uint8Array(await dl.data.arrayBuffer());
  const verify=await sha(verifyBytes);
  if(verify!==digest)throw new Error(`archive checksum mismatch for ${table}`);

  const pkValues=rows.map(r=>r[pk]).filter(v=>v!==null&&v!==undefined);
  const manifest=await db.from("brian_archive_manifests").upsert({
    archive_id:archiveId,table_name:table,storage_bucket:BUCKET,storage_path:path,row_count:rows.length,
    first_observed_at:first,last_observed_at:last,content_sha256:digest,compressed_bytes:verifyBytes.byteLength,
    state:"UPLOADED_VERIFIED",verified_at:new Date().toISOString(),pk_values:pkValues,
    metadata:{time_column:timeCol,pk_column:pk,last_pk:lastPk,hot_retention_days:Number(policy.hot_retention_days),format:"jsonl+gzip",restore_mode:"upsert_by_primary_key",delete_requires_verified_archive_and_24h_grace:true},
    updated_at:new Date().toISOString()
  },{onConflict:"archive_id"});
  if(manifest.error)throw manifest.error;
  return {table,status:"UPLOADED_VERIFIED",archive_id:archiveId,rows:rows.length,first,last,bytes:verifyBytes.byteLength};
}

async function purgeVerified(){
  const before=new Date(Date.now()-GRACE_HOURS*3600_000).toISOString();
  const q=await db.from("brian_archive_manifests")
    .select("archive_id,table_name,storage_path,content_sha256,pk_values,verified_at,state,metadata")
    .eq("state","UPLOADED_VERIFIED").lt("verified_at",before)
    .order("verified_at",{ascending:true}).limit(MAX_PURGES_PER_RUN);
  if(q.error)throw q.error;
  const results:any[]=[];
  for(const m of q.data??[]){
    const p=await db.from("brian_archive_policies").select("pk_column,purge_enabled").eq("table_name",m.table_name).single();
    if(p.error)throw p.error;
    if(!p.data.purge_enabled){results.push({archive_id:m.archive_id,table:m.table_name,status:"RETAIN_HOT_POLICY"});continue}

    const dl=await db.storage.from(BUCKET).download(String(m.storage_path));
    if(dl.error)throw dl.error;
    const compressed=new Uint8Array(await dl.data.arrayBuffer());
    if(await sha(compressed)!==String(m.content_sha256))throw new Error(`purge checksum mismatch: ${m.archive_id}`);

    const ids=Array.isArray(m.pk_values)?m.pk_values:[];
    let deleted=0;
    for(const group of chunks(ids,200)){
      const del=await db.from(String(m.table_name)).delete().in(String(p.data.pk_column),group).select(String(p.data.pk_column));
      if(del.error)throw del.error;
      deleted+=(del.data??[]).length;
    }
    const upd=await db.from("brian_archive_manifests").update({state:"PURGED",purged_at:new Date().toISOString(),updated_at:new Date().toISOString(),metadata:{...(m.metadata??{}),purge_verified_again:true,deleted_rows:deleted}}).eq("archive_id",m.archive_id);
    if(upd.error)throw upd.error;
    results.push({archive_id:m.archive_id,table:m.table_name,status:"PURGED",deleted});
  }
  return results;
}

async function restoreArchive(archiveId:string){
  const m=await db.from("brian_archive_manifests").select("*").eq("archive_id",archiveId).single();
  if(m.error)throw m.error;
  const table=String(m.data.table_name);
  if(!ALLOWED.has(table))throw new Error("restore table not allowed");
  const p=await db.from("brian_archive_policies").select("pk_column").eq("table_name",table).single();
  if(p.error)throw p.error;
  const dl=await db.storage.from(BUCKET).download(String(m.data.storage_path));
  if(dl.error)throw dl.error;
  const compressed=new Uint8Array(await dl.data.arrayBuffer());
  if(await sha(compressed)!==String(m.data.content_sha256))throw new Error("restore checksum mismatch");
  const { ungzip }=await import("npm:pako@2.1.0");
  const raw=new TextDecoder().decode(ungzip(compressed));
  const rows=raw.split("\n").filter(Boolean).map(line=>JSON.parse(line));
  let restored=0;
  for(const group of chunks(rows,150)){
    const up=await db.from(table).upsert(group,{onConflict:String(p.data.pk_column),ignoreDuplicates:true});
    if(up.error)throw up.error;
    restored+=group.length;
  }
  const upd=await db.from("brian_archive_manifests").update({last_restored_at:new Date().toISOString(),updated_at:new Date().toISOString(),metadata:{...(m.data.metadata??{}),last_restore_rows:restored,restore_mode:"upsert_by_primary_key"}}).eq("archive_id",archiveId);
  if(upd.error)throw upd.error;
  return {status:"RESTORED",archive_id:archiveId,table,rows:restored,archive_retained:true};
}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);
  try{await requireCronAuth(req,db)}catch(e){return out({status:"UNAUTHORIZED",error:errText(e),shadow_only:true,live_execution:false},401)}
  try{
    await ensureBucket();
    const body=await req.json().catch(()=>({}));
    const action=String(body?.action??"archive");
    if(action==="restore"){
      if(!body?.archive_id)return out({error:"archive_id required"},400);
      return out(await restoreArchive(String(body.archive_id)));
    }
    const health=await healthyEnough();
    if(!health.ok)return out({status:"SKIPPED_BUSY",health,archive_loss:false,shadow_only:true,live_execution:false});

    const purged=await purgeVerified();
    const policies=await db.from("brian_archive_policies").select("*").eq("archive_enabled",true).order("priority",{ascending:true});
    if(policies.error)throw policies.error;
    const archived:any[]=[];
    for(const policy of policies.data??[]){
      if(archived.length>=MAX_ARCHIVES_PER_RUN)break;
      const result=await archivePolicy(policy);
      if(result.status!=="NO_CANDIDATES")archived.push(result);
    }
    return out({status:"SUCCESS",archived,purged,bucket:BUCKET,grace_hours:GRACE_HOURS,backup_first:true,delete_only_after_verified_archive:true,restore_supported:true,shadow_only:true,live_execution:false});
  }catch(e){
    return out({status:"FAILED_CLOSED",error:errText(e),backup_first:true,delete_only_after_verified_archive:true,shadow_only:true,live_execution:false},500);
  }
});
