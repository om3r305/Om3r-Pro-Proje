import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import { withCollectorLease } from "../_shared/collector_lease.ts";

const SUPABASE_URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(SUPABASE_URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const COLLECTOR_ID="brian-source-registry-v2";
const EVIDENCE="PROSPECTIVE_EVOLUTION_SHADOW";
const LEASE_SECONDS=180;
const BATCH=16;
const CONCURRENCY=4;
const RUNTIME="REGISTRY_V4";

function out(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}})}
function errorText(error:unknown){if(error instanceof Error)return `${error.name}: ${error.message}`;if(error&&typeof error==="object"){const row=error as Record<string,unknown>;const fields=["code","message","details","hint","status","statusText"].filter(k=>row[k]!=null).map(k=>`${k}=${String(row[k])}`);if(fields.length)return fields.join(" | ");try{return JSON.stringify(error)}catch{}}return String(error)}
function clamp(v:number){return Math.max(0,Math.min(1,v))}
async function sha(v:string|Uint8Array){const b=typeof v==="string"?new TextEncoder().encode(v):v;const d=new Uint8Array(await crypto.subtle.digest("SHA-256",b));return[...d].map(x=>x.toString(16).padStart(2,"0")).join("")}
function hostMatches(host:string,domain:string){const h=host.toLowerCase().replace(/^www\./,"");const d=domain.toLowerCase().replace(/^www\./,"");return h===d||h.endsWith(`.${d}`)}
function authorityPrior(tier:string){if(tier==="T1_OFFICIAL_PRIMARY")return .98;if(tier==="T0_RAW_TELEMETRY")return .95;if(tier==="T2_INSTITUTIONAL")return .90;if(tier==="T3_TOP_TIER_NEWS")return .82;if(tier==="T4_SPECIALIST")return .67;return .42}
function originalityPrior(tier:string){if(tier==="T0_RAW_TELEMETRY"||tier==="T1_OFFICIAL_PRIMARY")return 1;if(tier==="T2_INSTITUTIONAL")return .9;if(tier==="T3_TOP_TIER_NEWS")return .7;if(tier==="T4_SPECIALIST")return .6;return .35}
function leadPrior(tier:string){if(tier==="T0_RAW_TELEMETRY")return .95;if(tier==="T1_OFFICIAL_PRIMARY")return .88;if(tier==="T2_INSTITUTIONAL")return .62;if(tier==="T3_TOP_TIER_NEWS")return .55;if(tier==="T4_SPECIALIST")return .58;return .45}

async function readPrefix(r:Response,max=512_000){if(!r.body)return new Uint8Array();const reader=r.body.getReader();const chunks:Uint8Array[]=[];let total=0;try{while(total<max){const {value,done}=await reader.read();if(done)break;if(!value)continue;const take=Math.min(value.length,max-total);chunks.push(value.subarray(0,take));total+=take;if(take<value.length)break}}finally{try{await reader.cancel()}catch{}}const bytes=new Uint8Array(total);let off=0;for(const c of chunks){bytes.set(c,off);off+=c.length}return bytes}
function parseable(kind:string,contentType:string,text:string){const ct=contentType.toLowerCase();const t=text.trim().slice(0,2000).toLowerCase();if(kind==="RSS"||kind==="ATOM"||kind==="STATUSPAGE_ATOM")return /<(rss|feed)(\s|>)/i.test(text)||ct.includes("xml")||ct.includes("rss")||ct.includes("atom");if(kind==="JSON_API"){try{JSON.parse(text);return true}catch{return ct.includes("json")&&text.trim().length>2}}if(kind==="HTML")return ct.includes("html")||t.includes("<html")||t.includes("<!doctype");if(kind==="DATASET")return text.length>0;return false}

async function probe(endpoint:any){
  const started=Date.now();const observedAt=new Date().toISOString();let status:number|null=null,ctype="",bytes=0,hash:string|null=null,reachable=false,pars=false,origin=false,change=false,errorClass:string|null=null,errorMessage:string|null=null,resolvedUrl=endpoint.endpoint_url;
  try{
    const timeoutMs=endpoint.endpoint_id==="sec_edgar_current"?12_000:9_000;
    const response=await fetch(endpoint.endpoint_url,{redirect:"follow",headers:{accept:endpoint.expected_content==="json"?"application/json,*/*;q=0.1":"application/rss+xml,application/atom+xml,application/xml,text/html,application/json;q=0.8,*/*;q=0.1","user-agent":"Brian-Market-OS/2.1 source-registry contact=owner"},signal:AbortSignal.timeout(timeoutMs)});
    status=response.status;ctype=response.headers.get("content-type")||"";resolvedUrl=response.url||endpoint.endpoint_url;let resolvedHost="";try{resolvedHost=new globalThis.URL(resolvedUrl).hostname}catch{}
    origin=hostMatches(resolvedHost,endpoint.canonical_domain);
    const raw=await readPrefix(response);bytes=raw.length;hash=raw.length?await sha(raw):null;const text=new TextDecoder("utf-8",{fatal:false}).decode(raw);pars=response.ok&&parseable(endpoint.endpoint_kind,ctype,text);reachable=response.ok&&origin&&bytes>0;
    const prev=await db.from("brian_source_endpoint_health_v2").select("content_hash").eq("endpoint_id",endpoint.endpoint_id).not("content_hash","is",null).order("observed_at",{ascending:false}).limit(1).maybeSingle();
    change=Boolean(hash&&prev.data?.content_hash&&prev.data.content_hash!==hash);
    if(!response.ok){errorClass="HTTP";errorMessage=`HTTP ${response.status}`}else if(!origin){errorClass="ORIGIN_MISMATCH";errorMessage=`redirected to ${resolvedHost||'unresolved-host'}`}else if(!pars){errorClass="UNPARSEABLE";errorMessage=`kind=${endpoint.endpoint_kind} content-type=${ctype}`}
  }catch(error){errorClass="FETCH";errorMessage=errorText(error).slice(0,800)}

  const latency=Date.now()-started;const healthId=await sha(`${endpoint.endpoint_id}|${observedAt}|${status}|${hash||errorClass||""}`);
  const healthInsert=await db.from("brian_source_endpoint_health_v2").insert({health_id:healthId,endpoint_id:endpoint.endpoint_id,observed_at:observedAt,reachable,parseable:pars,origin_match:origin,http_status:status,latency_ms:latency,content_type:ctype||null,payload_bytes:bytes,content_hash:hash,change_detected:change,error_class:errorClass,error_message:errorMessage,metadata:{resolved_url:resolvedUrl,source_arch_version:"V2",runtime:RUNTIME},shadow_only:true,live_execution:false});if(healthInsert.error)throw healthInsert.error;
  const recent=await db.from("brian_source_endpoint_health_v2").select("reachable,parseable,origin_match").eq("endpoint_id",endpoint.endpoint_id).order("observed_at",{ascending:false}).limit(3);if(recent.error)throw recent.error;
  let streak=0;for(const row of recent.data||[]){if(row.reachable&&row.parseable&&row.origin_match)streak++;else break}
  const healthScore=clamp(streak/2),authority=authorityPrior(endpoint.tier),lead=leadPrior(endpoint.tier),originality=originalityPrior(endpoint.tier),relevance=clamp(Number(endpoint.priority||50)/100),precision=.5,manipulation=clamp(Number(endpoint.manipulation_risk||.1));
  const composite=clamp(.30*authority+.15*lead+.15*originality+.15*relevance+.15*healthScore+.10*precision-.15*manipulation);
  const researchEligible=streak>=2&&composite>=.72&&["T0_RAW_TELEMETRY","T1_OFFICIAL_PRIMARY","T2_INSTITUTIONAL"].includes(endpoint.tier);
  const scoreId=await sha(`${endpoint.endpoint_id}|${observedAt}|${composite.toFixed(6)}|${streak}`);const reasons=[`health_streak=${streak}`,`composite=${composite.toFixed(3)}`,researchEligible?"research_eligible":"research_not_yet_eligible","decision_evidence_locked"];
  const scoreInsert=await db.from("brian_source_scores_v2").insert({score_id:scoreId,endpoint_id:endpoint.endpoint_id,assessed_at:observedAt,authority_score:authority,lead_time_score:lead,originality_score:originality,market_relevance_score:relevance,manipulation_risk:manipulation,historical_precision_score:precision,health_score:healthScore,composite_score:composite,sample_count:(recent.data||[]).length,eligible_for_research:researchEligible,eligible_for_decision_evidence:false,reasons,metadata:{source_arch_version:"V2",health_streak:streak,decision_evidence_locked:true,runtime:RUNTIME},shadow_only:true,live_execution:false});if(scoreInsert.error)throw scoreInsert.error;
  // World candidate metadata and assessments are synchronized by the SECURITY DEFINER score trigger.
  // brian_world_source_candidates is append-only for normal roles; never mutate it here.
  const lifecycle=researchEligible?"ACTIVE":(reachable&&pars&&origin?"VERIFYING":"DEGRADED");const update=await db.from("brian_source_endpoints_v2").update({lifecycle_state:lifecycle,updated_at:new Date().toISOString()}).eq("endpoint_id",endpoint.endpoint_id);if(update.error)throw update.error;
  return {endpoint_id:endpoint.endpoint_id,reachable,parseable:pars,origin_match:origin,http_status:status,latency_ms:latency,health_streak:streak,composite_score:Number(composite.toFixed(4)),research_eligible:researchEligible,lifecycle,error:errorMessage};
}

async function run(){
  const runStarted=new Date().toISOString();const now=Date.now();
  const endpoints=await db.from("brian_source_endpoints_v2").select("*").eq("access_mode","PUBLIC_NO_KEY").neq("lifecycle_state","DISABLED").neq("lifecycle_state","REJECTED").limit(500);if(endpoints.error)throw endpoints.error;
  const health=await db.from("brian_source_endpoint_health_v2").select("endpoint_id,observed_at").order("observed_at",{ascending:false}).limit(5000);if(health.error)throw health.error;
  const last=new Map<string,number>();for(const row of health.data||[]){const id=String(row.endpoint_id);if(!last.has(id))last.set(id,Date.parse(row.observed_at))}
  const due=(endpoints.data||[]).filter((e:any)=>now-(last.get(String(e.endpoint_id))||0)>=Number(e.polling_seconds||900)*1000).sort((a:any,b:any)=>(last.get(String(a.endpoint_id))||0)-(last.get(String(b.endpoint_id))||0)||Number(b.priority)-Number(a.priority)).slice(0,BATCH);
  const results:any[]=[];for(let i=0;i<due.length;i+=CONCURRENCY){const group=due.slice(i,i+CONCURRENCY);const settled=await Promise.all(group.map(async endpoint=>{try{return await probe(endpoint)}catch(error){return {endpoint_id:endpoint.endpoint_id,error:errorText(error),research_eligible:false}}}));results.push(...settled)}
  const ok=results.filter(r=>r.reachable&&r.parseable&&r.origin_match).length,eligible=results.filter(r=>r.research_eligible).length,degraded=results.filter(r=>r.error);
  const runId=await sha(`${COLLECTOR_ID}|${runStarted}|${results.length}|${ok}`);const receipt=await db.from("brian_collector_runs").insert({run_id:runId,collector_id:COLLECTOR_ID,started_at:runStarted,finished_at:new Date().toISOString(),status:degraded.length?"DEGRADED":"SUCCESS",observed_records:results.length,stored_records:results.length,degraded_sources:degraded.map(r=>`${r.endpoint_id}:${r.error}`).slice(0,24),error_class:null,error_message:null,evidence_class:EVIDENCE,shadow_only:true,live_execution:false,metadata:{source_arch_version:"V2",runtime:RUNTIME,batch_size:BATCH,concurrency:CONCURRENCY,ok,eligible,due_total:due.length,decision_evidence_locked:true,fair_queue:"OLDEST_DUE_FIRST"}});if(receipt.error)throw receipt.error;
  return {status:degraded.length?"DEGRADED":"SUCCESS",probed:results.length,ok,research_eligible:eligible,results,shadow_only:true,live_execution:false,decision_evidence_locked:true,dip_touched:false};
}

Deno.serve(async(req:Request)=>{if(req.method!=="POST")return out({error:"POST required"},405);try{await requireCronAuth(req,db)}catch(error){return out({status:"UNAUTHORIZED",error:errorText(error),shadow_only:true,live_execution:false},401)}try{const lease=await withCollectorLease(db,COLLECTOR_ID,LEASE_SECONDS,run);if(lease.contended)return out({status:"SKIPPED_LEASE_CONTENDED",shadow_only:true,live_execution:false,dip_touched:false});return out(lease.value)}catch(error){return out({status:"FAILED_CLOSED",error:errorText(error),shadow_only:true,live_execution:false,dip_touched:false},500)}});
