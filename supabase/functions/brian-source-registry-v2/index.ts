import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import { withCollectorLease } from "../_shared/collector_lease.ts";

const SUPABASE_URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(SUPABASE_URL,SERVICE,{auth:{persistSession:false,autoRefreshToken:false}});
const COLLECTOR_ID="brian-source-registry-v2";
const EVIDENCE="PROSPECTIVE_EVOLUTION_SHADOW";
const LEASE_SECONDS=180;
const BATCH=12;

function out(body:unknown,status=200){return new Response(JSON.stringify(body),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}})}
function err(e:unknown){return e instanceof Error?`${e.name}: ${e.message}`:String(e)}
function clamp(v:number){return Math.max(0,Math.min(1,v))}
async function sha(v:string|Uint8Array){const b=typeof v==="string"?new TextEncoder().encode(v):v;const d=new Uint8Array(await crypto.subtle.digest("SHA-256",b));return[...d].map(x=>x.toString(16).padStart(2,"0")).join("")}
function hostMatches(host:string,domain:string){const h=host.toLowerCase().replace(/^www\./,"");const d=domain.toLowerCase().replace(/^www\./,"");return h===d||h.endsWith(`.${d}`)}
function authorityPrior(tier:string){if(tier==="T1_OFFICIAL_PRIMARY")return .98;if(tier==="T0_RAW_TELEMETRY")return .95;if(tier==="T2_INSTITUTIONAL")return .90;if(tier==="T3_TOP_TIER_NEWS")return .82;if(tier==="T4_SPECIALIST")return .67;return .42}
function originalityPrior(tier:string){if(tier==="T0_RAW_TELEMETRY"||tier==="T1_OFFICIAL_PRIMARY")return 1;if(tier==="T2_INSTITUTIONAL")return .9;if(tier==="T3_TOP_TIER_NEWS")return .7;if(tier==="T4_SPECIALIST")return .6;return .35}
function leadPrior(tier:string){if(tier==="T0_RAW_TELEMETRY")return .95;if(tier==="T1_OFFICIAL_PRIMARY")return .88;if(tier==="T2_INSTITUTIONAL")return .62;if(tier==="T3_TOP_TIER_NEWS")return .55;if(tier==="T4_SPECIALIST")return .58;return .45}

async function readPrefix(r:Response,max=512_000){if(!r.body)return new Uint8Array();const reader=r.body.getReader();const chunks:Uint8Array[]=[];let total=0;try{while(total<max){const {value,done}=await reader.read();if(done)break;if(!value)continue;const take=Math.min(value.length,max-total);chunks.push(value.subarray(0,take));total+=take;if(take<value.length)break}}finally{try{await reader.cancel()}catch{}}const out=new Uint8Array(total);let off=0;for(const c of chunks){out.set(c,off);off+=c.length}return out}
function parseable(kind:string,contentType:string,text:string){const ct=contentType.toLowerCase();const t=text.trim().slice(0,2000).toLowerCase();if(kind==="RSS"||kind==="ATOM"||kind==="STATUSPAGE_ATOM")return /<(rss|feed)(\s|>)/i.test(text)||ct.includes("xml")||ct.includes("rss")||ct.includes("atom");if(kind==="JSON_API"){try{JSON.parse(text);return true}catch{return ct.includes("json")&&text.trim().length>2}}if(kind==="HTML")return ct.includes("html")||t.includes("<html")||t.includes("<!doctype");if(kind==="DATASET")return text.length>0;return false}

async function probe(endpoint:any){
  const started=Date.now();const observedAt=new Date().toISOString();let status:number|null=null,ctype="",bytes=0,hash:string|null=null,reachable=false,pars=false,origin=false,change=false,errorClass:string|null=null,errorMessage:string|null=null,resolvedUrl=endpoint.endpoint_url;
  try{
    const r=await fetch(endpoint.endpoint_url,{redirect:"follow",headers:{accept:endpoint.expected_content==="json"?"application/json,*/*;q=0.1":"application/rss+xml,application/atom+xml,application/xml,text/html,application/json;q=0.8,*/*;q=0.1","user-agent":"Brian-Market-OS/2.0 source-registry owner=operator"},signal:AbortSignal.timeout(9000)});
    status=r.status;ctype=r.headers.get("content-type")||"";resolvedUrl=r.url||endpoint.endpoint_url;let resolvedHost="";try{resolvedHost=new globalThis.URL(resolvedUrl).hostname}catch{}
    origin=hostMatches(resolvedHost,endpoint.canonical_domain);
    const raw=await readPrefix(r);bytes=raw.length;hash=raw.length?await sha(raw):null;const text=new TextDecoder("utf-8",{fatal:false}).decode(raw);pars=r.ok&&parseable(endpoint.endpoint_kind,ctype,text);reachable=r.ok&&origin&&bytes>0;
    const prev=await db.from("brian_source_endpoint_health_v2").select("content_hash").eq("endpoint_id",endpoint.endpoint_id).not("content_hash","is",null).order("observed_at",{ascending:false}).limit(1).maybeSingle();
    change=Boolean(hash&&prev.data?.content_hash&&prev.data.content_hash!==hash);
    if(!r.ok){errorClass="HTTP";errorMessage=`HTTP ${r.status}`}else if(!origin){errorClass="ORIGIN_MISMATCH";errorMessage=`redirected to ${resolvedHost||'unresolved-host'}`}else if(!pars){errorClass="UNPARSEABLE";errorMessage=`kind=${endpoint.endpoint_kind} content-type=${ctype}`}
  }catch(e){errorClass="FETCH";errorMessage=err(e).slice(0,800)}
  const latency=Date.now()-started;const healthId=await sha(`${endpoint.endpoint_id}|${observedAt}|${status}|${hash||errorClass||""}`);
  const ins=await db.from("brian_source_endpoint_health_v2").insert({health_id:healthId,endpoint_id:endpoint.endpoint_id,observed_at:observedAt,reachable,parseable:pars,origin_match:origin,http_status:status,latency_ms:latency,content_type:ctype||null,payload_bytes:bytes,content_hash:hash,change_detected:change,error_class:errorClass,error_message:errorMessage,metadata:{resolved_url:resolvedUrl,source_arch_version:"V2"},shadow_only:true,live_execution:false});if(ins.error)throw ins.error;

  const recent=await db.from("brian_source_endpoint_health_v2").select("reachable,parseable,origin_match").eq("endpoint_id",endpoint.endpoint_id).order("observed_at",{ascending:false}).limit(3);
  if(recent.error)throw recent.error;const samples=recent.data||[];let streak=0;for(const x of samples){if(x.reachable&&x.parseable&&x.origin_match)streak++;else break}
  const healthScore=clamp(streak/2);const authority=authorityPrior(endpoint.tier);const lead=leadPrior(endpoint.tier);const originality=originalityPrior(endpoint.tier);const relevance=clamp(Number(endpoint.priority||50)/100);const precision=.5;const manipulation=clamp(Number(endpoint.manipulation_risk||.1));
  const composite=clamp(.30*authority+.15*lead+.15*originality+.15*relevance+.15*healthScore+.10*precision-.15*manipulation);
  const researchEligible=streak>=2&&composite>=.72&&["T0_RAW_TELEMETRY","T1_OFFICIAL_PRIMARY","T2_INSTITUTIONAL"].includes(endpoint.tier);
  const scoreId=await sha(`${endpoint.endpoint_id}|${observedAt}|${composite.toFixed(6)}|${streak}`);
  const reasons=[`health_streak=${streak}`,`composite=${composite.toFixed(3)}`,researchEligible?"research_eligible":"research_not_yet_eligible","decision_evidence_locked"];
  const sq=await db.from("brian_source_scores_v2").insert({score_id:scoreId,endpoint_id:endpoint.endpoint_id,assessed_at:observedAt,authority_score:authority,lead_time_score:lead,originality_score:originality,market_relevance_score:relevance,manipulation_risk:manipulation,historical_precision_score:precision,health_score:healthScore,composite_score:composite,sample_count:samples.length,eligible_for_research:researchEligible,eligible_for_decision_evidence:false,reasons,metadata:{source_arch_version:"V2",health_streak:streak,decision_evidence_locked:true},shadow_only:true,live_execution:false});if(sq.error)throw sq.error;

  const lifecycle=researchEligible?"ACTIVE":(reachable&&pars&&origin?"VERIFYING":"DEGRADED");
  const uq=await db.from("brian_source_endpoints_v2").update({lifecycle_state:lifecycle,updated_at:new Date().toISOString()}).eq("endpoint_id",endpoint.endpoint_id);if(uq.error)throw uq.error;
  const cq=await db.from("brian_world_source_candidates").update({stage:researchEligible?"RESEARCHING":"VERIFYING",metadata:{...(endpoint.metadata||{}),source_arch_version:"V2",endpoint_id:endpoint.endpoint_id,tier:endpoint.tier,category:endpoint.category,region:endpoint.region,official_origin:endpoint.official_origin,origin_verification_pending:!researchEligible,decision_evidence_locked:true,health_streak:streak,composite_score:composite}}).eq("candidate_id",`source-arch-v2:${endpoint.endpoint_id}`);if(cq.error)throw cq.error;

  const assessmentId=await sha(`source-arch-v2|${endpoint.source_id}|${observedAt}|${composite.toFixed(6)}`);
  const aq=await db.from("brian_world_source_assessments").insert({assessment_id:assessmentId,source_id:endpoint.source_id,assessed_at:observedAt,authority_score:authority,freshness_score:healthScore,manipulation_penalty:manipulation,corroboration_penalty:endpoint.corroboration_required ? .10 : 0,access_penalty:endpoint.access_mode==="PUBLIC_NO_KEY"?0:.25,trust_score:composite,eligible_for_research:researchEligible,eligible_for_decision_evidence:false,reasons,metadata:{source_arch_version:"V2",endpoint_id:endpoint.endpoint_id,tier:endpoint.tier,lead_time_score:lead,originality_score:originality,market_relevance_score:relevance,historical_precision_score:precision,health_score:healthScore,decision_evidence_locked:true},evidence_class:EVIDENCE,shadow_only:true,live_execution:false});if(aq.error)throw aq.error;
  return {endpoint_id:endpoint.endpoint_id,reachable,parseable:pars,origin_match:origin,http_status:status,latency_ms:latency,health_streak:streak,composite_score:composite,research_eligible:researchEligible,lifecycle,error:errorMessage};
}

async function run(){
  const now=new Date();const endpoints=await db.from("brian_source_endpoints_v2").select("*").eq("access_mode","PUBLIC_NO_KEY").neq("lifecycle_state","DISABLED").neq("lifecycle_state","REJECTED").order("priority",{ascending:false}).limit(200);if(endpoints.error)throw endpoints.error;
  const health=await db.from("brian_source_endpoint_health_v2").select("endpoint_id,observed_at").order("observed_at",{ascending:false}).limit(5000);if(health.error)throw health.error;
  const last=new Map<string,number>();for(const h of health.data||[]){const id=String(h.endpoint_id);if(!last.has(id))last.set(id,Date.parse(h.observed_at))}
  const due=(endpoints.data||[]).filter((e:any)=>{const t=last.get(String(e.endpoint_id))||0;return now.getTime()-t>=Number(e.polling_seconds||900)*1000}).sort((a:any,b:any)=>Number(b.priority)-Number(a.priority)||(last.get(String(a.endpoint_id))||0)-(last.get(String(b.endpoint_id))||0)).slice(0,BATCH);
  const results=[];for(const endpoint of due){try{results.push(await probe(endpoint))}catch(e){results.push({endpoint_id:endpoint.endpoint_id,error:err(e),research_eligible:false})}}
  const ok=results.filter((r:any)=>r.reachable&&r.parseable&&r.origin_match).length;const eligible=results.filter((r:any)=>r.research_eligible).length;
  const startedAt=new Date().toISOString();const runId=await sha(`${COLLECTOR_ID}|${startedAt}|${results.length}|${ok}`);await db.from("brian_collector_runs").insert({run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:new Date().toISOString(),status:results.some((r:any)=>r.error)?"DEGRADED":"SUCCESS",observed_records:results.length,stored_records:results.length,degraded_sources:results.filter((r:any)=>r.error).map((r:any)=>`${r.endpoint_id}:${r.error}`).slice(0,20),error_class:null,error_message:null,evidence_class:EVIDENCE,shadow_only:true,live_execution:false,metadata:{source_arch_version:"V2",batch_size:BATCH,ok,eligible,due_total:due.length,decision_evidence_locked:true}});
  return {status:"OK",probed:results.length,ok,research_eligible:eligible,results,shadow_only:true,live_execution:false,decision_evidence_locked:true,dip_touched:false};
}

Deno.serve(async(req:Request)=>{if(req.method!=="POST")return out({error:"POST required"},405);try{await requireCronAuth(req,db)}catch(e){return out({status:"UNAUTHORIZED",error:err(e),shadow_only:true,live_execution:false},401)}try{const lease=await withCollectorLease(db,COLLECTOR_ID,LEASE_SECONDS,run);if(lease.contended)return out({status:"SKIPPED_LEASE_CONTENDED",shadow_only:true,live_execution:false,dip_touched:false});return out(lease.value)}catch(e){return out({status:"FAILED_CLOSED",error:err(e),shadow_only:true,live_execution:false,dip_touched:false},500)}});
