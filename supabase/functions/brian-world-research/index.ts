import { createClient } from 'npm:@supabase/supabase-js@2.116.0';
import { withCollectorLease } from '../_shared/collector_lease.ts';
import { requireRealtimeInternal } from '../_shared/realtime_internal_auth.ts';
import { RESEARCH_VERSION, readOfficial, assessPrimaryText, searchCounterEvidence, reactionProbe, type CounterEvent, type ReactionPoint } from './logic.ts';
import { documentText } from './reader.ts';
const db=createClient(Deno.env.get('SUPABASE_URL')!,Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!,{auth:{persistSession:false,autoRefreshToken:false}});
const ID='brian-world-evidence-research-v1';
const out=(body:unknown,status=200)=>new Response(JSON.stringify(body),{status,headers:{'content-type':'application/json','cache-control':'no-store'}});
async function sha(s:string){return[...new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(s)))].map(b=>b.toString(16).padStart(2,'0')).join('');}
async function enabled(){const q=await db.from('brian_evolution_runtime_config').select('config_value').eq('config_key','brian_system_enabled').abortSignal(AbortSignal.timeout(4000)).single();if(q.error)throw Error('OPERATOR_CONTROL_UNAVAILABLE');return q.data?.config_value==='true';}
type Event=CounterEvent&{canonical_domain:string;counter_pool_scan_limit?:number;counter_pool_scan_truncated?:boolean;probes:Array<{asset_id:string;points:ReactionPoint[]}>;counter_pool:CounterEvent[]};
Deno.serve(async(req:Request)=>{
 if(req.method!=='POST')return out({error:'POST_REQUIRED'},405);
 try{await requireRealtimeInternal(req);}catch{return out({status:'UNAUTHORIZED'},401);}
 const started=new Date().toISOString();
 try{
  if(!await enabled())return out({status:'OPERATOR_STOPPED',shadow_only:true,live_execution:false});
  const lease=await withCollectorLease(db,ID,50,async()=>{
   const last=await db.from('brian_collector_runs').select('started_at').eq('collector_id',ID).order('started_at',{ascending:false}).limit(1).abortSignal(AbortSignal.timeout(4000)).maybeSingle();
   if(last.error)throw Error('RESEARCH_CADENCE_UNAVAILABLE');
   if(last.data?.started_at&&Date.now()-Date.parse(last.data.started_at)<240000)return out({status:'SKIPPED_INTERVAL',shadow_only:true,live_execution:false});
   const input=await db.rpc('brian_world_research_inputs_v1').abortSignal(AbortSignal.timeout(5000));if(input.error)throw Error('RESEARCH_INPUTS_UNAVAILABLE');
   const events=(input.data?.events??[]) as Event[];if(events.length>2)throw Error('RESEARCH_INPUT_BOUND_EXCEEDED');
   const sourceAccess=input.data?.source_access??{status:'UNKNOWN',cooldown_sources:[]};
   const contactQ=await db.rpc('brian_research_contact_v1').abortSignal(AbortSignal.timeout(4000));const contact=!contactQ.error&&typeof contactQ.data==='string'?contactQ.data:null;
   const cooldownSources=Array.isArray(sourceAccess.cooldown_sources)?sourceAccess.cooldown_sources:[];
   let stored=0,failed=0;
   for(const event of events){
    let status='FAILED',uri:string|null=null,hash:string|null=null,document:Record<string,unknown>={body_received:false,claim_verified_independently:false},error:string|null=null;
    try{
     let fetched=await readOfficial(event.provenance_uri!,event.canonical_domain,undefined,contact),parsed=documentText(fetched.text,fetched.uri),followError:string|null=null;
     if(parsed.indexOnly&&parsed.primaryDocument){try{const next=await readOfficial(parsed.primaryDocument,event.canonical_domain,undefined,contact);const nextParsed=documentText(next.text,next.uri);fetched=next;parsed=nextParsed;}catch(e){followError=e instanceof Error?e.message:'PRIMARY_DOCUMENT_UNAVAILABLE';}}
     uri=fetched.uri;hash=fetched.hash;document={...assessPrimaryText(event.claim,parsed.title,parsed.text,parsed.indexOnly),response_sha256:hash,response_bytes:fetched.bytes,received_at:fetched.received_at,primary_document_error:followError};status=parsed.indexOnly?'INDEX_ONLY':'READ';
    }catch(e){failed++;error=e instanceof Error?e.message:'DOCUMENT_READ_FAILED';}
    const assessed=new Date().toISOString();
    const review={event:{event_id:event.event_id,source_id:event.source_id,claim:event.claim,first_observed_at:event.first_observed_at,published_at:event.published_at??null},document,
     counter_evidence:{...searchCounterEvidence(event,event.counter_pool,assessed),selection_scan_limit:event.counter_pool_scan_limit??null,selection_scan_truncated:event.counter_pool_scan_truncated??null},
     market_reaction:{role:'BTC_ETH_BROAD_MARKET_PROBES_NOT_EVENT_ATTRIBUTION',probes:event.probes.map(p=>({asset_id:p.asset_id,...reactionProbe(p.points,event.first_observed_at,event.published_at??null,assessed)}))},
     error,external_content_used_as_instruction:false,trade_ready:false,direct_alpha_influence:false};
    if(!await enabled())return out({status:'OPERATOR_STOPPED',stored,shadow_only:true,live_execution:false});
    const ins=await db.from('brian_world_evidence_reviews').insert({review_id:await sha(`${RESEARCH_VERSION}|${event.event_id}|${assessed}|${hash??error}`),event_id:event.event_id,assessed_at:assessed,version:RESEARCH_VERSION,status,document_uri:uri,content_sha256:hash,review,shadow_only:true,live_execution:false}).abortSignal(AbortSignal.timeout(4000));
    if(ins.error)throw Error('RESEARCH_REVIEW_PERSIST_FAILED');stored++;
   }
   const status=failed||cooldownSources.length?'DEGRADED':'SUCCESS',finished=new Date().toISOString();
   const record=await db.from('brian_collector_runs').insert({run_id:await sha(`${ID}|${started}|${finished}`),collector_id:ID,started_at:started,finished_at:finished,status,observed_records:events.length,stored_records:stored,degraded_sources:failed||cooldownSources.length?['primary_document_reads_partial']:[],error_class:null,error_message:null,evidence_class:'PROSPECTIVE_EVOLUTION_SHADOW',shadow_only:true,live_execution:false,metadata:{version:RESEARCH_VERSION,failed_document_reads:failed,source_access:sourceAccess,external_claims_independently_verified:0,market_reaction_is_not_priced_in_probability:true}}).abortSignal(AbortSignal.timeout(4000));
   if(record.error)throw Error('RESEARCH_RECEIPT_UNAVAILABLE');
   return out({status,assessed:events.length,stored,failed_document_reads:failed,source_access:sourceAccess,shadow_only:true,live_execution:false});
  });
  return lease.contended?out({status:'SKIPPED_LEASE_CONTENDED',shadow_only:true,live_execution:false}):lease.value!;
 }catch(e){
  const error=e instanceof Error?e.message:'RESEARCH_FAILED',finished=new Date().toISOString();
  try { await db.from('brian_collector_runs').insert({run_id:await sha(`${ID}|${started}|${finished}|failed`),collector_id:ID,started_at:started,finished_at:finished,status:'FAILED',observed_records:0,stored_records:0,degraded_sources:['research_unavailable'],error_class:'RESEARCH_FAILED',error_message:error.slice(0,200),evidence_class:'PROSPECTIVE_EVOLUTION_SHADOW',shadow_only:true,live_execution:false,metadata:{version:RESEARCH_VERSION,counts_unavailable:true}}).abortSignal(AbortSignal.timeout(4000)); } catch { /* response still reports failure */ }
  return out({status:'FAILED_CLOSED',error,shadow_only:true,live_execution:false},503);
 }
});
