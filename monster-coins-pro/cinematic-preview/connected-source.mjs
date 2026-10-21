// Keep protected status reads on the verified same-origin web API route.
export const DATA_ROOT='/api/brian';
export const DASHBOARD_KEY='mcp-dashboard-key-v1';
const FIELDS=['decision_id','compiler_version','observed_at','asset_id','observed_reference_price','action','direction','evidence_score','independent_group_count','support_groups','conflict_groups','source_cost_quote_id','estimated_round_trip_cost_bps','gross_edge_bps','net_edge_bps','veto_reason','reason','source_observation_ids','source_intrabar_event_ids'];
export function decisionRecord(row){return Object.fromEntries(FIELDS.filter(k=>Object.hasOwn(row,k)).map(k=>[k,row[k]]));}
export function connectDecision(heartbeat,control){
 const decisions=Array.isArray(control?.alpha_v2?.decisions)?control.alpha_v2.decisions:[];
 if(control?.live_execution===true||control?.alpha_v2?.live_execution===true)return {heartbeat,decisions:[],matched:false};
 const rows=decisions.filter(d=>d&&typeof d==='object'&&typeof d.decision_id==='string'&&Number.isFinite(Date.parse(d.observed_at))&&d.live_execution!==true).map(decisionRecord);
 const a=heartbeat.alpha;const matches=a?rows.filter(d=>a.decision_id?d.decision_id===a.decision_id:d.asset_id===a.asset_id&&d.action===a.action&&Date.parse(d.observed_at)===Date.parse(a.observed_at)):[];
 return {heartbeat:matches.length===1?{...heartbeat,alpha:{...a,...matches[0]}}:heartbeat,decisions:rows,matched:matches.length===1};
}
export async function readConnectedSource({fetcher=fetch,storage=localStorage,timeout=15000}={}){
 const response=await fetcher(DATA_ROOT+'/brian-frontier-heartbeat-public',{credentials:'omit',cache:'no-store',signal:AbortSignal.timeout(timeout)});
 if(!response.ok)throw Error('Heartbeat HTTP '+response.status);const heartbeat=await response.json();
 let key='';try{key=(storage.getItem(DASHBOARD_KEY)||'').trim();}catch{}
 if(!key)return {heartbeat,decisions:[],connection:'public',matched:false,fetchedAt:new Date().toISOString()};
 try{
  const res=await fetcher(DATA_ROOT+'/brian-control-center',{method:'POST',headers:{'content-type':'application/json','x-brian-dashboard-key':key},body:JSON.stringify({action:'status'}),credentials:'omit',cache:'no-store',signal:AbortSignal.timeout(Math.max(timeout,30000))});
  if(!res.ok)return {heartbeat,decisions:[],connection:res.status===401||res.status===403?'locked':'unavailable',matched:false,connectionDetail:'HTTP '+res.status,fetchedAt:new Date().toISOString()};
  const control=await res.json(),result=connectDecision(heartbeat,control);return {...result,connection:'authenticated',fetchedAt:new Date().toISOString()};
 }catch(error){return {heartbeat,decisions:[],connection:'unavailable',matched:false,connectionDetail:['TimeoutError','AbortError','SyntaxError','TypeError'].includes(error?.name)?error.name:'REQUEST_FAILED',fetchedAt:new Date().toISOString()};}
}
