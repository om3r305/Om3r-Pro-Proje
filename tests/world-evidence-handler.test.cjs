const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {stripTypeScriptTypes}=require('node:module'),{webcrypto}=require('node:crypto');
const dir='supabase/functions/brian-world-research/';
const compile=p=>stripTypeScriptTypes(fs.readFileSync(p,'utf8')).replace(/^import[^\n]+\n/gm,'').replace(/export /g,'');
const logic=vm.runInNewContext(compile(dir+'logic.ts')+';({RESEARCH_VERSION,readOfficial,assessPrimaryText,searchCounterEvidence,reactionProbe})',{URL,AbortSignal,crypto:webcrypto,TextEncoder,TextDecoder,Uint8Array,fetch});
function load(){
 const state={enabled:true,fail:false,events:[],documentFails:false},calls=[];let handler;
 const db={from(table){const q={select(){return q},eq(){return q},order(){return q},limit(){return q},abortSignal(){return q},single(){return Promise.resolve({data:{config_value:state.enabled?'true':'false'}})},maybeSingle(){return Promise.resolve({data:null})},insert(row){calls.push({table,row});return q},then(resolve){resolve({data:null,error:null})}};return q},rpc(){calls.push('inputs');return{abortSignal(){return Promise.resolve(state.fail?{error:{message:'offline'}}:{data:{events:state.events}})}}}};
 vm.runInNewContext(compile(dir+'index.ts'),{...logic,createClient:()=>db,requireRealtimeInternal:async req=>{if(req.headers.get('x-brian-internal-key')!=='test-internal')throw Error('UNAUTHORIZED')},withCollectorLease:async(db,id,ttl,work)=>{calls.push('lease');return{contended:false,value:await work()}},documentText:()=>({title:'Title',text:'Central bank interest rates increase. '.repeat(10),indexOnly:false,primaryDocument:null}),readOfficial:async()=>{calls.push('fetch');if(state.documentFails)throw Error('OFFICIAL_HTTP_403');return{uri:'https://example.org/a',hash:'a'.repeat(64),text:'body',bytes:1000,received_at:new Date().toISOString()}},crypto:webcrypto,TextEncoder,Date,Response,Request,AbortSignal,console,Deno:{env:{get:()=> 'test-only'},serve:fn=>{handler=fn}}});
 return {state,calls,handler};
}
const request=key=>new Request('https://test.invalid/',{method:'POST',headers:key?{'x-brian-internal-key':key}:{}});
test('research worker authenticates before queries, respects operator stop and owns no trading writes',async()=>{
 const h=load();assert.equal((await h.handler(request())).status,401);assert.deepEqual(h.calls,[]);h.state.enabled=false;const d=await(await h.handler(request('test-internal'))).json();assert.equal(d.status,'OPERATOR_STOPPED');assert.deepEqual(h.calls,[]);
});
test('actual worker persists official body receipt with explicit unresolved truth/price semantics',async()=>{
 const h=load();h.state.events=[{event_id:'e',source_id:'s',claim:'Central bank interest rates increase',canonical_domain:'example.org',provenance_uri:'https://example.org/a',first_observed_at:'2026-10-19T12:00:00Z',counter_pool:[],probes:[]}];
 const response=await h.handler(request('test-internal')),d=await response.json();assert.equal(response.status,200);assert.equal(d.stored,1);
 const row=h.calls.find(c=>c.table==='brian_world_evidence_reviews').row;assert.equal(row.status,'READ');assert.equal(row.review.document.claim_verified_independently,false);assert.equal(row.review.trade_ready,false);assert.equal(row.review.external_content_used_as_instruction,false);assert.equal(row.shadow_only,true);assert.equal(row.live_execution,false);
 assert.ok(h.calls.filter(c=>c?.table).every(c=>['brian_world_evidence_reviews','brian_collector_runs'].includes(c.table)));
});
test('provider failure is stored as failure and operation failure emits FAILED_CLOSED receipt',async()=>{
 const h=load();h.state.documentFails=true;h.state.events=[{event_id:'e',source_id:'s',claim:'claim',canonical_domain:'example.org',provenance_uri:'https://example.org/a',first_observed_at:'2026-10-19T12:00:00Z',counter_pool:[],probes:[]}];
 const d=await(await h.handler(request('test-internal'))).json();assert.equal(d.status,'DEGRADED');assert.equal(h.calls.find(c=>c.table==='brian_world_evidence_reviews').row.status,'FAILED');
 const fail=load();fail.state.fail=true;assert.equal((await fail.handler(request('test-internal'))).status,503);assert.equal(fail.calls.find(c=>c.table==='brian_collector_runs').row.status,'FAILED');
});
