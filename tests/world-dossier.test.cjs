'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const {stripTypeScriptTypes}=require('node:module'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../supabase/functions/_shared/world_dossier.ts'),'utf8');
const box={URL};vm.runInNewContext(stripTypeScriptTypes(source,{mode:'strip'}).replace(/^export /gm,'')+'\nglobalThis.api={buildWorldDossiers,selectDossierEventIds,safeSourceUri};',box);
const {buildWorldDossiers,selectDossierEventIds,safeSourceUri}=box.api;
const id=n=>n.toString(16).padStart(64,'0'),ref=n=>'intel_event:'+id(n);
const at='2026-10-19T09:00:00Z',before='2026-10-19T08:00:00Z',after='2026-10-19T10:00:00Z';
const m=(refs=[ref(1)])=>({mechanism_id:'m',narrative_id:'energy',observed_at:at,confidence:.97,cause:'conditional mechanism',transmission:['supply','prices'],evidence_refs:refs});
const s=(n=1)=>({event_id:id(n),source_id:'official:publisher',claim:'an unverified claim',provenance_uri:'https://www.example.org/news/'+n,trust_class:'OFFICIAL_PRIMARY',first_observed_at:before,created_at:before,published_at:before});
const dossier=(mechanism=m(),sources=[s()],scenarios=[],state='AVAILABLE')=>JSON.parse(JSON.stringify(buildWorldDossiers([mechanism],sources,scenarios,at,state)[0]));
test('heuristic .97 and official label never become calibrated probability or verified claims',()=>{
 const d=dossier();assert.equal(d.score.kind,'HEURISTIC_UNCALIBRATED');assert.equal(d.score.calibrated_probability,null);
 assert.equal(d.trade_ready,false);assert.equal(d.source_claims[0].claim_verification,'NOT_INDEPENDENTLY_VERIFIED');
 assert.equal(d.source_summary.source_labels_are_verification,false);assert.equal(d.live_execution,false);
});
test('many copies on one publisher remain one host group with unknown independent groups',()=>{
 const sources=Array.from({length:50},(_,i)=>({...s(i+1),trust_class:'UNVERIFIED_DISCOVERY'}));
 const d=dossier(m(sources.map(x=>'intel_event:'+x.event_id)),sources);
 assert.equal(d.source_summary.publisher_host_groups,1);assert.equal(d.source_summary.independently_verified_source_groups,null);
 assert.equal(d.source_summary.discovery_label,50);assert.equal(d.trade_ready,false);assert.equal(d.source_claims.length,8);
});
test('duplicate refs do not multiply evidence; unsupported and missing refs remain incomplete',()=>{
 const d=dossier(m([ref(1),ref(1),ref(2),'unsupported:claim']));
 assert.equal(d.coverage.references,3);assert.equal(d.coverage.available,1);assert.equal(d.coverage.unresolved,2);assert.equal(d.coverage.complete,false);
});
test('future capture, persistence and publication cannot enter an earlier mechanism dossier',()=>{
 for(const field of ['first_observed_at','created_at','published_at']){
  const d=dossier(m(),[{...s(),[field]:after}]);assert.equal(d.coverage.available,0);assert.equal(d.coverage.complete,false);
 }
});
test('missing knowledge time and invalid/future mechanism time cannot claim source coverage',()=>{
 for(const row of [{...s(),first_observed_at:undefined},{...s(),created_at:undefined}])assert.equal(dossier(m(),[row]).coverage.available,0);
 for(const value of ['bad-time',after]){const d=dossier({...m(),observed_at:value});assert.equal(d.coverage.available,0);assert.ok(d.missing_requirements.includes('MECHANISM_TIME_INVALID'));}
});
test('lookup failure is explicit and cannot reuse supplied rows as successful evidence',()=>{
 const d=dossier(m(),[s()],[],'QUERY_FAILED');assert.equal(d.coverage.lookup_state,'QUERY_FAILED');assert.equal(d.coverage.available,0);
 assert.ok(d.missing_requirements.includes('SOURCE_LOOKUP_FAILED'));assert.equal(d.coverage.complete,false);
});
test('source URI policy rejects execution, private hosts, credentials and query leakage',()=>{
 for(const uri of ['javascript:alert(1)','http://example.org','https://name:secret@example.org','https://127.0.0.1','https://intranet.local','https://example.org:8443'])assert.equal(safeSourceUri(uri),null);
 assert.equal(safeSourceUri('https://example.org/news?token=hidden#secret'),'https://example.org/news');
});
test('bounded lookup is fair to later narratives and deterministic',()=>{
 const mechanisms=Array.from({length:40},(_,i)=>({...m(Array.from({length:100},(_,j)=>ref(1+i*100+j))),mechanism_id:'m'+i}));
 const ids=selectDossierEventIds(mechanisms);assert.equal(ids.length,400);
 for(let i=0;i<40;i++)assert.ok(ids.includes(id(1+i*100)));
 assert.equal(JSON.stringify(ids),JSON.stringify(selectDossierEventIds(mechanisms)));
});
test('generic invalidators are proposals, never counter-evidence receipts',()=>{
 const d=dossier(m(),[s()],[{mechanism_id:'m',branch:'MECHANISM_HOLDS',invalidators:['source retraction']}]);
 assert.equal(d.counter_evidence.status,'NOT_ASSESSED');assert.deepEqual(d.counter_evidence.receipts,[]);
 assert.equal(d.counter_evidence.proposed_invalidators_are_evidence,false);assert.equal(d.priced_in.status,'NOT_ASSESSED');assert.equal(d.actionable_horizon.seconds,null);
});
test('builder never mutates historical scores, sources or scenarios',()=>{
 const inputs=[m(),[s()],[]],prior=JSON.stringify(inputs);dossier(...inputs);assert.equal(JSON.stringify(inputs),prior);
 assert.equal(dossier({...m(),confidence:NaN}).score.value,null);assert.equal(dossier({...m(),confidence:1.2}).score.value,null);
 assert.throws(()=>buildWorldDossiers([],[],[],'bad-time'),/DOSSIER_AS_OF_INVALID/);
});
function ui(){const nodes=new Map();const get=id=>{if(!nodes.has(id))nodes.set(id,{innerHTML:'',textContent:'',value:'',classList:{toggle(){},add(){},remove(){}},addEventListener(){}});return nodes.get(id)};
 const ctx={URL,document:{getElementById:get},localStorage:{getItem(){return''}},setInterval(){},setTimeout(){},clearTimeout(){}};
 vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../monster-coins-pro/world.js'),'utf8'),ctx);return{ctx,nodes,get};}
test('real world renderer escapes claims, blocks executable links and labels incomplete dossiers',()=>{
 const {ctx,get}=ui();const d=dossier(m(),[{...s(),claim:'<img src=x onerror=alert(1)>',provenance_uri:'javascript:alert(1)'}]);
 ctx.render({summary:{},dossiers:[d]});const html=get('dossierList').innerHTML;
 assert.match(html,/&lt;img/);assert.doesNotMatch(html,/<img|href="javascript:/);assert.match(html,/İşlem için hazır değil/);
 assert.match(html,/kalibre edilmiş olasılık değildir/);assert.match(html,/bağımsız olarak doğrulanmadı/);
});
test('legacy API and missing scores do not imply verified dossiers or zero confidence',()=>{
 const {ctx,get}=ui();ctx.render({summary:{}});assert.match(get('dossierList').innerHTML,/henüz yok/);
 assert.equal(ctx.num(null),'—');assert.equal(ctx.num(0),'0.00');
 assert.equal(ctx.sourceLink('https://x:y@example.org','bad'),'bad');
});

function endpoint(lookupFailure=false){
 let handler;const reads=[];
 const data={brian_dashboard_auth:{dashboard_key_sha256:require('node:crypto').createHash('sha256').update('test-only-dashboard-key').digest('hex')},
  brian_world_causal_mechanisms:[m()],brian_world_scenario_snapshots:[],brian_intel_events:[s()]};
 const db={from(table){reads.push(table);const q={select(){return q},eq(){return q},in(){return q},order(){return q},limit(n){if(table==='brian_intel_events')assert.equal(n,400);return q},abortSignal(){return q},single(){return q},then(resolve,reject){return Promise.resolve({data:data[table]??[],error:table==='brian_intel_events'&&lookupFailure?{message:'test query failure'}:null}).then(resolve,reject)}};return q}};
 const ctx={Request,Response,TextEncoder,AbortSignal,crypto:require('node:crypto').webcrypto,
  createClient(){return db},buildWorldDossiers,selectDossierEventIds,
  Deno:{env:{get(){return'test-only'}},serve(fn){handler=fn}}};
 const status=fs.readFileSync(path.join(__dirname,'../supabase/functions/brian-world-status/index.ts'),'utf8');
 vm.runInNewContext(stripTypeScriptTypes(status,{mode:'strip'}).replace(/^import .*;\s*$/gm,''),ctx);
 return{handler,reads};
}
test('actual status handler joins source evidence and real UI renders its dossier response',async()=>{
 const {handler,reads}=endpoint();const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{'x-brian-dashboard-key':'test-only-dashboard-key'}}));
 assert.equal(response.status,200);const body=await response.json();assert.equal(body.status,'ONLINE');assert.equal(body.dossiers.length,1);
 assert.equal(body.dossiers[0].coverage.available,1);assert.equal(body.semantics.confidence_kind,'HEURISTIC_UNCALIBRATED');
 assert.ok(reads.includes('brian_intel_events'));const {ctx,get}=ui();ctx.render(body);assert.match(get('dossierList').innerHTML,/an unverified claim/);
});
test('actual status handler keeps wrong and missing dashboard keys closed before evidence lookup',async()=>{
 for(const headers of [{},{'x-brian-dashboard-key':'incorrect-test-value'}]){
  const {handler,reads}=endpoint();const response=await handler(new Request('https://example.invalid',{method:'POST',headers}));
  assert.equal(response.status,401);assert.ok(!reads.includes('brian_intel_events'));
 }
});
test('actual status handler reports source-query degradation and never ready dossiers',async()=>{
 const {handler}=endpoint(true);const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{'x-brian-dashboard-key':'test-only-dashboard-key'}}));
 const body=await response.json();assert.equal(body.status,'DEGRADED');assert.equal(body.dossiers[0].coverage.lookup_state,'QUERY_FAILED');assert.equal(body.dossiers[0].trade_ready,false);
});
test('status endpoint rejects unsupported methods before auth or source queries',async()=>{
 const {handler,reads}=endpoint();const response=await handler(new Request('https://example.invalid'));
 assert.equal(response.status,405);assert.equal(reads.length,0);
});
