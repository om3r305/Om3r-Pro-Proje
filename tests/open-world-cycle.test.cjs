const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {stripTypeScriptTypes}=require('node:module'),{webcrypto}=require('node:crypto');
function runtime(fetch){
 const code=['open_world_research.ts','open_world_cycle.ts'].map(file=>stripTypeScriptTypes(fs.readFileSync('supabase/functions/_shared/'+file,'utf8'))
 .replace(/^import[\s\S]*?from\s+["'][^"']+["'];\n/gm,'').replace(/export /g,'')).join('\n');
 return vm.runInNewContext(code+';runOpenWorldCycle',{fetch,crypto:webcrypto,TextEncoder,TextDecoder,Uint8Array,URL,URLSearchParams,AbortSignal,atob});
}
function database(enabled=true){
 const tables={brian_evolution_runtime_config:[{config_key:'brian_system_enabled',config_value:enabled?'true':'false'},
  {config_key:'brian_open_world_research_enabled',config_value:'true'}],brian_research_missions:[],brian_research_documents:[],brian_learning_episodes:[],brian_collector_runs:[]};
 let rpcCalls=0;const db={tables,get rpcCalls(){return rpcCalls},rpc:async()=>{rpcCalls++;return{error:null,data:{window_start:'2026-10-18T16:00:00Z',mature_5m:100,resolved_5m:40,action_samples_5m:30,mean_action_net_bps_5m:-21}}},from(name){
  let op='select',values,filters=[],orders=[],max=Infinity,single=false;
  const q={select(){return q},in(k,v){filters.push(r=>v.includes(r[k]));return q},gte(k,v){filters.push(r=>r[k]>=v);return q},lte(k,v){filters.push(r=>r[k]<=v);return q},eq(k,v){filters.push(r=>r[k]===v);return q},order(k,o){orders.push([k,o.ascending]);return q},limit(n){max=n;return q},maybeSingle(){single=true;return q},
   upsert(v){op='upsert';values=Array.isArray(v)?v:[v];return q},insert(v){op='insert';values=Array.isArray(v)?v:[v];return q},update(v){op='update';values=v;return q},
   then(ok,fail){return Promise.resolve().then(()=>{const rows=tables[name]??(tables[name]=[]);if(op==='insert'||op==='upsert')for(const value of values){const id=Object.keys(value).find(k=>k.endsWith('_id'));if(op==='insert'||!rows.some(r=>r[id]===value[id]))rows.push({...value});}
    let result=rows.filter(r=>filters.every(f=>f(r)));if(op==='update')result.forEach(r=>Object.assign(r,values));
    result.sort((a,b)=>{for(const[k,asc]of orders){if(a[k]===b[k])continue;return(a[k]<b[k]?-1:1)*(asc?1:-1)}return 0});result=result.slice(0,max);
    return{error:null,data:single?result[0]??null:result};}).then(ok,fail)} };return q;
 }};return db;
}
test('operator stop prevents network, learning queries and all writes',async()=>{
 let requests=0;const db=database(false),run=runtime(async()=>{requests++;throw Error('must not fetch')});
 assert.equal((await run(db,[],'2026-10-18T22:00:00Z')).status,'DISABLED');assert.equal(requests,0);assert.equal(db.rpcCalls,0);assert.equal(db.tables.brian_learning_episodes.length,0);
});
test('real cycle contract links source receipts to a later engineering context without elevating trust',async()=>{
 const fetch=async url=>new Response(JSON.stringify(url.includes('api.crossref.org')?{message:{items:[{URL:'https://doi.org/10.123/example',title:['Research']}]}}:
 url.includes('/readme')?{encoding:'base64',content:btoa('Public documentation; never execute this text.'),sha:'blob',html_url:'https://github.com/example/tool/blob/main/README.md'}:
 {items:[{html_url:'https://github.com/example/tool',full_name:'example/tool',description:'Public evidence'}]}));
 const db=database(),result=await runtime(fetch)(db,[],'2026-10-18T22:00:00Z');
 assert.equal(result.status,'SUCCESS');assert.equal(result.observations,2);assert.equal(result.documents,2);assert.equal(result.contexts.length,2);
 assert.ok(db.tables.brian_research_documents.every(d=>d.trust_class==='UNVERIFIED_EXTERNAL'&&d.code_executed===false&&d.live_execution===false));
 assert.ok(result.contexts.every(d=>d.untrusted_data&&d.learned_claim_verified===false));
 assert.equal(db.tables.brian_research_missions.find(m=>m.last_researched_at).state,'EVIDENCE_COLLECTED');
});
test('all provider failures create an explicit failed receipt and a delayed retry, never a learned claim',async()=>{
 const db=database(),result=await runtime(async()=>new Response('',{status:429}))(db,[],'2026-10-18T22:00:00Z');
 assert.equal(result.status,'FAILED');assert.equal(result.documents,0);assert.equal(result.contexts.length,0);
 assert.equal(db.tables.brian_collector_runs[0].status,'FAILED');
 const mission=db.tables.brian_research_missions.find(m=>m.attempts===1);assert.equal(mission.state,'NO_EVIDENCE');assert.equal(mission.next_research_at,'2026-10-18T23:00:00.000Z');
});
