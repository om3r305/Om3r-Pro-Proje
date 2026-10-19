const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {stripTypeScriptTypes}=require('node:module');
const {webcrypto,createHash}=require('node:crypto');
function load(){
 const compile=p=>stripTypeScriptTypes(fs.readFileSync(p,'utf8')).replace(/^import[^\n]+\n/gm,'').replace(/export /g,'');
 const coverage=vm.runInNewContext(compile('supabase/functions/brian-shadow-scorecard/coverage.ts')+';({COHORT,OUTCOME_COVERAGE_SQL})');
 const model=vm.runInNewContext(compile('supabase/functions/brian-shadow-scorecard/model.ts')+';({summarize,alphaSummary})');
 const calls=[],state={failed:false};let handler;
 const tx=async(parts)=>{const text=parts.join('?');calls.push(text);if(text.includes('transaction_timestamp'))return[{at:new Date('2026-10-19T12:00:00Z')}];return[];};
 tx.savepoint=async fn=>{calls.push('savepoint');return fn(tx)};
 tx.unsafe=async(query,args)=>{calls.push({query,args});if(state.failed)throw Error('timeout');return[{coverage:{as_of:args[0],rows:[],rolling_truncated:false,prospective_truncated:false,compiler_versions:[]}}]};
 const sql=async()=>[{dashboard_key_sha256:createHash('sha256').update('test-only-key').digest('hex'),cron_key_sha256:null}];
 sql.begin=async fn=>{calls.push('begin');return fn(tx)};sql.end=async()=>{calls.push('end')};
 vm.runInNewContext(compile('supabase/functions/brian-shadow-scorecard/index.ts'),{...coverage,...model,postgres:()=>sql,crypto:webcrypto,Date,Response,Request,TextEncoder,console,Deno:{env:{get:()=> 'test-only-db'},serve:fn=>{handler=fn}}});
 return{handler,calls,state,coverage};
}
const req=(key,body='{}')=>new Request('https://test.invalid/',{method:'POST',headers:key?{'x-brian-dashboard-key':key}:{},body});
test('missing and wrong dashboard keys never execute coverage',async()=>{
 const h=load();assert.equal((await h.handler(req())).status,401);assert.equal((await h.handler(req('wrong'))).status,401);assert.ok(!h.calls.includes('begin'));
});
test('actual handler uses read-only snapshot clock and frozen window, ignoring client dates',async()=>{
 const h=load();const response=await h.handler(req('test-only-key','{"starts_at":"1900-01-01"}'));assert.equal(response.status,200);
 const d=await response.json();assert.equal(d.outcome_coverage.status,'AVAILABLE');assert.equal(d.outcome_coverage.automatic_promotion,false);assert.equal(d.outcome_coverage.protocol.id,h.coverage.COHORT.id);
 const call=h.calls.find(x=>x?.query);assert.equal(call.query,h.coverage.OUTCOME_COVERAGE_SQL);assert.deepEqual([...call.args],['2026-10-19T12:00:00.000Z',h.coverage.COHORT.starts_at,h.coverage.COHORT.ends_at]);
 assert.ok(h.calls.some(x=>typeof x==='string'&&x.includes('repeatable read, read only')));assert.ok(h.calls.includes('savepoint'));assert.equal(d.shadow_only,true);assert.equal(d.live_execution,false);
});
test('coverage failure is explicit and preserves existing reports',async()=>{
 const h=load();h.state.failed=true;const response=await h.handler(req('test-only-key'));assert.equal(response.status,200);const d=await response.json();assert.equal(d.outcome_coverage.status,'UNAVAILABLE');assert.equal(d.outcome_coverage.rows,null);assert.ok(Array.isArray(d.reports));assert.ok(Array.isArray(d.net_scoreboard));
});
test('cached payload still requires authentication for every request',async()=>{
 const h=load();assert.equal((await h.handler(req('test-only-key'))).status,200);assert.equal((await h.handler(req('wrong'))).status,401);assert.equal((await h.handler(req('test-only-key'))).status,200);assert.equal(h.calls.filter(x=>x==='begin').length,1);
});
