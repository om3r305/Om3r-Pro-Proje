const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const vm=require('node:vm');const {stripTypeScriptTypes}=require('node:module');
const source=stripTypeScriptTypes(fs.readFileSync('supabase/functions/_shared/operator_state.ts','utf8')).replace(/export /g,'');
const {readOperatorState}=vm.runInNewContext(source+';({readOperatorState})',{AbortSignal,fetch});
test('authoritative enable is required and uses the authenticated control route',async()=>{
 let called=false;const r=await readOperatorState('test',async(url,init)=>{called=true;assert.match(url,/brian-core-scheduler-bridge$/);assert.equal(JSON.parse(init.body).action,'operator_status');assert.equal(init.headers['x-brian-internal-key'],'test');return {ok:true,json:async()=>({status:'SUCCESS',enabled:true})}});assert.ok(called);assert.equal(r.enabled,true);
});
test('operator STOP never falls back to a cached enabled state',async()=>{
 const r=await readOperatorState('test',async()=>({ok:true,json:async()=>({status:'SUCCESS',enabled:false})}));assert.equal(r.enabled,false);assert.equal(r.status,'OPERATOR_STOPPED');
});
test('failed, malformed and unreachable control all fail closed',async()=>{
 for(const f of [async()=>({ok:false,json:async()=>({enabled:true})}),async()=>({ok:true,json:async()=>({status:'SUCCESS',enabled:'true'})}),async()=>{throw Error('network')}]){
 const r=await readOperatorState('test',f);assert.equal(r.enabled,false);assert.equal(r.status,'CONTROL_UNAVAILABLE');
 }
});
