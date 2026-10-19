'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const {stripTypeScriptTypes}=require('node:module');
const {webcrypto,createHash}=require('node:crypto');
const vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../supabase/functions/_shared/cron_auth.ts'),'utf8');
const code=stripTypeScriptTypes(source,{mode:'strip'}).replace(/^export /gm,'')+'\nglobalThis.check=requireCronAuth;';
const sandbox={crypto:webcrypto,TextEncoder,AbortSignal,setTimeout};vm.runInNewContext(code,sandbox);
const digest=value=>createHash('sha256').update(value).digest('hex');
function client(result,abortable=true){
 const calls=[]; const query={then(resolve,reject){return Promise.resolve(result).then(resolve,reject)}};
 if(abortable)query.abortSignal=signal=>{calls.push('abort');assert.ok(signal instanceof AbortSignal);return Promise.resolve(result)};
 return {calls,from(name){calls.push(name);return{select(fields){assert.equal(fields,'cron_key_sha256');return{eq(field,value){assert.equal(field,'auth_id');assert.equal(value,'custom');return{single(){return query}}}}}}}};
}
const request=value=>new Request('https://example.invalid',{headers:value?{'x-brian-cron-key':value}:{}});
test('missing cron credential fails before database access',async()=>{
 const db=client(null);await assert.rejects(sandbox.check(request(''),db,'custom'),/UNAUTHORIZED_CRON/);assert.deepEqual(db.calls,[]);
});
test('valid custom key supports both abortable query and legacy thenable',async()=>{
 for(const abortable of [true,false]){
  const db=client({data:{cron_key_sha256:digest('test-only-value')},error:null},abortable);
  await sandbox.check(request('test-only-value'),db,'custom');
  assert.equal(db.calls.includes('abort'),abortable);
 }
});
test('wrong custom key is rejected without changing lookup authorization',async()=>{
 const db=client({data:{cron_key_sha256:digest('different-test-value')},error:null});
 await assert.rejects(sandbox.check(request('test-only-value'),db,'custom'),/UNAUTHORIZED_CRON/);
});
test('missing custom auth row and transient custom failure remain closed',async()=>{
 for(const result of [{data:null,error:null},{data:null,error:{message:'schema cache timeout'}}]){
  await assert.rejects(sandbox.check(request('test-only-value'),client(result),'custom'),/CRON_AUTH_UNAVAILABLE/);
 }
});
