const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');const vm=require('node:vm');const {stripTypeScriptTypes}=require('node:module');const {webcrypto}=require('node:crypto');
const source=fs.readFileSync('supabase/functions/brian-outcome-price-eye/logic.ts','utf8');
const logic=vm.runInNewContext(stripTypeScriptTypes(source).replace(/export /g,'')+';({parseOutcomeQuotes,readBookSnapshot,BOOK_URI})',{Date,Response,Request,AbortSignal,TextDecoder,TextEncoder,Uint8Array,crypto:webcrypto,fetch});
const book=(symbol='BTCUSDT',extra={})=>({symbol,bidPrice:'100.01',askPrice:'100.03',bidQty:'2',askQty:'3',...extra});
test('requested assets are observed even when absent from radar or an action signal',()=>{
 const r=logic.parseOutcomeQuotes([book(),book('ROTATEDUSDT')],['crypto:ROTATEDUSDT']);assert.equal(r.quotes.length,1);assert.equal(r.quotes[0].asset_id,'crypto:ROTATEDUSDT');assert.equal(r.missing.length,0);
});
test('raw decimal prices are preserved; unknown or invalid quotes do not become zero',()=>{
 const r=logic.parseOutcomeQuotes([book('AUSDT',{bidPrice:'.1',askPrice:'.2'}),book('BUSDT',{bidPrice:null}),book('CUSDT',{askPrice:'NaN'}),book('DUSDT',{bidQty:0}),book('EUSDT',{askPrice:'99'})],['crypto:AUSDT','crypto:BUSDT','crypto:CUSDT','crypto:DUSDT','crypto:EUSDT']);
 assert.equal(r.quotes.length,1);assert.equal(r.quotes[0].bid_price,'.1');assert.equal(r.missing.length,4);
});
test('duplicate symbols are ambiguous and cannot be silently selected',()=>{
 const r=logic.parseOutcomeQuotes([book(),book(),book()],['crypto:BTCUSDT']);assert.equal(r.quotes.length,0);assert.equal(r.missing.length,1);
});
test('invalid watch lists and object-shaped provider payloads fail closed',()=>{
 assert.throws(()=>logic.parseOutcomeQuotes({},[]));assert.throws(()=>logic.parseOutcomeQuotes([],['crypto:BTCUSDT','crypto:BTCUSDT']));assert.throws(()=>logic.parseOutcomeQuotes([],['https://internal']));
});
test('transport receives one public snapshot, no credentials or redirects, and hashes actual bytes',async()=>{
 let calls=0;const r=await logic.readBookSnapshot(async(uri,init)=>{calls++;assert.equal(uri,logic.BOOK_URI);assert.equal(init.redirect,'error');assert.ok(!JSON.stringify(init.headers).includes('key'));return new Response(JSON.stringify([book()]));});
 assert.equal(calls,1);assert.equal(r.responseHash.length,64);assert.ok(Date.parse(r.receivedAt)>=Date.parse(r.requestStartedAt));
});
test('rate limits honor a bounded cooldown without retry loops',async()=>{
 let calls=0;await assert.rejects(()=>logic.readBookSnapshot(async()=>{calls++;return new Response('',{status:429,headers:{'retry-after':'300'}})}),e=>e.message==='BOOK_HTTP_429'&&e.retryAfterSeconds===300);assert.equal(calls,1);
});
test('oversized bodies and malformed JSON are never accepted',async()=>{
 await assert.rejects(()=>logic.readBookSnapshot(async()=>new Response('[]',{headers:{'content-length':'2000001'}})),/BOOK_TOO_LARGE/);
 await assert.rejects(()=>logic.readBookSnapshot(async()=>new Response('[')));
});
