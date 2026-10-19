const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const vm=require('node:vm');const {stripTypeScriptTypes}=require('node:module');const {webcrypto}=require('node:crypto');
const compile=p=>stripTypeScriptTypes(fs.readFileSync(p,'utf8')).replace(/^import[^\n]+\n/gm,'').replace(/export /g,'');
function load(){
 const state={enabled:[true,true],assets:['crypto:BTCUSDT','crypto:ROTATEDUSDT'],fetches:0,rows:[],runs:[],last:null,contended:false,providerError:null,dbStopped:false};let handler;
 const logic=vm.runInNewContext(compile('supabase/functions/brian-outcome-price-eye/logic.ts')+';({parseOutcomeQuotes,BOOK_URI,PRICE_CAPTURE_VERSION})');
 const db={from(table){let insert=null;const q={select(){return q},eq(){return q},in(){return q},order(){return q},limit(){return q},maybeSingle(){return q},single(){return q},insert(v){insert=v;return q},async abortSignal(){if(table==='brian_evolution_runtime_config')return{data:{config_value:String(state.enabled.length>1?state.enabled.shift():state.enabled[0])},error:null};if(insert){state.runs.push(insert);return{error:null}}return{data:state.last,error:null}}};return q},rpc(name,args){return{async abortSignal(){if(name==='brian_outcome_watch_assets_v1')return{data:state.assets.map(asset_id=>({asset_id})),error:null};state.rows=args.p_rows;return{data:state.dbStopped?{status:'OPERATOR_STOPPED',stored:0}:{status:'SUCCESS',received:state.rows.length,stored:state.rows.length},error:null}}}}};
 const readBookSnapshot=async()=>{state.fetches++;if(state.providerError)throw state.providerError;return{payload:[{symbol:'BTCUSDT',bidPrice:'100',askPrice:'102',bidQty:'1',askQty:'1'}],receivedAt:new Date().toISOString(),requestStartedAt:new Date(Date.now()-10).toISOString(),responseHash:'a'.repeat(64)}};
 vm.runInNewContext(compile('supabase/functions/brian-outcome-price-eye/index.ts'),{...logic,readBookSnapshot,createClient:()=>db,withCollectorLease:async(db,id,ttl,work)=>state.contended?{contended:true}:{contended:false,value:await work()},requireRealtimeInternal:async req=>{if(req.headers.get('x-brian-internal-key')!=='test-only')throw Error('UNAUTHORIZED')},Date,Request,Response,AbortSignal,TextEncoder,crypto:webcrypto,console,Deno:{env:{get:()=> 'test-only'},serve:fn=>handler=fn}});
 return{state,handler};
}
const req=key=>new Request('https://test.invalid/',{method:'POST',headers:key?{'x-brian-internal-key':key}:{},body:'{}'});
test('unauthorized requests and STOP do no provider or price work',async()=>{
 const h=load();assert.equal((await h.handler(req())).status,401);h.state.enabled=[false];assert.equal((await(await h.handler(req('test-only'))).json()).status,'OPERATOR_STOPPED');assert.equal(h.state.fetches,0);assert.equal(h.state.rows.length,0);assert.equal(h.state.runs.length,0);
});
test('one bulk snapshot persists available rotated assets honestly, unknown cost is not supplied',async()=>{
 const h=load();const d=await(await h.handler(req('test-only'))).json();assert.equal(d.status,'DEGRADED');assert.equal(d.requested_assets,2);assert.equal(d.stored,1);assert.equal(h.state.fetches,1);assert.equal(h.state.rows[0].source_metadata.exchange_timestamp,null);assert.equal(h.state.rows[0].source_metadata.quote_time_semantics,'HTTP_RECEIPT_TIME_EXCHANGE_TIME_UNKNOWN');assert.ok(!Object.hasOwn(h.state.rows[0],'estimated_round_trip_cost_bps'));assert.equal(h.state.runs[0].shadow_only,true);assert.equal(h.state.runs[0].live_execution,false);
});
test('STOP after network receipt prevents persistence and operational success receipt',async()=>{
 const h=load();h.state.enabled=[true,false];const d=await(await h.handler(req('test-only'))).json();assert.equal(d.status,'OPERATOR_STOPPED');assert.equal(h.state.fetches,1);assert.equal(h.state.rows.length,0);assert.equal(h.state.runs.length,0);
});
test('database STOP is not rewritten as a successful collection',async()=>{
 const h=load();h.state.dbStopped=true;const d=await(await h.handler(req('test-only'))).json();assert.equal(d.status,'OPERATOR_STOPPED');assert.equal(h.state.runs.length,0);
});
test('lease contention, cadence guard and provider cooldown do no fetch',async()=>{
 const h=load();h.state.contended=true;assert.equal((await(await h.handler(req('test-only'))).json()).status,'SKIPPED_LEASE_CONTENDED');h.state.contended=false;h.state.last={started_at:new Date().toISOString(),metadata:{}};assert.equal((await(await h.handler(req('test-only'))).json()).status,'SKIPPED_INTERVAL');h.state.last={metadata:{next_allowed_at:new Date(Date.now()+300000).toISOString()}};assert.equal((await(await h.handler(req('test-only'))).json()).status,'SKIPPED_PROVIDER_COOLDOWN');assert.equal(h.state.fetches,0);
});
test('429 creates bounded delayed retry evidence, no synthetic prices',async()=>{
 const h=load();h.state.providerError=Object.assign(Error('BOOK_HTTP_429'),{retryAfterSeconds:300});assert.equal((await h.handler(req('test-only'))).status,503);assert.equal(h.state.rows.length,0);assert.equal(h.state.runs[0].status,'FAILED');assert.ok(Date.parse(h.state.runs[0].metadata.next_allowed_at)>Date.now()+290000);
});
test('empty watch lists skip network and capacity overflow fails visibly',async()=>{
 const h=load();h.state.assets=[];assert.equal((await(await h.handler(req('test-only'))).json()).status,'NO_PENDING_ASSETS');h.state.assets=Array.from({length:501},(_,i)=>`crypto:S${i}USDT`);const d=await(await h.handler(req('test-only'))).json();assert.equal(d.error,'WATCH_CAPACITY_EXCEEDED');assert.equal(h.state.fetches,0);
});
test('actual orchestrator starts price observation before a stalled universe lane completes',async()=>{
 let handler,release;const calls=[];const stalled=new Promise(resolve=>release=resolve);class Clock extends Date{static now(){return Date.parse('2026-10-19T13:30:00Z')}}
 vm.runInNewContext(compile('supabase/functions/brian-realtime-orchestrator/index.ts'),{Date:Clock,Request,Response,AbortSignal,readOperatorState:async()=>({enabled:true}),requireRealtimeInternal:async()=> 'test-only',Deno:{serve:fn=>handler=fn},fetch:async url=>{calls.push(url);if(url.endsWith('brian-universe-collector')){await stalled;return new Response('{"status":"FAILED"}')}return new Response('{"status":"SUCCESS"}')}});
 const pending=handler(req('test-only'));await new Promise(resolve=>setImmediate(resolve));assert.ok(calls.some(x=>x.endsWith('brian-outcome-price-eye')));release();const d=await(await pending).json();assert.ok(d.results.some(x=>x.name==='outcome_price'&&x.ok));assert.ok(d.results.some(x=>x.name==='universe'&&!x.ok));
});
