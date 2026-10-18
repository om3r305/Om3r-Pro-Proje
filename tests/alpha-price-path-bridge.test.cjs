const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {stripTypeScriptTypes}=require('node:module');
function handler({pathError=null}={}){
 let serve,calls=[];
 const db={from(table){calls.push(table);const q={};for(const method of ['select','gte','gt','eq','order','limit'])q[method]=()=>q;
 q.then=(ok,fail)=>Promise.resolve(table==='brian_alpha_decisions'?{data:[{decision_id:'d'}],error:null}:{data:[{event_id:'e',asset_id:'crypto:BTCUSDT',observed_mid_price:100}],error:pathError}).then(ok,fail);return q;}};
 let source=stripTypeScriptTypes(fs.readFileSync('supabase/functions/brian-realtime-alpha-export/index.ts','utf8'))
  .replace(/^import .*;\n/gm,'');
 vm.runInNewContext(source,{Deno:{env:{get:()=> 'test'},serve:fn=>{serve=fn}},createClient:()=>db,
 requireBridgeAuth:async req=>{if(req.headers.get('x-brian-internal-key')!=='valid')throw Error('unauthorized')},Response,Date});
 return {run:body=>serve(new Request('https://example.test',{method:'POST',headers:{'x-brian-internal-key':'valid'},body:JSON.stringify(body)})),raw:()=>serve(new Request('https://example.test',{method:'POST',body:'{}'})),calls};
}
test('legacy bridge callers keep decisions while independent price export is explicit',async()=>{
 const h=handler();const data=await(await h.run({})).json();assert.equal(data.count,1);assert.equal(data.price_path_count,0);assert.deepEqual(h.calls,['brian_alpha_decisions']);
});
test('authenticated price bridge carries real event identifiers and fails closed on path query errors',async()=>{
 const h=handler();const data=await(await h.run({include_price_path:true})).json();assert.equal(data.price_path_count,1);assert.equal(data.price_path[0].event_id,'e');
 const failed=await handler({pathError:{message:'unavailable'}}).run({include_price_path:true});assert.equal(failed.status,500);assert.equal((await failed.json()).status,'FAILED_CLOSED');
});
test('unauthorized request cannot query either private table',async()=>{const h=handler();assert.equal((await h.raw()).status,401);assert.equal(h.calls.length,0);});
test('independent prices resolve the exact horizon without relaxing terminal tolerance or guessing cost',()=>{
 const source=stripTypeScriptTypes(fs.readFileSync('supabase/functions/_shared/alpha_audit.ts','utf8')).replace(/export /g,'');
 const resolve=vm.runInNewContext(source+';resolveAlphaAuditHorizon');
 const d={observedAt:'2026-10-18T22:00:00Z',action:'WAIT',direction:0,referencePrice:100,estimatedRoundTripCostBps:20};
 const points=[{observed_at:d.observedAt,observed_mid_price:100,estimated_round_trip_cost_bps:null},
 {observed_at:'2026-10-18T22:04:00Z',observed_mid_price:101,estimated_round_trip_cost_bps:null},
 {observed_at:'2026-10-18T22:05:30Z',observed_mid_price:101,estimated_round_trip_cost_bps:null}];
 assert.equal(resolve(d,300,points).classification,'WAIT_MISSED_LONG');
 assert.equal(resolve({...d,estimatedRoundTripCostBps:null},300,points),null);
 assert.equal(resolve(d,300,[points[0],{...points[2],observed_at:'2026-10-18T22:07:01Z'}]),null);
});
