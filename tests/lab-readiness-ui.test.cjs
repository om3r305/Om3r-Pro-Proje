const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
function page(payload){
 let mounted,changed,calls=0,authenticated=false,opened=false;
 const events={};
 const room={classList:{contains:()=>opened}};
 const side={prepend:p=>{mounted=p}};
 const document={hidden:false,readyState:'complete',head:{appendChild(){}},createElement:()=>({}),querySelector:()=>side,getElementById:id=>id==='brianAnatomyLive'?room:id==='labReadinessLiveV1'?mounted:id==='unlockBtn'?{addEventListener:(_,fn)=>{events.unlock=fn}}:null};
 vm.runInNewContext(fs.readFileSync('monster-coins-pro/lab-readiness-v1.js','utf8'),{document,ROOT:'read-only',key:()=>authenticated?'test-key':'',post:async()=>{calls++;return payload},setTimeout:fn=>fn(),setInterval(){},MutationObserver:class{constructor(fn){changed=fn}observe(){}}});
 return {calls:()=>calls,html:()=>mounted?.innerHTML||'',async open(){opened=true;changed();await new Promise(resolve=>setImmediate(resolve))},async unlock(){authenticated=true;events.unlock();await new Promise(resolve=>setImmediate(resolve))}};
}
test('lab makes no unauthorized startup request and refreshes when a programmatic room opens after unlock',async()=>{
 const p=page({observed_at:new Date().toISOString(),shadow_only:true,live_execution:false,metrics:{pipeline_health_pct:null,evidence_maturity_pct:23,promotion_readiness_pct:0},candidates:[]});
 assert.equal(p.calls(),0);await p.open();assert.equal(p.calls(),0);assert.match(p.html(),/kilidini aç/);
 await p.unlock();assert.equal(p.calls(),1);assert.match(p.html(),/GÜNCEL RAPOR/);assert.match(p.html(),/>—<\/strong>/);assert.match(p.html(),/0\.0%/);
});
test('server stale flag prevents an old lab response from being labelled live',async()=>{
 const p=page({observed_at:new Date().toISOString(),stale:true,shadow_only:true,live_execution:false,metrics:{},candidates:[]});await p.open();await p.unlock();
 assert.match(p.html(),/SON KAYIT · GÜNCEL DEĞİL/);assert.doesNotMatch(p.html(),/>GÜNCEL RAPOR</);
});
