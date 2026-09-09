import assert from 'node:assert/strict';
import { observeTarget, withLevelObservations } from './level_observation.ts';
import type { Bar } from '../_shared/dip_v8_dual.ts';
function fixture(mode='none',down=false):Bar[]{
 const rows=Array.from({length:16},(_,i)=>({t:i*60000,ct:(i+1)*60000-1,o:100,h:101,l:99,c:100,v:10}));
 rows[4].h=104;
 if(mode==='touch')rows[10].h=104;
 if(mode==='sweep')rows[10]={...rows[10],h:106,c:103};
 if(mode==='cross')rows[10]={...rows[10],h:106,c:105};
 return down?rows.map(b=>({...b,o:200-b.o,h:200-b.l,l:200-b.h,c:200-b.c})):rows;
}
for(const down of [false,true])for(const [mode,status] of [['none','UNTOUCHED'],['touch','TOUCHED'],['sweep','SWEPT'],['cross','CLOSE_CROSSED']]){
 Deno.test(`${down?'low':'high'} ${status} is diagnostic closed-bar evidence`,()=>{
  const rows=fixture(mode,down),result=observeTarget(rows,'1m',down?96:104,960000).find(x=>x.origin==='PIVOT'&&x.pivot_at[0]===240000)!;
  assert.equal(result.status,status);assert.equal(result.confirmed_at,480000);
  assert.equal(result.first_touch_bar,mode==='none'?null:600000);
  assert.equal(result.first_close_cross_bar,mode==='cross'?600000:null);
 });
}
Deno.test('equal pair only exists after second pivot right confirmation; no prior touch attributed',()=>{
 const rows=fixture();rows[10].h=104.01;
 assert.equal(observeTarget(rows.slice(0,13),'1m',104.005,780000).length,0);
 const equal=observeTarget(rows.slice(0,14),'1m',104.005,840000).find(x=>x.origin==='EQUAL_PAIR')!;
 assert.equal(equal.confirmed_at,840000);assert.equal(equal.status,'UNTOUCHED');assert.deepEqual(equal.pivot_at,[240000,600000]);
 rows[14]={...rows[14],h:106,c:103};
 const swept=observeTarget(rows,'1m',104.005,960000).find(x=>x.origin==='EQUAL_PAIR')!;
 assert.equal(swept.status,'SWEPT');assert.equal(swept.first_sweep_bar,840000);
});
Deno.test('open and future candles cannot change the observation',()=>{
 const a=fixture('none'),b=fixture('cross');
 assert.deepEqual(observeTarget(a,'1m',104,600000),observeTarget(b,'1m',104,600000));
});
function candidate(){return{thesis:{veto:['CALIBRATING'],thesis_id:'unchanged'},decision:{evidence:{veto:['CALIBRATING']},occurrence_id:'unchanged'},target:104,entry:102,inv:100,size:{qty:1},episode:'same',canEnter:true};}
Deno.test('adding diagnostics preserves entire decision contract and never mutates inputs',()=>{
 const c=candidate(),before=structuredClone(c),rows=fixture('sweep'),bars={'1m':rows},beforeBars=structuredClone(bars);
 const out=withLevelObservations(c,bars,960000);
 assert.notEqual(out,c);assert.deepEqual(c,before);assert.deepEqual(bars,beforeBars);
 assert.ok((out.thesis as Record<string,unknown>).level_observation);assert.ok((out.decision!.evidence as Record<string,unknown>).level_observation);
 const plain=structuredClone(out);delete (plain.thesis as Record<string,unknown>).level_observation;delete (plain.decision!.evidence as Record<string,unknown>).level_observation;
 assert.deepEqual(plain,before);
});
Deno.test('malformed optional data and oversized diagnostics leave decision unchanged',()=>{
 const c=candidate(),rows=fixture();rows[9].t++;
 assert.equal(withLevelObservations(c,{'1m':rows},960000),c);
 const big={...c,decision:{evidence:{padding:'x'.repeat(11000)}}};
 assert.equal(withLevelObservations(big,{'1m':fixture()},960000),big);
});
