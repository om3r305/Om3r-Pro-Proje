/** Characterization replay: observed behavior, NOT approval to discard old levels.
 * All bars are synthetic and sealed before each decision. No network, DB or orders.
 * Run: deno test --no-check tests/replay/dip_v83_level_lifecycle.test.ts
 */
import assert from 'node:assert/strict';
import { structure } from '../../supabase/functions/brian-dip-dual-shadow-worker/structure.ts';
import { selectEconomicTarget } from '../../supabase/functions/brian-dip-dual-shadow-worker/decision.ts';
import type { Bar } from '../../supabase/functions/_shared/dip_v8_dual.ts';

type History='UNTOUCHED'|'WICK_CROSSED'|'CLOSE_CROSSED';
function fixture(history:History,down=false):Bar[]{
  const rows=Array.from({length:16},(_,i)=>({t:i*60000,ct:(i+1)*60000-1,o:100,h:101,l:99,c:100,v:10}));
  rows[4].h=104; // Confirmed at the close of bar 7 (RIGHT=3).
  if(history!=='UNTOUCHED')rows[10]={...rows[10],h:106,c:history==='CLOSE_CROSSED'?105:103};
  for(let i=13;i<16;i++)rows[i]={...rows[i],o:102,h:103,l:101,c:102};
  return down?rows.map(b=>({...b,o:200-b.o,h:200-b.l,l:200-b.h,c:200-b.c})):rows;
}
for(const down of [false,true])for(const history of ['UNTOUCHED','WICK_CROSSED','CLOSE_CROSSED'] as const){
  Deno.test(`${down?'SHORT':'LONG'}: nearest old pivot still selected after ${history}`,async()=>{
    const bars=fixture(history,down),s=await structure('1m',bars),level=down?96:104,entry=down?98:102;
    const pivot=s.pivots.find(p=>p.t===bars[4].t&&p.kind===(down?'L':'H'));
    assert.ok(pivot);assert.equal(pivot.p,level);
    const later=bars.slice(8); // Only bars after pivot confirmation.
    const touched=later.some(b=>down?b.l<level:b.h>level);
    const closedThrough=later.some(b=>down?b.c<level:b.c>level);
    assert.equal(touched,history!=='UNTOUCHED');assert.equal(closedThrough,history==='CLOSE_CROSSED');
    assert.equal(selectEconomicTarget(down?'DOWN':'UP',entry,20,[s]),level);
    // None of these histories is encoded in the existing Pivot contract.
    assert.equal(Object.hasOwn(pivot,'status'),false);
  });
}
Deno.test('pivot is unavailable until three right-hand candles have closed',async()=>{
  const bars=fixture('WICK_CROSSED');
  assert.equal((await structure('1m',bars.slice(0,7))).pivots.some(p=>p.t===bars[4].t),false);
  assert.equal((await structure('1m',bars.slice(0,8))).pivots.some(p=>p.t===bars[4].t),true);
});
Deno.test('a replay decision only receives the prefix available at its decision time',async()=>{
  const untouched=fixture('UNTOUCHED'),broken=fixture('CLOSE_CROSSED');
  // The two worlds differ in the future; the input available at minute 10 is identical.
  const a=await structure('1m',untouched.slice(0,10)),b=await structure('1m',broken.slice(0,10));
  assert.deepEqual(a,b);
  assert.equal(selectEconomicTarget('UP',102,20,[a]),selectEconomicTarget('UP',102,20,[b]));
});
