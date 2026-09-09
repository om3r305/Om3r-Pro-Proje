import assert from 'node:assert/strict';
import { executionEconomics, refreshEntryEvaluation } from './execution.ts';
import { closePosition, initialRuntime, type Position } from '../_shared/dip_v8_dual.ts';
import type { CandidateResult } from './decision.ts';

Deno.test('economic reward and loss match actual shadow close for LONG and SHORT',()=>{
  for(const direction of ['UP','DOWN'] as const){
    const entry=100.2,target=direction==='UP'?105:95,stop=direction==='UP'?98:102;
    const e=executionEconomics(direction,entry,target,stop,10,1,2,0);
    for(const win of [true,false]){
      const rt=initialRuntime(1000);
      rt.pos={side:direction==='UP'?'LONG':'SHORT',entry,qty:1,margin:entry,fees_open:entry*.001,fee_bps:10,slippage_bps:1,spread_bps:2} as Position;
      rt.cash-=entry+entry*.001;
      const event=closePosition(rt,{reason:win?'TARGET_FIRST':'INVALIDATION_FIRST',hit:win,price:win?target:stop,eventAt:1000,eventRangeStart:1000,checkedUntil:1000},'x',0,1001)!;
      const expected=(win?e.net_reward_bps:-e.net_risk_bps)*entry/10000;
      assert.ok(Math.abs(Number(event.realized_pnl)-expected)<1e-10);
    }
  }
});
Deno.test('opening slippage is not charged a second time by economic model',()=>{
  const e=executionEconomics('UP',100,101,99,0,10,0,0);
  assert.ok(Math.abs(e.net_reward_bps-89.9)<1e-8);
});
Deno.test('funding reserve reduces reward and increases risk without inventing a credit',()=>{
  const a=executionEconomics('DOWN',100,99,101,10,1,1,0),b=executionEconomics('DOWN',100,99,101,10,1,1,2);
  assert.ok(Math.abs(a.net_reward_bps-b.net_reward_bps-2)<1e-9);
  assert.ok(Math.abs(b.net_risk_bps-a.net_risk_bps-2)<1e-9);
});
Deno.test('eligible forecast gets immutable re-evaluation identity, preserving parent',async()=>{
  const original={canEnter:true,occurrence:'forecast',episode:'episode',thesis:{thesis_id:'forecast'},decision:{occurrence_id:'forecast',episode_id:'episode',evidence:{veto:[]}}} as unknown as CandidateResult;
  const a=structuredClone(original),b=structuredClone(original);
  const d=await refreshEntryEvaluation(a,7,1000);
  assert.notEqual(d?.occurrence_id,'forecast');assert.equal(d?.episode_id,'episode');
  assert.equal(d?.evidence.parent_forecast_id,'forecast');assert.equal(a.thesis.thesis_id,d?.occurrence_id);
  assert.equal((await refreshEntryEvaluation(b,7,1000))?.occurrence_id,d?.occurrence_id);
  assert.equal(original.decision?.occurrence_id,'forecast');
});
Deno.test('vetoed forecast is not promoted',async()=>{
  assert.equal(await refreshEntryEvaluation({canEnter:false} as CandidateResult,7,1000),null);
});
