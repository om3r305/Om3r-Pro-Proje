import assert from "node:assert/strict";
import { firstForwardLevel } from "./levels.ts";
import type { Struct } from "../_shared/dip_v84_contract.ts";

function s(tf:string,pivots:Struct["pivots"],atr=10):Struct{return{tf,lastClose:100,atr,pivots,lastHigh:pivots.filter(x=>x.kind==="H").at(-1)??null,lastLow:pivots.filter(x=>x.kind==="L").at(-1)??null,trend:"RANGE",bos:null,choch:null,sweep:null,failedBreak:null,equalHigh:null,equalLow:null,fingerprint:tf};}

Deno.test("L1 is nearest forward level from actual fill and cannot skip to farther economics",async()=>{
  const out=await firstForwardLevel({direction:"UP",fill:100,signalAt:10_000_000,tickSize:.01,structs:[s("1m",[{i:0,t:0,p:100.5,kind:"H"},{i:1,t:60_000,p:105,kind:"H"}])]});
  assert.equal(out.l1?.normalized_price,100.5);
  assert.equal(out.levels[1]?.normalized_price,105);
});

Deno.test("levels not confirmed by signal time cannot become L1",async()=>{
  const out=await firstForwardLevel({direction:"UP",fill:100,signalAt:200_000,tickSize:.01,structs:[s("1m",[{i:0,t:0,p:101,kind:"H"},{i:1,t:60_000,p:100.5,kind:"H"}])]});
  // 1m pivot confirmation is t + 4m. Neither candidate is known by 200s.
  assert.equal(out.l1,null);
});

Deno.test("same tick price deterministically prefers higher timeframe",async()=>{
  const out=await firstForwardLevel({direction:"DOWN",fill:100,signalAt:20_000_000,tickSize:.01,structs:[s("1m",[{i:0,t:0,p:99,kind:"L"}]),s("15m",[{i:0,t:0,p:99.004,kind:"L"}])]});
  assert.equal(out.l1?.normalized_price,99);
  assert.equal(out.l1?.timeframe,"15m");
});
