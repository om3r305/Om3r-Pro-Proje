import assert from "node:assert/strict";
import { calibrate, chooseShadowLeverage, closePosition, evaluatePath, initialRuntime, sizePosition, validateSession, ENGINE_VERSION, POLICY_VERSION, type Position, type Struct } from "../_shared/dip_v8_dual.ts";
import { selectEconomicTarget } from "./decision.ts";
import { episodeEntered, readRuntime } from "./worker.ts";
import { positionFunding } from "./market.ts";
const rules={minNotional:20,minQty:.001,maxQty:100,stepSize:.001};
function structure(levels:number[],kind:"H"|"L"):Struct{return{tf:"1m",lastClose:2500,atr:10,pivots:levels.map((p,i)=>({i,t:i,p,kind})),lastHigh:null,lastLow:null,trend:"RANGE",bos:null,choch:null,sweep:null,failedBreak:null,equalHigh:null,equalLow:null,fingerprint:"test"};}
Deno.test("nearest obstacle cannot be skipped to manufacture economic RR in either direction",()=>{
 assert.equal(selectEconomicTarget("UP",2500,22,[structure([2503,2520],"H")]),2503);
 assert.equal(selectEconomicTarget("DOWN",2500,22,[structure([2497,2480],"L")]),2497);
});
Deno.test("cold size never exceeds 8 percent even at maximum raw score",()=>{
 for(const raw of [.6,.7,.8,1])for(const direction of ["UP","DOWN"] as const){
  const size=sizePosition(500,500,direction,2500,direction==="UP"?2499:2501,calibrate([]),10,1,.1,rules,1,raw);
  assert.ok(size);assert.ok(size.notional<=40);assert.ok(size.worst_loss<=2.5);
 }
});
Deno.test("exchange minimum cannot override exposure cap and cold leverage cannot become 2x",()=>{
 assert.equal(sizePosition(50,50,"UP",2500,2499,calibrate([]),10,1,.1,rules,1,.9),null);
 assert.equal(sizePosition(500,500,"UP",2500,2499,calibrate([]),10,1,.1,rules,2,.9),null);
});
Deno.test("calibrated sizing retains a 20 percent gross cap including leverage",()=>{
 const cal=calibrate(Array.from({length:60},(_,i)=>({hit:i<50})));
 const size=sizePosition(500,500,"UP",2500,2499,cal,10,1,.1,rules,2,.9);
 assert.ok(size);assert.ok(size.notional<=100);assert.ok(size.margin<=50);
 assert.equal(chooseShadowLeverage({maxAllowed:2,cal,raw:.8,economicRR:3,targetBps:100,costBps:20,flowScore:.5}),2);
 assert.equal(chooseShadowLeverage({maxAllowed:2,cal:calibrate([]),raw:1,economicRR:3,targetBps:100,costBps:20,flowScore:.5}),1);
});
Deno.test("episode check reads the ledger even if lastLock was overwritten by opposite trade",async()=>{
 const calls:unknown[]=[];const q={select:()=>q,eq:(k:string,v:string)=>{calls.push([k,v]);return q;},in:()=>q,limit:async()=>({data:[{transition_id:"earlier-long"}],error:null})};
 assert.equal(await episodeEntered({from:()=>q} as any,"old-long-episode"),true);
 assert.deepEqual(calls,[["episode_id","old-long-episode"]]);
});
Deno.test("ledger and runtime read errors fail closed",async()=>{
 const q={select:()=>q,eq:()=>q,in:()=>q,limit:async()=>({data:null,error:{message:"offline"}}),maybeSingle:async()=>({data:null,error:{message:"offline"}})};
 await assert.rejects(()=>episodeEntered({from:()=>q} as any,"x"),/EPISODE_READ_FAILED/);
 await assert.rejects(()=>readRuntime({from:()=>q} as any,"x"),/READ_FAILED/);
});
function pos(side:"LONG"|"SHORT"):Position{return{side,position_id:"p",thesis_id:"t",episode_id:"e",setup:"TEST",regime:"RANGE",entry:100,qty:1,notional:100,margin:100,leverage:1,target:side==="LONG"?110:90,stop:side==="LONG"?95:105,opened_at:new Date(0).toISOString(),due_at:new Date(5400000).toISOString(),fees_open:.1,fee_bps:10,slippage_bps:0,spread_bps:0,venue:"SHADOW_PERP",policy_version:POLICY_VERSION,checked_until:0,market_price:100,actual_fraction:.1,gross_fraction:.1};}
Deno.test("settled funding is signed, based on settlement mark and requested over the held interval",async()=>{
 for(const side of ["LONG","SHORT"] as const){
  const fetcher=(async(input:RequestInfo|URL,init?:RequestInit)=>{const url=new URL(String(input));assert.equal(init?.method,"GET");assert.equal(url.pathname,"/fapi/v1/fundingRate");assert.equal(url.searchParams.get("startTime"),"1");assert.equal(url.searchParams.get("endTime"),"2000");return Response.json([{symbol:"ETHUSDT",fundingTime:1000,fundingRate:"0.001",markPrice:"102"}]);}) as typeof fetch;
  const f=await positionFunding(pos(side),2000,fetcher);assert.ok(Math.abs(f.cashflow-(side==="LONG"?-.102:.102))<1e-12);
 }
});
Deno.test("funding evidence outside holding window is rejected",async()=>{
 const fetcher=(async()=>Response.json([{symbol:"ETHUSDT",fundingTime:3000,fundingRate:"0.001",markPrice:"102"}])) as typeof fetch;
 await assert.rejects(()=>positionFunding(pos("LONG"),2000,fetcher),/INVALID_FUNDING_HISTORY/);
});
Deno.test("LONG and SHORT close balance includes funding once and matches realized PnL",()=>{
 for(const side of ["LONG","SHORT"] as const){const rt=initialRuntime(1000);rt.pos=pos(side);rt.pos.funding_accrued=side==="LONG"?-.102:.102;rt.cash=899.9;
  const price=side==="LONG"?110:90;
  const e=closePosition(rt,{reason:"TARGET_FIRST",hit:true,price,eventAt:2000,eventRangeStart:2000,checkedUntil:2000},"fp",0,2001)!;
  assert.ok(Math.abs(rt.cash-1000-rt.realized)<1e-10);assert.equal(e.funding_cashflow,side==="LONG"?-.102:.102);assert.equal(rt.trades,1);assert.equal(closePosition(rt,{reason:"TARGET_FIRST",hit:true,price,eventAt:2000,eventRangeStart:2000,checkedUntil:2000},"fp",0,2001),null);
 }
});
Deno.test("both barriers in one sealed candle remain ambiguous in shared evaluator",()=>{
 const r=evaluatePath({direction:"UP",target:110,stop:95,start:0,due:60000,now:61000,end:60000,entry:100,segments:[{kind:"BAR",start:0,end:60000,o:100,h:111,l:94,c:101}]});assert.equal(r.reason,"AMBIGUOUS");assert.equal(r.hit,null);
});
Deno.test("dual worker rejects live execution and non ETH universe",()=>{
 const cfg={engine_version:ENGINE_VERSION,policy_version:POLICY_VERSION,symbols:["ETHUSDT"],shadow_only:true,live_execution:false,browser_execution:false,server_authoritative:true,allow_shadow_short:true,max_shadow_leverage:2,execution_mode:"SHADOW_PAPER"};
 validateSession(cfg);assert.throws(()=>validateSession({...cfg,live_execution:true}),/SHADOW/);assert.throws(()=>validateSession({...cfg,symbols:["ETHUSDT","XRPUSDT"]}),/RESTART/);
});

import { entryGuard } from "./entry_guard.ts";
import type { Bar } from "../_shared/dip_v8_dual.ts";
function bars(values:number[]):Bar[]{return values.map((c,i)=>({t:(i+10)*60000,ct:(i+11)*60000-1,o:c,h:c+.1,l:c-.1,c,v:1}));}
Deno.test("real 2478 reversal: friction cannot clear 2478.30 resistance and jump target",()=>{
 const s=structure([2478.12,2478.30,2496.31],"H");
 assert.equal(selectEconomicTarget("UP",2478.14,22.6,[s]),2478.30);
 const rows=bars([2471.16,2473,2474,2475,2476,2476.5,2477,2478.14]);
 const g=entryGuard("UP","EARLY_REVERSAL",2478.14,2478.387814,rows,[s],1.177,2e6);
 assert.ok(g.veto.includes("WAIT_RETEST"));assert.ok(g.veto.includes("ENTRY_TOO_LATE"));
});
Deno.test("a tick over resistance cannot unlock distant target; mirrored SHORT also waits",()=>{
 for(const sign of [1,-1]){const cv=(v:number)=>sign===1?v:200-v;
 const s=structure([cv(100),cv(110)],sign===1?"H":"L");
 const rows=bars([99,99.1,99.2,99.3,99.4,99.5,99.6,99.7].map(cv));
 assert.ok(entryGuard(sign===1?"UP":"DOWN","BOS_RETEST",cv(100.1),cv(100.12),rows,[s],1,2e6).veto.includes("WAIT_RETEST"));}
});
Deno.test("two sealed closes followed by a holding retest can clear a crossed obstacle",()=>{
 const s=structure([100,110],"H");const rows=bars([99,99,99,99,99,100.4,100.5,100.3]);rows[7].l=100.05;
 assert.deepEqual(entryGuard("UP","BOS_RETEST",100.4,100.42,rows,[s],1,2e6).veto,[]);
 rows[7].ct=3e6;
 assert.ok(entryGuard("UP","BOS_RETEST",100.4,100.42,rows,[s],1,2e6).veto.includes("ENTRY_DATA_INCOMPLETE"));
});
