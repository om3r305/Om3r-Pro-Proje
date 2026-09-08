import assert from "node:assert/strict";
import type { Cal, Struct } from "../_shared/dip_v8_dual.ts";
import { chooseShadowLeverage } from "../_shared/dip_v8_dual.ts";
import { microReclaimReady, microRejectReady, selectEconomicTarget } from "./decision.ts";
function s(tf:string,trend:Struct["trend"],high:number,low:number,highLabel="HH",lowLabel="HL",pivots:Struct["pivots"]=[]):Struct{return{tf,lastClose:(high+low)/2,atr:1.7,pivots,lastHigh:{i:1,t:1,p:high,kind:"H",label:highLabel},lastLow:{i:2,t:2,p:low,kind:"L",label:lowLabel},trend,bos:null,choch:null,sweep:null,failedBreak:null,equalHigh:null,equalLow:null,fingerprint:tf};}
Deno.test("micro reclaim detects HH/HL long continuation",()=>{assert.equal(microReclaimReady(s("1m","UP",2492.85,2488),s("5m","UP",2501.74,2486.98),.84,2490.82),true);});
Deno.test("micro rejection mirrors LH/LL short continuation",()=>{assert.equal(microRejectReady(s("1m","DOWN",2502,2495,"LH","LL"),s("5m","DOWN",2505,2489,"LH","LL"),-.81,2498),true);});
Deno.test("mirror signal stays closed when flow disagrees",()=>{assert.equal(microRejectReady(s("1m","DOWN",2502,2495,"LH","LL"),s("5m","DOWN",2505,2489,"LH","LL"),.20,2498),false);});
Deno.test("down target ladder selects an economic lower pivot",()=>{const a=s("1m","DOWN",2502,2495,"LH","LL",[{i:1,t:1,p:2497,kind:"L",label:"LL"},{i:2,t:2,p:2484,kind:"L",label:"LL"}]),b=s("5m","DOWN",2505,2489,"LH","LL",[{i:3,t:3,p:2478,kind:"L",label:"LL"}]);assert.equal(selectEconomicTarget("DOWN",2500,22,[a,b]),2484);});
Deno.test("2x remains locked before calibrated evidence",()=>{const cal:Cal={samples:20,hits:16,p:null,lower:.6,upper:.9,ambiguous:0};assert.equal(chooseShadowLeverage({maxAllowed:2,cal,raw:.9,rr:4,targetBps:100,costBps:20,ofi:.8}),1);});
Deno.test("2x opens only on strong calibrated edge",()=>{const cal:Cal={samples:60,hits:45,p:.75,lower:.62,upper:.84,ambiguous:1};assert.equal(chooseShadowLeverage({maxAllowed:2,cal,raw:.8,rr:3,targetBps:90,costBps:20,ofi:.5}),2);});
Deno.test("2x falls back to 1x when any edge gate weakens",()=>{const cal:Cal={samples:60,hits:45,p:.75,lower:.62,upper:.84,ambiguous:1};assert.equal(chooseShadowLeverage({maxAllowed:2,cal,raw:.65,rr:3,targetBps:90,costBps:20,ofi:.5}),1);assert.equal(chooseShadowLeverage({maxAllowed:2,cal,raw:.8,rr:2.1,targetBps:90,costBps:20,ofi:.5}),1);});
