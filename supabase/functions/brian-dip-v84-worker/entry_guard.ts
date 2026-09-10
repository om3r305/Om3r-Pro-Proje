import type { Bar, Struct } from "../_shared/dip_v84_contract.ts";
import { ENTRY_GUARD_VERSION } from "../_shared/dip_v84_contract.ts";

// V8.4 Package 1 intentionally preserves the confirmed V8.3 entry-zone semantics.
// Market structure uses raw quote; simulated fill friction cannot break a level.
// A crossed local obstacle remains active until three sealed closes beyond the level
// and the third close's candle retests within the existing ATR tolerance.
export function entryGuard(direction:"UP"|"DOWN",setup:string,quote:number,fill:number,rows:Bar[],structs:Struct[],atr:number,now:number){
  const sign=direction==="UP"?1:-1,closed=rows.filter(b=>b.ct<now),recent=closed.slice(-8),veto:string[]=[];
  if(recent.length<8||!(atr>0)||!Number.isFinite(quote)||!Number.isFinite(fill))return{version:ENTRY_GUARD_VERSION,veto:["ENTRY_DATA_INCOMPLETE"],obstacles:[],extension_atr:null};
  const low=Math.min(...recent.map(b=>b.l)),high=Math.max(...recent.map(b=>b.h)),tol=atr*.15;
  const levels=new Map<number,number>();
  for(const s of structs)for(const p of s.pivots){
    if(p.kind!==(direction==="UP"?"H":"L"))continue;
    const tfMs=({"1m":60000,"5m":300000,"15m":900000,"1h":3600000} as Record<string,number>)[s.tf]||60000;
    const confirmed=p.t+4*tfMs;
    if(p.p>=Math.min(low,quote,fill)-tol&&p.p<=Math.max(high,quote,fill)+tol)levels.set(p.p,Math.max(levels.get(p.p)||0,confirmed));
  }
  const obstacles:Array<{price:number;confirmed_at:number;retested:boolean}>=[];
  for(const [level,confirmed] of levels){
    if(sign*(quote-level)<-tol&&sign*(fill-level)<0)continue;
    const bars=recent.filter(b=>b.t>=confirmed);let retested=false;
    for(let i=2;i<bars.length;i++){
      const a=bars[i-2],b=bars[i-1],c=bars[i];
      if(sign*(a.c-level)>tol&&sign*(b.c-level)>tol&&sign*(c.c-level)>tol&&
        (direction==="UP"?c.l<=level+tol&&c.l>=level-tol:c.h>=level-tol&&c.h<=level+tol))retested=true;
    }
    if(sign*(quote-level)<=tol||!retested)veto.push("WAIT_RETEST");
    obstacles.push({price:level,confirmed_at:confirmed,retested});
  }
  const extension=(direction==="UP"?quote-low:high-quote)/atr;
  if(setup==="EARLY_REVERSAL"&&extension>2)veto.push("ENTRY_TOO_LATE");
  return{version:ENTRY_GUARD_VERSION,veto:[...new Set(veto)],obstacles,extension_atr:extension};
}
