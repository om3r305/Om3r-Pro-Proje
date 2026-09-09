// Diagnostic evidence only. Never changes target, veto, identity, sizing or execution.
import type { Bar, J, Pivot } from '../_shared/dip_v8_dual.ts';
import { atr, classifyPivots } from './structure.ts';
const RIGHT=3;
export type LevelObservation={
  tf:string; kind:'H'|'L'; origin:'PIVOT'|'EQUAL_PAIR'; price:number;
  pivot_at:number[]; confirmed_at:number; as_of:number;
  first_touch_bar:number|null; first_sweep_bar:number|null; first_close_cross_bar:number|null;
  status:'UNTOUCHED'|'TOUCHED'|'SWEPT'|'CLOSE_CROSSED';
  precision:'CLOSED_BAR_START'; coverage:'SINCE_CONFIRMATION';
};
const samePrice=(a:number,b:number)=>Math.abs(a-b)<=Math.max(1,Math.abs(a),Math.abs(b))*1e-10;
export function observeTarget(rows:Bar[],tf:string,target:number,decisionAt:number):LevelObservation[]{
  if(!Number.isFinite(decisionAt)||!Number.isFinite(target)||target<=0)return[];
  const sealed=rows.filter(b=>b.ct<decisionAt);
  for(let i=0;i<sealed.length;i++){
    const b=sealed[i];
    if(![b.t,b.ct,b.o,b.h,b.l,b.c].every(Number.isFinite)||b.ct<b.t||b.l<=0||b.h<Math.max(b.o,b.c)||b.l>Math.min(b.o,b.c)||(i&&b.t!==sealed[i-1].ct+1))throw Error('OBSERVATION_INVALID_BARS');
  }
  if(sealed.length<8)return[];
  const pivots=classifyPivots(sealed),definitions:{kind:'H'|'L';origin:'PIVOT'|'EQUAL_PAIR';pivots:Pivot[];price:number}[]=[];
  // Mirror the reader's visible pivot window, without filtering its target selection.
  for(const p of pivots.slice(-12))if(samePrice(p.p,target))definitions.push({kind:p.kind,origin:'PIVOT',pivots:[p],price:p.p});
  for(const kind of ['H','L'] as const){
    const ps=pivots.filter(p=>p.kind===kind);
    for(let i=ps.length-1;i>0;i--){
      if(Math.abs(ps[i].p-ps[i-1].p)<=atr(sealed)*.15){
        const price=(ps[i].p+ps[i-1].p)/2;
        if(samePrice(price,target))definitions.push({kind,origin:'EQUAL_PAIR',pivots:[ps[i-1],ps[i]],price});
        break;
      }
    }
  }
  return definitions.slice(-4).map(d=>{
    // Pair existence requires BOTH pivots to be confirmed. No earlier events are inferred.
    const confirmed=sealed[Math.max(...d.pivots.map(p=>p.i))+RIGHT].ct+1;
    let touch:number|null=null,sweep:number|null=null,cross:number|null=null;
    for(const b of sealed){
      if(b.t<confirmed)continue;
      const reached=d.kind==='H'?b.h>=d.price:b.l<=d.price;
      const swept=d.kind==='H'?b.h>d.price&&b.c<d.price:b.l<d.price&&b.c>d.price;
      const closed=d.kind==='H'?b.c>d.price:b.c<d.price;
      if(reached&&touch===null)touch=b.t;
      if(swept&&sweep===null)sweep=b.t;
      if(closed&&cross===null)cross=b.t;
    }
    return{tf,kind:d.kind,origin:d.origin,price:d.price,pivot_at:d.pivots.map(p=>p.t),confirmed_at:confirmed,as_of:decisionAt,first_touch_bar:touch,first_sweep_bar:sweep,first_close_cross_bar:cross,status:cross!==null?'CLOSE_CROSSED':sweep!==null?'SWEPT':touch!==null?'TOUCHED':'UNTOUCHED',precision:'CLOSED_BAR_START',coverage:'SINCE_CONFIRMATION'};
  });
}
export function withLevelObservations<T extends {thesis:J;decision:({evidence:J}|null);target:number|null}>(candidate:T,bars:Record<string,Bar[]>,at:number):T{
  try{
    const levels=candidate.target?['1m','5m','15m','1h'].flatMap(tf=>observeTarget(bars[tf]||[],tf,candidate.target!,at)).slice(0,8):[];
    const observation={version:'dip-level-observation-v1',mode:'OBSERVATION_ONLY',levels};
    const evidence=candidate.decision?{...candidate.decision.evidence,level_observation:observation}:null;
    // Leave headroom beneath the existing 15 KB SQL decision envelope (jsonb adds spaces).
    if(evidence&&new TextEncoder().encode(JSON.stringify({...candidate.decision,evidence})).length>11000)return candidate;
    return{...candidate,thesis:{...candidate.thesis,level_observation:observation},decision:candidate.decision?{...candidate.decision,evidence:evidence!}:null};
  }catch{
    // Optional diagnostics must not interrupt position handling or alter trade eligibility.
    return candidate;
  }
}
