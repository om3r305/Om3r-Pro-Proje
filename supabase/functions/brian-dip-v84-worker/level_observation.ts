import type { Bar } from "../_shared/dip_v84_contract.ts";
import type { StructuralLevel, LevelFreshness } from "./levels.ts";

export type ObservedLevel=StructuralLevel&{
  as_of:number;
  first_touch_bar:number|null;
  first_sweep_bar:number|null;
  first_close_cross_bar:number|null;
  freshness:LevelFreshness;
};

export function observeLevel(level:StructuralLevel,rows:Bar[],decisionAt:number):ObservedLevel{
  const sealed=rows.filter(b=>b.ct<decisionAt&&b.t>=level.confirmed_at);
  let touch:number|null=null,sweep:number|null=null,cross:number|null=null;
  for(const b of sealed){
    const reached=level.kind==="H"?b.h>=level.normalized_price:b.l<=level.normalized_price;
    const swept=level.kind==="H"?b.h>level.normalized_price&&b.c<level.normalized_price:b.l<level.normalized_price&&b.c>level.normalized_price;
    const closed=level.kind==="H"?b.c>level.normalized_price:b.c<level.normalized_price;
    if(reached&&touch===null)touch=b.t;if(swept&&sweep===null)sweep=b.t;if(closed&&cross===null)cross=b.t;
  }
  const freshness:LevelFreshness=cross!==null?"CLOSE_CROSSED":sweep!==null?"SWEPT":touch!==null?"TOUCHED":"UNTOUCHED";
  return{...level,as_of:decisionAt,first_touch_bar:touch,first_sweep_bar:sweep,first_close_cross_bar:cross,freshness};
}
