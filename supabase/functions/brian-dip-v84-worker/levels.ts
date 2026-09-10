import { hash, type Direction, type Struct } from "../_shared/dip_v84_contract.ts";

export type LevelFreshness="UNTOUCHED"|"TOUCHED"|"SWEPT"|"CLOSE_CROSSED"|"REVALIDATED";
export type StructuralLevel={
  level_id:string;
  price:number;
  normalized_price:number;
  kind:"H"|"L";
  origin:"PIVOT"|"EQUAL_PAIR";
  timeframe:string;
  confirmed_at:number;
  source_times:number[];
  freshness:LevelFreshness;
};

const TF_MS:Record<string,number>={"1m":60_000,"5m":300_000,"15m":900_000,"1h":3_600_000,"4h":14_400_000};
const TF_PRIORITY:Record<string,number>={"4h":5,"1h":4,"15m":3,"5m":2,"1m":1};
const PIVOT_RIGHT=3;

function normalizedPrice(price:number,tickSize:number):number{
  if(!(price>0)||!(tickSize>0)||![price,tickSize].every(Number.isFinite))throw Error("V84_INVALID_LEVEL_PRICE");
  const ticks=Math.round(price/tickSize);
  return Number((ticks*tickSize).toFixed(12));
}

async function makeLevel(input:Omit<StructuralLevel,"level_id"|"freshness">):Promise<StructuralLevel>{
  const level_id=await hash([input.kind,input.origin,input.timeframe,input.normalized_price,...input.source_times,input.confirmed_at].join("|"));
  return{...input,level_id,freshness:"UNTOUCHED"};
}

export async function collectStructuralLevels(structs:Struct[],tickSize:number,signalAt:number):Promise<StructuralLevel[]>{
  if(!Number.isFinite(signalAt)||signalAt<=0)throw Error("V84_INVALID_SIGNAL_AT");
  const raw:StructuralLevel[]=[];
  for(const s of structs){
    const tfMs=TF_MS[s.tf];if(!tfMs)continue;
    for(const p of s.pivots){
      const confirmed=p.t+(PIVOT_RIGHT+1)*tfMs;
      if(confirmed>signalAt)continue;
      const normalized_price=normalizedPrice(p.p,tickSize);
      raw.push(await makeLevel({price:p.p,normalized_price,kind:p.kind,origin:"PIVOT",timeframe:s.tf,confirmed_at:confirmed,source_times:[p.t]}));
    }
    for(const kind of ["H","L"] as const){
      const ps=s.pivots.filter(p=>p.kind===kind);
      for(let i=ps.length-1;i>0;i--){
        if(Math.abs(ps[i].p-ps[i-1].p)<=s.atr*.15){
          const price=(ps[i].p+ps[i-1].p)/2;
          const confirmed=Math.max(ps[i].t,ps[i-1].t)+(PIVOT_RIGHT+1)*tfMs;
          if(confirmed<=signalAt){
            const normalized_price=normalizedPrice(price,tickSize);
            raw.push(await makeLevel({price,normalized_price,kind,origin:"EQUAL_PAIR",timeframe:s.tf,confirmed_at:confirmed,source_times:[ps[i-1].t,ps[i].t]}));
          }
          break;
        }
      }
    }
  }
  // Canonicalize equal tick prices. Preserve the higher timeframe, then earlier confirmation, then stable id.
  const byPrice=new Map<string,StructuralLevel>();
  for(const level of raw){
    const key=`${level.kind}:${level.normalized_price.toFixed(12)}`;
    const old=byPrice.get(key);
    if(!old||compareTie(level,old)<0)byPrice.set(key,level);
  }
  return [...byPrice.values()];
}

function compareTie(a:StructuralLevel,b:StructuralLevel):number{
  const tf=(TF_PRIORITY[b.timeframe]??0)-(TF_PRIORITY[a.timeframe]??0);
  if(tf)return tf;
  if(a.confirmed_at!==b.confirmed_at)return a.confirmed_at-b.confirmed_at;
  return a.level_id.localeCompare(b.level_id);
}

export async function firstForwardLevel(input:{direction:Direction;fill:number;signalAt:number;tickSize:number;structs:Struct[]}):Promise<{l1:StructuralLevel|null;levels:StructuralLevel[]}>{
  const {direction,fill,signalAt,tickSize,structs}=input;
  if(!(fill>0)||!Number.isFinite(fill))throw Error("V84_INVALID_FILL");
  const kind=direction==="UP"?"H":"L";
  const levels=(await collectStructuralLevels(structs,tickSize,signalAt)).filter(x=>x.kind===kind&&(direction==="UP"?x.normalized_price>fill:x.normalized_price<fill));
  levels.sort((a,b)=>{
    const da=Math.abs(a.normalized_price-fill),db=Math.abs(b.normalized_price-fill);
    if(Math.abs(da-db)>tickSize/2)return da-db;
    return compareTie(a,b);
  });
  return{l1:levels[0]??null,levels};
}
