// Brian DIP V8.3 dual-direction SHADOW contracts. No live exchange execution.
export const SYMBOL = "ETHUSDT";
export const ENGINE_VERSION = "brian-dip-chart-reader-v8-dual";
export const POLICY_VERSION = "dip-v8-dual-20260908.2";
export const METRIC_VERSION = "target-before-invalidation-v8.2";
export const RESOLVER_VERSION = "dip-v8-path-20260908.5";
export const MAX_HOLD_MS = 90 * 60_000;
export const MIN_CAL_SAMPLES = 40;
export const MAX_SHADOW_LEVERAGE = 2;
export const ACCOUNT_RISK_FRACTION = 0.005;
export type J = Record<string, unknown>;
export type Bar = { t:number; ct:number; o:number; h:number; l:number; c:number; v:number };
export type Pivot = { i:number; t:number; p:number; kind:"H"|"L"; label?:string };
export type Struct = {
  tf:string; lastClose:number; atr:number; pivots:Pivot[];
  lastHigh:Pivot|null; lastLow:Pivot|null;
  trend:"UP"|"DOWN"|"RANGE"; bos:"UP"|"DOWN"|null; choch:"UP"|"DOWN"|null;
  sweep:"BULL"|"BEAR"|null; failedBreak:"BULL"|"BEAR"|null;
  equalHigh:number|null; equalLow:number|null; fingerprint:string;
};
export type Rules = { minNotional:number; minQty:number; maxQty:number; stepSize:number };
export type Cal = { samples:number; hits:number; p:number|null; lower:number; upper:number; ambiguous:number; unavailable?:boolean };
export type Position = {
  side:"LONG"|"SHORT";
  position_id:string; thesis_id:string; episode_id:string; setup:string; regime:string;
  entry:number; qty:number; notional:number; target:number; stop:number;
  opened_at:string; due_at:string; fees_open:number; fee_bps:number; slippage_bps:number; spread_bps:number;
  venue:"SHADOW_PERP"; policy_version:string; checked_until:number; market_price:number;
  leverage:1|2; margin:number; actual_fraction:number; gross_fraction:number;
  funding_accrued?:number; funding_settlements?:FundingSettlement[];
};
export type FundingSettlement = { fundingTime:number; fundingRate:number; markPrice:number; cashflow:number };
export type Runtime = {
  start:number; cash:number; realized:number; trades:number; wins:number; losses:number;
  pos:Position|null;
  lastLock:{episode_id:string;fingerprint:string;last5m_t:number;reason:string}|null;
  latestThesis:J|null; lastOccurrence:string|null; lastSnapshotHour:string|null;
  marketCursor:number; lastClosedAt:number|null;
};
export type Segment = { start:number; end:number; o:number; h:number; l:number; c:number; kind:"BAR"|"TRADES"; points?:{t:number;id:number;p:number}[] };
export type Resolution = {
  reason:"TARGET_FIRST"|"INVALIDATION_FIRST"|"AMBIGUOUS"|"EXPIRED_NO_BARRIER"|"PENDING"|"INDETERMINATE";
  hit:boolean|null; price:number; eventAt:number|null; eventRangeStart:number|null; checkedUntil:number; detail?:string;
};

export function n(value:unknown,fallback=0):number { const x=Number(value); return Number.isFinite(x)?x:fallback; }
export function clip(x:number,low:number,high:number):number { return Math.max(low,Math.min(high,x)); }
export function mean(xs:number[]):number { return xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:0; }
export async function hash(s:string):Promise<string> {
  const b=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(s)));
  return [...b].map(x=>x.toString(16).padStart(2,"0")).join("");
}
export function same(a:string,b:string):boolean { if(a.length!==b.length)return false; let d=0; for(let i=0;i<a.length;i++)d|=a.charCodeAt(i)^b.charCodeAt(i); return d===0; }
export function initialRuntime(start:number):Runtime {
  if(!Number.isFinite(start)||start<=0)throw Error("INVALID_STARTING_EQUITY");
  return {start,cash:start,realized:0,trades:0,wins:0,losses:0,pos:null,lastLock:null,latestThesis:null,lastOccurrence:null,lastSnapshotHour:null,marketCursor:0,lastClosedAt:null};
}
export function validateSession(config:J):void {
  if(config.engine_version!==ENGINE_VERSION||config.policy_version!==POLICY_VERSION||JSON.stringify(config.symbols)!==JSON.stringify([SYMBOL]))throw Error("WAIT_V8_DUAL_CLEAN_RESTART");
  if(config.shadow_only!==true||config.live_execution!==false||config.browser_execution!==false||config.server_authoritative!==true)throw Error("INVALID_SHADOW_CONTRACT");
  if(config.allow_shadow_short!==true||n(config.max_shadow_leverage,0)!==MAX_SHADOW_LEVERAGE)throw Error("INVALID_DUAL_DIRECTION_CONTRACT");
  if(!["OBSERVE","SHADOW_PAPER"].includes(String(config.execution_mode)))throw Error("INVALID_EXECUTION_MODE");
}
export function calibrate(rows:{hit:boolean|null}[],unavailable=false):Cal {
  const clean=rows.filter(x=>x.hit!==null),samples=clean.length,hits=clean.filter(x=>x.hit).length;
  const p=samples?hits/samples:0,z=1.96,d=1+z*z/Math.max(samples,1);
  const centre=(p+z*z/(2*Math.max(samples,1)))/d;
  const width=z*Math.sqrt((p*(1-p)+z*z/(4*Math.max(samples,1)))/Math.max(samples,1))/d;
  return {samples,hits,p:samples>=MIN_CAL_SAMPLES?p:null,lower:samples?Math.max(0,centre-width):0,upper:samples?Math.min(1,centre+width):1,ambiguous:rows.length-clean.length,unavailable};
}

export function chooseShadowLeverage(input:{maxAllowed:number;cal:Cal;raw:number;economicRR:number;targetBps:number;costBps:number;flowScore:number}):1|2 {
  const {maxAllowed,cal,raw,economicRR,targetBps,costBps,flowScore}=input;
  if(maxAllowed<2||cal.unavailable||cal.p===null||cal.samples<MIN_CAL_SAMPLES)return 1;
  const ambiguousRate=cal.ambiguous/Math.max(1,cal.samples+cal.ambiguous);
  return cal.p>=0.68&&cal.lower>=0.55&&raw>=0.72&&economicRR>=2.5&&targetBps>=3.5*costBps&&Math.abs(flowScore)>=0.25&&ambiguousRate<=0.10?2:1;
}

export function sizePosition(
  cash:number,tradeNotional:number,direction:"UP"|"DOWN",entry:number,stop:number,cal:Cal,
  feeBps:number,slippageBps:number,spreadBps:number,rules:Rules,leverage:1|2,raw:number,
) {
  const validStop=direction==="UP"?entry>stop:stop>entry;
  if(!(cash>0&&tradeNotional>0&&entry>0&&stop>0&&validStop)||cal.unavailable)return null;
  const cold=cal.p===null||cal.samples<MIN_CAL_SAMPLES;
  if(cold&&leverage!==1)return null;
  const maxNotional=Math.min(tradeNotional,cash*(cold?.08:.20));
  const maxMargin=Math.min(cash,maxNotional/leverage);
  const alloc=cal.p===null
    ? (raw>=.78?1:raw>=.70?.75:raw>=.64?.55:.35)
    : clip(.55+Math.max(0,cal.p-.55)*2.2,.55,1);
  const riskMove=Math.abs(entry-stop);
  const roundTripBps=feeBps*2+slippageBps*2+spreadBps;
  const lossPerUnit=riskMove+entry*roundTripBps/10000;
  const riskBudget=cash*ACCOUNT_RISK_FRACTION;
  const byRisk=riskBudget/lossPerUnit;
  const byExposure=maxNotional/entry;
  const byCash=cash/(entry/leverage+entry*feeBps/10000);
  const budgetQty=Math.min(byRisk,byExposure,byCash,rules.maxQty);
  const qty=Math.floor((budgetQty+Number.EPSILON)/rules.stepSize)*rules.stepSize;
  const notional=qty*entry,margin=notional/leverage,fees_open=notional*feeBps/10000;
  const worstLoss=qty*lossPerUnit;
  if(!Number.isFinite(qty)||qty<rules.minQty||notional<rules.minNotional||margin+fees_open>cash+1e-8||margin>maxMargin+1e-8||worstLoss>riskBudget+1e-8)return null;
  return {qty,notional,margin,leverage,actual_fraction:margin/cash,gross_fraction:notional/cash,fees_open,worst_loss:worstLoss,risk_fraction:worstLoss/cash,allocation:alloc};
}

// Windows are [start,end). Sealed evidence only.
export function evaluatePath(input:{direction:"UP"|"DOWN";target:number;stop:number;start:number;due:number;now:number;end:number;entry:number;segments:Segment[]}):Resolution {
  const {direction,target,stop,start,due,now,end,entry,segments}=input;
  const result=(reason:Resolution["reason"],hit:boolean|null,price:number,eventAt:number|null,checkedUntil:number,eventRangeStart:number|null=eventAt):Resolution=>({reason,hit,price,eventAt,eventRangeStart,checkedUntil});
  if(!(target>0&&stop>0&&start<due&&end<=now&&end<=due&&end>=start))return result("INDETERMINATE",null,entry,null,start);
  let cursor=start,last=entry;
  const hit=(p:number)=>direction==="UP"?{win:p>=target,loss:p<=stop}:{win:p<=target,loss:p>=stop};
  for(const s of segments){
    if(s.start!==cursor||s.end<=s.start||s.end>end||![s.o,s.h,s.l,s.c].every(Number.isFinite)||s.l<=0||s.h<s.l)return result("INDETERMINATE",null,last,null,cursor);
    if(s.kind==="TRADES"){
      let previousAt=s.start,previousId=-1;
      for(const p of s.points||[]){
        if(p.t<previousAt||p.t<s.start||p.t>=s.end||p.id<=previousId||!(p.p>0))return result("INDETERMINATE",null,last,null,cursor);
        const h=hit(p.p); previousAt=p.t; previousId=p.id; last=p.p;
        if(h.loss)return result("INVALIDATION_FIRST",false,p.p,p.t,s.end);
        if(h.win)return result("TARGET_FIRST",true,target,p.t,s.end);
      }
    }else{
      const open=hit(s.o);
      if(open.loss)return result("INVALIDATION_FIRST",false,s.o,s.start,s.end);
      if(open.win)return result("TARGET_FIRST",true,target,s.start,s.end);
      const win=direction==="UP"?s.h>=target:s.l<=target,loss=direction==="UP"?s.l<=stop:s.h>=stop;
      if(win&&loss)return result("AMBIGUOUS",null,stop,s.end-1,s.end,s.start);
      if(loss)return result("INVALIDATION_FIRST",false,stop,s.end-1,s.end,s.start);
      if(win)return result("TARGET_FIRST",true,target,s.end-1,s.end,s.start);
      last=s.c;
    }
    cursor=s.end;
  }
  if(cursor!==end)return result("INDETERMINATE",null,last,null,cursor);
  return end===due?result("EXPIRED_NO_BARRIER",false,last,due,end):result("PENDING",null,last,null,cursor);
}

export function closePosition(rt:Runtime,resolution:Resolution,fingerprint:string,last5m:number,recordedAt:number):J|null {
  const p=rt.pos;
  if(!p||["PENDING","INDETERMINATE"].includes(resolution.reason))return null;
  const friction=(p.spread_bps/2+p.slippage_bps)/10000;
  const exit=resolution.price*(p.side==="LONG"?1-friction:1+friction);
  const feeClose=exit*p.qty*p.fee_bps/10000;
  const gross=(p.side==="LONG"?exit-p.entry:p.entry-exit)*p.qty;
  const funding=p.funding_accrued??0;
  if(!Number.isFinite(funding))throw Error("INVALID_FUNDING");
  const net=gross-p.fees_open-feeClose+funding;
  rt.cash+=p.margin+gross-feeClose+funding;
  rt.realized+=net; rt.trades++;
  if(net>0)rt.wins++; else if(net<0)rt.losses++;
  rt.lastLock={episode_id:p.episode_id,fingerprint,last5m_t:last5m,reason:resolution.reason};
  rt.lastClosedAt=recordedAt; rt.pos=null;
  return {
    event_kind:p.side==="LONG"?"SELL":"SHORT_CLOSE",
    position_id:p.position_id,occurrence_id:p.thesis_id,episode_id:p.episode_id,
    price:exit,entry_price:p.entry,exit_price:exit,quantity:p.qty,notional:p.notional,
    fees:p.fees_open+feeClose,funding_cashflow:funding,realized_pnl:net,cash_after:rt.cash,equity_after:rt.cash,
    metadata:{server_v8:true,side:p.side,funding_cashflow:funding,funding_settlements:p.funding_settlements??[],funding_model:"SETTLED_RATE_MARK_PRICE",exit_reason:resolution.reason==="AMBIGUOUS"?"AMBIGUOUS_CONSERVATIVE_STOP":resolution.reason,thesis_id:p.thesis_id,setup:p.setup,venue:p.venue,policy_version:p.policy_version,resolution,execution_model:"dual_shadow_perp_barrier_v83",fee_bps:p.fee_bps,slippage_bps:p.slippage_bps,leverage:p.leverage,margin:p.margin}
  };
}
