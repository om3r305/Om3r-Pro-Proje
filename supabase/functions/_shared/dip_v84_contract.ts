// Brian DIP V8.4 Package-1 contracts. GitHub preparation only; SHADOW ONLY.
// IMPORTANT: this file is intentionally separate from dip_v8_dual.ts so V8.3 remains byte-for-byte frozen.

export const SYMBOL = "ETHUSDT" as const;
export const RELEASE_ID = "dip-v84-package1-evidence-20260910.1" as const;
export const RELEASE_STAGE = "GITHUB_PREPARED_NOT_DEPLOYED" as const;
export const ENGINE_VERSION = "brian-dip-v84" as const;
export const POLICY_VERSION = "dip-v84-l1-evidence-20260910.1" as const;
export const DECISION_REVISION = "dip-v84-l1-occurrence-20260910.1" as const;
export const METRIC_VERSION = "target-before-invalidation-v84.1" as const;
export const RESOLVER_VERSION = "dip-v84-path-20260910.1" as const;
export const ENTRY_GUARD_VERSION = "entry-zone-v1" as const;
export const TARGET_PLANNER_VERSION = "l1-nearest-structural-v84.1" as const;
export const EXECUTION_MODEL_VERSION = "fill-forward-frozen-exit-v84.1" as const;
export const COST_MODEL_VERSION = "FILL_FORWARD_V84_1" as const;
export const MARKET_DATA_CONTRACT_VERSION = "binance-usdm-sealed-v84.1" as const;
export const DB_CONTRACT_VERSION = "brian-dip-v84-db-1" as const;
export const CALIBRATION_FAMILY_ID = "dip-v84-package1-evidence-family-1" as const;
export const STRATEGY_MANIFEST_HASH = "1b4938913e189d04d0b325727e0318b786c4c6fa4734d575f8b278a1d0696fb0" as const;

// This is deliberately not sealed until the GitHub implementation commit is final.
// Any deployed worker must replace this with a real immutable source/logic hash and register it in DB.
export const LOGIC_HASH = "UNSEALED_GITHUB_ONLY" as const;

export const MAX_HOLD_MS = 90 * 60_000;
export const MIN_EXECUTION_CAL_SAMPLES = 40;
export const ACCOUNT_RISK_FRACTION = 0.005;
export const COLD_MAX_NOTIONAL_FRACTION = 0.08;
export const WARM_RISK_PROMOTION_ENABLED = false;
export const MAX_SHADOW_LEVERAGE: 1 = 1;
export const MIN_ECONOMIC_RR = 1.25;
export const TARGET_COST_MULTIPLE = 2.5;

export type J = Record<string, unknown>;
export type Direction = "UP" | "DOWN";
export type Setup = "SWEEP_RECLAIM" | "FAILED_BREAK" | "BOS_RETEST" | "EARLY_REVERSAL";
export type CalibrationState = "UNAVAILABLE" | "COLD_NEW_FAMILY" | "WARM";

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
export type FundingSettlement = { fundingTime:number; fundingRate:number; markPrice:number; cashflow:number };
export type Segment = { start:number; end:number; o:number; h:number; l:number; c:number; kind:"BAR"|"TRADES"; points?:{t:number;id:number;p:number}[] };
export type Resolution = {
  reason:"TARGET_FIRST"|"INVALIDATION_FIRST"|"AMBIGUOUS"|"EXPIRED_NO_BARRIER"|"PENDING"|"INDETERMINATE";
  hit:boolean|null; price:number; eventAt:number|null; eventRangeStart:number|null; checkedUntil:number; detail?:string;
};

export type ExecutionCalibration = {
  state:CalibrationState;
  validContract:boolean;
  samples:number;
  episodes:number;
  days:number;
  wins:number;
  losses:number;
  p:number|null;
  lower:number|null;
  upper:number|null;
  ambiguousLosses:number;
  unavailableReason:string|null;
};

export type Position = {
  side:"LONG"|"SHORT";
  position_id:string; thesis_id:string; episode_id:string; setup:Setup; regime:string;
  entry:number; qty:number; notional:number; target:number; stop:number;
  stop_source:string; stop_structure_id:string|null;
  opened_at:string; due_at:string; fees_open:number; fee_bps:number; slippage_bps:number;
  expected_exit_spread_bps:number; expected_exit_slippage_bps:number;
  venue:"SHADOW_PERP"; policy_version:string; release_id:string; calibration_family_id:string;
  checked_until:number; market_price:number;
  leverage:1; margin:number; actual_fraction:number; gross_fraction:number;
  funding_accrued?:number; funding_settlements?:FundingSettlement[];
};

export type Runtime = {
  start:number; cash:number; realized:number; trades:number; wins:number; losses:number;
  pos:Position|null;
  latestThesis:J|null;
  lastOccurrence:string|null;
  lastSnapshotHour:string|null;
  marketCursor:number;
  lastClosedAt:number|null;
};

export type ReleaseContract = {
  release_id:string;
  logic_hash:string;
  strategy_manifest_hash:string;
  calibration_family_id:string;
  db_contract_version:string;
  engine_version:string;
  policy_version:string;
  decision_revision:string;
  metric_version:string;
  resolver_version:string;
  entry_guard_version:string;
  target_planner_version:string;
  execution_model_version:string;
  cost_model_version:string;
  market_data_contract_version:string;
};

export const RELEASE_CONTRACT:ReleaseContract = Object.freeze({
  release_id:RELEASE_ID,
  logic_hash:LOGIC_HASH,
  strategy_manifest_hash:STRATEGY_MANIFEST_HASH,
  calibration_family_id:CALIBRATION_FAMILY_ID,
  db_contract_version:DB_CONTRACT_VERSION,
  engine_version:ENGINE_VERSION,
  policy_version:POLICY_VERSION,
  decision_revision:DECISION_REVISION,
  metric_version:METRIC_VERSION,
  resolver_version:RESOLVER_VERSION,
  entry_guard_version:ENTRY_GUARD_VERSION,
  target_planner_version:TARGET_PLANNER_VERSION,
  execution_model_version:EXECUTION_MODEL_VERSION,
  cost_model_version:COST_MODEL_VERSION,
  market_data_contract_version:MARKET_DATA_CONTRACT_VERSION,
});

export const STRATEGY_MANIFEST = Object.freeze({
  browser_execution:false,
  calibration:{execution_source:"ACTUAL_SHADOW_FILLS_ONLY",forecast_authority:false,min_samples_for_probability:40,warm_activation_enabled:false},
  cost:{entry_price:"ask/bid plus opening slippage",exit_model:"frozen-at-open executable-side conversion once",expected_funding_horizon_minutes:90,version:COST_MODEL_VERSION},
  decision_cadence_seconds:15,
  entry_guard:{early_reversal_max_extension_atr:2,version:ENTRY_GUARD_VERSION},
  live_execution:false,
  max_hold_ms:MAX_HOLD_MS,
  risk:{account_fraction:ACCOUNT_RISK_FRACTION,cold_max_notional_fraction:COLD_MAX_NOTIONAL_FRACTION,max_shadow_leverage:MAX_SHADOW_LEVERAGE,risk_promotion_enabled:false},
  server_authoritative:true,
  shadow_only:true,
  structure:{atr_period:14,bos_buffer_atr:.15,pivot_left:3,pivot_right:3},
  symbol:SYMBOL,
  target:{live_l2:false,min_economic_rr:MIN_ECONOMIC_RR,policy:"NEAREST_STRUCTURAL_THEN_ECONOMIC_GATE",skip_obstacle:false,target_cost_multiple:TARGET_COST_MULTIPLE},
});

export function n(value:unknown,fallback=0):number { const x=Number(value); return Number.isFinite(x)?x:fallback; }
export function clip(x:number,low:number,high:number):number { return Math.max(low,Math.min(high,x)); }
export function mean(xs:number[]):number { return xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:0; }
export async function hash(s:string):Promise<string> {
  const b=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(s)));
  return [...b].map(x=>x.toString(16).padStart(2,"0")).join("");
}
export function same(a:string,b:string):boolean { if(a.length!==b.length)return false; let d=0; for(let i=0;i<a.length;i++)d|=a.charCodeAt(i)^b.charCodeAt(i); return d===0; }

function canonicalNumber(x:number):number {
  if(!Number.isFinite(x))throw Error("NON_FINITE_MANIFEST_NUMBER");
  return Object.is(x,-0)?0:x;
}
export function canonicalize(value:unknown):unknown {
  if(value===null||typeof value==="string"||typeof value==="boolean")return value;
  if(typeof value==="number")return canonicalNumber(value);
  if(Array.isArray(value))return value.map(canonicalize);
  if(typeof value==="object"){
    const input=value as Record<string,unknown>,out:Record<string,unknown>={};
    for(const key of Object.keys(input).sort()){
      const v=input[key];
      if(v===undefined)throw Error("UNDEFINED_MANIFEST_VALUE:"+key);
      out[key]=canonicalize(v);
    }
    return out;
  }
  throw Error("UNSUPPORTED_MANIFEST_VALUE");
}
export function canonicalJson(value:unknown):string { return JSON.stringify(canonicalize(value)); }
export async function manifestHash(value:unknown):Promise<string> { return hash(canonicalJson(value)); }

export function initialRuntime(start:number):Runtime {
  if(!Number.isFinite(start)||start<=0)throw Error("INVALID_STARTING_EQUITY");
  return {start,cash:start,realized:0,trades:0,wins:0,losses:0,pos:null,latestThesis:null,lastOccurrence:null,lastSnapshotHour:null,marketCursor:0,lastClosedAt:null};
}

export function validateSession(config:J):void {
  if(config.release_id!==RELEASE_ID||config.strategy_manifest_hash!==STRATEGY_MANIFEST_HASH||config.calibration_family_id!==CALIBRATION_FAMILY_ID||config.db_contract_version!==DB_CONTRACT_VERSION)throw Error("V84_RELEASE_CONTRACT_MISMATCH");
  if(config.engine_version!==ENGINE_VERSION||config.policy_version!==POLICY_VERSION||config.decision_revision!==DECISION_REVISION||config.metric_version!==METRIC_VERSION)throw Error("V84_VERSION_CONTRACT_MISMATCH");
  if(JSON.stringify(config.symbols)!==JSON.stringify([SYMBOL]))throw Error("V84_SYMBOL_CONTRACT_MISMATCH");
  if(config.shadow_only!==true||config.live_execution!==false||config.browser_execution!==false||config.server_authoritative!==true)throw Error("V84_INVALID_SHADOW_CONTRACT");
  if(config.allow_shadow_short!==true||n(config.max_shadow_leverage,0)!==MAX_SHADOW_LEVERAGE)throw Error("V84_INVALID_RISK_CONTRACT");
  if(!["OBSERVE","SHADOW_PAPER"].includes(String(config.execution_mode)))throw Error("V84_INVALID_EXECUTION_MODE");
  if(WARM_RISK_PROMOTION_ENABLED)throw Error("V84_EVIDENCE_RELEASE_CANNOT_ENABLE_WARM_RISK");
}

function wilson(samples:number,hits:number):{p:number;lower:number;upper:number}{
  const p=samples?hits/samples:0,z=1.96,d=1+z*z/Math.max(samples,1);
  const centre=(p+z*z/(2*Math.max(samples,1)))/d;
  const width=z*Math.sqrt((p*(1-p)+z*z/(4*Math.max(samples,1)))/Math.max(samples,1))/d;
  return{p,lower:samples?Math.max(0,centre-width):0,upper:samples?Math.min(1,centre+width):1};
}

export function unavailableCalibration(reason:string):ExecutionCalibration {
  return {state:"UNAVAILABLE",validContract:false,samples:0,episodes:0,days:0,wins:0,losses:0,p:null,lower:null,upper:null,ambiguousLosses:0,unavailableReason:reason};
}

export function executionCalibration(input:{wins:number;losses:number;episodes:number;days:number;ambiguousLosses:number}):ExecutionCalibration {
  const values=[input.wins,input.losses,input.episodes,input.days,input.ambiguousLosses];
  if(values.some(x=>!Number.isSafeInteger(x)||x<0))return unavailableCalibration("INVALID_EXECUTION_CALIBRATION_COUNTS");
  const samples=input.wins+input.losses;
  if(input.episodes>samples||input.ambiguousLosses>input.losses)return unavailableCalibration("INVALID_EXECUTION_CALIBRATION_RELATION");
  const state:CalibrationState=samples<MIN_EXECUTION_CAL_SAMPLES?"COLD_NEW_FAMILY":"WARM";
  if(state==="COLD_NEW_FAMILY")return{state,validContract:true,samples,episodes:input.episodes,days:input.days,wins:input.wins,losses:input.losses,p:null,lower:null,upper:null,ambiguousLosses:input.ambiguousLosses,unavailableReason:null};
  const w=wilson(samples,input.wins);
  return{state,validContract:true,samples,episodes:input.episodes,days:input.days,wins:input.wins,losses:input.losses,p:w.p,lower:w.lower,upper:w.upper,ambiguousLosses:input.ambiguousLosses,unavailableReason:null};
}

export function calibrationNoEdge(cal:ExecutionCalibration,economicRR:number):boolean|null {
  if(cal.state!=="WARM"||cal.lower===null)return null;
  if(!(economicRR>0)&&economicRR!==0)return true;
  return cal.lower<=1/(1+Math.max(0,economicRR));
}

// Package-1 evidence release never grants risk promotion even if execution calibration reaches WARM.
export function executionPermission(cal:ExecutionCalibration):{entryAllowed:boolean;leverage:1;maxNotionalFraction:number;reason:string}{
  if(cal.state==="UNAVAILABLE"||!cal.validContract)return{entryAllowed:false,leverage:1,maxNotionalFraction:0,reason:"FAIL_CLOSED_CALIBRATION_UNAVAILABLE"};
  return{entryAllowed:true,leverage:1,maxNotionalFraction:COLD_MAX_NOTIONAL_FRACTION,reason:cal.state==="COLD_NEW_FAMILY"?"COLD_EVIDENCE_COLLECTION":"WARM_STATS_RISK_PROMOTION_FROZEN"};
}

export function evaluatePath(input:{direction:Direction;target:number;stop:number;start:number;due:number;now:number;end:number;entry:number;segments:Segment[]}):Resolution {
  const {direction,target,stop,start,due,now,end,entry,segments}=input;
  const result=(reason:Resolution["reason"],hit:boolean|null,price:number,eventAt:number|null,checkedUntil:number,eventRangeStart:number|null=eventAt):Resolution=>({reason,hit,price,eventAt,eventRangeStart,checkedUntil});
  if(!(target>0&&stop>0&&start<due&&end<=now&&end<=due&&end>=start))return result("INDETERMINATE",null,entry,null,start);
  let cursor=start,last=entry;
  const touched=(p:number)=>direction==="UP"?{win:p>=target,loss:p<=stop}:{win:p<=target,loss:p>=stop};
  for(const s of segments){
    if(s.start!==cursor||s.end<=s.start||s.end>end||![s.o,s.h,s.l,s.c].every(Number.isFinite)||s.l<=0||s.h<s.l)return result("INDETERMINATE",null,last,null,cursor);
    if(s.kind==="TRADES"){
      let previousAt=s.start,previousId=-1;
      for(const p of s.points||[]){
        if(p.t<previousAt||p.t<s.start||p.t>=s.end||p.id<=previousId||!(p.p>0))return result("INDETERMINATE",null,last,null,cursor);
        const h=touched(p.p); previousAt=p.t; previousId=p.id; last=p.p;
        if(h.loss)return result("INVALIDATION_FIRST",false,p.p,p.t,s.end);
        if(h.win)return result("TARGET_FIRST",true,target,p.t,s.end);
      }
    }else{
      const open=touched(s.o);
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
