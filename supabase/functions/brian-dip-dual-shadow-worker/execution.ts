import { hash, type J } from '../_shared/dip_v8_dual.ts';
import type { CandidateResult } from './decision.ts';

// Entry already includes ask/bid and opening slippage. Match closePosition exactly.
export function executionEconomics(side:'UP'|'DOWN',entry:number,target:number,stop:number,fee:number,slip:number,spread:number,funding:number){
  if(![entry,target,stop,fee,slip,spread,funding].every(Number.isFinite)||Math.min(entry,target,stop)<=0||Math.min(fee,slip,spread,funding)<0)throw Error('INVALID_EXECUTION_COST');
  const sign=side==='UP'?1:-1,friction=(slip+spread/2)/10000;
  const pnl=(barrier:number)=>{const exit=barrier*(1-sign*friction);return sign*(exit-entry)-entry*fee/10000-exit*fee/10000-entry*funding/10000;};
  const reward=pnl(target),risk=-pnl(stop);
  return {model:'FILL_TO_FILL_V1',net_reward_bps:Math.max(0,reward/entry*10000),net_risk_bps:risk/entry*10000,economic_rr:risk>0?Math.max(0,reward)/risk:0};
}

// A prior forecast is not an executed trade. Record a fresh, immutable entry evaluation.
// The caller still holds the lease and checks the durable episode-entry ledger.
export async function refreshEntryEvaluation(c:CandidateResult,version:number,at:number){
  if(!c.canEnter||!c.decision)return null;
  const parent=c.occurrence,id=await hash(['entry-evaluation-v1',parent,version].join('|'));
  c.occurrence=id;c.thesis={...c.thesis,thesis_id:id,occurrence_id:id,parent_forecast_id:parent};
  const evidence:J={...c.decision.evidence,parent_forecast_id:parent,evaluation_kind:'ENTRY_REEVALUATION'};
  c.decision={...c.decision,occurrence_id:id,decision_at:new Date(at).toISOString(),evidence};
  return c.decision;
}
