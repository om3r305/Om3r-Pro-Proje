import { EXECUTION_MODEL_VERSION } from "../_shared/dip_v84_authority_contract.ts";
import type { J, Position, Resolution, Runtime } from "../_shared/dip_v84_contract.ts";
import type { Market } from "../brian-dip-v84-worker/market.ts";

function exitFromBarrier(pos:Position,barrier:number):number{
  if(!(barrier>0)||!Number.isFinite(barrier))throw Error("V841_INVALID_EXIT_BARRIER");
  const friction=(pos.expected_exit_spread_bps/2+pos.expected_exit_slippage_bps)/10000;
  return barrier*(pos.side==="LONG"?1-friction:1+friction);
}
function executionBarrier(pos:Position,resolution:Resolution):number{
  if(resolution.reason==="TARGET_FIRST")return pos.target;
  if(resolution.reason==="INVALIDATION_FIRST"||resolution.reason==="AMBIGUOUS")return pos.stop;
  return resolution.price;
}
function applyClose(rt:Runtime,p:Position,exit:number,recordedAt:number,reason:string,extra:J={}):J{
  const feeClose=exit*p.qty*p.fee_bps/10000,gross=(p.side==="LONG"?exit-p.entry:p.entry-exit)*p.qty;
  const funding=p.funding_accrued??0;if(!Number.isFinite(funding))throw Error("V841_INVALID_FUNDING");
  const net=gross-p.fees_open-feeClose+funding;
  rt.cash+=p.margin+gross-feeClose+funding;rt.realized+=net;rt.trades++;
  if(net>0)rt.wins++;else if(net<0)rt.losses++;
  rt.lastClosedAt=recordedAt;rt.pos=null;
  return{event_kind:p.side==="LONG"?"SELL":"SHORT_CLOSE",position_id:p.position_id,occurrence_id:p.thesis_id,episode_id:p.episode_id,price:exit,entry_price:p.entry,exit_price:exit,quantity:p.qty,notional:p.notional,fees:p.fees_open+feeClose,funding_cashflow:funding,realized_pnl:net,cash_after:rt.cash,equity_after:rt.cash,metadata:{server_v84:true,brian_authority:true,side:p.side,setup:p.setup,regime:p.regime,release_id:p.release_id,calibration_family_id:p.calibration_family_id,funding_cashflow:funding,funding_settlements:p.funding_settlements??[],funding_model:"SETTLED_RATE_MARK_PRICE",exit_reason:reason,execution_model:EXECUTION_MODEL_VERSION,fee_bps:p.fee_bps,expected_exit_spread_bps:p.expected_exit_spread_bps,expected_exit_slippage_bps:p.expected_exit_slippage_bps,leverage:p.leverage,margin:p.margin,...extra}};
}
export function closePosition(rt:Runtime,resolution:Resolution,recordedAt:number):J|null{
  const p=rt.pos;if(!p||["PENDING","INDETERMINATE"].includes(resolution.reason))return null;
  const triggerPrice=resolution.price,barrier=executionBarrier(p,resolution),exit=exitFromBarrier(p,barrier);
  return applyClose(rt,p,exit,recordedAt,resolution.reason==="AMBIGUOUS"?"AMBIGUOUS_CONSERVATIVE_STOP":resolution.reason,{resolution,path_trigger_price:triggerPrice,execution_barrier_price:barrier,execution_ambiguous_loss:resolution.reason==="AMBIGUOUS"});
}
export function closeOnBrianReversal(rt:Runtime,market:Market,recordedAt:number,thesis:J):J|null{
  const p=rt.pos;if(!p)return null;
  const action=String(thesis.authority_action||"WAIT"),opposite=p.side==="LONG"?action==="SHORT":action==="LONG";
  const confidence=Number(thesis.authority_confidence||0);
  if(!opposite||!Number.isFinite(confidence)||confidence<0.62)return null;
  // bid/ask is already executable-side, so spread is not subtracted again; only exit slippage is applied once.
  const slip=p.expected_exit_slippage_bps/10000,exit=p.side==="LONG"?market.book.bid*(1-slip):market.book.ask*(1+slip);
  return applyClose(rt,p,exit,recordedAt,"BRIAN_REVERSAL_EXIT",{authority_action:action,authority_confidence:confidence,authority_reason:thesis.authority_reason,authority_scores:thesis.authority_scores,market_bid:market.book.bid,market_ask:market.book.ask,spread_applied_via_executable_side:true,execution_ambiguous_loss:false});
}
