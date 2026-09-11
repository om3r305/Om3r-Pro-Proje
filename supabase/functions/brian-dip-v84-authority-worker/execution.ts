import { EXECUTION_MODEL_VERSION } from "../_shared/dip_v84_authority_contract.ts";
import type { J, Position, Runtime } from "../_shared/dip_v84_contract.ts";
import type { Market } from "../brian-dip-v84-worker/market.ts";

function applyClose(rt:Runtime,p:Position,exit:number,recordedAt:number,reason:string,extra:J={}):J{
  if(p.side!=="LONG")throw Error("V843_SHORT_POSITION_FORBIDDEN");
  const feeClose=exit*p.qty*p.fee_bps/10000,gross=(exit-p.entry)*p.qty;
  const funding=p.funding_accrued??0;if(!Number.isFinite(funding))throw Error("V843_INVALID_FUNDING");
  const net=gross-p.fees_open-feeClose+funding;
  rt.cash+=p.margin+gross-feeClose+funding;rt.realized+=net;rt.trades++;
  if(net>0)rt.wins++;else if(net<0)rt.losses++;
  rt.lastClosedAt=recordedAt;rt.pos=null;
  return{event_kind:"SELL",position_id:p.position_id,occurrence_id:p.thesis_id,episode_id:p.episode_id,price:exit,entry_price:p.entry,exit_price:exit,quantity:p.qty,notional:p.notional,fees:p.fees_open+feeClose,funding_cashflow:funding,realized_pnl:net,cash_after:rt.cash,equity_after:rt.cash,metadata:{server_v84:true,brian_authority:true,long_only:true,side:"LONG",setup:p.setup,regime:p.regime,release_id:p.release_id,calibration_family_id:p.calibration_family_id,funding_cashflow:funding,funding_settlements:p.funding_settlements??[],funding_model:"SETTLED_RATE_MARK_PRICE",exit_reason:reason,execution_model:EXECUTION_MODEL_VERSION,fee_bps:p.fee_bps,expected_exit_spread_bps:p.expected_exit_spread_bps,expected_exit_slippage_bps:p.expected_exit_slippage_bps,leverage:p.leverage,margin:p.margin,...extra}};
}

function executableLongExit(p:Position,market:Market):number{
  const slip=p.expected_exit_slippage_bps/10000;
  return market.book.bid*(1-slip);
}

export function closeOnStop(rt:Runtime,barrier:number,recordedAt:number,extra:J={}):J|null{
  const p=rt.pos;if(!p)return null;if(p.side!=="LONG")throw Error("V843_SHORT_POSITION_FORBIDDEN");
  if(!(barrier>0)||!Number.isFinite(barrier))throw Error("V843_INVALID_STOP_BARRIER");
  const friction=(p.expected_exit_spread_bps/2+p.expected_exit_slippage_bps)/10000;
  return applyClose(rt,p,barrier*(1-friction),recordedAt,"INVALIDATION_FIRST",{execution_barrier_price:barrier,...extra});
}

export function closeOnBrianSell(rt:Runtime,market:Market,recordedAt:number,thesis:J,sellVotes:number):J|null{
  const p=rt.pos;if(!p)return null;if(p.side!=="LONG")throw Error("V843_SHORT_POSITION_FORBIDDEN");
  const action=String(thesis.authority_action||"HOLD");
  if(action!=="SELL")return null;
  const scores=(thesis.authority_scores??{}) as J;
  const strength=String(scores.sell_strength||"SOFT");
  if(strength!=="STRONG"&&sellVotes<2)return null;
  const confidence=Number(thesis.authority_confidence||0);
  return applyClose(rt,p,executableLongExit(p,market),recordedAt,strength==="STRONG"?"BRIAN_STRONG_SELL":"BRIAN_STATEFUL_SELL",{authority_action:action,authority_confidence:confidence,authority_reason:thesis.authority_reason,authority_scores:scores,sell_strength:strength,sell_reason:scores.sell_reason??null,sell_votes:sellVotes,market_bid:market.book.bid,market_ask:market.book.ask,spread_applied_via_executable_side:true,execution_ambiguous_loss:false});
}

export function closeOnMaxHold(rt:Runtime,market:Market,recordedAt:number):J|null{
  const p=rt.pos;if(!p)return null;if(p.side!=="LONG")throw Error("V843_SHORT_POSITION_FORBIDDEN");
  return applyClose(rt,p,executableLongExit(p,market),recordedAt,"MAX_HOLD_STATEFUL_EXIT",{market_bid:market.book.bid,market_ask:market.book.ask});
}