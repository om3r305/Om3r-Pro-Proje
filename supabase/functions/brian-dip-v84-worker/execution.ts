import { EXECUTION_MODEL_VERSION, type J, type Position, type Resolution, type Runtime } from "../_shared/dip_v84_contract.ts";

function exitFromBarrier(pos:Position,barrier:number):number{
  if(!(barrier>0)||!Number.isFinite(barrier))throw Error("V84_INVALID_EXIT_BARRIER");
  // Package 1 authoritative model freezes expected exit spread/slippage at open.
  // Barrier is a structural/reference price: executable-side conversion once + slippage once.
  const friction=(pos.expected_exit_spread_bps/2+pos.expected_exit_slippage_bps)/10000;
  return barrier*(pos.side==="LONG"?1-friction:1+friction);
}

export function closePosition(rt:Runtime,resolution:Resolution,recordedAt:number):J|null{
  const p=rt.pos;
  if(!p||["PENDING","INDETERMINATE"].includes(resolution.reason))return null;
  const barrier=resolution.reason==="AMBIGUOUS"?p.stop:resolution.price;
  const exit=exitFromBarrier(p,barrier);
  const feeClose=exit*p.qty*p.fee_bps/10000;
  const gross=(p.side==="LONG"?exit-p.entry:p.entry-exit)*p.qty;
  const funding=p.funding_accrued??0;if(!Number.isFinite(funding))throw Error("V84_INVALID_FUNDING");
  const net=gross-p.fees_open-feeClose+funding;
  rt.cash+=p.margin+gross-feeClose+funding;rt.realized+=net;rt.trades++;
  if(net>0)rt.wins++;else if(net<0)rt.losses++;
  rt.lastClosedAt=recordedAt;rt.pos=null;
  return{
    event_kind:p.side==="LONG"?"SELL":"SHORT_CLOSE",
    position_id:p.position_id,occurrence_id:p.thesis_id,episode_id:p.episode_id,
    price:exit,entry_price:p.entry,exit_price:exit,quantity:p.qty,notional:p.notional,
    fees:p.fees_open+feeClose,funding_cashflow:funding,realized_pnl:net,cash_after:rt.cash,equity_after:rt.cash,
    metadata:{server_v84:true,side:p.side,setup:p.setup,regime:p.regime,release_id:p.release_id,calibration_family_id:p.calibration_family_id,funding_cashflow:funding,funding_settlements:p.funding_settlements??[],funding_model:"SETTLED_RATE_MARK_PRICE",exit_reason:resolution.reason==="AMBIGUOUS"?"AMBIGUOUS_CONSERVATIVE_STOP":resolution.reason,resolution,execution_model:EXECUTION_MODEL_VERSION,fee_bps:p.fee_bps,expected_exit_spread_bps:p.expected_exit_spread_bps,expected_exit_slippage_bps:p.expected_exit_slippage_bps,leverage:p.leverage,margin:p.margin,execution_ambiguous_loss:resolution.reason==="AMBIGUOUS"}
  };
}
