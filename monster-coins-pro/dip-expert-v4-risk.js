function v4PairLocked(sym){return Number(v4PairGuard[sym]||0)>v4Now();}
function v4GlobalLocked(){return v4SessionLossLockUntil>v4Now();}
function v4RecordOutcome(sym,pnlValue,reason){
  const o={t:v4Now(),sym,pnl:Number(pnlValue||0),reason};v4ClosedOutcomes.push(o);while(v4ClosedOutcomes.length>30)v4ClosedOutcomes.shift();
  const recentPair=v4ClosedOutcomes.filter(x=>x.sym===sym).slice(-3),losses=recentPair.filter(x=>x.pnl<=0).length;if(recentPair.length>=3&&losses>=2)v4PairGuard[sym]=v4Now()+V4_PAIR_LOCK_MS;
  const r=v4ClosedOutcomes.slice(-10),bad=r.filter(x=>x.pnl<0&&['HARD_5M_STRUCTURE_STOP','FLOW_INVALIDATION','TIME_EXIT','HTF_INVALIDATION'].includes(x.reason)).length;if(r.length>=5&&bad>=3)v4SessionLossLockUntil=Math.max(v4SessionLossLockUntil,v4Now()+V4_GLOBAL_LOCK_MS);
}
function v4EntryGuard(sym,ctx){
  if(v4NeedsRestart)return'RESTART_REQUIRED';if(v4CloudFault)return'CLOUD_PERSIST_FAULT';if(v4PairLocked(sym))return'PAIR_COOLDOWN';if(v4GlobalLocked())return'GLOBAL_STOP_GUARD';
  const m=metrics();if(m.equity<=book.start*.96){v4SessionLossLockUntil=Math.max(v4SessionLossLockUntil,v4Now()+V4_DD_LOCK_MS);return'SESSION_DD_GUARD';}
  if(v4OpenCount()>=V4_MAX_OPEN)return'MAX_OPEN';if(v4Heat()>=V4_MAX_HEAT)return'PORTFOLIO_HEAT';if(!ctx||ctx.bk.ageMs>3500)return'STALE_BOOK';if(ctx.bk.spreadBps>V4_MAX_ENTRY_SPREAD_BPS)return'SPREAD_TOO_WIDE';
  if(ctx.edgeRatio<V4_COST_EDGE_MULT)return'EDGE_BELOW_COST';return null;
}
function v4TradePlan(ctx,side){
  const costPct=ctx.roundTripCostBps/100;
  const structure=side==='LONG'&&ctx.f5.swingLow?Math.max(0,v4Pct(ctx.bk.mid,ctx.f5.swingLow)):side==='SHORT'&&ctx.f5.swingHigh?Math.max(0,v4Pct(ctx.f5.swingHigh,ctx.bk.mid)):0;
  let stopPct=v4Clamp(Math.max(ctx.f5.atrPct*1.05,costPct*2.5,structure*.55),.35,1.60);
  let targetPct=Math.max(costPct*3.0,ctx.f5.atrPct*1.45,stopPct*V4_MIN_RR);
  targetPct=v4Clamp(targetPct,.55,2.75);
  if(targetPct<stopPct*V4_MIN_RR)return null;
  const expectedPct=ctx.expectedMoveBps/100;if(expectedPct<Math.max(costPct*V4_COST_EDGE_MULT,targetPct*.82))return null;
  const eq=Math.max(1,metrics().equity),risk=eq*V4_RISK_PER_TRADE,raw=risk/(stopPct/100),notional=Math.min(raw,eq*V4_MAX_POSITION,book.cash*.95);
  if(notional<Math.max(12,eq*.035))return null;
  return{side,stopPct,targetPct,costPct,notional,trailArmPct:Math.max(costPct*1.65,targetPct*.62),trailGapPct:v4Clamp(Math.max(ctx.f5.atrPct*.38,costPct*.45),.10,.55),maxHoldMs:side==='SHORT'?V4_SHORT_HOLD_MS:V4_LONG_HOLD_MS};
}
function v4Fill(sym,venue,side,notional,isExit=false){
  const bk=v4BookMetrics(sym,venue),slipCfg=Number(book.cfg?.slippage_bps??1),impact=v4ImpactBps(sym,venue,notional),slipBps=Math.max(slipCfg,impact),mid=bk.mid||Number(live[sym]);
  let px=mid;
  if(side==='LONG')px=(bk.ask||mid)*(1+slipBps/10000);
  else px=(bk.bid||mid)*(1-slipBps/10000);
  if(isExit){if(side==='LONG')px=(bk.bid||mid)*(1-slipBps/10000);else px=(bk.ask||mid)*(1+slipBps/10000);}
  return{px,slipBps,spreadBps:bk.spreadBps};
}
