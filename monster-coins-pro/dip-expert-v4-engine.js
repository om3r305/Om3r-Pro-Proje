function v4Open(st,ctx,side,mode){
  const guard=v4EntryGuard(st.symbol,ctx);if(guard){st.v4.lastVeto=guard;return v4Reject(guard)};
  const plan=v4TradePlan(ctx,side);if(!plan){st.v4.lastVeto='PAYOFF_NOT_GOOD_ENOUGH';return v4Reject('PAYOFF_NOT_GOOD_ENOUGH')};
  const venue=side==='SHORT'?'USDM_PERP':'SPOT';if(side==='SHORT'&&!v4FuturesSymbols.has(st.symbol)){st.v4.lastVeto='NO_PERP_MARKET';return v4Reject('NO_PERP_MARKET')};
  const c=book.cfg||cfgUi(),feeSideBps=side==='SHORT'?5:Number(c.fee_bps||10),fill=v4Fill(st.symbol,venue,side,plan.notional,false),entry=fill.px,entryFee=plan.notional*(feeSideBps/10000),required=plan.notional+entryFee;
  if(book.cash<required){st.v4.lastVeto='NO_CASH';return v4Reject('NO_CASH')};
  const qty=plan.notional/entry,stop=side==='LONG'?entry*(1-plan.stopPct/100):entry*(1+plan.stopPct/100),target=side==='LONG'?entry*(1+plan.targetPct/100):entry*(1-plan.targetPct/100);
  book.cash-=required;st.pos={v4:true,side,venue,entry,qty,margin:plan.notional,notional:plan.notional,exposure:plan.notional,leverage:1,entryFee,at:iso(),mode,stop,stopPct:plan.stopPct,target,targetPct:plan.targetPct,costPct:plan.costPct,trailArmPct:plan.trailArmPct,trailGapPct:plan.trailGapPct,trailPrice:null,bestPrice:entry,maxHoldMs:plan.maxHoldMs,entrySwingLow:ctx.f5.swingLow,entrySwingHigh:ctx.f5.swingHigh,entryHtf:side==='LONG'?ctx.htfLong:ctx.htfShort,entryBeta:ctx.beta};
  st.armed=false;st.v4.phase='POSITION';st.v4.armSide=null;st.lastAction=side==='LONG'?'LONG':'SHORT';v4Funnel.entered++;
  event(side==='LONG'?'BUY':'SHORT_OPEN',st.symbol,ctx.bk.mid,{entry_price:entry,quantity:qty,notional:plan.notional,fees:entryFee,metadata:{expert_v4:true,side,venue,mode,margin:plan.notional,leverage:1,stop_price:stop,stop_pct:plan.stopPct,target_price:target,target_pct:plan.targetPct,cost_pct:plan.costPct,edge_ratio:ctx.edgeRatio,spread_bps:ctx.bk.spreadBps,impact_bps:ctx.impact,ofi:ctx.flow.ofi,book_pressure:ctx.bk.pressure,beta_btc:ctx.beta,idio_z:ctx.idioZ,htf:side==='LONG'?ctx.htfLong:ctx.htfShort}});
  render();return true;
}
function v4Close(st,ctx,reason){
  const q=st.pos;if(!q)return;const side=q.side||'LONG',venue=q.venue||'SPOT',feeSideBps=venue==='USDM_PERP'?5:Number(book.cfg?.fee_bps||10),fill=v4Fill(st.symbol,venue,side,Number(q.notional||q.margin||0),true),exit=fill.px;
  const raw=side==='LONG'?q.qty*(exit-q.entry):q.qty*(q.entry-exit),exitNotional=Math.max(0,q.qty*exit),exitFee=exitNotional*(feeSideBps/10000),r=raw-Number(q.entryFee||0)-exitFee,margin=Number(q.margin||q.notional||0);
  book.cash+=Math.max(0,margin+raw-exitFee);book.realized+=r;book.trades++;r>0?book.wins++:book.losses++;st.realized=Number(st.realized||0)+r;st.pos=null;st.v4.phase='WATCH';st.v4.armSide=null;st.v4.armAt=0;st.v4.lastVeto=reason;st.lastAction='COOLDOWN';v4RecordOutcome(st.symbol,r,reason);
  event(side==='LONG'?'SELL':'SHORT_CLOSE',st.symbol,ctx?.bk?.mid||Number(live[st.symbol]),{entry_price:q.entry,exit_price:exit,quantity:q.qty,notional:q.notional,fees:Number(q.entryFee||0)+exitFee,realized_pnl:r,metadata:{expert_v4:true,side,venue,mode:q.mode,exit_reason:reason,hold_min:(v4Now()-new Date(q.at).getTime())/60000,stop_pct:q.stopPct,target_pct:q.targetPct,cost_pct:q.costPct,spread_bps:fill.spreadBps,slippage_bps:fill.slipBps}});
  render();
}
function v4Manage(st,ctx){
  const q=st.pos;if(!q||!ctx)return;const p=ctx.bk.mid||Number(live[st.symbol]);if(!(p>0))return;const side=q.side||'LONG',move=side==='LONG'?v4Pct(p,q.entry):v4Pct(q.entry,p),hold=v4Now()-new Date(q.at).getTime();q.bestPrice=side==='LONG'?Math.max(Number(q.bestPrice||p),p):Math.min(Number(q.bestPrice||p),p);
  if((side==='LONG'&&p<=q.stop)||(side==='SHORT'&&p>=q.stop))return v4Close(st,ctx,'HARD_5M_STRUCTURE_STOP');
  const bars5=v4Closed(st.symbol,'5m'),last5=bars5.at(-1),prevSwingLow=q.entrySwingLow,prevSwingHigh=q.entrySwingHigh;
  if(side==='LONG'&&last5&&prevSwingLow&&last5.c<prevSwingLow&&move<q.costPct*.5)return v4Close(st,ctx,'HTF_INVALIDATION');
  if(side==='SHORT'&&last5&&prevSwingHigh&&last5.c>prevSwingHigh&&move<q.costPct*.5)return v4Close(st,ctx,'HTF_INVALIDATION');
  const flowBad=side==='LONG'?(ctx.flow.ofi<-.28&&ctx.bk.pressure<.88):(ctx.flow.ofi>.28&&ctx.bk.pressure>1.14);
  if(move< -Math.max(q.costPct*.55,ctx.f5.atrPct*.30)&&flowBad)return v4Close(st,ctx,'FLOW_INVALIDATION');
  if(move>=Number(q.trailArmPct||999)){
    const costFloor=Number(q.costPct||0)*1.12,gap=Number(q.trailGapPct||.2);
    if(side==='LONG'){const floor=q.entry*(1+costFloor/100),trail=q.bestPrice*(1-gap/100);q.trailPrice=Math.max(Number(q.trailPrice||0),floor,trail);}
    else{const ceil=q.entry*(1-costFloor/100),trail=q.bestPrice*(1+gap/100);q.trailPrice=q.trailPrice==null?Math.min(ceil,trail):Math.min(q.trailPrice,ceil,trail);}
  }
  if(q.trailPrice!=null&&((side==='LONG'&&p<=q.trailPrice)||(side==='SHORT'&&p>=q.trailPrice)))return v4Close(st,ctx,'PROFIT_TRAIL_V4');
  if((side==='LONG'&&p>=q.target)||(side==='SHORT'&&p<=q.target))return v4Close(st,ctx,'EXPERT_TARGET_V4');
  if(hold>=Number(q.maxHoldMs||V4_LONG_HOLD_MS))return v4Close(st,ctx,'TIME_EXIT');
}

function v4ArmLong(st,ctx,p){
  const trigger=v4Clamp(ctx.f5.atrPct*.55,.35,1.80),swingHigh=ctx.f5.swingHigh||p,drop=Math.max(0,v4Pct(p,swingHigh)*-1);
  const idioDip=ctx.idioZ<=-1.05,deepEnough=drop>=trigger;
  if(!deepEnough&&!idioDip)return false;
  st.v4.phase='ARM_LONG';st.v4.armSide='LONG';st.v4.armLow=p;st.v4.armHigh=swingHigh;st.v4.armAt=v4Now();st.dip=p;st.armed=true;st.lastAction='DIP_HUNT';
  event('DIP_ARMED',st.symbol,p,{metadata:{expert_v4:true,side:'LONG',trigger_pct:trigger,drop_pct:drop,idio_z:ctx.idioZ,beta_btc:ctx.beta,atr5_pct:ctx.f5.atrPct,edge_ratio:ctx.edgeRatio}});return true;
}
function v4TryLong(st,ctx,p){
  if(st.v4.phase!=='ARM_LONG')return false;if(v4Now()-Number(st.v4.armAt||0)>12*60*1000){st.v4.phase='WATCH';st.armed=false;st.v4.lastVeto='STALE_DIP_ARM';return v4Reject('STALE_DIP_ARM')}if(p<Number(st.v4.armLow||p))st.v4.armLow=p;
  const rebound=v4Pct(p,Number(st.v4.armLow||p)),need=v4Clamp(ctx.f5.atrPct*.30,.12,.70),flowOk=ctx.flow.ofi>=.10&&ctx.flow.ratio>=1.15,bookOk=ctx.bk.pressure>=1.05,htfOk=ctx.htfLong>-.28,btcOk=ctx.btcLongRisk>-.38,volumeOk=ctx.f1.volRel>=.75;
  if(rebound>ctx.f5.atrPct*1.10){st.v4.phase='WATCH';st.armed=false;st.v4.lastVeto='SKIP_CHASE';event('SKIP_CHASE',st.symbol,p,{metadata:{expert_v4:true,rebound_pct:rebound,atr5_pct:ctx.f5.atrPct}});return false;}
  if(rebound<need)return false;if(!btcOk){st.v4.lastVeto='BTC_RISK_LONG';return v4Reject('BTC_RISK_LONG')}if(!htfOk){st.v4.lastVeto='HTF_LONG_MISMATCH';return v4Reject('HTF_LONG_MISMATCH')}if(!flowOk){st.v4.lastVeto='NO_BUY_FLOW';return v4Reject('NO_BUY_FLOW')}if(!bookOk){st.v4.lastVeto='NO_BID_PRESSURE';return v4Reject('NO_BID_PRESSURE')}if(!volumeOk){st.v4.lastVeto='WEAK_VOLUME';return v4Reject('WEAK_VOLUME')}
  return v4Open(st,ctx,'LONG','V4_DIP_RECLAIM');
}
function v4TryShort(st,ctx,p){
  if(!v4FuturesSymbols.has(st.symbol))return false;
  const pctx=v4Context(st.symbol,'USDM_PERP');if(!pctx)return false;
  const guard=v4EntryGuard(st.symbol,pctx);if(guard){st.v4.lastVeto=guard;return v4Reject(guard)};
  const recentLow=pctx.f5.swingLow||p,breakNeed=v4Clamp(pctx.f5.atrPct*.18,.08,.45);
  const regime=pctx.htfShort>.18&&pctx.f15.trend<-.08&&pctx.f1h.trend<-.05&&pctx.btcLongRisk<.48;
  const flow=pctx.flow.ofi<=-.12&&pctx.flow.ratio<=.87,book=pctx.bk.pressure<=.95,priceBreak=p<recentLow*(1-breakNeed/100);
  // Strongly negative funding means shorts pay longs; avoid entering a crowded/expensive short shadow.
  const fundingOk=pctx.fundingRate>-0.0005;
  if(!(regime&&flow&&book&&priceBreak&&fundingOk)){if(!fundingOk)st.v4.lastVeto='SHORT_FUNDING_COST';return false;}
  return v4Open(st,pctx,'SHORT','V4_DOWNTREND_BREAK');
}
function v4Evaluate(sym,p){
  if(!running||v4Booting||v4NeedsRestart||!sid||!v4Universe.includes(sym))return;const now=v4Now();if(now-Number(v4EvalAt[sym]||0)<240)return;v4EvalAt[sym]=now;v4Funnel.evaluated++;
  const st=v4Ensure(sym),ctx=v4Context(sym,st.pos?.venue||'SPOT');if(!ctx){st.v4.lastVeto='WAIT_NATIVE_TF';return}
  if(st.pos)return v4Manage(st,ctx);
  const general=v4EntryGuard(sym,ctx);if(general){st.v4.lastVeto=general;return v4Reject(general)};
  const mid=ctx.bk.mid||Number(p);if(!(mid>0))return;
  if(st.v4.phase==='ARM_LONG'){if(v4TryLong(st,ctx,mid))return}else v4ArmLong(st,ctx,mid);
  if(!st.pos&&st.v4.phase!=='ARM_LONG')v4TryShort(st,ctx,mid);
}

tick=function(sym,p){p=Number(p);if(!(p>0))return;live[sym]=p;const st=v4Ensure(sym);st.last=p;v4Evaluate(sym,p);};
