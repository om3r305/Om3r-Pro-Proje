// Final V4 execution override: model book-walk impact relative to touch, not mid, and use the correct side of book.
v4ImpactBps=function(sym,venue='SPOT',notional=100,bookSide='BUY'){
  const x=(venue==='USDM_PERP'?v4PerpBook[sym]:v4SpotBook[sym])||{},levels=bookSide==='SELL'?(x.bids||[]):(x.asks||[]);
  if(!levels.length)return 2;const touch=Number(levels[0][0]);if(!(touch>0))return 2;
  let rem=Math.max(1,notional),spent=0,qty=0;
  for(const [p,q] of levels){const cap=p*q,take=Math.min(rem,cap);spent+=take;qty+=take/p;rem-=take;if(rem<=0)break;}
  if(rem>0)return Math.min(50,8+rem/notional*25);
  const vwap=qty?spent/qty:touch;return Math.max(0,bookSide==='SELL'?(touch/vwap-1)*10000:(vwap/touch-1)*10000);
};
v4Fill=function(sym,venue,side,notional,isExit=false){
  const bk=v4BookMetrics(sym,venue),slipCfg=Number(book.cfg?.slippage_bps??1),bookSide=(!isExit&&side==='LONG')||(isExit&&side==='SHORT')?'BUY':'SELL',impact=v4ImpactBps(sym,venue,notional,bookSide),slipBps=Math.max(slipCfg,impact),mid=bk.mid||Number(live[sym]);
  let px=mid;if(side==='LONG')px=(bk.ask||mid)*(1+slipBps/10000);else px=(bk.bid||mid)*(1-slipBps/10000);if(isExit){if(side==='LONG')px=(bk.bid||mid)*(1-slipBps/10000);else px=(bk.ask||mid)*(1+slipBps/10000);}return{px,slipBps,spreadBps:bk.spreadBps};
};

note=function(e){const m=e.metadata||{};if(m.expert_v4){if(e.event_kind==='ENGINE_START')return`V4 · ${m.universe_size||12} market · max ${m.max_open||2} · microstructure`;if(e.event_kind==='DIP_ARMED')return`${m.side||'LONG'} arm · ATR5 ${Number(m.atr5_pct||0).toFixed(2)}% · idioZ ${Number(m.idio_z||0).toFixed(2)} · cost edge ${Number(m.edge_ratio||0).toFixed(1)}x`;if(e.event_kind==='BUY'||e.event_kind==='SHORT_OPEN')return`${m.mode} · ${m.venue} · ${cash(m.margin)} · stop ${Number(m.stop_pct||0).toFixed(2)}% · TP ${Number(m.target_pct||0).toFixed(2)}% · cost ${Number(m.cost_pct||0).toFixed(2)}%`;if(e.event_kind==='SELL'||e.event_kind==='SHORT_CLOSE')return`${m.exit_reason||'EXIT'} · ${Number(m.hold_min||0).toFixed(1)} dk · ${m.venue||''}`;if(e.event_kind==='INFO'&&m.info==='ENTRY_FUNNEL')return`Funnel ${m.entered}/${m.evaluated} · ${Object.entries(m.rejected||{}).map(([k,v])=>`${k}:${v}`).join(' ')}`;}return _v4PrevNote(e);};

function v4RadarRow(sym){
  const st=v4Ensure(sym),ctx=v4Context(sym,st.pos?.venue||'SPOT'),rank=v4Ranks[sym]||{},p=ctx?.bk?.mid||Number(live[sym]||0),pos=st.pos;
  let state=pos?(pos.side==='SHORT'?'SHORT':'LONG'):(st.v4.phase==='ARM_LONG'?'DIP HUNT':'WATCH'),klass=pos?(pos.side==='SHORT'?'skip':'long'):(st.v4.phase==='ARM_LONG'?'hunt':'');
  const opp=ctx?Math.round(v4Clamp(35+18*Math.min(3,ctx.edgeRatio)+14*Math.abs(ctx.htfLong)+10*Math.abs(ctx.flow.ofi)+8*Math.abs(ctx.idioZ),0,100)):Math.round(rank.score||0),cost=ctx?ctx.roundTripCostBps:0;
  const veto=pos?`${pos.mode} · TP ${Number(pos.targetPct||0).toFixed(2)}% / SL ${Number(pos.stopPct||0).toFixed(2)}%`:(st.v4.lastVeto||'WAIT');
  return`<div class="coinCard ${sym===selected?'selected':''}" data-s="${v4Esc(sym)}"><div><div class="coinTop"><span class="coinSymbol">${v4Esc(sym.replace('USDT',''))}</span><span class="coinState ${klass}">${state}</span></div><div class="coinMeta">#${v4Universe.indexOf(sym)+1} · edge ${ctx?ctx.edgeRatio.toFixed(1):'—'}x · cost ${cost?cost.toFixed(0):'—'}bps · ATR5 ${ctx?ctx.f5.atrPct.toFixed(2):'—'}%<br>OFI ${ctx?ctx.flow.ofi.toFixed(2):'—'} · book ${ctx?ctx.bk.pressure.toFixed(2):'—'} · BTCβ ${ctx?ctx.beta.toFixed(2):'—'} · ${v4Esc(veto)}</div><div class="scoreBar"><i style="width:${opp}%"></i></div><div class="scoreLabel">V4 Opportunity ${opp}/100 · HTF ${ctx?ctx.htfLong.toFixed(2):'—'}</div></div><div class="coinRight"><div class="coinPrice">${price(p)}</div>${st.realized?`<div class="coinMove ${st.realized>=0?'pos':'neg'}">${pnl(st.realized)}</div>`:''}</div></div>`;
}
renderRadar=function(){
  if(!$('coinRadar'))return;const sy=[...v4Universe].sort((a,b)=>{const pa=states[a]?.pos?1:0,pb=states[b]?.pos?1:0;if(pa!==pb)return pb-pa;const ca=v4Context(a,states[a]?.pos?.venue||'SPOT'),cb=v4Context(b,states[b]?.pos?.venue||'SPOT');return(Number(cb?.edgeRatio||v4Ranks[b]?.score||0)-Number(ca?.edgeRatio||v4Ranks[a]?.score||0))});$('coinRadar').innerHTML=sy.map(v4RadarRow).join('')||'<div class="row">V4 evreni yükleniyor…</div>';document.querySelectorAll('#coinRadar [data-s]').forEach(x=>x.onclick=()=>{selected=x.dataset.s;renderRadar();_v4PrevDraw()});if($('radarStatus'))$('radarStatus').textContent=v4NeedsRestart?'RESTART':running?'V4 LIVE':'WAIT';
};
renderKpi=function(){_v4PrevRenderKpi();if($('kpiEngine')){$('kpiEngine').textContent=v4NeedsRestart?'V4 RESTART':running?'EXPERT V4':'V4 IDLE';$('kpiEngine').className=`value ${running?'pos':'amber'}`}if($('kpiEngineMeta'))$('kpiEngineMeta').textContent=`AUTO ${v4Universe.length||V4_UNIVERSE_SIZE} · max ${V4_MAX_OPEN} · micro + HTF + cost gate`;if($('kpiOpenMeta'))$('kpiOpenMeta').textContent=`max ${V4_MAX_OPEN} · heat ${(v4Heat()*100).toFixed(0)}%`;};
render=function(){_v4PrevRender();v4UiPatch();renderRadar();renderKpi();};

function v4UiPatch(){
  const h=document.querySelector('.desktopTitle h1');if(h)h.textContent='Aggressive Market Expert V4';const mb=document.querySelector('.mobileBrand b');if(mb)mb.textContent='Dip Expert V4';const sub=document.querySelector('.desktopTitle .sub');if(sub)sub.textContent='Native 1m/5m/15m/1h · BTC regime · order-flow · book pressure · cost-aware shadow execution';
  const banner=document.querySelector('.dipBanner>div:first-child');if(banner)banner.innerHTML='<strong>EXPERT V4 · SHADOW ONLY</strong> · Dip tek başına giriş değildir. V4 maliyet, BTC rejimi, 5m/15m/1h yapı, taker akışı ve order-book baskısını birlikte görmeden pozisyon açmaz.';
  const badge=document.querySelector('.expertModeCard .badge');if(badge)badge.textContent='MICROSTRUCTURE EXPERT V4';const expert=document.querySelector('.expertModeCard');if(expert){const b=expert.querySelector('b');if(b)b.textContent='Rejim + order flow + gerçek spread/depth + risk sizing';const s=expert.querySelector('small');if(s)s.textContent='12 dinamik market · max 2 pozisyon · 1x validation · BTC beta · 5m structure';}
  if($('startBtn'))$('startBtn').textContent='▶ Expert V4 Başlat';if($('restartBtn'))$('restartBtn').textContent='↻ V4 Temiz Kasa ile Restart';
  const rule=document.querySelector('.dipRuleLine');if(rule)rule.innerHTML='V4: <b>dip = radar</b> → native 5m ATR ile yeterli hareket → BTC/15m/1h rejim → <b>OFI + bid/ask pressure</b> ile dönüş onayı → beklenen move en az <b>2.5× gerçekçi round-trip cost</b> → risk tabanlı size → max 2 pozisyon → 5m structure / flow / hard-time / target-trail çıkışı.';
  const title=document.querySelector('#watchlist .title');if(title)title.textContent='Coin Radar · Expert V4';
  const logic=document.querySelector('.logicSteps');if(logic)logic.innerHTML='<div><b>1</b><span>Binance evreninden likit/hareketli marketleri tarar; maliyet başına hareketi düşük olanı eler.</span></div><div><b>2</b><span>Native 1m + 5m + 15m + 1h ve BTC rejimi/beta ile altcoin hareketini piyasa dumpından ayırır.</span></div><div><b>3</b><span>Dip sadece adaydır; aggTrade order-flow ve depth5 bid/ask pressure dönüşü doğrulamadan LONG yok.</span></div><div><b>4</b><span>SHORT yalnız USD-M perpetual marketi olan coinde, futures tape/book ve HTF düşüşüyle simüle edilir.</span></div><div><b>5</b><span>Beklenen hareket maliyetin 2.5 katından küçükse işlem yok; stop/target en az 1.35 R:R korunur.</span></div><div><b>6</b><span>Max 2 pozisyon, portfolio heat, pair-loss guard, global stop guard ve sert time-exit sermayeyi korur.</span></div>';
  if(v4NeedsRestart&&$('sessionBanner'))$('sessionBanner').textContent='V3 SESSION · V4 RESTART GEREKLİ';
  if(document.querySelector('.pickerTitle span'))document.querySelector('.pickerTitle span').textContent='V4 otomatik tarar · ekrandaki eski 6 buton execution seçimi değildir';
}

addEventListener('load',()=>{
  try{clearInterval(v3UniverseTimer);v3UniverseTimer=null}catch{}
  v4UiPatch();
  setTimeout(async()=>{try{await v4LoadHistory();v4ScheduleUniverse();connect();render()}catch(e){console.warn('dip-v4-init',e);toast('V4 market verisi yüklenemedi: '+String(e.message||e).slice(0,80))}},50);
});
