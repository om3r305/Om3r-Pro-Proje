'use strict';

(function(){
  const baseRenderThesis=renderThesis;
  const baseRender=render;
  const baseRenderHealth=renderHealth;

  const safeText=(v)=>{
    if(v==null)return '';
    if(typeof v==='string')return v;
    if(v instanceof Error)return v.message||String(v);
    if(typeof v==='object'){
      if(typeof v.message==='string')return v.message;
      if(typeof v.error==='string')return v.error;
      if(v.error&&typeof v.error.message==='string')return v.error.message;
      if(typeof v.status==='string')return v.status;
      try{return JSON.stringify(v);}catch{return 'Beklenmeyen hata';}
    }
    return String(v);
  };
  const livePx=v=>typeof px==='function'?px(v):Number(v||0).toFixed(2);
  const livePnl=v=>{const n=Number(v||0);return `${n>=0?'+':''}$${n.toFixed(3)}`;};
  const ageSec=v=>{const t=Date.parse(v||'');return Number.isFinite(t)?Math.max(0,Math.round((Date.now()-t)/1000)):null;};
  const tfSummary=(s)=>s?`${s.tf||''} ${s.trend||'RANGE'}${s.bos?` · BOS ${s.bos}`:''}${s.choch?` · CHOCH ${s.choch}`:''}${s.sweep?` · SWEEP ${s.sweep}`:''}`:'—';
  const eventOcc=e=>String(e?.occurrence_id||e?.thesis_id||e?.metadata?.thesis_id||'');
  const isOpen=k=>k==='BUY'||k==='SHORT_OPEN';
  const isClose=k=>k==='SELL'||k==='SHORT_CLOSE';
  const reasonText=r=>r==='TARGET_FIRST'?'TP / HEDEF':r==='INVALIDATION_FIRST'?'STOP / İPTAL':r==='EXPIRED_NO_BARRIER'?'SÜRE DOLDU':r==='AMBIGUOUS_CONSERVATIVE_STOP'?'BELİRSİZ · STOP':'KAPANDI';

  toast=function(text,kind='ok'){
    const el=$('toast');if(!el)return;
    el.textContent=safeText(text)||'İşlem tamamlandı';
    el.className=`toast show ${kind}`;
    clearTimeout(el._t);el._t=setTimeout(()=>el.className='toast',3200);
  };

  trader=async function(action,body={}){
    const key=dashboardKey();if(!key){showLock(true);throw new Error('Dashboard anahtarı gerekli.');}
    const r=await fetch(TRADER_API,{method:'POST',cache:'no-store',headers:{'content-type':'application/json','x-brian-dashboard-key':key},body:JSON.stringify({action,...body})});
    const d=await r.json().catch(()=>({}));
    if(r.status===401){localStorage.removeItem(KEY_NAME);showLock(true);}
    if(!r.ok)throw new Error(safeText(d.error??d.status??d)||`HTTP ${r.status}`);
    return d;
  };

  // REST seeds USD-M candle history; the public stream updates the display.
  // Brian karar/pozisyon motoru server tarafında USD-M Perpetual olarak kalır.
  loadChart=async function(){
    if(chartBusy||document.hidden)return;chartBusy=true;
    try{
      const r=await fetch(`${CHART_API}?view=perp1m&limit=180&_=${Date.now()}`,{cache:'no-store'});
      const d=await r.json().catch(()=>({}));
      if(!r.ok)throw new Error(safeText(d.error)||`HTTP ${r.status}`);
      if(d.source!=='BINANCE_USDM_PERP')throw new Error('Grafik piyasa kaynağı uyuşmuyor');
      if(!Array.isArray(d.candles)||d.candles.length<30)throw new Error('Binance USD-M Perp 1m mum verisi yetersiz');
      model.chart=d;model.chartError=null;model.chartAt=Date.now();
    }catch(e){model.chartError=safeText(e);}finally{chartBusy=false;renderChart();renderHealth();}
  };

  renderThesis=function(){
    const host=$('thesisBox'),pos=position(),t=thesis();
    if(!pos){baseRenderThesis();if(t?.direction==='WAIT'){const entry=host.querySelector('.thesis-levels > span');if(entry)entry.textContent='Giriş planı yok · sinyal bekleniyor';}return;}
    const side=String(pos.side||'').toUpperCase(),dc=side==='LONG'?'up':'down';
    const engineCurrent=Number(model?.snapshot?.state?.symbols?.ETHUSDT?.price||pos.market_price||0);
    const visualCurrent=Number(model?.chart?.last_price||0);
    const unreal=Number(model?.snapshot?.unrealized_pnl||0);
    const margin=Number(pos.margin??(pos.notional/Math.max(1,Number(pos.leverage||1))||0));
    const due=pos.due_at?berlinTime(pos.due_at):'—';
    const waitText=side==='SHORT'
      ? `Brian ${livePx(pos.target)} aşağı hedefi veya ${livePx(pos.stop)} yukarı stopundan hangisi önce gelirse onu bekliyor.`
      : `Brian ${livePx(pos.target)} yukarı hedefi veya ${livePx(pos.stop)} aşağı stopundan hangisi önce gelirse onu bekliyor.`;
    host.innerHTML=`<div class="thesis-main"><b>AKTİF POZİSYON</b><span class="dir ${dc}">${side}</span><span>Kullanılan kasa <strong>$${margin.toFixed(2)}</strong></span><span>Pozisyon <strong>$${Number(pos.notional||0).toFixed(2)}</strong></span><span>Entry <strong>${livePx(pos.entry)}</strong></span><span>Motor <strong>${livePx(engineCurrent)}</strong></span><span>TP <strong>${livePx(pos.target)}</strong></span><span>Stop <strong>${livePx(pos.stop)}</strong></span><span>Kaldıraç <strong>${Number(pos.leverage||1).toFixed(0)}x</strong></span><span class="${unreal>0?'pos':unreal<0?'neg':''}">Açık P&L <strong>${livePnl(unreal)}</strong></span></div><div class="thesis-levels"><span>Ne bekliyor? <b>${esc(waitText)}</b></span><span>En geç <b>${due}</b></span><small>Motor USD-M Perpetual ile yönetir. Görsel mumlar Binance USD-M Perp 1m'dir${visualCurrent?` · PERP ${livePx(visualCurrent)}`:''}.</small></div>`;
  };

  renderKpis=function(){
    const z=model.snapshot||{},rt=z.state?.v8||{},pos=position();
    const real=num(z.realized_pnl??rt.realized),unreal=num(z.unrealized_pnl),start=num(model.session?.starting_equity??rt.start??500),cash=num(z.cash??rt.cash),equity=num(z.equity,start+real+unreal);
    const trades=num(z.trade_count??rt.trades),wins=num(z.win_count??rt.wins),losses=num(z.loss_count??rt.losses),closed=wins+losses;
    $('kpiEquity').textContent=money(equity);$('kpiEquity').className=`value ${equity>start?'pos':equity<start?'neg':''}`;
    $('kpiPnl').textContent=pnl(real);$('kpiPnl').className=`value ${real>0?'pos':real<0?'neg':''}`;
    $('kpiOpen').textContent=pos?String(pos.side||'AÇIK').toUpperCase():'—';$('kpiOpen').className=`value ${pos?.side==='SHORT'?'neg':pos?'pos':''}`;
    $('kpiWin').textContent=closed?`${(wins/closed*100).toFixed(1)}%`:'—';$('kpiTrades').textContent=String(trades);
    $('kpiEngine').textContent=workerFresh()?'BRIAN V8.3 DUAL':model.session?.status==='PAUSED'&&pos?'V8.3 EXIT ONLY':model.session?.status==='PAUSED'?'V8.3 PAUSED':'V8.3 WAIT';$('kpiEngine').className=`value ${workerFresh()?'pos':'amber'}`;
    $('kpiEquityMeta').textContent=`Toplam hesap değeri · serbest nakit ${money(cash)} · başlangıç ${money(start)}`;
    $('kpiPnlMeta').textContent=`Realized ${pnl(real)} · açık ${pnl(unreal)}`;
    $('kpiOpenMeta').textContent=pos?`Kasa ${money(pos.margin??0)} · notional ${money(pos.notional)} · ${num(pos.leverage,1)}x`:'Pozisyon yok';
    $('kpiWinMeta').textContent=`${wins} win / ${losses} loss`;$('kpiTradesMeta').textContent='kapalı round trips';$('kpiEngineMeta').textContent=`Motor: USD-M PERP · ${Number(runtime()?.decision_cadence_seconds)||15} sn karar`;
  };

  renderHealth=function(){
    baseRenderHealth();
    const pos=position(),sr=runtime(),hb=ageSec(sr?.generated_at||model?.snapshot?.observed_at),exitLive=Boolean(model.session?.status==='PAUSED'&&pos&&sr?.status==='OK'&&hb!=null&&hb<150);
    if($('feedMeta'))$('feedMeta').textContent=workerFresh()&&chartFresh()?`Motor USD-M · Görsel Binance USD-M Perp 1m · karar ${num(sr?.decision_cadence_seconds,60)} sn`:(model.chartError||sr?.market_error||'heartbeat bekleniyor');
    if(exitLive){$('feedState').textContent='EXIT TRACKING';$('feedState').className='good';$('cloudState').textContent='EXIT ONLY';$('cloudState').className='good';$('topStatus').textContent='V8.3 EXIT TRACKING';$('topStatus').className='pill good';}
  };

  function line(ctx,y,L,R,w,label,value,color,dash=[5,4],right=false){
    const v=Number(value);if(!(v>0))return false;const yy=y(v);if(!Number.isFinite(yy)||yy<18||yy>ctx.canvas.clientHeight-32)return false;
    ctx.save();ctx.setLineDash(dash);ctx.strokeStyle=color;ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.setLineDash([]);ctx.font='700 10px system-ui';const text=`${label} ${livePx(v)}`,tw=ctx.measureText(text).width,x=right?Math.max(L+4,w-R-tw-8):L+6;ctx.fillStyle='rgba(5,10,18,.90)';ctx.fillRect(x-3,yy-15,tw+7,15);ctx.fillStyle=color;ctx.fillText(text,x,yy-4);ctx.restore();return true;
  }
  function pvt(s,k){const x=s?.[k];return Number(x?.p)>0?Number(x.p):null;}
  function eventMarker(ctx,a,L,R,w,h,xw,y,e){
    const at=Date.parse(e?.observed_at||e?.recorded_at||'');if(!Number.isFinite(at)||!a.length)return;
    let idx=0,best=Infinity;for(let i=0;i<a.length;i++){const d=Math.abs(a[i].t-at);if(d<best){best=d;idx=i;}}
    if(best>120000)return;const k=String(e.event_kind||''),price=Number(e.price||e.entry_price||e.exit_price);if(!(price>0))return;
    const x=L+xw*idx+xw/2,yy=y(price);if(!Number.isFinite(yy)||yy<22||yy>h-38)return;
    const isBuy=k==='BUY'||k==='SHORT_CLOSE',color=isBuy?'#21d89b':'#ff5b72',label=k==='BUY'?'BUY':k==='SELL'?'SELL':k==='SHORT_OPEN'?'SHORT':'CLOSE';
    ctx.save();ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,yy,4,0,Math.PI*2);ctx.fill();ctx.font='800 9px system-ui';const tw=ctx.measureText(label).width;ctx.fillStyle='rgba(5,10,18,.92)';ctx.fillRect(Math.max(L,x-tw/2-4),yy-21,tw+8,14);ctx.fillStyle=color;ctx.fillText(label,Math.max(L+2,x-tw/2),yy-10);ctx.restore();
  }

  renderChart=function(){
    const cv=$('candleCanvas'),box=$('chartWrap');if(!cv||!box)return;
    const ctx=cv.getContext('2d'),dpr=window.devicePixelRatio||1,w=Math.max(240,box.clientWidth),h=Math.max(280,box.clientHeight);
    cv.width=w*dpr;cv.height=h*dpr;cv.style.width=w+'px';cv.style.height=h+'px';ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#07101a';ctx.fillRect(0,0,w,h);
    const raw=model.chart?.candles||[],a=raw.slice(-(w<650?45:70)).map(c=>({t:num(c.t),o:num(c.o),h:num(c.h),l:num(c.l),c:num(c.c),v:num(c.v)}));
    if(!a.length){ctx.fillStyle='#91a2b8';ctx.font='14px system-ui';ctx.fillText(model.chartError||'Binance USD-M Perp 1m bekleniyor…',20,35);return;}
    const t=thesis()||{},S=t.structure||{},s1=S.s1||{},s5=S.s5||{},pos=position();
    const entry=Number(pos?.entry||0),target=Number(pos?.target||0),stop=Number(pos?.stop||0),h1=pvt(s1,'lastHigh'),l1=pvt(s1,'lastLow'),h5=pvt(s5,'lastHigh'),l5=pvt(s5,'lastLow');
    const candleLo=Math.min(...a.map(x=>x.l)),candleHi=Math.max(...a.map(x=>x.h)),candleSpan=Math.max(candleHi-candleLo,.5),near=v=>Number(v)>0&&Number(v)>=candleLo-candleSpan*.25&&Number(v)<=candleHi+candleSpan*.25;
    const scaleLevels=[entry,stop,h1,l1,h5,l5].filter(near);let lo=Math.min(candleLo,...scaleLevels),hi=Math.max(candleHi,...scaleLevels),pad=Math.max((hi-lo)*.08,.20);lo-=pad;hi+=pad;
    const L=16,R=90,T=35,B=38,plotW=w-L-R,plotH=h-T-B,xw=plotW/a.length,y=v=>T+(hi-v)/(hi-lo)*plotH;
    ctx.font='700 10px system-ui';ctx.fillStyle='#8fa1b9';ctx.fillText(`${tfSummary(s1)}   |   ${tfSummary(s5)}`,L,T-13);
    ctx.font='10px system-ui';for(let i=0;i<=5;i++){const yy=T+plotH*i/5,v=hi-(hi-lo)*i/5;ctx.strokeStyle='#172334';ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.fillStyle='#718198';ctx.fillText(livePx(v),w-R+8,yy+3);}
    a.forEach((c,i)=>{const x=L+xw*i+xw/2,up=c.c>=c.o,col=up?'#12d996':'#ff5068';ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,y(c.h));ctx.lineTo(x,y(c.l));ctx.stroke();const top=y(Math.max(c.o,c.c)),bot=y(Math.min(c.o,c.c));ctx.fillRect(x-Math.max(1,xw*.28),top,Math.max(2,xw*.56),Math.max(1,bot-top));});
    if(h1)line(ctx,y,L,R,w,`1m ${s1.lastHigh?.label||'H'}`,h1,'#92a7c8',[1,5]);
    if(l1)line(ctx,y,L,R,w,`1m ${s1.lastLow?.label||'L'}`,l1,'#92a7c8',[1,5]);
    if(h5)line(ctx,y,L,R,w,`5m ${s5.lastHigh?.label||'H'}`,h5,'#ae89ff',[3,6]);
    if(l5)line(ctx,y,L,R,w,`5m ${s5.lastLow?.label||'L'}`,l5,'#ae89ff',[3,6]);
    if(entry)line(ctx,y,L,R,w,pos?.side==='SHORT'?'SHORT ENTRY':'BUY ENTRY',entry,'#65a6ff',[2,3]);
    if(stop&&!line(ctx,y,L,R,w,'STOP',stop,'#ff6379',[6,4],true)){ctx.fillStyle='#ff6379';ctx.font='800 10px system-ui';ctx.fillText(`${stop>hi?'STOP ↑':'STOP ↓'} ${livePx(stop)}`,w-R-100,stop>hi?T+12:h-B-5);}
    if(target&&!line(ctx,y,L,R,w,'TP / HEDEF',target,'#19d69a',[6,4],true)){ctx.fillStyle='#19d69a';ctx.font='800 10px system-ui';ctx.fillText(`${target>hi?'TP ↑':'TP ↓'} ${livePx(target)}`,w-R-100,target>hi?T+12:h-B-5);}
    for(const e of (model.events||[]).slice(0,20))eventMarker(ctx,a,L,R,w,h,xw,y,e);
    const last=num(model.chart?.last_price||a.at(-1).c),yy=Math.max(T+10,Math.min(h-B-10,y(last))),col=a.at(-1).c>=a.at(-1).o?'#12d996':'#ff5068';ctx.setLineDash([3,3]);ctx.strokeStyle=col;ctx.globalAlpha=.65;ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.globalAlpha=1;ctx.setLineDash([]);ctx.fillStyle=col;ctx.fillRect(w-R+3,yy-11,R-6,22);ctx.fillStyle='#fff';ctx.font='800 11px system-ui';ctx.fillText(livePx(last),w-R+8,yy+4);
    ctx.fillStyle='#6f8197';ctx.font='10px system-ui';for(let i=0;i<5;i++){const idx=Math.min(a.length-1,Math.round((a.length-1)*i/4)),x=L+xw*idx+xw/2,d=new Date(a[idx].t);ctx.fillText(d.toLocaleTimeString('de-DE',{timeZone:'Europe/Berlin',hour:'2-digit',minute:'2-digit'}),Math.max(L,Math.min(w-R-36,x-18)),h-10);}
    const badge=$('chartSource');if(badge)badge.textContent='BINANCE USD-M PERP · 1m';
    const title=document.querySelector('.chartHead .title');if(title)title.textContent='ETHUSDT · Binance USD-M Perp · 1m';
    const note=document.querySelector('.chartHead .note');if(note)note.textContent='Son işlem fiyatı · USD-M Perpetual · 1 dakika mumları';
    $('lastPrice').textContent=livePx(last);
  };

  renderLog=function(){
    const rows=Array.isArray(model.events)?model.events:[],host=$('eventRows');if(!host)return;
    if(!rows.length){host.innerHTML='<tr><td colspan="10">Bu V8.3 session’da henüz işlem yok.</td></tr>';return;}
    const closes=new Map(),opens=[];
    for(const e of rows){const k=String(e.event_kind||''),occ=eventOcc(e);if(isClose(k)&&occ)closes.set(occ,e);if(isOpen(k))opens.push(e);}
    const pos=position();
    host.innerHTML=opens.slice(0,80).map(e=>{
      const k=String(e.event_kind||''),occ=eventOcc(e),m=e.metadata||{},close=closes.get(occ),active=Boolean(pos&&String(pos.thesis_id||'')===occ),target=Number(m.target||0),stop=Number(m.stop||0),amount=Number(m.margin??e.notional??0),entry=Number(e.entry_price||e.price||0),exit=Number(close?.exit_price||0),rp=close?Number(close.realized_pnl||0):0,reason=close?.metadata?.exit_reason||close?.metadata?.resolution?.reason||'';
      const status=active?`AÇIK · ${k==='SHORT_OPEN'?`TP ${livePx(target)} aşağı / STOP ${livePx(stop)} yukarı`:`TP ${livePx(target)} yukarı / STOP ${livePx(stop)} aşağı`} BEKLİYOR`:close?`KAPANDI · ${reasonText(reason)}`:'BEKLİYOR';
      const tagClass=k.includes('SHORT')?'short':'buy';
      return `<tr><td>${berlinTime(e.observed_at)}</td><td><span class="tag ${tagClass}">${esc(k)}</span></td><td>ETH</td><td><b>${money(amount)}</b></td><td>${livePx(entry)}</td><td class="pos">${livePx(target)}</td><td class="neg">${livePx(stop)}</td><td>${exit?livePx(exit):'—'}</td><td class="${rp>0?'pos':rp<0?'neg':''}">${close?livePnl(rp):'—'}</td><td>${esc(status)}</td></tr>`;
    }).join('');
  };

  render=function(){
    baseRender();renderLog();
    const pos=position();if(pos){for(const id of ['startBtn','restartBtn']){const b=$(id);if(b){b.disabled=true;b.title='Açık pozisyon kapanana kadar session kontrolü kilitli.';}}}
  };

})();
