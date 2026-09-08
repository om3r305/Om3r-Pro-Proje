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
  const ma=(arr,n,i)=>{if(i+1<n)return null;let s=0;for(let k=i-n+1;k<=i;k++)s+=arr[k].c;return s/n;};
  const ageSec=v=>{const t=Date.parse(v||'');return Number.isFinite(t)?Math.max(0,Math.round((Date.now()-t)/1000)):null;};

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

  loadChart=async function(){
    if(chartBusy||document.hidden)return;chartBusy=true;
    try{
      const r=await fetch(`${CHART_API}?view=spot1s&limit=600&_=${Date.now()}`,{cache:'no-store'});
      const d=await r.json().catch(()=>({}));
      if(!r.ok)throw new Error(safeText(d.error)||`HTTP ${r.status}`);
      if(!Array.isArray(d.candles)||d.candles.length<120)throw new Error('Binance Spot 1s mum verisi yetersiz');
      model.chart=d;model.chartError=null;model.chartAt=Date.now();
    }catch(e){model.chartError=safeText(e);}finally{chartBusy=false;renderChart();renderHealth();}
  };

  renderThesis=function(){
    const host=$('thesisBox'),pos=position(),t=thesis();
    if(!pos){baseRenderThesis();return;}
    const side=String(pos.side||'').toUpperCase(),dc=side==='LONG'?'up':'down';
    const currentDir=t?.direction==='UP'?'YUKARI':t?.direction==='DOWN'?'AŞAĞI':'WAIT';
    const unreal=Number(model?.snapshot?.unrealized_pnl||0);
    const engineCurrent=Number(model?.snapshot?.state?.symbols?.ETHUSDT?.price||pos.market_price||0);
    const latestSetup=String(t?.setup||'NONE'),latestVeto=Array.isArray(t?.veto)&&t.veto.length?t.veto.join(' · '):'YOK';
    host.innerHTML=`<div class="thesis-main"><b>AKTİF POZİSYON</b><span class="dir ${dc}">${side}</span><span>Entry <strong>${livePx(pos.entry)}</strong></span><span>Motor fiyatı <strong>${livePx(engineCurrent)}</strong></span><span>Hedef <strong>${livePx(pos.target)}</strong></span><span>İptal <strong>${livePx(pos.stop)}</strong></span><span>Kaldıraç <strong>${Number(pos.leverage||1).toFixed(0)}x</strong></span><span>Notional <strong>$${Number(pos.notional||0).toFixed(2)}</strong></span><span class="${unreal>0?'pos':unreal<0?'neg':''}">Açık P&L <strong>${livePnl(unreal)}</strong></span></div><div class="thesis-levels"><span>Yeni sinyal <b>${currentDir}</b></span><span>Setup <b>${esc(latestSetup)}</b></span><small>Pozisyon USD-M motor kontratıyla yönetiliyor. Üstteki canlı grafik Binance Spot 1s görsel referanstır.<br>Yeni veto: ${esc(latestVeto)}</small></div>`;
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
    $('kpiOpenMeta').textContent=pos?`${num(pos.leverage,1)}x · notional ${money(pos.notional)} · margin ${money(pos.margin??pos.notional/num(pos.leverage,1))}`:'Pozisyon yok';
    $('kpiWinMeta').textContent=`${wins} win / ${losses} loss`;$('kpiTradesMeta').textContent='kapalı round trips';$('kpiEngineMeta').textContent='Motor: USD-M PERP · 60 sn karar';
  };

  renderHealth=function(){
    baseRenderHealth();
    const pos=position(),sr=runtime(),hb=ageSec(sr?.generated_at||model?.snapshot?.observed_at),exitLive=Boolean(model.session?.status==='PAUSED'&&pos&&sr?.status==='OK'&&hb!=null&&hb<150);
    if(exitLive){$('feedState').textContent='EXIT TRACKING';$('feedState').className='good';$('feedMeta').textContent=`USD-M pozisyon takibi · worker ${hb} sn`;$('cloudState').textContent='EXIT ONLY';$('cloudState').className='good';$('topStatus').textContent='V8.3 EXIT TRACKING';$('topStatus').className='pill good';}
  };

  renderChart=function(){
    const cv=$('candleCanvas'),box=$('chartWrap');if(!cv||!box)return;
    const ctx=cv.getContext('2d'),dpr=window.devicePixelRatio||1,w=Math.max(620,box.clientWidth),h=Math.max(560,box.clientHeight);
    cv.width=w*dpr;cv.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#07101a';ctx.fillRect(0,0,w,h);
    const raw=model.chart?.candles||[],a=raw.slice(-480).map(c=>({t:num(c.t),o:num(c.o),h:num(c.h),l:num(c.l),c:num(c.c),v:num(c.v)}));
    if(!a.length){ctx.fillStyle='#91a2b8';ctx.font='14px system-ui';ctx.fillText(model.chartError||'Binance Spot 1s bekleniyor…',20,35);return;}
    let lo=Math.min(...a.map(x=>x.l)),hi=Math.max(...a.map(x=>x.h)),pad=Math.max((hi-lo)*.10,.12);lo-=pad;hi+=pad;
    const L=18,R=82,T=24,B=34,plotW=w-L-R,plotH=h-T-B,xw=plotW/a.length,y=v=>T+(hi-v)/(hi-lo)*plotH;
    ctx.font='10px system-ui';
    for(let i=0;i<=5;i++){const yy=T+plotH*i/5,v=hi-(hi-lo)*i/5;ctx.strokeStyle='#172334';ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.fillStyle='#718198';ctx.fillText(livePx(v),w-R+7,yy+3);}
    const drawMA=(n,color)=>{ctx.strokeStyle=color;ctx.lineWidth=1.2;ctx.beginPath();let started=false;for(let i=0;i<a.length;i++){const m=ma(a,n,i);if(m==null)continue;const x=L+xw*i+xw/2,yy=y(m);if(!started){ctx.moveTo(x,yy);started=true;}else ctx.lineTo(x,yy);}ctx.stroke();};
    drawMA(7,'#e1b72f');drawMA(25,'#d94aa3');drawMA(99,'#8b6ed1');
    a.forEach((c,i)=>{const x=L+xw*i+xw/2,up=c.c>=c.o,col=up?'#21c997':'#ff566d';ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,y(c.h));ctx.lineTo(x,y(c.l));ctx.stroke();const top=y(Math.max(c.o,c.c)),bot=y(Math.min(c.o,c.c));ctx.fillRect(x-Math.max(.45,xw*.36),top,Math.max(1,xw*.72),Math.max(1,bot-top));});
    const last=num(model.chart?.last_price||a.at(-1).c),yy=Math.max(T+10,Math.min(h-B-10,y(last))),col=a.at(-1).c>=a.at(-1).o?'#21c997':'#ff566d';
    ctx.setLineDash([3,3]);ctx.strokeStyle=col;ctx.globalAlpha=.65;ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.globalAlpha=1;ctx.setLineDash([]);ctx.fillStyle=col;ctx.fillRect(w-R+2,yy-11,R-4,22);ctx.fillStyle='#fff';ctx.font='800 11px system-ui';ctx.fillText(livePx(last),w-R+8,yy+4);
    ctx.fillStyle='#6f8197';ctx.font='10px system-ui';for(let i=0;i<5;i++){const idx=Math.min(a.length-1,Math.round((a.length-1)*i/4)),x=L+xw*idx+xw/2,d=new Date(a[idx].t);ctx.fillText(d.toLocaleTimeString('de-DE',{timeZone:'Europe/Berlin',hour:'2-digit',minute:'2-digit',second:'2-digit'}),Math.max(L,Math.min(w-R-45,x-22)),h-10);}
    const badge=$('chartSource');if(badge)badge.textContent='BINANCE SPOT · 1s';
    const title=document.querySelector('.chartHead .title');if(title)title.textContent='ETHUSDT · Binance Spot · 1s';
    const note=document.querySelector('.chartHead .note');if(note)note.textContent='Binance ETHUSDT Spot 1s ile aynı görsel kaynak. Brian karar motoru: USD-M Perpetual 1m.';
    $('lastPrice').textContent=livePx(last);
  };

  render=function(){
    baseRender();
    const pos=position();
    if(pos){for(const id of ['startBtn','restartBtn','pauseBtn']){const b=$(id);if(b){b.disabled=true;b.title='Açık pozisyon kapanana kadar session kontrolü kilitli.';}}}
  };

  window.addEventListener('load',()=>setInterval(()=>{
    const bar=$('freshnessBar');if(!bar)return;
    const sr=runtime(),marketAge=Math.max(0,Math.round((Date.now()-(model.chartAt||Date.now()))/1000)),workerAge=ageSec(sr?.generated_at||model?.snapshot?.observed_at),thesisAge=ageSec(thesis()?.generated_at||thesis()?.decision_time||sr?.generated_at);
    bar.textContent=`Grafik Spot 1s ${marketAge} sn · Thesis ${thesisAge==null?'—':thesisAge+' sn'} · Worker ${workerAge==null?'—':workerAge+' sn'} · Motor karar 60 sn`;
    bar.style.color=workerAge!=null&&workerAge>135?'#ff6379':'#8ea1b8';
  },500));
})();
