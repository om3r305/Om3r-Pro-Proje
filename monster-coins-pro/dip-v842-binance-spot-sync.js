'use strict';

(()=>{
  const STREAMS=['wss://stream.binance.com:9443/stream?streams=ethusdt@aggTrade/ethusdt@kline_1s','wss://stream.binance.com:443/stream?streams=ethusdt@aggTrade/ethusdt@kline_1s','wss://data-stream.binance.vision/stream?streams=ethusdt@aggTrade/ethusdt@kline_1s'];
  const REST=['https://api.binance.com','https://data-api.binance.vision'];
  const HISTORY=180;
  let ws=null,streamIndex=0,retryTimer=null,restTimer=null,lastTickAt=0,lastEventAt=0,lastPrice=0,lastSource='WAIT',historyBusy=false,historyReady=false,paintQueued=false,lastPaintAt=0;
  let candles=[];
  let viewport={lo:null,hi:null,at:0};
  const el=id=>document.getElementById(id);
  const n=(v,f=0)=>Number.isFinite(Number(v))?Number(v):f;
  const fmt=v=>{const x=Number(v);return x>0?x.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}):'—';};
  const age=t=>t?Math.max(0,Date.now()-t):Infinity;
  const tSafe=()=>{try{return typeof thesis==='function'?thesis():null;}catch{return null;}};
  const srSafe=()=>{try{return typeof serverRuntime==='function'?serverRuntime():null;}catch{return null;}};
  const posSafe=()=>{try{return typeof position==='function'?position():null;}catch{return null;}};

  // Locked transport contract: this is the exact Binance Spot feed used by the visible chart.
  window.__v842BinanceSpot={price:0,eventAt:0,receivedAt:0,source:'WAIT'};

  function setLive(p,eventAt,source){
    if(!(p>0))return;
    lastPrice=p;lastEventAt=eventAt||Date.now();lastTickAt=Date.now();lastSource=source;
    window.__v842BinanceSpot={price:p,eventAt:lastEventAt,receivedAt:lastTickAt,source};
    try{
      if(typeof model!=='undefined'){
        model.chart=model.chart||{};
        model.chart.last_price=p;
        model.chart.candles=candles;
        model.chartError=null;
        model.chartAt=lastTickAt;
      }
    }catch{}
    const pEl=el('lastPrice');if(pEl)pEl.textContent=fmt(p);
    const sEl=el('chartSource');if(sEl)sEl.textContent=source==='SPOT_WS'?'BINANCE SPOT LIVE':'BINANCE SPOT REST';
  }

  function upsertKline(k){
    const t=n(k?.t),o=n(k?.o),h=n(k?.h),l=n(k?.l),c=n(k?.c),v=n(k?.v),ct=n(k?.T,t+999);
    if(!(t>0&&o>0&&h>0&&l>0&&c>0))return;
    const row={t,ct,o,h,l,c,v};
    const last=candles.at(-1);
    if(last&&n(last.t)===t)candles[candles.length-1]=row;
    else if(!last||t>n(last.t)){candles.push(row);if(candles.length>HISTORY)candles.splice(0,candles.length-HISTORY);}
    setLive(c,n(k?.T)||Date.now(),'SPOT_WS');
    schedulePaint();
  }

  function schedulePaint(){
    if(paintQueued)return;paintQueued=true;
    const wait=Math.max(0,900-(Date.now()-lastPaintAt));
    setTimeout(()=>requestAnimationFrame(()=>{paintQueued=false;lastPaintAt=Date.now();try{renderChart();}catch{}}),wait);
  }

  async function fetchSpot(path){
    let err='SPOT_UNAVAILABLE';
    for(const host of REST){
      try{const r=await fetch(host+path,{cache:'no-store',headers:{accept:'application/json'}});if(!r.ok){err=`HTTP_${r.status}`;continue;}return await r.json();}catch(e){err=String(e?.message||e);}
    }
    throw new Error(err);
  }

  async function bootstrap(){
    if(historyBusy||historyReady)return;historyBusy=true;
    try{
      const raw=await fetchSpot(`/api/v3/klines?symbol=ETHUSDT&interval=1s&limit=${HISTORY}`);
      if(!Array.isArray(raw)||raw.length<30)throw new Error('SPOT_1S_HISTORY_UNAVAILABLE');
      candles=raw.map(x=>({t:n(x[0]),ct:n(x[6]),o:n(x[1]),h:n(x[2]),l:n(x[3]),c:n(x[4]),v:n(x[5])})).filter(x=>x.t>0&&x.o>0&&x.h>0&&x.l>0&&x.c>0).slice(-HISTORY);
      historyReady=candles.length>=30;
      const last=candles.at(-1);if(last)setLive(last.c,last.ct,'SPOT_REST');
      try{if(typeof model!=='undefined'){model.chart={candles,last_price:last?.c||0,source:'BINANCE_SPOT_1S'};model.chartError=null;model.chartAt=Date.now();}}catch{}
      schedulePaint();
    }catch(e){try{if(typeof model!=='undefined')model.chartError=String(e?.message||e);}catch{}}
    finally{historyBusy=false;}
  }

  async function spotRestTick(){
    if(age(lastTickAt)<2500)return;
    try{const d=await fetchSpot('/api/v3/ticker/price?symbol=ETHUSDT');const p=n(d?.price);if(p>0)setLive(p,Date.now(),'SPOT_REST');}catch{}
  }

  function connect(){
    clearTimeout(retryTimer);try{ws?.close();}catch{}
    const url=STREAMS[streamIndex%STREAMS.length];
    try{
      ws=new WebSocket(url);
      ws.onopen=()=>{streamIndex=0;};
      ws.onmessage=e=>{try{const m=JSON.parse(e.data),d=m?.data||m;if(d?.e==='aggTrade'){const p=n(d.p),t=n(d.T||d.E);if(p>0)setLive(p,t,'SPOT_WS');}else if(d?.e==='kline'&&d?.k?.i==='1s')upsertKline(d.k);}catch{}};
      ws.onerror=()=>{try{ws.close();}catch{}};
      ws.onclose=()=>{streamIndex=(streamIndex+1)%STREAMS.length;retryTimer=setTimeout(connect,900);};
    }catch{streamIndex=(streamIndex+1)%STREAMS.length;retryTimer=setTimeout(connect,1200);}
  }

  function axisRange(rows){
    const lows=rows.map(x=>x.l),highs=rows.map(x=>x.h),rawLo=Math.min(...lows),rawHi=Math.max(...highs),span=Math.max(rawHi-rawLo,rawHi*.00015,0.05),pad=span*.12;
    if(!(viewport.lo>0&&viewport.hi>viewport.lo)){viewport={lo:rawLo-pad,hi:rawHi+pad,at:Date.now()};return viewport;}
    const vr=viewport.hi-viewport.lo,nearLo=rawLo<viewport.lo+vr*.08,nearHi=rawHi>viewport.hi-vr*.08;
    if(nearLo)viewport.lo=Math.min(viewport.lo,rawLo-pad);
    if(nearHi)viewport.hi=Math.max(viewport.hi,rawHi+pad);
    const rawSpan=rawHi-rawLo;
    if(Date.now()-viewport.at>60000&&rawSpan<(viewport.hi-viewport.lo)*.52)viewport={lo:rawLo-pad,hi:rawHi+pad,at:Date.now()};
    return viewport;
  }

  function line(ctx,y,L,R,w,label,value,color,dash=[5,4],right=false){const v=n(value);if(!(v>0))return false;const yy=y(v);if(!Number.isFinite(yy)||yy<18||yy>ctx.canvas.clientHeight-28)return false;ctx.save();ctx.setLineDash(dash);ctx.strokeStyle=color;ctx.lineWidth=1.1;ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.setLineDash([]);ctx.font='700 10px system-ui';const text=`${label} ${fmt(v)}`,tw=ctx.measureText(text).width,x=right?Math.max(L+4,w-R-tw-7):L+6;ctx.fillStyle='rgba(5,10,18,.90)';ctx.fillRect(x-3,yy-15,tw+7,15);ctx.fillStyle=color;ctx.fillText(text,x,yy-4);ctx.restore();return true;}
  function pivot(s,k){const x=s?.[k];return n(x?.p)>0?n(x.p):0;}
  function brianBasis(t){const decision=n(t?.decision_market_price??t?.entry_price);return lastPrice>0&&decision>0?lastPrice-decision:0;}
  function toSpot(v,basis){const x=n(v);return x>0?x+basis:0;}
  function inView(v,lo,hi){return v>lo&&v<hi;}

  function drawBrianOverlays(ctx,y,L,R,w,lo,hi,t,live){
    if(!t)return;
    const basis=brianBasis(t),S=t.structure||{},s1=S.s1||{},s5=S.s5||{},seen=new Set();
    const draw=(label,raw,color,dash,right=false)=>{
      const v=toSpot(raw,basis),key=Math.round(v*100);if(!inView(v,lo,hi)||seen.has(key))return;
      if(line(ctx,y,L,R,w,label,v,color,dash,right))seen.add(key);
    };
    draw(`1m ÜST ${s1.lastHigh?.label||''}`.trim(),pivot(s1,'lastHigh'),'#91a9c8',[1,5]);
    draw(`1m DIP ${s1.lastLow?.label||''}`.trim(),pivot(s1,'lastLow'),'#5fb6c9',[2,5]);
    draw(`5m ÜST ${s5.lastHigh?.label||''}`.trim(),pivot(s5,'lastHigh'),'#ad8cff',[3,6]);
    draw(`5m DIP ${s5.lastLow?.label||''}`.trim(),pivot(s5,'lastLow'),'#7d76d8',[3,6]);
    const plans=Array.isArray(t?.target_plan?.levels)?t.target_plan.levels:[];
    const forward=plans.map(x=>({p:toSpot(x?.price??x?.normalized_price,basis),tf:String(x?.timeframe||'')})).filter(x=>inView(x.p,lo,hi)&&x.p>live+.01).sort((a,b)=>a.p-b.p);
    let rank=0;for(const x of forward){const key=Math.round(x.p*100);if(seen.has(key))continue;rank++;line(ctx,y,L,R,w,`BRIAN ÜST ${rank}${x.tf?' '+x.tf:''}`,x.p,'#20d5a0',[7,4],true);seen.add(key);if(rank>=3)break;}
    ctx.save();ctx.font='700 9px system-ui';ctx.fillStyle='#72879f';ctx.fillText(`Brian yapı overlay · Spot eşdeğer basis ${basis>=0?'+':''}${basis.toFixed(2)}`,L+4,31);ctx.restore();
  }

  const legacyRender=typeof renderChart==='function'?renderChart:null;
  renderChart=function(){
    const cv=el('candleCanvas'),box=el('chartWrap');if(!cv||!box)return;
    const rows=candles.slice(-HISTORY);if(rows.length<2){if(legacyRender)legacyRender();return;}
    const ctx=cv.getContext('2d'),dpr=window.devicePixelRatio||1,w=Math.max(500,box.clientWidth),h=Math.max(360,box.clientHeight),pw=Math.round(w*dpr),ph=Math.round(h*dpr);
    if(cv.width!==pw||cv.height!==ph){cv.width=pw;cv.height=ph;cv.style.width=`${w}px`;cv.style.height=`${h}px`;}
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#07101a';ctx.fillRect(0,0,w,h);
    const {lo,hi}=axisRange(rows),L=15,R=92,T=42,B=38,plotW=w-L-R,plotH=h-T-B,xw=plotW/rows.length,y=v=>T+(hi-v)/(hi-lo)*plotH;
    ctx.font='10px system-ui';for(let i=0;i<=5;i++){const yy=T+plotH*i/5,v=hi-(hi-lo)*i/5;ctx.strokeStyle='#172334';ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.fillStyle='#718198';ctx.fillText(fmt(v),w-R+8,yy+3);}
    rows.forEach((c,i)=>{const x=L+xw*i+xw/2,up=c.c>=c.o,col=up?'#12d996':'#ff5068';ctx.strokeStyle=col;ctx.fillStyle=col;ctx.beginPath();ctx.moveTo(x,y(c.h));ctx.lineTo(x,y(c.l));ctx.stroke();const top=y(Math.max(c.o,c.c)),bot=y(Math.min(c.o,c.c));ctx.fillRect(x-Math.max(.7,xw*.32),top,Math.max(1.2,xw*.64),Math.max(1,bot-top));});
    const t=tSafe()||{},pos=posSafe(),basis=brianBasis(t),entry=toSpot(pos?.entry??t.entry_price,basis),target=toSpot(pos?.target??t.target??t.target_price,basis),stop=toSpot(pos?.stop??t.invalidation??t.invalidation_price,basis);
    line(ctx,y,L,R,w,pos?'POZİSYON FILL':'ADAY FILL',entry,'#6da3ff',[2,3]);
    line(ctx,y,L,R,w,pos?'POZİSYON STOP':'ADAY STOP',stop,'#ff6379',[6,4],true);
    if(inView(target,lo,hi))line(ctx,y,L,R,w,pos?'ADVISORY HEDEF':'ADAY HEDEF',target,'#19d69a',[6,4],true);
    const live=lastPrice||rows.at(-1).c;
    drawBrianOverlays(ctx,y,L,R,w,lo,hi,t,live);
    const yy=Math.max(T+10,Math.min(h-B-10,y(live))),up=live>=rows.at(-1).o,col=up?'#12d996':'#ff5068';ctx.setLineDash([3,3]);ctx.strokeStyle=col;ctx.globalAlpha=.68;ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.globalAlpha=1;ctx.setLineDash([]);ctx.fillStyle=col;ctx.fillRect(w-R+3,yy-11,R-6,22);ctx.fillStyle='#fff';ctx.font='700 10px system-ui';ctx.fillText(fmt(live),w-R+9,yy+4);
    ctx.fillStyle='#a8b6ca';ctx.font='700 10px system-ui';ctx.fillText('BINANCE SPOT · ETHUSDT · 1s LIVE',L+4,15);
    for(let i=0;i<rows.length;i+=30){const x=L+xw*i+xw/2;ctx.fillStyle='#66778e';ctx.font='9px system-ui';const d=new Date(rows[i].t);ctx.fillText(new Intl.DateTimeFormat('de-DE',{timeZone:'Europe/Berlin',hour:'2-digit',minute:'2-digit',second:'2-digit'}).format(d),x-23,h-12);}
    const pEl=el('lastPrice');if(pEl)pEl.textContent=fmt(live);const sEl=el('chartSource');if(sEl)sEl.textContent=age(lastTickAt)<2500?'BINANCE SPOT LIVE':'BINANCE SPOT REST';
  };

  const legacyLoad=typeof loadChart==='function'?loadChart:null;
  loadChart=async function(){
    if(!historyReady)await bootstrap();
    try{if(typeof model!=='undefined'&&historyReady){model.chart={candles,last_price:lastPrice||candles.at(-1)?.c||0,source:'BINANCE_SPOT_1S'};model.chartError=null;model.chartAt=lastTickAt||Date.now();}}catch{}
    renderChart();try{if(typeof renderHealth==='function')renderHealth();}catch{}
  };

  function decorate(){
    const t=tSafe(),sr=srSafe(),decision=n(t?.decision_market_price??t?.entry_price),decisionAt=Date.parse(String(t?.decision_time||t?.generated_at||0)),workerAt=Date.parse(String(sr?.generated_at||0)),diff=lastPrice&&decision?lastPrice-decision:0,liveAge=age(lastTickAt),wsLive=lastSource==='SPOT_WS'&&liveAge<2500;
    const source=wsLive?'BINANCE SPOT LIVE':lastPrice?'BINANCE SPOT REST':'BINANCE SPOT WAIT';
    const src=el('chartSource');if(src)src.textContent=source;
    const pEl=el('lastPrice');if(pEl&&lastPrice)pEl.textContent=fmt(lastPrice);
    const fresh=el('freshnessBar');if(fresh)fresh.textContent=`Binance Spot ${lastPrice?(wsLive?Math.round(liveAge)+' ms':'REST'):'WAIT'} · Brian karar ${Number.isFinite(decisionAt)?(age(decisionAt)/1000).toFixed(1)+' sn':'—'} · Worker ${Number.isFinite(workerAt)?(age(workerAt)/1000).toFixed(1)+' sn':'—'} · Motor 10 sn · SHORT KAPALI`;
    const mark=el('markPriceText');if(mark){const action=String(t?.authority_action||'WAIT'),q=t?.authority_entry_quality==null?'—':Math.round(n(t.authority_entry_quality)*100)+'%';mark.textContent=`Brian ${action} · USD-M karar ${fmt(decision)} · Binance Spot ${fmt(lastPrice)} · basis ${diff>=0?'+':''}${diff.toFixed(2)} · entry quality ${q}`;}
    const feed=el('feedMeta');if(feed)feed.textContent=`Ekran: Binance Spot 1s ${wsLive?'WebSocket':'REST'} · Brian motoru: USD-M SHADOW · karar 10 sn`;
    const feedState=el('feedState');if(feedState){feedState.textContent=lastPrice?'SPOT LIVE':'SPOT WAIT';feedState.className=lastPrice?'good':'warn';}
    const engine=el('kpiEngine');if(engine)engine.textContent='V8.4.3 LONG';
    const engineMeta=el('kpiEngineMeta');if(engineMeta)engineMeta.textContent='Brian USD-M SHADOW · Binance Spot 1s kilitli · profit protect';
    const ctx=el('decisionContext');if(ctx&&t){const scores=t.authority_scores||{},mem=(typeof model!=='undefined'?model.snapshot?.state?.v84?.v842:null)||{},protect=scores.profit_protect_armed===true?'ON':'OFF',strength=String(scores.sell_strength||'NONE'),reason=String(scores.sell_reason||'NONE');const html=`<div class="thesis-main"><b>V8.4.3 LIVE SYNC + PROFIT PROTECT</b><span>Mode <strong>LONG / SELL ONLY</strong></span><span>Brian action <strong>${String(t.authority_action||'WAIT')}</strong></span><span>Direction conf <strong>${Math.round(n(t.authority_confidence)*100)}%</strong></span><span>Entry quality <strong>${Math.round(n(t.authority_entry_quality)*100)}%</strong></span><span>Range <strong>${n(scores.range_position).toFixed(2)}</strong></span><span>Live mom <strong>${n(scores.momentum_atr).toFixed(2)} ATR</strong></span><span>Peak <strong>+${n(scores.peak_profit_bps).toFixed(1)} bps</strong></span><span>Giveback <strong>${n(scores.giveback_bps).toFixed(1)} bps</strong></span><span>Protect <strong>${protect}</strong></span><span>Sell <strong>${strength}</strong></span><span>Reason <strong>${reason}</strong></span><span>Sell votes <strong>${n(mem.sellVotes)}</strong></span><span>SHORT <strong>OFF</strong></span></div>`;if(ctx.dataset.v843!==html){ctx.innerHTML=html;ctx.dataset.v843=html;}}
  }

  bootstrap();connect();restTimer=setInterval(spotRestTick,1000);setInterval(decorate,250);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)return;if(!ws||ws.readyState>1)connect();if(age(lastTickAt)>2500)spotRestTick();});
})();