'use strict';

(()=>{
  let ws=null,retry=null,lastTickAt=0,lastEventAt=0,lastPrice=0,renderPending=false;
  const $=id=>document.getElementById(id);
  const num=(v,f=0)=>Number.isFinite(Number(v))?Number(v):f;
  const px=v=>{const n=Number(v);return n>0?n.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}):'—'};
  const age=t=>t?Math.max(0,Date.now()-t):Infinity;

  function thesisSafe(){try{return typeof thesis==='function'?thesis():null;}catch{return null;}}
  function serverSafe(){try{return typeof serverRuntime==='function'?serverRuntime():null;}catch{return null;}}

  function applyTick(p,eventAt){
    lastPrice=p;lastTickAt=Date.now();lastEventAt=eventAt||lastTickAt;
    try{
      if(typeof model!=='undefined'&&model.chart){
        model.chart.last_price=p;
        const a=model.chart.candles;
        if(Array.isArray(a)&&a.length){
          const minute=Math.floor(lastEventAt/60000)*60000,last=a[a.length-1];
          if(num(last.t)===minute){last.c=p;last.h=Math.max(num(last.h,p),p);last.l=Math.min(num(last.l,p),p);}
          else if(num(last.t)<minute){a.push({t:minute,ct:minute+59999,o:p,h:p,l:p,c:p,v:0});if(a.length>240)a.splice(0,a.length-240);}
        }
      }
      if(!renderPending){renderPending=true;requestAnimationFrame(()=>{renderPending=false;try{renderChart();}catch{}});}
    }catch{}
  }

  function connect(){
    clearTimeout(retry);try{ws?.close();}catch{}
    try{
      ws=new WebSocket('wss://fstream.binance.com/ws/ethusdt@aggTrade');
      ws.onmessage=e=>{try{const d=JSON.parse(e.data),p=Number(d.p),t=Number(d.T||d.E);if(p>0)applyTick(p,t);}catch{}};
      ws.onclose=()=>{retry=setTimeout(connect,1500);};
      ws.onerror=()=>{try{ws.close();}catch{}};
    }catch{retry=setTimeout(connect,2000);}
  }

  function decorate(){
    const t=thesisSafe(),sr=serverSafe(),live=lastPrice||num(typeof model!=='undefined'?model.chart?.last_price:0),decision=num(t?.decision_market_price??t?.entry_price),decisionAt=Date.parse(String(t?.decision_time||t?.generated_at||0)),workerAt=Date.parse(String(sr?.generated_at||0));
    const liveAge=age(lastTickAt),decisionAge=age(decisionAt),workerAge=age(workerAt),diff=live&&decision?live-decision:0;
    const wsLive=liveAge<3000;
    if($('chartSource'))$('chartSource').textContent=wsLive?'BINANCE USD-M LIVE':'BINANCE REST FALLBACK';
    if($('freshnessBar'))$('freshnessBar').textContent=`Binance ${wsLive?Math.round(liveAge)+' ms':'REST'} · Brian karar ${Number.isFinite(decisionAge)?(decisionAge/1000).toFixed(1)+' sn':'—'} · Worker ${Number.isFinite(workerAge)?(workerAge/1000).toFixed(1)+' sn':'—'} · Motor 10 sn · SHORT KAPALI`;
    if($('markPriceText')){
      const action=String(t?.authority_action||'WAIT'),q=t?.authority_entry_quality==null?'—':Math.round(num(t.authority_entry_quality)*100)+'%';
      $('markPriceText').textContent=`Brian ${action} · karar fiyatı ${px(decision)} · Binance ${px(live)} · fark ${diff>=0?'+':''}${diff.toFixed(2)} · entry quality ${q}`;
    }
    if($('kpiEngine'))$('kpiEngine').textContent='BRIAN V8.4.2 LONG';
    if($('kpiEngineMeta'))$('kpiEngineMeta').textContent='USD-M PERP · 10 sn · LONG/SELL only · 1x';
    if($('cloudState')&&String($('cloudState').textContent).includes('V8.4'))$('cloudState').textContent='BULUT V8.4.2';
    const ctx=$('decisionContext');
    if(ctx&&t){
      const scores=t.authority_scores||{},mem=(typeof model!=='undefined'?model.snapshot?.state?.v84?.v842:null)||{};
      const liveBox=`<div class="thesis-main"><b>V8.4.2 LIVE SYNC</b><span>Mode <strong>LONG / SELL ONLY</strong></span><span>Brian action <strong>${String(t.authority_action||'WAIT')}</strong></span><span>Direction conf <strong>${Math.round(num(t.authority_confidence)*100)}%</strong></span><span>Entry quality <strong>${Math.round(num(t.authority_entry_quality)*100)}%</strong></span><span>Range <strong>${num(scores.range_position).toFixed(2)}</strong></span><span>Live mom <strong>${num(scores.momentum_atr).toFixed(2)} ATR</strong></span><span>Sell votes <strong>${num(mem.sellVotes)}</strong></span><span>SHORT <strong>OFF</strong></span></div>`;
      if(!ctx.dataset.v842||ctx.dataset.v842!==liveBox){ctx.innerHTML=liveBox;ctx.dataset.v842=liveBox;}
    }
  }

  window.addEventListener('load',()=>{connect();setInterval(decorate,250);});
})();