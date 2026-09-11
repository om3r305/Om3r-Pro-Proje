'use strict';

(()=>{
  const el=id=>document.getElementById(id);
  const n=(v,f=0)=>Number.isFinite(Number(v))?Number(v):f;
  const fmt=v=>{const x=Number(v);return x>0?x.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}):'—';};
  const pct=v=>`${Math.round(n(v)*100)}%`;
  const txt=(node,value)=>{if(node&&node.textContent!==value)node.textContent=value;};

  function currentThesis(){try{return typeof thesis==='function'?thesis():null;}catch{return null;}}
  function currentRuntime(){try{return typeof serverRuntime==='function'?serverRuntime():null;}catch{return null;}}

  function patchStatic(){
    document.title='Monster Coins Pro — Brian V8.4.4 Cycle Forecast + Harvest';
    const unlock=document.querySelector('#unlock h2');txt(unlock,'Brian V8.4.4 Kilidi');
    const active=document.querySelector('.side .nav a.active');txt(active,'⚡ Brian V8.4.4');
    const h1=document.querySelector('main .top h1');txt(h1,'Brian V8.4.4 · Cycle Forecast + Harvest Lab');
    const refresh=el('refreshBtn');txt(refresh,'↻ V8.4.4 Yenile');
    const logTitle=document.querySelector('#dipLog .title');txt(logTitle,'V8.4.4 Shadow AL / HOLD / SAT Planı / Sonuç');
    const sessionTitle=document.querySelector('#healthPanel > .title');txt(sessionTitle,'V8.4.4 Session / Server');
    const sessionRow=document.querySelector('#healthPanel .healthRow .rtitle');txt(sessionRow,'V8.4.4 Session');
    const targetTitle=document.querySelector('.right .panel:nth-of-type(2) .title');txt(targetTitle,'Brian V8.4.4 Cycle hedefi');
    const mobile=document.querySelector('.mobileNav a:first-child span:last-child');txt(mobile,'V8.4.4');
  }

  function patchLive(){
    patchStatic();
    const t=currentThesis(),sr=currentRuntime();
    const running=sr?.status==='OK'&&sr?.authoritative===true;
    const top=el('topStatus');if(top){txt(top,running?'V8.4.4 RUNNING':'V8.4.4 WAIT');top.className=`pill ${running?'good':'warn'}`;}
    const engine=el('kpiEngine');if(engine){txt(engine,running?'V8.4.4 CYCLE':'V8.4.4 WAIT');engine.className=`value ${running?'pos':'amber'}`;}
    txt(el('kpiEngineMeta'),'Forecast-first · net edge gate · rebase · harvest · Binance Spot 1s');
    const cloud=el('cloudState');if(cloud&&running)txt(cloud,'BULUT V8.4.4');
    if(!t)return;

    const scores=t.authority_scores||{};
    const phase=String(t.cycle_phase||'SEEK_DIP');
    const action=String(t.authority_action||'WAIT');
    const timing=String(t.entry_timing_state||'WAIT');
    const forecast=n(t.forecast_destination_price);
    const stretch=n(t.stretch_target_price);
    const edge=n(t.forecast_net_edge_bps);
    const prob=n(t.forecast_probability);
    const rebase=t.rebase_required===true?(t.rebase_ready===true?'READY':'WAIT'):'N/A';
    const sellStrength=String(scores.sell_strength||'NONE');
    const sellReason=String(scores.sell_reason||'NONE');
    const currentNet=n(scores.current_net_bps);
    const peakNet=n(scores.peak_net_bps);
    const ctx=el('decisionContext');
    if(ctx){
      const html=`<div class="thesis-main"><b>V8.4.4 CYCLE FORECAST + HARVEST</b><span>Phase <strong>${phase}</strong></span><span>Action <strong>${action}</strong></span><span>Entry timing <strong>${timing}</strong></span><span>Primary forecast <strong>${fmt(forecast)}</strong></span><span>Forecast probability <strong>${pct(prob)}</strong></span><span>Net edge <strong>${edge.toFixed(2)} bps</strong></span><span>Stretch only <strong>${fmt(stretch)}</strong></span><span>Entry quality <strong>${pct(t.authority_entry_quality)}</strong></span><span>Range <strong>${n(scores.range_position).toFixed(2)}</strong></span><span>Momentum <strong>${n(scores.momentum_atr).toFixed(2)} ATR</strong></span><span>Rebase <strong>${rebase}</strong></span><span>Net now <strong>${currentNet.toFixed(1)} bps</strong></span><span>Peak net <strong>${peakNet.toFixed(1)} bps</strong></span><span>Sell <strong>${sellStrength}</strong></span><span>Reason <strong>${sellReason}</strong></span><span>SHORT <strong>OFF</strong></span></div>`;
      if(ctx.dataset.v844!==html){ctx.innerHTML=html;ctx.dataset.v844=html;}
    }
    const mark=el('markPriceText');
    if(mark){txt(mark,`Brian ${action} · Cycle ${phase} · Primary ${fmt(forecast)} · net edge ${edge.toFixed(2)} bps · timing ${timing}`);}
    const summary=document.querySelector('#decisionDetails summary');txt(summary,'Brian kararı, cycle forecast, rebase ve harvest nedenleri');
  }

  // Binance-style rolling low marker. It is display-only: Brian's blue/red/green
  // decision overlays stay untouched. The marker follows the lowest 1s Spot wick
  // in the currently visible 180-second chart window and moves immediately when
  // a new lower low arrives (or when the previous low scrolls out of view).
  let dipViewport={lo:null,hi:null,at:0};
  function dipAxis(rows){
    const rawLo=Math.min(...rows.map(x=>n(x.l))),rawHi=Math.max(...rows.map(x=>n(x.h))),span=Math.max(rawHi-rawLo,rawHi*.00015,.05),pad=span*.12;
    if(!(dipViewport.lo>0&&dipViewport.hi>dipViewport.lo)){dipViewport={lo:rawLo-pad,hi:rawHi+pad,at:Date.now()};return dipViewport;}
    const vr=dipViewport.hi-dipViewport.lo,nearLo=rawLo<dipViewport.lo+vr*.08,nearHi=rawHi>dipViewport.hi-vr*.08;
    if(nearLo)dipViewport.lo=Math.min(dipViewport.lo,rawLo-pad);
    if(nearHi)dipViewport.hi=Math.max(dipViewport.hi,rawHi+pad);
    const rawSpan=rawHi-rawLo;
    if(Date.now()-dipViewport.at>60000&&rawSpan<(dipViewport.hi-dipViewport.lo)*.52)dipViewport={lo:rawLo-pad,hi:rawHi+pad,at:Date.now()};
    return dipViewport;
  }
  function drawLiveDip(){
    try{
      const cv=el('candleCanvas'),box=el('chartWrap');if(!cv||!box||typeof model==='undefined')return;
      const rows=(Array.isArray(model?.chart?.candles)?model.chart.candles:[]).slice(-180).map(c=>({t:n(c.t),h:n(c.h),l:n(c.l)})).filter(c=>c.t>0&&c.h>0&&c.l>0);if(rows.length<2)return;
      let dipIndex=0;for(let i=1;i<rows.length;i++)if(rows[i].l<=rows[dipIndex].l)dipIndex=i;
      const dip=rows[dipIndex],ctx=cv.getContext('2d'),dpr=window.devicePixelRatio||1,w=Math.max(500,box.clientWidth),h=Math.max(360,box.clientHeight),{lo,hi}=dipAxis(rows),L=15,R=92,T=42,B=38,plotW=w-L-R,plotH=h-T-B,xw=plotW/rows.length;
      const x=L+xw*dipIndex+xw/2,y=T+(hi-dip.l)/(hi-lo)*plotH;if(!Number.isFinite(x)||!Number.isFinite(y))return;
      ctx.save();ctx.setTransform(dpr,0,0,dpr,0,0);ctx.strokeStyle='#d9e7f5';ctx.fillStyle='#d9e7f5';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x-12,y+7);ctx.lineTo(x+12,y+7);ctx.stroke();ctx.font='700 10px system-ui';const label=`CANLI DIP ${fmt(dip.l)}`,tw=ctx.measureText(label).width,labelX=Math.max(L+3,Math.min(w-R-tw-4,x-tw/2)),labelY=Math.min(h-B-4,y+21);ctx.fillStyle='rgba(7,16,26,.94)';ctx.fillRect(labelX-3,labelY-11,tw+6,14);ctx.fillStyle='#d9e7f5';ctx.fillText(label,labelX,labelY);ctx.restore();
    }catch{}
  }
  const chartRender=typeof renderChart==='function'?renderChart:null;
  if(chartRender)renderChart=function(){chartRender();drawLiveDip();};

  window.addEventListener('load',()=>{patchLive();setInterval(patchLive,120);});
})();
