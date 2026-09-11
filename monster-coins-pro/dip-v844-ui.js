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

  window.addEventListener('load',()=>{patchLive();setInterval(patchLive,120);});
})();
