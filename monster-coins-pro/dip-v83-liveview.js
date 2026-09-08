'use strict';

(function(){
  const baseRenderThesis=renderThesis;
  const baseRenderChart=renderChart;

  function livePx(v){return typeof px==='function'?px(v):Number(v||0).toFixed(2);}
  function livePnl(v){const n=Number(v||0);return `${n>=0?'+':''}$${n.toFixed(3)}`;}

  renderThesis=function(){
    const host=$('thesisBox');
    const pos=position();
    const t=thesis();
    if(!pos){baseRenderThesis();return;}

    const side=String(pos.side||'').toUpperCase();
    const dc=side==='LONG'?'up':'down';
    const currentDir=t?.direction==='UP'?'YUKARI':t?.direction==='DOWN'?'AŞAĞI':'WAIT';
    const unreal=Number(model?.snapshot?.unrealized_pnl||0);
    const current=Number(model?.chart?.last_price||model?.snapshot?.state?.symbols?.ETHUSDT?.price||pos.market_price||0);
    const latestSetup=String(t?.setup||'NONE');
    const latestVeto=Array.isArray(t?.veto)&&t.veto.length?t.veto.join(' · '):'YOK';
    host.innerHTML=`<div class="thesis-main"><b>AKTİF POZİSYON</b><span class="dir ${dc}">${side}</span><span>Entry <strong>${livePx(pos.entry)}</strong></span><span>Şimdi <strong>${livePx(current)}</strong></span><span>Hedef <strong>${livePx(pos.target)}</strong></span><span>İptal <strong>${livePx(pos.stop)}</strong></span><span>Kaldıraç <strong>${Number(pos.leverage||1).toFixed(0)}x</strong></span><span>Notional <strong>$${Number(pos.notional||0).toFixed(2)}</strong></span><span class="${unreal>0?'pos':unreal<0?'neg':''}">Açık P&L <strong>${livePnl(unreal)}</strong></span></div><div class="thesis-levels"><span>Güncel piyasa thesis'i <b>${currentDir}</b></span><span>Setup <b>${esc(latestSetup)}</b></span><small>Aktif pozisyon kendi entry/target/stop kontratıyla yönetiliyor. Yeni thesis WAIT/NONE olsa bile mevcut pozisyon kapanmış sayılmaz.<br>Yeni veto: ${esc(latestVeto)}</small></div>`;
  };

  renderChart=function(){
    const original=model.chart;
    if(original&&Array.isArray(original.candles)&&original.candles.length>45){
      model.chart={...original,candles:original.candles.slice(-45)};
      try{baseRenderChart();}
      finally{model.chart=original;}
      const badge=$('chartSource');if(badge)badge.textContent='SERVER USD-M · 45m';
      return;
    }
    baseRenderChart();
    const badge=$('chartSource');if(badge)badge.textContent='SERVER USD-M · 45m';
  };
})();
