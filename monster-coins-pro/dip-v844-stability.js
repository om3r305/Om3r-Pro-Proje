'use strict';

(()=>{
  const el=id=>document.getElementById(id);
  const n=(v,f=0)=>Number.isFinite(Number(v))?Number(v):f;
  const fmt=v=>{const x=Number(v);return x>0?x.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}):'—';};
  const txt=(node,value)=>{if(node&&node.textContent!==value)node.textContent=value;};
  const ageSeconds=value=>{
    const t=typeof value==='number'?value:Date.parse(String(value||''));
    if(!Number.isFinite(t)||t<=0)return '—';
    return `${Math.min(999,Math.max(0,(Date.now()-t)/1000)).toFixed(1)} sn`;
  };
  const thesisSafe=()=>{try{return typeof thesis==='function'?thesis():null;}catch{return null;}};
  const runtimeSafe=()=>{try{return typeof serverRuntime==='function'?serverRuntime():null;}catch{return null;}};
  let queued=false;

  document.documentElement.dataset.v844StableOwner='1';
  window.__v844UiStabilityVersion='20260912.1';

  function installCss(){
    if(el('v844NoJumpStyle'))return;
    const style=document.createElement('style');
    style.id='v844NoJumpStyle';
    style.textContent=`
      html,body,#overview,#chartPanel,#chartWrap,#healthPanel,#thesisBox,#decisionContext,#dipLog{overflow-anchor:none!important}
      #kpiEngineMeta,#markPriceText,#freshnessBar,#feedMeta,#feedState,#chartSource{font-variant-numeric:tabular-nums}
      #kpiEngineMeta{height:2.65em!important;min-height:2.65em!important;overflow:hidden!important}
      #markPriceText{height:2.65em!important;min-height:2.65em!important;overflow:hidden!important;line-height:1.32!important}
      #freshnessBar{height:2.65em!important;min-height:2.65em!important;overflow:hidden!important;line-height:1.32!important}
      #feedMeta{height:2.75em!important;min-height:2.75em!important;overflow:hidden!important;line-height:1.32!important}
      #feedState{display:inline-flex!important;align-items:center!important;justify-content:flex-end!important;min-width:92px!important;white-space:nowrap!important}
      #chartSource{display:inline-flex!important;align-items:center!important;justify-content:center!important;min-width:128px!important;min-height:34px!important;white-space:nowrap!important}
      #healthPanel .healthRow{min-height:82px!important}
      #decisionContext{overflow-anchor:none!important}
      @media(max-width:760px){
        #overview .kpi{min-height:112px!important}
        #kpiEngine{white-space:nowrap!important}
        #kpiEngineMeta{height:2.8em!important;min-height:2.8em!important;line-height:1.35!important}
        #markPriceText{height:3.15em!important;min-height:3.15em!important;line-height:1.42!important}
        #freshnessBar{height:3.15em!important;min-height:3.15em!important;line-height:1.42!important}
        #feedMeta{height:3.2em!important;min-height:3.2em!important;line-height:1.42!important}
        #feedState{min-width:96px!important}
        #chartSource{min-width:116px!important}
      }
    `;
    document.head.appendChild(style);
  }

  function ownTelemetry(){
    installCss();
    const t=thesisSafe()||{},sr=runtimeSafe()||{},spot=window.__v842BinanceSpot||{};
    const spotAt=n(spot.receivedAt),spotAge=spotAt?Date.now()-spotAt:Infinity;
    const spotLive=spotAge<2500&&n(spot.price)>0;
    const spotRest=!spotLive&&n(spot.price)>0;
    const source=spotLive?'BINANCE SPOT LIVE':spotRest?'BINANCE SPOT REST':'BINANCE SPOT WAIT';
    const feedState=spotLive?'SPOT LIVE':spotRest?'SPOT REST':'SPOT WAIT';
    const phase=String(t.cycle_phase||'SEEK_DIP');
    const action=String(t.authority_action||'WAIT');
    const timing=String(t.entry_timing_state||'WAIT');
    const forecast=n(t.forecast_destination_price);
    const edge=n(t.forecast_net_edge_bps);
    const decisionAt=t.decision_time||t.generated_at||t.signal_at;
    const workerAt=sr.generated_at;

    txt(el('kpiEngineMeta'),'Forecast-first · rebase · harvest · Spot 1s');
    txt(el('markPriceText'),`Brian ${action} · Cycle ${phase} · Primary ${fmt(forecast)} · net edge ${edge.toFixed(2)} bps · timing ${timing}`);
    txt(el('freshnessBar'),`Binance Spot ${spotAt?ageSeconds(spotAt):'—'} · Brian karar ${ageSeconds(decisionAt)} · Worker ${ageSeconds(workerAt)} · Motor 10 sn · SHORT KAPALI`);
    txt(el('feedMeta'),'Ekran: Binance Spot 1s · Brian motoru: USD-M SHADOW · karar 10 sn');
    const state=el('feedState');if(state){txt(state,feedState);state.className=spotLive||spotRest?'good':'warn';}
    txt(el('chartSource'),source);
  }

  function queueOwn(){
    if(queued)return;
    queued=true;
    queueMicrotask(()=>{queued=false;ownTelemetry();});
  }

  function guardLegacyWriters(){
    const ids=['kpiEngineMeta','markPriceText','freshnessBar','feedMeta','feedState','chartSource'];
    const observer=new MutationObserver(queueOwn);
    for(const id of ids){const node=el(id);if(node)observer.observe(node,{subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:['class']});}
  }

  // The 1s canvas deliberately used narrow candle bodies (about 64% of each slot),
  // which looked like broken/dotted price paths on mobile. Widen only the tiny candle
  // body rectangles on this one canvas. This is display-only; market/Brian data is untouched.
  function installCandleContinuity(){
    const proto=window.CanvasRenderingContext2D?.prototype;
    if(!proto||proto.__v844GapFixInstalled)return;
    const original=proto.fillRect;
    Object.defineProperty(proto,'__v844GapFixInstalled',{value:true,configurable:false});
    proto.fillRect=function(x,y,w,h){
      if(this?.canvas?.id==='candleCanvas'&&Number.isFinite(w)&&w>=1&&w<=8&&Number.isFinite(h)&&h>=1){
        const widened=Math.min(8,Math.max(w,w*1.48));
        return original.call(this,x-(widened-w)/2,y,widened,h);
      }
      return original.call(this,x,y,w,h);
    };
  }

  installCss();
  installCandleContinuity();
  window.addEventListener('load',()=>{
    ownTelemetry();
    guardLegacyWriters();
    setInterval(ownTelemetry,1000);
    try{if(typeof renderChart==='function')renderChart();}catch{}
  });
})();
