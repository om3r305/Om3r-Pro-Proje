'use strict';

(()=>{
  const el=id=>document.getElementById(id);
  const n=(v,f=0)=>Number.isFinite(Number(v))?Number(v):f;
  const fmt=v=>{const x=Number(v);return x>0?x.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}):'—';};
  const fmtCoin=v=>{const x=Number(v);if(!(x>0))return '—';const d=x<1?6:x<100?4:2;return x.toLocaleString('en-US',{minimumFractionDigits:Math.min(2,d),maximumFractionDigits:d});};
  const signedPct=v=>{const x=Number(v);return Number.isFinite(x)?`${x>=0?'+':''}${x.toFixed(Math.abs(x)>=100?0:1)}%`:'—';};
  const txt=(node,value)=>{if(node&&node.textContent!==value)node.textContent=value;};
  const esc=value=>String(value??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
  const ageSeconds=value=>{
    const t=typeof value==='number'?value:Date.parse(String(value||''));
    if(!Number.isFinite(t)||t<=0)return '—';
    return `${Math.min(999,Math.max(0,(Date.now()-t)/1000)).toFixed(1)} sn`;
  };
  const thesisSafe=()=>{try{return typeof thesis==='function'?thesis():null;}catch{return null;}};
  const runtimeSafe=()=>{try{return typeof serverRuntime==='function'?serverRuntime():null;}catch{return null;}};
  const ALPHA_API='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-control-center';
  const KEY_STORAGE='mcp-dashboard-key-v1';
  let queued=false;

  document.documentElement.dataset.v844StableOwner='1';
  window.__v844UiStabilityVersion='20260914.2';

  function installCss(){
    if(el('v844NoJumpStyle'))return;
    const style=document.createElement('style');
    style.id='v844NoJumpStyle';
    style.textContent=`
      html,body,#overview,#chartPanel,#chartWrap,#healthPanel,#thesisBox,#decisionContext,#dipLog,#alphaRadarPanel{overflow-anchor:none!important}
      #kpiEngineMeta,#markPriceText,#freshnessBar,#feedMeta,#feedState,#chartSource{font-variant-numeric:tabular-nums}
      #kpiEngineMeta{height:2.65em!important;min-height:2.65em!important;overflow:hidden!important}
      #markPriceText{height:2.65em!important;min-height:2.65em!important;overflow:hidden!important;line-height:1.32!important}
      #freshnessBar{height:2.65em!important;min-height:2.65em!important;overflow:hidden!important;line-height:1.32!important}
      #feedMeta{height:2.75em!important;min-height:2.75em!important;overflow:hidden!important;line-height:1.32!important}
      #feedState{display:inline-flex!important;align-items:center!important;justify-content:flex-end!important;min-width:92px!important;white-space:nowrap!important}
      #chartSource{display:inline-flex!important;align-items:center!important;justify-content:center!important;min-width:128px!important;min-height:34px!important;white-space:nowrap!important}
      #healthPanel .healthRow{min-height:82px!important}
      #decisionContext{overflow-anchor:none!important}
      #alphaRadarPanel{min-height:410px!important}
      #alphaRadarPanel .alpha-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:9px}
      #alphaRadarPanel .alpha-title{font-weight:850;font-size:16px;color:#eef6ff}
      #alphaRadarPanel .alpha-meta{font-size:11px;line-height:1.45;color:#8091a6;margin-top:3px}
      #alphaRadarPanel .alpha-live{white-space:nowrap;color:#35f0ae;font-weight:850;font-size:11px}
      #alphaRadarPanel .alpha-list{display:grid;gap:6px;min-height:278px}
      #alphaRadarPanel .alpha-section{margin:4px 1px 0;font-size:10px;font-weight:900;letter-spacing:.08em;color:#8da4bb;text-transform:uppercase}
      #alphaRadarPanel .alpha-row{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:7px 9px;border:1px solid #183246;border-radius:9px;background:rgba(7,17,28,.56)}
      #alphaRadarPanel .alpha-row.hot{border-color:#24513f;background:rgba(8,31,25,.58)}
      #alphaRadarPanel .alpha-coin{font-weight:850;color:#e7f2ff;font-size:12px}
      #alphaRadarPanel .alpha-sub{font-size:10px;color:#8193a8;margin-top:2px;line-height:1.35}
      #alphaRadarPanel .alpha-side{font-size:11px;font-weight:900;color:#35f0ae;white-space:nowrap}
      #alphaRadarPanel .alpha-side.down{color:#ff6278}
      #alphaRadarPanel .alpha-side.score{color:#62c7ff}
      #alphaRadarPanel .alpha-wait{color:#f3c969}
      #alphaRadarPanel .alpha-link{display:inline-flex;margin-top:9px;font-size:11px;font-weight:750;color:#62c7ff;text-decoration:none}
      @media(max-width:760px){
        #overview .kpi{min-height:112px!important}
        #kpiEngine{white-space:nowrap!important}
        #kpiEngineMeta{height:2.8em!important;min-height:2.8em!important;line-height:1.35!important}
        #markPriceText{height:3.15em!important;min-height:3.15em!important;line-height:1.42!important}
        #freshnessBar{height:3.15em!important;min-height:3.15em!important;line-height:1.42!important}
        #feedMeta{height:3.2em!important;min-height:3.2em!important;line-height:1.42!important}
        #feedState{min-width:96px!important}
        #chartSource{min-width:116px!important}
        #alphaRadarPanel{min-height:430px!important}
      }
    `;
    document.head.appendChild(style);
  }

  function ensureAlphaRadarPanel(){
    installCss();
    const nav=document.querySelector('.side .nav');
    if(nav&&!nav.querySelector('[data-alpha-radar-link]')){
      const a=document.createElement('a');
      a.href='/alpha';a.dataset.alphaRadarLink='1';a.innerHTML='<span>🔥</span><span>Patlama Radar</span>';
      const health=nav.querySelector('a[href="#healthPanel"]');
      if(health)nav.insertBefore(a,health);else nav.appendChild(a);
    }
    if(el('alphaRadarPanel'))return el('alphaRadarPanel');
    const right=document.querySelector('.right');
    if(!right)return null;
    const panel=document.createElement('section');
    panel.id='alphaRadarPanel';panel.className='panel';
    panel.innerHTML=`<div class="alpha-head"><div><div class="alpha-title">🔥 Patlama Radar / ALPHA</div><div class="alpha-meta">Binance piyasa evrenindeki gerçek sıcak coinler + ayrı ALPHA SHADOW pozisyonları.</div></div><div id="alphaRadarState" class="alpha-live">BAĞLANIYOR</div></div><div id="alphaRadarList" class="alpha-list"><div class="alpha-row"><div><div class="alpha-coin">Piyasa radarı yükleniyor…</div><div class="alpha-sub">Binance universe snapshot bekleniyor.</div></div><span class="alpha-side alpha-wait">WAIT</span></div></div><a class="alpha-link" href="/alpha">Tam ALPHA / Radar ekranını aç →</a>`;
    right.insertBefore(panel,right.firstChild);
    return panel;
  }

  function hotRow(row){
    const asset=String(row.base_asset||row.symbol||'').replace(/USDT$/i,'').replace(/[^A-Z0-9_-]/gi,'');
    const change=n(row.price_change_pct);
    const score=Math.round(n(row.radar_score)*100);
    const range=n(row.range_pct);
    const spread=Number.isFinite(Number(row.spread_bps))?`${n(row.spread_bps).toFixed(1)} bps`:'—';
    return `<div class="alpha-row hot"><div><div class="alpha-coin">🔥 ${esc(asset)} · ${esc(signedPct(change))}</div><div class="alpha-sub">Radar ${score}/100 · 24s range ${range.toFixed(1)}% · spread ${esc(spread)}</div></div><span class="alpha-side ${change<0?'down':'score'}">${change>=0?'HOT':'VOL'}</span></div>`;
  }

  function positionRow(p){
    const asset=String(p.asset_id||'').replace('crypto:','').replace(/[^A-Z0-9_-]/gi,'');
    const side=n(p.position)>0?'LONG':'SHORT';
    return `<div class="alpha-row"><div><div class="alpha-coin">${esc(asset)} · SHADOW ${side}</div><div class="alpha-sub">Entry ${fmtCoin(p.entry_price)} · Son ref ${fmtCoin(p.last_reference_price)}</div></div><span class="alpha-side ${side==='SHORT'?'down':''}">${side}</span></div>`;
  }

  async function refreshAlphaRadar(){
    ensureAlphaRadarPanel();
    const state=el('alphaRadarState'),list=el('alphaRadarList');
    if(!list)return;
    const key=localStorage.getItem(KEY_STORAGE)||'';
    if(!key){txt(state,'KİLİTLİ');return;}
    try{
      const response=await fetch(ALPHA_API,{method:'POST',headers:{'content-type':'application/json','x-brian-dashboard-key':key},body:JSON.stringify({action:'status'})});
      if(!response.ok)throw new Error(`HTTP ${response.status}`);
      const data=await response.json(),alpha=data?.alpha_v2||{},radar=alpha?.market_radar||data?.market_radar||{};
      const hot=Array.isArray(radar.hot)?radar.hot:[];
      const positions=Array.isArray(alpha.positions)?alpha.positions:[];
      const radarOnline=String(radar.status||'').toUpperCase()==='ONLINE';
      const alphaOnline=String(alpha.status||'').toUpperCase()==='ONLINE';
      const radarAge=Number.isFinite(Number(radar.age_seconds))?`${Math.round(n(radar.age_seconds))}s`:'—';
      txt(state,radarOnline?`RADAR LIVE · ${radarAge}`:hot.length?`RADAR ${String(radar.status||'STALE')}`:alphaOnline?'ALPHA LIVE':'RADAR WAIT');
      state.className=`alpha-live${radarOnline?'':' alpha-wait'}`;

      const rows=[];
      rows.push('<div class="alpha-section">🔥 Piyasa sıcakları</div>');
      if(hot.length){
        for(const row of hot.slice(0,5))rows.push(hotRow(row));
      }else{
        rows.push('<div class="alpha-row"><div><div class="alpha-coin">Radar snapshot bekleniyor</div><div class="alpha-sub">Universe collector canlı veriyi hazırlıyor.</div></div><span class="alpha-side alpha-wait">WAIT</span></div>');
      }
      rows.push('<div class="alpha-section">ALPHA SHADOW açık pozisyonlar</div>');
      if(positions.length){
        for(const p of positions.slice(0,3))rows.push(positionRow(p));
      }else{
        rows.push('<div class="alpha-row"><div><div class="alpha-coin">Açık SHADOW pozisyon yok</div><div class="alpha-sub">Radar sıcak coinleri göstermeye devam eder.</div></div><span class="alpha-side alpha-wait">0</span></div>');
      }
      list.innerHTML=rows.join('');
    }catch(error){
      txt(state,'RADAR ERROR');state.className='alpha-live alpha-wait';
      list.innerHTML='<div class="alpha-row"><div><div class="alpha-coin">Radar paneli veri alamadı</div><div class="alpha-sub">ETH V8.4.4 motoru bundan bağımsız çalışmaya devam ediyor.</div></div><span class="alpha-side alpha-wait">RETRY</span></div>';
    }
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
    const feed=el('feedState');if(feed){txt(feed,feedState);const cls=spotLive||spotRest?'good':'warn';if(feed.className!==cls)feed.className=cls;}
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
    ensureAlphaRadarPanel();
    refreshAlphaRadar();
    guardLegacyWriters();
    setInterval(ownTelemetry,1000);
    setInterval(()=>{if(!document.hidden)refreshAlphaRadar();},15000);
    try{if(typeof renderChart==='function')renderChart();}catch{}
  });
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)refreshAlphaRadar();});
})();
