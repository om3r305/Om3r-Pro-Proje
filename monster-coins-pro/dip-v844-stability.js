'use strict';

(()=>{
  const el=id=>document.getElementById(id);
  const n=(v,f=0)=>Number.isFinite(Number(v))?Number(v):f;
  const fmt=v=>{const x=Number(v);return x>0?x.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}):'—';};
  const fmtCoin=v=>{const x=Number(v);if(!(x>0))return '—';const d=x<1?6:x<100?4:2;return x.toLocaleString('en-US',{minimumFractionDigits:Math.min(2,d),maximumFractionDigits:d});};
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
  window.__v844UiStabilityVersion='20260914.1';

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
      #alphaRadarPanel{min-height:228px!important}
      #alphaRadarPanel .alpha-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:8px}
      #alphaRadarPanel .alpha-title{font-weight:850;font-size:16px;color:#eef6ff}
      #alphaRadarPanel .alpha-meta{font-size:11px;line-height:1.45;color:#8091a6;margin-top:3px}
      #alphaRadarPanel .alpha-live{white-space:nowrap;color:#35f0ae;font-weight:850;font-size:12px}
      #alphaRadarPanel .alpha-list{display:grid;gap:7px;min-height:126px}
      #alphaRadarPanel .alpha-row{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:8px 10px;border:1px solid #183246;border-radius:10px;background:rgba(7,17,28,.56)}
      #alphaRadarPanel .alpha-coin{font-weight:820;color:#e7f2ff;font-size:12px}
      #alphaRadarPanel .alpha-sub{font-size:10px;color:#8193a8;margin-top:2px;line-height:1.35}
      #alphaRadarPanel .alpha-side{font-size:11px;font-weight:850;color:#35f0ae;white-space:nowrap}
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
        #alphaRadarPanel{min-height:248px!important}
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
    panel.innerHTML=`<div class="alpha-head"><div><div class="alpha-title">🔥 Patlama Radar / ALPHA</div><div class="alpha-meta">ETH grafiği V8.4.4 DIP motoruna özel. Bu kart piyasa genelindeki ayrı ALPHA SHADOW motorunu gösterir.</div></div><div id="alphaRadarState" class="alpha-live">BAĞLANIYOR</div></div><div id="alphaRadarList" class="alpha-list"><div class="alpha-row"><div><div class="alpha-coin">Piyasa radarı yükleniyor…</div><div class="alpha-sub">LSK, CVC ve diğer USDT adayları ayrı motorda taranıyor.</div></div><span class="alpha-side alpha-wait">WAIT</span></div></div><a class="alpha-link" href="/alpha">Tam ALPHA / Radar ekranını aç →</a>`;
    right.insertBefore(panel,right.firstChild);
    return panel;
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
      const data=await response.json(),alpha=data?.alpha_v2||{};
      const positions=Array.isArray(alpha.positions)?alpha.positions:[];
      const decisions=Array.isArray(alpha.decisions)?alpha.decisions:[];
      const online=String(alpha.status||'').toUpperCase()==='ONLINE';
      txt(state,online?'ALPHA LIVE':String(alpha.status||'ALPHA WAIT'));
      state.className=`alpha-live${online?'':' alpha-wait'}`;
      const rows=[];
      for(const p of positions.slice(0,4)){
        const asset=String(p.asset_id||'').replace('crypto:','').replace(/[^A-Z0-9_-]/gi,'');
        const side=n(p.position)>0?'LONG':'SHORT';
        rows.push(`<div class="alpha-row"><div><div class="alpha-coin">${esc(asset)} · SHADOW ${side}</div><div class="alpha-sub">Entry ${fmtCoin(p.entry_price)} · Son ref ${fmtCoin(p.last_reference_price)}</div></div><span class="alpha-side">${side}</span></div>`);
      }
      const seen=new Set(positions.map(p=>String(p.asset_id||'')));
      for(const d of decisions){
        if(rows.length>=6)break;
        const id=String(d.asset_id||'');if(!id||seen.has(id))continue;seen.add(id);
        const asset=id.replace('crypto:','').replace(/[^A-Z0-9_-]/gi,'');
        const action=String(d.action||'WAIT');
        rows.push(`<div class="alpha-row"><div><div class="alpha-coin">${esc(asset)} · ${esc(action)}</div><div class="alpha-sub">Evidence ${n(d.evidence_score).toFixed(3)} · Net edge ${Number.isFinite(Number(d.net_edge_bps))?n(d.net_edge_bps).toFixed(1)+' bps':'—'}</div></div><span class="alpha-side ${action==='WAIT'?'alpha-wait':''}">${esc(action)}</span></div>`);
      }
      list.innerHTML=rows.length?rows.join(''):'<div class="alpha-row"><div><div class="alpha-coin">ALPHA aday bekliyor</div><div class="alpha-sub">Radar canlı; ekonomik giriş oluşunca burada görünecek.</div></div><span class="alpha-side alpha-wait">WAIT</span></div>';
    }catch(error){
      txt(state,'ALPHA ERROR');state.className='alpha-live alpha-wait';
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
      return original.call(this,x,y,widened,h);
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
