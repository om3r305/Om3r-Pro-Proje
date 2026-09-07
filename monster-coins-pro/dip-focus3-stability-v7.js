/* Brian DIP V7 Focus-3 stability layer.
   View-only/UI hardening: keeps foresight + last detected dip pinned, throttles chart redraws,
   and removes legacy non-focus symbols from the visible lab. SHADOW ONLY. */
(function(){
  const FOCUS=['XRPUSDT','ETHUSDT','DOGEUSDT'];
  const focusSet=new Set(FOCUS);
  const stableForecast={};
  const stableDip={};
  let drawTimer=null,lastDrawAt=0;

  function focus(){
    v4Universe=[...FOCUS];
    for(const s of FOCUS)try{v4Ensure(s)}catch{}
    if(!focusSet.has(selected))selected='XRPUSDT';
  }
  function storeKey(kind,sym){return `brian-v7-${kind}:${sid||'no-session'}:${sym}`;}
  function readPinned(kind,sym,maxAgeMs){
    try{const raw=localStorage.getItem(storeKey(kind,sym));if(!raw)return null;const x=JSON.parse(raw);if(!x||Date.now()-Number(x.savedAt||0)>maxAgeMs)return null;return x.value||null;}catch{return null;}
  }
  function writePinned(kind,sym,value){try{localStorage.setItem(storeKey(kind,sym),JSON.stringify({savedAt:Date.now(),value}));}catch{}}

  // A server snapshot may still contain a legacy universe. The visible lab is always Focus-3.
  const prevRestore=restore;
  restore=function(d){
    prevRestore(d);
    focus();
  };

  // A stale browser snapshot must never re-introduce ARB/other symbols into a new session config.
  const prevStart=start;
  start=async function(restart=false){
    focus();
    if(book?.cfg)book.cfg={...book.cfg,symbols:[...FOCUS],universe_size:FOCUS.length,auto_universe:false};
    if(session?.config)session.config={...session.config,symbols:[...FOCUS],universe_size:FOCUS.length,auto_universe:false};
    return prevStart(restart);
  };

  // Keep the last valid server forecast visible while the next refresh is in flight.
  const prevForesight=v7Foresight;
  v7Foresight=function(sym=selected){
    if(!focusSet.has(sym))return null;
    const fresh=prevForesight(sym);
    if(fresh&&Number(fresh.price)>0){stableForecast[sym]=fresh;writePinned('forecast',sym,fresh);return fresh;}
    if(stableForecast[sym])return stableForecast[sym];
    const pinned=readPinned('forecast',sym,3*60_000);if(pinned){stableForecast[sym]=pinned;return pinned;}
    return null;
  };

  // A DIP line is sticky inside the same session: it only moves when a newer server dip arrives.
  // Temporary status/snapshot refreshes can no longer make the line disappear.
  if(typeof v7LatestDip==='function'){
    const prevDip=v7LatestDip;
    v7LatestDip=function(sym){
      if(!focusSet.has(sym))return null;
      const fresh=prevDip(sym);
      if(fresh&&Number(fresh.price)>0){stableDip[sym]=fresh;writePinned('dip',sym,fresh);return fresh;}
      if(stableDip[sym])return {...stableDip[sym],active:false,source:'pinned'};
      const pinned=readPinned('dip',sym,6*60*60_000);if(pinned){stableDip[sym]=pinned;return {...pinned,active:false,source:'pinned'};}
      return null;
    };
  }

  // Do not rebuild the foresight DOM on every aggTrade tick. That was causing visible layout jitter.
  const prevRenderForesightBar=renderForesightBar;
  renderForesightBar=function(){
    const bar=$('v7ForesightBar');
    const f=v7Foresight();
    const sig=f?[
      selected,f.generated_at,f.direction,Number(f.confidence||0).toFixed(5),
      f.accuracy==null?'':Number(f.accuracy).toFixed(5),Number(f.samples||0),
      Number(f.peak||0).toPrecision(10),Number(f.trough||0).toPrecision(10)
    ].join('|'):`${selected}|WAIT`;
    if(bar&&bar.dataset.stableSig===sig)return;
    prevRenderForesightBar();
    const next=$('v7ForesightBar');if(next)next.dataset.stableSig=sig;
  };

  // Radar is strictly XRP / ETH / DOGE. Clicking a row uses the final foresight-aware draw(),
  // not the old base chart renderer that temporarily hid the forecast overlay.
  renderRadar=function(){
    focus();
    const host=$('coinRadar');if(!host)return;
    const sy=[...FOCUS].sort((a,b)=>{
      const pa=states[a]?.pos?1:0,pb=states[b]?.pos?1:0;if(pa!==pb)return pb-pa;
      const ca=typeof v4Context==='function'?v4Context(a,states[a]?.pos?.venue||'SPOT'):null;
      const cb=typeof v4Context==='function'?v4Context(b,states[b]?.pos?.venue||'SPOT'):null;
      return Number(cb?.edgeRatio||v4Ranks?.[b]?.score||0)-Number(ca?.edgeRatio||v4Ranks?.[a]?.score||0);
    });
    host.innerHTML=sy.map(v4RadarRow).join('')||'<div class="row">Focus-3 verisi yükleniyor…</div>';
    host.querySelectorAll('[data-s]').forEach(x=>x.onclick=()=>{const s=x.dataset.s;if(!focusSet.has(s))return;selected=s;renderRadar();draw();});
    if($('radarStatus'))$('radarStatus').textContent=running?(v7ServerRuntime?.authoritative?'V7 CLOUD LIVE':'V7 HANDOFF'):'V7 WAIT';
  };

  // The chart does not need to repaint dozens of times per second. 8 fps is visually live and
  // removes canvas clear/repaint flashing on busy aggTrade bursts.
  const finalDraw=draw;
  draw=function(){
    const now=performance.now(),wait=Math.max(0,120-(now-lastDrawAt));
    if(drawTimer)return;
    drawTimer=setTimeout(()=>{
      drawTimer=null;
      requestAnimationFrame(()=>{lastDrawAt=performance.now();focus();finalDraw();});
    },wait);
  };

  function installStyle(){
    if(document.getElementById('v7-focus3-stability-style'))return;
    const s=document.createElement('style');s.id='v7-focus3-stability-style';s.textContent=`
      .v7ForesightBar{min-height:48px;box-sizing:border-box;contain:layout paint}
      #chartWrap{contain:layout paint}
      #candleCanvas{display:block}
    `;document.head.appendChild(s);
  }

  focus();installStyle();
  addEventListener('load',()=>{focus();installStyle();renderRadar();renderForesightBar();draw();});
})();
