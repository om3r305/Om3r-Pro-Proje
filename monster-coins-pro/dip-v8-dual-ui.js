/* Brian V8.3 Dual UI — single-source server-authoritative stability layer. SHADOW ONLY. */
const V83_POLICY='dip-v8-dual-20260908.2';
const V83_ENGINE='brian-dip-chart-reader-v8-dual';
const V83_METRIC='target-before-invalidation-v8.2';
const V83_REV='dip-v8-dual-rootfix-20260908.3';
let v83ServerSnapshot=null;
let v83ServerFresh=false;
let v83BrowserWs=null;
let v83BrowserWsOk=false;
let v83ReconnectTimer=null;

const v83BaseParams=params;
params=function(){
  const p=v83BaseParams();
  p.config={...(p.config||{}),symbols:['ETHUSDT'],auto_universe:false,universe_size:1,engine_version:V83_ENGINE,policy_version:V83_POLICY,measurement:V83_METRIC,shadow_only:true,live_execution:false,browser_execution:false,server_authoritative:true,allow_shadow_short:true,max_shadow_leverage:2,execution_mode:'SHADOW_PAPER',sizing_policy:'V83_RISK_BUDGETED_USABLE_CAPITAL',leverage_policy:'1X_BASE__2X_ONLY_AFTER_40_CALIBRATED_EDGE',chart_reader_version:'v8.3-dual',decision_cadence_seconds:60,market_source:'BINANCE_USDM_PERP',max_account_risk_fraction:.005};
  return p;
};

function v83Snapshot(){return v83ServerSnapshot||(typeof v8ServerSnapshot!=='undefined'?v8ServerSnapshot:null);}
function v83RawThesis(){const z=v83Snapshot();return z?.state?.symbols?.ETHUSDT?.thesis||z?.state?.v8?.latestThesis||null;}
function v83NormThesis(t){
  if(!t)return null;
  return {...t,target:t.target??t.target_price??null,invalidation:t.invalidation??t.invalidation_price??null,structural_invalidation:t.structural_invalidation??t.structural_invalidation_price??null,rr:Number(t.economic_rr??t.rr??0),economic_rr:Number(t.economic_rr??t.rr??0)};
}

// Current server thesis is the chart/radar truth. Resolver forecasts are historical outcome tracking,
// not a prerequisite for showing the current WAIT/UP/DOWN decision.
v8Foresight=function(){return v83NormThesis(v83RawThesis())||null;};
v8ChartThesis=function(){
  const f=v8Foresight(),z=v83Snapshot(),p=z?.state?.v8?.pos||z?.state?.symbols?.ETHUSDT?.pos||null;
  if(!p)return f?{...f,level_source:'SERVER'}:null;
  return {...(f||{}),thesis_id:p.thesis_id,entry_low:p.entry,entry_high:p.entry,target:p.target,invalidation:p.stop,structural_invalidation:null,level_source:'POSITION',position_id:p.position_id};
};

function v83ServerRuntime(){return v83Snapshot()?.state?.serverRuntime||null;}
function v83Healthy(){
  const z=v83Snapshot(),sr=v83ServerRuntime(),generated=Date.parse(sr?.generated_at||z?.observed_at||0),age=generated?Date.now()-generated:Infinity;
  return Boolean(session?.status==='RUNNING'&&sr?.authoritative===true&&sr?.market_source==='BINANCE_USDM_PERP'&&!sr?.market_error&&age<150000);
}
function v83Text(el,v){if(el&&el.textContent!==String(v))el.textContent=String(v);}
function v83Class(el,v){if(el&&el.className!==v)el.className=v;}
function v83FeedUi(){
  const sr=v83ServerRuntime(),z=v83Snapshot(),generated=Date.parse(sr?.generated_at||z?.observed_at||0),age=generated?Date.now()-generated:Infinity;
  v83ServerFresh=v83Healthy();
  if(v83ServerFresh){
    // Keep legacy browser-health timer harmless and keep the visible state single-source.
    lastWs=Date.now();
    const sec=Math.max(0,Math.round(age/1000));
    v83Text($('feedState'),'SERVER LIVE');v83Class($('feedState'),'pos');
    v83Text($('feedMeta'),`USD-M worker · ${sec} sn · karar ${Number(sr?.decision_cadence_seconds||60)} sn${v83BrowserWsOk?' · browser WS var':''}`);
    v83Text($('onlineText'),'USD-M SERVER LIVE');
    v83Text($('marketTextSide'),'Binance USD-M · SERVER LIVE');
    try{$('marketDotSide')?.classList.remove('off');$('onlinePill')?.querySelector('.dot')?.classList.remove('off');}catch{}
    return;
  }
  if(session?.status==='RUNNING'){
    v83Text($('feedState'),'SERVER WAIT');v83Class($('feedState'),'amber');
    v83Text($('feedMeta'),sr?.market_error?`market: ${String(sr.market_error).slice(0,70)}`:'authoritative heartbeat bekleniyor');
  }else{
    v83Text($('feedState'),'IDLE');v83Class($('feedState'),'amber');v83Text($('feedMeta'),'session kapalı');
  }
}

function v83ApplyServerPrice(){
  const z=v83Snapshot(),sym=z?.state?.symbols?.ETHUSDT,t=v83RawThesis(),p=Number(sym?.price??sym?.last??t?.entry_low??t?.entry_high??0);
  if(!(p>0))return;
  live.ETHUSDT=p;if(states.ETHUSDT)states.ETHUSDT.last=p;
  const stamp=Date.parse(t?.signal_at||v83ServerRuntime()?.generated_at||Date.now()),minute=Math.floor(stamp/60000)*60000,a=candles.ETHUSDT||(candles.ETHUSDT=[]),prev=Number(a.at(-1)?.c||p);
  let x=a.find(q=>Number(q.t)===minute);
  if(!x){x={t:minute,o:prev,h:p,l:p,c:p,v:0,closed:true};a.push(x);}else{x.h=Math.max(Number(x.h||p),p);x.l=Math.min(Number(x.l||p),p);x.c=p;x.closed=true;}
  if(a.length>180)a.splice(0,a.length-180);
  if(typeof v4Bars!=='undefined'){v4Bars.ETHUSDT=v4Bars.ETHUSDT||{'1m':[],'5m':[],'15m':[],'1h':[]};v4Bars.ETHUSDT['1m']=a;}
}

const v83BaseRestore=restore;
restore=function(d){
  v83BaseRestore(d);
  v83ServerSnapshot=d?.snapshot||null;
  const c=session?.config;if(c)Object.assign(c,{symbols:['ETHUSDT'],auto_universe:false,universe_size:1,engine_version:V83_ENGINE,policy_version:V83_POLICY,measurement:V83_METRIC,shadow_only:true,live_execution:false,browser_execution:false,server_authoritative:true,allow_shadow_short:true,max_shadow_leverage:2,market_source:'BINANCE_USDM_PERP',decision_cadence_seconds:60});
  v83ApplyServerPrice();v83FeedUi();
};

function v83Dir(t){return t?.direction==='UP'?'YUKARI':t?.direction==='DOWN'?'AŞAĞI':'WAIT';}
function v83Price(v){return Number(v)>0?price(v):'—';}
renderV8ThesisBar=function(){
  const bar=$('v7ForesightBar');if(!bar)return;
  const f=v8Foresight();
  if(!f){bar.innerHTML='<div><b>BRIAN V8.3 · SERVER THESIS</b><span>Server heartbeat / structure bekleniyor…</span></div><span class="v7ForesightStatus wait">SERVER WAIT</span>';return;}
  const raw=f.raw_conviction==null?'—':`${Math.round(Number(f.raw_conviction)*100)}/100`;
  const cal=f.calibrated_probability==null?`CALIBRATING · n=${Number(f.calibration_samples||f.samples||0)}`:`${Math.round(Number(f.calibrated_probability)*100)}% · n=${Number(f.calibration_samples||f.samples||0)}`;
  const dir=v83Dir(f),cls=f.direction==='UP'?'up':f.direction==='DOWN'?'down':'flat',veto=Array.isArray(f.veto)&&f.veto.length?f.veto.join(' · '):'YOK';
  const why=Array.isArray(f.why)&&f.why.length?f.why.join(' · '):'—';
  bar.innerHTML=`<div class="v7ForesightMain"><b>BRIAN V8.3 · SERVER THESIS</b><span class="${cls}">${dir}</span><span>Setup <strong>${v4Esc(String(f.setup||'NONE'))}</strong></span><span>Rejim <strong>${v4Esc(String(f.regime||'—'))}</strong></span><span>Ham görüş <strong>${raw}</strong></span><span>Kalibrasyon <strong>${v4Esc(cal)}</strong></span></div><div class="v7ForesightLevels"><span>Entry <b>${v83Price(f.entry_low)} – ${v83Price(f.entry_high)}</b></span><span>Hedef <b>${v83Price(f.target)}</b></span><span>İptal <b>${v83Price(f.invalidation)}</b></span><span>Econ R <b>${Number(f.economic_rr||0).toFixed(2)}</b></span><small>Veto: ${v4Esc(veto)} · Yapı: ${v4Esc(why)}</small></div>`;
};

const v83BaseRenderKpi=renderKpi;
renderKpi=function(){
  v83BaseRenderKpi();
  const z=v83Snapshot(),realized=Number(z?.realized_pnl??book.realized??0),p=$('kpiPnl');
  if(p){p.textContent=pnl(realized);p.className=`value ${realized>0?'pos':realized<0?'neg':''}`;}
  const sr=v83ServerRuntime();
  if($('kpiEngine')){$('kpiEngine').textContent=v83ServerFresh?'BRIAN V8.3 DUAL':session?.status==='RUNNING'?'V8.3 SERVER WAIT':'V8.3 IDLE';$('kpiEngine').className=`value ${v83ServerFresh?'pos':'amber'}`;}
  if($('kpiEngineMeta'))$('kpiEngineMeta').textContent=`USD-M PERP · ${Number(sr?.decision_cadence_seconds||60)} sn karar · ${sr?.worker_version||V83_REV}`;
  if($('kpiOpenMeta'))$('kpiOpenMeta').textContent='max 1 · risk ≤%0.5/trade · 1x base / edge-gated 2x · SHADOW';
};

function v83PatchUi(){
  const h=document.querySelector('.desktopTitle h1');if(h)h.textContent='Brian V8.3 · Dual Direction Lab';
  const mb=document.querySelector('.mobileBrand b');if(mb)mb.textContent='Brian V8.3 Dual';
  const sub=document.querySelector('.desktopTitle .sub');if(sub)sub.textContent='ETH USD-M PERP · LONG + SHORT · 1-minute server decisions · SHADOW ONLY';
  const banner=document.querySelector('.dipBanner>div:first-child');if(banner)banner.innerHTML='<strong>BRIAN V8.3 · SERVER AUTHORITATIVE</strong> · USD-M perpetual · 1 dk karar · early reversal + direction referee · economic R:R · SHADOW ONLY.';
  const badge=document.querySelector('.expertModeCard .badge');if(badge)badge.textContent='V8.3 DUAL · 1M · USD-M PERP';
  if($('startBtn'))$('startBtn').textContent='▶ Brian V8.3 Dual Başlat';if($('restartBtn'))$('restartBtn').textContent='↻ V8.3 Temiz Test Session';
  if($('sourceBadge'))$('sourceBadge').textContent='SERVER USD-M';
  if($('chartSub'))$('chartSub').textContent='USD-M Perpetual · server-authoritative structure/thesis · browser stream yalnız görüntü desteği';
  const title=document.querySelector('#dipLog .title');if(title)title.textContent='V8.3 LONG / SHORT Açılış-Kapanış Log';
  const radarTitle=document.querySelector('#watchlist .title');if(radarTitle)radarTitle.textContent='ETH Dual Radar · V8.3';
}

const v83BaseDraw=draw;
draw=function(){
  v83BaseDraw();
  if($('sourceBadge'))$('sourceBadge').textContent='SERVER USD-M';
  if($('chartSub'))$('chartSub').textContent='USD-M Perpetual · server-authoritative structure/thesis · browser stream yalnız görüntü desteği';
};

const v83BaseStart=start;
start=async function(restart=false){
  const r=await v83BaseStart(restart);
  setTimeout(()=>{status(false).catch(()=>{});connect();},150);
  return r;
};

// Browser stream is optional display acceleration. It never changes engine/feed health.
connect=function(){
  try{v83BrowserWs?.close()}catch{}
  try{v83BrowserWs=new WebSocket('wss://fstream.binance.com/stream?streams=ethusdt@aggTrade/ethusdt@kline_1m')}catch{return;}
  ws=v83BrowserWs;
  v83BrowserWs.onopen=()=>{v83BrowserWsOk=true;};
  v83BrowserWs.onmessage=e=>{let m;try{const q=JSON.parse(e.data);m=q?.data||q}catch{return}if(m.e==='aggTrade'){const p=Number(m.p);if(p>0){live.ETHUSDT=p;if(states.ETHUSDT)states.ETHUSDT.last=p;}draw();renderRadar();renderKpi();}else if(m.e==='kline'){const k=m.k,p=Number(k.c);if(p>0){const a=candles.ETHUSDT||(candles.ETHUSDT=[]),x={t:Number(k.t),o:Number(k.o),h:Number(k.h),l:Number(k.l),c:p,v:Number(k.v),closed:Boolean(k.x)},i=a.findIndex(z=>z.t===x.t);i>=0?a[i]=x:a.push(x);if(a.length>180)a.splice(0,a.length-180);}draw();}};
  v83BrowserWs.onclose=()=>{v83BrowserWsOk=false;clearTimeout(v83ReconnectTimer);v83ReconnectTimer=setTimeout(connect,5000);};
  v83BrowserWs.onerror=()=>{try{v83BrowserWs.close()}catch{}};
};

function v83InstallFeedGuard(){
  const feed=$('feedState');if(!feed)return;
  const obs=new MutationObserver(()=>{
    if(!v83ServerFresh)return;
    const t=String(feed.textContent||'');
    if(t==='LIVE'||t==='STALE'||t==='LIVE + SERVER'){feed.textContent='SERVER LIVE';feed.className='pos';}
  });
  obs.observe(feed,{childList:true,characterData:true,subtree:true});
}

addEventListener('load',()=>{
  try{
    v83PatchUi();v83InstallFeedGuard();connect();
    setInterval(()=>{v83ApplyServerPrice();v83FeedUi();renderV8ThesisBar();renderKpi();},500);
    setTimeout(()=>status(false).catch(()=>{}),250);
  }catch(e){console.warn('v83-stability',e);}
});
