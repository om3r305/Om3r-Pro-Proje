/* Brian V8.3 Dual UI — server-authoritative view. SHADOW ONLY. */
const V83_POLICY='dip-v8-dual-20260908.2';
const V83_ENGINE='brian-dip-chart-reader-v8-dual';
const V83_METRIC='target-before-invalidation-v8.2';
const V83_REV='dip-v8-dual-rootfix-20260908.3';
let v83ServerSnapshot=null;
let v83ServerFresh=false;
let v83BrowserWsOk=false;
let v83BrowserWs=null;
let v83ReconnectTimer=null;

const _v83Params=params;
params=function(){
  const p=_v83Params();
  p.config={...(p.config||{}),symbols:['ETHUSDT'],auto_universe:false,universe_size:1,engine_version:V83_ENGINE,policy_version:V83_POLICY,measurement:V83_METRIC,shadow_only:true,live_execution:false,browser_execution:false,server_authoritative:true,allow_shadow_short:true,max_shadow_leverage:2,execution_mode:'SHADOW_PAPER',sizing_policy:'V83_RISK_BUDGETED_USABLE_CAPITAL',leverage_policy:'1X_BASE__2X_ONLY_AFTER_40_CALIBRATED_EDGE',chart_reader_version:'v8.3-dual',decision_cadence_seconds:60,market_source:'BINANCE_USDM_PERP',max_account_risk_fraction:.005};
  return p;
};

function v83SetFeedUi(mode,meta,ok=true){
  if($('feedState')){$('feedState').textContent=mode;$('feedState').className=ok?'pos':'amber';}
  if($('feedMeta'))$('feedMeta').textContent=meta||'';
  if($('onlineText'))$('onlineText').textContent=ok?`USD-M ${mode}`:'USD-M SERVER BEKLENİYOR';
  if($('marketTextSide'))$('marketTextSide').textContent=ok?`Binance USD-M · ${mode}`:'USD-M server heartbeat bekleniyor…';
  try{const d=$('marketDotSide');if(d)d.classList.toggle('off',!ok);const p=$('onlinePill')?.querySelector('.dot');if(p)p.classList.toggle('off',!ok);}catch{}
}

function v83ApplyServerCandle(priceValue,stamp){
  const p=Number(priceValue);if(!(p>0))return;
  const t=Math.floor(Number(stamp||Date.now())/60000)*60000;
  const a=candles.ETHUSDT||(candles.ETHUSDT=[]);
  const prev=Number(a.at(-1)?.c||p);
  let x=a.find(z=>Number(z.t)===t);
  if(!x){x={t,o:prev,h:p,l:p,c:p,v:0,closed:true};a.push(x);}else{x.h=Math.max(Number(x.h||p),p);x.l=Math.min(Number(x.l||p),p);x.c=p;x.closed=true;}
  if(a.length>180)a.splice(0,a.length-180);
  live.ETHUSDT=p;states.ETHUSDT.last=p;
  if(typeof v4Bars!=='undefined'){v4Bars.ETHUSDT=v4Bars.ETHUSDT||{'1m':[],'5m':[],'15m':[],'1h':[]};v4Bars.ETHUSDT['1m']=a;}
}

function v83ApplyServerStatus(d){
  const z=d?.snapshot||null,sr=z?.state?.serverRuntime||null,sym=z?.state?.symbols?.ETHUSDT||null;
  v83ServerSnapshot=z;
  const generated=Date.parse(sr?.generated_at||z?.observed_at||0);
  const age=generated?Date.now()-generated:Infinity;
  const healthy=Boolean(session?.status==='RUNNING'&&sr?.authoritative===true&&sr?.market_source==='BINANCE_USDM_PERP'&&!sr?.market_error&&age<150000);
  v83ServerFresh=healthy;
  const thesis=sym?.thesis||z?.state?.v8?.latestThesis||null;
  const p=Number(sym?.price??sym?.last??thesis?.entry_low??thesis?.entry_high??0);
  if(p>0)v83ApplyServerCandle(p,Date.parse(thesis?.signal_at||sr?.generated_at||Date.now()));
  if(healthy){
    const sec=Math.max(0,Math.round(age/1000));
    v83SetFeedUi(v83BrowserWsOk?'LIVE + SERVER':'SERVER LIVE',`USD-M worker · ${sec} sn · karar ${Number(sr?.decision_cadence_seconds||60)} sn`,true);
  }else if(session?.status==='RUNNING'){
    v83SetFeedUi('SERVER WAIT',sr?.market_error?`market: ${String(sr.market_error).slice(0,60)}`:'authoritative heartbeat bekleniyor',false);
  }else v83SetFeedUi('IDLE','session kapalı',false);
}

const _v83Restore=restore;
restore=function(d){
  _v83Restore(d);
  const c=session?.config;if(c)Object.assign(c,{engine_version:V83_ENGINE,policy_version:V83_POLICY,measurement:V83_METRIC,symbols:['ETHUSDT'],allow_shadow_short:true,max_shadow_leverage:2,shadow_only:true,live_execution:false,browser_execution:false,server_authoritative:true,market_source:'BINANCE_USDM_PERP',decision_cadence_seconds:60});
  v83ApplyServerStatus(d);
};

const _v83Note=note;
note=function(e){
  const m=e?.metadata||{},lev=Number(m.leverage||1);
  if(m.server_v8&&(e.event_kind==='BUY'||e.event_kind==='SHORT_OPEN')){const side=e.event_kind==='SHORT_OPEN'?'SHORT':'LONG';return `V8.3 ${side} · ${m.setup||''} · ${lev}x · margin $${Number(m.margin||0).toFixed(2)} · notional $${Number(e.notional||m.notional||0).toFixed(2)} · econ R ${Number(m.economic_rr??m.rr??0).toFixed(2)} · max risk $${Number(m.worst_loss||0).toFixed(2)}`;}
  if(m.server_v8&&(e.event_kind==='SELL'||e.event_kind==='SHORT_CLOSE'))return `V8.3 ${e.event_kind==='SHORT_CLOSE'?'SHORT CLOSE':'LONG CLOSE'} · ${lev}x · ${m.exit_reason||'EXIT'}`;
  return _v83Note(e);
};

const _v83UiPatch=v4UiPatch;
v4UiPatch=function(){
  _v83UiPatch();
  const h=document.querySelector('.desktopTitle h1');if(h)h.textContent='Brian V8.3 · Dual Direction Lab';
  const mb=document.querySelector('.mobileBrand b');if(mb)mb.textContent='Brian V8.3 Dual';
  const sub=document.querySelector('.desktopTitle .sub');if(sub)sub.textContent='ETH USD-M PERP · LONG + SHORT · 1-minute server decisions · risk-budgeted · SHADOW ONLY';
  const banner=document.querySelector('.dipBanner>div:first-child');if(banner)banner.innerHTML='<strong>BRIAN V8.3 · ROOTFIX · LONG + SHORT</strong> · 1 dk server karar · USD-M perpetual truth source · early reversal + direction referee · economic R:R · SHADOW ONLY.';
  const badge=document.querySelector('.expertModeCard .badge');if(badge)badge.textContent='V8.3 DUAL · 1M · USD-M PERP';
  const expert=document.querySelector('.expertModeCard');if(expert){const b=expert.querySelector('b');if(b)b.textContent='Early reversal + direction referee + fast/slow futures flow';const s=expert.querySelector('small');if(s)s.textContent='Server authoritative. Browser/VPN Binance bağlantısı kesilse de karar motoru çalışır. İşlem başı worst-case risk ≤ %0.5; 2x yalnız kalibre edge.';}
  if($('startBtn'))$('startBtn').textContent='▶ Brian V8.3 Dual Başlat';if($('restartBtn'))$('restartBtn').textContent='↻ V8.3 Temiz Test Session';
  const rule=document.querySelector('.dipRuleLine');if(rule)rule.innerHTML='V8.3: <b>1 dk server karar</b> → early reversal + confirmed structure → direction referee → fast/slow futures flow → fee/slippage/spread/funding sonrası <b>economic R:R</b> → risk-budgeted sizing. Browser feed yalnız görüntü hızlandırıcısıdır.';
  const title=document.querySelector('#dipLog .title');if(title)title.textContent='V8.3 LONG / SHORT Açılış-Kapanış Log';
  const lognote=document.querySelector('#dipLog .note');if(lognote)lognote.textContent='Yalnız server shadow pozisyon olayları · USD-M PERP truth source';
  const radarTitle=document.querySelector('#watchlist .title');if(radarTitle)radarTitle.textContent='ETH Dual Radar · V8.3';
  if($('sourceBadge'))$('sourceBadge').textContent='SERVER USD-M';
  if($('chartSub'))$('chartSub').textContent='USD-M Perpetual · server authoritative 1m truth-source · browser stream opsiyonel';
  const logic=document.querySelector('.logicSteps');if(logic)logic.innerHTML='<div><b>1</b><span>1m/5m/15m/1h/4h kapanmış mum yapısını okur; gelecek mum yok.</span></div><div><b>2</b><span>Confirmed pivot yanında provisional EARLY_REVERSAL ile hızlı dönüşü takip eder.</span></div><div><b>3</b><span>Direction referee karşı yön BOS/trendi veto eder; tek sweep büyük resmi ezemez.</span></div><div><b>4</b><span>Server USD-M perpetual fast/slow flow, book ve funding aynı kararda birleşir.</span></div><div><b>5</b><span>R:R fee/slippage/spread/funding sonrası ekonomik R:R’dır.</span></div><div><b>6</b><span>Kasa kullanılabilir üst sınır; risk bütçeli sizing. 2x edge-gated ve risk bütçesini büyütmez.</span></div>';
  if(v83ServerFresh)v83ApplyServerStatus({snapshot:v83ServerSnapshot});
};

const _v83RenderKpi=renderKpi;
renderKpi=function(){
  _v83RenderKpi();
  const realized=Number(v83ServerSnapshot?.realized_pnl??(typeof v8ServerSnapshot!=='undefined'?v8ServerSnapshot?.realized_pnl:undefined)??book.realized??0),p=$('kpiPnl');
  if(p){p.textContent=pnl(realized);p.className=`value ${realized>0?'pos':realized<0?'neg':''}`;}
  if($('kpiEngine')){$('kpiEngine').textContent=running?(v83ServerFresh?'BRIAN V8.3 DUAL':'V8.3 SERVER WAIT'):'V8.3 IDLE';$('kpiEngine').className=`value ${running&&v83ServerFresh?'pos':'amber'}`;}
  const sr=v83ServerSnapshot?.state?.serverRuntime;if($('kpiEngineMeta'))$('kpiEngineMeta').textContent=`USD-M PERP · ${Number(sr?.decision_cadence_seconds||60)} sn karar · ${sr?.worker_version||V83_REV}`;
  if($('kpiOpenMeta'))$('kpiOpenMeta').textContent='max 1 · risk ≤%0.5/trade · 1x base / edge-gated 2x · SHADOW';
};

if(typeof renderV8ThesisBar==='function'){
  const _v83RenderBar=renderV8ThesisBar;
  renderV8ThesisBar=function(){_v83RenderBar();try{const f=typeof v8Foresight==='function'?v8Foresight():null,bar=$('v7ForesightBar');if(!f||!bar)return;const main=bar.querySelector('.v7ForesightMain');if(main&&!main.querySelector('.v83EconChip')){const chip=document.createElement('span');chip.className='v83EconChip';chip.innerHTML=`Econ R <strong>${Number(f.economic_rr??f.rr??0).toFixed(2)}</strong> · Kaldıraç <strong>${Number(f.leverage||1)}x</strong>`;main.appendChild(chip);}}catch{}};
}

const _v83RenderLog=renderLog;
renderLog=function(){_v83RenderLog();const host=$('eventRows');if(host&&/V7 session|henüz Dip/i.test(host.textContent||''))host.innerHTML='<tr class="v7EmptyRow"><td colspan="8">Bu V8.3 session’da henüz LONG / SHORT açılış-kapanış olayı yok.</td></tr>';};

// Optional browser WebSocket: never controls engine health. If blocked by VPN/Opera, server stays LIVE.
connect=function(){
  try{v83BrowserWs?.close()}catch{}
  try{v83BrowserWs=new WebSocket('wss://fstream.binance.com/stream?streams=ethusdt@aggTrade/ethusdt@kline_1m')}catch{return;}
  ws=v83BrowserWs;
  v83BrowserWs.onopen=()=>{v83BrowserWsOk=true;if(v83ServerFresh)v83SetFeedUi('LIVE + SERVER','USD-M WebSocket + authoritative worker',true);};
  v83BrowserWs.onmessage=e=>{let m;try{const q=JSON.parse(e.data);m=q?.data||q}catch{return}lastWs=Date.now();if(m.e==='aggTrade'){const p=Number(m.p);if(p>0){live.ETHUSDT=p;states.ETHUSDT.last=p;}draw();renderRadar();renderKpi();}else if(m.e==='kline'){const k=m.k,p=Number(k.c);if(p>0)v83ApplyServerCandle(p,Number(k.t));draw();}};
  v83BrowserWs.onclose=()=>{v83BrowserWsOk=false;clearTimeout(v83ReconnectTimer);v83ReconnectTimer=setTimeout(connect,5000);if(v83ServerFresh)v83SetFeedUi('SERVER LIVE','Browser stream kapalı · server worker aktif',true);};
  v83BrowserWs.onerror=()=>{try{v83BrowserWs.close()}catch{}};
};

// Existing V7 status poll runs every 3s. Keep it; it is the UI heartbeat.
const _v83Start=start;
start=async function(restart=false){const r=await _v83Start(restart);setTimeout(()=>{status(false).catch(()=>{});connect();},150);return r;};

addEventListener('load',()=>{try{v4UiPatch();render();connect();setTimeout(()=>status(false).catch(()=>{}),300);}catch(e){console.warn('v83 server-primary ui',e);}});
