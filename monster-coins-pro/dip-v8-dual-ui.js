/* Brian V8.3 Dual UI overlay — view/config only. Server owns SHADOW execution. */
const V83_POLICY='dip-v8-dual-20260908.2';
const V83_ENGINE='brian-dip-chart-reader-v8-dual';
const V83_METRIC='target-before-invalidation-v8.2';
const V83_REV='dip-v8-dual-rootfix-20260908.3';

// Browser chart/radar must view the same public USD-M perpetual venue as the server worker.
v4Spot=async function(path){
  const p=String(path||'').replace('/api/v3/klines','/fapi/v1/klines').replace('/api/v3/depth','/fapi/v1/depth').replace('/api/v3/aggTrades','/fapi/v1/aggTrades').replace('/api/v3/exchangeInfo','/fapi/v1/exchangeInfo');
  let last='';
  for(const h of ['https://fapi.binance.com','https://fapi1.binance.com','https://fapi2.binance.com']){
    try{const r=await fetch(h+p,{cache:'no-store'});if(!r.ok){last='HTTP '+r.status;continue;}return await r.json();}catch(e){last=String(e?.message||e)}
  }
  throw Error('USD-M market data: '+last);
};
connect=function(){
  try{ws?.close()}catch{}
  ws=new WebSocket('wss://fstream.binance.com/stream?streams=ethusdt@aggTrade/ethusdt@kline_1m');
  ws.onopen=()=>{if($('onlineText'))$('onlineText').textContent='BINANCE USD-M LIVE';if($('feedState')){$('feedState').textContent='LIVE';$('feedState').className='pos';}if($('marketTextSide'))$('marketTextSide').textContent='Binance USD-M Perp canlı';};
  ws.onmessage=e=>{lastWs=Date.now();let m;try{m=JSON.parse(e.data).data}catch{return}if(m.e==='aggTrade'){live.ETHUSDT=Number(m.p);states.ETHUSDT.last=Number(m.p);draw();renderRadar();renderKpi();}else if(m.e==='kline'){const k=m.k,x={t:Number(k.t),o:Number(k.o),h:Number(k.h),l:Number(k.l),c:Number(k.c),v:Number(k.v),closed:Boolean(k.x)},a=candles.ETHUSDT||(candles.ETHUSDT=[]),i=a.findIndex(z=>z.t===x.t);i>=0?a[i]=x:a.push(x);if(a.length>180)a.splice(0,a.length-180);live.ETHUSDT=x.c;if(v4Bars?.ETHUSDT)v4Bars.ETHUSDT['1m']=a;draw();}};
  ws.onclose=()=>setTimeout(connect,1400);ws.onerror=()=>{try{ws.close()}catch{}};
};

const _v83Params=params;
params=function(){const p=_v83Params();p.config={...(p.config||{}),symbols:['ETHUSDT'],auto_universe:false,universe_size:1,engine_version:V83_ENGINE,policy_version:V83_POLICY,measurement:V83_METRIC,shadow_only:true,live_execution:false,browser_execution:false,server_authoritative:true,allow_shadow_short:true,max_shadow_leverage:2,execution_mode:'SHADOW_PAPER',sizing_policy:'V83_RISK_BUDGETED_USABLE_CAPITAL',leverage_policy:'1X_BASE__2X_ONLY_AFTER_40_CALIBRATED_EDGE',chart_reader_version:'v8.3-dual',decision_cadence_seconds:60,market_source:'BINANCE_USDM_PERP',max_account_risk_fraction:.005};return p;};

const _v83Restore=restore;
restore=function(d){_v83Restore(d);const c=session?.config;if(c){Object.assign(c,{engine_version:V83_ENGINE,policy_version:V83_POLICY,measurement:V83_METRIC,symbols:['ETHUSDT'],allow_shadow_short:true,max_shadow_leverage:2,shadow_only:true,live_execution:false,browser_execution:false,server_authoritative:true});}};

const _v83Note=note;
note=function(e){const m=e?.metadata||{},lev=Number(m.leverage||1);if(m.server_v8&&(e.event_kind==='BUY'||e.event_kind==='SHORT_OPEN')){const side=e.event_kind==='SHORT_OPEN'?'SHORT':'LONG';return `V8.3 ${side} · ${m.setup||''} · ${lev}x · margin $${Number(m.margin||0).toFixed(2)} · notional $${Number(e.notional||m.notional||0).toFixed(2)} · econ R ${Number(m.economic_rr??m.rr??0).toFixed(2)} · max risk $${Number(m.worst_loss||0).toFixed(2)}`;}if(m.server_v8&&(e.event_kind==='SELL'||e.event_kind==='SHORT_CLOSE'))return `V8.3 ${e.event_kind==='SHORT_CLOSE'?'SHORT CLOSE':'LONG CLOSE'} · ${lev}x · ${m.exit_reason||'EXIT'}`;return _v83Note(e);};

const _v83UiPatch=v4UiPatch;
v4UiPatch=function(){_v83UiPatch();const h=document.querySelector('.desktopTitle h1');if(h)h.textContent='Brian V8.3 · Dual Direction Lab';const mb=document.querySelector('.mobileBrand b');if(mb)mb.textContent='Brian V8.3 Dual';const sub=document.querySelector('.desktopTitle .sub');if(sub)sub.textContent='ETH USD-M PERP · LONG + SHORT · 1-minute server decisions · risk-budgeted · SHADOW ONLY';const banner=document.querySelector('.dipBanner>div:first-child');if(banner)banner.innerHTML='<strong>BRIAN V8.3 · ROOTFIX · LONG + SHORT</strong> · 1 dk karar · USD-M perpetual truth source · early reversal + direction referee · maliyet sonrası ekonomik R:R · kasa risk bütçeli · SHADOW ONLY.';const badge=document.querySelector('.expertModeCard .badge');if(badge)badge.textContent='V8.3 DUAL · 1M · USD-M PERP';const expert=document.querySelector('.expertModeCard');if(expert){const b=expert.querySelector('b');if(b)b.textContent='Early reversal + direction referee + fast/slow order flow';const s=expert.querySelector('small');if(s)s.textContent='Kullanılabilir kasa gerçek üst sınırdır; boyut sinyal + stop + maliyet ile seçilir. İşlem başı worst-case risk ≤ %0.5. 2x yalnız güçlü kalibre edge.';}if($('startBtn'))$('startBtn').textContent='▶ Brian V8.3 Dual Başlat';if($('restartBtn'))$('restartBtn').textContent='↻ V8.3 Temiz Test Session';const rule=document.querySelector('.dipRuleLine');if(rule)rule.innerHTML='V8.3: <b>1 dk</b> → early reversal + confirmed structure → direction referee → fast/slow futures flow → gerçek fee/slippage/spread/funding → <b>economic R:R</b> → trade-notional üst sınırı + <b>≤%0.5 hesap riski</b>. 2x yalnız ≥40 kalibre sonuç ve çok güçlü edge.';const title=document.querySelector('#dipLog .title');if(title)title.textContent='V8.3 LONG / SHORT Açılış-Kapanış Log';const lognote=document.querySelector('#dipLog .note');if(lognote)lognote.textContent='Yalnız server shadow pozisyon olayları · USD-M PERP truth source';const radarTitle=document.querySelector('#watchlist .title');if(radarTitle)radarTitle.textContent='ETH Dual Radar · V8.3';const logic=document.querySelector('.logicSteps');if(logic)logic.innerHTML='<div><b>1</b><span>1m/5m/15m/1h/4h yapıyı yalnız kapanmış mumlarla okur; gelecek mum kullanmaz.</span></div><div><b>2</b><span>Confirmed pivot beklerken ayrıca provisional EARLY_REVERSAL ile hızlı dönüşü takip eder.</span></div><div><b>3</b><span>Direction referee karşı yöndeki güçlü BOS/trendi veto eder; tek sweep büyük resmi ezemez.</span></div><div><b>4</b><span>USD-M perpetual fast + slow aggTrade flow, order book ve funding aynı kararda birleşir.</span></div><div><b>5</b><span>R:R artık fee/slippage/spread/funding sonrası ekonomik R:R; sahte 25R göstermez.</span></div><div><b>6</b><span>Kullanılabilir kasa üst sınır; boyut risk bütçeli. 2x gross exposure edge-gated, risk bütçesi büyümez.</span></div>';if($('cloudMeta'))$('cloudMeta').textContent=`${V83_REV} · karar 1 dk · USD-M PERP · SHADOW ONLY`;};

const _v83RenderKpi=renderKpi;
renderKpi=function(){_v83RenderKpi();const realized=Number(v8ServerSnapshot?.realized_pnl??book.realized??0),p=$('kpiPnl');if(p){p.textContent=pnl(realized);p.className=`value ${realized>0?'pos':realized<0?'neg':''}`;}if($('kpiEngine')){$('kpiEngine').textContent=running?'BRIAN V8.3 DUAL':'V8.3 IDLE';$('kpiEngine').className=`value ${running?'pos':'amber'}`;}const sr=v8ServerSnapshot?.state?.serverRuntime;if($('kpiEngineMeta'))$('kpiEngineMeta').textContent=`USD-M PERP · ${Number(sr?.decision_cadence_seconds||60)} sn karar · ${sr?.worker_version||V83_REV}`;if($('kpiOpenMeta'))$('kpiOpenMeta').textContent='max 1 · risk ≤%0.5/trade · 1x base / edge-gated 2x · SHADOW';};

const _v83RenderBar=renderV8ThesisBar;
renderV8ThesisBar=function(){_v83RenderBar();const f=v8Foresight(),bar=$('v7ForesightBar');if(!f||!bar)return;const main=bar.querySelector('.v7ForesightMain');if(main){const chip=document.createElement('span');chip.innerHTML=`Econ R <strong>${Number(f.economic_rr??f.rr??0).toFixed(2)}</strong> · Kaldıraç <strong>${Number(f.leverage||1)}x</strong>`;main.appendChild(chip);}};

const _v83RenderLog=renderLog;
renderLog=function(){_v83RenderLog();const host=$('eventRows');if(host&&/V7 session|henüz Dip/i.test(host.textContent||''))host.innerHTML='<tr class="v7EmptyRow"><td colspan="8">Bu V8.3 session’da henüz LONG / SHORT açılış-kapanış olayı yok.</td></tr>';};

addEventListener('load',()=>{try{v4UiPatch();render();}catch(e){console.warn('v83-dual-ui',e);}});
