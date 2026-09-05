/* Expert V4 hotfix — dynamic portfolio metrics + browser liveness diagnostics + stale-safe engine takeover.
   SHADOW/PAPER only. Does not touch frozen Brian Phase 3.7. */

// Legacy V1/V2 render() still references these readout nodes. V4 no longer displays them,
// but keeping hidden compatibility shims prevents a null dereference from killing the lifecycle.
for(const id of ['ruleRebound','ruleTp']){
  if(!document.getElementById(id)){
    const x=document.createElement('span');x.id=id;x.hidden=true;x.setAttribute('aria-hidden','true');document.body.appendChild(x);
  }
}

// V4 clean-room: V2/V3 are still loaded for backward compatibility, but they must never
// own the universe/timer once V4 is present. The old V3 5-minute refresh was polluting
// the V4 12-market state and could make the UI show coins that V4 was not evaluating.
function v4QuarantineLegacyAdapters(){
  try{if(typeof v3UniverseTimer!=='undefined'&&v3UniverseTimer){clearInterval(v3UniverseTimer);v3UniverseTimer=null;}}catch{}
  try{if(typeof v3ScheduleUniverse==='function')v3ScheduleUniverse=function(){try{if(v3UniverseTimer){clearInterval(v3UniverseTimer);v3UniverseTimer=null;}}catch{}};}catch{}
  try{if(typeof discoverUniverse==='function')discoverUniverse=async function(){return [...v4Universe];};}catch{}
  try{if(typeof v3LoadHistory==='function')v3LoadHistory=async function(){return true;};}catch{}
}
v4QuarantineLegacyAdapters();
setTimeout(v4QuarantineLegacyAdapters,0);
setTimeout(v4QuarantineLegacyAdapters,1500);
setInterval(v4QuarantineLegacyAdapters,30000);

let v4WakeLock = null;
let v4ResumeBusy = false;
let v4RuntimeReportAt = 0;
let v4LeaseClaimAt = 0;

metrics = function(){
  const c = book.cfg || cfgUi();
  let eq = Number(book.cash || 0), un = 0, open = 0;
  for(const st of Object.values(states)){
    const q = st?.pos;
    if(!q) continue;
    open++;
    const venue = q.venue || 'SPOT', side = q.side || 'LONG';
    const bk = typeof v4BookMetrics === 'function' ? v4BookMetrics(st.symbol, venue) : null;
    const mark = Number(bk?.mid || st.last || live[st.symbol] || q.entry || 0);
    const entry = Number(q.entry || 0), qty = Number(q.qty || 0);
    const margin = Number(q.margin || q.notional || 0), entryFee = Number(q.entryFee || 0);
    const feeBps = venue === 'USDM_PERP' ? 5 : Number(c.fee_bps ?? 10);
    const exitFee = Math.max(0, qty * mark) * feeBps / 10000;
    const raw = side === 'SHORT' ? qty * (entry - mark) : qty * (mark - entry);
    const positionEquity = Math.max(0, margin + raw - exitFee);
    eq += positionEquity;
    un += raw - entryFee - exitFee;
  }
  return {equity:eq, unrealized:un, open, returnPct:book.start ? 100 * (eq / book.start - 1) : 0};
};

// Keep a core liquid sleeve in the execution universe while leaving the remaining slots
// to the V4 tradeability/opportunity scanner. This prevents obvious liquid movers such as
// BNB/DOGE from disappearing only because a 24h ranking was temporarily dominated by exotic coins.
const V4_CORE_LIQUID=['BNBUSDT','DOGEUSDT','ETHUSDT','XRPUSDT'];
const _v4NativeDiscoverUniverse=v4DiscoverUniverse;
v4DiscoverUniverse=async function(force=false){
  const native=[...(await _v4NativeDiscoverUniverse(force))];
  const ranked=Object.values(v4Ranks||{});
  const momentum=[...ranked].sort((a,b)=>((Math.abs(Number(b.change||0))+1)*Number(b.tradeability||0))-((Math.abs(Number(a.change||0))+1)*Number(a.tradeability||0))).map(x=>x.symbol);
  const quality=[...ranked].sort((a,b)=>Number(b.score||0)-Number(a.score||0)).map(x=>x.symbol);
  const open=Object.keys(states).filter(s=>states[s]?.pos),next=[];
  for(const s of [...open,...V4_CORE_LIQUID,...momentum.slice(0,4),...native,...quality]){
    if(s&&!next.includes(s)&&next.length<V4_UNIVERSE_SIZE)next.push(s);
  }
  v4Universe=next;v4UniverseUpdatedAt=v4Now();v4Universe.forEach(v4Ensure);
  return v4Universe;
};

// -------- V4.1 chart-expert entries: breakout + momentum continuation + trend pullback --------
// V4 originally knew only DIP_RECLAIM LONG and DOWNTREND_BREAK SHORT. That means a strong
// bullish expansion could be visible on the chart but literally had no LONG entry path.
// These modes are still cost/HTF/BTC/order-flow gated and remain SHADOW only.
function v4TapePulse(sym,ms=15000){
  const a=(v4TapeSpot[sym]||[]),cut=v4Now()-ms,x=a.filter(q=>q.t>=cut&&Number(q.p)>0);
  if(x.length<2)return{retPct:0,ofi:0,ratio:1,prints:x.length,buy:0,sell:0};
  const first=Number(x[0].p),last=Number(x.at(-1).p),buy=v4Sum(x.map(q=>q.buy)),sell=v4Sum(x.map(q=>q.sell)),tot=buy+sell;
  return{retPct:v4Pct(last,first),ofi:tot?(buy-sell)/tot:0,ratio:sell>0?buy/sell:(buy>0?9:1),prints:x.length,buy,sell};
}
function v4RecentOneMin(sym,n=12){const r=v4Closed(sym,'1m');return r.slice(-Math.max(3,n));}
function v4TryBreakoutLong(st,ctx,p){
  const rows=v4RecentOneMin(st.symbol,14);if(rows.length<8)return false;
  const refHigh=Math.max(...rows.map(x=>Number(x.h))),pulse=v4TapePulse(st.symbol,15000);
  const breakNeed=v4Clamp(ctx.f1.atrPct*.10,.025,.18),breakPct=v4Pct(p,refHigh);
  const freshBreak=breakPct>=breakNeed;
  const accelNeed=v4Clamp(ctx.f1.atrPct*.20,.035,.24),accelerating=pulse.retPct>=accelNeed;
  const trendOk=(ctx.f5.trend>=.18&&ctx.htfLong>=-.06)||(ctx.f1.trend>=.58&&ctx.f15.trend>=-.12);
  const flowOk=(ctx.flow.ofi>=.10&&ctx.flow.ratio>=1.12)||(pulse.ofi>=.16&&pulse.ratio>=1.22);
  const bookOk=ctx.bk.pressure>=1.03,btcOk=ctx.btcLongRisk>-.38;
  const extension=ctx.f1.ema9>0?v4Pct(p,ctx.f1.ema9):0,maxExtension=Math.max(.55,ctx.f5.atrPct*1.20);
  const notLate=extension<=maxExtension&&breakPct<=Math.max(.55,ctx.f5.atrPct*.85);
  if(!(freshBreak&&accelerating&&trendOk&&flowOk&&bookOk&&btcOk&&notLate))return false;
  st.v4.lastSignal={mode:'V4_BREAKOUT_MOMENTUM',at:v4Now(),refHigh,breakPct,pulsePct:pulse.retPct};
  return v4Open(st,ctx,'LONG','V4_BREAKOUT_MOMENTUM');
}
function v4TryTrendPullbackLong(st,ctx,p){
  const rows=v4RecentOneMin(st.symbol,10);if(rows.length<7)return false;
  const hi=Math.max(...rows.map(x=>Number(x.h))),lo=Math.min(...rows.map(x=>Number(x.l))),pulse=v4TapePulse(st.symbol,12000);
  const depth=v4Pct(lo,hi)*-1,depthMin=Math.max(.08,ctx.f5.atrPct*.18),depthMax=Math.max(.45,ctx.f5.atrPct*1.35);
  const trendOk=ctx.f5.trend>=.28&&ctx.htfLong>=.08&&ctx.f15.trend>=-.05;
  const reclaimed=p>ctx.f1.ema9&&ctx.f1.ema9>=ctx.f1.ema21*.998;
  const touchedMean=lo<=Math.max(ctx.f1.ema21,ctx.f1.ema9)*1.004;
  const flowOk=(ctx.flow.ofi>=.07&&ctx.flow.ratio>=1.08)||(pulse.ofi>=.12&&pulse.retPct>0);
  const bookOk=ctx.bk.pressure>=1.02,btcOk=ctx.btcLongRisk>-.35,depthOk=depth>=depthMin&&depth<=depthMax;
  if(!(trendOk&&reclaimed&&touchedMean&&flowOk&&bookOk&&btcOk&&depthOk))return false;
  st.v4.lastSignal={mode:'V4_TREND_PULLBACK',at:v4Now(),depthPct:depth,pulsePct:pulse.retPct};
  return v4Open(st,ctx,'LONG','V4_TREND_PULLBACK');
}

// Override the V4 evaluator so momentum modes are checked BEFORE dip-arm logic.
// A coin that is already ARM_LONG is also allowed to graduate into a breakout/continuation
// entry instead of being forced to wait for the old dip-reclaim path or SKIP_CHASE.
v4Evaluate=function(sym,p){
  if(!running||v4Booting||v4NeedsRestart||!sid||!v4Universe.includes(sym))return;
  const now=v4Now();if(now-Number(v4EvalAt[sym]||0)<240)return;v4EvalAt[sym]=now;v4Funnel.evaluated++;
  const st=v4Ensure(sym),ctx=v4Context(sym,st.pos?.venue||'SPOT');if(!ctx){st.v4.lastVeto='WAIT_NATIVE_TF';return;}
  if(st.pos)return v4Manage(st,ctx);
  const general=v4EntryGuard(sym,ctx);if(general){st.v4.lastVeto=general;return v4Reject(general);}
  const mid=ctx.bk.mid||Number(p);if(!(mid>0))return;

  // 1) Fresh upside expansion: catch a real breakout while it is still fresh.
  if(v4TryBreakoutLong(st,ctx,mid))return;
  // 2) Existing bullish structure: buy the controlled pullback/reclaim continuation.
  if(v4TryTrendPullbackLong(st,ctx,mid))return;
  // 3) Original V4 dip-reclaim specialist.
  if(st.v4.phase==='ARM_LONG'){if(v4TryLong(st,ctx,mid))return;}else v4ArmLong(st,ctx,mid);
  // 4) Original futures downtrend-break short specialist.
  if(!st.pos&&st.v4.phase!=='ARM_LONG')v4TryShort(st,ctx,mid);
};

async function v4AcquireWakeLock(){
  if(!running || document.visibilityState !== 'visible' || !('wakeLock' in navigator)) return;
  try{
    if(v4WakeLock && !v4WakeLock.released) return;
    v4WakeLock = await navigator.wakeLock.request('screen');
    v4WakeLock.addEventListener?.('release',()=>{ v4WakeLock=null; });
  }catch{}
}

function v4ReportRuntimeFault(source, err){
  const now = Date.now();
  if(now - v4RuntimeReportAt < 15000) return;
  v4RuntimeReportAt = now;
  const message = String(err?.message || err || 'unknown').slice(0,220);
  console.error('V4 runtime fault', source, err);
  if($('cloudState')){$('cloudState').textContent='V4 RUNTIME';$('cloudState').className='neg';}
  if($('cloudMeta'))$('cloudMeta').textContent=`${source}: ${message}`;
  if(running && sid){
    api('event',{
      session_id:sid,engine_token:token(),event_id:`dip-v4-runtime-${crypto.randomUUID()}`,
      observed_at:iso(),event_kind:'INFO',symbol:null,price:null,cash_after:book.cash,
      equity_after:Number(metrics().equity),metadata:{expert_v4:true,info:'CLIENT_RUNTIME_ERROR',source,message}
    }).catch(()=>{});
  }
}

addEventListener('error', e=>v4ReportRuntimeFault('window.error', e.error || e.message));
addEventListener('unhandledrejection', e=>v4ReportRuntimeFault('unhandledrejection', e.reason));

async function v4EnsureEngineLease(){
  if(!session || session.status!=='RUNNING' || v4NeedsRestart || !sid) return false;
  try{
    await api('engine_check',{session_id:sid,engine_token:token()});
    running=true;return true;
  }catch(e){
    const msg=String(e?.message||e);
    if(!/UNAUTHORIZED_DIP_ENGINE|BRIAN_DIP_ENGINE_LEASE_ACTIVE/.test(msg)){running=false;throw e;}
    try{
      const c=await api('claim_engine',{session_id:sid,engine_token:token()});
      running=true;v4LeaseClaimAt=Date.now();
      try{event('INFO',null,null,{metadata:{expert_v4:true,info:'ENGINE_LEASE_CLAIMED',lease_generation:c.lease_generation||null}})}catch{}
      toast('V4 motor bu cihaz tarafından devralındı.');
      return true;
    }catch(claimErr){
      if(String(claimErr?.message||claimErr).includes('BRIAN_DIP_ENGINE_LEASE_ACTIVE')){running=false;return false;}
      running=false;throw claimErr;
    }
  }
}

// Replace V4 status with lease-aware status. A stale/dead browser can no longer leave a zombie RUNNING session.
status = async function(initial=false){
  try{
    const d=await api('status');restore(d);
    if(session?.status==='RUNNING'&&!v4NeedsRestart) await v4EnsureEngineLease(); else running=false;
    v4UiPatch();render();
    if(initial)toast(v4NeedsRestart?'V3 session bulundu · V4 için Yeni Kasa ile Restart.':running?'Dip Expert V4 canlı.':'V4 session başka aktif cihazda.');
  }catch(e){engineErr(e);if(initial)toast(String(e.message||e))}
};

async function v4ResumeForeground(){
  if(document.visibilityState !== 'visible' || v4ResumeBusy) return;
  v4ResumeBusy = true;
  try{
    await status(false);
    if(session?.status === 'RUNNING' && running && !v4NeedsRestart){
      await v4LoadHistory();
      connect();
      await snapshot();
    }
    await v4AcquireWakeLock();
  }catch(e){v4ReportRuntimeFault('foreground-resume',e)}
  finally{v4ResumeBusy=false;try{render()}catch(e){v4ReportRuntimeFault('foreground-render',e)}}
}

document.addEventListener('visibilitychange',()=>{if(document.visibilityState === 'visible') v4ResumeForeground();});
addEventListener('pageshow',()=>setTimeout(v4ResumeForeground,100));

setInterval(async()=>{
  if(!session || session.status!=='RUNNING') return;
  const age = lastWs ? Date.now()-lastWs : Infinity;
  if(running){
    try{await api('engine_check',{session_id:sid,engine_token:token()})}catch(e){
      running=false;try{await v4EnsureEngineLease()}catch(err){v4ReportRuntimeFault('engine-heartbeat',err)}
    }
  }else if(document.visibilityState==='visible'){
    try{await v4EnsureEngineLease();if(running){await v4LoadHistory();connect();await snapshot();}}catch{}
  }
  if(running && document.visibilityState==='visible' && age>10000){try{connect()}catch(e){v4ReportRuntimeFault('ws-watchdog',e)}}
  if(running) v4AcquireWakeLock();
},7000);

setInterval(()=>{
  if(running && sid && !v4NeedsRestart){Promise.resolve(snapshot()).catch(e=>v4ReportRuntimeFault('snapshot-heartbeat',e));}
},7000);

const _v4HotfixRenderKpi = renderKpi;
renderKpi = function(){
  _v4HotfixRenderKpi();
  if(!$('kpiEngine')) return;
  const age = lastWs ? Date.now()-lastWs : Infinity;
  if(running && age <= 10000){
    $('kpiEngine').textContent='EXPERT V4.1';$('kpiEngine').className='value pos';
    if($('kpiEngineMeta'))$('kpiEngineMeta').textContent=`LIVE · DIP + BREAKOUT + PULLBACK · AUTO ${v4Universe.length||V4_UNIVERSE_SIZE} · feed ${Math.round(age/1000)}s`;
  }else if(running){
    $('kpiEngine').textContent='V4.1 STALE';$('kpiEngine').className='value amber';
    if($('kpiEngineMeta'))$('kpiEngineMeta').textContent='Session sahibi bu cihaz · Binance feed bayat, reconnect deneniyor';
  }else if(session?.status==='RUNNING'){
    $('kpiEngine').textContent='V4.1 VIEW';$('kpiEngine').className='value amber';
    if($('kpiEngineMeta'))$('kpiEngineMeta').textContent='Başka canlı lease var veya 25 sn devralma süresi bekleniyor';
  }else{
    $('kpiEngine').textContent='V4.1 IDLE';$('kpiEngine').className='value amber';
  }
};

// Upgrade visible explanation so the UI describes what the engine actually evaluates.
const _v4HotfixUiPatch=v4UiPatch;
v4UiPatch=function(){
  _v4HotfixUiPatch();
  const h=document.querySelector('.desktopTitle h1');if(h)h.textContent='Aggressive Chart Expert V4.1';
  const mb=document.querySelector('.mobileBrand b');if(mb)mb.textContent='Chart Expert V4.1';
  const banner=document.querySelector('.dipBanner>div:first-child');if(banner)banner.innerHTML='<strong>EXPERT V4.1 · SHADOW ONLY</strong> · Dört setup: dip-reclaim, breakout momentum, trend-pullback continuation ve futures downtrend short. BTC rejimi + 1m/5m/15m/1h + order-flow + book pressure + gerçekçi cost gate birlikte çalışır.';
  const rule=document.querySelector('.dipRuleLine');if(rule)rule.innerHTML='V4.1: <b>DIP tek giriş yolu değildir.</b> Fresh breakout + momentum, trend içi pullback/reclaim, klasik dip-reclaim ve futures downtrend short ayrı setup olarak değerlendirilir. Her biri maliyet, BTC/HTF, OFI, depth ve risk sizing kapısından geçer.';
  const logic=document.querySelector('.logicSteps');if(logic)logic.innerHTML='<div><b>1</b><span>Likidite + tradeability tarar; BNB/DOGE/ETH/XRP çekirdek likit sleeve, kalan slotlar dinamik fırsatlar.</span></div><div><b>2</b><span>Fresh 1m breakout ve 15sn tape acceleration varsa momentum LONG yolunu açar; dip beklemek zorunda değildir.</span></div><div><b>3</b><span>5m/15m trend güçlü ise kontrollü pullback sonrası EMA reclaim + order-flow dönüşünde continuation LONG arar.</span></div><div><b>4</b><span>Dip-reclaim hâlâ ayrı setup; ARM_LONG olan coin breakout fırsatı üretirse artık bloke edilmez.</span></div><div><b>5</b><span>SHORT yalnız USD-M perpetual + futures tape/book + HTF düşüş yapısıyla simüle edilir.</span></div><div><b>6</b><span>Tüm yollar cost edge, max 2 pozisyon, heat, structure stop, flow invalidation, trail ve hard-time exit ile korunur.</span></div>';
};

// iOS standalone + black-translucent places page content underneath the system status bar.
addEventListener('DOMContentLoaded',()=>{
  const style=document.createElement('style');
  style.textContent=`@supports (padding-top: env(safe-area-inset-top)){body:before{content:"";position:fixed;z-index:2147483000;top:0;left:0;right:0;height:env(safe-area-inset-top);background:#05070d;pointer-events:none}.dipMain{padding-top:max(10px,env(safe-area-inset-top))}}`;
  document.head.appendChild(style);
});
