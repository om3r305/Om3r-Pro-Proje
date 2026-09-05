/* Expert V4 hotfix — dynamic portfolio metrics + browser liveness diagnostics + stale-safe engine takeover.
   SHADOW/PAPER only. Does not touch frozen Brian Phase 3.7. */

// Legacy V1/V2 render() still references these readout nodes. V4 no longer displays them,
// but keeping hidden compatibility shims prevents a null dereference from killing the lifecycle.
for(const id of ['ruleRebound','ruleTp']){
  if(!document.getElementById(id)){
    const x=document.createElement('span');x.id=id;x.hidden=true;x.setAttribute('aria-hidden','true');document.body.appendChild(x);
  }
}

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
    $('kpiEngine').textContent='EXPERT V4';$('kpiEngine').className='value pos';
    if($('kpiEngineMeta'))$('kpiEngineMeta').textContent=`LIVE · AUTO ${v4Universe.length||V4_UNIVERSE_SIZE} · max ${V4_MAX_OPEN} · feed ${Math.round(age/1000)}s`;
  }else if(running){
    $('kpiEngine').textContent='V4 STALE';$('kpiEngine').className='value amber';
    if($('kpiEngineMeta'))$('kpiEngineMeta').textContent='Session sahibi bu cihaz · Binance feed bayat, reconnect deneniyor';
  }else if(session?.status==='RUNNING'){
    $('kpiEngine').textContent='V4 VIEW';$('kpiEngine').className='value amber';
    if($('kpiEngineMeta'))$('kpiEngineMeta').textContent='Başka canlı lease var veya 25 sn devralma süresi bekleniyor';
  }else{
    $('kpiEngine').textContent='V4 IDLE';$('kpiEngine').className='value amber';
  }
};

// iOS standalone + black-translucent places page content underneath the system status bar.
addEventListener('DOMContentLoaded',()=>{
  const style=document.createElement('style');
  style.textContent=`@supports (padding-top: env(safe-area-inset-top)){body:before{content:"";position:fixed;z-index:2147483000;top:0;left:0;right:0;height:env(safe-area-inset-top);background:#05070d;pointer-events:none}.dipMain{padding-top:max(10px,env(safe-area-inset-top))}}`;
  document.head.appendChild(style);
});
