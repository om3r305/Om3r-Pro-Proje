/* Brian Dip V4 runtime stability guard.
   SHADOW/PAPER ONLY. Keeps partial market-data/network faults from wedging the UI,
   while failing closed for new entries when cloud lease/persistence is uncertain. */

const V4_RUNTIME_FETCH_TIMEOUT_MS = 6000;
const V4_RUNTIME_MIN_READY_MARKETS = 3;
const V4_RUNTIME_LEASE_GRACE_MS = 25000;
let v4RuntimeLastLeaseOkAt = 0;
let v4RuntimeLeaseDegraded = false;
let v4RuntimeLastDataWarningAt = 0;

function v4RuntimeSetCloud(state, text, klass='amber'){
  if($('cloudState')){$('cloudState').textContent=state;$('cloudState').className=klass;}
  if($('cloudMeta'))$('cloudMeta').textContent=text;
}
function v4RuntimeNetworkish(err){
  const s=String(err?.message||err||'').toLowerCase();
  return /fetch|network|timeout|abort|load failed|connection|offline|http 5\d\d/.test(s);
}
function v4RuntimeSymbolReady(sym){
  const b=v4Bars[sym]||{};
  return ['1m','5m','15m','1h'].every(tf=>Array.isArray(b[tf])&&b[tf].length>=30);
}

// Every REST request is bounded. A dead Binance host can no longer leave the page on
// “market data loading” forever.
v4FetchJson = async function(host,path){
  const ctrl=new AbortController(),timer=setTimeout(()=>ctrl.abort('timeout'),V4_RUNTIME_FETCH_TIMEOUT_MS);
  try{
    const r=await fetch(host+path,{cache:'no-store',signal:ctrl.signal});
    if(!r.ok)throw Error(`HTTP ${r.status} ${path}`);
    return await r.json();
  }catch(e){
    if(e?.name==='AbortError'||String(e).includes('timeout'))throw Error(`TIMEOUT ${host}${path}`);
    throw e;
  }finally{clearTimeout(timer);}
};

// Race the official spot mirrors and accept the first healthy response instead of waiting
// serially for several dead hosts.
v4Spot = async function(path){
  const hosts=['https://api.binance.com','https://api1.binance.com','https://api3.binance.com'];
  try{return await Promise.any(hosts.map(h=>v4FetchJson(h,path)));}
  catch(e){throw Error(`Binance spot unavailable: ${String(e?.message||e).slice(0,120)}`);}
};
v4Perp = async function(path){return v4FetchJson('https://fapi.binance.com',path);};

// One bad altcoin must not reject an entire scanner/history batch.
v4Batch = async function(items,n,fn){
  const failures=[];
  for(let i=0;i<items.length;i+=Math.max(1,n)){
    const part=items.slice(i,i+Math.max(1,n));
    const settled=await Promise.allSettled(part.map(fn));
    settled.forEach((r,j)=>{if(r.status==='rejected')failures.push({item:part[j],reason:String(r.reason?.message||r.reason||'failed')});});
  }
  if(failures.length){
    const now=Date.now();
    if(now-v4RuntimeLastDataWarningAt>15000){console.warn('V4 partial market-data batch',failures.slice(0,8));v4RuntimeLastDataWarningAt=now;}
  }
  return{ok:items.length-failures.length,failures};
};

const _v4RuntimeNativeLoadHistory=v4LoadHistory;
v4LoadHistory=async function(){
  const overlay=$('chartOverlay'),label=overlay?.querySelector('span');
  if(overlay)overlay.classList.add('show');
  if(label)label.textContent='V4 market verisi yükleniyor…';
  try{
    const result=await _v4RuntimeNativeLoadHistory();
    const btcReady=v4RuntimeSymbolReady('BTCUSDT');
    const open=Object.keys(states).filter(s=>states[s]?.pos);
    const ready=v4Universe.filter(s=>s!=='BTCUSDT'&&v4RuntimeSymbolReady(s));
    const next=[];
    for(const s of [...open,...ready])if(s&&!next.includes(s)&&next.length<V4_UNIVERSE_SIZE)next.push(s);
    v4Universe=next;v4UniverseUpdatedAt=v4Now();v4Universe.forEach(v4Ensure);
    if(!btcReady||ready.length<V4_RUNTIME_MIN_READY_MARKETS){
      throw Error(`Yeterli native market verisi yok (${ready.length}/${V4_RUNTIME_MIN_READY_MARKETS}).`);
    }
    if(!selected||!v4Universe.includes(selected))selected=v4Universe[0]||ready[0]||'ETHUSDT';
    if($('radarStatus'))$('radarStatus').textContent=ready.length>=V4_UNIVERSE_SIZE?'V4 READY':`DATA ${ready.length}/${V4_UNIVERSE_SIZE}`;
    return result;
  }finally{
    if(overlay)overlay.classList.remove('show');
  }
};
historyLoad=v4LoadHistory;

// Lease handling: an ownership conflict still stops this browser. A short network/API fault
// does not turn a healthy RUNNING session into a zombie “stopped” UI. During uncertainty,
// v4CloudFault blocks NEW entries; existing positions continue to be risk-managed locally.
v4EnsureEngineLease = async function(){
  if(!session||session.status!=='RUNNING'||v4NeedsRestart||!sid)return false;
  try{
    await api('engine_check',{session_id:sid,engine_token:token()});
    v4RuntimeLastLeaseOkAt=Date.now();v4RuntimeLeaseDegraded=false;running=true;
    return true;
  }catch(e){
    const msg=String(e?.message||e);
    if(/UNAUTHORIZED_DIP_ENGINE|BRIAN_DIP_ENGINE_LEASE_ACTIVE/.test(msg)){
      try{
        const c=await api('claim_engine',{session_id:sid,engine_token:token()});
        running=true;v4RuntimeLastLeaseOkAt=Date.now();v4RuntimeLeaseDegraded=false;v4CloudFault=false;
        try{event('INFO',null,null,{metadata:{expert_v4:true,info:'ENGINE_LEASE_CLAIMED',lease_generation:c.lease_generation||null}})}catch{}
        toast('V4 motor bu cihaz tarafından devralındı.');
        return true;
      }catch(claimErr){
        const cm=String(claimErr?.message||claimErr);
        if(cm.includes('BRIAN_DIP_ENGINE_LEASE_ACTIVE')){running=false;v4RuntimeLeaseDegraded=false;return false;}
        e=claimErr;
      }
    }
    const now=Date.now(),feedFresh=Boolean(lastWs&&now-lastWs<15000),recentLease=!v4RuntimeLastLeaseOkAt||now-v4RuntimeLastLeaseOkAt<V4_RUNTIME_LEASE_GRACE_MS;
    if(v4RuntimeNetworkish(e)&&feedFresh&&recentLease){
      running=true;v4RuntimeLeaseDegraded=true;v4CloudFault=true;
      v4RuntimeSetCloud('SYNC BEKLE','Geçici cloud/lease bağlantı sorunu · yeni girişler fail-closed, market motoru recovery deniyor.','amber');
      return true;
    }
    running=false;throw e;
  }
};

// Surface recovery state without changing the existing V4/V5 decision logic.
setInterval(()=>{
  if(!session||session.status!=='RUNNING')return;
  const feedAge=lastWs?Date.now()-lastWs:Infinity;
  if(running&&feedAge>10000&&document.visibilityState==='visible'){
    try{connect();}catch(e){v4ReportRuntimeFault?.('runtime-guard-ws',e);}
  }
  if(v4RuntimeLeaseDegraded&&feedAge<10000&&$('feedState')){
    $('feedState').textContent='LIVE · SYNC';$('feedState').className='amber';
  }
},5000);

/* BRIAN_DIP_LIVE_V6 — production recovery + V5 reasoner/executor bridge.
   SHADOW/PAPER ONLY. Does not add any live order endpoint and does not mutate MAIN/Phase 3.7. */
let v6LastHealthyUniverse = [];

function v6UniqueSymbols(items){
  const out=[];
  for(const s of items||[]){if(typeof s==='string'&&s.endsWith('USDT')&&s!=='BTCUSDT'&&!out.includes(s))out.push(s);}
  return out;
}
function v6RememberHealthyUniverse(){
  const ready=v6UniqueSymbols(v4Universe).filter(v4RuntimeSymbolReady);
  if(ready.length>=V4_RUNTIME_MIN_READY_MARKETS)v6LastHealthyUniverse=ready.slice(0,V4_UNIVERSE_SIZE);
}
function v6RecoverUniverse(){
  const open=Object.keys(states).filter(s=>states[s]?.pos);
  const configured=Array.isArray(session?.config?.symbols)?session.config.symbols:[];
  const runtimeReady=Object.keys(states).filter(s=>s!=='BTCUSDT'&&v4RuntimeSymbolReady(s));
  const currentReady=v6UniqueSymbols(v4Universe).filter(v4RuntimeSymbolReady);
  const rememberedReady=v6UniqueSymbols(v6LastHealthyUniverse).filter(v4RuntimeSymbolReady);
  const next=v6UniqueSymbols([...open,...currentReady,...rememberedReady,...configured,...runtimeReady]).slice(0,V4_UNIVERSE_SIZE);
  if(next.length<V4_RUNTIME_MIN_READY_MARKETS)return false;
  v4Universe=next;v4UniverseUpdatedAt=v4Now();v4Universe.forEach(v4Ensure);v6LastHealthyUniverse=[...next];
  if(!selected||!v4Universe.includes(selected))selected=v4Universe[0]||'ETHUSDT';
  if($('radarStatus'))$('radarStatus').textContent=running?'BRIAN V5 LIVE':`DATA ${next.length}/${V4_UNIVERSE_SIZE}`;
  try{renderRadar();}catch{}
  return true;
}

const _v6PriorLoadHistory=v4LoadHistory;
v4LoadHistory=async function(){
  const before=v6UniqueSymbols(v4Universe);
  if(before.length>=V4_RUNTIME_MIN_READY_MARKETS)v6LastHealthyUniverse=[...before];
  try{
    const result=await _v6PriorLoadHistory();
    if(!v4Universe.length)v6RecoverUniverse();
    v6RememberHealthyUniverse();
    return result;
  }catch(e){
    if(before.length){v4Universe=[...before];v4UniverseUpdatedAt=v4Now();}
    v6RecoverUniverse();
    throw e;
  }finally{
    if(!v4Universe.length)v6RecoverUniverse();
    try{renderRadar();}catch{}
  }
};
historyLoad=v4LoadHistory;

// The V5 layer is loaded after this guard. Install the bridge on the next task so
// it sees V5 globals and can extend only the reasoner entry path.
setTimeout(()=>{
  if(typeof v5ReasonerEntry!=='function'||typeof v5ControllerDecision!=='function')return;
  v5ReasonerEntry=function(st,ctx,p,d){
    if(!st||!ctx||!d||!['BUY','SELL'].includes(d.action))return false;
    if(d.hardDrift||d.ood||(d.vetoReasons||[]).length)return false;
    const cost=Math.max(0,Number(d.costBps||0)),net=Number(d.netEdgeBps||0);
    const setup=String(d.setup||'');
    const confFloor=['PULLBACK_CONTINUATION','TREND_EXHAUSTION'].includes(setup)?.62:.64;
    if(Number(d.confidence||0)<confFloor||Number(d.agreement||0)<.56)return false;
    if(!(net>Math.max(3,cost*.45))||Number(ctx.edgeRatio||0)<V4_COST_EDGE_MULT)return false;

    if(d.action==='BUY'){
      const allowed=['LIQUIDITY_SWEEP_REVERSAL','RANGE_REJECTION','PULLBACK_CONTINUATION','BREAKOUT_RETEST_CONTINUATION','DIP_RECLAIM'];
      if(!allowed.includes(setup))return false;
      const flowOk=Number(ctx.flow?.ofi||0)>=.03&&Number(ctx.bk?.pressure||0)>=1.01;
      const trendFloor=['LIQUIDITY_SWEEP_REVERSAL','RANGE_REJECTION','DIP_RECLAIM'].includes(setup)?-.28:-.12;
      if(!flowOk||Number(ctx.btcLongRisk||0)<=-.42||Number(ctx.htfLong||0)<=trendFloor)return false;
      return v4Open(st,ctx,'LONG',`V5_${setup}`);
    }

    if(!v4FuturesSymbols.has(st.symbol))return false;
    const pctx=v4Context(st.symbol,'USDM_PERP');if(!pctx)return false;
    const pd=v5ControllerDecision(st.symbol,pctx,pctx.bk.mid||Number(p));
    v5DecisionBySymbol[st.symbol]=pd;st.v4.brainV5=v5SlimDecision(pd);
    if(pd.action!=='SELL'||pd.hardDrift||pd.ood||(pd.vetoReasons||[]).length)return false;
    const psetup=String(pd.setup||setup),allowed=['TREND_EXHAUSTION','DOWNTREND_BREAK','FAILED_BREAK_REVERSAL'];
    if(!allowed.includes(psetup))return false;
    const pcost=Math.max(0,Number(pd.costBps||0)),pnet=Number(pd.netEdgeBps||0);
    if(Number(pd.confidence||0)<.62||Number(pd.agreement||0)<.56||!(pnet>Math.max(3,pcost*.45)))return false;
    if(Number(pctx.edgeRatio||0)<V4_COST_EDGE_MULT)return false;
    const flowOk=Number(pctx.flow?.ofi||0)<=-.03&&Number(pctx.bk?.pressure||1)<=.99;
    const htfOk=Number(pctx.htfShort||0)>-.08||psetup==='FAILED_BREAK_REVERSAL';
    const fundingOk=Number(pctx.fundingRate||0)>-0.0005;
    if(!flowOk||!htfOk||!fundingOk)return false;
    return v4Open(st,pctx,'SHORT',`V5_${psetup}`);
  };
},0);

setInterval(()=>{
  if(!session||session.status!=='RUNNING')return;
  if(v4Universe.length>=V4_RUNTIME_MIN_READY_MARKETS){v6RememberHealthyUniverse();return;}
  v6RecoverUniverse();
},3000);
