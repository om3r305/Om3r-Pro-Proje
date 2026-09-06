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
