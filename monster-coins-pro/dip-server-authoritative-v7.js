/* Brian DIP V7 browser handoff.
   SHADOW ONLY. The browser keeps Binance charts/radar rendering, but it no longer owns entries,
   exits, snapshots, heartbeat, or portfolio accounting. Supabase cron + brian-dip-shadow-worker
   is authoritative at 1-minute cadence even when iOS suspends/closes this page. */

const BRIAN_DIP_SERVER_AUTHORITATIVE_V7 = true;
let v7ServerRuntime = null;

// Older V4/V5 files install 7-second engine_check/claim_engine timers before this V7 overlay loads.
// Intercept those lease-only actions so a view-only browser can never keep refreshing the legacy
// engine lease and block the server-authoritative takeover. All real session/status controls still
// go to brian-dip-trader normally.
const _v7Api = api;
api = async function(action,body={}){
  if(action==='engine_check'||action==='claim_engine'){
    return {status:'SERVER_AUTHORITATIVE_VIEW_ONLY',browser_execution:false,shadow_only:true,live_execution:false};
  }
  return _v7Api(action,body);
};
v4EnsureEngineLease = async function(){
  running = Boolean(session?.status === 'RUNNING');
  return running;
};

// Disable every browser execution path after all V4/V5 wrappers have loaded.
v4Evaluate = function(){ return; };
snapshot = async function(){ return; };

// Keep local UI helpers from persisting diagnostic events. Server events remain append-only in DB
// and arrive through status().
event = function(kind,sym,p,extra={}){
  return {
    event_id:`dip-browser-view-${crypto.randomUUID()}`,
    session_id:sid,
    observed_at:iso(),
    event_kind:kind,
    symbol:sym||null,
    price:Number.isFinite(Number(p))?Number(p):null,
    metadata:{browser_view_only:true,...(extra.metadata||{})}
  };
};

const _v7Restore = restore;
restore = function(d){
  _v7Restore(d);
  v7ServerRuntime = d?.snapshot?.state?.serverRuntime || null;
  // Session state, not a browser engine lease, is the source of truth now.
  running = Boolean(session?.status === 'RUNNING');
  if(v7ServerRuntime?.authoritative === true){
    v4CloudFault=false;
    if($('cloudState')){$('cloudState').textContent='BULUT V7';$('cloudState').className='pos';}
    if($('cloudMeta'))$('cloudMeta').textContent=`Server karar ${clock(v7ServerRuntime.generated_at||d?.snapshot?.observed_at)} · 1 dk cadence`;
  }
};

status = async function(initial=false){
  try{
    const d=await api('status');
    restore(d);
    v4UiPatch();
    render();
    if(initial)toast(v7ServerRuntime?.authoritative===true?'Brian Dip V7 bulutta 24/7 aktif.':'Brian Dip V7 server devri bekleniyor.');
  }catch(e){
    const t=String(e?.message||e);
    if($('cloudState')){$('cloudState').textContent='SERVER CHECK';$('cloudState').className='amber';}
    if($('cloudMeta'))$('cloudMeta').textContent=t.slice(0,100);
    if(initial)toast(t);
  }
};

start = async function(restart=false){
  v4Booting=true;
  try{
    await v4LoadHistory();
    const p=params();
    // Persist the V7 ownership contract in the append-only session START config. The worker can
    // distinguish V7 cloud sessions from legacy browser-owned V4/V5 sessions without guessing.
    p.config={...p.config,symbols:[...v4Universe],server_authoritative:true,sizing_policy:'BRAIN_CONFIDENCE_V7',browser_execution:false};
    const d=await api(restart?'restart':'start',p);
    if(!restart&&d.status==='RESUMED'){
      await status(false);
      connect();
      render();
      toast('Aynı V7 cloud session devam ediyor.');
      return;
    }
    const cfg={...p.config,...(d.config||{}),symbols:[...v4Universe],engine_version:DIP_EXPERT_V4,allow_shadow_short:true,max_shadow_leverage:1,server_authoritative:true,sizing_policy:'BRAIN_CONFIDENCE_V7',browser_execution:false};
    session={session_id:d.session_id,status:'RUNNING',started_at:d.started_at||iso(),starting_equity:d.starting_equity,trade_notional:d.trade_notional,config:cfg};
    sid=d.session_id;
    v4ClearRuntimeStates();
    book={start:Number(d.starting_equity),cash:Number(d.starting_equity),realized:0,trades:0,wins:0,losses:0,cfg};
    history=[];
    v4NeedsRestart=false;
    v4CloudFault=false;
    v4SessionLossLockUntil=0;
    v4ClosedOutcomes.splice(0);
    for(const k of Object.keys(v4PairGuard))delete v4PairGuard[k];
    for(const s of v4Universe)v4SeedState(s);
    running=true;
    v7ServerRuntime=null;
    connect();
    v4UiPatch();
    render();
    toast(restart?'Yeni V7 cloud shadow session açıldı.':'Brian Dip V7 cloud session başladı.');
    // Cron owns the first authoritative snapshot; refresh shortly without creating a browser lease.
    setTimeout(()=>status(false).catch(()=>{}),4000);
  }catch(e){toast(String(e.message||e));}
  finally{v4Booting=false;render();}
};

pause = async function(){
  try{
    await api('pause');
    running=false;
    if(session)session.status='PAUSED';
    render();
    toast('Brian Dip V7 cloud session pause edildi.');
  }catch(e){toast(String(e.message||e));}
};

const _v7UiPatch = v4UiPatch;
v4UiPatch = function(){
  _v7UiPatch();
  const h=document.querySelector('.desktopTitle h1');if(h)h.textContent='Brian Dip V7 · Cloud Expert';
  const mb=document.querySelector('.mobileBrand b');if(mb)mb.textContent='Brian Dip V7';
  const sub=document.querySelector('.desktopTitle .sub');if(sub)sub.textContent='SERVER AUTHORITATIVE · 24/7 · Binance public market data · brain-confidence sizing · SHADOW ONLY';
  const banner=document.querySelector('.dipBanner>div:first-child');if(banner)banner.innerHTML='<strong>BRIAN DIP V7 · BULUT 24/7 · SHADOW ONLY</strong> · Telefon/sayfa yalnız görüntüler. Giriş, çıkış, stop, target, time-exit, öğrenme ve kasa Supabase server worker tarafından her dakika yürütülür.';
  const badge=document.querySelector('.expertModeCard .badge');if(badge)badge.textContent='BRAIN V7 · SERVER AUTHORITATIVE';
  const expert=document.querySelector('.expertModeCard');if(expert){const b=expert.querySelector('b');if(b)b.textContent='Brian karar verir · confidence/agreement/net-edge ile dinamik pozisyon boyutu';const s=expert.querySelector('small');if(s)s.textContent='max 2 pozisyon · toplam heat ≤ %80 · tek pozisyon ≤ %72 · 1x SHADOW · güçlü sinyal $500+ olabilir';}
  if($('startBtn'))$('startBtn').textContent='▶ Brian Dip V7 Cloud Başlat';
  if($('restartBtn'))$('restartBtn').textContent='↻ V7 Temiz Cloud Session Restart';
  const rule=document.querySelector('.dipRuleLine');if(rule)rule.innerHTML='V7: <b>Brian expert committee</b> → setup/rejim + wick/reclaim + trend/momentum + OFI/depth + BTC risk + gerçek cost → <b>brain quality</b> → risk/heat kontrollü dinamik size. Çok güçlü sinyalde $500+ mümkündür; zayıf sinyalde küçük kalır veya WAIT. Browser execution = 0.';
  if($('radarStatus'))$('radarStatus').textContent=running?(v7ServerRuntime?.authoritative?'V7 CLOUD LIVE':'V7 HANDOFF'):'V7 WAIT';
};

const _v7RenderKpi = renderKpi;
renderKpi = function(){
  _v7RenderKpi();
  if($('kpiEngine')){
    $('kpiEngine').textContent=running?(v7ServerRuntime?.authoritative?'BRIAN V7 CLOUD':'V7 HANDOFF'):'V7 IDLE';
    $('kpiEngine').className=`value ${running?'pos':'amber'}`;
  }
  if($('kpiEngineMeta'))$('kpiEngineMeta').textContent=running
    ?(v7ServerRuntime?.authoritative?'SERVER 24/7 · brain sizing · browser bağımsız':'Server ilk authoritative snapshot bekleniyor')
    :'Cloud session kapalı';
  if($('kpiOpenMeta')&&running)$('kpiOpenMeta').textContent=`max 2 · heat ≤80% · brain size ≤72% · açık ${metrics().open}`;
};

const _v7Note = note;
note = function(e){
  const m=e?.metadata||{};
  if(m.server_v7&&(e.event_kind==='BUY'||e.event_kind==='SHORT_OPEN')){
    const q=Number(m.brain_quality||0)*100, f=Number(m.actual_fraction||0)*100;
    return `Brian V7 ${m.mode||''} · size ${Number(e.notional||0).toFixed(2)}$ (${f.toFixed(0)}%) · Q ${q.toFixed(0)}% · net ${Number(m.brain_net_edge_bps||0).toFixed(0)}bps`;
  }
  if(m.server_v7&&(e.event_kind==='SELL'||e.event_kind==='SHORT_CLOSE'))return `V7 ${m.exit_reason||'EXIT'} · server lifecycle`;
  if(m.server_v7&&m.info==='SERVER_AUTHORITATIVE_TAKEOVER_V7')return 'Browser → server 24/7 devri tamamlandı';
  return _v7Note(e);
};

addEventListener('load',()=>{try{v4UiPatch();render();}catch(e){console.warn('dip-v7-ui',e);}});
