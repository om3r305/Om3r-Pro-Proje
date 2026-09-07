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

/* V7.1 dashboard clarity + live-view overlay.
   View-only UI changes: no browser execution, no DB deletion, no Phase 3.7 mutation. */
let v7ExpectedSessionId = null;
let v7RestartPending = false;
let v7SessionFenceUntil = 0;
let v7LiveBusy = false;
let v7LiveTimer = null;

function v7ScopeStatusPayload(raw){
  if(!raw || typeof raw!=='object') return raw;
  const currentId=raw.session?.session_id||null;
  const events=Array.isArray(raw.events)?raw.events.filter(e=>!currentId||e?.session_id===currentId):[];
  const snapshot=raw.snapshot && (!currentId||raw.snapshot.session_id===currentId) ? raw.snapshot : null;
  return {...raw,events,snapshot};
}

// Every caller (including legacy 7s UI polling) receives only the currently returned session.
const _v7ScopedApi = api;
api = async function(action,body={}){
  const raw=await _v7ScopedApi(action,body);
  if(action!=='status') return raw;
  if(v7RestartPending){
    return {
      schema_version:raw?.schema_version||'brian.aggressive-dip.status.v3',
      generated_at:raw?.generated_at||iso(),
      shadow_only:true,live_execution:false,
      session:session||raw?.session||null,snapshot:null,events:[],engine_lease:null
    };
  }
  const d=v7ScopeStatusPayload(raw);
  if(Date.now()<v7SessionFenceUntil && v7ExpectedSessionId && d?.session?.session_id && d.session.session_id!==v7ExpectedSessionId){
    return {...d,session,snapshot:null,events:[]};
  }
  return d;
};

const _v7ScopedRestore = restore;
restore = function(raw){
  const d=v7ScopeStatusPayload(raw);
  const incoming=d?.session?.session_id||null;
  const changed=Boolean(incoming && sid && incoming!==sid);
  if(changed) history=[];
  _v7ScopedRestore(d);
  if(incoming){
    v7ExpectedSessionId=incoming;
    history=(Array.isArray(d.events)?d.events:[]).filter(e=>e?.session_id===incoming);
  }else history=[];
};

const _v7CleanStart = start;
start = async function(restart=false){
  if(restart){
    v7RestartPending=true;
    history=[];
    try{renderLog();draw();}catch{}
  }
  const before=sid;
  try{
    await _v7CleanStart(restart);
    if(sid && (restart||sid!==before)){
      v7ExpectedSessionId=sid;
      v7SessionFenceUntil=Date.now()+15000;
      history=[];
      if($('eventRows'))renderLog();
      draw();
    }
  }finally{v7RestartPending=false;}
};

function v7EventLabel(kind,pnlValue){
  const k=String(kind||'').toUpperCase();
  if(k==='BUY')return{label:'ALINDI',tag:'v7TagBuy',row:'v7RowBuy'};
  if(k==='SELL')return{label:'SATILDI',tag:'v7TagSell',row:'v7RowSell'};
  if(k==='SHORT_OPEN')return{label:'SHORT AÇILDI',tag:'v7TagSell',row:'v7RowSell'};
  if(k==='SHORT_CLOSE'){
    const positive=Number(pnlValue||0)>=0;
    return{label:'SHORT KAPANDI',tag:positive?'v7TagBuy':'v7TagSell',row:positive?'v7RowBuy':'v7RowSell'};
  }
  if(k==='DIP_ARMED')return{label:'DİP HAZIR',tag:'v7TagDip',row:'v7RowDip'};
  if(k==='DIP_NEW_LOW')return{label:'YENİ DİP',tag:'v7TagDip',row:'v7RowDip'};
  if(k==='SKIP_CHASE')return{label:'KOVALANMADI',tag:'v7TagWait',row:'v7RowWait'};
  if(k==='NO_CASH')return{label:'KASA YETERSİZ',tag:'v7TagSell',row:'v7RowSell'};
  if(k==='ENGINE_START')return{label:'SESSION BAŞLADI',tag:'v7TagInfo',row:'v7RowInfo'};
  if(k==='ENGINE_PAUSE')return{label:'SESSION DURDU',tag:'v7TagInfo',row:'v7RowInfo'};
  return{label:k.replaceAll('_',' '),tag:'v7TagInfo',row:'v7RowInfo'};
}
function v7SessionEvents(){return (Array.isArray(history)?history:[]).filter(e=>!sid||e?.session_id===sid);}
renderLog = function(){
  const host=$('eventRows');if(!host)return;
  const rows=v7SessionEvents().slice(0,160);
  host.innerHTML=rows.length?rows.map(e=>{
    const ui=v7EventLabel(e.event_kind,e.realized_pnl);
    const pclass=Number(e.realized_pnl||0)>0?'pos':Number(e.realized_pnl||0)<0?'neg':'';
    return `<tr class="${ui.row}">
      <td>${v4Esc(clock(e.observed_at))}</td>
      <td><span class="tag v7EventTag ${ui.tag}">${v4Esc(ui.label)}</span></td>
      <td><b>${v4Esc((e.symbol||'—').replace('USDT',''))}</b></td>
      <td>${v4Esc(price(e.price))}</td>
      <td>${v4Esc(price(e.dip_low))}</td>
      <td>${v4Esc(price(e.entry_price))}</td>
      <td class="${pclass}">${e.realized_pnl==null?'—':v4Esc(pnl(e.realized_pnl))}</td>
      <td class="noteCell">${v4Esc(note(e))}</td>
    </tr>`;
  }).join(''):'<tr class="v7EmptyRow"><td colspan="8">Bu V7 session’da henüz Dip / Alım / Satım olayı yok.</td></tr>';
};

function v7LatestDip(sym){
  const st=states[sym]||{};
  const active=Number(st.dip||st.v4?.armLow||0);
  if(active>0)return{price:active,active:Boolean(st.armed||st.v4?.phase==='ARM_LONG'),source:'state'};
  const e=v7SessionEvents().find(x=>x?.symbol===sym&&(x.event_kind==='DIP_NEW_LOW'||x.event_kind==='DIP_ARMED'));
  const p=Number(e?.dip_low||e?.price||0);
  return p>0?{price:p,active:false,source:'session'}:null;
}
function v7ChartLevel(ctx,y,L,R,w,label,value,color,dash=[5,4],side='left'){
  if(!(Number(value)>0))return;
  const yy=y(Number(value));if(!Number.isFinite(yy))return;
  ctx.save();ctx.setLineDash(dash);ctx.strokeStyle=color;ctx.lineWidth=1.2;
  ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.setLineDash([]);
  ctx.font='600 10px system-ui';
  const text=`${label} ${price(value)}`,tw=ctx.measureText(text).width;
  const x=side==='right'?Math.max(L+4,w-R-tw-12):L+6;
  const ty=Math.max(15,Math.min(yy-5,ctx.canvas.height/(devicePixelRatio||1)-34));
  ctx.fillStyle='rgba(5,10,18,.88)';ctx.fillRect(x-4,ty-11,tw+8,15);
  ctx.fillStyle=color;ctx.fillText(text,x,ty);ctx.restore();
}

draw = function(){
  const cv=$('candleCanvas'),box=$('chartWrap');if(!cv||!box)return;
  const ctx=cv.getContext('2d'),dpr=devicePixelRatio||1,w=Math.max(320,box.clientWidth),h=Math.max(280,box.clientHeight);
  cv.width=w*dpr;cv.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);ctx.fillStyle='#080e17';ctx.fillRect(0,0,w,h);
  const a=(candles[selected]||[]).slice(-100);if(!a.length)return;
  const st=states[selected]||{},q=st.pos||null,dipInfo=v7LatestDip(selected);
  const values=a.flatMap(c=>[Number(c.l),Number(c.h)]).filter(Number.isFinite);
  const lp=Number(live[selected]||a.at(-1)?.c||0),levels=[];
  if(dipInfo?.price>0&&(!lp||Math.abs(dipInfo.price/lp-1)<=.12))levels.push(dipInfo.price);
  if(q)for(const v of [q.entry,q.stop,q.target])if(Number(v)>0)levels.push(Number(v));
  let lo=Math.min(...values,...levels),hi=Math.max(...values,...levels),pad=(hi-lo)*.075||1;lo-=pad;hi+=pad;
  const L=12,R=82,T=12,B=28,plotW=w-L-R,plotH=h-T-B,xw=plotW/a.length,y=v=>T+(hi-v)/(hi-lo)*plotH;
  ctx.lineWidth=1;ctx.font='9px system-ui';
  for(let i=0;i<=5;i++){
    const yy=T+plotH*i/5,v=hi-(hi-lo)*i/5;
    ctx.strokeStyle='#182235';ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();
    ctx.fillStyle='#728096';ctx.fillText(price(v),w-R+8,yy+3);
  }
  a.forEach((c,i)=>{
    const x=L+xw*i+xw/2,up=Number(c.c)>=Number(c.o),col=up?'#0ecb81':'#f6465d';
    ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(x,y(Number(c.h)));ctx.lineTo(x,y(Number(c.l)));ctx.stroke();
    const top=y(Math.max(Number(c.o),Number(c.c))),bot=y(Math.min(Number(c.o),Number(c.c)));
    ctx.fillRect(x-Math.max(1,xw*.27),top,Math.max(2,xw*.54),Math.max(1,bot-top));
  });
  if(dipInfo?.price>0)v7ChartLevel(ctx,y,L,R,w,dipInfo.active?'AKTİF DİP':'SON DİP',dipInfo.price,'#f0b90b',[4,4],'left');
  if(q){
    const side=String(q.side||'LONG').toUpperCase();
    v7ChartLevel(ctx,y,L,R,w,side==='SHORT'?'SHORT GİRİŞ':'ALIM GİRİŞ',q.entry,side==='SHORT'?'#f6465d':'#0ecb81',[2,2],'left');
    v7ChartLevel(ctx,y,L,R,w,'TP',q.target,'#2af0a3',[6,4],'right');
    v7ChartLevel(ctx,y,L,R,w,'SL',q.stop,'#ff6b7a',[6,4],'right');
  }
  if(lp>0){
    const yy=Math.max(T+9,Math.min(h-B-9,y(lp))),up=Number(a.at(-1)?.c)>=Number(a.at(-1)?.o),col=up?'#0ecb81':'#f6465d';
    ctx.setLineDash([3,3]);ctx.strokeStyle=col;ctx.globalAlpha=.65;ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.globalAlpha=1;ctx.setLineDash([]);
    ctx.fillStyle=col;ctx.fillRect(w-R+3,yy-10,R-6,20);ctx.fillStyle='#fff';ctx.font='600 9px system-ui';ctx.fillText(price(lp),w-R+8,yy+3);
  }
  if(dipInfo?.price>0&&lp>0&&Math.abs(dipInfo.price/lp-1)>.12){ctx.font='600 10px system-ui';ctx.fillStyle='#f0b90b';ctx.fillText(`SON DİP ${price(dipInfo.price)} · grafik aralığı dışında`,L+4,T+12);}
  $('chartSymbol').textContent=selected;$('lastPrice').textContent=price(lp);$('chartSub').textContent=`Binance Spot · 1m · ${a.length} mum · V7 session seviyeleri`;
};

const _v7ClarityUiPatch=v4UiPatch;
v4UiPatch=function(){
  _v7ClarityUiPatch();
  const head=document.querySelector('#dipLog .head');
  if(head){
    const n=head.querySelector('.note');if(n)n.textContent='Yalnız mevcut V7 cloud session · Dip / Alım / Satım · otomatik canlı güncelleme';
    const b=head.querySelector('.badge');if(b){b.textContent='BU SESSION · CANLI';b.className='badge green';}
  }
  const legend=document.querySelector('.chartLegend');
  if(legend)legend.innerHTML='<span><i class="legendDot dip"></i>DİP</span><span><i class="legendDot buy"></i>ALIM / GİRİŞ</span><span><i class="legendDot sell"></i>SHORT / SL</span><span><i class="legendDot ma7"></i>TP etiketli</span>';
};

function v7InstallClarityStyles(){
  if(document.getElementById('v7-clarity-styles'))return;
  const s=document.createElement('style');s.id='v7-clarity-styles';
  s.textContent=`
    .dipTable tbody tr td{transition:background .18s ease,border-color .18s ease}
    .dipTable tbody tr.v7RowBuy td{background:rgba(14,203,129,.075);border-bottom-color:rgba(14,203,129,.16)}
    .dipTable tbody tr.v7RowSell td{background:rgba(246,70,93,.075);border-bottom-color:rgba(246,70,93,.16)}
    .dipTable tbody tr.v7RowDip td{background:rgba(240,185,11,.065);border-bottom-color:rgba(240,185,11,.15)}
    .dipTable tbody tr.v7RowWait td{background:rgba(125,142,165,.04)}
    .v7EventTag{min-width:88px;text-align:center;font-weight:800;letter-spacing:.02em}
    .v7TagBuy{color:#0ecb81!important;background:rgba(14,203,129,.13)!important;border-color:rgba(14,203,129,.32)!important}
    .v7TagSell{color:#ff6b7a!important;background:rgba(246,70,93,.13)!important;border-color:rgba(246,70,93,.32)!important}
    .v7TagDip{color:#f0b90b!important;background:rgba(240,185,11,.12)!important;border-color:rgba(240,185,11,.30)!important}
    .v7TagInfo,.v7TagWait{color:#aab6c8!important;background:rgba(125,142,165,.08)!important;border-color:rgba(125,142,165,.2)!important}
    .v7EmptyRow td{text-align:center;color:#8290a5;padding:22px 10px!important}
  `;
  document.head.appendChild(s);
}

async function v7LiveRefresh(){
  if(v7LiveBusy||v7RestartPending||document.visibilityState==='hidden')return;
  v7LiveBusy=true;
  try{const d=await api('status');restore(d);v4UiPatch();render();}
  catch(e){const t=String(e?.message||e);if($('cloudMeta'))$('cloudMeta').textContent=`Canlı görünüm: ${t.slice(0,80)}`;}
  finally{v7LiveBusy=false;}
}
function v7StartLivePolling(){if(v7LiveTimer)clearInterval(v7LiveTimer);v7LiveTimer=setInterval(v7LiveRefresh,3000);}

addEventListener('load',()=>{
  try{v7InstallClarityStyles();v4UiPatch();renderLog();draw();v7StartLivePolling();setTimeout(v7LiveRefresh,800);}
  catch(e){console.warn('dip-v7-clarity',e);}
});
document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible')v7LiveRefresh();});
