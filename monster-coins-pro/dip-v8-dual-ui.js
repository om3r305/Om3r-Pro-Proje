/* Brian DIP V8.2 Dual UI overlay.
   View/config only. Server worker owns all accounting and SHADOW execution. */
const V82_POLICY='dip-v8-dual-20260908.2';
const V82_ENGINE='brian-dip-chart-reader-v8-dual';
const V82_METRIC='target-before-invalidation-v8.2';

const _v82Params=params;
params=function(){
  const p=_v82Params();
  p.config={...(p.config||{}),
    symbols:['ETHUSDT'],auto_universe:false,universe_size:1,
    engine_version:V82_ENGINE,policy_version:V82_POLICY,measurement:V82_METRIC,
    shadow_only:true,live_execution:false,browser_execution:false,server_authoritative:true,
    allow_shadow_short:true,max_shadow_leverage:2,execution_mode:'SHADOW_PAPER',
    sizing_policy:'V8_DUAL_RISK_CAPPED',leverage_policy:'1X_BASE__2X_ONLY_AFTER_40_CALIBRATED_EDGE',
    chart_reader_version:'v8.2-dual'
  };
  return p;
};

const _v82Restore=restore;
restore=function(d){
  _v82Restore(d);
  const c=session?.config;
  if(c){
    c.engine_version=V82_ENGINE;c.policy_version=V82_POLICY;c.measurement=V82_METRIC;
    c.symbols=['ETHUSDT'];c.allow_shadow_short=true;c.max_shadow_leverage=2;
    c.shadow_only=true;c.live_execution=false;c.browser_execution=false;c.server_authoritative=true;
  }
};

const _v82Note=note;
note=function(e){
  const m=e?.metadata||{},lev=Number(m.leverage||1);
  if(m.server_v8&&(e.event_kind==='BUY'||e.event_kind==='SHORT_OPEN')){
    const side=e.event_kind==='SHORT_OPEN'?'SHORT':'LONG';
    return `V8.2 ${side} · ${m.setup||''} · ${lev}x · margin $${Number(m.margin||0).toFixed(2)} · target ${price(m.target)} · iptal ${price(m.stop)} · R:R ${Number(m.rr||0).toFixed(2)}`;
  }
  if(m.server_v8&&(e.event_kind==='SELL'||e.event_kind==='SHORT_CLOSE')){
    return `V8.2 ${e.event_kind==='SHORT_CLOSE'?'SHORT CLOSE':'LONG CLOSE'} · ${lev}x · ${m.exit_reason||'EXIT'}`;
  }
  return _v82Note(e);
};

const _v82UiPatch=v4UiPatch;
v4UiPatch=function(){
  _v82UiPatch();
  const h=document.querySelector('.desktopTitle h1');if(h)h.textContent='Brian V8.2 · Dual Direction Lab';
  const mb=document.querySelector('.mobileBrand b');if(mb)mb.textContent='Brian V8.2 Dual';
  const sub=document.querySelector('.desktopTitle .sub');if(sub)sub.textContent='ETH · LONG + SHORT · server-authoritative · 1x base / edge-gated 2x · SHADOW ONLY';
  const banner=document.querySelector('.dipBanner>div:first-child');if(banner)banner.innerHTML='<strong>BRIAN V8.2 · LONG + SHORT · SHADOW ONLY</strong> · Brian hem dipten LONG hem tepeden SHORT fırsatı arar. 2x yalnız en az 40 sonuçlanmış aynı-politika örneği ve güçlü kalibre edge oluşursa açılır; gerçek emir yoktur.';
  const badge=document.querySelector('.expertModeCard .badge');if(badge)badge.textContent='V8.2 DUAL · LONG + SHORT';
  const expert=document.querySelector('.expertModeCard');if(expert){const b=expert.querySelector('b');if(b)b.textContent='Aynalanmış yapı + order flow + cost/R:R + risk-capped margin';const s=expert.querySelector('small');if(s)s.textContent='ETH only · tek açık pozisyon · 1x normal · 2x yalnız güçlü kalibre edge · SHADOW';}
  if($('startBtn'))$('startBtn').textContent='▶ Brian V8.2 Dual Başlat';
  if($('restartBtn'))$('restartBtn').textContent='↻ V8.2 Dual Temiz Session';
  const rule=document.querySelector('.dipRuleLine');if(rule)rule.innerHTML='V8.2: <b>LONG</b> = dip/reclaim/HH-HL · <b>SHORT</b> = top/rejection/LH-LL → flow + structure + gerçek cost → R:R ≥ 2 → risk-capped size. <b>2x otomatik değildir:</b> ≥40 sonuç, kalibre p ≥68%, Wilson alt sınır ≥55%, ham görüş ≥72%, R:R ≥2.5 ve güçlü flow birlikte gerekir.';
  if($('kpiEngineMeta'))$('kpiEngineMeta').textContent='DUAL SHADOW · 1x base · 2x calibrated edge';
  const title=document.querySelector('#dipLog .title');if(title)title.textContent='LONG / SHORT Açılış-Kapanış Log';
  const lognote=document.querySelector('#dipLog .note');if(lognote)lognote.textContent='Yalnız gerçek shadow pozisyon olayları · BUY/SELL/SHORT OPEN/SHORT CLOSE';
  const radarTitle=document.querySelector('#watchlist .title');if(radarTitle)radarTitle.textContent='ETH Dual Radar · V8.2';
};

const _v82RenderV8ThesisBar=renderV8ThesisBar;
renderV8ThesisBar=function(){
  _v82RenderV8ThesisBar();
  const f=v8Foresight(),bar=$('v7ForesightBar');if(!f||!bar)return;
  const lev=Number(f.leverage||1),policy=String(f.leverage_policy||'BASE_1X');
  const main=bar.querySelector('.v7ForesightMain');
  if(main){const chip=document.createElement('span');chip.innerHTML=`Kaldıraç <strong>${lev}x</strong> · ${v4Esc(policy==='EDGE_GATED_2X'?'EDGE GATED':'BASE')}`;main.appendChild(chip);}
};

addEventListener('load',()=>{try{v4UiPatch();render();}catch(e){console.warn('v82-dual-ui',e);}});
