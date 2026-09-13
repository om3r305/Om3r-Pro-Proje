'use strict';
(()=>{
  const ENDPOINT=()=>`${ROOT}/brian-development-status`;
  const $=(q,r=document)=>r.querySelector(q);
  const $$=(q,r=document)=>Array.from(r.querySelectorAll(q));
  const clamp=n=>Math.max(0,Math.min(100,Number(n)||0));
  const fmt=n=>Number.isFinite(Number(n))?`${Number(n).toFixed(1)}%`:'—';
  const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const state={data:null,busy:false,last:0,error:null};

  function component(id){return state.data?.components?.find?.(x=>x.id===id)||null}
  function score(c,field='maturity_pct'){
    if(!c)return null;
    if(Number(c.evidence_pct||0)<10)return null;
    const n=Number(c[field]);
    return Number.isFinite(n)?clamp(n):null;
  }
  function overall(k){const n=Number(state.data?.overall?.[k]);return Number.isFinite(n)?clamp(n):null}
  function energy(){
    const h=overall('data_exchange_health_pct')||0,c=overall('evidence_confidence_pct')||0,p=clamp(h*.75+c*.25);
    return {p,label:p>=85?'YÜKSEK':p>=60?'ORTA':p>=35?'DÜŞÜK':'KRİTİK'};
  }
  function derivedLearning(){
    const lab=component('lab'),q=overall('measured_quality_pct')||0,m=score(lab)||0;
    return clamp(m*.6+q*.4);
  }
  function decision(){
    const a=component('alpha');
    return score(a,'quality_pct') ?? overall('measured_quality_pct');
  }
  function organRows(){
    const a=component('alpha'),s=component('sensors'),w=component('world'),l=component('lab');
    const health=overall('data_exchange_health_pct')||0,conf=overall('evidence_confidence_pct')||0;
    return [
      ['🧠','Beyin - ALPHA',score(a)],['👁','Göz / Sinir',score(s)],['🫁','Sol Akciğer',score(w)],
      ['🫁','Sağ Akciğer',score(w,'quality_pct')],['🫀','Kalp / Enerji',health],['◒','Karaciğer',score(w,'quality_pct')],
      ['◉','Mide / Bağırsak',score(l,'quality_pct')],['🦴','İskelet / Yapı',conf],['✹','Bağışıklık',Math.min(100,(health+conf)/2)],
      ['🧬','DNA / Çekirdek',Math.min(100,((score(l)||0)+conf)/2)]
    ];
  }
  function activity(){
    const out=[];
    (state.data?.feature_signals||[]).slice(0,3).forEach(x=>out.push(['DENEY',x.experiment_id||x.decision||'Yeni deney sonucu']));
    (state.data?.capability_gaps||[]).slice(0,2).forEach(x=>out.push([String(x.severity||'GAP').toUpperCase(),`${x.domain||''} · ${x.capability_id||''}`]));
    (state.data?.strengths||[]).slice(0,2).forEach(x=>out.push(['GÜÇLÜ',`${x.name||'Güçlü alan'} · ${fmt(x.maturity_pct)}`]));
    return out.slice(0,6);
  }

  function injectCss(){
    if($('#baxRefCss'))return;
    const s=document.createElement('style');s.id='baxRefCss';s.textContent=`
#brianAnatomyLive{background:#010611!important;overflow:auto!important;overscroll-behavior:contain}
#brianAnatomyLive .bal-head,#brianAnatomyLive #balBody{display:none!important}
#brianAnatomyLive .bal-shell{min-height:100vh!important;padding:0!important;background:#010611!important;display:flex!important;justify-content:center!important;align-items:flex-start!important}
.bax-ref-wrap{position:relative;width:min(100%,1448px);aspect-ratio:1448/1086;flex:0 0 auto;background:#010611 url('/brian-anatomy-dashboard-reference.webp?v=20260913-live1') center top/100% 100% no-repeat;overflow:hidden;box-shadow:0 0 80px #000;color:#eaffff;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.bax-hit{position:absolute;z-index:10;border:0;background:transparent;cursor:pointer}
.bax-refresh{left:80.8%;top:1%;width:10.1%;height:5.4%}.bax-close{right:.3%;top:.4%;width:2.6%;height:4.2%;opacity:0;transition:.2s}.bax-close:hover{opacity:1;background:#ff476b;border-radius:50%;color:#fff}.bax-close:after{content:'×';font-size:18px;font-weight:900}
.bax-val{position:absolute;z-index:8;display:flex;align-items:center;justify-content:center;min-width:4.8%;height:2.45%;padding:0 .35%;border-radius:5px;background:linear-gradient(180deg,rgba(4,23,40,.98),rgba(2,13,26,.98));box-shadow:0 0 11px rgba(37,225,255,.10);font-weight:900;color:#efffff;letter-spacing:.015em;font-size:clamp(8px,1.14vw,16px);line-height:1;white-space:nowrap}
.bax-v-general{left:39.25%;top:10%}.bax-v-health{left:49.15%;top:10%}.bax-v-learn{left:58.45%;top:10%}.bax-v-decision{left:67.25%;top:10%}.bax-v-energy{left:82%;top:10.55%;min-width:7.8%;height:2.55%;font-size:clamp(9px,1.23vw,18px);color:#63f4b0}
.bax-v-alpha{left:10.35%;top:19.25%;color:#64f4af}.bax-v-sensors{left:10.35%;top:31.15%;color:#55eaff}.bax-v-world{left:10.35%;top:43%;color:#6cf4a6}.bax-v-liver{left:10.35%;top:55.25%;color:#57edd1}.bax-v-gut{left:10.35%;top:67.4%;color:#d9ef7b}.bax-v-dna{left:87.1%;top:19.35%;color:#61eaff}.bax-v-cosmic{left:84.1%;top:69.05%;color:#57e9d0}
.bax-sync{position:absolute;right:8.4%;top:5%;z-index:9;color:#79a8ba;font-size:clamp(5px,.48vw,8px);font-weight:700;white-space:nowrap;text-shadow:0 1px 4px #000}.bax-sync.ok:before{content:'● ';color:#52efad}.bax-sync.bad:before{content:'● ';color:#ff6b7d}
.bax-organs{position:absolute;z-index:7;left:28.25%;top:78.58%;width:39.4%;height:14.2%;display:grid;grid-template-columns:1fr 1fr;grid-auto-rows:1fr;column-gap:.7%;row-gap:.6%;padding:2.45% .8% .65%;box-sizing:border-box;background:linear-gradient(180deg,rgba(2,17,31,.98),rgba(2,12,24,.99));border:1px solid rgba(39,201,239,.55);border-radius:9px;box-shadow:inset 0 0 25px rgba(23,190,230,.04)}
.bax-organs:before{content:'⚕  ORGAN DURUM LİSTESİ';position:absolute;left:2.2%;top:4%;font-size:clamp(6px,.69vw,10px);font-weight:900;color:#e6fbff;letter-spacing:.035em}.bax-organ{display:grid;grid-template-columns:minmax(0,1fr) 22% 29%;align-items:center;gap:1.5%;padding:0 3%;border:1px solid rgba(55,183,220,.15);border-radius:4px;background:rgba(4,27,43,.72);font-size:clamp(5px,.59vw,8.5px);color:#b9d9e5;white-space:nowrap}.bax-organ b{color:#f5ffff;text-align:right;font-size:1.05em}.bax-organ em{font-style:normal;color:#53efbc;text-align:right}.bax-organ em:before{content:'● ';text-shadow:0 0 6px #52efb5}
.bax-events{position:absolute;z-index:7;left:68.1%;top:78.58%;width:28.45%;height:14.2%;padding:2.45% .75% .55%;box-sizing:border-box;background:linear-gradient(180deg,rgba(2,17,31,.98),rgba(2,12,24,.99));border:1px solid rgba(39,201,239,.55);border-radius:9px;box-shadow:inset 0 0 25px rgba(23,190,230,.04);overflow:hidden}.bax-events:before{content:'⚡  SON GELİŞİM FAALİYETLERİ';position:absolute;left:3%;top:4%;font-size:clamp(6px,.69vw,10px);font-weight:900;color:#e6fbff;letter-spacing:.035em}.bax-event{height:16%;display:grid;grid-template-columns:14% 1fr;align-items:center;border-bottom:1px solid rgba(61,172,208,.13);font-size:clamp(5px,.56vw,8px);color:#9bc1d0;overflow:hidden}.bax-event b{color:#69bde5;font-size:.95em}.bax-event span{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.bax-empty{display:grid;place-items:center;height:100%;font-size:clamp(6px,.6vw,9px);color:#7292a2}
.bax-mobile{display:none}
@media(max-width:800px){
  #brianAnatomyLive{overflow-x:hidden!important;overflow-y:auto!important;-webkit-overflow-scrolling:touch;background:radial-gradient(circle at 50% 12%,#06243b 0,#020914 34%,#01050d 68%,#01040a 100%)!important}
  #brianAnatomyLive .bal-shell{display:block!important;width:100%!important;min-height:100dvh!important;overflow:visible!important}
  .bax-ref-wrap{width:100%!important;min-width:0!important;max-width:none!important;aspect-ratio:auto!important;min-height:100dvh!important;background:none!important;overflow:visible!important;box-shadow:none!important;padding:0 0 calc(22px + env(safe-area-inset-bottom))!important;box-sizing:border-box}
  .bax-ref-wrap>.bax-hit,.bax-ref-wrap>.bax-sync,.bax-ref-wrap>.bax-val,.bax-ref-wrap>.bax-organs,.bax-ref-wrap>.bax-events{display:none!important}
  .bax-mobile{display:block;min-height:100dvh;background:linear-gradient(180deg,rgba(1,7,15,.18),rgba(1,5,12,.96));}
  .bax-m-head{position:sticky;top:0;z-index:40;display:flex;align-items:center;justify-content:space-between;gap:10px;padding:calc(10px + env(safe-area-inset-top)) 14px 10px;background:rgba(1,8,17,.86);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border-bottom:1px solid rgba(61,220,255,.16)}
  .bax-m-title{min-width:0}.bax-m-kicker{font-size:10px;font-weight:900;letter-spacing:.14em;color:#44e5ff}.bax-m-title h2{margin:3px 0 0;font-size:16px;line-height:1.1;color:#f2feff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .bax-m-actions{display:flex;align-items:center;gap:8px;flex:none}.bax-m-live{font-size:10px;font-weight:800;color:#74f6bf;white-space:nowrap}.bax-m-live:before{content:'● ';text-shadow:0 0 10px #57f3b3}.bax-m-live.bad{color:#ff8191}.bax-m-live.bad:before{color:#ff6579;text-shadow:0 0 10px #ff6579}
  .bax-m-btn{width:34px;height:34px;border-radius:11px;border:1px solid rgba(73,220,255,.28);background:rgba(8,35,52,.82);color:#dbfbff;font-size:18px;font-weight:900;display:grid;place-items:center;padding:0}
  .bax-m-metrics{display:grid;grid-template-columns:1fr 1fr;gap:9px;padding:12px 12px 10px}
  .bax-metric{min-height:76px;border:1px solid rgba(45,211,247,.22);border-radius:15px;padding:11px 12px;background:linear-gradient(145deg,rgba(6,31,48,.92),rgba(3,16,29,.94));box-shadow:inset 0 0 24px rgba(27,192,236,.035)}
  .bax-metric span{display:block;font-size:9px;font-weight:800;letter-spacing:.06em;color:#77aabd;text-transform:uppercase}.bax-metric strong{display:block;margin-top:6px;font-size:23px;line-height:1;color:#f0feff}.bax-metric.energy strong{color:#66f4b6}.bax-metric.wide{grid-column:1/-1;display:flex;align-items:center;justify-content:space-between;min-height:58px}.bax-metric.wide strong{margin:0;font-size:20px}
  .bax-m-hero{position:relative;height:48dvh;min-height:390px;max-height:520px;margin:2px 12px 12px;border:1px solid rgba(40,211,248,.24);border-radius:22px;overflow:hidden;background:#020b15 url('/brian-anatomy-dashboard-reference.webp?v=20260913-live1') 50% 17%/cover no-repeat;box-shadow:0 20px 60px rgba(0,0,0,.42),inset 0 0 50px rgba(29,196,232,.04)}
  .bax-m-hero:before{content:'';position:absolute;inset:0;background:linear-gradient(180deg,rgba(1,7,14,.05) 0%,rgba(1,7,14,0) 55%,rgba(1,7,14,.62) 100%);pointer-events:none}.bax-m-hero:after{content:'BRIAN • GELİŞİM ANATOMİSİ';position:absolute;left:14px;bottom:12px;font-size:10px;font-weight:900;letter-spacing:.12em;color:#bff8ff;text-shadow:0 2px 9px #000}
  .bax-m-section{margin:0 12px 12px;border:1px solid rgba(40,211,248,.22);border-radius:18px;background:linear-gradient(180deg,rgba(4,24,39,.95),rgba(2,13,24,.98));overflow:hidden}
  .bax-m-section h3{margin:0;padding:13px 14px 10px;font-size:11px;letter-spacing:.08em;color:#dffcff;border-bottom:1px solid rgba(60,184,219,.12)}
  .bax-m-organs{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:10px}.bax-m-organ{min-width:0;border:1px solid rgba(62,193,228,.13);border-radius:12px;background:rgba(7,31,46,.72);padding:10px}.bax-m-organ-top{display:flex;align-items:center;gap:6px;font-size:10px;color:#b9d7e2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.bax-m-organ strong{display:block;margin-top:6px;font-size:18px;color:#f3ffff}.bax-m-organ small{display:block;margin-top:4px;font-size:9px;color:#61efb5}.bax-m-organ small.warn{color:#ffd36e}.bax-m-organ small.wait{color:#8ca7b2}
  .bax-m-events{padding:4px 12px 10px}.bax-m-event{display:grid;grid-template-columns:48px minmax(0,1fr);gap:8px;padding:10px 0;border-bottom:1px solid rgba(64,178,211,.12);font-size:10px}.bax-m-event:last-child{border-bottom:0}.bax-m-event b{color:#61c9ef}.bax-m-event span{min-width:0;color:#a9c9d5;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.bax-m-event strong{color:#e9fcff}.bax-m-empty{padding:24px 12px;text-align:center;color:#7897a5;font-size:11px}
}
@media(max-width:380px){.bax-m-metrics{gap:7px;padding-left:9px;padding-right:9px}.bax-m-hero,.bax-m-section{margin-left:9px;margin-right:9px}.bax-metric strong{font-size:21px}.bax-m-organs{grid-template-columns:1fr}}
`;document.head.appendChild(s);
  }

  function make(){
    const shell=$('#balShell');if(!shell||$('.bax-ref-wrap',shell))return false;
    injectCss();
    const wrap=document.createElement('section');wrap.className='bax-ref-wrap';wrap.setAttribute('aria-label','Brian canlı gelişim anatomisi');
    wrap.innerHTML=`
      <button class="bax-hit bax-refresh" title="Canlı veriyi yenile" aria-label="Canlı veriyi yenile"></button>
      <button class="bax-hit bax-close" title="Kapat" aria-label="Kapat"></button>
      <div class="bax-sync">CANLI VERİ</div>
      <div class="bax-val bax-v-general"></div><div class="bax-val bax-v-health"></div><div class="bax-val bax-v-learn"></div><div class="bax-val bax-v-decision"></div><div class="bax-val bax-v-energy"></div>
      <div class="bax-val bax-v-alpha"></div><div class="bax-val bax-v-sensors"></div><div class="bax-val bax-v-world"></div><div class="bax-val bax-v-liver"></div><div class="bax-val bax-v-gut"></div>
      <div class="bax-val bax-v-dna"></div><div class="bax-val bax-v-cosmic"></div>
      <div class="bax-organs"></div><div class="bax-events"></div>
      <div class="bax-mobile">
        <header class="bax-m-head">
          <div class="bax-m-title"><div class="bax-m-kicker">BRIAN v2</div><h2>Gelişim Anatomisi</h2></div>
          <div class="bax-m-actions"><div class="bax-m-live">CANLI</div><button class="bax-m-btn bax-m-refresh" aria-label="Yenile">↻</button><button class="bax-m-btn bax-m-close" aria-label="Kapat">×</button></div>
        </header>
        <div class="bax-m-metrics">
          <div class="bax-metric"><span>Genel Gelişim</span><strong data-bind="general">—</strong></div>
          <div class="bax-metric"><span>Sistem Sağlığı</span><strong data-bind="health">—</strong></div>
          <div class="bax-metric"><span>Öğrenme</span><strong data-bind="learn">—</strong></div>
          <div class="bax-metric"><span>Karar Kalitesi</span><strong data-bind="decision">—</strong></div>
          <div class="bax-metric wide energy"><span>Enerji / Canlılık</span><strong data-bind="energy">—</strong></div>
        </div>
        <div class="bax-m-hero" aria-label="Brian anatomi görünümü"></div>
        <section class="bax-m-section"><h3>⚕ ORGAN DURUM LİSTESİ</h3><div class="bax-m-organs"></div></section>
        <section class="bax-m-section"><h3>⚡ SON GELİŞİM FAALİYETLERİ</h3><div class="bax-m-events"></div></section>
      </div>`;
    shell.appendChild(wrap);
    $('.bax-refresh',wrap).onclick=()=>load(true);
    $('.bax-close',wrap).onclick=()=>$('#balClose')?.click();
    $('.bax-m-refresh',wrap).onclick=()=>load(true);
    $('.bax-m-close',wrap).onclick=()=>$('#balClose')?.click();
    document.addEventListener('keydown',e=>{if(e.key==='Escape'&&$('#brianAnatomyLive.bal-show'))$('#balClose')?.click()},{passive:true});
    return true;
  }

  function set(q,v){const e=$(q);if(e)e.textContent=v==null?'—':fmt(v)}
  function bind(name,v,raw=false){$$(`[data-bind="${name}"]`).forEach(e=>e.textContent=raw?String(v??'—'):(v==null?'—':fmt(v)))}
  function render(){
    if(!make())return;
    if(!state.data)return;
    const a=component('alpha'),s=component('sensors'),w=component('world'),l=component('lab');
    const general=overall('brain_development_pct'),health=overall('data_exchange_health_pct'),conf=overall('evidence_confidence_pct'),learn=derivedLearning(),dec=decision();
    set('.bax-v-general',general);set('.bax-v-health',health);set('.bax-v-learn',learn);set('.bax-v-decision',dec);
    bind('general',general);bind('health',health);bind('learn',learn);bind('decision',dec);
    const en=energy(),ee=$('.bax-v-energy');if(ee)ee.textContent=en.label;bind('energy',en.label,true);
    set('.bax-v-alpha',score(a));set('.bax-v-sensors',score(s));set('.bax-v-world',score(w));set('.bax-v-liver',score(w,'quality_pct'));set('.bax-v-gut',score(l,'quality_pct'));
    set('.bax-v-dna',Math.min(100,((score(l)||0)+(conf||0))/2));set('.bax-v-cosmic',score(w));
    const sync=$('.bax-sync');if(sync){sync.className=`bax-sync ${state.error?'bad':'ok'}`;sync.textContent=state.error?'SON VERİ / BAĞLANTI BEKLENİYOR':`CANLI · ${new Date(state.last).toLocaleTimeString('tr-TR',{hour:'2-digit',minute:'2-digit',second:'2-digit'})}`}
    const mlive=$('.bax-m-live');if(mlive){mlive.classList.toggle('bad',!!state.error);mlive.textContent=state.error?'SON VERİ':'CANLI'}
    const rows=organRows();
    const org=$('.bax-organs');if(org)org.innerHTML=rows.map(([icon,name,n])=>`<div class="bax-organ"><span>${icon} ${esc(name)}</span><b>${n==null?'—':fmt(n)}</b><em>${n==null?'Kanıt bekliyor':n<35?'Uyarı':'Çalışıyor'}</em></div>`).join('');
    const morg=$('.bax-m-organs');if(morg)morg.innerHTML=rows.map(([icon,name,n])=>`<div class="bax-m-organ"><div class="bax-m-organ-top">${icon} ${esc(name)}</div><strong>${n==null?'—':fmt(n)}</strong><small class="${n==null?'wait':n<35?'warn':''}">${n==null?'Kanıt bekliyor':n<35?'Uyarı':'● Çalışıyor'}</small></div>`).join('');
    const items=activity();
    const ev=$('.bax-events');if(ev)ev.innerHTML=items.length?items.map((x,i)=>`<div class="bax-event"><b>${new Date(Date.now()-i*60000).toLocaleTimeString('tr-TR',{hour:'2-digit',minute:'2-digit'})}</b><span><strong>${esc(x[0])}</strong> · ${esc(x[1])}</span></div>`).join(''):'<div class="bax-empty">Yeni gelişim faaliyeti bekleniyor.</div>';
    const mev=$('.bax-m-events');if(mev)mev.innerHTML=items.length?items.map((x,i)=>`<div class="bax-m-event"><b>${new Date(Date.now()-i*60000).toLocaleTimeString('tr-TR',{hour:'2-digit',minute:'2-digit'})}</b><span><strong>${esc(x[0])}</strong> · ${esc(x[1])}</span></div>`).join(''):'<div class="bax-m-empty">Yeni gelişim faaliyeti bekleniyor.</div>';
  }

  async function load(force=false){
    if(state.busy)return;if(!force&&Date.now()-state.last<28000)return;
    if(typeof key==='function'&&!key()){setTimeout(()=>load(false),1500);return}
    state.busy=true;
    try{
      if(typeof post!=='function')throw new Error('Brian API köprüsü hazır değil');
      state.data=await post(ENDPOINT(),{});state.error=null;state.last=Date.now();
      try{localStorage.setItem('brian-anatomy-reference-live',JSON.stringify({t:state.last,d:state.data}))}catch(_e){}
    }catch(e){
      state.error=String(e?.message||e);
      if(!state.data){try{const c=JSON.parse(localStorage.getItem('brian-anatomy-reference-live')||'null');if(c?.d&&Date.now()-Number(c.t||0)<6*60*60*1000){state.data=c.d;state.last=Number(c.t||0)}}catch(_e){}}
    }finally{state.busy=false;render()}
  }

  function boot(){
    let tries=0;const ready=setInterval(()=>{tries++;if(make()||$('#balShell')){clearInterval(ready);make();load(true)}else if(tries>40)clearInterval(ready)},150);
    setInterval(()=>{if(document.visibilityState==='visible'&&$('#brianAnatomyLive.bal-show'))load(false)},30000);
    const mo=new MutationObserver(()=>{if($('#brianAnatomyLive')&&!$('.bax-ref-wrap')){make();render()}});mo.observe(document.documentElement,{subtree:true,childList:true});
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',boot,{once:true});else boot();
})();
