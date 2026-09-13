'use strict';
(()=>{
  const ENDPOINT=()=>`${ROOT}/brian-development-status`;
  const $=(q,r=document)=>r.querySelector(q);
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
    const a=component('alpha'),s=component('sensors'),w=component('world'),l=component('lab'),t=component('treasury');
    const health=overall('data_exchange_health_pct')||0,conf=overall('evidence_confidence_pct')||0;
    const rows=[
      ['🧠','Beyin - ALPHA',score(a)],['👁','Göz / Sinir',score(s)],['🫁','Sol Akciğer',score(w)],
      ['🫁','Sağ Akciğer',score(w,'quality_pct')],['🫀','Kalp / Enerji',health],['◒','Karaciğer',score(w,'quality_pct')],
      ['◉','Mide / Bağırsak',score(l,'quality_pct')],['🦴','İskelet / Yapı',conf],['✹','Bağışıklık',Math.min(100,(health+conf)/2)],
      ['🧬','DNA / Çekirdek',Math.min(100,((score(l)||0)+conf)/2)]
    ];
    return rows;
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
#brianAnatomyLive{background:#010611!important;overflow:auto!important}
#brianAnatomyLive .bal-head,#brianAnatomyLive #balBody{display:none!important}
#brianAnatomyLive .bal-shell{min-height:100vh!important;padding:0!important;background:#010611!important;display:flex!important;justify-content:center!important;align-items:flex-start!important}
.bax-ref-wrap{position:relative;width:min(100%,1448px);aspect-ratio:1448/1086;flex:0 0 auto;background:#010611 url('/brian-anatomy-dashboard-reference.webp?v=20260913-live1') center top/100% 100% no-repeat;overflow:hidden;box-shadow:0 0 80px #000}
.bax-hit{position:absolute;z-index:10;border:0;background:transparent;cursor:pointer}
.bax-refresh{left:80.8%;top:1.0%;width:10.1%;height:5.4%}.bax-close{right:.3%;top:.4%;width:2.6%;height:4.2%;opacity:0;transition:.2s}.bax-close:hover{opacity:1;background:#ff476b;border-radius:50%;color:#fff}.bax-close:after{content:'×';font-size:18px;font-weight:900}
.bax-val{position:absolute;z-index:8;display:flex;align-items:center;justify-content:center;min-width:4.8%;height:2.45%;padding:0 .35%;border-radius:5px;background:linear-gradient(180deg,rgba(4,23,40,.98),rgba(2,13,26,.98));box-shadow:0 0 11px rgba(37,225,255,.10);font-weight:900;color:#efffff;letter-spacing:.015em;font-size:clamp(8px,1.14vw,16px);line-height:1;white-space:nowrap}
.bax-val.green{color:#78f7bd}.bax-val.cyan{color:#54eaff}.bax-val.pink{color:#ff6394}.bax-val.amber{color:#ffd06b}
.bax-v-general{left:39.25%;top:10.0%}.bax-v-health{left:49.15%;top:10.0%}.bax-v-learn{left:58.45%;top:10.0%}.bax-v-decision{left:67.25%;top:10.0%}
.bax-v-energy{left:82.0%;top:10.55%;min-width:7.8%;height:2.55%;font-size:clamp(9px,1.23vw,18px);color:#63f4b0}
.bax-v-alpha{left:10.35%;top:19.25%;color:#64f4af}.bax-v-sensors{left:10.35%;top:31.15%;color:#55eaff}.bax-v-world{left:10.35%;top:43.0%;color:#6cf4a6}.bax-v-liver{left:10.35%;top:55.25%;color:#57edd1}.bax-v-gut{left:10.35%;top:67.4%;color:#d9ef7b}
.bax-v-dna{left:87.1%;top:19.35%;color:#61eaff}.bax-v-cosmic{left:84.1%;top:69.05%;color:#57e9d0}
.bax-sync{position:absolute;right:8.4%;top:5.0%;z-index:9;color:#79a8ba;font-size:clamp(5px,.48vw,8px);font-weight:700;white-space:nowrap;text-shadow:0 1px 4px #000}.bax-sync.ok:before{content:'● ';color:#52efad}.bax-sync.bad:before{content:'● ';color:#ff6b7d}
.bax-organs{position:absolute;z-index:7;left:28.25%;top:78.58%;width:39.4%;height:14.2%;display:grid;grid-template-columns:1fr 1fr;grid-auto-rows:1fr;column-gap:.7%;row-gap:.6%;padding:2.45% .8% .65%;box-sizing:border-box;background:linear-gradient(180deg,rgba(2,17,31,.98),rgba(2,12,24,.99));border:1px solid rgba(39,201,239,.55);border-radius:9px;box-shadow:inset 0 0 25px rgba(23,190,230,.04)}
.bax-organs:before{content:'⚕  ORGAN DURUM LİSTESİ';position:absolute;left:2.2%;top:4%;font-size:clamp(6px,.69vw,10px);font-weight:900;color:#e6fbff;letter-spacing:.035em}.bax-organ{display:grid;grid-template-columns:minmax(0,1fr) 22% 29%;align-items:center;gap:1.5%;padding:0 3%;border:1px solid rgba(55,183,220,.15);border-radius:4px;background:rgba(4,27,43,.72);font-size:clamp(5px,.59vw,8.5px);color:#b9d9e5;white-space:nowrap}.bax-organ b{color:#f5ffff;text-align:right;font-size:1.05em}.bax-organ em{font-style:normal;color:#53efbc;text-align:right}.bax-organ em:before{content:'● ';text-shadow:0 0 6px #52efb5}
.bax-events{position:absolute;z-index:7;left:68.1%;top:78.58%;width:28.45%;height:14.2%;padding:2.45% .75% .55%;box-sizing:border-box;background:linear-gradient(180deg,rgba(2,17,31,.98),rgba(2,12,24,.99));border:1px solid rgba(39,201,239,.55);border-radius:9px;box-shadow:inset 0 0 25px rgba(23,190,230,.04);overflow:hidden}.bax-events:before{content:'⚡  SON GELİŞİM FAALİYETLERİ';position:absolute;left:3%;top:4%;font-size:clamp(6px,.69vw,10px);font-weight:900;color:#e6fbff;letter-spacing:.035em}.bax-event{height:16%;display:grid;grid-template-columns:14% 1fr;align-items:center;border-bottom:1px solid rgba(61,172,208,.13);font-size:clamp(5px,.56vw,8px);color:#9bc1d0;overflow:hidden}.bax-event b{color:#69bde5;font-size:.95em}.bax-event span{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.bax-empty{display:grid;place-items:center;height:100%;font-size:clamp(6px,.6vw,9px);color:#7292a2}
.bax-mobile-note{display:none}
@media(max-width:800px){#brianAnatomyLive{overflow:auto!important}.bax-ref-wrap{width:1180px;min-width:1180px}.bax-mobile-note{display:block;position:fixed;left:8px;bottom:8px;z-index:70;background:#031626e8;border:1px solid #37dfff55;border-radius:999px;padding:7px 10px;color:#a9f7ff;font:700 9px system-ui;pointer-events:none}}
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
      <div class="bax-mobile-note">← Kaydırarak tam ekranı inceleyebilirsin →</div>`;
    shell.appendChild(wrap);
    $('.bax-refresh',wrap).onclick=()=>load(true);
    $('.bax-close',wrap).onclick=()=>$('#balClose')?.click();
    document.addEventListener('keydown',e=>{if(e.key==='Escape'&&$('#brianAnatomyLive.bal-show'))$('#balClose')?.click()},{passive:true});
    return true;
  }

  function set(q,v){const e=$(q);if(e)e.textContent=v==null?'—':fmt(v)}
  function render(){
    if(!make())return;
    if(!state.data)return;
    const a=component('alpha'),s=component('sensors'),w=component('world'),l=component('lab');
    const health=overall('data_exchange_health_pct'),conf=overall('evidence_confidence_pct');
    set('.bax-v-general',overall('brain_development_pct'));
    set('.bax-v-health',health);set('.bax-v-learn',derivedLearning());set('.bax-v-decision',decision());
    const en=energy(),ee=$('.bax-v-energy');if(ee)ee.textContent=en.label;
    set('.bax-v-alpha',score(a));set('.bax-v-sensors',score(s));set('.bax-v-world',score(w));set('.bax-v-liver',score(w,'quality_pct'));set('.bax-v-gut',score(l,'quality_pct'));
    set('.bax-v-dna',Math.min(100,((score(l)||0)+(conf||0))/2));set('.bax-v-cosmic',score(w));
    const sync=$('.bax-sync');if(sync){sync.className=`bax-sync ${state.error?'bad':'ok'}`;sync.textContent=state.error?'SON VERİ / BAĞLANTI BEKLENİYOR':`CANLI · ${new Date(state.last).toLocaleTimeString('tr-TR',{hour:'2-digit',minute:'2-digit',second:'2-digit'})}`}
    const org=$('.bax-organs');if(org)org.innerHTML=organRows().map(([icon,name,n])=>`<div class="bax-organ"><span>${icon} ${esc(name)}</span><b>${n==null?'—':fmt(n)}</b><em>${n==null?'Kanıt bekliyor':n<35?'Uyarı':'Çalışıyor'}</em></div>`).join('');
    const ev=$('.bax-events'),items=activity();if(ev)ev.innerHTML=items.length?items.map((x,i)=>`<div class="bax-event"><b>${new Date(Date.now()-i*60000).toLocaleTimeString('tr-TR',{hour:'2-digit',minute:'2-digit'})}</b><span><strong>${esc(x[0])}</strong> · ${esc(x[1])}</span></div>`).join(''):'<div class="bax-empty">Yeni gelişim faaliyeti bekleniyor.</div>';
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
