'use strict';
(()=>{
  const $=(q,r=document)=>r.querySelector(q);
  const $$=(q,r=document)=>Array.from(r.querySelectorAll(q));
  const clamp=n=>Math.max(0,Math.min(100,Number(n)||0));
  const fmt=n=>Number.isFinite(Number(n))?`${Number(n).toFixed(1)}%`:'—';
  const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let data=null,last=0,busy=false;

  function cached(){
    try{
      const a=JSON.parse(localStorage.getItem('brian-anatomy-reference-live')||'null');
      if(a?.d){last=Number(a.t||Date.now());return a.d}
    }catch(_e){}
    try{
      const b=JSON.parse(localStorage.getItem('brian-development-status-v1')||'null');
      if(b?.text){const d=JSON.parse(b.text);last=Number(b.savedAt||Date.now());return d}
    }catch(_e){}
    return null;
  }
  function component(id){return data?.components?.find?.(x=>x.id===id)||null}
  function score(c,field='maturity_pct'){
    if(!c||Number(c.evidence_pct||0)<10)return null;
    const n=Number(c[field]);return Number.isFinite(n)?clamp(n):null;
  }
  function overall(k){const n=Number(data?.overall?.[k]);return Number.isFinite(n)?clamp(n):null}
  function learning(){const l=component('lab'),q=overall('measured_quality_pct')||0,m=score(l)||0;return clamp(m*.6+q*.4)}
  function decision(){const a=component('alpha');return score(a,'quality_pct') ?? overall('measured_quality_pct')}
  function energy(){const h=overall('data_exchange_health_pct')||0,c=overall('evidence_confidence_pct')||0,p=clamp(h*.75+c*.25);return p>=85?'YÜKSEK':p>=60?'ORTA':p>=35?'DÜŞÜK':'KRİTİK'}
  function rows(){
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
    (data?.feature_signals||[]).slice(0,3).forEach(x=>out.push(['DENEY',x.experiment_id||x.decision||'Yeni deney sonucu']));
    (data?.capability_gaps||[]).slice(0,2).forEach(x=>out.push([String(x.severity||'GAP').toUpperCase(),`${x.domain||''} · ${x.capability_id||''}`]));
    (data?.strengths||[]).slice(0,2).forEach(x=>out.push(['GÜÇLÜ',`${x.name||'Güçlü alan'} · ${fmt(x.maturity_pct)}`]));
    return out.slice(0,6);
  }

  function ensureArtwork(){
    const hero=$('.bax-m-hero');if(!hero)return;
    let img=$('.bax-m-art',hero);
    if(!img){
      img=document.createElement('img');img.className='bax-m-art';img.alt='Brian gelişim anatomisi';
      img.decoding='async';img.loading='eager';
      img.src='/brian-anatomy-dashboard-reference.webp?v=20260913-mobile4';
      img.onerror=()=>{
        if(img.dataset.fallback)return;
        img.dataset.fallback='1';
        img.src='https://raw.githubusercontent.com/om3r305/Om3r-Pro-Proje/brian-2026/monster-coins-pro/brian-anatomy-dashboard-reference.webp';
      };
      hero.prepend(img);
    }
  }
  function css(){
    if($('#baxRuntimeHotfixCss'))return;
    const s=document.createElement('style');s.id='baxRuntimeHotfixCss';s.textContent=`
@media(max-width:800px){
  .bax-m-hero{background:#020b15!important;height:auto!important;min-height:0!important;max-height:none!important;aspect-ratio:390/500!important;display:block!important}
  .bax-m-art{position:absolute;inset:0;width:100%;height:100%;display:block;object-fit:cover;object-position:50% 47%;z-index:0;filter:saturate(1.06) contrast(1.04);opacity:1!important}
  .bax-m-hero:before{z-index:1}.bax-m-hero:after{z-index:2}
  .bax-metrics{padding-top:10px!important}
}
`;document.head.appendChild(s);
  }
  function bind(name,val,raw=false){$$(`[data-bind="${name}"]`).forEach(e=>e.textContent=raw?String(val??'—'):(val==null?'—':fmt(val)))}
  function set(q,val){const e=$(q);if(e)e.textContent=val==null?'—':fmt(val)}
  function render(){
    if(!data)return;
    css();ensureArtwork();
    const general=overall('brain_development_pct'),health=overall('data_exchange_health_pct'),conf=overall('evidence_confidence_pct'),learn=learning(),dec=decision(),en=energy();
    bind('general',general);bind('health',health);bind('learn',learn);bind('decision',dec);bind('energy',en,true);
    set('.bax-v-general',general);set('.bax-v-health',health);set('.bax-v-learn',learn);set('.bax-v-decision',dec);
    const ee=$('.bax-v-energy');if(ee)ee.textContent=en;
    const a=component('alpha'),s=component('sensors'),w=component('world'),l=component('lab');
    set('.bax-v-alpha',score(a));set('.bax-v-sensors',score(s));set('.bax-v-world',score(w));set('.bax-v-liver',score(w,'quality_pct'));set('.bax-v-gut',score(l,'quality_pct'));
    set('.bax-v-dna',Math.min(100,((score(l)||0)+(conf||0))/2));set('.bax-v-cosmic',score(w));
    const live=$('.bax-m-live');if(live){live.classList.remove('bad');live.textContent='CANLI'}
    const sync=$('.bax-sync');if(sync){sync.className='bax-sync ok';sync.textContent=`CANLI · ${new Date(last||Date.now()).toLocaleTimeString('tr-TR',{hour:'2-digit',minute:'2-digit',second:'2-digit'})}`}
    const rs=rows();
    const mo=$('.bax-m-organs');if(mo)mo.innerHTML=rs.map(([icon,name,n])=>`<div class="bax-m-organ"><div class="bax-m-organ-top">${icon} ${esc(name)}</div><strong>${n==null?'—':fmt(n)}</strong><small class="${n==null?'wait':n<35?'warn':''}">${n==null?'Kanıt bekliyor':n<35?'Uyarı':'● Çalışıyor'}</small></div>`).join('');
    const o=$('.bax-organs');if(o)o.innerHTML=rs.map(([icon,name,n])=>`<div class="bax-organ"><span>${icon} ${esc(name)}</span><b>${n==null?'—':fmt(n)}</b><em>${n==null?'Kanıt bekliyor':n<35?'Uyarı':'Çalışıyor'}</em></div>`).join('');
    const items=activity();
    const me=$('.bax-m-events');if(me)me.innerHTML=items.length?items.map((x,i)=>`<div class="bax-m-event"><b>${new Date((last||Date.now())-i*60000).toLocaleTimeString('tr-TR',{hour:'2-digit',minute:'2-digit'})}</b><span><strong>${esc(x[0])}</strong> · ${esc(x[1])}</span></div>`).join(''):'<div class="bax-m-empty">Yeni gelişim faaliyeti bekleniyor.</div>';
    const e=$('.bax-events');if(e)e.innerHTML=items.length?items.map((x,i)=>`<div class="bax-event"><b>${new Date((last||Date.now())-i*60000).toLocaleTimeString('tr-TR',{hour:'2-digit',minute:'2-digit'})}</b><span><strong>${esc(x[0])}</strong> · ${esc(x[1])}</span></div>`).join(''):'<div class="bax-empty">Yeni gelişim faaliyeti bekleniyor.</div>';
  }
  async function refresh(){
    if(busy)return;busy=true;
    try{
      if(typeof post==='function' && typeof ROOT!=='undefined'){
        const d=await post(`${ROOT}/brian-development-status`,{});
        if(d){data=d;last=Date.now();try{localStorage.setItem('brian-anatomy-reference-live',JSON.stringify({t:last,d:data}))}catch(_e){}}
      }
    }catch(_e){}
    if(!data)data=cached();
    busy=false;render();
  }
  function boot(){
    css();
    let n=0;const t=setInterval(()=>{
      n++;
      ensureArtwork();
      const c=cached();if(c){data=c;render();clearInterval(t)}
      if(n>20){clearInterval(t);refresh()}
    },250);
    setTimeout(refresh,800);
    setInterval(()=>{if(document.visibilityState==='visible'&&$('#brianAnatomyLive.bal-show')){const c=cached();if(c){data=c;render()}refresh()}},30000);
    document.addEventListener('click',e=>{if(e.target?.closest?.('.bax-m-refresh,.bax-refresh'))setTimeout(refresh,250)},true);
    const mo=new MutationObserver(()=>{if($('.bax-ref-wrap')){ensureArtwork();if(!data)data=cached();render()}});mo.observe(document.documentElement,{subtree:true,childList:true});
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',boot,{once:true});else boot();
})();