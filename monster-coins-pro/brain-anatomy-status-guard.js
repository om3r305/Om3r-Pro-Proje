'use strict';
(()=>{
  const ROOT='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1';
  const KEY_STORAGE='mcp-dashboard-key-v1';
  const DEV_CACHE_KEY='brian-development-status-v1';
  const OWN_CACHE_KEY='brian-anatomy-reference-live';
  const $=(q,r=document)=>r.querySelector(q);
  const $$=(q,r=document)=>Array.from(r.querySelectorAll(q));
  const clamp=n=>Math.max(0,Math.min(100,Number(n)||0));
  const fmt=n=>Number.isFinite(Number(n))?`${Number(n).toFixed(1)}%`:'—';
  let busy=false;
  let lastLiveAt=0;
  let lastError='';

  function key(){return(localStorage.getItem(KEY_STORAGE)||'').trim()}
  function component(data,id){return data?.components?.find?.(x=>x.id===id)||null}
  function score(c,field='maturity_pct'){
    if(!c||Number(c.evidence_pct||0)<10)return null;
    const n=Number(c[field]);return Number.isFinite(n)?clamp(n):null;
  }
  function overall(data,k){const n=Number(data?.overall?.[k]);return Number.isFinite(n)?clamp(n):null}
  function energy(data){
    const h=overall(data,'data_exchange_health_pct')||0,c=overall(data,'evidence_confidence_pct')||0,p=clamp(h*.75+c*.25);
    return p>=85?'YÜKSEK':p>=60?'ORTA':p>=35?'DÜŞÜK':'KRİTİK';
  }
  function learning(data){const l=component(data,'lab'),q=overall(data,'measured_quality_pct')||0,m=score(l)||0;return clamp(m*.6+q*.4)}
  function decision(data){const a=component(data,'alpha');return score(a,'quality_pct') ?? overall(data,'measured_quality_pct')}
  function bind(name,val,raw=false){$$(`[data-bind="${name}"]`).forEach(e=>e.textContent=raw?String(val??'—'):(val==null?'—':fmt(val)))}
  function set(q,val){const e=$(q);if(e)e.textContent=val==null?'—':fmt(val)}
  function applyData(data){
    if(!data)return;
    const general=overall(data,'brain_development_pct'),health=overall(data,'data_exchange_health_pct'),learn=learning(data),dec=decision(data),en=energy(data);
    bind('general',general);bind('health',health);bind('learn',learn);bind('decision',dec);bind('energy',en,true);
    set('.bax-v-general',general);set('.bax-v-health',health);set('.bax-v-learn',learn);set('.bax-v-decision',dec);
    const ee=$('.bax-v-energy');if(ee)ee.textContent=en;
    const a=component(data,'alpha'),s=component(data,'sensors'),w=component(data,'world'),l=component(data,'lab'),conf=overall(data,'evidence_confidence_pct');
    set('.bax-v-alpha',score(a));set('.bax-v-sensors',score(s));set('.bax-v-world',score(w));set('.bax-v-liver',score(w,'quality_pct'));set('.bax-v-gut',score(l,'quality_pct'));
    set('.bax-v-dna',Math.min(100,((score(l)||0)+(conf||0))/2));set('.bax-v-cosmic',score(w));
  }
  function cached(){
    try{
      const raw=localStorage.getItem(DEV_CACHE_KEY);
      if(raw){const c=JSON.parse(raw);if(c?.text)return{data:JSON.parse(c.text),savedAt:Number(c.savedAt||0)}}
    }catch(_e){}
    try{
      const raw=localStorage.getItem(OWN_CACHE_KEY);
      if(raw){const c=JSON.parse(raw);if(c?.d)return{data:c.d,savedAt:Number(c.t||0)}}
    }catch(_e){}
    return null;
  }
  function time(ts){
    if(!ts)return'—';
    return new Date(ts).toLocaleTimeString('tr-TR',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  }
  function paintStatus(state,label,detail=''){
    const live=$('.bax-m-live');
    if(live){live.classList.remove('bad','warn','ok');live.classList.add(state==='ok'?'ok':state==='warn'?'warn':'bad');live.textContent=label}
    const sync=$('.bax-sync');
    if(sync){sync.className=`bax-sync ${state==='ok'?'ok':state==='warn'?'warn':'bad'}`;sync.textContent=detail||label}
  }
  async function refresh(){
    if(busy)return;busy=true;
    const k=key();
    if(!k){
      const c=cached();if(c){applyData(c.data);paintStatus('warn','KİLİTLİ',`SON VERİ · ${time(c.savedAt)}`)}else paintStatus('bad','KİLİTLİ','Dashboard anahtarı gerekli');
      busy=false;return;
    }
    const ctl=new AbortController();const timer=setTimeout(()=>ctl.abort('BRIAN_DEVELOPMENT_TIMEOUT'),30000);
    try{
      const r=await fetch(`${ROOT}/brian-development-status`,{method:'POST',headers:{'content-type':'application/json','x-brian-dashboard-key':k},body:'{}',cache:'no-store',signal:ctl.signal});
      if(!r.ok){let msg=`HTTP ${r.status}`;try{const j=await r.json();msg=j?.error||msg}catch(_e){}throw new Error(msg)}
      const data=await r.json();
      lastLiveAt=Date.now();lastError='';
      try{localStorage.setItem(DEV_CACHE_KEY,JSON.stringify({savedAt:lastLiveAt,text:JSON.stringify(data)}));localStorage.setItem(OWN_CACHE_KEY,JSON.stringify({t:lastLiveAt,d:data}))}catch(_e){}
      applyData(data);
      paintStatus('ok','CANLI',`CANLI · ${time(lastLiveAt)}`);
    }catch(e){
      lastError=String(e?.message||e||'Bağlantı hatası');
      const c=cached();
      if(c){applyData(c.data);paintStatus('warn','SON VERİ',`BAĞLANTI SORUNU · ${time(c.savedAt)}`)}
      else paintStatus('bad','BAĞLANTI HATASI',lastError.slice(0,80));
    }finally{clearTimeout(timer);busy=false}
  }
  function boot(){
    refresh();
    setInterval(()=>{if(document.visibilityState==='visible')refresh()},30000);
    document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible')refresh()});
    document.addEventListener('click',e=>{if(e.target?.closest?.('.bax-m-refresh,.bax-refresh'))setTimeout(refresh,100)},true);
    setInterval(()=>{
      if(lastLiveAt&&Date.now()-lastLiveAt>90000){
        const c=cached();paintStatus(c?'warn':'bad',c?'SON VERİ':'BAĞLANTI HATASI',c?`CANLI AKIŞ GECİKTİ · ${time(c.savedAt)}`:(lastError||'Canlı veri bekleniyor'));
      }
    },15000);
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',boot,{once:true});else boot();
})();
