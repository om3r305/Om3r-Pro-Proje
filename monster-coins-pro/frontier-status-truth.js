'use strict';
(()=>{
  const CACHE_KEY='brian-frontier-heartbeat-lkg-v3';
  const CACHE_MAX_MS=30*60*1000;
  const PUBLIC_HEARTBEAT_ENDPOINT='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-frontier-heartbeat-public';
  const now=()=>Date.now();
  const parseAge=(v)=>{const t=Date.parse(String(v||''));return Number.isFinite(t)?Math.max(0,(now()-t)/1000):Infinity};
  const fmtAge=(v)=>{const s=parseAge(v);return !Number.isFinite(s)?'bilinmiyor':s<60?`${Math.round(s)} sn`:s<3600?`${Math.round(s/60)} dk`:`${Math.round(s/3600)} sa`};
  const okStatus=(r)=>['SUCCESS','ONLINE','OK'].includes(String(r?.status||'').toUpperCase());
  const liveCycleStatus=(r)=>['SUCCESS','ONLINE','OK','SKIPPED'].includes(String(r?.status||'').toUpperCase());
  const cached=()=>{try{const x=JSON.parse(localStorage.getItem(CACHE_KEY)||'null');return x?.hb&&now()-Number(x.saved_at||0)<=CACHE_MAX_MS?x:null}catch{return null}};
  const save=(hb)=>{try{localStorage.setItem(CACHE_KEY,JSON.stringify({saved_at:now(),hb}))}catch{}};
  const getHB=()=>{try{return typeof STABILITY!=='undefined'?STABILITY.heartbeat:null}catch{return null}};
  const setHB=(hb)=>{try{if(typeof STABILITY!=='undefined')STABILITY.heartbeat=hb}catch{}};
  const latestRun=(hb,id)=>hb?.collectors?.[id]||null;
  const runSuccessAge=(r)=>parseAge(r?.last_success_at||(okStatus(r)?(r?.finished_at||r?.started_at):null));
  const withEvidence=(state,meta)=>({state,meta});

  function hydrate(){
    if(getHB()) return;
    const c=cached();
    if(c?.hb){setHB({...c.hb,__cached:true,__cached_saved_at:c.saved_at});try{applyHeartbeat(c.hb)}catch{}}
  }

  const originalFrontierPost=typeof frontierPost==='function'?frontierPost:null;
  if(originalFrontierPost){
    frontierPost=async function(url,body={},timeoutMs=20000){
      const isHB=String(url||'').includes('/brian-frontier-heartbeat');
      try{
        const data=await originalFrontierPost(url,body,isHB?Math.max(25000,timeoutMs):timeoutMs);
        if(isHB&&data?.status==='OK'){save(data);setHB(data)}
        return data;
      }catch(e){
        if(isHB){
          const c=cached();
          if(c?.hb){
            const stale={...c.hb,__cached:true,__cached_saved_at:c.saved_at,__live_error:String(e?.message||e)};
            setHB(stale);
            return stale;
          }
        }
        throw e;
      }
    };
  }

  async function fetchJsonWithTimeout(url,init,timeoutMs,label){
    const controller=new AbortController();
    const timer=setTimeout(()=>controller.abort(),timeoutMs);
    try{
      const r=await fetch(url,{...init,cache:'no-store',signal:controller.signal});
      let data={};try{data=await r.json()}catch{}
      if(!r.ok){const e=new Error(data.error||data.status||`HTTP ${r.status}`);e.status=r.status;e.payload=data;throw e}
      return data;
    }catch(e){
      if(e?.name==='AbortError')throw new Error(label||'HEARTBEAT_TIMEOUT');
      throw e;
    }finally{clearTimeout(timer)}
  }

  async function publicHeartbeat(timeoutMs=16000){
    return fetchJsonWithTimeout(PUBLIC_HEARTBEAT_ENDPOINT,{
      method:'POST',headers:{'content-type':'application/json'},body:'{}'
    },timeoutMs,'HEARTBEAT_READONLY_TIMEOUT');
  }

  function hbObservedMs(hb){
    const t=Date.parse(String(hb?.observed_at||''));
    return Number.isFinite(t)?t:0;
  }
  function freshestHeartbeat(candidate){
    const options=[candidate,getHB(),cached()?.hb].filter(Boolean);
    options.sort((a,b)=>hbObservedMs(b)-hbObservedMs(a));
    return options[0]||candidate||null;
  }

  function truthRows(){
    hydrate();
    const hb=getHB();
    if(!hb){
      const why=(()=>{try{return STABILITY?.error?` · ${STABILITY.error}`:''}catch{return''}})();
      const meta=`Canlı heartbeat bekleniyor${why}`;
      return [
        {key:'world',name:'Dünya Gezgini',icon:'🌍',state:'warn',meta},
        {key:'behavior',name:'Davranış Motoru',icon:'🧠',state:'warn',meta},
        {key:'alpha',name:'ALPHA',icon:'α',state:'warn',meta},
        {key:'treasury',name:'Hazine / Portföy',icon:'◉',state:'warn',meta},
        {key:'research',name:'Araştırma Lab',icon:'⚗',state:'warn',meta},
        {key:'ocean',name:'Okyanus',icon:'≈',state:'warn',meta},
      ];
    }

    const c=hb.control||{};
    if(c.system_enabled===false){
      return [
        {key:'world',name:'Dünya Gezgini',icon:'🌍',state:'info',meta:'Sistem operatör tarafından durduruldu'},
        {key:'behavior',name:'Davranış Motoru',icon:'🧠',state:'info',meta:'Sistem operatör tarafından durduruldu'},
        {key:'alpha',name:'ALPHA',icon:'α',state:'info',meta:'Sistem operatör tarafından durduruldu'},
        {key:'treasury',name:'Hazine / Portföy',icon:'◉',state:'info',meta:'Sistem operatör tarafından durduruldu'},
        {key:'research',name:'Araştırma Lab',icon:'⚗',state:'info',meta:'Sistem operatör tarafından durduruldu'},
        {key:'ocean',name:'Okyanus',icon:'≈',state:'info',meta:'Sistem operatör tarafından durduruldu'},
      ];
    }

    const hbAge=parseAge(hb.observed_at);
    const cachedMode=Boolean(hb.__cached||hb.transport_degraded);
    const freshnessSuffix=cachedMode?` · son doğrulama ${Math.round((now()-Number(hb.__cached_saved_at||now()))/60000)} dk önce`:'';

    const alphaAge=parseAge(hb.alpha?.observed_at);
    const alphaRun=latestRun(hb,'brian-alpha-decision-compiler-v2');
    const alphaSuccessAge=runSuccessAge(alphaRun);
    let alpha=alphaAge<=720
      ?withEvidence('ok',`Karar akışı ${fmtAge(hb.alpha?.observed_at)} önce · ${String(hb.alpha?.asset_id||'').replace('crypto:','')||'ALPHA'} ${String(hb.alpha?.action||'')}`)
      :alphaAge<=1080||alphaSuccessAge<=600
        ?withEvidence('warn',`ALPHA senkronu canlı · son karar ${fmtAge(hb.alpha?.observed_at)} önce`)
        :withEvidence(alphaSuccessAge<=900?'warn':'bad',alphaRun?.error_class||`ALPHA kanıtı ${fmtAge(hb.alpha?.observed_at)} önce`);

    const wr=hb.world_run||{};
    const wrAge=parseAge(wr.finished_at||wr.started_at);
    const worldRun=latestRun(hb,'brian-world-brain-v1');
    const worldSuccessAge=runSuccessAge(worldRun);
    let world=(String(wr.status)==='SUCCESS'&&wrAge<=900)||worldSuccessAge<=900
      ?withEvidence('ok',`Dünya çekirdeği canlı · ${fmtAge(wr.finished_at||wr.started_at)} önce`)
      :wrAge<=1800||worldSuccessAge<=1800
        ?withEvidence('warn',`World Brain son sağlam kanıt ${fmtAge(worldRun?.last_success_at||wr.finished_at||wr.started_at)} önce`)
        :withEvidence('bad',worldRun?.error_class||'World Brain taze kanıt bekliyor');

    const t=c.treasury||{};
    const tAge=parseAge(t.observed_at);
    const tr=latestRun(hb,'brian-evolution-treasury-v1');
    const trSuccessAge=runSuccessAge(tr);
    const treasuryCycleLive=(liveCycleStatus(tr)&&parseAge(tr?.finished_at||tr?.started_at)<=900)||trSuccessAge<=900;
    let treasury=treasuryCycleLive&&t.equity_usd!=null&&tAge<=1800
      ?withEvidence('ok',`Kasa hattı canlı · ${Number(t.equity_usd).toLocaleString('tr-TR')} · son snapshot ${fmtAge(t.observed_at)} önce`)
      :t.equity_usd!=null&&tAge<=1800
        ?withEvidence('warn',`Kasa son kanıt ${fmtAge(t.observed_at)} önce`)
        :withEvidence(trSuccessAge<=1800?'warn':'bad',tr?.error_class||'Hazine taze kanıt bekliyor');

    const evo=latestRun(hb,'brian-evolution-orchestrator-v1')||latestRun(hb,'brian-evolution-sandbox-v1')||latestRun(hb,'brian-evolution-researcher-v1');
    const evoSuccessAge=runSuccessAge(evo);
    let research=evoSuccessAge<=2400?withEvidence('ok',`Evolution / Lab canlı · son başarı ${fmtAge(evo?.last_success_at||evo?.finished_at||evo?.started_at)} önce`):evoSuccessAge<=3600?withEvidence('warn',`Araştırma hattı son başarı ${fmtAge(evo?.last_success_at||evo?.finished_at||evo?.started_at)} önce`):withEvidence('warn',evo?.error_class||'Araştırma hattı yeni cycle bekliyor');

    const oceanRun=latestRun(hb,'brian-evolution-ocean-worker-v1');
    const oceanSuccessAge=runSuccessAge(oceanRun);
    let ocean=oceanSuccessAge<=2400?withEvidence('ok',`Ocean worker canlı · ${fmtAge(oceanRun?.last_success_at||oceanRun?.finished_at||oceanRun?.started_at)} önce`):withEvidence('warn',oceanRun?.error_class||'Ocean yeni cycle bekliyor');

    const b=hb.behavior||{};
    const behaviorContextAt=b.observed_at||b.decision_observed_at;
    const behaviorContextAge=parseAge(behaviorContextAt);
    const behaviorPipelineAt=b.pipeline_observed_at||null;
    const behaviorPipelineAge=parseAge(behaviorPipelineAt);
    const behaviorPipelineLive=String(b.pipeline_status||'').toUpperCase()==='SUCCESS'&&behaviorPipelineAge<=720;
    let behavior=behaviorPipelineLive
      ?withEvidence('ok',`Davranış üreticisi canlı · pipeline ${fmtAge(behaviorPipelineAt)} önce · ALPHA bağlamı ${fmtAge(behaviorContextAt)} önce`)
      :behaviorContextAge<=720
        ?withEvidence('ok',`Davranış ${String(b.state||'CANLI').replaceAll('_',' ')} · ${fmtAge(behaviorContextAt)} önce`)
        :behaviorContextAge<=1080
          ?withEvidence('warn',`Davranış pipeline heartbeat bekliyor · son bağlı bağlam ${fmtAge(behaviorContextAt)} önce`)
          :alpha.state==='bad'
            ?withEvidence('bad','Davranış pipeline heartbeat ve ALPHA bağlantısı taze değil')
            :withEvidence('warn','Davranış hattı yeni pipeline kanıtı bekliyor');

    const pipeline=hb.news_pipeline;
    if(!pipeline || !['OK','SUCCESS','HEALTHY'].includes(pipeline.status) || pipeline.pipeline_stalled){
      if(world.state==='ok')world.state='warn';
      world.meta+=pipeline?.pipeline_stalled?' · haber işleme kuyruğu gecikiyor':pipeline?' · haber kaynakları kısmi; Direct Wire '+(pipeline.direct_wire?.status==='SUCCESS'&&parseAge(pipeline.direct_wire?.last_run_at)<=360?'canlı':'bekleniyor'):' · haber kapsamı doğrulanamadı';
    }
    if(cachedMode||hbAge>120){
      [world,behavior,alpha,treasury,research,ocean].forEach(x=>{if(x.state==='ok')x.state='warn';x.meta+=freshnessSuffix||` · heartbeat ${Math.round(hbAge)} sn önce`});
    }

    return [
      {key:'world',name:'Dünya Gezgini',icon:'🌍',...world},
      {key:'behavior',name:'Davranış Motoru',icon:'🧠',...behavior},
      {key:'alpha',name:'ALPHA',icon:'α',...alpha},
      {key:'treasury',name:'Hazine / Portföy',icon:'◉',...treasury},
      {key:'research',name:'Araştırma Lab',icon:'⚗',...research},
      {key:'ocean',name:'Okyanus',icon:'≈',...ocean},
    ];
  }

  try{moduleRows=truthRows}catch{}

  const baseNews=renderNews;
  renderNews=function(){
    baseNews();
    const hb=getHB(), pipeline=hb?.news_pipeline;
    const stale=!hb||hb.__cached||hb.transport_degraded||parseAge(hb.observed_at)>120;
    const partial=stale||!pipeline||!['OK','SUCCESS','HEALTHY'].includes(pipeline.status)||pipeline.pipeline_stalled;
    const badge=document.getElementById('newsBadge');
    if(partial&&badge){badge.textContent=stale?'BAĞLANTI BEKLENİYOR':'KAPSAM KISMİ';badge.className='badge warn';}
    if(!news().length){
      const feed=document.getElementById('criticalNews');
      if(feed)feed.innerHTML='<div class="news"><div class="news-title">'+(partial?'Haber kapsamı tamamlanmadı.':'Doğrulanan akışta yeni kritik gelişme yok.')+'</div><div class="news-meta">'+(partial?'Kaynak veya bağlantı eksikliği sürüyor; boş liste önemli haber olmadığı anlamına gelmez.':'Son başarılı tarama izleniyor.')+'</div></div>';
      const ticker=document.querySelector('.ticker');
      if(ticker&&partial)ticker.innerHTML='<span class="radar-label">BRIAN RADAR</span> · Haber kapsamı kısmi · kaynaklar doğrulanıyor';
    }
  };

  async function directHeartbeat(){
    try{
      const incoming=await publicHeartbeat(25000);
      if(incoming?.status==='OK'){
        const hb=freshestHeartbeat(incoming);
        const incomingIsBest=hb===incoming;
        if(incomingIsBest&&!incoming.transport_degraded)save(incoming);
        setHB(hb);
        try{if(typeof STABILITY!=='undefined')STABILITY.error=null}catch{}
        try{if(typeof S!=='undefined'&&S.errors)delete S.errors.systemControl}catch{}
        try{applyHeartbeat(hb)}catch{}
        try{if(typeof render==='function')render()}catch{}
        const el=document.getElementById('syncText');
        if(el)el.textContent=incomingIsBest&&!incoming.transport_degraded
          ?'Brian heartbeat doğrulandı'
          :'Canlı bağlantı yenileniyor · son sağlam kanıt korunuyor';
      }
    }catch(e){
      const c=cached();
      const hb=freshestHeartbeat(c?.hb||null);
      if(hb){
        const stale={...hb,__cached:true,__cached_saved_at:c?.saved_at||now(),__live_error:String(e?.message||e)};
        setHB(stale);
        try{applyHeartbeat(hb)}catch{}
      }
      const el=document.getElementById('syncText');
      if(el)el.textContent=hb?'Canlı bağlantı yenileniyor · son sağlam kanıt gösteriliyor':'Heartbeat gecikti · canlı kanıt bekleniyor';
      try{if(typeof render==='function')render()}catch{}
    }
  }

  hydrate();
  try{if(typeof render==='function')render()}catch{}
  // Single-writer rule: frontier-stability owns heartbeat polling.
  // This layer only supplies truth semantics and last-known-good hydration.
  if(!getHB()&&!window.__FRONTIER_COMPOSED_BOOT__) setTimeout(directHeartbeat,500);
})();
