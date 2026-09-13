'use strict';
(()=>{
  const DEV='/brian-development-status';
  const CACHE='brian-development-status-v1';
  const wrapped=window.fetch.bind(window);
  const readCache=new Map();
  const readInflight=new Map();
  let memoryText='';
  let memoryAt=0;

  const TTL=[
    ['/brian-world-status',45000],
    ['/brian-evolution-status',60000],
    ['/brian-evolution-treasury-status',20000],
    ['/brian-evolution-ocean-status',45000],
    ['/brian-evolution-lab-status',60000],
    ['/brian-evolution-alpha-intelligence-status',30000],
    ['/brian-frontier-news',45000],
  ];

  const clamp=n=>Math.max(0,Math.min(100,Number(n)||0));
  function visible(){return !!document.querySelector('#brianAnatomyLive.bal-show')}
  function response(txt,tag='memory',status=200){return new Response(txt,{status,headers:{'content-type':'application/json; charset=utf-8','x-brian-cache':tag}})}
  function ageSec(v){const t=Date.parse(String(v||''));return Number.isFinite(t)?Math.max(0,(Date.now()-t)/1000):999999}
  function freshness(v,good=120,bad=900){const a=ageSec(v);if(a<=good)return 100;if(a>=bad)return 0;return clamp(100*(bad-a)/(bad-good))}
  function runScore(run,good=180,bad=1200){if(!run)return 0;const st=String(run.status||'').toUpperCase();if(st==='FAILED')return 0;let base=st==='SUCCESS'?100:st==='DEGRADED'||st.startsWith('SKIPPED')?65:45;return clamp(base*freshness(run.finished_at||run.started_at,good,bad)/100)}
  function tier(v){return v>=85?'OLGUN':v>=70?'UZMAN':v>=55?'YETKİN':v>=35?'GELİŞEN':v>=15?'ÇIRAK':'ÇEKİRDEK'}

  function currentHeartbeat(){
    try{return typeof STABILITY!=='undefined'&&STABILITY?.heartbeat?.status==='OK'?STABILITY.heartbeat:null}catch(_e){return null}
  }
  function syntheticDevelopment(){
    const hb=currentHeartbeat();
    if(!hb)return null;
    const collectors=Object.entries(hb.collectors||{}).filter(([id])=>!String(id).toLowerCase().includes('dip'));
    const runs=collectors.map(([,r])=>r).filter(Boolean);
    const collectorScore=runs.length?runs.reduce((s,r)=>s+runScore(r),0)/runs.length:0;
    const alphaFresh=freshness(hb.alpha?.observed_at,120,900);
    const treasury=hb.control?.treasury||{};
    const treasuryFresh=freshness(treasury.observed_at,120,900);
    const worldRun=hb.collectors?.['brian-world-brain-v1']||null;
    const worldScore=Math.max(runScore(worldRun,300,1500),freshness(hb.world_run?.finished_at||hb.world_run?.started_at,300,1500));
    const evolutionRuns=['brian-evolution-orchestrator-v1','brian-evolution-sandbox-v1','brian-evolution-researcher-v1'].map(id=>hb.collectors?.[id]).filter(Boolean);
    const labScore=evolutionRuns.length?evolutionRuns.reduce((s,r)=>s+runScore(r,300,1800),0)/evolutionRuns.length:collectorScore;
    const total=Number(hb.control?.managed_jobs_total||0),active=Number(hb.control?.managed_jobs_active||0);
    const jobs=total>0?clamp(100*active/total):collectorScore;
    const health=clamp(collectorScore*.42+jobs*.23+alphaFresh*.18+treasuryFresh*.17);
    const measured=clamp(alphaFresh*.42+worldScore*.25+labScore*.23+treasuryFresh*.10);
    const evidence=clamp(45+Math.min(35,runs.length*3)+health*.20);
    const general=clamp(measured*.45+health*.35+evidence*.20);
    const mk=(id,name,q,e=90)=>({id,name,quality_pct:Math.round(q*10)/10,evidence_pct:Math.round(e*10)/10,maturity_pct:Math.round((q*e/100)*10)/10,samples:runs.length,metrics:{source:'frontier-heartbeat'}});
    const arms=collectors.map(([id,r])=>({id,success_pct:Math.round(runScore(r)*10)/10,runs:1,success:String(r?.status||'').toUpperCase()==='SUCCESS'?1:0,degraded:String(r?.status||'').toUpperCase()==='DEGRADED'?1:0,failed:String(r?.status||'').toUpperCase()==='FAILED'?1:0,observed_records:Number(r?.observed_records||0),stored_records:Number(r?.stored_records||0)}));
    const strengths=[
      {name:'Canlı veri akışı',maturity_pct:health},
      {name:'ALPHA tazeliği',maturity_pct:alphaFresh},
      {name:'Dünya algısı',maturity_pct:worldScore},
    ].filter(x=>x.maturity_pct>=45);
    const gaps=[];
    if(alphaFresh<45)gaps.push({severity:'HIGH',domain:'alpha',capability_id:'alpha_freshness'});
    if(worldScore<45)gaps.push({severity:'MEDIUM',domain:'world',capability_id:'world_freshness'});
    if(collectorScore<60)gaps.push({severity:'MEDIUM',domain:'sensors',capability_id:'collector_health'});
    return {
      status:'OK',source:'frontier-heartbeat',generated_at:new Date().toISOString(),
      overall:{brain_development_pct:Math.round(general*10)/10,data_exchange_health_pct:Math.round(health*10)/10,measured_quality_pct:Math.round(measured*10)/10,evidence_confidence_pct:Math.round(evidence*10)/10,tier:tier(general)},
      components:[mk('alpha','ALPHA Karar Kalitesi',alphaFresh),mk('sensors','Veri Algısı ve Akış',collectorScore),mk('world','Dünya / Hafıza',worldScore),mk('lab','Araştırma / Gelişim',labScore),mk('treasury','Hazine / Uygulama',treasuryFresh)],
      arms,feature_signals:[],capability_gaps:gaps,strengths,
    };
  }
  function devCache(){
    if(memoryText&&Date.now()-memoryAt<6*60*60*1000)return{txt:memoryText,at:memoryAt};
    try{const raw=localStorage.getItem(CACHE);if(!raw)return null;const c=JSON.parse(raw);if(!c?.text)return null;const at=Number(c.savedAt||0);if(Date.now()-at>6*60*60*1000)return null;memoryText=String(c.text);memoryAt=at;return{txt:memoryText,at}}catch(_e){return null}
  }
  function anatomyResponse(){
    const live=syntheticDevelopment();
    if(live){const txt=JSON.stringify(live);memoryText=txt;memoryAt=Date.now();try{localStorage.setItem(CACHE,JSON.stringify({savedAt:memoryAt,text:txt}))}catch(_e){};return response(txt,'anatomy-heartbeat')}
    const c=devCache();
    if(c)return response(c.txt,'anatomy-last-known');
    return null;
  }
  function bodyObject(init){try{return typeof init?.body==='string'&&init.body?JSON.parse(init.body):{}}catch(_e){return null}}
  function safeRead(url,init){if(String(init?.method||'GET').toUpperCase()!=='POST')return null;const body=bodyObject(init);if(body===null)return null;if(Object.keys(body).length&&body.action!=='status')return null;const match=TTL.find(([needle])=>url.includes(needle));return match?{ttl:match[1],key:`${match[0]}|${JSON.stringify(body)}`}:null}
  async function liveRead(input,init,meta){
    const now=Date.now(),cached=readCache.get(meta.key);if(cached&&now-cached.at<meta.ttl)return response(cached.txt,'frontier-fresh');
    if(readInflight.has(meta.key))return readInflight.get(meta.key).then(x=>response(x.txt,'frontier-deduped',x.status));
    const p=wrapped(input,init).then(async r=>{const txt=await r.clone().text();if(r.ok)readCache.set(meta.key,{txt,at:Date.now(),status:r.status});return{txt,status:r.status,ok:r.ok}}).catch(err=>{const stale=readCache.get(meta.key);if(stale)return{txt:stale.txt,status:200,ok:true,stale:true};throw err}).finally(()=>setTimeout(()=>readInflight.delete(meta.key),0));
    readInflight.set(meta.key,p);const x=await p;return response(x.txt,x.stale?'frontier-stale':'frontier-live',x.status||200)
  }

  window.fetch=(input,init={})=>{
    const url=typeof input==='string'?input:String(input?.url||'');
    if(url.includes(DEV)){
      const local=anatomyResponse();
      if(local)return Promise.resolve(local);
      if(!visible())return Promise.reject(new Error('BRIAN_ANATOMY_DORMANT'));
      return wrapped(input,init);
    }
    const meta=safeRead(url,init);if(meta)return liveRead(input,init,meta);return wrapped(input,init);
  };

  document.addEventListener('click',e=>{if(e.target?.closest?.('#balOpen,.bax-m-refresh,.bax-refresh'))setTimeout(()=>{const local=anatomyResponse();if(local&&visible())document.dispatchEvent(new CustomEvent('brian-anatomy-heartbeat-ready'))},60)},true);
})();