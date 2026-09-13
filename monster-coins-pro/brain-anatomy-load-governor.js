'use strict';
(()=>{
  const DEV='/brian-development-status';
  const CACHE='brian-development-status-v1';
  const wrapped=window.fetch.bind(window);
  let devInflight=null;
  let memoryText='';
  let memoryAt=0;
  const readCache=new Map();
  const readInflight=new Map();

  const TTL=[
    ['/brian-world-status',45000],
    ['/brian-evolution-status',60000],
    ['/brian-evolution-treasury-status',20000],
    ['/brian-evolution-ocean-status',45000],
    ['/brian-evolution-lab-status',60000],
    ['/brian-evolution-alpha-intelligence-status',30000],
    ['/brian-frontier-news',45000],
  ];

  function visible(){return !!document.querySelector('#brianAnatomyLive.bal-show')}
  function devCache(){
    if(memoryText&&Date.now()-memoryAt<6*60*60*1000)return{txt:memoryText,at:memoryAt};
    try{
      const raw=localStorage.getItem(CACHE);if(!raw)return null;
      const c=JSON.parse(raw);if(!c?.text)return null;
      const at=Number(c.savedAt||0);if(Date.now()-at>6*60*60*1000)return null;
      memoryText=String(c.text);memoryAt=at;return{txt:memoryText,at};
    }catch(_e){return null}
  }
  function response(txt,tag='memory',status=200){
    return new Response(txt,{status,headers:{'content-type':'application/json; charset=utf-8','x-brian-cache':tag}});
  }
  function dormant(){
    const c=devCache();
    if(c)return Promise.resolve(response(c.txt,'anatomy-hidden-cache'));
    return Promise.reject(new Error('BRIAN_ANATOMY_DORMANT'));
  }
  function bodyObject(init){
    try{return typeof init?.body==='string'&&init.body?JSON.parse(init.body):{}}catch(_e){return null}
  }
  function safeRead(url,init){
    if(String(init?.method||'GET').toUpperCase()!=='POST')return null;
    const body=bodyObject(init);if(body===null)return null;
    // Never cache commands. Empty-body status endpoints are reads; action=status is also read-only.
    if(Object.keys(body).length&&body.action!=='status')return null;
    const match=TTL.find(([needle])=>url.includes(needle));
    return match?{ttl:match[1],key:`${match[0]}|${JSON.stringify(body)}`}:null;
  }
  async function liveRead(input,init,meta){
    const now=Date.now(),cached=readCache.get(meta.key);
    if(cached&&now-cached.at<meta.ttl)return response(cached.txt,'frontier-fresh');
    if(readInflight.has(meta.key))return readInflight.get(meta.key).then(x=>response(x.txt,'frontier-deduped',x.status));
    const p=wrapped(input,init).then(async r=>{
      const txt=await r.clone().text();
      if(r.ok)readCache.set(meta.key,{txt,at:Date.now(),status:r.status});
      return{txt,status:r.status,ok:r.ok};
    }).catch(err=>{
      const stale=readCache.get(meta.key);
      if(stale)return{txt:stale.txt,status:200,ok:true,stale:true};
      throw err;
    }).finally(()=>setTimeout(()=>readInflight.delete(meta.key),0));
    readInflight.set(meta.key,p);
    const x=await p;
    return response(x.txt,x.stale?'frontier-stale':'frontier-live',x.status||200);
  }

  window.fetch=(input,init={})=>{
    const url=typeof input==='string'?input:String(input?.url||'');

    if(url.includes(DEV)){
      if(!visible())return dormant();
      const c=devCache();
      if(c&&Date.now()-c.at<25000)return Promise.resolve(response(c.txt,'anatomy-fresh'));
      if(devInflight)return devInflight.then(x=>response(x.txt,'anatomy-deduped',x.status));
      devInflight=wrapped(input,init).then(async r=>{
        const txt=await r.clone().text();
        if(r.ok){memoryText=txt;memoryAt=Date.now();try{localStorage.setItem(CACHE,JSON.stringify({savedAt:memoryAt,text:txt}))}catch(_e){}}
        return{txt,status:r.status,ok:r.ok};
      }).finally(()=>setTimeout(()=>{devInflight=null},0));
      return devInflight.then(x=>response(x.txt,x.ok?'anatomy-live':'anatomy-error',x.status||500));
    }

    const meta=safeRead(url,init);
    if(meta)return liveRead(input,init,meta);
    return wrapped(input,init);
  };

  // Wake the exact anatomy state immediately on user open, but still only one real request is allowed.
  document.addEventListener('click',e=>{
    if(!e.target?.closest?.('#balOpen'))return;
    setTimeout(()=>{if(visible())document.querySelector('.bax-ref-wrap .bax-refresh')?.click()},80);
  },true);

  document.addEventListener('visibilitychange',()=>{
    if(document.visibilityState!=='visible'||!visible())return;
    const c=devCache();
    if(!c||Date.now()-c.at>60000)setTimeout(()=>document.querySelector('.bax-ref-wrap .bax-refresh')?.click(),80);
  });
})();
