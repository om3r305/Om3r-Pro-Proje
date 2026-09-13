'use strict';
(()=>{
  const DEV='/brian-development-status';
  const CACHE='brian-development-status-v1';
  const wrapped=window.fetch.bind(window);
  let inflight=null;
  let memoryText='';
  let memoryAt=0;

  function visible(){return !!document.querySelector('#brianAnatomyLive.bal-show')}
  function readCache(){
    if(memoryText&&Date.now()-memoryAt<6*60*60*1000)return{txt:memoryText,at:memoryAt};
    try{
      const raw=localStorage.getItem(CACHE);if(!raw)return null;
      const c=JSON.parse(raw);if(!c?.text)return null;
      const at=Number(c.savedAt||0);if(Date.now()-at>6*60*60*1000)return null;
      memoryText=String(c.text);memoryAt=at;return{txt:memoryText,at};
    }catch(_e){return null}
  }
  function response(txt,tag='memory'){
    return new Response(txt,{status:200,headers:{'content-type':'application/json; charset=utf-8','x-brian-anatomy-cache':tag}});
  }
  function dormant(){
    const c=readCache();
    if(c)return Promise.resolve(response(c.txt,'hidden-cache'));
    return Promise.reject(new Error('BRIAN_ANATOMY_DORMANT'));
  }

  window.fetch=(input,init={})=>{
    const url=typeof input==='string'?input:String(input?.url||'');
    if(!url.includes(DEV))return wrapped(input,init);

    // Anatomy is an on-demand diagnostic. Never let hidden UI polling hammer Brian/DB.
    if(!visible())return dormant();

    // Multiple legacy anatomy layers can ask at once; serve one live request to all of them.
    const c=readCache();
    if(c&&Date.now()-c.at<25000)return Promise.resolve(response(c.txt,'fresh-cache'));
    if(inflight)return inflight.then(({txt,status,ok})=>ok?response(txt,'deduped'):new Response(txt,{status:status||500,headers:{'content-type':'application/json; charset=utf-8'}}));

    inflight=wrapped(input,init).then(async r=>{
      const txt=await r.clone().text();
      if(r.ok){
        memoryText=txt;memoryAt=Date.now();
        try{localStorage.setItem(CACHE,JSON.stringify({savedAt:memoryAt,text:txt}))}catch(_e){}
      }
      return{txt,status:r.status,ok:r.ok};
    }).finally(()=>{setTimeout(()=>{inflight=null},0)});

    return inflight.then(x=>x.ok?response(x.txt,'live'):new Response(x.txt,{status:x.status||500,headers:{'content-type':'application/json; charset=utf-8'}}));
  };

  // The exact skin has its own state. Wake it immediately after the user opens anatomy.
  document.addEventListener('click',e=>{
    if(!e.target?.closest?.('#balOpen'))return;
    setTimeout(()=>{
      if(!visible())return;
      const refresh=document.querySelector('.bax-ref-wrap .bax-refresh');
      if(refresh)refresh.click();
    },80);
  },true);

  document.addEventListener('visibilitychange',()=>{
    if(document.visibilityState!=='visible'||!visible())return;
    const c=readCache();
    if(!c||Date.now()-c.at>60000)setTimeout(()=>document.querySelector('.bax-ref-wrap .bax-refresh')?.click(),80);
  });
})();
