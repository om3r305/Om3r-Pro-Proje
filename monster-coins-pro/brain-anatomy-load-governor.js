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
    if(inflight)return inflight.then(({txt})=>response(txt,'deduped'));

    inflight=wrapped(input,init).then(async r=>{
      if(!r.ok)return{txt:await r.clone().text(),status:r.status,ok:false,headers:r.headers};
      const txt=await r.clone().text();
      memoryText=txt;memoryAt=Date.now();
      try{localStorage.setItem(CACHE,JSON.stringify({savedAt:memoryAt,text:txt}))}catch(_e){}
      return{txt,status:r.status,ok:true,headers:r.headers};
    }).finally(()=>{setTimeout(()=>{inflight=null},0)});

    return inflight.then(x=>{
      if(x.ok)return response(x.txt,'live');
      return new Response(x.txt,{status:x.status||500,headers:{'content-type':'application/json; charset=utf-8'}});
    });
  };
})();
