'use strict';

(()=>{
  const DIRECT='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-dip-multiasset-status';
  const PROXY='/api/brian/brian-dip-multiasset-status';
  const CACHE_KEY='dip-v880-last-good-status';
  const nativeFetch=window.fetch.bind(window);
  const MAX_CACHE_AGE=45_000;
  let cachedBody='',cachedAt=0;

  try{
    const saved=JSON.parse(sessionStorage.getItem(CACHE_KEY)||'null');
    if(saved&&typeof saved.body==='string'&&Number.isFinite(Number(saved.at))&&Date.now()-Number(saved.at)<MAX_CACHE_AGE){cachedBody=saved.body;cachedAt=Number(saved.at);}
  }catch{}

  function requestInfo(input,init){
    const url=typeof input==='string'?input:String(input?.url||'');
    const target=url===DIRECT||url===PROXY;
    const method=String(init?.method||input?.method||'GET').toUpperCase();
    const body=typeof init?.body==='string'?init.body:'';
    const isStatus=target&&method==='POST'&&(!body||body.includes('"action":"status"')||body.includes('"action": "status"'));
    return{url,target,isStatus};
  }
  function versionAtLeast880(body){
    try{
      const d=JSON.parse(body||'{}'),m=String(d?.engine_version||d?.last_scan?.engine_version||'').match(/^V?(\d+)\.(\d+)\.(\d+)/);
      if(!m)return false;
      const major=Number(m[1]),minor=Number(m[2]);
      return major>8||(major===8&&minor>=8);
    }catch{return false;}
  }
  function save(body){
    if(!versionAtLeast880(body))return;
    cachedBody=body;cachedAt=Date.now();
    try{sessionStorage.setItem(CACHE_KEY,JSON.stringify({body,at:cachedAt}));}catch{}
  }
  function cachedResponse(){
    if(!cachedBody||Date.now()-cachedAt>MAX_CACHE_AGE||!versionAtLeast880(cachedBody))return null;
    return new Response(cachedBody,{status:200,headers:{'content-type':'application/json; charset=utf-8','cache-control':'no-store','x-guardian-cache':'last-good-v880'}});
  }

  window.fetch=async function guardedFetch(input,init){
    const info=requestInfo(input,init);
    if(!info.target)return nativeFetch(input,init);
    const proxied=info.url===DIRECT?PROXY:input;
    try{
      const response=await nativeFetch(proxied,init);
      if(info.isStatus&&response.ok){
        const body=await response.clone().text();
        if(body&&body.includes('"engine_id"'))save(body);
        return response;
      }
      if(info.isStatus&&!response.ok)return cachedResponse()||response;
      return response;
    }catch(error){
      if(info.isStatus){const fallback=cachedResponse();if(fallback)return fallback;}
      throw error;
    }
  };
})();
