'use strict';

(()=>{
  const STATUS_URL='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-dip-multiasset-status';
  const CACHE_KEY='dip-v855-last-good-status';
  const nativeFetch=window.fetch.bind(window);
  const MAX_CACHE_AGE=5*60_000;
  let cachedBody='',cachedAt=0;

  try{
    const saved=JSON.parse(sessionStorage.getItem(CACHE_KEY)||'null');
    if(saved&&typeof saved.body==='string'&&Number.isFinite(Number(saved.at))&&Date.now()-Number(saved.at)<MAX_CACHE_AGE){cachedBody=saved.body;cachedAt=Number(saved.at);}
  }catch{}

  function isStatusRequest(input,init){
    const url=typeof input==='string'?input:String(input?.url||'');
    if(url!==STATUS_URL)return false;
    const method=String(init?.method||input?.method||'GET').toUpperCase();
    if(method!=='POST')return false;
    const body=typeof init?.body==='string'?init.body:'';
    return !body||body.includes('"action":"status"')||body.includes('"action": "status"');
  }
  function save(body){cachedBody=body;cachedAt=Date.now();try{sessionStorage.setItem(CACHE_KEY,JSON.stringify({body,at:cachedAt}));}catch{}}
  function cachedResponse(){
    if(!cachedBody||Date.now()-cachedAt>MAX_CACHE_AGE)return null;
    return new Response(cachedBody,{status:200,headers:{'content-type':'application/json; charset=utf-8','cache-control':'no-store','x-guardian-cache':'last-good'}});
  }

  window.fetch=async function guardedFetch(input,init){
    if(!isStatusRequest(input,init))return nativeFetch(input,init);
    try{
      const response=await nativeFetch(input,init);
      if(response.ok){
        const body=await response.clone().text();
        if(body&&body.includes('"engine_id"'))save(body);
        return response;
      }
      return cachedResponse()||response;
    }catch(error){
      const fallback=cachedResponse();
      if(fallback)return fallback;
      throw error;
    }
  };
})();
