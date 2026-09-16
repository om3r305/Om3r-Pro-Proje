'use strict';

(()=>{
  const STATUS_PATH='brian-dip-multiasset-status';
  let latestEvents=[];
  let applying=false;
  let observer=null;

  const money=v=>{
    const x=Number(v);
    return Number.isFinite(x)?`$${x.toFixed(2)}`:'—';
  };

  function apply(){
    if(applying)return;
    const body=document.getElementById('mdEvents');
    const table=body?.closest('table');
    const head=table?.querySelector('thead tr');
    if(!body||!head)return;
    applying=true;
    try{
      const headers=[...head.children];
      if(!headers.some(x=>x.dataset?.notionalCol==='1')){
        const th=document.createElement('th');
        th.dataset.notionalCol='1';
        th.textContent='Tutar ($)';
        head.insertBefore(th,headers[4]||null);
      }

      const rows=[...body.querySelectorAll('tr')];
      rows.forEach((tr,i)=>{
        const cells=[...tr.children];
        if(cells.length===1){
          if(cells[0].getAttribute('colspan')!=='7')cells[0].setAttribute('colspan','7');
          return;
        }
        const ev=latestEvents[i];
        if(!ev)return;
        let td=tr.querySelector('td[data-notional-col="1"]');
        if(!td){
          td=document.createElement('td');
          td.dataset.notionalCol='1';
          tr.insertBefore(td,tr.children[4]||null);
        }
        const value=money(ev.notional);
        if(td.textContent!==value)td.textContent=value;
        const title=String(ev.action||'').toUpperCase()==='BUY'?'Alım için kullanılan USDT':'Satış işlem tutarı';
        if(td.title!==title)td.title=title;
        if(td.style.fontVariantNumeric!=='tabular-nums')td.style.fontVariantNumeric='tabular-nums';
        if(td.style.fontWeight!=='800')td.style.fontWeight='800';
      });
    }finally{
      applying=false;
    }
  }

  function observe(){
    const body=document.getElementById('mdEvents');
    if(!body||observer)return false;
    observer=new MutationObserver(()=>{
      if(!applying)queueMicrotask(apply);
    });
    observer.observe(body,{childList:true,subtree:true});
    apply();
    return true;
  }

  const originalFetch=window.fetch.bind(window);
  window.fetch=async(...args)=>{
    const response=await originalFetch(...args);
    try{
      const url=String(args?.[0]?.url||args?.[0]||'');
      if(url.includes(STATUS_PATH)){
        response.clone().json().then(data=>{
          if(Array.isArray(data?.recent_events)){
            latestEvents=data.recent_events.slice(0,40);
            queueMicrotask(()=>{observe();apply();});
          }
        }).catch(()=>{});
      }
    }catch{}
    return response;
  };

  window.addEventListener('load',()=>{
    if(observe())return;
    const timer=setInterval(()=>{if(observe())clearInterval(timer);},250);
    setTimeout(()=>clearInterval(timer),10000);
  });
})();
