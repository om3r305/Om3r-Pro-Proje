'use strict';

(()=>{
  const API='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-dip-v84-view';
  const KEY_NAME='mcp-dashboard-key-v1';
  const $=id=>document.getElementById(id);
  const toast=(text,kind='ok')=>{const el=$('toast');if(!el)return;el.textContent=text;el.className=`toast show ${kind}`;clearTimeout(el._t);el._t=setTimeout(()=>el.className='toast',3500);};

  async function restart(){
    const capital=Number($('v841Capital')?.value);
    if(!Number.isFinite(capital)||capital<10||capital>1_000_000){toast('Test kasası 10–1.000.000 USDT arasında olmalı.','err');return;}
    const key=localStorage.getItem(KEY_NAME)||'';
    if(!key){toast('Önce dashboard kilidini aç.','err');return;}
    const btn=$('v841Restart');if(btn)btn.disabled=true;
    try{
      const r=await fetch(API,{method:'POST',cache:'no-store',headers:{'content-type':'application/json','x-brian-dashboard-key':key},body:JSON.stringify({action:'restart',starting_equity:capital})});
      const d=await r.json().catch(()=>({}));
      if(!r.ok)throw new Error(d.error||`HTTP ${r.status}`);
      toast(`${capital.toFixed(0)} USDT ile yeni SHADOW session açıldı.`);
      setTimeout(()=>location.reload(),650);
    }catch(e){
      const m=String(e?.message||e);
      toast(m.includes('OPEN_POSITION')?'Açık pozisyon varken yeni kasa başlatılamaz.':m,'err');
    }finally{if(btn)btn.disabled=false;}
  }

  window.addEventListener('load',()=>{
    const input=$('v841Capital'),btn=$('v841Restart');
    if(input){
      const current=Number(document.getElementById('kpiEquity')?.textContent?.replace(/[^0-9.]/g,''));
      if(Number.isFinite(current)&&current>=10)input.value=String(Math.round(current));
    }
    if(btn)btn.onclick=restart;
  });
})();
