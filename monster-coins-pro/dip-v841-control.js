'use strict';

(()=>{
  const CORE_API='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-dip-v84-view';
  const MULTI_API='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-dip-multiasset-status';
  const KEY_NAME='mcp-dashboard-key-v1';
  const $=id=>document.getElementById(id);
  const toast=(text,kind='ok')=>{const el=$('toast');if(!el)return;el.textContent=text;el.className=`toast show ${kind}`;clearTimeout(el._t);el._t=setTimeout(()=>el.className='toast',5000);};
  const post=async(url,key,body)=>{const r=await fetch(url,{method:'POST',cache:'no-store',headers:{'content-type':'application/json','x-brian-dashboard-key':key},body:JSON.stringify(body)});const d=await r.json().catch(()=>({}));if(!r.ok)throw new Error(d.error||d.status||`HTTP ${r.status}`);return d;};
  const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));

  async function restart(){
    const capital=Number($('v841Capital')?.value);
    if(!Number.isFinite(capital)||capital<10||capital>1_000_000){toast('Test kasası 10–1.000.000 USDT arasında olmalı.','err');return;}
    const key=localStorage.getItem(KEY_NAME)||'';
    if(!key){toast('Önce dashboard kilidini aç.','err');return;}
    const btn=$('v841Restart');if(btn)btn.disabled=true;
    try{
      // Unified DIP rule: no session reset while either ETH core or an altcoin DIP position is open.
      const multiBefore=await post(MULTI_API,key,{action:'status'}).catch(()=>null);
      if(Array.isArray(multiBefore?.positions)&&multiBefore.positions.length){
        throw new Error(`OPEN_POSITION_MULTI:${multiBefore.positions.map(p=>p.symbol).join(', ')}`);
      }

      // ETH core is the session authority. If it rejects restart, multi-asset state is untouched.
      const core=await post(CORE_API,key,{action:'restart',starting_equity:capital});
      const sourceSessionId=String(core?.session?.session_id||'');
      if(!sourceSessionId)throw new Error('CORE_SESSION_ID_MISSING');

      let multi=null,lastError=null;
      for(let attempt=0;attempt<3;attempt++){
        try{
          multi=await post(MULTI_API,key,{action:'restart',starting_equity:capital,source_session_id:sourceSessionId});
          if(String(multi?.source_session_id||'')!==sourceSessionId)throw new Error('MULTI_SESSION_LINK_MISMATCH');
          break;
        }catch(e){lastError=e;if(attempt<2)await sleep(300*(attempt+1));}
      }
      if(!multi)throw new Error(`MULTI_SESSION_SYNC_FAILED:${String(lastError?.message||lastError||'unknown')}`);

      window.dispatchEvent(new CustomEvent('dip:session-restarted',{detail:{sessionId:sourceSessionId,startingEquity:capital}}));
      toast(`${capital.toFixed(0)} USDT · tek DIP SHADOW session hazır: ETH + altcoinler birlikte yenilendi.`);
      setTimeout(()=>location.reload(),800);
    }catch(e){
      const m=String(e?.message||e);
      if(m.includes('OPEN_POSITION_MULTI'))toast(`Altcoin DIP pozisyonu açık (${m.split(':').slice(1).join(':')}). Önce kapanmasını bekle.`, 'err');
      else if(m.includes('OPEN_POSITION'))toast('Açık DIP pozisyonu varken yeni session başlatılamaz.','err');
      else if(m.includes('MULTI_SESSION_SYNC_FAILED'))toast('ETH session açıldı ama altcoin DIP bağlantısı tamamlanamadı. Yenilemeden önce kontrol et.','err');
      else toast(m,'err');
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
