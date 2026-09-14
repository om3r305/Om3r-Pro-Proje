'use strict';

(()=>{
  const API='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-dip-multiasset-status';
  const KEY='mcp-dashboard-key-v1';
  const el=id=>document.getElementById(id);
  const toast=(text,kind='ok')=>{const n=el('toast');if(!n)return;n.textContent=text;n.className=`toast show ${kind}`;clearTimeout(n._t);n._t=setTimeout(()=>n.className='toast',4200);};
  const post=async(key,body)=>{const r=await fetch(API,{method:'POST',cache:'no-store',headers:{'content-type':'application/json','x-brian-dashboard-key':key},body:JSON.stringify(body)});const d=await r.json().catch(()=>({}));if(!r.ok)throw Object.assign(new Error(d.error||d.status||`HTTP ${r.status}`),{status:r.status});return d;};
  const key=()=>localStorage.getItem(KEY)||'';
  const showLock=show=>el('unlock')?.classList.toggle('show',show);

  async function validateAndUnlock(){
    const input=el('unlockKey'),value=String(input?.value||'').trim();
    if(!value){toast('Dashboard anahtarı gerekli.','err');return;}
    const btn=el('unlockBtn');if(btn)btn.disabled=true;
    try{
      const data=await post(value,{action:'status'});
      localStorage.setItem(KEY,value);showLock(false);if(input)input.value='';
      window.dispatchEvent(new CustomEvent('dip:guardian-status',{detail:data}));
      window.dispatchEvent(new CustomEvent('dip:guardian-refresh'));
      toast('Guardian bağlantısı açıldı.');
    }catch(e){localStorage.removeItem(KEY);showLock(true);toast(e?.status===401?'Dashboard anahtarı geçersiz.':`Guardian bağlantısı açılamadı: ${e?.message||e}`,'err');}
    finally{if(btn)btn.disabled=false;}
  }

  async function restart(){
    const capital=Number(el('v841Capital')?.value);
    if(!Number.isFinite(capital)||capital<10||capital>1_000_000){toast('Test kasası 10–1.000.000 USDT arasında olmalı.','err');return;}
    const k=key();if(!k){showLock(true);toast('Önce Guardian kilidini aç.','err');return;}
    const btn=el('v841Restart');if(btn)btn.disabled=true;
    try{
      const before=await post(k,{action:'status'});
      if(Array.isArray(before?.positions)&&before.positions.length)throw new Error(`OPEN_POSITION_MULTI:${before.positions.map(p=>p.symbol).join(', ')}`);
      const data=await post(k,{action:'restart',starting_equity:capital});
      window.dispatchEvent(new CustomEvent('dip:session-restarted',{detail:{sessionId:data?.source_session_id||'',startingEquity:capital}}));
      window.dispatchEvent(new CustomEvent('dip:guardian-status',{detail:data}));
      toast(`${capital.toFixed(0)} USDT · yeni Guardian SHADOW session başladı.`);
      setTimeout(()=>location.reload(),650);
    }catch(e){const m=String(e?.message||e);if(m.includes('OPEN_POSITION_MULTI'))toast(`Açık DIP pozisyonu var (${m.split(':').slice(1).join(':')}). Önce kapanmalı.`,'err');else toast(m,'err');}
    finally{if(btn)btn.disabled=false;}
  }

  window.addEventListener('load',()=>{
    showLock(!key());
    const unlock=el('unlockBtn'),input=el('unlockKey'),restartBtn=el('v841Restart'),refresh=el('refreshBtn');
    if(unlock)unlock.onclick=validateAndUnlock;
    if(input)input.addEventListener('keydown',e=>{if(e.key==='Enter')validateAndUnlock();});
    if(restartBtn)restartBtn.onclick=restart;
    if(refresh)refresh.onclick=()=>{window.dispatchEvent(new CustomEvent('dip:guardian-refresh'));toast('Guardian verisi yenileniyor…');};
  });
})();
