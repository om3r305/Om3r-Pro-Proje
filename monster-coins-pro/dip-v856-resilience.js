'use strict';

(()=>{
  const el=id=>document.getElementById(id);
  let last=null,lastAt=0;

  function n(v,f=0){const x=Number(v);return Number.isFinite(x)?x:f;}
  function remember(d){if(!d||!d.engine_id)return;last=d;lastAt=Date.now();}
  function fallbackText(){
    const d=last||window.__DIP_GUARDIAN_STATE__||null;
    if(!d)return 'GUARDIAN SYNC · bağlantı yenileniyor';
    const scan=d.last_scan||{},version=String(d.engine_version||scan.engine_version||'V8.5.6'),mode=String(d.risk_mode||scan.risk_mode||'DEFENSIVE').toUpperCase(),age=Math.max(0,Math.round(n(d.age_seconds,(Date.now()-lastAt)/1000)));
    return `${version} ${mode} · SYNC · son sağlam worker ${age} sn`;
  }
  function heal(){
    const node=el('multiDipState');if(!node)return;
    const text=String(node.textContent||'').toUpperCase();
    if(text.includes('VERİ HATASI')||text.includes('DATA ERROR')){
      node.textContent=fallbackText();
      node.style.color='#f3c969';
      node.dataset.transient='1';
    }
  }
  function install(){
    const node=el('multiDipState');if(!node){setTimeout(install,120);return;}
    new MutationObserver(heal).observe(node,{childList:true,characterData:true,subtree:true});
    heal();
  }
  window.addEventListener('dip:guardian-status',e=>{remember(e?.detail);heal();});
  window.addEventListener('load',()=>{remember(window.__DIP_GUARDIAN_STATE__);install();});
})();
