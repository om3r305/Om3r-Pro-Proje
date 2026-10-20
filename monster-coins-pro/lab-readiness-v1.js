'use strict';
(()=>{
  const ID='labReadinessLiveV1';
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct=v=>v!==null&&v!==undefined&&v!==''&&Number.isFinite(Number(v))?`${Number(v).toFixed(1)}%`:'—';
  const state={data:null,error:null,busy:false,last:0};
  function style(){if(document.getElementById('labReadinessCssV1'))return;const s=document.createElement('style');s.id='labReadinessCssV1';s.textContent=`#${ID}{border:1px solid #35ccff33;background:linear-gradient(145deg,#061f33,#020d1a);border-radius:16px;padding:13px}.lrv1h{display:flex;justify-content:space-between;align-items:center;gap:8px}.lrv1h b{font-size:12px}.lrv1badge{font-size:8px;font-weight:900;border:1px solid #49dfbd55;color:#70f2c9;border-radius:999px;padding:4px 7px}.lrv1grid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-top:10px}.lrv1k{background:#041827;border:1px solid #35ccff1a;border-radius:11px;padding:9px}.lrv1k span{display:block;color:#83a6b5;font-size:7px;letter-spacing:.06em}.lrv1k strong{display:block;font-size:18px;margin-top:4px}.lrv1bar{height:4px;background:#0b2a3b;border-radius:99px;overflow:hidden;margin-top:6px}.lrv1bar i{display:block;height:100%;background:linear-gradient(90deg,#25dfff,#62f1b6)}.lrv1cand{margin-top:8px;padding:8px;border-radius:10px;background:#031420;font-size:8px;color:#8faab6}.lrv1cand b{color:#dffcff}.lrv1warn{color:#ffc55a}.lrv1ok{color:#62f1b6}@media(max-width:520px){.lrv1grid{grid-template-columns:1fr}.lrv1k strong{font-size:22px}}`;document.head.appendChild(s)}
  function mount(){style();if(document.getElementById(ID))return document.getElementById(ID);const side=document.querySelector('#brianAnatomyLive .ba2side');if(!side)return null;const p=document.createElement('section');p.id=ID;side.prepend(p);return p}
  function render(){
    const p=mount();if(!p)return;
    const d=state.data,m=d?.metrics||{};
    const stale=state.error||d?.stale===true||!Number.isFinite(Date.parse(d?.observed_at))||Date.parse(d.observed_at)>Date.now()||Date.now()-Date.parse(d.observed_at)>180000;
    const stamp=d?.observed_at?new Date(d.observed_at).toLocaleString('tr-TR',{timeZone:'Europe/Berlin'}):'Zaman bilinmiyor';
    const header=`<div class="lrv1h"><b>⚗ LABORATUVAR HAZIRLIĞI</b><span class="lrv1badge">${!d?'VERİ BEKLENİYOR':stale?'SON KAYIT · GÜNCEL DEĞİL':'GÜNCEL RAPOR'}</span></div>`;
    if(!d){p.innerHTML=header+`<div class="lrv1cand lrv1warn">${esc(state.error||'Hazırlık verisi henüz alınmadı.')}</div>`;return}
    const cards=[['VERİ HATTI SAĞLIĞI',m.pipeline_health_pct],['KANIT OLGUNLUĞU',m.evidence_maturity_pct],['TERFİ HAZIRLIĞI',m.promotion_readiness_pct]];
    p.innerHTML=header+`<p class="lrv1cand">${esc(stamp)}${state.error?' · '+esc(state.error):''}</p><div class="lrv1grid">${cards.map(([name,v])=>`<div class="lrv1k"><span>${name}</span><strong>${pct(v)}</strong></div>`).join('')}</div>`+
      (d.candidates||[]).map(c=>`<div class="lrv1cand"><b>${esc(c.family)}</b> · ${esc(c.samples??'—')}/${esc(c.minimum_samples??'—')} örnek · ${esc(c.regimes??'—')}/${esc(c.minimum_regimes??'—')} rejim · net ${c.net_edge_bps==null?'—':esc(Number(c.net_edge_bps).toFixed(1))+' bps'}<br><span class="lrv1warn">${esc(c.council_decision||'Kanıt bekleniyor')}${c.promotion_ready===true?' · Terfi adayı, onay gerekli':''}</span></div>`).join('')+
      '<div class="lrv1cand">Terfi için insan onayı gerekir. Bu rapor deneyi veya işlemi başlatmaz.</div>';
  }
  async function load(force=false){
    if(state.busy||document.hidden)return;
    if(!force&&state.data&&Date.now()-state.last<120000){render();return}
    if(typeof post!=='function'||typeof ROOT==='undefined')return;
    if(typeof key!=='function'||!key()){state.error='Laboratuvar raporu için dashboard kilidini aç.';render();return}
    state.busy=true;state.error=null;render();
    try{
      const next=await post(`${ROOT}/brian-lab-readiness-status`,{});
      if(next?.shadow_only!==true||next?.live_execution!==false||!next?.metrics)throw Error('Laboratuvar raporu doğrulanamadı.');
      state.data=next;state.last=Date.now();
    }catch(e){state.error='Laboratuvar verisi yenilenemedi. Son rapor varsa güncel değildir.';if(e?.status===401)state.data=null}
    finally{state.busy=false;render()}
  }
  function wire(){
    mount();
    const modal=document.getElementById('brianAnatomyLive');
    const isOpen=()=>modal?.classList.contains('show')||modal?.classList.contains('bal-show');
    if(modal)new MutationObserver(()=>{if(isOpen())load(false)}).observe(modal,{attributes:true,attributeFilter:['class']});
    document.getElementById('ba2refresh')?.addEventListener('click',()=>load(true));
    document.getElementById('unlockBtn')?.addEventListener('click',()=>{if(isOpen())load(true)});
    if(isOpen())load(false);
    setInterval(()=>{if(isOpen())load(false)},60000);
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',()=>setTimeout(wire,700));else setTimeout(wire,700);
})();
