'use strict';
(()=>{
  const $=(q,r=document)=>r.querySelector(q);
  const clamp=n=>Math.max(0,Math.min(100,Number(n)||0));
  const pct=n=>Number.isFinite(Number(n))?`${Number(n).toFixed(1)}%`:'—';
  function hb(){return window.STABILITY?.heartbeat||null}
  function alphaPct(h){const a=h?.alpha;if(!a)return null;const q=Number(a.confidence??a.score??a.edge_confidence);return Number.isFinite(q)?clamp(q*100):75}
  function healthPct(h){if(!h)return null;const runs=Object.values(h.collectors||{});if(!runs.length)return null;const ok=runs.filter(r=>String(r?.status||'').toUpperCase()==='SUCCESS').length;return clamp(ok/runs.length*100)}
  function worldPct(h){const w=h?.world_run;if(!w)return null;const x=Number(w.entity_observations||0)+Number(w.narrative_snapshots||0)+Number(w.asset_impacts||0);return clamp(45+Math.log10(1+x)*14)}
  function metrics(){
    const h=hb(),health=healthPct(h),alpha=alphaPct(h),world=worldPct(h);
    const general=[health,alpha,world].filter(Number.isFinite).reduce((a,b)=>a+b,0)/Math.max(1,[health,alpha,world].filter(Number.isFinite).length);
    const learn=world==null?null:clamp(world*.55+(health||0)*.45);
    const decision=alpha==null?null:clamp(alpha*.75+(health||0)*.25);
    const energy=health==null?'BEKLE':health>=85?'YÜKSEK':health>=65?'ORTA':health>=40?'DÜŞÜK':'KRİTİK';
    return {general,health,learn,decision,energy,alpha,world};
  }
  function style(){if($('#baclean-css'))return;const s=document.createElement('style');s.id='baclean-css';s.textContent=`
#brianAnatomyLive{position:fixed;inset:0;display:none;z-index:12000;background:#020711;color:#eaffff;overflow:auto;-webkit-overflow-scrolling:touch}
#brianAnatomyLive.bal-show{display:block}.bac{min-height:100dvh;background:radial-gradient(circle at 50% 10%,#07344f 0,#020914 34%,#01050d 72%)}
.bac-head{position:sticky;top:0;z-index:20;display:flex;justify-content:space-between;align-items:center;padding:calc(12px + env(safe-area-inset-top)) 16px 12px;background:#020914ef;border-bottom:1px solid #35ccff33;backdrop-filter:blur(16px)}
.bac-title b{display:block;font-size:18px}.bac-title span{font-size:10px;color:#55dfff;letter-spacing:.14em}.bac-act{display:flex;align-items:center;gap:8px}.bac-live{font-weight:900;font-size:12px;color:#64f1b5}.bac-btn{width:40px;height:40px;border-radius:12px;border:1px solid #45d9ff55;background:#08253a;color:#eaffff;font-size:21px}
.bac-kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;padding:14px}.bac-kpi{border:1px solid #35ccff33;background:linear-gradient(145deg,#061f33,#020d1a);border-radius:16px;padding:12px}.bac-kpi span{display:block;font-size:10px;color:#79aabc;letter-spacing:.08em}.bac-kpi b{display:block;font-size:22px;margin-top:7px}.bac-kpi.energy b{color:#65f2b5}
.bac-hero{margin:0 14px 14px;border:1px solid #35ccff33;border-radius:24px;overflow:hidden;min-height:580px;position:relative;background:#020b15;display:grid;place-items:center}.bac-hero img{width:100%;height:100%;object-fit:cover;object-position:center 42%;position:absolute;inset:0}.bac-veil{position:absolute;inset:0;background:linear-gradient(180deg,transparent 50%,rgba(1,6,12,.68));pointer-events:none}.bac-tag{position:absolute;left:18px;bottom:16px;font-size:12px;font-weight:900;letter-spacing:.12em}.bac-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:0 14px 28px}.bac-card{border:1px solid #35ccff28;background:#041725;border-radius:16px;padding:13px}.bac-card span{color:#8bb0bf;font-size:10px}.bac-card b{display:block;font-size:20px;margin-top:5px}
@media(max-width:800px){.bac-kpis{grid-template-columns:1fr 1fr}.bac-kpi.energy{grid-column:1/-1}.bac-hero{min-height:56dvh;max-height:660px}.bac-grid{grid-template-columns:1fr 1fr}.bac-head{padding-left:14px;padding-right:14px}.bac-title b{font-size:17px}}
` ;document.head.appendChild(s)}
  function ensure(){
    style();
    document.querySelectorAll('#developmentLaunch').forEach(x=>x.remove());
    if(!$('#developmentLaunchClean')){
      const anchor=$('#autonomyRow')||$('#chatPanel')?.closest('section')||$('.metrics');
      if(anchor?.parentNode){const q=document.createElement('section');q.id='developmentLaunchClean';q.className='card card-pad';q.innerHTML='<div style="display:flex;align-items:center;justify-content:space-between;gap:12px"><div><b>🧬 Brian Gelişim Anatomisi</b><div style="color:#7f9bab;font-size:10px;margin-top:4px">Canlı heartbeat verisiyle, ağır analiz çağrısı olmadan.</div></div><button class="btn primary" id="balOpenClean">CANLI ANATOMİ →</button></div>';anchor.parentNode.insertBefore(q,anchor)}
    }
    document.querySelectorAll('#brianAnatomyLive').forEach(x=>x.remove());
    const m=document.createElement('div');m.id='brianAnatomyLive';m.innerHTML=`<div class="bac"><header class="bac-head"><div class="bac-title"><span>BRIAN v2</span><b>Gelişim Anatomisi</b></div><div class="bac-act"><div class="bac-live" id="bacLive">CANLI</div><button class="bac-btn" id="bacRefresh">↻</button><button class="bac-btn" id="bacClose">×</button></div></header><div class="bac-kpis"><div class="bac-kpi"><span>GENEL GELİŞİM</span><b data-k="general">—</b></div><div class="bac-kpi"><span>SİSTEM SAĞLIĞI</span><b data-k="health">—</b></div><div class="bac-kpi"><span>ÖĞRENME</span><b data-k="learn">—</b></div><div class="bac-kpi"><span>KARAR KALİTESİ</span><b data-k="decision">—</b></div><div class="bac-kpi energy"><span>ENERJİ / CANLILIK</span><b data-k="energy">—</b></div></div><div class="bac-hero"><img src="/brian-anatomy-mobile.jpg?v=20260913-clean1" alt="Brian gelişim anatomisi"><div class="bac-veil"></div><div class="bac-tag">BRIAN • GELİŞİM ANATOMİSİ</div></div><div class="bac-grid"><div class="bac-card"><span>BEYİN / ALPHA</span><b data-k="alpha">—</b></div><div class="bac-card"><span>DÜNYA / HAFIZA</span><b data-k="world">—</b></div><div class="bac-card"><span>COLLECTOR SAĞLIĞI</span><b data-k="health2">—</b></div><div class="bac-card"><span>SON HEARTBEAT</span><b data-k="time">—</b></div></div></div>`;document.body.appendChild(m);
    $('#balOpenClean').onclick=()=>{m.classList.add('bal-show');document.body.style.overflow='hidden';render()};
    $('#bacClose').onclick=()=>{m.classList.remove('bal-show');document.body.style.overflow=''};
    $('#bacRefresh').onclick=()=>{if(typeof window.refresh==='function')Promise.resolve(window.refresh()).finally(()=>setTimeout(render,250));else render()};
  }
  function render(){const h=hb(),x=metrics();const set=(k,v,raw=false)=>{const e=$(`[data-k="${k}"]`);if(e)e.textContent=raw?String(v??'—'):(v==null?'—':pct(v))};set('general',x.general);set('health',x.health);set('learn',x.learn);set('decision',x.decision);set('energy',x.energy,true);set('alpha',x.alpha);set('world',x.world);set('health2',x.health);set('time',h?new Date().toLocaleTimeString('tr-TR',{hour:'2-digit',minute:'2-digit',second:'2-digit'}):'BEKLENİYOR',true);const live=$('#bacLive');if(live){live.textContent=h?'CANLI':'HEARTBEAT BEKLENİYOR';live.style.color=h?'#64f1b5':'#ffc55a'}}
  function boot(){ensure();render();setInterval(()=>{if(document.visibilityState==='visible')render()},5000)}
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',boot,{once:true});else boot();
})();
