'use strict';

const ROOT = 'https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1';
const CONTROL = `${ROOT}/brian-control-center`;
const KEY_STORAGE = 'mcp-dashboard-key-v1';
const EP = {
  world: `${ROOT}/brian-world-status`,
  evolution: `${ROOT}/brian-evolution-status`,
  treasury: `${ROOT}/brian-evolution-treasury-status`,
  ocean: `${ROOT}/brian-evolution-ocean-status`,
  lab: `${ROOT}/brian-evolution-lab-status`,
  alphaIntel: `${ROOT}/brian-evolution-alpha-intelligence-status`,
  news: `${ROOT}/brian-frontier-news`,
};
const $ = (id) => document.getElementById(id);
const $$ = (s) => [...document.querySelectorAll(s)];
const S = { control:null, world:null, evolution:null, treasury:null, ocean:null, lab:null, alphaIntel:null, news:null, errors:{}, lastSync:null };
const TR = {
  'AI compute demand':'Yapay zekâ hesaplama talebi',
  'Semiconductor supply chain':'Yarı iletken tedarik zinciri',
  'Monetary policy':'Para politikası',
  'Inflation':'Enflasyon',
  'Geopolitical risk':'Jeopolitik risk',
  'Energy supply':'Enerji arzı',
  'Crypto regulation':'Kripto düzenlemeleri',
  'ETF flows':'ETF para akışları',
  'Cybersecurity / exploit risk':'Siber güvenlik / saldırı riski',
  'Product launch cycle':'Ürün / teknoloji lansmanı',
  'Corporate earnings':'Şirket finansal sonuçları',
  'Token listing / unlock':'Token listeleme / kilit açılımı'
};
function key(){ return (localStorage.getItem(KEY_STORAGE)||'').trim(); }
function esc(v){ return String(v??'').replace(/[&<>"']/g,(c)=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function num(v){ const n=Number(v); return Number.isFinite(n)?n:null; }
function money(v){ const n=num(v); return n==null?'—':new Intl.NumberFormat('tr-TR',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(n); }
function pct(v){ const n=num(v); return n==null?'—':`${Math.round(n*100)}%`; }
function clock(v){ if(!v)return '—'; try{return new Intl.DateTimeFormat('tr-TR',{timeZone:'Europe/Berlin',hour:'2-digit',minute:'2-digit'}).format(new Date(v));}catch{return '—';} }
function age(v){ const t=Date.parse(String(v||'')); if(!Number.isFinite(t))return '—'; const s=Math.max(0,(Date.now()-t)/1000); return s<60?`${Math.round(s)} sn`:s<3600?`${Math.round(s/60)} dk`:`${Math.round(s/3600)} sa`; }
function trLabel(v){ return TR[String(v)]||String(v||'Bilinmeyen gelişme'); }
function act(v){ return ({OPEN_LONG:'LONG AÇ',OPEN_SHORT:'SHORT AÇ',WAIT:'BEKLE',VETO:'VETO',BUY:'AL',SELL:'SAT',HOLD:'TUT'})[String(v)]||String(v||'—').replaceAll('_',' '); }
function tone(v){ const s=String(v||'').toUpperCase(); if(['ONLINE','SUCCESS','RUNNING','HEALTHY','ACTIVE'].includes(s))return 'ok'; if(['DEGRADED','STALE','WAITING_FOR_FIRST_CYCLE','IDLE','NO_DATA','PAUSED','SKIPPED'].some((x)=>s.includes(x)))return 'warn'; return 'bad'; }
function statusText(t){ return t==='ok'?'Çevrim içi':t==='warn'?'Kısmi / bekliyor':'Bağlantı sorunu'; }
function setDot(id,t){ const el=$(id); if(el)el.className=`dot ${t}`; }
function setCable(id,t){ const el=$(id); if(!el)return; const healthy=id==='cableAlpha'||id==='cableTreasury'?'ok':'info'; el.className=`cable ${t==='bad'?'bad':t==='warn'?'warn':healthy}`; }
function unlock(show){ $('unlock')?.classList.toggle('show',show); }
async function post(url,body={}){ const k=key(); if(!k)throw new Error('UNAUTHORIZED_DASHBOARD'); const r=await fetch(url,{method:'POST',headers:{'content-type':'application/json','x-brian-dashboard-key':k},body:JSON.stringify(body),cache:'no-store'}); let data={}; try{data=await r.json();}catch{} if(!r.ok)throw new Error(data.error||data.status||`HTTP ${r.status}`); return data; }
async function safe(name,p){ try{S[name]=await p;delete S.errors[name];}catch(e){S[name]=null;S.errors[name]=String(e?.message||e);} }
async function refresh(){
  if(!key()){unlock(true);return;}
  if($('syncText'))$('syncText').textContent='Canlı servisler okunuyor…';
  await Promise.all([
    safe('control',post(CONTROL,{action:'status'})),safe('world',post(EP.world)),safe('evolution',post(EP.evolution)),
    safe('treasury',post(EP.treasury)),safe('ocean',post(EP.ocean)),safe('lab',post(EP.lab)),
    safe('alphaIntel',post(EP.alphaIntel)),safe('news',post(EP.news))
  ]);
  S.lastSync=new Date(); render();
}
function modules(){
  const a=S.control?.alpha_v2||{};
  return [
    {key:'world',name:'Dünya Gezgini',icon:'🌍',status:S.world?.status||'ERROR',meta:S.world?`${S.world.summary?.unique_entities??0} varlık · ${S.world.summary?.narratives??0} anlatı`:S.errors.world},
    {key:'alpha',name:'ALPHA',icon:'α',status:a.online?'ONLINE':a.status||'ERROR',meta:a.online?`Son karar ${Math.round(Number(a.decision_age_seconds||0))} sn önce`:S.errors.control||'ALPHA heartbeat yok'},
    {key:'treasury',name:'Hazine / Portföy',icon:'◉',status:S.treasury?.status||'ERROR',meta:S.treasury?`${money(S.treasury.summary?.equity_usd)} · ${S.treasury.summary?.open_positions??0} pozisyon`:S.errors.treasury},
    {key:'research',name:'Araştırma / Evolution',icon:'⚗',status:S.evolution?.status||'ERROR',meta:S.evolution?`${S.evolution.gaps?.length??0} açık · ${S.evolution.journal?.length??0} günlük kaydı`:S.errors.evolution},
    {key:'ocean',name:'Okyanus',icon:'≈',status:S.ocean?.status||'ERROR',meta:S.ocean?.active_run?'Aktif Ocean sınavı':S.ocean?'Aktif sınav yok':S.errors.ocean},
    {key:'lab',name:'Deney Laboratuvarı',icon:'🧪',status:S.lab?.status||'ERROR',meta:S.lab?`${S.lab.summary?.experiments??S.lab.experiments?.length??0} deney`:S.errors.lab}
  ];
}
function renderModules(){
  const rows=modules(); let good=0;
  $('moduleList').innerHTML=rows.map((r)=>{const t=tone(r.status);if(t==='ok')good++;return `<div class="module"><div class="module-icon">${r.icon}</div><div class="module-main"><div class="module-name">${esc(r.name)}</div><div class="module-meta">${esc(r.meta||statusText(t))}</div></div><span class="dot ${t}"></span><div class="mini-bars"><i></i><i></i><i></i><i></i></div></div>`;}).join('');
  $('moduleCount').textContent=`${good}/${rows.length} sağlıklı`;
  const map={world:['dotWorld','textWorld','cableWorld'],alpha:['dotAlpha','textAlpha','cableAlpha'],treasury:['dotTreasury','textTreasury','cableTreasury'],research:['dotResearch','textResearch','cableResearch'],ocean:['dotOcean','textOcean','cableOcean']};
  rows.forEach((r)=>{if(!map[r.key])return;const t=tone(r.status);const [d,x,c]=map[r.key];setDot(d,t);$(x).textContent=statusText(t);setCable(c,t);});
  const b=(S.world?.narratives||[]).length||(S.news?.items||[]).length?'ok':'warn';setDot('dotBehavior',b);$('textBehavior').textContent=b==='ok'?'Davranış kanıtı izleniyor':'Kanıt bekleniyor';setCable('cableBehavior',b);
  return {good,total:rows.length};
}
function news(){
  if(Array.isArray(S.news?.items)&&S.news.items.length)return S.news.items.slice(0,9).map((x)=>({urgency:x.urgency||'MEDIUM',title:x.title_tr||'Brian için önemli gelişme',summary:x.summary_tr||'',time:x.observed_at,source:x.source_id,asset:x.primary_asset,original:x.original_claim}));
  const out=[];
  (S.world?.narratives||[]).slice(0,5).forEach((n)=>out.push({urgency:Number(n.strength)>=.75?'HIGH':'MEDIUM',title:trLabel(n.label),summary:`Brian bu anlatıyı ${Math.round(Number(n.strength||0)*100)}% güçle izliyor.`,time:n.observed_at,source:'World Brain'}));
  (S.world?.upcoming_events||[]).slice(0,4).forEach((e)=>out.push({urgency:Number(e.confidence)>=.8?'HIGH':'MEDIUM',title:'Yaklaşan kritik olay',summary:String(e.title||e.event_kind||'Planlı olay'),time:e.first_observed_at,source:'Takvim'}));
  return out;
}
function renderNews(){
  const items=news(); $('newsBadge').textContent=items.length?`${items.length} BRIAN FİLTRESİ`:'AKIŞ YOK';$('newsBadge').className=`badge ${items.length?'ok':'warn'}`;
  $('criticalNews').innerHTML=items.length?items.map((n)=>`<div class="news"><div class="news-top"><div class="news-title">${esc(n.title)}</div><div class="severity ${String(n.urgency).toLowerCase()}">${n.urgency==='CRITICAL'?'KRİTİK':n.urgency==='HIGH'?'YÜKSEK':'ORTA'}</div></div><div class="news-meta">${esc(n.summary)}${n.asset?` · ${esc(n.asset)}`:''}<br>${clock(n.time)} · ${esc(n.source||'Brian')}</div>${n.original?`<details class="news-meta"><summary>Kaynak metni</summary>${esc(n.original)}</details>`:''}</div>`).join(''):'<div class="news"><div class="news-title">Brian için kritik gelişme akışı henüz veri üretmedi.</div><div class="news-meta">World Brain çevrim içi olduğunda piyasa açısından anlamlı gelişmeler burada görünür.</div></div>';
  const top=items.slice(0,5);const h=top.length?top.map((n)=>`<div class="ticker-item"><span class="dot ${n.urgency==='CRITICAL'?'bad':n.urgency==='HIGH'?'warn':'ok'}"></span><b>${n.urgency==='CRITICAL'?'KRİTİK':'Brian'}:</b> ${esc(n.title)}</div>`).join(''):'<div class="ticker-item"><span class="dot warn"></span><b>Brian:</b> kritik gelişme akışı bekleniyor…</div>'; $('tickerTrack').innerHTML=h+h;
}
function renderAlpha(){
  const a=S.control?.alpha_v2||{},ds=a.decisions||[],d=ds[0]; $('alphaValue').textContent=d?act(d.action):(a.online?'CANLI':'—'); $('alphaMeta').textContent=d?`${String(d.asset_id||'').replace('crypto:','')} · ${clock(d.observed_at)}`:'Karar bekleniyor'; $('alphaBadge').textContent=a.online?'CANLI':'BAĞLANTI'; $('alphaBadge').className=`badge ${a.online?'ok':'bad'}`;
  $('alphaFeed').innerHTML=ds.slice(0,6).map((x)=>`<div class="module"><div class="module-icon">α</div><div class="module-main"><div class="module-name">${esc(String(x.asset_id||'').replace('crypto:',''))} · ${esc(act(x.action))}</div><div class="module-meta">${clock(x.observed_at)} · kanıt ${Number(x.evidence_score||0).toFixed(2)} · maliyet ${Number(x.estimated_round_trip_cost_bps||0).toFixed(1)} bps</div></div><span class="dot ${['OPEN_LONG','OPEN_SHORT'].includes(x.action)?'ok':x.action==='VETO'?'bad':'warn'}"></span></div>`).join('')||'<div class="module"><span class="dot warn"></span><div class="module-main"><div class="module-name">ALPHA karar akışı bekleniyor</div></div></div>';
}
function renderWorld(){ const s=S.world?.summary;$('worldValue').textContent=s?String(s.unique_entities??0):'—';$('worldMeta').textContent=s?`${s.narratives??0} anlatı · ${s.asset_impact_candidates??0} etki adayı`:'World Brain bekleniyor'; const rows=(S.world?.narratives||[]).slice(0,4);$('worldMini').innerHTML=rows.map((n)=>`<div class="module"><span class="dot ${Number(n.strength)>=.7?'warn':'ok'}"></span><div class="module-main"><div class="module-name">${esc(trLabel(n.label))}</div><div class="module-meta">Güç ${Math.round(Number(n.strength||0)*100)}% · genişlik ${n.breadth??0}</div></div></div>`).join('')||'<div class="module"><span class="dot warn"></span><div class="module-main"><div class="module-name">Dünya verisi bekleniyor</div></div></div>'; }
function renderTreasury(){ const s=S.treasury?.summary;$('treasuryValue').textContent=s?money(s.equity_usd):'—';$('treasuryMeta').textContent=s?`${money(s.cash_usd)} nakit · ${pct(s.deployment_pct)} kullanım`:'Treasury worker bekleniyor'; }
function renderResearch(){ const e=S.evolution||{},j=e.journal||[],g=e.gaps||[];$('researchBadge').textContent=e.status==='ONLINE'?'CANLI':'BEKLİYOR';$('researchBadge').className=`badge ${e.status==='ONLINE'?'ok':'warn'}`; const rows=[...j.slice(0,3).map((x)=>({title:x.title||x.event_type,meta:x.summary||`${x.stage||''} · ${clock(x.occurred_at)}`})),...g.slice(0,3).map((x)=>({title:`Eksik yetenek: ${x.capabilityId||x.capability_id||x.domain}`,meta:x.reason||x.suggestedAction||x.suggested_action}))];$('researchFeed').innerHTML=rows.slice(0,6).map((x)=>`<div class="module"><div class="module-icon">⚗</div><div class="module-main"><div class="module-name">${esc(x.title||'Araştırma')}</div><div class="module-meta">${esc(x.meta||'')}</div></div></div>`).join('')||'<div class="module"><span class="dot warn"></span><div class="module-main"><div class="module-name">Evolution araştırma günlüğü bekleniyor</div></div></div>'; }
function renderBehavior(){ const n=(S.world?.narratives||[])[0],items=news(); if(n||items.length){$('behaviorText').textContent=`Brian ${n?trLabel(n.label):items[0].title} çevresindeki kitle tepkisini fiyatlama ve kaynak kanıtıyla birlikte izliyor. Metin hissi tek başına işlem yetkisi vermez.`;$('behaviorBadge').textContent='CANLI KANIT';$('behaviorBadge').className='badge ok';} }
function renderBelief(){ const edges=S.alphaIntel?.expected_edges||S.alphaIntel?.edges||[],e=edges[0];if(!e){$('beliefP').textContent='—';$('beliefQ').textContent='—';$('beliefX').textContent='—';return;} const p=num(e.expected_gross_move_bps),q=num(e.estimated_round_trip_cost_bps),x=num(e.expected_net_edge_bps);$('beliefP').textContent=p==null?'—':`${p.toFixed(1)}b`;$('beliefQ').textContent=q==null?'—':`${q.toFixed(1)}b`;$('beliefX').textContent=x==null?'—':`${x.toFixed(1)}b`;$('beliefMeta').textContent='Frontier Q ledger devreye girene kadar mevcut expected-edge bileşenleri gösteriliyor.'; }
function alerts(){ const rows=[];modules().forEach((m)=>{const t=tone(m.status);if(t!=='ok')rows.push({t,title:`${m.name}: ${statusText(t)}`,meta:m.meta||'Heartbeat yok'});});news().filter((x)=>x.urgency==='CRITICAL'||x.urgency==='HIGH').slice(0,3).forEach((n)=>rows.push({t:n.urgency==='CRITICAL'?'bad':'warn',title:n.title,meta:n.summary}));$('alertCount').textContent=String(rows.length);$('alertFeed').innerHTML=rows.length?rows.map((r)=>`<div class="module"><span class="dot ${r.t}"></span><div class="module-main"><div class="module-name">${esc(r.title)}</div><div class="module-meta">${esc(r.meta||'')}</div></div></div>`).join(''):'<div class="module"><span class="dot ok"></span><div class="module-main"><div class="module-name">Kritik bağlantı sorunu yok</div><div class="module-meta">Canlı hatlar stabil görünüyor</div></div></div>';return rows; }
function renderMaster(summary,a){ const bg=S.control?.system?.background||{},controlOk=Boolean(bg.browser_independent&&bg.continues_when_page_closed&&(bg.overall==='ONLINE'||bg.overall==='DEGRADED')); const ratio=summary.total?summary.good/summary.total:0,t=!controlOk||a.some((x)=>x.t==='bad')?'bad':ratio>=.8?'ok':'warn';setDot('masterDot',t);$('masterText').textContent=t==='ok'?'SİSTEM CANLI':t==='warn'?'KISMİ / İZLE':'SORUN VAR';$('healthValue').textContent=summary.total?`${Math.round(ratio*100)}%`:'—';$('healthMeta').textContent=`${summary.good}/${summary.total} Frontier modülü sağlıklı`;$('brainStatus').textContent=t==='ok'?'Brian düşünüyor · veri akışı canlı · SHADOW ONLY':t==='warn'?'Brian çalışıyor · bazı hatlar bekliyor/gecikmiş':'Brian uyarı veriyor · kırmızı hatları kontrol et';$('syncText').textContent=S.lastSync?`Son canlı senkron ${clock(S.lastSync.toISOString())}`:'Bağlanıyor';$('orbitFault')?.classList.toggle('hidden',!a.length); }
function renderMeeting(){ const rows=modules(),f=(k)=>rows.find((x)=>x.key===k),set=(id,r)=>{$(id).textContent=r?statusText(tone(r.status)):'—';};set('mWorld',f('world'));set('mAlpha',f('alpha'));set('mTreasury',f('treasury'));set('mResearch',f('research'));$('mSkeptic').textContent=news().length?'Kanıt ve karşı tez arıyor':'Kanıt bekliyor'; }
function render(){ const m=renderModules();renderNews();renderAlpha();renderWorld();renderTreasury();renderResearch();renderBehavior();renderBelief();const a=alerts();renderMaster(m,a);renderMeeting(); }
function bubble(text,type){ const el=document.createElement('div');el.className=`bubble ${type}`;el.textContent=text;$('chatLog').appendChild(el);$('chatLog').scrollTop=$('chatLog').scrollHeight; }
function answer(q){ const text=String(q||'').trim();if(!text)return;const l=text.toLocaleLowerCase('tr-TR');let r='';if(l.includes('haber')||l.includes('geliş')){const n=news()[0];r=n?`Şu an en yüksek öncelikli gelişme: ${n.title}. ${n.summary}`:'Henüz Brian filtresinden geçen kritik bir gelişme yok.';}else if(l.includes('sorun')||l.includes('hata')||l.includes('çalış')){const bad=modules().filter((m)=>tone(m.status)!=='ok');r=bad.length?`Dikkat istediğim hatlar: ${bad.map((x)=>`${x.name} (${statusText(tone(x.status))})`).join(', ')}.`:'Ana modüllerde kritik kopukluk görmüyorum.';}else if(l.includes('alpha')||l.includes('karar')){const d=S.control?.alpha_v2?.decisions?.[0];r=d?`ALPHA'nın son kararı ${String(d.asset_id||'').replace('crypto:','')} için ${act(d.action)}. Karar ${age(d.observed_at)} önce üretildi.`:"ALPHA'dan henüz okunabilir karar gelmedi.";}else if(l.includes('hazine')||l.includes('kasa')||l.includes('para')){const s=S.treasury?.summary;r=s?`Shadow hazine ${money(s.equity_usd)}. Nakit ${money(s.cash_usd)}, kullanılan sermaye ${pct(s.deployment_pct)}, açık pozisyon ${s.open_positions}.`:'Hazine ilk cycle/bağlantı verisini bekliyor.';}else if(l.includes('öğren')||l.includes('araştır')){const j=S.evolution?.journal?.[0],g=S.evolution?.gaps?.[0];r=j?`Son araştırma kaydı: ${j.title||j.event_type}. ${j.summary||''}`:g?`Şu an öne çıkan eksik yetenek: ${g.reason||g.capability_id||g.domain}.`:'Evolution günlüğü henüz veri üretmedi.';}else{const a=modules(),good=a.filter((x)=>tone(x.status)==='ok').length;r=`Şu an ${good}/${a.length} ana modül sağlıklı. Bana haber, ALPHA, Hazine, araştırma veya sistem sağlığı sorabilirsin.`;}bubble(text,'user');setTimeout(()=>bubble(r,'brian'),150); }
function open(id){$(id)?.classList.add('show');}function close(id){$(id)?.classList.remove('show');}
function bind(){
  const h=location.hash||'',m=h.match(/(?:^#|&)key=([^&]+)/);if(m){try{localStorage.setItem(KEY_STORAGE,decodeURIComponent(m[1]));}catch{}history.replaceState(null,'',location.pathname+location.search);}
  unlock(!key());$('unlockBtn')?.addEventListener('click',()=>{const v=$('unlockKey').value.trim();if(!v)return;localStorage.setItem(KEY_STORAGE,v);unlock(false);refresh();});$('unlockKey')?.addEventListener('keydown',(e)=>{if(e.key==='Enter')$('unlockBtn').click();});
  $('chatSend')?.addEventListener('click',()=>{const v=$('chatInput').value;$('chatInput').value='';answer(v);});$('chatInput')?.addEventListener('keydown',(e)=>{if(e.key==='Enter')$('chatSend').click();});$$('[data-q]').forEach((b)=>b.addEventListener('click',()=>answer(b.dataset.q)));
  $('openMeeting')?.addEventListener('click',()=>open('meetingModal'));$('open3d')?.addEventListener('click',()=>open('view3dModal'));$('mobileMeeting')?.addEventListener('click',()=>open('meetingModal'));$('mobileChat')?.addEventListener('click',()=>{document.querySelector('#chatPanel')?.scrollIntoView({behavior:'smooth'});$('chatInput')?.focus();});$$('[data-close]').forEach((b)=>b.addEventListener('click',()=>close(b.dataset.close)));$$('.modal').forEach((m0)=>m0.addEventListener('click',(e)=>{if(e.target===m0)m0.classList.remove('show');}));document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible'&&key())refresh();});
}
bind();if(key())refresh();setInterval(()=>{if(key()&&document.visibilityState==='visible')refresh();},15000);
