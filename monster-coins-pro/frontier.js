'use strict';

const ROOT='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1';
const CONTROL=`${ROOT}/brian-control-center`;
const KEY_STORAGE='mcp-dashboard-key-v1';
const ENDPOINTS={
  world:`${ROOT}/brian-world-status`,
  evolution:`${ROOT}/brian-evolution-status`,
  treasury:`${ROOT}/brian-evolution-treasury-status`,
  ocean:`${ROOT}/brian-evolution-ocean-status`,
  lab:`${ROOT}/brian-evolution-lab-status`,
  alphaIntel:`${ROOT}/brian-evolution-alpha-intelligence-status`,
  news:`${ROOT}/brian-frontier-news`,
};
const $=id=>document.getElementById(id);
const $$=sel=>[...document.querySelectorAll(sel)];
const state={control:null,world:null,evolution:null,treasury:null,ocean:null,lab:null,alphaIntel:null,news:null,errors:{},lastSync:null};

function key(){return(localStorage.getItem(KEY_STORAGE)||'').trim();}
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function finite(v){const n=Number(v);return Number.isFinite(n)?n:null;}
function money(v){const n=finite(v);return n==null?'—':new Intl.NumberFormat('tr-TR',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(n);}
function pct01(v){const n=finite(v);return n==null?'—':`${Math.round(n*100)}%`;}
function clock(v){if(!v)return'—';try{return new Intl.DateTimeFormat('tr-TR',{timeZone:'Europe/Berlin',hour:'2-digit',minute:'2-digit'}).format(new Date(v));}catch{return'—';}}
function rel(v){const t=Date.parse(String(v||''));if(!Number.isFinite(t))return'—';const s=Math.max(0,(Date.now()-t)/1000);if(s<60)return`${Math.round(s)} sn`;if(s<3600)return`${Math.round(s/60)} dk`;return`${Math.round(s/3600)} sa`;}
function action(v){return({OPEN_LONG:'LONG AÇ',OPEN_SHORT:'SHORT AÇ',WAIT:'BEKLE',VETO:'VETO',BUY:'AL',SELL:'SAT',HOLD:'TUT'})[String(v)]||String(v||'—').replaceAll('_',' ');}
function tone(status){const s=String(status||'').toUpperCase();if(['ONLINE','SUCCESS','RUNNING','HEALTHY','ACTIVE'].includes(s))return'ok';if(['DEGRADED','STALE','WAITING_FOR_FIRST_CYCLE','IDLE','NO_DATA','PAUSED','SKIPPED'].some(x=>s.includes(x)))return'warn';return'bad';}
function setDot(id,t){const el=$(id);if(el)el.className=`dot ${t}`;}
function setCable(id,t){const el=$(id);if(el)el.className=`cable ${t==='bad'?'bad':t==='warn'?'warn':id==='cableAlpha'||id==='cableTreasury'?'ok':'info'}`;}
function toast(msg){const el=$('toast');if(!el)return;el.textContent=String(msg);el.style.display='block';clearTimeout(toast.t);toast.t=setTimeout(()=>el.style.display='none',2800);}
function consumeSetupKey(){const h=location.hash||'';const m=h.match(/(?:^#|&)key=([^&]+)/);if(!m)return;try{const value=decodeURIComponent(m[1]);if(value)localStorage.setItem(KEY_STORAGE,value);}catch{}history.replaceState(null,'',location.pathname+location.search);}
function unlock(show){$('unlock')?.classList.toggle('show',show);}
async function post(url,body={}){const k=key();if(!k)throw new Error('UNAUTHORIZED_DASHBOARD');const r=await fetch(url,{method:'POST',headers:{'content-type':'application/json','x-brian-dashboard-key':k},body:JSON.stringify(body),cache:'no-store'});let data={};try{data=await r.json();}catch{}if(!r.ok)throw new Error(data.error||data.status||`HTTP ${r.status}`);return data;}
async function control(actionName='status',body={}){return post(CONTROL,{action:actionName,...body});}
async function safe(name,promise){try{state[name]=await promise;delete state.errors[name];return state[name];}catch(error){state.errors[name]=String(error?.message||error);state[name]=null;return null;}}

const narrativeTR={
  'AI compute demand':'Yapay zekâ hesaplama talebi','Semiconductor supply chain':'Yarı iletken tedarik zinciri','Monetary policy':'Para politikası','Inflation':'Enflasyon','Geopolitical risk':'Jeopolitik risk','Energy supply':'Enerji arzı','Crypto regulation':'Kripto düzenlemeleri','ETF flows':'ETF para akışları','Stablecoin payments':'Stablecoin ödemeleri','Cybersecurity / exploit risk':'Siber güvenlik / saldırı riski','Product launch cycle':'Ürün / teknoloji lansmanı','Corporate earnings':'Şirket finansal sonuçları','Token listing / unlock':'Token listeleme / kilit açılımı'
};
function trLabel(v){return narrativeTR[String(v)]||String(v||'Bilinmeyen gelişme');}
function statusText(t){return t==='ok'?'Çevrim içi':t==='warn'?'Kısmi / bekliyor':'Bağlantı sorunu';}

async function refresh(){
  if(!key()){unlock(true);return;}
  $('syncText').textContent='Canlı servisler okunuyor…';
  const jobs=[
    safe('control',control('status')),
    safe('world',post(ENDPOINTS.world)),
    safe('evolution',post(ENDPOINTS.evolution)),
    safe('treasury',post(ENDPOINTS.treasury)),
    safe('ocean',post(ENDPOINTS.ocean)),
    safe('lab',post(ENDPOINTS.lab)),
    safe('alphaIntel',post(ENDPOINTS.alphaIntel)),
    safe('news',post(ENDPOINTS.news)),
  ];
  await Promise.all(jobs);
  state.lastSync=new Date();render();
}

function moduleRows(){
  const c=state.control||{},bg=c.system?.background||{},a=c.alpha_v2||{};
  return [
    {key:'world',name:'Dünya Gezgini',icon:'🌍',status:state.world?.status||'ERROR',meta:state.world?`${state.world.summary?.unique_entities??0} varlık · ${state.world.summary?.narratives??0} anlatı`:state.errors.world},
    {key:'alpha',name:'ALPHA',icon:'α',status:a.online?'ONLINE':a.status||'ERROR',meta:a.online?`Son karar ${Math.round(Number(a.decision_age_seconds||0))} sn önce`:state.errors.control||'ALPHA heartbeat yok'},
    {key:'treasury',name:'Hazine / Portföy',icon:'◉',status:state.treasury?.status||'ERROR',meta:state.treasury?`${money(state.treasury.summary?.equity_usd)} equity · ${state.treasury.summary?.open_positions??0} pozisyon`:state.errors.treasury},
    {key:'research',name:'Araştırma / Evolution',icon:'⚗',status:state.evolution?.status||'ERROR',meta:state.evolution?`${state.evolution.gaps?.length??0} capability açığı · ${state.evolution.journal?.length??0} günlük kaydı`:state.errors.evolution},
    {key:'ocean',name:'Okyanus',icon:'≈',status:state.ocean?.status||'ERROR',meta:state.ocean?.active_run?`Aktif ${state.ocean.active_run.duration_hours||''}s sınav`:state.ocean?'Aktif sınav yok':state.errors.ocean},
    {key:'lab',name:'Deney Laboratuvarı',icon:'🧪',status:state.lab?.status||'ERROR',meta:state.lab?`${state.lab.summary?.experiments??state.lab.experiments?.length??0} deney`:state.errors.lab},
  ];
}
function renderModules(){
  const rows=moduleRows();let good=0;
  $('moduleList').innerHTML=rows.map(r=>{const t=tone(r.status);if(t==='ok')good++;return`<div class="module"><div class="module-icon">${r.icon}</div><div class="module-main"><div class="module-name">${esc(r.name)}</div><div class="module-meta">${esc(r.meta||statusText(t))}</div></div><span class="dot ${t}"></span><div class="mini-bars"><i></i><i></i><i></i><i></i></div></div>`}).join('');
  $('moduleCount').textContent=`${good}/${rows.length} sağlıklı`;
  const map={world:['dotWorld','textWorld','cableWorld'],alpha:['dotAlpha','textAlpha','cableAlpha'],treasury:['dotTreasury','textTreasury','cableTreasury'],research:['dotResearch','textResearch','cableResearch'],ocean:['dotOcean','textOcean','cableOcean']};
  for(const r of rows){if(!map[r.key])continue;const t=tone(r.status),[d,txt,cable]=map[r.key];setDot(d,t);$(txt).textContent=statusText(t);setCable(cable,t);}
  const behaviorEvidence=Boolean((state.world?.narratives||[]).length||state.news?.items?.length);const bt=behaviorEvidence?'ok':'warn';setDot('dotBehavior',bt);$('textBehavior').textContent=behaviorEvidence?'Davranış kanıtı izleniyor':'Kanıt bekleniyor';setCable('cableBehavior',bt);
  return{good,total:rows.length};
}

function newsItems(){
  if(Array.isArray(state.news?.items)&&state.news.items.length)return state.news.items.slice(0,9).map(x=>({urgency:x.urgency||'MEDIUM',title:x.title_tr||'Brian için önemli gelişme',summary:x.summary_tr||'',time:x.observed_at,source:x.source_id,asset:x.primary_asset,original:x.original_claim}));
  const out=[];
  for(const n of(state.world?.narratives||[]).slice(0,5))out.push({urgency:Number(n.strength)>=.75?'HIGH':'MEDIUM',title:trLabel(n.label),summary:`Brian bu anlatıyı ${Math.round(Number(n.strength||0)*100)}% güç ve ${n.breadth??0} bağımsız olay genişliğiyle izliyor.`,time:n.observed_at,source:'World Brain'});
  for(const e of(state.world?.upcoming_events||[]).slice(0,4))out.push({urgency:Number(e.confidence)>=.8?'HIGH':'MEDIUM',title:'Yaklaşan kritik olay',summary:String(e.title||e.event_kind||'Planlı olay'),time:e.first_observed_at,source:'Takvim'});
  return out;
}
function renderNews(){
  const items=newsItems();
  $('newsBadge').textContent=items.length?`${items.length} BRIAN FİLTRESİ`:'AKIŞ YOK';$('newsBadge').className=`badge ${items.length?'ok':'warn'}`;
  $('criticalNews').innerHTML=items.length?items.map(n=>`<div class="news"><div class="news-top"><div class="news-title">${esc(n.title)}</div><div class="severity ${String(n.urgency).toLowerCase()}">${esc(n.urgency==='CRITICAL'?'KRİTİK':n.urgency==='HIGH'?'YÜKSEK':'ORTA')}</div></div><div class="news-meta">${esc(n.summary)}${n.asset?` · ${esc(n.asset)}`:''}<br>${clock(n.time)} · ${esc(n.source||'Brian')}</div>${n.original?`<details class="news-meta"><summary>Kaynak metni</summary>${esc(n.original)}</details>`:''}</div>`).join(''):'<div class="news"><div class="news-title">Brian için kritik gelişme akışı henüz veri üretmedi.</div><div class="news-meta">World Brain / Frontier News çevrim içi olduğunda burada yalnız piyasa açısından anlamlı gelişmeler görünür.</div></div>';
  const ticker=items.slice(0,5);const html=ticker.length?ticker.map(n=>`<div class="ticker-item"><span class="dot ${n.urgency==='CRITICAL'?'bad':n.urgency==='HIGH'?'warn':'ok'}"></span><b>${n.urgency==='CRITICAL'?'KRİTİK':'Brian'}:</b> ${esc(n.title)}</div>`).join(''):'<div class="ticker-item"><span class="dot warn"></span><b>Brian:</b> kritik gelişme akışı bekleniyor…</div>';
  $('tickerTrack').innerHTML=html+html;
}

function renderAlpha(){
  const a=state.control?.alpha_v2||{},ds=a.decisions||[],latest=ds[0]||null;
  $('alphaValue').textContent=latest?action(latest.action):(a.online?'CANLI':'—');$('alphaMeta').textContent=latest?`${String(latest.asset_id||'').replace('crypto:','')} · ${clock(latest.observed_at)}`:'Karar bekleniyor';
  $('alphaBadge').textContent=a.online?'CANLI':'BAĞLANTI';$('alphaBadge').className=`badge ${a.online?'ok':'bad'}`;
  $('alphaFeed').innerHTML=ds.slice(0,6).map(d=>`<div class="module"><div class="module-icon">α</div><div class="module-main"><div class="module-name">${esc(String(d.asset_id||'').replace('crypto:',''))} · ${esc(action(d.action))}</div><div class="module-meta">${clock(d.observed_at)} · kanıt ${Number(d.evidence_score||0).toFixed(2)} · maliyet ${Number(d.estimated_round_trip_cost_bps||0).toFixed(1)} bps</div></div><span class="dot ${['OPEN_LONG','OPEN_SHORT'].includes(d.action)?'ok':d.action==='VETO'?'bad':'warn'}"></span></div>`).join('')||'<div class="module"><span class="dot warn"></span><div class="module-main"><div class="module-name">ALPHA karar akışı bekleniyor</div></div></div>';
}
function renderTreasury(){const s=state.treasury?.summary;$('treasuryValue').textContent=s?money(s.equity_usd):'—';$('treasuryMeta').textContent=s?`${money(s.cash_usd)} nakit · ${pct01(s.deployment_pct)} kullanım`:'Treasury worker bekleniyor';}
function renderWorld(){const s=state.world?.summary;$('worldValue').textContent=s?String(s.unique_entities??0):'—';$('worldMeta').textContent=s?`${s.narratives??0} anlatı · ${s.asset_impact_candidates??0} etki adayı`:'World Brain bekleniyor';const ns=(state.world?.narratives||[]).slice(0,4);$('worldMini').innerHTML=ns.map(n=>`<div class="module"><span class="dot ${Number(n.strength)>=.7?'warn':'ok'}"></span><div class="module-main"><div class="module-name">${esc(trLabel(n.label))}</div><div class="module-meta">Güç ${Math.round(Number(n.strength||0)*100)}% · genişlik ${n.breadth??0}</div></div></div>`).join('')||'<div class="module"><span class="dot warn"></span><div class="module-main"><div class="module-name">Dünya verisi bekleniyor</div></div></div>';}
function renderResearch(){const e=state.evolution||{},journal=e.journal||[],gaps=e.gaps||[];$('researchBadge').textContent=e.status==='ONLINE'?'CANLI':'BEKLİYOR';$('researchBadge').className=`badge ${e.status==='ONLINE'?'ok':'warn'}`;const rows=[...journal.slice(0,3).map(j=>({title:j.title||j.event_type,meta:j.summary||`${j.stage||''} · ${clock(j.occurred_at)}`})),...gaps.slice(0,3).map(g=>({title:`Eksik yetenek: ${g.capabilityId||g.capability_id||g.domain}`,meta:g.reason||g.suggestedAction||g.suggested_action}))];$('researchFeed').innerHTML=rows.slice(0,6).map(r=>`<div class="module"><div class="module-icon">⚗</div><div class="module-main"><div class="module-name">${esc(r.title||'Araştırma')}</div><div class="module-meta">${esc(r.meta||'')}</div></div></div>`).join('')||'<div class="module"><span class="dot warn"></span><div class="module-main"><div class="module-name">Evolution araştırma günlüğü bekleniyor</div></div></div>';}
function renderBehavior(){const n=(state.world?.narratives||[])[0],items=newsItems();if(n||items.length){$('behaviorText').textContent=`Brian ${n?trLabel(n.label):items[0].title} çevresindeki kitle tepkisini fiyatlama ve kaynak kanıtıyla birlikte izliyor. Metin hissi tek başına işlem yetkisi vermez.`;$('behaviorBadge').textContent='CANLI KANIT';$('behaviorBadge').className='badge ok';}else{$('behaviorText').textContent='Davranış katmanı haber değil, gerçek akış/konumlanma kanıtı bekliyor.';}}
function renderBelief(){const ai=state.alphaIntel||{},edges=ai.expected_edges||ai.edges||[],e=edges[0]||null;if(e){const p=finite(e.expected_gross_move_bps),q=finite(e.estimated_round_trip_cost_bps),x=finite(e.expected_net_edge_bps);$('beliefP').textContent=p==null?'—':`${p.toFixed(1)}b`;$('beliefQ').textContent=q==null?'—':`${q.toFixed(1)}b`;$('beliefX').textContent=x==null?'—':`${x.toFixed(1)}b`;$('beliefMeta').textContent='Bu kart Frontier P/Q kontratına geçiş sırasında mevcut expected-edge bileşenlerini gösterir; gerçek Q snapshot ledger devreye girdiğinde doğrudan piyasa-implied Q gösterilecektir.';}else{$('beliefP').textContent='—';$('beliefQ').textContent='—';$('beliefX').textContent='—';}}
function renderAlerts(){const rows=[];for(const m of moduleRows()){const t=tone(m.status);if(t!=='ok')rows.push({t,title:`${m.name}: ${statusText(t)}`,meta:m.meta||'Heartbeat yok'});}for(const n of newsItems().filter(x=>x.urgency==='CRITICAL'||x.urgency==='HIGH').slice(0,3))rows.push({t:n.urgency==='CRITICAL'?'bad':'warn',title:n.title,meta:n.summary});$('alertCount').textContent=String(rows.length);$('alertFeed').innerHTML=rows.length?rows.map(r=>`<div class="module"><span class="dot ${r.t}"></span><div class="module-main"><div class="module-name">${esc(r.title)}</div><div class="module-meta">${esc(r.meta||'')}</div></div></div>`).join(''):'<div class="module"><span class="dot ok"></span><div class="module-main"><div class="module-name">Kritik bağlantı sorunu yok</div><div class="module-meta">Canlı hatlar stabil görünüyor</div></div></div>';return rows;}
function renderMaster(summary,alerts){const bg=state.control?.system?.background||{},controlOk=Boolean(bg.browser_independent&&bg.continues_when_page_closed&&(bg.overall==='ONLINE'||bg.overall==='DEGRADED'));const ratio=summary.total?summary.good/summary.total:0;const t=!controlOk||alerts.some(x=>x.t==='bad')?'bad':ratio>=.8?'ok':'warn';setDot('masterDot',t);$('masterText').textContent=t==='ok'?'SİSTEM CANLI':t==='warn'?'KISMİ / İZLE':'SORUN VAR';$('healthValue').textContent=summary.total?`${Math.round(100*ratio)}%`:'—';$('healthMeta').textContent=`${summary.good}/${summary.total} Frontier modülü sağlıklı`;$('brainStatus').textContent=t==='ok'?'Brian düşünüyor · veri akışı canlı · SHADOW ONLY':t==='warn'?'Brian çalışıyor · bazı hatlar bekliyor/gecikmiş':'Brian uyarı veriyor · kırmızı hatları kontrol et';$('syncText').textContent=state.lastSync?`Son canlı senkron ${clock(state.lastSync.toISOString())}`:'Bağlanıyor';$('orbitFault')?.classList.toggle('hidden',!alerts.length);}
function renderMeeting(){const rows=moduleRows(),find=k=>rows.find(x=>x.key===k);const set=(id,r)=>{$(id).textContent=r?statusText(tone(r.status)):'—'};set('mWorld',find('world'));set('mAlpha',find('alpha'));set('mTreasury',find('treasury'));set('mResearch',find('research'));$('mSkeptic').textContent=newsItems().length?'Kanıt ve karşı tez arıyor':'Kanıt bekliyor';}
function render(){const summary=renderModules();renderNews();renderAlpha();renderTreasury();renderWorld();renderResearch();renderBehavior();renderBelief();const alerts=renderAlerts();renderMaster(summary,alerts);renderMeeting();}

function answer(q){const text=String(q||'').trim();if(!text)return;const lower=text.toLocaleLowerCase('tr-TR');let response='Sorunu canlı Brian durumu üzerinden değerlendirdim. ';if(lower.includes('haber')||lower.includes('geliş')){const n=newsItems()[0];response+=n?`Şu an en yüksek öncelikli gelişme: ${n.title}. ${n.summary}`:'Henüz Brian filtresinden geçen kritik bir gelişme yok.';}else if(lower.includes('sorun')||lower.includes('hata')||lower.includes('çalış')){const bad=moduleRows().filter(m=>tone(m.status)!=='ok');response+=bad.length?`Dikkat istediğim hatlar: ${bad.map(x=>`${x.name} (${statusText(tone(x.status))})`).join(', ')}.`:'Ana modüllerde kritik kopukluk görmüyorum.';}else if(lower.includes('alpha')||lower.includes('karar')){const d=state.control?.alpha_v2?.decisions?.[0];response+=d?`ALPHA'nın son kararı ${String(d.asset_id||'').replace('crypto:','')} için ${action(d.action)}. Karar ${rel(d.observed_at)} önce üretildi.`:'ALPHA'dan henüz okunabilir karar gelmedi.';}else if(lower.includes('hazine')||lower.includes('kasa')||lower.includes('para')){const s=state.treasury?.summary;response+=s?`Shadow hazine ${money(s.equity_usd)}. Nakit ${money(s.cash_usd)}, kullanılan sermaye ${pct01(s.deployment_pct)}, açık pozisyon ${s.open_positions}.`:'Hazine ilk cycle/bağlantı verisini bekliyor.';}else if(lower.includes('öğren')||lower.includes('araştır')){const g=state.evolution?.gaps?.[0],j=state.evolution?.journal?.[0];response+=j?`Son araştırma kaydı: ${j.title||j.event_type}. ${j.summary||''}`:g?`Şu an öne çıkan eksik yetenek: ${g.reason||g.capability_id||g.domain}.`:'Evolution günlüğü henüz veri üretmedi.';}else{const alerts=renderAlerts();response+=`Şu an ${moduleRows().filter(x=>tone(x.status)==='ok').length}/${moduleRows().length} ana modül sağlıklı. ${alerts.length?`${alerts.length} uyarıyı izliyorum.`:'Kritik uyarı yok.'} Bana haber, ALPHA, Hazine, araştırma veya sistem sağlığı sorabilirsin.`;}appendBubble(text,'user');setTimeout(()=>appendBubble(response,'brian'),180);}
function appendBubble(text,type){const el=document.createElement('div');el.className=`bubble ${type}`;el.textContent=text;$('chatLog').appendChild(el);$('chatLog').scrollTop=$('chatLog').scrollHeight;}
function openModal(id){$(id)?.classList.add('show');}
function closeModal(id){$(id)?.classList.remove('show');}
function bind(){
  consumeSetupKey();unlock(!key());
  $('unlockBtn')?.addEventListener('click',()=>{const v=$('unlockKey').value.trim();if(!v)return;localStorage.setItem(KEY_STORAGE,v);unlock(false);refresh();});
  $('unlockKey')?.addEventListener('keydown',e=>{if(e.key==='Enter')$('unlockBtn').click();});
  $('chatSend')?.addEventListener('click',()=>{const v=$('chatInput').value;$('chatInput').value='';answer(v);});
  $('chatInput')?.addEventListener('keydown',e=>{if(e.key==='Enter')$('chatSend').click();});
  $$('[data-q]').forEach(b=>b.addEventListener('click',()=>answer(b.dataset.q)));
  $('openMeeting')?.addEventListener('click',()=>openModal('meetingModal'));$('open3d')?.addEventListener('click',()=>openModal('view3dModal'));$('mobileMeeting')?.addEventListener('click',()=>openModal('meetingModal'));$('mobileChat')?.addEventListener('click',()=>{document.querySelector('#chatPanel')?.scrollIntoView({behavior:'smooth'});$('chatInput')?.focus();});
  $$('[data-close]').forEach(b=>b.addEventListener('click',()=>closeModal(b.dataset.close)));
  $$('.modal').forEach(m=>m.addEventListener('click',e=>{if(e.target===m)m.classList.remove('show');}));
  document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible')refresh();});
}

bind();if(key())refresh();setInterval(()=>{if(key()&&document.visibilityState==='visible')refresh();},15000);
