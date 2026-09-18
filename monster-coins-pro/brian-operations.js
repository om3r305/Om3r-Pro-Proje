/* Main dashboard presentation only. Existing nodes/listeners and all control APIs are preserved. */
(function(){
 'use strict';
 const $=id=>document.getElementById(id),E=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 if($('brianOperations')||!$('command'))return;
 const css=document.createElement('link');css.rel='stylesheet';css.href='/brian-operations.css?v=20260918-pos1';document.head.appendChild(css);
 const shell=document.createElement('div');shell.id='brianOperations';
 shell.innerHTML=`<nav class="ops-nav" aria-label="Brian ekran bölümleri"><a href="#opsOverview">Özet</a><a href="#opsFlow">Akış</a><a href="#opsRooms">Odalar</a><a href="#opsControl">Kontrol</a></nav>
 <section id="opsOverview" class="ops-section"><header class="ops-heading"><div><span class="ops-kicker">01 / OPERASYON</span><h2>Bir bakışta Brian.</h2></div><span class="ops-mode">SHADOW · USD</span></header><p class="ops-muted" id="opsFresh">Son kayıtlar bekleniyor.</p><div class="ops-stats"><article><span>Hazine değeri</span><strong id="opsEquity">—</strong><small id="opsTreasuryTime">Kasa kaydı bekleniyor</small></article><article id="opsPositionsCard" class="ops-position-card" role="button" tabindex="0" aria-label="Açık pozisyon ayrıntılarını göster"><span>Açık pozisyon</span><strong id="opsPositions">—</strong><small id="opsPositionsMeta">Son hazine kaydına göre · Dokun: varlık, neden ve kâr/zarar</small></article><article><span>Güncel veri hatları</span><strong id="opsHealthy">— / 6</strong><div class="ops-segments" id="opsSegments" aria-hidden="true"></div><small>Karar başarısı veya kâr oranı değildir</small></article></div><div id="opsAlerts"></div></section>
 <section id="opsFlow" class="ops-section"><header class="ops-heading"><div><span class="ops-kicker">02 / GÖZLEMDEN KARARA</span><h2>Dünyada ne oldu, Brian ne yaptı?</h2></div></header><p class="ops-muted">Haberler ve son kararlar kendi kayıtlarıyla gösterilir. Birlikte görünmeleri, aralarında doğrulanmış neden–sonuç bağı olduğu anlamına gelmez.</p><div class="ops-grid"><div id="opsRadar"></div><article class="card card-pad ops-decisions"><div class="section-title">α Son kararların izi</div><p class="ops-muted">Karar, gerçekleşmiş işlem değildir. İşlem sonucu Hazine Defteri’nde doğrulanır.</p><div id="opsDecisions"></div><button class="btn" type="button" data-ops-room="meeting">Konseyde kanıtları incele ↗</button></article></div><div id="opsBelief"></div></section>
 <section id="opsRooms" class="ops-section"><header class="ops-heading"><div><span class="ops-kicker">03 / BRIAN ODALARI</span><h2>Derine in.</h2></div></header><div class="ops-room-grid">${[['meeting','◎','Toplantı','Olay, değerlendirme ve karar kanıtları'],['accounting','◇','Muhasebe','Gelir, gider, pozisyonlar ve grafikler'],['anatomy','⌬','Gelişim Anatomisi','Organlar, öğrenme ve gelişim kanıtları'],['engineering','⟨/⟩','Mühendislik','Aday kod, testler ve ilerleme']].map(([id,icon,name,desc])=>`<button type="button" class="ops-room" data-ops-room="${id}"><i aria-hidden="true">${icon}</i><span><b>${name}</b><small>${desc}</small></span><em aria-hidden="true">↗</em></button>`).join('')}</div><p id="opsRoomFeedback" class="ops-muted" role="status"></p><div id="opsRoomDetails" class="ops-grid"></div></section>
 <section id="opsControl" class="ops-section"><header class="ops-heading"><div><span class="ops-kicker">04 / SİSTEM</span><h2>Kontrol sende.</h2></div></header><div id="opsControls" class="ops-grid"></div><div id="opsDiagnostics" class="ops-grid"></div><div id="opsChat"></div></section><footer class="ops-footer">BRIAN FRONTIER OS <span>Kaynaklı kayıtlar · Açık durumlar · SHADOW</span></footer>`;
 $('command').after(shell);
 const card=id=>$(id)?.closest('article.card');
 const move=(node,target)=>{if(node&&$(target)&&!$(target).contains(node))$(target).appendChild(node)};
 function fold(node,target,title,open=false){if(!node||node.closest('.ops-fold'))return;const d=document.createElement('details');d.className='ops-fold';d.open=open;const summary=document.createElement('summary');summary.textContent=title;d.appendChild(summary);d.appendChild(node);$(target).appendChild(d)}
 const oldSections=[...document.querySelectorAll('main.shell > section')].filter(x=>x.id!=='command');
 move(card('alertFeed'),'opsAlerts');move($('worldPanel'),'opsRadar');
 fold(card('beliefP'),'opsBelief','Tahmini hareket ve işlem maliyeti');
 fold($('treasuryLedger'),'opsRoomDetails','Hazine · açık pozisyonlar ve son işlemler');
 fold($('developerBrian'),'opsRoomDetails','Mühendislik · iş ve kaynak kayıtları');
 fold(card('researchFeed'),'opsRoomDetails','Araştırma · deney ve yetenek kayıtları');
 fold(card('openMeeting'),'opsRoomDetails','Toplantı · modül özeti');
 fold($('developmentLaunchV2'),'opsRoomDetails','Anatomi · ayrıntılı analiz');
 fold(document.querySelector('.treasury-control'),'opsControls','Sanal kasa tutarını ayarla');
 move(document.querySelector('.system-control'),'opsControls');
 fold(card('moduleList'),'opsDiagnostics','Modüller · bağlantı ve çalışma kanıtları');
 fold(card('alphaFeed'),'opsDiagnostics','ALPHA · ayrıntılı karar kayıtları');
 fold(document.querySelector('main.shell > .metrics'),'opsDiagnostics','Ek sistem göstergeleri');
 fold($('chatPanel'),'opsChat','Canlı durum asistanına sor');
 for(const el of oldSections)if(!el.children.length){el.hidden=true;el.style.display='none';}
 const label=(id,value)=>{const e=$(id)?.parentElement?.querySelector('span,label');if(e)e.textContent=value};
 label('beliefP','Tahmini brüt hareket');label('beliefQ','Tahmini işlem maliyeti');label('beliefX','Tahmini net avantaj');label('healthValue','Güncel modül oranı');
 const chatTitle=$('chatPanel')?.querySelector('.section-title');if(chatTitle)chatTitle.textContent='Canlı durum asistanı';
 const chatSub=$('chatPanel')?.querySelector('.section-sub');if(chatSub)chatSub.textContent='Sistem kayıtlarından hazır durum özetleri verir.';
 const roomActions={meeting:()=>$('openMeeting')?.click(),accounting:()=>window.openTreasuryAccounting?.(),positions:()=>window.openTreasuryAccountingPositions?.()||window.openTreasuryAccounting?.(),anatomy:()=>window.openBrianAnatomy?.(),engineering:()=>window.openBrianEngineeringRoom?.()};
 shell.addEventListener('click',e=>{const b=e.target.closest('[data-ops-room]');if(b){const id=b.dataset.opsRoom;const fn={accounting:'openTreasuryAccounting',positions:'openTreasuryAccountingPositions',anatomy:'openBrianAnatomy',engineering:'openBrianEngineeringRoom'}[id];const available=id==='meeting'?!!$('openMeeting'):typeof window[fn]==='function'||(id==='positions'&&typeof window.openTreasuryAccounting==='function');if(available){$('opsRoomFeedback').textContent='';roomActions[id]()}else $('opsRoomFeedback').textContent='Oda henüz hazırlanıyor. Biraz sonra tekrar deneyebilirsin.';}
 const a=e.target.closest('.ops-nav a');if(a){e.preventDefault();const target=$(a.hash.slice(1));target?.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});target?.setAttribute('tabindex','-1');target?.focus({preventScroll:true})}});
 // Hero shortcuts may point into a folded panel. Open the corresponding disclosure first.
 document.addEventListener('click',e=>{const b=e.target.closest('#command [data-module]')||(e.target.closest('#ncOpen')?document.querySelector('#command [data-module][aria-pressed="true"]'):null);if(!b)return;const ids={world:'worldPanel',behavior:'beliefP',alpha:'alphaFeed',treasury:'treasuryLedger',research:'researchFeed',ocean:'moduleList'};let n=$(ids[b.dataset.module]);while(n){if(n.tagName==='DETAILS')n.open=true;n=n.parentElement}},true);
 function normalizeTreasury(){
  const t=(typeof V4!=='undefined'?V4.autonomy?.treasury:null)||(typeof S!=='undefined'?S.treasury?.snapshot:null);
  const actions=typeof V4!=='undefined'?V4.autonomy?.treasury_actions:null;
  if(!Array.isArray(t?.positions)&&$('treasuryPositions'))$('treasuryPositions').textContent='Pozisyon ayrıntıları bu özete ulaşmadı. Muhasebe odasından kayıtları inceleyebilirsin.';
  if(!Array.isArray(actions)&&$('treasuryActions'))$('treasuryActions').textContent='İşlem ayrıntıları henüz alınmadı; bu, işlem yapılmadığı anlamına gelmez.';
  if(t?.promotion_gate_open==null&&$('treasuryLedgerBadge'))$('treasuryLedgerBadge').textContent='KOŞUL BİLİNMİYOR';
  for(const k of $('treasuryLedgerSummary')?.children||[]){const name=k.querySelector('span')?.textContent,b=k.querySelector('b');if(!b)continue;
   if(name==='Açık Pozisyon'&&!Array.isArray(t?.positions))b.textContent=Number.isInteger(t?.open_positions)?t.open_positions:'—';
   if(name==='Cycle İşlemi'&&t?.action_count==null)b.textContent='—';
   if(name==='Promotion Gate'&&t?.promotion_gate_open==null)b.textContent='—';
  }
 }
 if(typeof renderTreasuryV4==='function'){const original=renderTreasuryV4;renderTreasuryV4=function(){original.apply(this,arguments);normalizeTreasury()};}
 normalizeTreasury();
 const stamp=v=>v&&Number.isFinite(Date.parse(v))?new Date(v).toLocaleString('tr-TR'):'Zaman bilinmiyor';
 const money=v=>v===null?'—':new Intl.NumberFormat('tr-TR',{style:'currency',currency:'USD',maximumFractionDigits:2}).format(v);
 const action=v=>({WAIT:'Bekle',HOLD:'Tut',VETO:'Veto',OPEN_LONG:'Uzun yön adayı',OPEN_SHORT:'Kısa yön adayı',NO_TRADE:'İşlem yapma',CLOSE:'Kapatma kararı'}[v]||v||'Karar bekleniyor');
 const openPositions=()=>{if(typeof window.openTreasuryAccountingPositions==='function')window.openTreasuryAccountingPositions();else window.openTreasuryAccounting?.();};
 let signature='';
 function update(){if(document.hidden)return;const hb=typeof STABILITY!=='undefined'?STABILITY.heartbeat:null,err=typeof STABILITY!=='undefined'?STABILITY.error:null;
 const v=window.BrianCommandModel?.project(hb,typeof moduleRows==='function'?moduleRows():[],err);if(!v)return;
 $('opsFresh').textContent=`${v.heartbeatFresh?'Heartbeat güncel':'Güncel bağlantı doğrulanamadı; varsa son kayıtlar gösteriliyor'} · ${stamp(v.heartbeatAt)}`;
 $('opsEquity').textContent=money(v.equity);const liveSummary=typeof S!=='undefined'?S.treasury?.summary:null,liveSnap=typeof S!=='undefined'?S.treasury?.snapshot:null,count=liveSummary?.open_positions??hb?.control?.treasury?.open_positions;$('opsPositions').textContent=Number.isInteger(Number(count))?String(Number(count)):(v.positions??'—');const treasuryAt=liveSnap?.observed_at||hb?.control?.treasury?.observed_at;$('opsTreasuryTime').textContent=`${v.treasuryFresh?'Güncel kasa':'Son kayıt · güncellik doğrulanmadı'} · ${stamp(treasuryAt)}`;if($('opsPositionsMeta'))$('opsPositionsMeta').textContent=`${Number(count)||0} açık · ${stamp(treasuryAt)} · Dokun: varlık, neden ve kâr/zarar`;
 $('opsHealthy').textContent=v.heartbeatFresh?`${v.live} / 6`:'— / 6';$('opsSegments').innerHTML=v.modules.map(m=>`<i data-state="${m.state}"></i>`).join('');
 const ds=typeof S!=='undefined'?S.control?.alpha_v2?.decisions:[];const decisions=Array.isArray(ds)&&ds.length?ds.slice(0,3):hb?.alpha?[hb.alpha]:[];
 const next=JSON.stringify(decisions);if(next!==signature){const changed=!!signature;signature=next;$('opsDecisions').innerHTML=decisions.length?decisions.map(d=>`<article class="ops-trace ${changed?'ops-arrival':''}"><div class="ops-trace-top"><b>${E(String(d.asset_id||'Varlık belirtilmemiş').replace('crypto:',''))}</b><span>${E(action(d.action))}</span></div><time>${E(stamp(d.observed_at||d.created_at))}</time><p>${E(d.reason||d.rationale||'Bu kayıtta açıklanmış karar gerekçesi bulunmuyor.')}</p></article>`).join(''):'<p class="ops-muted">Karar kaydı bekleniyor. İşlem veya sonuç üretilmedi.</p>';}
 const list=$('criticalNews');if(list&&!$('opsMoreNews')){const b=document.createElement('button');b.id='opsMoreNews';b.className='btn';b.textContent='Diğer gelişmeleri göster';b.setAttribute('aria-expanded','false');b.onclick=()=>{const expanded=b.getAttribute('aria-expanded')!=='true';b.setAttribute('aria-expanded',String(expanded));list.classList.toggle('ops-expanded',expanded);b.textContent=expanded?'İlk üç gelişmeyi göster':'Diğer gelişmeleri göster'};list.after(b)}
 if($('opsMoreNews'))$('opsMoreNews').hidden=!list||list.children.length<=3;
 }
 const pc=$('opsPositionsCard');if(pc){pc.addEventListener('click',openPositions);pc.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();openPositions();}});}
 update();setInterval(update,3000);document.addEventListener('visibilitychange',update);
})();
