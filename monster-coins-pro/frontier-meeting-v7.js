'use strict';
/* Compact, read-only evidence council. Existing backend execution is unchanged. */
(()=>{
 const css=document.createElement('link');css.rel='stylesheet';css.href='/frontier-meeting-v7.css?v=20260916-1';document.head.appendChild(css);
 const $=s=>document.querySelector(s),E=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const state={selected:'alpha',tab:'evidence',paused:false,signature:'',seen:{},log:[],view:null,event:null,m:null};
 const labels={ok:'KANIT VAR',wait:'BEKLİYOR',blocked:'BLOKE'};
 const clock=t=>new Date(t).toLocaleTimeString('tr-TR',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
 const HISTORY_KEY='brian-council-observed-events-v1';
 function history(events){
  try{
   const read=k=>{const x=JSON.parse(localStorage.getItem(k)||'[]');return Array.isArray(x)?x:[]};
   const records=new Map();
   for(const x of [...read('brian-meeting-event-history-v7'),...read(HISTORY_KEY)]){const time=x.time||x.eventTime;const ms=typeof time==='number'?time:Date.parse(time);if(!Number.isFinite(ms)||Date.now()-ms>604800000||ms>Date.now())continue;const key=x.uri||x.title+'|'+ms;records.set(key,{title:String(x.title||''),time:ms,uri:meetingV6SafeUrl(x.uri)});}
   for(const e of events){const ms=Date.parse(e.time);records.set(e.uri||e.title+'|'+ms,{title:e.title,time:ms,uri:meetingV6SafeUrl(e.uri)});}
   const out=[...records.values()].sort((a,b)=>b.time-a.time).slice(0,80);const value=JSON.stringify(out);if(localStorage.getItem(HISTORY_KEY)!==value)localStorage.setItem(HISTORY_KEY,value);return out;
  }catch{return []}
 }
 function snapshot(){
  const m=meetingSnapshotV4();
  const events=meetingV6MajorEvents().filter(e=>BrianCouncilModel.fresh(e.time,Date.now(),e.urgency==='CRITICAL'?7200000:3600000));
  let event=events[0]||null;
  if(event){const raw=(Array.isArray(S.news?.items)?S.news.items:[]).find(x=>x.provenance_uri===event.uri&&x.observed_at===event.time);event={...event,event_id:raw?.event_id||null};}
  const evidence=event?meetingV6Evidence(event):{};
  const view=BrianCouncilModel.project({event,decision:m.d,treasury:m.t,evidence,rows:m.rows});
  return {m,event,evidence,view,events};
 }
 function draw(force=false){
  const room=$('#meetingModal .room');if(!room)return;
  const data=snapshot(),{view,event,m,evidence,events}=data;state.view=view;state.event=event;state.m=m;
  const archive=history(events);
  const title=event?.title||'Konsey yeni bir gelişme bekliyor';
  const signal=$('#meetingV6SignalStrip');if(signal){signal.className='meeting-v6-signal '+(view.blocked?'red':event?'orange':'calm');signal.textContent=`${view.title} · ${title}`;}
  const badge=$('#meetingCard .badge');if(badge)badge.textContent=view.title;
  $('#meetingCard')?.classList.remove('meeting-v6-red','meeting-v6-orange','meeting-alarm','meeting-review');
  const modal=$('#meetingModal');modal.querySelector('.section-title').textContent='Brian · Olay Konseyi';modal.querySelector('.section-sub').textContent='Olay, kanıt ve kararın gerekçesi';
  modal.querySelector('.modal-card')?.classList.remove('meeting-v6-modal-red','meeting-v6-modal-orange');
  const signatures={};for(const g of view.gates)signatures[g.id]=JSON.stringify([g.state,g.text,event?.event_id,event?.time,g.id==='alpha'?m.d:null,g.id==='treasury'?m.t:null]);
  const changed=Object.keys(signatures).filter(id=>state.seen[id]&&state.seen[id]!==signatures[id]);
  for(const id of changed){const g=view.gates.find(x=>x.id===id);state.log.unshift({time:Date.now(),text:`${g.name} kaydı değişti · ${labels[g.state]}. ${g.text}`});}
  state.log=state.log.slice(0,30);state.seen=signatures;
  const signature=JSON.stringify([view,event,evidence,events.map(e=>[e.title,e.time]),m.d?.action,m.d?.asset_id,m.d?.observed_at,m.t?.promotion_gate_open]);
  if(!force&&state.signature===signature)return;state.signature=signature;
  const focus=document.activeElement?.closest('[data-cv]')?.dataset.cv;
  const selected=view.gates.find(g=>g.id===state.selected)||view.gates[0];
  const url=meetingV6SafeUrl(event?.uri),publisher=event?meetingV6Publisher(event):'—';
  room.innerHTML=`<div class="cv ${state.paused?'cv-paused':''}"><section class="cv-summary"><div><div class="cv-eyebrow">CANLI DURUMDAN KANIT ÖZETİ</div><h2>${E(view.title)}</h2><div class="cv-topic">${E(title)}</div><p>${E(view.reason)}</p></div><div class="cv-next"><div class="cv-eyebrow">BİR SONRAKİ KOŞUL</div><p>${E(view.next)}</p><span class="cv-count">${view.ready}<small> / ${view.gates.length}</small></span><p class="cv-note">Görünür koşul sağlandı · başarı veya oylama puanı değildir.</p></div></section><section class="cv-stage"><div class="cv-stage-head"><div><div class="cv-eyebrow">KONSEY / KANIT AĞI</div><p>Bir üye seç, gerekçesini gör.</p></div><button type="button" class="cv-motion" data-cv="motion" aria-pressed="${state.paused}">${state.paused?'Hareketi aç':'Hareketi durdur'}</button></div><div class="cv-network">${view.gates.map(g=>`<button type="button" class="cv-node ${changed.includes(g.id)&&modal.classList.contains('show')?'changed':''}" data-cv="${g.id}" data-state="${g.state}" aria-pressed="${g.id===state.selected}"><i aria-hidden="true">${g.icon}</i><b>${g.name}</b><small>${labels[g.state]}</small></button>`).join('')}</div><div class="cv-detail"><strong>${E(selected.name)} · ${labels[selected.state]}</strong><p>${E(selected.text)}</p><p class="cv-note">Aranan kanıt: ${E(selected.need)}</p></div><div class="cv-bottom"><span>Parlama = bu oturumda değişen kayıt.<br>Üye açıklamaları kayıtlardan türetilen özettir.</span><span>Salt okunur<br>SHADOW</span></div></section><section><div class="cv-tabs" role="tablist" aria-label="Konsey ayrıntıları">${[['evidence','Kanıtlar'],['updates','Gelişmeler'],['result','Sonuç']].map(([id,name])=>`<button type="button" role="tab" id="cv-tab-${id}" aria-controls="cv-panel-${id}" aria-selected="${state.tab===id}" tabindex="${state.tab===id?0:-1}" data-cv="tab-${id}">${name}</button>`).join('')}</div><div class="cv-panel" id="cv-panel-evidence" role="tabpanel" aria-labelledby="cv-tab-evidence" ${state.tab!=='evidence'?'hidden':''}><div class="cv-row"><span class="cv-tag">KAYNAK</span>${E(publisher)} ${url?`<a href="${E(url)}" target="_blank" rel="noopener noreferrer">Haberi aç ↗</a>`:''}<small>${event?E(evidence.label):'Olay kaydı bekleniyor.'} · Alınma zamanı: ${event?.time?E(clock(event.time)):'—'}</small></div><div class="cv-row"><span class="cv-tag">OLAY BAĞI</span>${view.linked?'Olay ve varlık eşleşiyor.':'ALPHA ile doğrulanmış olay/varlık bağlantısı yok.'}<small>Son ALPHA: ${E(m.d?.action||'—')} · ${E(m.d?.asset_id||'—')} · ${E(m.d?.observed_at||m.d?.created_at||'Zaman bilinmiyor')}</small></div>${view.gates.map(g=>`<div class="cv-row"><span class="cv-tag">${g.name}</span>${E(g.text)}</div>`).join('')}<p class="cv-note">Kaynağın tanınması, haberin doğru veya fiyat etkisinin kanıtlanmış olduğu anlamına gelmez.</p></div><div class="cv-panel" id="cv-panel-updates" role="tabpanel" aria-labelledby="cv-tab-updates" ${state.tab!=='updates'?'hidden':''}><p class="cv-note">Bu sayfa açıkken görülen değişiklikler. Saatler ekranın değişikliği gördüğü andır; kalıcı toplantı tutanağı değildir.</p><ol class="cv-log">${state.log.length?state.log.map(l=>`<li><time>${clock(l.time)}</time>${E(l.text)}</li>`).join(''):'<li>Henüz yeni bir değişiklik gözlenmedi.</li>'}</ol><p class="cv-note">Bu tarayıcıdaki olay arşivi · son 7 gün, en çok 80 kayıt. Eski kayıtlar işlem sonucu veya onay olarak yorumlanmaz.</p>${archive.map(e=>`<div class="cv-row"><span class="cv-tag">OLAY KAYDI</span>${E(e.title)}<small>${E(new Date(e.time).toLocaleString('tr-TR'))}${e.uri?` · <a href="${E(e.uri)}" target="_blank" rel="noopener noreferrer">Kaynak ↗</a>`:''}</small></div>`).join('')}</div><div class="cv-panel" id="cv-panel-result" role="tabpanel" aria-labelledby="cv-tab-result" ${state.tab!=='result'?'hidden':''}><h2>${E(view.title)}</h2><p>${E(view.reason)}</p><div class="cv-detail"><strong>Ne değişirse ilerler?</strong><p>${E(view.next)}</p></div><p>Bu ekran işlem emri vermez. Alım/satımın gerçekleştiği, yalnız sunucunun ilgili işlem kaydıyla doğrulanabilir.</p><p class="cv-note">Araştırma ve Şüpheci için bu olaya bağlı ayrı bir değerlendirme kaydı bu ekrana gelmiyor; bağımsız konuşma veya oy üretilmez. Yazılımcı Brian kendi mühendislik kanalında çalışır.</p></div></section></div>`;
  if(focus)room.querySelector(`[data-cv="${focus}"]`)?.focus({preventScroll:true});
 }
 const modal=$('#meetingModal');
 // The legacy dashboard refresh still writes this status; keep its hook out of the new layout.
 if(modal&&!$('#roomSkeptic')){const legacy=document.createElement('span');legacy.id='roomSkeptic';legacy.hidden=true;modal.appendChild(legacy);}
 modal?.setAttribute('role','dialog');modal?.setAttribute('aria-modal','true');
 modal?.addEventListener('click',e=>{const b=e.target.closest('[data-cv]');if(!b)return;const id=b.dataset.cv;if(id==='motion')state.paused=!state.paused;else if(id.startsWith('tab-'))state.tab=id.slice(4);else state.selected=id;draw(true);});
 modal?.addEventListener('keydown',e=>{if(e.key==='Escape'){modal.classList.remove('show');$('#openMeeting')?.focus();}if(e.target.matches('[role=tab]')&&['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();const tabs=['evidence','updates','result'];const i=tabs.indexOf(state.tab);state.tab=e.key==='Home'?tabs[0]:e.key==='End'?tabs[2]:tabs[(i+(e.key==='ArrowRight'?1:2))%3];draw(true);$(`#cv-tab-${state.tab}`)?.focus();}});
 renderMeetingV4=()=>draw();
 const baseAnswer=answer;answer=function(q){if(/toplantı|konsey|alarm|büyük olay/i.test(String(q))){const {view}=snapshot();bubble(String(q),'user');bubble(`${view.title}. ${view.reason} Sonraki koşul: ${view.next}`,'brian');return;}return baseAnswer(q)};
 new MutationObserver(()=>{if(modal.classList.contains('show'))draw(true)}).observe(modal,{attributes:true,attributeFilter:['class']});
 setInterval(()=>{if(!document.hidden&&modal.classList.contains('show'))draw()},15000);
 draw(true);
})();
