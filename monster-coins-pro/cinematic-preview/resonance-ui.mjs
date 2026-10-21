import {createRecordStream,eventTone,phaseAt} from './resonance-model.mjs';
const $=id=>document.getElementById(id),esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const display=v=>typeof v==='object'?JSON.stringify(v):String(v??'—');
const clock=v=>Number.isFinite(Date.parse(v))?new Date(v).toLocaleTimeString('tr-TR',{hour12:false}):'—';
export function createResonance({onCue,onMood}){
 const stream=createRecordStream();let active=null,elapsed=0,lastPhase=-1,offline=false,graph=null,ledger=[],imported=false;
 function paint(record,mode){
  record={...record,delta:record.delta||[],observedAt:record.observedAt||graph?.at,mode:record.mode||'baseline'};
  if(record.mode==='import')mode='import';
  active={record,mode};elapsed=0;lastPhase=-1;const r=record.raw||{},tone=eventTone(record);
  document.body.style.setProperty('--event-color',tone);document.body.dataset.resonance='active';document.body.dataset.eventNode=record.id;let tag=document.getElementById('resSceneLabel');if(!tag){tag=document.createElement('div');tag.id='resSceneLabel';document.getElementById('stage').append(tag);}tag.textContent=(mode==='live'||mode==='revision'?'YENİ KAYIT':'KAYIT GÖSTERİMİ / YENİ OLAY DEĞİL')+' · '+record.label+' · '+clock(record.at);
  $('resMode').textContent=mode==='live'?'YENİ KAYIT / CANLI GÖZLEM':mode==='revision'?'KAYIT REVİZYONU / CANLI GÖZLEM':mode==='baseline'?'İLK GÖZLEM / VARIŞ İDDİASI YOK':mode==='import'?'İÇE AKTARILMIŞ KAYIT / CANLI DEĞİL':'KAYIT CANLANDIRMASI / YENİ OLAY DEĞİL';
  $('resTitle').textContent=record.label+' / '+(r.action||r.activity_state||r.status||'KAYIT');
  $('resIdentity').textContent=record.path+' · '+(record.record_id||r.decision_id||'kaynak zamanı '+clock(record.at));
  $('resSource').textContent=record.label;$('resRecordKind').textContent=record.group==='collector'?'TOPLAYICI KAYDI':'KAYNAK KAYDI';
  $('resTimes').innerHTML=[['KAYNAK',clock(record.at)],['YANIT',clock(record.observedAt)],['KİMLİK',record.record_id||r.decision_id||'Kaynakta yok']].map(([k,v])=>`<div><span>${esc(k)}</span><b title="${esc(v)}">${esc(v)}</b></div>`).join('');
  const fields=Object.entries(r).filter(([k])=>!['metadata','reason','veto_reason'].includes(k));
  $('resFieldCount').textContent=fields.length+' ALAN';$('resFields').innerHTML=fields.slice(0,8).map(([k,v])=>`<div class="${record.delta.includes(k)?'field-changed':''}"><span>${esc(k.replaceAll('_',' '))}</span><b title="${esc(display(v))}">${esc(display(v))}</b></div>`).join('');
  const alpha=record.id==='alpha';$('resLinkKind').textContent=alpha?'DOĞRUDAN /alpha':'NEDENSEL BAĞ YOK';$('resAction').textContent=alpha?(r.action||'Karar alanı yok'):'İLİŞKİ DOĞRULANMADI';
  $('resReason').textContent=alpha?(r.reason||r.veto_reason||'Bu kaynağın yanıtında karar gerekçesi yok. Yetkili arşiv gerekçeleri ayrı inceleme ekranında açılır.'):'Bu kayıt ile ALPHA kararı arasında açık kaynak kimliği bulunmadığı için bir neden-sonuç zinciri çizilmiyor.';
  $('resRelation').textContent=alpha?'Kanıt skoru, başarı olasılığı değildir.':'Aynı yanıt zarfı / '+record.path+' — karar veya işlem kanıtı değil.';
  $('resReplay').disabled=false;onCue?.(record,mode);onMood?.(tone,.55);
  document.querySelectorAll('.node-label,.node-row,.event,.relay-row').forEach(el=>el.classList.toggle('res-origin',el.dataset.select===record.id));
 }
 function renderLedger(){ledger=stream.records();$('resLedgerCount').textContent=ledger.length+' KAYIT';$('resLedger').innerHTML=ledger.slice(0,5).map(n=>`<button data-res-record="${n.seq}" title="Gerçek kaydı canlandır"><time>${clock(n.at)}</time><span>${esc(n.label)}</span><b>${n.mode==='baseline'?'İLK':n.mode==='revision'?'REV':n.mode==='import'?'JSON':'YENİ'}</b></button>`).join('');}
 $('resLedger').onclick=e=>{const b=e.target.closest('[data-res-record]');if(b){const n=ledger.find(n=>n.seq===Number(b.dataset.resRecord));if(n)paint(n,'replay');}};
 $('resReplay').onclick=()=>{if(active)paint(active.record,'replay');};
 function focus(id){const n=stream.records().find(n=>n.id===id)||graph?.nodes.find(n=>n.id===id);if(!n)return false;paint(n,'replay');return true;}
 document.addEventListener('click',e=>{const b=e.target.closest('[data-select]');if(b)focus(b.dataset.select);});
 return {
  focus,
  observe(g,options){if(!!options?.replay!==imported){active=null;imported=!!options?.replay;}graph=g;const added=stream.observe(g,options);renderLedger();if(!active&&added.length){const first=added.find(n=>n.id==='alpha')||added[0];paint(first,'baseline');elapsed=8;}},
  health(g,degraded){graph=g;offline=degraded||!g?.fresh;},
  tick(dt,{offline:degraded=false,reduced=false}={}){
   if(!active)return;document.body.classList.toggle('res-motion-paused',reduced);offline=degraded||!graph?.fresh;
   if(offline){stream.clear();document.body.style.setProperty('--event-energy','0');document.body.dataset.resonance='offline';$('resMode').textContent=(imported?'İÇE AKTARILMIŞ KAYIT / CANLI DEĞİL · ':'')+'KAYNAK ESKİ / BAĞLANTI YOK · SON KAYIT';return;}
   elapsed+=dt;
   if(elapsed>8){const next=stream.next();if(next)paint(next,next.mode);else document.body.dataset.resonance='idle';}
   const phase=phaseAt(elapsed);if(phase!==lastPhase){lastPhase=phase;document.body.dataset.resPhase=String(phase);document.querySelectorAll('#resPhases li').forEach((li,i)=>{li.classList.toggle('done',i<phase);li.classList.toggle('current',i===phase);});}
   document.body.style.setProperty('--event-energy',reduced?'0':String(Math.max(0,1-Math.abs(elapsed-3)/2)));
   $('resQueue').textContent=String(stream.pending()).padStart(2,'0')+' BEKLEYEN';
  }
 };
}