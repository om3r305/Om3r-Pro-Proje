const API='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-world-status';
const KEY_STORAGE='mcp-dashboard-key-v1';
const $=id=>document.getElementById(id);
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function key(){return(localStorage.getItem(KEY_STORAGE)||'').trim();}
function dt(v){if(!v)return'—';try{return new Intl.DateTimeFormat('tr-TR',{timeZone:'Europe/Berlin',day:'2-digit',month:'2-digit',year:'numeric',hour:'2-digit',minute:'2-digit'}).format(new Date(v));}catch{return'—';}}
function num(v,d=2){if(v==null||v==='')return'—';const n=Number(v);return Number.isFinite(n)?n.toFixed(d):'—';}
function dir(v){const n=Number(v);return n>0?['↑','up']:n<0?['↓','down']:['↔','neutral'];}
function pill(text,tone=''){return`<span class="world-pill ${tone}">${esc(text)}</span>`;}
function toast(message){const el=$('toast');if(!el)return;el.textContent=String(message);el.classList.add('show');clearTimeout(toast.t);toast.t=setTimeout(()=>el.classList.remove('show'),3200);}
function showLock(v){$('worldLock')?.classList.toggle('show',v);}
async function api(){const k=key();if(!k)throw Error('Anahtar yok');const r=await fetch(API,{method:'POST',headers:{'content-type':'application/json','x-brian-dashboard-key':k},body:'{}',cache:'no-store'});let data={};try{data=await r.json();}catch{}if(!r.ok){if(r.status===401){localStorage.removeItem(KEY_STORAGE);showLock(true);}throw Error(data.error||`HTTP ${r.status}`);}return data;}
function impactText(row){const [arrow,tone]=dir(row.conditional_direction);return`${pill(`${arrow} ${row.asset_id}`,tone)} ${esc(row.rationale)}`;}

function sourceLink(uri,label){
  try{const u=new URL(uri);if(u.protocol!=='https:'||u.username||u.password||u.port||!(/^[a-z][a-z0-9.-]+\.[a-z]{2,}$/i.test(u.hostname))||/(^|\.)(localhost|local|internal|test|invalid)$/i.test(u.hostname))return esc(label);u.search='';u.hash='';return '<a href="'+esc(u.toString())+'" target="_blank" rel="noopener noreferrer" style="color:#89d9dd;overflow-wrap:anywhere">'+esc(label)+'</a>';}catch{return esc(label);}
}
const requirementText={
 MECHANISM_TIME_INVALID:'Olay dosyasının zamanı doğrulanamadı',
 SOURCE_LOOKUP_FAILED:'Kaynak kayıtları alınamadı',
 SOURCE_REFERENCE_COVERAGE_INCOMPLETE:'Kaynak bağlantılarının bir bölümü eksik veya zaman bakımından uygun değil',
 PRIMARY_SOURCE_UNAVAILABLE:'Birincil kaynak kaydı bulunamadı',
 SOURCE_CLAIMS_NOT_INDEPENDENTLY_VERIFIED:'Kaynak iddiaları bağımsız olarak doğrulanmadı',
 COUNTER_EVIDENCE_NOT_ASSESSED:'Olaya özel karşı kanıt değerlendirilmedi',
 PRICED_IN_NOT_ASSESSED:'Etkinin fiyatlanmış olup olmadığı değerlendirilmedi',
 ACTIONABLE_HORIZON_UNRESOLVED:'Uygulanabilir zaman ufku belirlenmedi',
 PREDICTIVE_CALIBRATION_UNVERIFIED:'Tahmin olasılığı kalibre edilmedi'
};
function dossierCard(d){
 const s=d.source_summary||{},c=d.coverage||{},score=d.score||{};
 const sources=(d.source_claims||[]).map(x=>'<li>'+sourceLink(x.uri,x.claim||x.source_id||'Kaynak kaydı')+'<br><span class="world-muted">Yayın: '+dt(x.published_at)+' · Brian gördü: '+dt(x.captured_at)+' · Kaydedildi: '+dt(x.persisted_at)+'<br>İddia doğrulanmadı; tam metin okuması bu kayıttan kanıtlanmıyor.</span></li>').join('');
 const missing=(d.missing_requirements||[]).map(x=>'<li>'+esc(requirementText[x]||'Ek doğrulama gerekli')+'</li>').join('');
 const invalidators=(d.counter_evidence?.proposed_invalidators||[]).map(x=>'<li>'+esc(x)+'</li>').join('');
 return '<details class="world-row"><summary class="world-title" style="cursor:pointer">'+esc(d.narrative_id)+' '+pill('Araştırma eksik','neutral')+' '+pill('Kanıt puanı '+num(score.value),'neutral')+'</summary>'+
 '<div class="world-meta"><p><b>Hipotez:</b> '+esc(d.mechanism?.cause)+'<br><b>Aktarım:</b> '+(d.mechanism?.transmission||[]).map(esc).join(' → ')+'<br><b>Dosya zamanı:</b> '+dt(d.observed_at)+'</p>'+
 '<p>Bağlantısı bulunan kaynak: '+esc(c.available??0)+' / '+esc(c.references??0)+' · Birincil etiketli: '+esc(s.official_primary_label??0)+' · Profesyonel etiketli: '+esc(s.professional_label??0)+' · Doğrulanmamış keşif: '+esc(s.discovery_label??0)+'<br>Yayıncı alan adı grubu: '+esc(s.publisher_host_groups??0)+'; kaynak bağımsızlığı ayrıca doğrulanmalı.</p>'+
 '<b>İlişkili kaynak iddiaları</b><p>Bu kayıtların aynı konuyla ilişkilendirilmesi, mekanizmayı doğruladıkları anlamına gelmez. En fazla sekiz kaynak gösterilir.</p>'+(sources?'<ul>'+sources+'</ul>':'<p>Görüntülenebilir kaynak kaydı yok.</p>')+
 '<b>Tamamlanması gerekenler</b><ul>'+missing+'</ul>'+
 (invalidators?'<b>Önerilen geçersiz kılma koşulları</b><p>Bunlar elde edilmiş karşı kanıt değildir.</p><ul>'+invalidators+'</ul>':'')+
 '<p><b>İşlem için hazır değil.</b> Puan kalibre edilmiş olasılık değildir; bu dosya doğrudan işlem sinyali üretmez.</p></div></details>';
}

function render(data){
  const s=data.summary||{};
  $('worldSync').textContent=`Son durum ${dt(data.observed_at)}`;
  $('worldState').textContent=`◉ ${data.status||'—'} · Discovery ${s.discovery_status||'—'} · Brain ${s.world_brain_status||'—'}`;
  $('kNarratives').textContent=String(s.narratives??0);$('kEntities').textContent=String(s.unique_entities??0);$('kFuture').textContent=String(s.upcoming_events??0);$('kScenarios').textContent=String(s.scenario_branches??0);
  $('narrativeList').innerHTML=(data.narratives||[]).map(n=>`<div class="world-row"><div class="world-title">${esc(n.label)} ${pill(`strength ${num(n.strength,2)}`,Number(n.strength)>.65?'up':'neutral')}</div><div class="world-meta">${esc(n.narrative_id)} · breadth ${esc(n.breadth)} · balance ${num(n.direction_balance,2)} · ${dt(n.observed_at)}<br>${(n.entity_ids||[]).slice(0,8).map(x=>pill(x)).join(' ')}</div></div>`).join('')||'<div class="world-row">Narrative snapshot bekleniyor.</div>';
  $('futureList').innerHTML=(data.upcoming_events||[]).map(e=>`<div class="world-row"><div class="world-title">${esc(e.event_kind)} ${pill(e.stage,'neutral')}</div><div class="world-meta"><b>${dt(e.scheduled_at)}</b><br>${esc(e.title)}<br>${(e.asset_ids||[]).slice(0,8).map(x=>pill(x)).join(' ')}</div></div>`).join('')||'<div class="world-row">Explicit zamanlı yaklaşan olay yok.</div>';
  if($('dossierList'))$('dossierList').innerHTML=(data.dossiers||[]).map(dossierCard).join('')||'<div class="world-row">Kaynaklı dosya verisi henüz yok; bu durum doğrulama yapıldığı anlamına gelmez.</div>';
  $('mechanismList').innerHTML=(data.mechanisms||[]).map(m=>`<div class="world-row"><div class="world-title">${esc(m.narrative_id)} ${pill(`Kanıt puanı ${num(m.confidence,2)}`,'neutral')}</div><div class="world-meta"><b>Sebep:</b> ${esc(m.cause)}<br><b>Zincir:</b> ${(m.transmission||[]).map(esc).join(' → ')}<br><b>Karşı kanıt araştırması:</b> ${m.counter_evidence_required?'zorunlu':'—'}</div></div>`).join('')||'<div class="world-row">Causal mechanism bekleniyor.</div>';
  $('impactList').innerHTML=(data.impacts||[]).slice(0,40).map(i=>`<div class="world-row"><div class="world-title">${impactText(i)}</div><div class="world-meta">Sezgisel kanıt puanı ${num(i.confidence,2)} · ${esc(i.stage)} · ALPHA doğrudan etki: ${i.direct_alpha_influence?'EVET':'HAYIR'}</div></div>`).join('')||'<div class="world-row">Cross-asset impact adayı yok.</div>';
  $('relationList').innerHTML=(data.relations||[]).slice(0,35).map(r=>`<div class="world-row"><div class="world-title">${esc(r.src_entity_id)} → ${esc(r.dst_entity_id)} ${pill(r.relation,r.relation==='CO_MENTIONED'?'neutral':'')}</div><div class="world-meta">Sezgisel kanıt puanı ${num(r.confidence,2)} · ${esc(r.mechanism)}</div></div>`).join('')||'<div class="world-row">Entity relation bekleniyor.</div>';
  const c=data.collectors||{};
  $('runtimeList').innerHTML=[['World Discovery Eye',c.discovery],['World Brain',c.world_brain]].map(([name,r])=>`<div class="world-row"><div class="world-title">${esc(name)} ${pill(r?.status||'NO_DATA',r?.status==='SUCCESS'?'up':r?.status==='FAILED'?'down':'neutral')}</div><div class="world-meta">${r?`${dt(r.started_at)} · observed ${esc(r.observed_records)} · stored ${esc(r.stored_records)}${(r.degraded_sources||[]).length?` · degraded ${(r.degraded_sources||[]).map(esc).join(', ')}`:''}`:'Henüz runtime kaydı yok.'}</div></div>`).join('');
}
async function refresh(){try{const data=await api();showLock(false);render(data);}catch(e){$('worldSync').textContent='World Brain durumu alınamadı';toast(e.message||String(e));}}
$('worldUnlock')?.addEventListener('click',()=>{const v=($('worldKey')?.value||'').trim();if(!v)return;localStorage.setItem(KEY_STORAGE,v);refresh();});
$('worldKey')?.addEventListener('keydown',e=>{if(e.key==='Enter')$('worldUnlock')?.click();});
if(!key())showLock(true);else refresh();setInterval(()=>{if(key())refresh();},15000);
