const API='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-world-status';
const KEY_STORAGE='mcp-dashboard-key-v1';
const $=id=>document.getElementById(id);
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function key(){return(localStorage.getItem(KEY_STORAGE)||'').trim();}
function dt(v){if(!v)return'—';try{return new Intl.DateTimeFormat('tr-TR',{timeZone:'Europe/Berlin',day:'2-digit',month:'2-digit',year:'numeric',hour:'2-digit',minute:'2-digit'}).format(new Date(v));}catch{return'—';}}
function num(v,d=2){const n=Number(v);return Number.isFinite(n)?n.toFixed(d):'—';}
function dir(v){const n=Number(v);return n>0?['↑','up']:n<0?['↓','down']:['↔','neutral'];}
function pill(text,tone=''){return`<span class="world-pill ${tone}">${esc(text)}</span>`;}
function toast(message){const el=$('toast');if(!el)return;el.textContent=String(message);el.classList.add('show');clearTimeout(toast.t);toast.t=setTimeout(()=>el.classList.remove('show'),3200);}
function showLock(v){$('worldLock')?.classList.toggle('show',v);}
async function api(){const k=key();if(!k)throw Error('Anahtar yok');const r=await fetch(API,{method:'POST',headers:{'content-type':'application/json','x-brian-dashboard-key':k},body:'{}',cache:'no-store'});let data={};try{data=await r.json();}catch{}if(!r.ok){if(r.status===401){localStorage.removeItem(KEY_STORAGE);showLock(true);}throw Error(data.error||`HTTP ${r.status}`);}return data;}
function impactText(row){const [arrow,tone]=dir(row.conditional_direction);return`${pill(`${arrow} ${row.asset_id}`,tone)} ${esc(row.rationale)}`;}
function render(data){
  const s=data.summary||{};
  $('worldSync').textContent=`Son durum ${dt(data.observed_at)}`;
  $('worldState').textContent=`◉ ${data.status||'—'} · Discovery ${s.discovery_status||'—'} · Brain ${s.world_brain_status||'—'}`;
  $('kNarratives').textContent=String(s.narratives??0);$('kEntities').textContent=String(s.unique_entities??0);$('kFuture').textContent=String(s.upcoming_events??0);$('kScenarios').textContent=String(s.scenario_branches??0);
  $('narrativeList').innerHTML=(data.narratives||[]).map(n=>`<div class="world-row"><div class="world-title">${esc(n.label)} ${pill(`strength ${num(n.strength,2)}`,Number(n.strength)>.65?'up':'neutral')}</div><div class="world-meta">${esc(n.narrative_id)} · breadth ${esc(n.breadth)} · balance ${num(n.direction_balance,2)} · ${dt(n.observed_at)}<br>${(n.entity_ids||[]).slice(0,8).map(x=>pill(x)).join(' ')}</div></div>`).join('')||'<div class="world-row">Narrative snapshot bekleniyor.</div>';
  $('futureList').innerHTML=(data.upcoming_events||[]).map(e=>`<div class="world-row"><div class="world-title">${esc(e.event_kind)} ${pill(e.stage,'neutral')}</div><div class="world-meta"><b>${dt(e.scheduled_at)}</b><br>${esc(e.title)}<br>${(e.asset_ids||[]).slice(0,8).map(x=>pill(x)).join(' ')}</div></div>`).join('')||'<div class="world-row">Explicit zamanlı yaklaşan olay yok.</div>';
  $('mechanismList').innerHTML=(data.mechanisms||[]).map(m=>`<div class="world-row"><div class="world-title">${esc(m.narrative_id)} ${pill(`güven ${num(m.confidence,2)}`,'neutral')}</div><div class="world-meta"><b>Sebep:</b> ${esc(m.cause)}<br><b>Zincir:</b> ${(m.transmission||[]).map(esc).join(' → ')}<br><b>Counter-evidence:</b> ${m.counter_evidence_required?'zorunlu':'—'}</div></div>`).join('')||'<div class="world-row">Causal mechanism bekleniyor.</div>';
  $('impactList').innerHTML=(data.impacts||[]).slice(0,40).map(i=>`<div class="world-row"><div class="world-title">${impactText(i)}</div><div class="world-meta">Güven ${num(i.confidence,2)} · ${esc(i.stage)} · ALPHA doğrudan etki: ${i.direct_alpha_influence?'EVET':'HAYIR'}</div></div>`).join('')||'<div class="world-row">Cross-asset impact adayı yok.</div>';
  $('relationList').innerHTML=(data.relations||[]).slice(0,35).map(r=>`<div class="world-row"><div class="world-title">${esc(r.src_entity_id)} → ${esc(r.dst_entity_id)} ${pill(r.relation,r.relation==='CO_MENTIONED'?'neutral':'')}</div><div class="world-meta">Güven ${num(r.confidence,2)} · ${esc(r.mechanism)}</div></div>`).join('')||'<div class="world-row">Entity relation bekleniyor.</div>';
  const c=data.collectors||{};
  $('runtimeList').innerHTML=[['World Discovery Eye',c.discovery],['World Brain',c.world_brain]].map(([name,r])=>`<div class="world-row"><div class="world-title">${esc(name)} ${pill(r?.status||'NO_DATA',r?.status==='SUCCESS'?'up':r?.status==='FAILED'?'down':'neutral')}</div><div class="world-meta">${r?`${dt(r.started_at)} · observed ${esc(r.observed_records)} · stored ${esc(r.stored_records)}${(r.degraded_sources||[]).length?` · degraded ${(r.degraded_sources||[]).map(esc).join(', ')}`:''}`:'Henüz runtime kaydı yok.'}</div></div>`).join('');
}
async function refresh(){try{const data=await api();showLock(false);render(data);}catch(e){$('worldSync').textContent='World Brain durumu alınamadı';toast(e.message||String(e));}}
$('worldUnlock')?.addEventListener('click',()=>{const v=($('worldKey')?.value||'').trim();if(!v)return;localStorage.setItem(KEY_STORAGE,v);refresh();});
$('worldKey')?.addEventListener('keydown',e=>{if(e.key==='Enter')$('worldUnlock')?.click();});
if(!key())showLock(true);else refresh();setInterval(()=>{if(key())refresh();},15000);
