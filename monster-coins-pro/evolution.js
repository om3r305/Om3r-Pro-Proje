const EVO_API='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-evolution-status';
const LAB_API='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-evolution-lab-status';
const KEY_STORAGE='mcp-dashboard-key-v1';
const $=id=>document.getElementById(id);
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function key(){return(localStorage.getItem(KEY_STORAGE)||'').trim();}
function clock(v){if(!v)return'—';try{return new Intl.DateTimeFormat('tr-TR',{timeZone:'Europe/Berlin',day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit'}).format(new Date(v));}catch{return'—';}}
function pill(text,tone=''){return`<span class="evo-pill ${tone}">${esc(text)}</span>`;}
function healthTone(v){return v==='HEALTHY'?'ok':v==='MISSING'||v==='DEGRADED'?'bad':'warn';}
function severityTone(v){return v==='CRITICAL'?'bad':v==='HIGH'?'warn':'';}
function decisionTone(v){return v==='PROMOTE_CANDIDATE'||v==='READY_FOR_HUMAN_REVIEW'?'ok':v==='REJECT'||v==='BLOCK'?'bad':'warn';}
function toast(message){const el=$('toast');if(!el)return;el.textContent=String(message);el.classList.add('show');clearTimeout(toast.t);toast.t=setTimeout(()=>el.classList.remove('show'),3200);}
function showLock(v){$('evoLock')?.classList.toggle('show',v);}
async function call(url,optional=false){const k=key();if(!k)throw Error('Anahtar yok');const r=await fetch(url,{method:'POST',headers:{'content-type':'application/json','x-brian-dashboard-key':k},body:'{}',cache:'no-store'});let data={};try{data=await r.json();}catch{}if(!r.ok){if(r.status===401&&!optional){localStorage.removeItem(KEY_STORAGE);showLock(true);}if(optional)return{status:'UNAVAILABLE',error:data.error||`HTTP ${r.status}`};throw Error(data.error||`HTTP ${r.status}`);}return data;}
async function api(){const [core,lab]=await Promise.all([call(EVO_API),call(LAB_API,true)]);return{core,lab};}
function renderCore(data){
  const d=data.dashboard||{},counts=d.capabilityCounts||{},sources=d.sourceCounts||{};
  $('evoSync').textContent=`Son durum ${clock(data.observed_at)}`;
  $('evoState').textContent=`◉ ${d.overall||data.status||'—'}`;
  $('evoHealthy').textContent=String(counts.HEALTHY??0);
  $('evoHealthyMeta').textContent=`${(data.capabilities||[]).length} toplam yetenek · ${counts.STALE??0} stale · ${counts.DEGRADED??0} degraded`;
  $('evoCritical').textContent=String((d.criticalGaps||[]).length);
  $('evoCriticalMeta').textContent=`${(data.gaps||[]).length} görünür capability gap`;
  $('evoSources').textContent=String(sources.total??0);
  $('evoSourcesMeta').textContent=`${sources.official??0} resmî · ${sources.researchEligible??0} araştırmaya uygun`;
  $('evoDecisionSources').textContent=String(sources.decisionEligible??0);

  $('evoCapabilities').innerHTML=(data.capabilities||[]).map(c=>`<div class="evo-row"><div class="evo-title">${esc(c.name)} ${pill(c.health,healthTone(c.health))} ${pill(c.stage)}</div><div class="evo-meta">${esc(c.description)}<br>${esc(c.capabilityId)}${c.metadata?.age_seconds!=null?` · yaş ${esc(c.metadata.age_seconds)} sn`:''}</div></div>`).join('')||'<div class="evo-row">Capability snapshot bekleniyor.</div>';
  $('evoGaps').innerHTML=(data.gaps||[]).slice(0,20).map(g=>`<div class="evo-row"><div class="evo-title">${esc(g.capabilityId)} ${pill(g.severity,severityTone(g.severity))}</div><div class="evo-meta">${esc(g.reason)}<br><b>Öneri:</b> ${esc(g.suggestedAction)}</div></div>`).join('')||'<div class="evo-row">Aktif gap yok.</div>';
  $('evoSourceList').innerHTML=(data.sources||[]).slice(0,20).map(s=>`<div class="evo-row evo-source"><div class="evo-title">${esc(s.provider)} ${pill(s.authorityClass,s.assessment?.eligibleForResearch?'ok':'warn')}</div><div class="evo-meta">${esc(s.canonicalUri)}<br>Trust ${Number(s.assessment?.trustScore??0).toFixed(2)} · manipülasyon ${Number(s.manipulationRisk??0).toFixed(2)} · ${esc(s.stage)}<br>${s.corroborationRequired?'Bağımsız doğrulama gerekli':'Birincil kaynak sınıfı'} · ALPHA doğrudan etki: hayır</div></div>`).join('')||'<div class="evo-row">Kaynak keşif kaydı bekleniyor.</div>';
  $('evoJournal').innerHTML=(data.journal||[]).slice(0,16).map(e=>`<div class="evo-row"><div class="evo-title">${esc(e.title)} ${pill(e.stage)}</div><div class="evo-meta">${clock(e.occurred_at)} · ${esc(e.summary)}</div></div>`).join('')||'<div class="evo-row">Evolution journal bekleniyor.</div>';
  $('evoRuns').innerHTML=(data.runs||[]).slice(0,14).map(r=>`<div class="evo-row"><div class="evo-title">${clock(r.started_at)} ${pill(r.status,r.status==='SUCCESS'?'ok':r.status==='FAILED'?'bad':'warn')}</div><div class="evo-meta">cap ${esc(r.capability_snapshots)} · gap ${esc(r.gap_snapshots)} · source ${esc(r.source_candidates)} · assessment ${esc(r.source_assessments)} · journal ${esc(r.journal_events)}${r.error_message?`<br>${esc(r.error_message)}`:''}</div></div>`).join('')||'<div class="evo-row">Orchestrator henüz çalışmadı.</div>';
}
function renderLab(data){
  const s=data.summary||{};
  $('evoHypothesisCount').textContent=String(s.hypotheses??0);
  $('evoExperimentCount').textContent=String(s.experiments??0);
  $('evoCodeCount').textContent=String(s.code_candidates??0);
  $('evoHumanReady').textContent=String(s.ready_for_human_review??0);
  if(data.status==='UNAVAILABLE'){
    $('evoHypotheses').innerHTML='<div class="evo-row">Lab status endpoint henüz deployment bekliyor.</div>';
    $('evoExperiments').innerHTML='<div class="evo-row">Prospective experiment verisi deployment sonrası gelecek.</div>';
    $('evoCode').innerHTML='<div class="evo-row">Self-coding sandbox deployment sonrası aktif olacak.</div>';
    $('evoLabRuns').innerHTML='<div class="evo-row">Cloud lab workerları deployment bekliyor.</div>';
    return;
  }
  $('evoHypotheses').innerHTML=(data.hypotheses||[]).slice(0,16).map(h=>{const m=h.metadata||{};return`<div class="evo-row"><div class="evo-title">${esc(m.hypothesis_kind||h.title||'HYPOTHESIS')} ${pill(h.stage)} ${pill(`P ${Number(m.priority??0).toFixed(2)}`)}</div><div class="evo-meta">${clock(h.observed_at)} · ${esc(h.problem_statement)}<br><b>Mekanizma:</b> ${esc(h.proposed_mechanism)}</div></div>`}).join('')||'<div class="evo-row">Research hypothesis bekleniyor.</div>';
  const promoByExp=new Map((data.promotions||[]).map(p=>[String(p.experiment_id),p]));
  $('evoExperiments').innerHTML=(data.experiments||[]).slice(0,16).map(e=>{const p=promoByExp.get(String(e.experiment_id));return`<div class="evo-row"><div class="evo-title">${esc(e.challenger_version)} ${pill(e.mode)} ${p?pill(p.decision,decisionTone(p.decision)):pill('ÖLÇÜM BEKLİYOR','warn')}</div><div class="evo-meta">${clock(e.created_at_source)} · min ${esc(e.minimum_samples)} sample / ${esc(e.minimum_regimes)} regime${p?`<br>Score ${Number(p.score??0).toFixed(2)} · ${esc((p.reasons||[]).join(' · '))}`:''}</div></div>`}).join('')||'<div class="evo-row">Prospective experiment bekleniyor.</div>';
  const reviewByCandidate=new Map((data.code_reviews||[]).map(r=>[String(r.candidate_id),r]));
  $('evoCode').innerHTML=(data.code_requests||[]).slice(0,16).map(c=>{const r=reviewByCandidate.get(String(c.candidate_id));return`<div class="evo-row"><div class="evo-title">${esc(c.branch_name)} ${r?pill(r.verdict,decisionTone(r.verdict)):pill('PLANNED','warn')}</div><div class="evo-meta">${clock(c.requested_at)} · ${esc(c.objective)}<br>${(c.changed_paths||[]).map(esc).join(' · ')}${r?`<br>${esc((r.reasons||[]).join(' · '))}`:''}</div></div>`}).join('')||'<div class="evo-row">Kod adayı planı bekleniyor.</div>';
  $('evoLabRuns').innerHTML=(data.runs||[]).slice(0,18).map(r=>`<div class="evo-row"><div class="evo-title">${esc(r.collector_id)} ${pill(r.status,r.status==='SUCCESS'?'ok':r.status==='FAILED'?'bad':'warn')}</div><div class="evo-meta">${clock(r.started_at)} · observed ${esc(r.observed_records)} · stored ${esc(r.stored_records)}${r.error_message?`<br>${esc(r.error_message)}`:''}</div></div>`).join('')||'<div class="evo-row">Lab worker kaydı bekleniyor.</div>';
}
function render(payload){renderCore(payload.core||{});renderLab(payload.lab||{});}
async function refresh(){try{const data=await api();showLock(false);render(data);}catch(e){$('evoSync').textContent='Evolution durumu alınamadı';toast(e.message||String(e));}}
function bind(){
  $('evoUnlock')?.addEventListener('click',()=>{const v=($('evoKey')?.value||'').trim();if(!v)return;localStorage.setItem(KEY_STORAGE,v);refresh();});
  $('evoKey')?.addEventListener('keydown',e=>{if(e.key==='Enter')$('evoUnlock')?.click();});
}
bind();if(!key())showLock(true);else refresh();setInterval(()=>{if(key())refresh();},15000);
