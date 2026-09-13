'use strict';

/* Brian Engineer live console — read-only view of PR #100 engineering state-machine. */
const ENGINEERING_STATUS_ENDPOINT = `${ROOT}/brian-frontier-engineering-status`;
const ENGINEER = { data: null, error: null, lastSync: null };
const ENGINEER_PHASES = [
  ['CLAIMED','GÖREV'],['UNDERSTAND','ANLA'],['PLAN','PLAN'],['CODE','KOD'],
  ['COMPILE','DERLE'],['TEST','TEST'],['REPLAY','REPLAY'],['STRESS','STRESS'],
  ['REVIEW','REVIEW'],['PR','PR'],['PREVIEW','PREVIEW'],['MEASURE','ÖLÇ'],
  ['HUMAN_APPROVAL','ÖMER ONAYI'],['DEPLOY','DEPLOY'],['MONITOR','İZLE'],['COMPLETE','TAMAM']
];

(function installEngineerStyles(){
  if(document.getElementById('engineerConsoleStyle')) return;
  const style=document.createElement('style');
  style.id='engineerConsoleStyle';
  style.textContent=`
    .eng-summary{grid-template-columns:repeat(6,minmax(0,1fr))!important}
    .eng-pipeline{display:flex;gap:5px;flex-wrap:wrap;padding:10px 0 12px}
    .eng-step{font-size:8px;font-weight:900;letter-spacing:.04em;padding:6px 7px;border-radius:8px;border:1px solid rgba(105,137,165,.22);background:rgba(7,21,35,.75);color:#7790a6}
    .eng-step.pass{border-color:rgba(65,242,180,.45);background:rgba(22,113,84,.18);color:#65f5c1}
    .eng-step.current{border-color:rgba(53,215,255,.72);background:rgba(25,121,154,.22);color:#88eaff;box-shadow:0 0 14px rgba(34,199,255,.16)}
    .eng-step.blocked{border-color:rgba(255,97,115,.65);background:rgba(135,25,42,.2);color:#ff8593}
    .eng-head{padding:10px;border:1px solid rgba(66,203,255,.18);background:rgba(4,22,38,.72);border-radius:12px;margin-bottom:8px}
    .eng-head b{color:#ecfbff;font-size:12px}.eng-meta{color:#7f96aa;font-size:9px;margin-top:5px;overflow-wrap:anywhere}
    .eng-reason{margin-top:7px;padding:7px 8px;border-radius:8px;background:rgba(123,25,39,.17);border:1px solid rgba(255,97,115,.25);color:#ff9aa5;font-size:9px;line-height:1.4}
    .eng-queue{display:grid;gap:6px;margin-top:8px}.eng-queue-row{display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:start;padding:8px;border-radius:9px;background:rgba(4,22,38,.62);border:1px solid rgba(74,166,211,.12);font-size:9px}.eng-priority{color:#66f1d0;font-weight:900}.eng-id{color:#6f879d;font-family:ui-monospace,monospace}.eng-status-note{font-size:9px;color:#8aa1b6;margin:5px 0 2px}
    @media(max-width:900px){.eng-summary{grid-template-columns:repeat(3,minmax(0,1fr))!important}}
    @media(max-width:560px){.eng-summary{grid-template-columns:repeat(2,minmax(0,1fr))!important}.eng-step{font-size:7px;padding:5px 6px}}
  `;
  document.head.appendChild(style);
})();

function engShortSha(v){const s=String(v||'');return s?s.slice(0,10):'—';}
function engEvent(run,kind){return v4Array(run?.recent_events).some(e=>String(e.event_kind)===kind&&e.passed===true);}
function engPassed(run,phase){
  if(!run) return false;
  if(phase==='CLAIMED') return true;
  if(phase==='UNDERSTAND') return engEvent(run,'UNDERSTAND');
  if(phase==='PLAN') return engEvent(run,'PLAN');
  if(phase==='CODE') return engEvent(run,'CODE')||Boolean(run.commit_sha);
  if(phase==='COMPILE') return run.compile_passed===true;
  if(phase==='TEST') return run.tests_passed===true;
  if(phase==='REPLAY') return run.replay_passed===true;
  if(phase==='STRESS') return run.stress_passed===true;
  if(phase==='REVIEW') return run.review_passed===true;
  if(phase==='PR') return Boolean(run.pr_number||run.pr_url);
  if(phase==='PREVIEW') return run.preview_passed===true;
  if(phase==='MEASURE') return run.measurement_passed===true;
  if(phase==='HUMAN_APPROVAL') return run.human_approval_status==='APPROVED'||run.phase==='HUMAN_APPROVAL';
  if(phase==='DEPLOY') return Boolean(run.deployed_sha);
  if(phase==='MONITOR') return Boolean(run.monitor_status);
  if(phase==='COMPLETE') return run.phase==='COMPLETE'&&run.status==='COMPLETE';
  return false;
}

function engCurrentIndex(run){
  if(!run) return -1;
  if(run.phase==='REPLAY'&&run.replay_passed===true&&run.stress_passed!==true) return ENGINEER_PHASES.findIndex(([p])=>p==='STRESS');
  return ENGINEER_PHASES.findIndex(([p])=>p===String(run.phase));
}

async function refreshEngineerConsole(){
  if(!key()) return;
  try{
    ENGINEER.data=await post(ENGINEERING_STATUS_ENDPOINT,{});
    ENGINEER.error=null;
    ENGINEER.lastSync=new Date();
  }catch(error){ENGINEER.error=String(error?.message||error);}
  renderEngineerConsole();
}

function renderEngineerConsole(){
  const a=V4?.autonomy;
  const e=ENGINEER.data;
  const badge=$('autonomyBadge');
  if(!badge||!document.getElementById('developerBrian')) return;
  const title=document.querySelector('#developerBrian .section-title');
  const sub=document.querySelector('#developerBrian .section-sub');
  if(title) title.textContent='👨‍💻 Yazılımcı Brian / Otonom Mühendislik';
  if(sub) sub.textContent='Eksik yetenek → araştırma → gerçek kod → replay/stress → bağımsız review → PR → preview → ölçüm → Ömer onayı.';
  if(!e){
    if(ENGINEER.error){badge.textContent='ENGINEER HATASI';badge.className='badge bad';}
    return;
  }

  const s=e.summary||{}, c=e.control||{};
  const active=e.current_run||null, approval=e.approval_run||null;
  const focus=approval||active||v4Array(e.recent_runs)[0]||null;
  if(approval){badge.textContent='ÖMER ONAYI BEKLİYOR';badge.className='badge warn';}
  else if(active){badge.textContent='BRIAN KODLUYOR';badge.className='badge ok';}
  else if(c.autonomous_claim_enabled===true){badge.textContent='OTONOM CLAIM AÇIK';badge.className='badge ok';}
  else {badge.textContent='KONTROLLÜ · CLAIM KAPALI';badge.className='badge info';}

  const summary=$('autonomySummary');
  if(summary){
    summary.classList.add('eng-summary');
    summary.innerHTML=`
      <div class="v4-stat"><span>Otonom Claim</span><b>${c.autonomous_claim_enabled===true?'AÇIK':'KAPALI'}</b></div>
      <div class="v4-stat"><span>Bekleyen İş</span><b>${Number(s.eligible_queue||0)}</b></div>
      <div class="v4-stat"><span>Aktif Run</span><b>${Number(s.active_runs||0)}</b></div>
      <div class="v4-stat"><span>Review Geçen</span><b>${Number(s.review_passed_runs||0)}</b></div>
      <div class="v4-stat"><span>Ömer Onayı</span><b>${Number(s.waiting_human_approval||0)}</b></div>
      <div class="v4-stat"><span>Tamamlanan</span><b>${Number(s.completed_runs||0)}</b></div>`;
  }

  const gov=$('autonomyGovernance');
  if(gov) gov.innerHTML=`
    <span class="v4-pill">PR #100 ENGINEER MOTORU</span>
    <span class="v4-pill">MAX RUN ${Number(c.max_concurrent_runs||1)}</span>
    <span class="v4-pill lock">İNSAN ONAYI ${c.require_human_approval===true?'ZORUNLU':'HATA'}</span>
    <span class="v4-pill lock">LIVE EXECUTION KAPALI</span>
    <span class="v4-pill lock">DIP KİLİTLİ / AYRI</span>`;

  const stream=$('codeStream');
  if(!stream) return;
  const blocks=[];
  if(focus){
    const currentIndex=engCurrentIndex(focus);
    blocks.push(`<div class="eng-head"><b>${esc(v4Short(focus.objective||focus.hypothesis_id||'Brian Engineer run',120))}</b><div class="eng-meta">Run ${esc(String(focus.run_id||'').slice(0,8))} · branch ${esc(focus.branch_name||'—')} · candidate ${esc(engShortSha(focus.commit_sha))} · ${age(focus.updated_at)} önce</div>${focus.status==='BLOCKED'?`<div class="eng-reason"><b>FAIL-CLOSED:</b> ${esc(v4Short(focus.failure_reason||'Gate başarısız',260))}</div>`:''}</div>`);
    blocks.push(`<div class="eng-pipeline">${ENGINEER_PHASES.map(([phase,label],i)=>{
      const passed=engPassed(focus,phase);const blocked=focus.status==='BLOCKED'&&i===currentIndex;const current=!blocked&&!passed&&i===currentIndex;
      return `<span class="eng-step ${blocked?'blocked':passed?'pass':current?'current':''}">${esc(label)}${passed?' ✓':''}</span>`;
    }).join('')}</div>`);
    if(focus.pr_url) blocks.push(`<div class="eng-status-note">PR #${Number(focus.pr_number||0)} · ${esc(v4Short(focus.pr_url,110))}</div>`);
    if(focus.preview_url) blocks.push(`<div class="eng-status-note">Preview: ${esc(v4Short(focus.preview_url,110))}</div>`);
  }else{
    blocks.push('<div class="eng-head"><b>Aktif Engineer run yok.</b><div class="eng-meta">Scheduled workflow hazır; kontrollü canary veya autonomous claim bekleniyor.</div></div>');
  }

  const queue=v4Array(e.eligible_queue);
  blocks.push(`<div class="eng-status-note">SIRADAKİ UYGUN MÜHENDİSLİK İŞLERİ · ${queue.length}/${Number(s.eligible_queue||0)}</div>`);
  blocks.push(`<div class="eng-queue">${queue.slice(0,6).map((q,i)=>`<div class="eng-queue-row"><span class="eng-priority">#${i+1}</span><div><b>${esc(v4Short(q.objective||q.hypothesis_id,105))}</b><div class="eng-id">${esc(String(q.request_id||'').slice(0,16))} · ${esc(v4Array(q.changed_paths).slice(0,2).join(' · '))}</div></div><span class="v4-tag">${Math.round(Number(q.metadata?.priority||0)*100)}%</span></div>`).join('')||'<div class="v4-muted">Uygun queue işi yok.</div>'}</div>`);
  stream.innerHTML=blocks.join('');
}

const previousRenderAutonomyV4=renderAutonomyV4;
renderAutonomyV4=function(){
  previousRenderAutonomyV4();
  renderEngineerConsole();
};

setTimeout(refreshEngineerConsole,400);
setInterval(refreshEngineerConsole,15000);
