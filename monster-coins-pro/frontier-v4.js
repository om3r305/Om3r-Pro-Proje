'use strict';

/* Brian Frontier V4 — additive live console. Keeps DIP fully isolated. */
const AUTONOMY_ENDPOINT = `${ROOT}/brian-frontier-autonomy-status`;
let V4 = { autonomy: null, error: null, lastSync: null };

(function installV4Style(){
  if (document.getElementById('frontierV4Style')) return;
  const style=document.createElement('style');
  style.id='frontierV4Style';
  style.textContent=`
  .v4-console{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:linear-gradient(180deg,rgba(1,14,29,.96),rgba(1,8,18,.96));border:1px solid rgba(33,205,255,.22);border-radius:14px;padding:12px;max-height:330px;overflow:auto}
  .v4-line{display:grid;grid-template-columns:auto 1fr;gap:9px;padding:8px 0;border-bottom:1px solid rgba(120,190,230,.09);font-size:11px;line-height:1.45}.v4-line:last-child{border-bottom:0}.v4-tag{color:#35f0ce;font-weight:800}.v4-muted{color:#8399ad}.v4-path{color:#55cfff;overflow-wrap:anywhere}.v4-ok{color:#41f2b4}.v4-warn{color:#ffbd55}.v4-bad{color:#ff6173}
  .v4-summary{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:11px 0}.v4-stat{background:rgba(5,27,45,.72);border:1px solid rgba(59,203,255,.15);border-radius:11px;padding:10px}.v4-stat span{display:block;color:#7892aa;font-size:9px;text-transform:uppercase;letter-spacing:.08em}.v4-stat b{display:block;margin-top:4px;font-size:17px;color:#effcff}
  .v4-governance{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}.v4-pill{border:1px solid rgba(54,220,188,.3);background:rgba(17,98,83,.15);color:#7ef5d2;padding:5px 8px;border-radius:999px;font-size:9px;font-weight:800}.v4-pill.lock{border-color:rgba(255,189,85,.3);background:rgba(113,75,12,.17);color:#ffcf79}
  .treasury-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:10px 0}.treasury-kpi{background:rgba(6,27,43,.7);border:1px solid rgba(75,190,235,.14);border-radius:11px;padding:9px}.treasury-kpi span{display:block;color:#7f96aa;font-size:9px}.treasury-kpi b{display:block;color:#f2fdff;font-size:15px;margin-top:4px}.position-row,.action-row,.source-row{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:8px;align-items:center;padding:8px 0;border-bottom:1px solid rgba(120,190,230,.09);font-size:10px}.position-row:last-child,.action-row:last-child,.source-row:last-child{border-bottom:0}
  .meeting-alarm{border-color:rgba(255,57,86,.72)!important;box-shadow:0 0 0 1px rgba(255,57,86,.2),0 0 30px rgba(255,30,61,.18)!important;animation:v4alarm 1.25s ease-in-out infinite}.meeting-alarm .section-title{color:#ff8492}.meeting-alarm .badge{background:rgba(170,20,41,.32)!important;border-color:#ff536a!important;color:#ffd7dc!important}
  @keyframes v4alarm{0%,100%{box-shadow:0 0 0 1px rgba(255,57,86,.16),0 0 20px rgba(255,30,61,.10)}50%{box-shadow:0 0 0 2px rgba(255,57,86,.52),0 0 42px rgba(255,30,61,.30)}}
  .meeting-brief{margin:12px 0;padding:12px;border-radius:13px;background:linear-gradient(135deg,rgba(8,40,64,.9),rgba(12,20,38,.9));border:1px solid rgba(74,204,255,.22)}.meeting-trigger{font-weight:900;font-size:13px}.meeting-decision{margin-top:8px;color:#70f5d2;font-weight:900}.meeting-agents{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin-top:9px}.meeting-agent{background:rgba(2,15,29,.65);padding:8px;border-radius:9px;border:1px solid rgba(76,165,210,.14);font-size:10px;line-height:1.4}.meeting-agent b{display:block;color:#aeeeff;margin-bottom:3px}
  .ticker .radar-label{color:#3ef4d0;font-weight:900;letter-spacing:.08em}.ticker .radar-age{color:#91a7bb}.secondary-warning{color:#ffbd55;font-size:9px;margin-left:5px}
  @media(max-width:760px){.v4-summary,.treasury-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.meeting-agents{grid-template-columns:1fr}.v4-console{max-height:260px}}
  `;
  document.head.appendChild(style);
})();

async function refreshAutonomyV4(){
  if(!key()) return;
  try{
    V4.autonomy=await post(AUTONOMY_ENDPOINT,{});
    V4.error=null;
    V4.lastSync=new Date();
  }catch(e){
    V4.error=String(e?.message||e);
  }
  renderAutonomyV4();
  renderTreasuryV4();
  renderMeetingV4();
}

function v4Array(v){ return Array.isArray(v)?v:[]; }
function v4Short(v,n=70){const s=String(v??'');return s.length>n?s.slice(0,n-1)+'…':s;}
function v4Stage(s){return String(s||'').replaceAll('_',' ');}
function v4Direction(d){return Number(d)===1?'LONG':Number(d)===-1?'SHORT':'—';}

function injectV4Panels(){
  if(document.getElementById('autonomyRow')) return;
  const chat=$('chatPanel')?.closest('section');
  if(!chat) return;
  const row=document.createElement('section');
  row.id='autonomyRow';
  row.className='grid2';
  row.innerHTML=`
    <article id="developerBrian" class="card card-pad">
      <div class="card-head"><div><div class="section-title">👨‍💻 Yazılımcı Brian / Otonom Geliştirme</div><div class="section-sub">Brian'ın keşfettiği eksikler, yazdığı aday kodlar, test kanıtı ve dünya kaynak kütüphanesi.</div></div><span id="autonomyBadge" class="badge warn">OKUNUYOR</span></div>
      <div id="autonomySummary" class="v4-summary"></div>
      <div id="autonomyGovernance" class="v4-governance"></div>
      <div class="section-sub" style="margin:12px 0 6px">CANLI KOD / DENEY AKIŞI</div><div id="codeStream" class="v4-console"></div>
      <div class="section-sub" style="margin:12px 0 6px">DÜNYA KAYNAK KÜTÜPHANESİ</div><div id="sourceStream" class="v4-console"></div>
    </article>
    <article id="treasuryLedger" class="card card-pad">
      <div class="card-head"><div><div class="section-title">💰 Hazine Defteri / Pozisyonlar</div><div class="section-sub">Brian'ın gerçekten açtığı-kapattığı SHADOW işlemler, açık pozisyonlar, maliyet ve gate durumu.</div></div><span id="treasuryLedgerBadge" class="badge info">CANLI KASA</span></div>
      <div id="treasuryLedgerSummary" class="treasury-grid"></div>
      <div class="section-sub" style="margin:12px 0 6px">AÇIK POZİSYONLAR</div><div id="treasuryPositions" class="v4-console"></div>
      <div class="section-sub" style="margin:12px 0 6px">SON HAZİNE İŞLEMLERİ</div><div id="treasuryActions" class="v4-console"></div>
    </article>`;
  chat.parentNode.insertBefore(row,chat);

  const quick=$('chatPanel')?.querySelector('.quick');
  if(quick && !quick.querySelector('[data-v4="code"]')){
    [['code','Kod / Gelişim','Brian kendi kodunda ne yapıyor?'],['sources','Kaynaklar','Dünya kaynaklarında ne buldun?'],['meeting','Toplantı','Toplantı var mı?'],['positions','Pozisyonlar','Ne aldın ne sattın?']].forEach(([id,label,q])=>{
      const b=document.createElement('button');b.className='btn';b.dataset.v4=id;b.dataset.q=q;b.textContent=label;quick.appendChild(b);b.addEventListener('click',()=>answer(q));
    });
  }
  const meetingCard=$('openMeeting')?.closest('article');
  if(meetingCard) meetingCard.id='meetingCard';
  const room=document.querySelector('#meetingModal .room');
  if(room && !document.getElementById('meetingBrief')){
    const brief=document.createElement('div');brief.id='meetingBrief';brief.className='meeting-brief';room.parentNode.insertBefore(brief,room);
  }
}

function renderAutonomyV4(){
  injectV4Panels();
  const a=V4.autonomy;
  const badge=$('autonomyBadge');
  if(!badge) return;
  if(!a){badge.textContent=V4.error?'BAĞLANTI HATASI':'BEKLENİYOR';badge.className=`badge ${V4.error?'bad':'warn'}`;return;}
  badge.textContent='SANDBOX CANLI';badge.className='badge ok';
  const s=a.summary||{};
  $('autonomySummary').innerHTML=`
    <div class="v4-stat"><span>Kod İsteği</span><b>${Number(s.codegen_requests||0)}</b></div>
    <div class="v4-stat"><span>Aday Kod</span><b>${Number(s.code_candidates||0)}</b></div>
    <div class="v4-stat"><span>Test/Artifact</span><b>${Number(s.artifact_receipts||0)}</b></div>
    <div class="v4-stat"><span>Keşfedilen Kaynak</span><b>${Number(s.discovered_world_sources||0)}</b></div>`;
  $('autonomyGovernance').innerHTML=`
    <span class="v4-pill">ARAŞTIRMA OTONOM</span><span class="v4-pill">ADAY KOD ÜRETİMİ AÇIK</span><span class="v4-pill">TEST KANITI AÇIK</span><span class="v4-pill lock">CANONICAL UYGULAMA İNSAN ONAYLI</span><span class="v4-pill lock">DIP KİLİTLİ / AYRI</span>`;
  const requests=v4Array(a.codegen_requests), candidates=v4Array(a.code_candidates), artifacts=v4Array(a.artifact_receipts), hypotheses=v4Array(a.hypotheses);
  const lines=[];
  requests.slice(0,3).forEach(r=>lines.push(`<div class="v4-line"><span class="v4-tag">İSTEK</span><div><b>${esc(v4Short(r.objective||r.hypothesis_id,95))}</b><div class="v4-muted">${age(r.requested_at)} önce · ${esc(r.branch_name||'candidate branch')}</div><div class="v4-path">${esc(v4Array(r.changed_paths).join(' · ')||'path planlanıyor')}</div></div></div>`));
  candidates.slice(0,3).forEach(c=>lines.push(`<div class="v4-line"><span class="v4-tag">ADAY</span><div><b>${esc(v4Stage(c.stage))}</b><div class="v4-muted">${age(c.proposed_at)} önce · ${esc(c.candidate_id)}</div><div class="v4-path">${esc(v4Array(c.changed_paths).join(' · '))}</div></div></div>`));
  artifacts.slice(0,4).forEach(r=>lines.push(`<div class="v4-line"><span class="${r.passed===false?'v4-bad':r.passed===true?'v4-ok':'v4-warn'}">${esc(r.evidence_kind)}</span><div><b>${r.passed===false?'BAŞARISIZ':r.passed===true?'GEÇTİ':'KANIT'}</b><div class="v4-muted">${age(r.observed_at)} önce · ${esc(r.generated_by||'Brian Evolution')}</div><div class="v4-path">${esc(v4Array(r.changed_paths).join(' · '))}</div></div></div>`));
  if(!lines.length && hypotheses.length) hypotheses.slice(0,5).forEach(h=>lines.push(`<div class="v4-line"><span class="v4-tag">HİPOTEZ</span><div><b>${esc(v4Short(h.title,95))}</b><div class="v4-muted">${esc(v4Stage(h.stage))} · belirsizlik ${Math.round(Number(h.uncertainty||0)*100)}%</div></div></div>`));
  $('codeStream').innerHTML=lines.join('')||'<div class="v4-line"><span class="v4-warn">BEKLE</span><div>Henüz yeni kod üretim kanıtı yok.</div></div>';

  const sources=v4Array(a.source_library);
  $('sourceStream').innerHTML=sources.slice(0,8).map(src=>{const as=src.assessment||{};return `<div class="source-row"><span class="dot ${as.eligible_for_research?'ok':'warn'}"></span><div><b>${esc(src.provider||src.source_id)}</b><div class="v4-muted">${esc(src.source_kind||'kaynak')} · ${esc(src.authority_class||'UNKNOWN')} · ${age(src.discovered_at)} önce</div><div class="v4-path">${esc(v4Short(src.canonical_uri||'',95))}</div></div><span class="v4-tag">${as.trust_score!=null?Math.round(Number(as.trust_score)*100)+'%':'—'}</span></div>`}).join('')||'<div class="v4-line"><span class="v4-warn">KAYNAK</span><div>Yeni kaynak adayı bekleniyor.</div></div>';
}

function renderTreasuryV4(){
  injectV4Panels();
  const t=V4.autonomy?.treasury || S.treasury?.snapshot || null;
  if(!t) return;
  const positions=v4Array(t.positions), actions=v4Array(V4.autonomy?.treasury_actions);
  const gate=Boolean(t.promotion_gate_open);
  $('treasuryLedgerBadge').textContent=gate?'GATE AÇIK':'GATE KAPALI';$('treasuryLedgerBadge').className=`badge ${gate?'ok':'info'}`;
  $('treasuryLedgerSummary').innerHTML=`
    <div class="treasury-kpi"><span>Equity</span><b>${money(t.equity_usd)}</b></div><div class="treasury-kpi"><span>Nakit</span><b>${money(t.cash_usd)}</b></div><div class="treasury-kpi"><span>Dağıtım</span><b>${pct(t.deployment_pct)}</b></div><div class="treasury-kpi"><span>Gerçekleşen PnL</span><b>${money(t.realized_pnl_usd)}</b></div>
    <div class="treasury-kpi"><span>Toplam Maliyet</span><b>${money(t.cumulative_costs_usd)}</b></div><div class="treasury-kpi"><span>Açık Pozisyon</span><b>${positions.length}</b></div><div class="treasury-kpi"><span>Cycle İşlemi</span><b>${Number(t.action_count||0)}</b></div><div class="treasury-kpi"><span>Promotion Gate</span><b>${gate?'AÇIK':'KAPALI'}</b></div>`;
  $('treasuryPositions').innerHTML=positions.length?positions.map(p=>`<div class="position-row"><span class="dot ok"></span><div><b>${esc(String(p.assetId||p.asset_id||'ASSET').replace('crypto:',''))} · ${esc(v4Direction(p.direction))}</b><div class="v4-muted">Sermaye ${money(p.capitalUsd||p.capital_usd)} · giriş ${Number(p.entryPrice||p.entry_price||0).toLocaleString('tr-TR')}</div></div><span>${esc(v4Short(p.reason||'',35))}</span></div>`).join(''):`<div class="v4-line"><span class="v4-warn">NAKİT</span><div><b>Açık pozisyon yok.</b><div class="v4-muted">${esc(t.promotion_gate_reason||v4Array(t.blocked_reasons).join(' · ')||'Brian uygun after-cost edge ve açık promotion gate bekliyor.')}</div></div></div>`;
  $('treasuryActions').innerHTML=actions.length?actions.map(a=>`<div class="action-row"><span class="dot ${a.kind==='OPEN'?'ok':'info'}"></span><div><b>${esc(a.kind)} · ${esc(String(a.asset_id||'').replace('crypto:',''))} · ${esc(v4Direction(a.direction))}</b><div class="v4-muted">${age(a.observed_at)} önce · ${money(a.capital_usd)} · maliyet ${money(a.cost_usd)}</div><div>${esc(v4Short(a.reason||'',100))}</div></div><span>${Number(a.expected_net_edge_bps||0).toFixed(1)} bps</span></div>`).join(''):'<div class="v4-line"><span class="v4-warn">0 İŞLEM</span><div><b>Henüz Hazine OPEN / EXIT üretmedi.</b><div class="v4-muted">Bu, işlemlerin gizlendiği anlamına gelmez; mevcut canlı ledger gerçekten boş.</div></div></div>';
}

/* World core stays green when only the optional discovery sensor is degraded. */
const moduleRowsV3 = moduleRows;
moduleRows = function(){
  const rows=moduleRowsV3();
  const world=rows.find(r=>r.key==='world');
  const behavior=rows.find(r=>r.key==='behavior');
  const alpha=rows.find(r=>r.key==='alpha');
  const disc=S.world?.collectors?.discovery||null;
  if(world && world.state==='warn' && String(world.meta||'').includes('Ana Dünya Beyni canlı')){
    world.state='ok';
    world.meta=`Ana Dünya Beyni canlı${disc?.status==='FAILED'?' · keşif sensörü ayrı uyarıda':''}`;
  }
  if(behavior && world?.state==='ok' && alpha?.state==='ok' && S.news){
    behavior.state='ok';behavior.meta='World Core + ALPHA + kritik olay akışı canlı';
  }
  return rows;
};

renderAlerts = function(rows){
  const alerts=[];
  rows.forEach(m=>{if(m.state==='bad'||m.state==='warn')alerts.push({state:m.state,title:`${m.name}: ${stateText(m.state)}`,meta:m.meta})});
  const disc=S.world?.collectors?.discovery;
  if(disc && String(disc.status)==='FAILED') alerts.push({state:'warn',title:'Dünya Keşif Sensörü: harici kaynak gecikmesi',meta:disc.error_message||disc.error_class||'Keşif gözü başarısız'});
  if(V4.error) alerts.push({state:'warn',title:'Yazılımcı Brian görünümü gecikti',meta:V4.error});
  $('alertCount').textContent=String(alerts.length);$('alertCount').className=`badge ${alerts.some(a=>a.state==='bad')?'bad':alerts.length?'warn':'ok'}`;
  $('alertFeed').innerHTML=alerts.length?alerts.slice(0,8).map(a=>`<div class="module"><span class="dot ${a.state}"></span><div class="module-main"><div class="module-name">${esc(a.title)}</div><div class="module-meta">${esc(a.meta||'')}</div></div></div>`).join(''):'<div class="module"><span class="dot ok"></span><div class="module-main"><div class="module-name">Kritik teknik sorun yok</div><div class="module-meta">Haberler burada değil; yalnız gerçek sistem/bağlantı uyarıları gösterilir.</div></div></div>';
};

renderNews = function(){
  const items=news();
  $('newsBadge').textContent=items.length?`${items.length} ÖNEMLİ`:'AKIŞ YOK';$('newsBadge').className=`badge ${items.length?'ok':'warn'}`;
  $('criticalNews').innerHTML=items.length?items.slice(0,7).map(n=>`<div class="news"><div class="news-top"><div class="news-title">${esc(n.title)}</div><div class="severity ${String(n.urgency).toLowerCase()}">${n.urgency==='CRITICAL'?'KRİTİK':n.urgency==='HIGH'?'YÜKSEK':'ORTA'}</div></div><div class="news-meta">${esc(n.summary)}${n.asset?` · ${esc(n.asset)}`:''}<br>${age(n.time)} önce · ${esc(n.source||'Brian')}</div>${n.original?`<details class="news-meta"><summary>Orijinal claim</summary>${esc(n.original)}</details>`:''}</div>`).join(''):'<div class="news"><div class="news-title">Brian filtresinden geçen kritik gelişme henüz yok.</div></div>';
  const freshItems=items.filter(n=>{const s=ageSec(n.time);return s!=null&&s<=3600}).slice(0,4);
  const pieces=freshItems.length?freshItems.map(n=>`<div class="ticker-item"><span class="radar-label">BRIAN RADAR</span><span class="dot ${n.urgency==='CRITICAL'?'bad':n.urgency==='HIGH'?'warn':'info'}"></span><b>${esc(n.title)}</b><span class="radar-age">${esc(age(n.time))} önce · ${esc(n.source||'Brian')}</span></div>`):[`<div class="ticker-item"><span class="radar-label">BRIAN RADAR • CANLI</span><span class="dot ok"></span><b>Yeni kritik gelişme yok</b><span class="radar-age">tarama devam ediyor</span></div>`];
  $('tickerTrack').innerHTML=[...pieces,...pieces].join('');
};

function meetingSnapshotV4(){
  const rows=moduleRows();
  const bad=rows.filter(r=>r.state==='bad');
  const critical=news().find(n=>n.urgency==='CRITICAL' && (ageSec(n.time)??999999)<=3600);
  const d=S.control?.alpha_v2?.decisions?.[0];
  const t=V4.autonomy?.treasury||S.treasury?.snapshot||null;
  let active=bad.length>0||Boolean(critical), trigger='Rutin operasyon', decision='İZLE — olağan akış devam ediyor.';
  if(bad.length){trigger=`Teknik alarm: ${bad.map(x=>x.name).join(', ')}`;decision='FAIL-CLOSED — yeni risk alma; kırmızı hattı izole et ve taze kanıt bekle.'}
  else if(critical){trigger=`Kritik olay: ${critical.title}`;decision=d&&['OPEN_LONG','OPEN_SHORT'].includes(d.action)?`ALPHA ${act(d.action)} üretti; yalnız SHADOW Hazine gate ve maliyet kontrolünden geçir.`:'BEKLE / KANIT TOPLA — olay kritik, ALPHA henüz trade doğrulamıyor.'}
  return {active,trigger,decision,rows,bad,critical,d,t};
}
function renderMeetingV4(){
  injectV4Panels();
  const m=meetingSnapshotV4(),card=$('meetingCard'),badge=card?.querySelector('.badge');
  card?.classList.toggle('meeting-alarm',m.active);
  if(badge){badge.textContent=m.active?'ALARM TOPLANTISI':'RUTİN İZLEME';badge.className=`badge ${m.active?'bad':'info'}`;}
  const brief=$('meetingBrief');if(!brief)return;
  const world=S.world?.summary||{},d=m.d,t=m.t,latestReq=v4Array(V4.autonomy?.codegen_requests)[0];
  brief.innerHTML=`<div class="meeting-trigger">${m.active?'🚨':'🟢'} ${esc(m.trigger)}</div><div class="meeting-decision">Brian kararı: ${esc(m.decision)}</div><div class="meeting-agents">
    <div class="meeting-agent"><b>🌍 Dünya</b>${esc(`${world.narratives??0} anlatı · ${world.asset_impact_candidates??0} etki adayı${m.critical?' · kritik olay masada':''}`)}</div>
    <div class="meeting-agent"><b>α ALPHA</b>${d?esc(`${String(d.asset_id||'').replace('crypto:','')} · ${act(d.action)} · kanıt ${Number(d.evidence_score||0).toFixed(2)}`):'Karar kanıtı bekleniyor'}</div>
    <div class="meeting-agent"><b>◉ Hazine</b>${t?esc(`${money(t.equity_usd)} equity · ${money(t.cash_usd)} nakit · ${pct(t.deployment_pct)} kullanım`):'Kasa okunuyor'}</div>
    <div class="meeting-agent"><b>👨‍💻 Yazılımcı Brian</b>${latestReq?esc(v4Short(latestReq.objective,110)):'Yeni codegen talebi yok'}</div>
    <div class="meeting-agent"><b>🛡 Şüpheci</b>${m.bad.length?'Teknik kanıt eksik; fail-closed veto.':m.critical?'Priced-in, kaynak ve karşı kanıt kontrolü istiyor.':'Karşı kanıt taraması rutin.'}</div>
    <div class="meeting-agent"><b>🧠 Brian</b>${esc(m.decision)}</div></div>`;
}

const renderV3=render;
render=function(){renderV3();injectV4Panels();renderAutonomyV4();renderTreasuryV4();renderMeetingV4();};

const answerV3=answer;
answer=function(q){
  const text=String(q||'').trim(),l=text.toLocaleLowerCase('tr-TR');if(!text)return;
  const a=V4.autonomy;
  if(l.includes('kod')||l.includes('geliştir')||l.includes('yazılım')){
    const r=v4Array(a?.codegen_requests)[0],c=v4Array(a?.code_candidates)[0],ar=v4Array(a?.artifact_receipts)[0];
    const msg=a?`Yazılımcı Brian: ${a.summary?.codegen_requests??0} codegen isteği, ${a.summary?.code_candidates??0} aday kod, ${a.summary?.artifact_receipts??0} artifact/test kanıtı var. ${r?'Son hedef: '+v4Short(r.objective,120)+'. ':''}${c?'Son aday '+c.candidate_id+' '+v4Stage(c.stage)+'. ':''}${ar?'Son kanıt '+ar.evidence_kind+' '+(ar.passed===true?'geçti':ar.passed===false?'başarısız':'bekliyor')+'. ':''}Canonical ana koda otomatik uygulama kapalı; insan onayı zorunlu.`:'Otonom geliştirme servisi henüz okunamadı.';
    bubble(text,'user');setTimeout(()=>bubble(msg,'brian'),80);return;
  }
  if(l.includes('kaynak')||l.includes('kütüphane')||l.includes('dünya')){
    const src=v4Array(a?.source_library),disc=S.world?.collectors?.discovery;
    const msg=a?`Dünya kütüphanesinde ${a.summary?.discovered_world_sources??0} keşfedilmiş kaynak adayı var. Son örnekler: ${src.slice(0,3).map(x=>x.provider||x.source_id).join(', ')||'yok'}. ${disc?.status==='FAILED'?'Ana Dünya Beyni canlı; harici Discovery Eye şu an gecikiyor/hata alıyor.':'Discovery Eye canlı.'}`:'Kaynak kütüphanesi okunamadı.';
    bubble(text,'user');setTimeout(()=>bubble(msg,'brian'),80);return;
  }
  if(l.includes('toplantı')||l.includes('alarm')){
    const m=meetingSnapshotV4();bubble(text,'user');setTimeout(()=>bubble(`${m.active?'Alarm toplantısı aktif.':'Şu an alarm toplantısı yok.'} ${m.trigger}. Brian kararı: ${m.decision}`,'brian'),80);return;
  }
  if(l.includes('pozisyon')||l.includes('aldın')||l.includes('sattın')||l.includes('işlem')){
    const t=a?.treasury,actions=v4Array(a?.treasury_actions),pos=v4Array(t?.positions);let msg=t?`Hazine ${money(t.equity_usd)}. Açık pozisyon ${pos.length}. `:'Hazine okunamadı. ';
    msg+=actions.length?`Son işlem: ${actions[0].kind} ${String(actions[0].asset_id||'').replace('crypto:','')} ${money(actions[0].capital_usd)}.`:'Canlı Hazine ledgerında henüz OPEN/EXIT işlemi yok.';
    bubble(text,'user');setTimeout(()=>bubble(msg,'brian'),80);return;
  }
  return answerV3(q);
};

injectV4Panels();
refreshAutonomyV4();
setInterval(()=>{if(key()&&document.visibilityState==='visible'&&!S.busy)refreshAutonomyV4()},15000);
document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible'&&key())refreshAutonomyV4()});
