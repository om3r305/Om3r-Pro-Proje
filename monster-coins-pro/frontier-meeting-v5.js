'use strict';

/* Brian Meeting V5 — evidence-aware operations council. Frontend-only; DIP untouched. */
(function installMeetingV5(){
  if(document.getElementById('frontierMeetingV5Style')) return;
  const style=document.createElement('style');
  style.id='frontierMeetingV5Style';
  style.textContent=`
    .meeting-review{border-color:rgba(255,188,75,.58)!important;box-shadow:0 0 0 1px rgba(255,188,75,.10),0 0 28px rgba(255,166,31,.10)!important}
    .meeting-review .section-title{color:#ffd58a}
    .meeting-brief-v5{display:grid;gap:12px}
    .meeting-topic{padding:13px;border-radius:12px;background:linear-gradient(135deg,rgba(7,34,55,.92),rgba(8,20,37,.94));border:1px solid rgba(73,195,241,.20)}
    .meeting-topic-title{font-size:14px;font-weight:900;line-height:1.35;color:#f2fbff}
    .meeting-topic-sub{margin-top:5px;color:#8da6ba;font-size:10px;line-height:1.5}
    .meeting-source-strip{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}
    .meeting-source-cell{padding:9px 10px;border-radius:10px;background:rgba(2,16,29,.72);border:1px solid rgba(72,169,211,.15);min-width:0}
    .meeting-source-cell span{display:block;color:#718ca3;font-size:8px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px}
    .meeting-source-cell b{display:block;color:#e8f8ff;font-size:10px;overflow-wrap:anywhere}
    .meeting-source-cell a{color:#66dafa;text-decoration:none}
    .meeting-source-state{display:inline-flex!important;width:max-content;max-width:100%;padding:3px 7px;border-radius:999px;font-size:8px!important;font-weight:900;letter-spacing:.04em}
    .meeting-source-state.ok{color:#70f1c9;background:rgba(26,120,95,.20);border:1px solid rgba(62,218,179,.28)}
    .meeting-source-state.warn{color:#ffd37b;background:rgba(126,82,15,.19);border:1px solid rgba(255,190,72,.30)}
    .meeting-source-state.bad{color:#ff9cab;background:rgba(137,30,48,.18);border:1px solid rgba(255,92,116,.28)}
    .meeting-original{padding:9px 10px;border-radius:10px;background:rgba(3,14,26,.58);border:1px dashed rgba(105,173,207,.18);font-size:9px;color:#9fb4c6;line-height:1.45}
    .meeting-original summary{cursor:pointer;color:#78cce8;font-weight:800}
    .meeting-consensus{padding:12px;border-radius:12px;border:1px solid rgba(56,222,186,.22);background:rgba(10,76,65,.14)}
    .meeting-consensus-label{font-size:8px;color:#79ad9f;text-transform:uppercase;letter-spacing:.09em}
    .meeting-consensus-text{margin-top:5px;color:#78f2cf;font-size:12px;line-height:1.45;font-weight:900}
    .meeting-votes{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
    .meeting-vote{font-size:8px;font-weight:850;border-radius:999px;padding:4px 7px;background:rgba(5,22,37,.72);border:1px solid rgba(93,156,188,.16);color:#b7cada}
    .meeting-vote.ok{color:#70f1c9;border-color:rgba(62,218,179,.25)}
    .meeting-vote.warn{color:#ffd37b;border-color:rgba(255,190,72,.25)}
    .meeting-vote.bad{color:#ff9cab;border-color:rgba(255,92,116,.25)}
    .meeting-agents-v5{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
    .meeting-member{padding:10px;border-radius:11px;background:rgba(2,15,29,.70);border:1px solid rgba(76,165,210,.14);min-width:0}
    .meeting-member-head{display:flex;align-items:flex-start;justify-content:space-between;gap:7px;margin-bottom:6px}
    .meeting-member-head b{font-size:10px;color:#c9f4ff}
    .meeting-stance{flex:0 0 auto;font-size:7px;font-weight:900;letter-spacing:.04em;padding:3px 6px;border-radius:999px;border:1px solid rgba(91,157,188,.20);color:#a9bfd0}
    .meeting-stance.ok{color:#6bf0c6;border-color:rgba(62,218,179,.30);background:rgba(20,112,88,.17)}
    .meeting-stance.warn{color:#ffd27a;border-color:rgba(255,190,72,.30);background:rgba(120,79,16,.16)}
    .meeting-stance.bad{color:#ff9aaa;border-color:rgba(255,92,116,.30);background:rgba(130,27,45,.16)}
    .meeting-stance.info{color:#75d9f8;border-color:rgba(81,192,230,.30);background:rgba(18,89,117,.16)}
    .meeting-member p{margin:0;color:#a7bac9;font-size:9px;line-height:1.5;overflow-wrap:anywhere}
    .meeting-member small{display:block;margin-top:5px;color:#6f879a;font-size:8px}
    @media(max-width:760px){.meeting-source-strip,.meeting-agents-v5{grid-template-columns:1fr}.meeting-topic-title{font-size:13px}}
  `;
  document.head.appendChild(style);
})();

const meetingV5NewsBase = news;
news = function(){
  if(Array.isArray(S.news?.items) && S.news.items.length){
    return S.news.items.slice(0,12).map(x=>({
      urgency:x.urgency||'MEDIUM',
      importance:Number(x.importance||0),
      title:x.title_tr||'Brian için önemli gelişme',
      summary:x.summary_tr||'',
      time:x.observed_at,
      publishedAt:x.published_at||null,
      source:x.source_id||'Bilinmeyen kaynak',
      sourceTrust:x.source_trust_class||'UNKNOWN',
      uri:x.provenance_uri||null,
      asset:x.primary_asset||null,
      original:x.original_claim||'',
      eventKind:x.event_kind||null,
      entityIds:Array.isArray(x.entity_ids)?x.entity_ids:[],
      narrativeIds:Array.isArray(x.narrative_ids)?x.narrative_ids:[]
    }));
  }
  return meetingV5NewsBase();
};

function meetingV5Hostname(value){
  try{return new URL(String(value||'')).hostname.replace(/^www\./,'').toLowerCase()}catch{return''}
}
function meetingV5Publisher(item){
  const host=meetingV5Hostname(item?.uri);
  if(host.endsWith('bbc.co.uk')||host.endsWith('bbc.com'))return'BBC';
  if(host.endsWith('reuters.com'))return'Reuters';
  if(host.endsWith('apnews.com'))return'Associated Press';
  if(host.endsWith('nytimes.com'))return'The New York Times';
  if(host.endsWith('coindesk.com'))return'CoinDesk';
  if(host.endsWith('bloomberg.com'))return'Bloomberg';
  if(host)return host;
  const source=String(item?.source||'').trim();
  return source==='public-rss'?'Harici haber akışı':source||'Bilinmeyen yayıncı';
}
function meetingV5SafeUrl(value){
  try{const u=new URL(String(value||''));return ['http:','https:'].includes(u.protocol)?u.href:''}catch{return''}
}
function meetingV5SourceAssessment(item){
  const sources=v4Array(V4.autonomy?.source_library);
  const wantedHost=meetingV5Hostname(item?.uri);
  const wantedSource=String(item?.source||'').toLowerCase();
  return sources.find(src=>{
    const srcHost=meetingV5Hostname(src.canonical_uri);
    const ids=[src.source_id,src.provider].map(v=>String(v||'').toLowerCase());
    return (wantedHost&&srcHost===wantedHost)||(wantedSource&&ids.includes(wantedSource));
  })||null;
}
function meetingV5Evidence(item){
  if(!item)return{verified:false,label:'OLAY YOK',tone:'info',trust:null,decisionEligible:false};
  const lib=meetingV5SourceAssessment(item),assessment=lib?.assessment||{};
  const trust=Number.isFinite(Number(assessment.trust_score))?Number(assessment.trust_score):null;
  const decisionEligible=assessment.eligible_for_decision_evidence===true;
  const trustClass=String(item.sourceTrust||'UNKNOWN');
  const explicitlyTrusted=/PRIMARY|OFFICIAL|REGULATOR|EXCHANGE|FILING|VERIFIED/i.test(trustClass)&&!/UNVERIFIED/i.test(trustClass);
  if(/UNVERIFIED|DISCOVERY/i.test(trustClass))return{verified:false,label:'DOĞRULAMA BEKLİYOR',tone:'warn',trust,decisionEligible,library:lib};
  if(explicitlyTrusted||decisionEligible)return{verified:true,label:'KAYNAK DOĞRULANDI',tone:'ok',trust,decisionEligible,library:lib};
  return{verified:false,label:'KAYNAK İNCELENİYOR',tone:'warn',trust,decisionEligible,library:lib};
}
function meetingV5FreshEvent(){
  const rank={CRITICAL:3,HIGH:2,MEDIUM:1};
  return news().filter(n=>{
    const sec=ageSec(n.time);
    return sec!=null&&sec<=3600&&['CRITICAL','HIGH'].includes(String(n.urgency));
  }).sort((a,b)=>(rank[b.urgency]||0)-(rank[a.urgency]||0)||(Number(b.importance||0)-Number(a.importance||0))||(Date.parse(b.time)-Date.parse(a.time)))[0]||null;
}
function meetingV5AlphaStance(d){
  if(!d)return{status:'KANIT BEKLİYOR',tone:'warn',group:'wait',text:'ALPHA henüz bu olay için kullanılabilir karar kanıtı üretmedi.'};
  const action=String(d.action||'WAIT');
  const asset=String(d.asset_id||'').replace('crypto:','')||'varlık';
  const score=Number(d.evidence_score||0);
  if(action==='OPEN_LONG'||action==='OPEN_SHORT')return{status:'KOŞULLU ONAY',tone:'ok',group:'support',text:`${asset} için ${act(action)} sinyali var; kanıt ${score.toFixed(2)}. Hazine ve maliyet kapıları ayrıca geçilmeli.`};
  if(action==='VETO')return{status:'REDDETTİ',tone:'bad',group:'object',text:`${asset} için ALPHA veto verdi; kanıt ${score.toFixed(2)}.`};
  return{status:'İŞLEMİ ONAYLAMADI',tone:'warn',group:'wait',text:`${asset} için ${act(action)}; kanıt ${score.toFixed(2)}. Haber tek başına işlem sebebi sayılmıyor.`};
}
function meetingV5TreasuryStance(t){
  if(!t)return{status:'KASA OKUNUYOR',tone:'warn',group:'wait',text:'Hazine durumu henüz toplantıya ulaşmadı.'};
  const gate=Boolean(t.promotion_gate_open);
  return{status:gate?'GATE AÇIK':'GATE KAPALI',tone:gate?'ok':'warn',group:gate?'support':'wait',text:`Özsermaye ${money(t.equity_usd)} · nakit ${money(t.cash_usd)} · kullanım ${pct(t.deployment_pct)}. ${gate?'Shadow işlem için sermaye kapısı uygun.':'Yeni pozisyon için sermaye/promotion kapısı açılmadı.'}`};
}

meetingSnapshotV4 = function(){
  const rows=moduleRows();
  const bad=rows.filter(r=>r.state==='bad');
  const event=meetingV5FreshEvent();
  const evidence=meetingV5Evidence(event);
  const d=S.control?.alpha_v2?.decisions?.[0]||null;
  const t=V4.autonomy?.treasury||S.treasury?.snapshot||null;
  const alpha=meetingV5AlphaStance(d),treasury=meetingV5TreasuryStance(t);
  let mode='routine',active=false,trigger='Rutin operasyon toplantısı',decision='İZLE — olağan akış devam ediyor.';
  if(bad.length){
    mode='technical';active=true;
    trigger=`Teknik alarm: ${bad.map(x=>x.name).join(', ')}`;
    decision='GÜVENLİ DURUŞ — yeni risk alınmıyor; sorun izole edilip taze sistem kanıtı bekleniyor.';
  }else if(event){
    mode='event';active=true;
    trigger=evidence.verified?`Doğrulanmış gelişme masada: ${event.title}`:`İzlenen gelişme — doğrulama sürüyor: ${event.title}`;
    if(!evidence.verified)decision='BEKLE / KAYNAĞI DOĞRULA — keşif haberi tek başına işlem kanıtı değildir.';
    else if(alpha.group!=='support')decision='BEKLE / KANIT TOPLA — haber doğrulandı; ALPHA henüz işlemi doğrulamıyor.';
    else if(treasury.group!=='support')decision=`ALPHA ${act(d.action)} sinyali verdi; Hazine gate kapalı olduğu için işlem yok.`;
    else decision=`KOŞULLU SHADOW ONAY — ALPHA ${act(d.action)} sinyali ve Hazine gate uygun; yalnız gölge işlem/maliyet kuralları içinde değerlendir.`;
  }
  return{active,mode,trigger,decision,rows,bad,critical:event,event,evidence,d,t,alpha,treasury};
};

function meetingV5Member(icon,name,status,tone,text,detail='',group='info'){
  return{icon,name,status,tone,text,detail,group};
}
function meetingV5Participants(m){
  const event=m.event,ev=m.evidence;
  const publisher=meetingV5Publisher(event);
  const worldSummary=S.world?.summary||{};
  const sourceMember=event?meetingV5Member('📰','Kaynak Doğrulama',ev.verified?'ONAYLADI':'DOĞRULUYOR',ev.tone,ev.verified?`${publisher} kaynağı karar kanıtı olarak kullanılabilir seviyede.`:`${publisher} haberi henüz tek başına karar kanıtı değil; ikinci bağımsız doğrulama aranıyor.`,`${String(event.sourceTrust||'UNKNOWN').replaceAll('_',' ')}${ev.trust!=null?' · güven '+Math.round(ev.trust*100)+'%':''}`,ev.verified?'support':'wait'):meetingV5Member('📰','Kaynak Doğrulama','RUTİN','info','Aktif olay doğrulaması yok.','','info');
  const worldMember=event?meetingV5Member('🌍','Dünya Gezgini','SUNDU','info',`Haberi ${publisher} üzerinden masaya getirdi; olay ile varlık etkisi arasında bağ arıyor.`,`${worldSummary.narratives??0} anlatı · ${worldSummary.asset_impact_candidates??0} etki adayı`,'info'):meetingV5Member('🌍','Dünya Gezgini','İZLİYOR','info','Dünya akışı taranıyor; toplantıya taşınan taze yüksek öncelikli olay yok.','','info');
  let skeptic;
  if(m.mode==='technical')skeptic=meetingV5Member('🛡','Şüpheci','VETO','bad','Teknik kanıt eksik olduğu için yeni risk alınmasına itiraz ediyor.','Fail-closed güvenlik kuralı','object');
  else if(event&&!ev.verified)skeptic=meetingV5Member('🛡','Şüpheci','İTİRAZ','bad','Tek kaynağa dayanarak işlem açılmasına karşı; karşı kanıt, priced-in etkisi ve ikinci kaynak istiyor.','Kaynak doğrulanmadan trade yok','object');
  else if(event&&m.alpha.group!=='support')skeptic=meetingV5Member('🛡','Şüpheci','BEKLİYOR','warn','Haber doğrulansa bile fiyatın bunu zaten yansıtıp yansıtmadığını ve ALPHA kanıtını bekliyor.','Karşı kanıt taraması sürüyor','wait');
  else skeptic=meetingV5Member('🛡','Şüpheci','KOŞULLU ONAY','ok','Kaynak ve ALPHA kanıtı uyumlu; yine de maliyet ve ters senaryo kontrolü korunuyor.','Veto sebebi görünmüyor','support');
  const dev=m.mode==='technical'?meetingV5Member('👨‍💻','Yazılımcı Brian','TEKNİK İNCELEME','warn','Sorun yazılım/altyapı kaynaklıysa mühendislik kuyruğuna taşır; piyasa kararını kendisi vermez.','Kodlama kanalı ayrı','wait'):meetingV5Member('👨‍💻','Yazılımcı Brian','OTURUM DIŞI','info','Bu olay bir yazılım değişikliği gerektirmiyor; piyasa kararına veya ALPHA oylamasına müdahale etmiyor.','Mühendislik ve trade yetkileri ayrıdır','info');
  const brianTone=m.mode==='technical'||(event&&!ev.verified)?'warn':m.alpha.group==='support'&&m.treasury.group==='support'?'ok':'warn';
  const brianStatus=m.mode==='technical'?'GÜVENLİ DURUŞ':m.alpha.group==='support'&&m.treasury.group==='support'&&ev.verified?'SHADOW ONAY':event?'BEKLE':'İZLE';
  const brian=meetingV5Member('🧠','Brian',brianStatus,brianTone,m.decision,'Nihai toplantı kararı','info');
  return[
    worldMember,
    sourceMember,
    meetingV5Member('α','ALPHA',m.alpha.status,m.alpha.tone,m.alpha.text,'Fiyat/yön/edge kanıtı',m.alpha.group),
    meetingV5Member('◉','Hazine',m.treasury.status,m.treasury.tone,m.treasury.text,'Sermaye ve gate kontrolü',m.treasury.group),
    skeptic,dev,brian
  ];
}
function meetingV5VoteSummary(participants){
  const voters=participants.filter(p=>!['Dünya Gezgini','Yazılımcı Brian','Brian'].includes(p.name));
  return{
    support:voters.filter(p=>p.group==='support').length,
    wait:voters.filter(p=>p.group==='wait').length,
    object:voters.filter(p=>p.group==='object').length
  };
}

renderMeetingV4 = function(){
  injectV4Panels();
  const m=meetingSnapshotV4(),card=$('meetingCard'),badge=card?.querySelector('.badge');
  card?.classList.remove('meeting-alarm','meeting-review');
  if(m.mode==='technical')card?.classList.add('meeting-alarm');
  else if(m.mode==='event')card?.classList.add('meeting-review');
  if(badge){
    if(m.mode==='technical'){badge.textContent='TEKNİK ALARM';badge.className='badge bad';}
    else if(m.mode==='event'&&!m.evidence.verified){badge.textContent='DOĞRULAMA TOPLANTISI';badge.className='badge warn';}
    else if(m.mode==='event'){badge.textContent='OLAY TOPLANTISI';badge.className='badge ok';}
    else{badge.textContent='RUTİN İZLEME';badge.className='badge info';}
  }
  const modalSub=document.querySelector('#meetingModal .modal-top .section-sub');
  if(modalSub)modalSub.textContent='Canlı olay değerlendirmesi · kaynak, ALPHA, risk ve Hazine mutabakatı';
  const brief=$('meetingBrief');if(!brief)return;
  const event=m.event,participants=meetingV5Participants(m),votes=meetingV5VoteSummary(participants),publisher=meetingV5Publisher(event),safeUrl=meetingV5SafeUrl(event?.uri);
  const sourceState=m.evidence||{label:'—',tone:'info',trust:null};
  const published=event?.publishedAt?`${clock(event.publishedAt)} · ${age(event.publishedAt)} önce`:'Yayın zamanı bilinmiyor';
  const topicTitle=m.mode==='technical'?m.trigger:event?event.title:'Rutin sistem değerlendirmesi';
  const topicSub=m.mode==='technical'?'Piyasa toplantısı askıda; önce teknik bütünlük doğrulanıyor.':event?(event.summary||'Gelişme; piyasa etkisi ve karşı kanıt açısından inceleniyor.'):'Yeni yüksek öncelikli olay yok; sistemler olağan akışta.';
  const sourceHtml=event?`<div class="meeting-source-strip">
      <div class="meeting-source-cell"><span>Haberi getiren</span><b>🌍 Dünya Gezgini</b></div>
      <div class="meeting-source-cell"><span>Yayınlayan</span><b>${esc(publisher)}</b></div>
      <div class="meeting-source-cell"><span>Toplama kanalı</span><b>${esc(event.source||'—')}</b></div>
      <div class="meeting-source-cell"><span>Kaynak durumu</span><b><span class="meeting-source-state ${sourceState.tone}">${esc(sourceState.label)}</span></b></div>
      <div class="meeting-source-cell"><span>Yayın zamanı</span><b>${esc(published)}</b></div>
      <div class="meeting-source-cell"><span>Kaynak güveni</span><b>${sourceState.trust!=null?Math.round(sourceState.trust*100)+'%':'Henüz puanlanmadı'}${safeUrl?` · <a href="${esc(safeUrl)}" target="_blank" rel="noopener noreferrer">Kaynağı aç ↗</a>`:''}</b></div>
    </div>${event.original?`<details class="meeting-original"><summary>Orijinal haber başlığı / claim</summary>${esc(event.original)}</details>`:''}`:'';
  brief.innerHTML=`<div class="meeting-brief-v5">
    <div class="meeting-topic"><div class="meeting-topic-title">${m.mode==='technical'?'🚨':'🗣️'} ${esc(topicTitle)}</div><div class="meeting-topic-sub">${esc(topicSub)}</div></div>
    ${sourceHtml}
    <div class="meeting-consensus"><div class="meeting-consensus-label">Toplantı sonucu</div><div class="meeting-consensus-text">${esc(m.decision)}</div><div class="meeting-votes"><span class="meeting-vote ok">ONAY ${votes.support}</span><span class="meeting-vote warn">BEKLE ${votes.wait}</span><span class="meeting-vote bad">İTİRAZ ${votes.object}</span></div></div>
    <div class="meeting-agents-v5">${participants.map(p=>`<div class="meeting-member"><div class="meeting-member-head"><b>${p.icon} ${esc(p.name)}</b><span class="meeting-stance ${p.tone}">${esc(p.status)}</span></div><p>${esc(p.text)}</p>${p.detail?`<small>${esc(p.detail)}</small>`:''}</div>`).join('')}</div>
  </div>`;
};

const meetingV5AnswerBase=answer;
answer=function(q){
  const text=String(q||'').trim(),l=text.toLocaleLowerCase('tr-TR');
  if(text&&(l.includes('toplantı')||l.includes('alarm'))){
    const m=meetingSnapshotV4(),event=m.event,publisher=meetingV5Publisher(event),parts=meetingV5Participants(m),votes=meetingV5VoteSummary(parts);
    const topic=event?`Masadaki gelişmeyi Dünya Gezgini getirdi; yayıncı ${publisher}, kaynak durumu ${m.evidence.label}. `:m.mode==='technical'?`${m.trigger}. `:'Masada taze yüksek öncelikli olay yok. ';
    const msg=`${topic}ALPHA: ${m.alpha.status}. Hazine: ${m.treasury.status}. Mutabakat: ${votes.support} onay, ${votes.wait} bekle, ${votes.object} itiraz. Brian kararı: ${m.decision}`;
    bubble(text,'user');setTimeout(()=>bubble(msg,'brian'),80);return;
  }
  return meetingV5AnswerBase(q);
};

renderMeetingV4();
