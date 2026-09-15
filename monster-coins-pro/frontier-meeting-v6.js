'use strict';

/* Brian Meeting V6 — one incident, one roundtable, one evidence-backed decision. DIP untouched. */
(function installMeetingV6(){
  if(document.getElementById('frontierMeetingV6Style')) return;
  const style=document.createElement('style');
  style.id='frontierMeetingV6Style';
  style.textContent=`
    #meetingCard{position:relative;isolation:isolate;overflow:visible!important;transition:border-color .25s ease,box-shadow .25s ease,transform .25s ease}
    #meetingCard.meeting-v6-red,#meetingCard.meeting-v6-orange{z-index:2}
    #meetingCard.meeting-v6-red:before,#meetingCard.meeting-v6-red:after,#meetingCard.meeting-v6-orange:before,#meetingCard.meeting-v6-orange:after{content:"";position:absolute;pointer-events:none;border-radius:18px;inset:-3px;z-index:-1}
    #meetingCard.meeting-v6-red:before{border:2px solid rgba(255,50,78,.92);box-shadow:0 0 16px rgba(255,40,65,.62),0 0 42px rgba(255,25,50,.34);animation:meetingV6RedPulse 1.05s ease-in-out infinite}
    #meetingCard.meeting-v6-red:after{inset:-10px;border:1px solid rgba(255,55,78,.38);animation:meetingV6Halo 1.05s ease-out infinite}
    #meetingCard.meeting-v6-orange:before{border:2px solid rgba(255,155,44,.92);box-shadow:0 0 15px rgba(255,135,31,.54),0 0 38px rgba(255,119,19,.28);animation:meetingV6OrangePulse 1.45s ease-in-out infinite}
    #meetingCard.meeting-v6-orange:after{inset:-9px;border:1px solid rgba(255,160,45,.34);animation:meetingV6Halo 1.45s ease-out infinite}
    @keyframes meetingV6RedPulse{0%,100%{box-shadow:0 0 8px rgba(255,40,65,.38),0 0 22px rgba(255,25,50,.18);opacity:.72}50%{box-shadow:0 0 24px rgba(255,58,80,.92),0 0 62px rgba(255,25,50,.52);opacity:1}}
    @keyframes meetingV6OrangePulse{0%,100%{box-shadow:0 0 8px rgba(255,140,31,.30),0 0 22px rgba(255,119,19,.14);opacity:.70}50%{box-shadow:0 0 22px rgba(255,164,55,.86),0 0 56px rgba(255,119,19,.40);opacity:1}}
    @keyframes meetingV6Halo{0%{transform:scale(.985);opacity:.72}100%{transform:scale(1.025);opacity:0}}
    .meeting-v6-signal{margin-top:10px;padding:8px 10px;border-radius:10px;font-size:9px;font-weight:900;line-height:1.4;display:flex;align-items:center;gap:8px;cursor:pointer}
    .meeting-v6-signal.red{color:#ffd7dd;background:rgba(128,20,39,.28);border:1px solid rgba(255,68,92,.46)}
    .meeting-v6-signal.orange{color:#ffe0ae;background:rgba(127,72,11,.25);border:1px solid rgba(255,163,55,.42)}
    .meeting-v6-signal.calm{color:#9debdc;background:rgba(19,89,76,.16);border:1px solid rgba(64,214,183,.22)}
    .meeting-v6-beacon{width:9px;height:9px;border-radius:50%;flex:0 0 auto;background:currentColor;box-shadow:0 0 12px currentColor}.meeting-v6-signal.red .meeting-v6-beacon,.meeting-v6-signal.orange .meeting-v6-beacon{animation:meetingV6Beacon .9s ease-in-out infinite alternate}@keyframes meetingV6Beacon{to{transform:scale(1.5);opacity:.45}}

    #meetingModal .modal-card{width:min(1540px,96vw)!important;max-width:none!important;max-height:94vh!important;overflow:auto!important;background:radial-gradient(circle at 50% -5%,rgba(11,53,76,.95),rgba(2,13,23,.98) 38%,rgba(1,7,13,.99) 100%)!important;border-radius:20px!important}
    #meetingModal .modal-card.meeting-v6-modal-red{border-color:rgba(255,63,88,.72)!important;box-shadow:0 0 0 1px rgba(255,63,88,.18),0 0 48px rgba(255,32,57,.22)!important;animation:meetingV6ModalRed 1.25s ease-in-out infinite}
    #meetingModal .modal-card.meeting-v6-modal-orange{border-color:rgba(255,161,52,.64)!important;box-shadow:0 0 0 1px rgba(255,161,52,.14),0 0 42px rgba(255,125,27,.18)!important;animation:meetingV6ModalOrange 1.7s ease-in-out infinite}
    @keyframes meetingV6ModalRed{50%{box-shadow:0 0 0 2px rgba(255,68,91,.40),0 0 65px rgba(255,30,55,.32)}}
    @keyframes meetingV6ModalOrange{50%{box-shadow:0 0 0 2px rgba(255,166,65,.32),0 0 58px rgba(255,126,22,.26)}}
    #meetingModal .modal-top{position:sticky;top:0;z-index:12;background:rgba(2,12,21,.92);backdrop-filter:blur(18px);padding-bottom:10px}
    #meetingModal .room{display:block!important;position:relative!important;min-height:0!important;padding:0!important;margin:0!important;background:none!important;border:0!important}
    #meetingModal .agent-chip,#meetingModal .room-core{display:none!important}
    #meetingBrief{margin:10px 0 12px!important;padding:0!important;background:none!important;border:0!important}

    .meeting-v6-shell{display:grid;gap:12px}
    .meeting-v6-alert{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:12px;align-items:center;padding:12px 14px;border-radius:14px;background:rgba(5,25,39,.82);border:1px solid rgba(74,185,226,.18)}
    .meeting-v6-alert.red{background:linear-gradient(90deg,rgba(122,17,36,.32),rgba(28,15,25,.82));border-color:rgba(255,71,94,.46)}
    .meeting-v6-alert.orange{background:linear-gradient(90deg,rgba(125,71,11,.30),rgba(28,21,13,.82));border-color:rgba(255,166,57,.42)}
    .meeting-v6-alert-icon{font-size:27px;filter:drop-shadow(0 0 12px currentColor)}
    .meeting-v6-alert.red .meeting-v6-alert-icon{color:#ff526b}.meeting-v6-alert.orange .meeting-v6-alert-icon{color:#ffad4f}.meeting-v6-alert.calm .meeting-v6-alert-icon{color:#58e3c0}
    .meeting-v6-alert-title{font-size:15px;font-weight:950;color:#f3fbff;line-height:1.3}.meeting-v6-alert-sub{font-size:9px;color:#8fa8bb;margin-top:4px;line-height:1.45}
    .meeting-v6-severity{font-size:8px;font-weight:950;letter-spacing:.09em;padding:6px 9px;border-radius:999px;border:1px solid rgba(132,180,203,.22);white-space:nowrap}.meeting-v6-alert.red .meeting-v6-severity{color:#ff9baa;border-color:rgba(255,88,108,.44)}.meeting-v6-alert.orange .meeting-v6-severity{color:#ffd189;border-color:rgba(255,174,73,.40)}.meeting-v6-alert.calm .meeting-v6-severity{color:#83efd0;border-color:rgba(70,221,182,.32)}

    .meeting-v6-kpis{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:7px}.meeting-v6-kpi{padding:9px 10px;border-radius:11px;background:rgba(3,17,29,.78);border:1px solid rgba(70,165,202,.13)}.meeting-v6-kpi span{display:block;font-size:7px;text-transform:uppercase;letter-spacing:.08em;color:#708a9e}.meeting-v6-kpi b{display:block;margin-top:4px;color:#e8f9ff;font-size:11px;overflow-wrap:anywhere}.meeting-v6-kpi .ok{color:#65edc1}.meeting-v6-kpi .warn{color:#ffd07c}.meeting-v6-kpi .bad{color:#ff8799}

    .meeting-v6-council{display:grid;grid-template-columns:minmax(190px,1fr) minmax(360px,1.55fr) minmax(190px,1fr);grid-template-areas:"s0 s1 s2" "s7 table s3" "s6 s5 s4";gap:9px;align-items:stretch;padding:12px;border-radius:22px;background:radial-gradient(ellipse at center,rgba(19,78,102,.23),rgba(2,11,19,.45) 60%,rgba(1,7,12,.30));border:1px solid rgba(65,180,220,.13)}
    .meeting-v6-seat{position:relative;padding:10px;border-radius:13px;background:linear-gradient(180deg,rgba(5,25,40,.92),rgba(2,15,26,.94));border:1px solid rgba(75,173,211,.15);min-width:0;box-shadow:0 10px 30px rgba(0,0,0,.16)}
    .meeting-v6-seat:after{content:"";position:absolute;width:22px;height:2px;background:linear-gradient(90deg,transparent,rgba(91,210,246,.35),transparent);left:50%;bottom:-6px;transform:translateX(-50%)}
    .meeting-v6-seat.s0{grid-area:s0}.meeting-v6-seat.s1{grid-area:s1}.meeting-v6-seat.s2{grid-area:s2}.meeting-v6-seat.s3{grid-area:s3}.meeting-v6-seat.s4{grid-area:s4}.meeting-v6-seat.s5{grid-area:s5}.meeting-v6-seat.s6{grid-area:s6}.meeting-v6-seat.s7{grid-area:s7}
    .meeting-v6-seat-head{display:flex;align-items:center;justify-content:space-between;gap:7px}.meeting-v6-person{display:flex;align-items:center;gap:7px;min-width:0}.meeting-v6-avatar{width:28px;height:28px;border-radius:50%;display:grid;place-items:center;background:rgba(17,73,96,.38);border:1px solid rgba(98,208,242,.22);font-size:14px;box-shadow:inset 0 0 12px rgba(72,199,235,.08)}.meeting-v6-person b{font-size:9px;color:#dff8ff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.meeting-v6-seat p{margin:7px 0 0;font-size:8px;color:#9eb2c1;line-height:1.45}.meeting-v6-seat small{display:block;margin-top:5px;font-size:7px;color:#667f93;line-height:1.35}
    .meeting-v6-stance{flex:0 0 auto;font-size:6.5px;font-weight:950;padding:3px 5px;border-radius:999px;border:1px solid rgba(97,151,177,.20);color:#adc0ce}.meeting-v6-stance.ok{color:#6df0c5;border-color:rgba(61,223,178,.32);background:rgba(16,105,80,.16)}.meeting-v6-stance.warn{color:#ffd17c;border-color:rgba(255,186,68,.32);background:rgba(115,76,16,.16)}.meeting-v6-stance.bad{color:#ff8fa0;border-color:rgba(255,87,111,.34);background:rgba(126,25,42,.16)}.meeting-v6-stance.info{color:#7edffb;border-color:rgba(75,195,232,.30);background:rgba(15,81,106,.16)}

    .meeting-v6-table{grid-area:table;align-self:stretch;position:relative;display:flex;flex-direction:column;justify-content:center;min-height:250px;padding:22px 28px;border-radius:48% / 26%;background:radial-gradient(ellipse at 50% 42%,rgba(16,68,91,.96),rgba(5,31,48,.98) 58%,rgba(2,17,29,.98));border:2px solid rgba(87,205,240,.28);box-shadow:inset 0 0 45px rgba(39,180,219,.08),0 24px 55px rgba(0,0,0,.28);text-align:center;overflow:hidden}
    .meeting-v6-table:before{content:"";position:absolute;inset:9px 18px;border-radius:48% / 27%;border:1px solid rgba(105,218,245,.12);pointer-events:none}.meeting-v6-table-label{font-size:7px;letter-spacing:.12em;color:#6f9db1;font-weight:900}.meeting-v6-table-topic{font-size:14px;line-height:1.35;font-weight:950;color:#f3fbff;margin:7px auto 0;max-width:680px}.meeting-v6-table-summary{font-size:8.5px;line-height:1.45;color:#90a9ba;margin:6px auto 0;max-width:680px}.meeting-v6-decision{margin:12px auto 0;padding:9px 12px;border-radius:11px;background:rgba(11,87,72,.18);border:1px solid rgba(58,226,180,.24);color:#75f1cc;font-size:10px;font-weight:950;line-height:1.4;max-width:720px}.meeting-v6-table.red .meeting-v6-decision{background:rgba(123,25,42,.15);border-color:rgba(255,75,98,.28);color:#ffb0bb}.meeting-v6-table.orange .meeting-v6-decision{background:rgba(123,75,15,.15);border-color:rgba(255,169,63,.28);color:#ffdaa0}
    .meeting-v6-votes{display:flex;justify-content:center;gap:6px;flex-wrap:wrap;margin-top:9px}.meeting-v6-vote{font-size:7px;font-weight:900;padding:4px 7px;border-radius:999px;border:1px solid rgba(99,154,180,.20);color:#a9bdcb}.meeting-v6-vote.ok{color:#6bf0c6;border-color:rgba(61,223,178,.30)}.meeting-v6-vote.warn{color:#ffd07a;border-color:rgba(255,186,68,.30)}.meeting-v6-vote.bad{color:#ff8e9f;border-color:rgba(255,87,111,.30)}

    .meeting-v6-lower{display:grid;grid-template-columns:1.15fr .85fr;gap:10px}.meeting-v6-panel{padding:12px;border-radius:14px;background:rgba(2,15,27,.78);border:1px solid rgba(72,166,205,.13)}.meeting-v6-panel h4{margin:0 0 8px;font-size:10px;color:#d9f5ff}.meeting-v6-docket{display:grid;gap:5px}.meeting-v6-docket-row{display:grid;grid-template-columns:95px minmax(0,1fr) auto;gap:7px;align-items:center;padding:7px 8px;border-radius:9px;background:rgba(3,21,35,.72);border:1px solid rgba(73,157,191,.10);font-size:7.5px}.meeting-v6-docket-row span{color:#6f899e;text-transform:uppercase}.meeting-v6-docket-row b{color:#d9edf6;overflow-wrap:anywhere}.meeting-v6-docket-row em{font-style:normal;font-weight:900}.meeting-v6-docket-row em.ok{color:#67edc1}.meeting-v6-docket-row em.warn{color:#ffd07b}.meeting-v6-docket-row em.bad{color:#ff8999}
    .meeting-v6-agenda{display:grid;gap:5px}.meeting-v6-agenda-item{padding:7px 8px;border-radius:9px;background:rgba(3,21,35,.72);border:1px solid rgba(73,157,191,.10)}.meeting-v6-agenda-item b{display:block;font-size:8px;color:#d9edf6;line-height:1.35}.meeting-v6-agenda-item span{display:block;margin-top:3px;font-size:7px;color:#72899c}.meeting-v6-empty{font-size:8px;color:#7890a2;padding:9px;border:1px dashed rgba(102,151,175,.20);border-radius:9px}
    .meeting-v6-transcript{display:grid;gap:5px;margin-top:8px}.meeting-v6-line{display:grid;grid-template-columns:110px minmax(0,1fr);gap:8px;padding:7px 8px;border-radius:9px;background:rgba(3,18,30,.68);font-size:7.5px;line-height:1.4}.meeting-v6-line b{color:#84dff6}.meeting-v6-line span{color:#9dafbd}

    @media(max-width:1100px){.meeting-v6-kpis{grid-template-columns:repeat(3,1fr)}.meeting-v6-council{grid-template-columns:1fr 1.35fr 1fr}.meeting-v6-table{padding:18px;min-height:270px}}
    @media(max-width:820px){.meeting-v6-alert{grid-template-columns:auto 1fr}.meeting-v6-severity{grid-column:1/-1;width:max-content}.meeting-v6-kpis{grid-template-columns:repeat(2,1fr)}.meeting-v6-council{grid-template-columns:1fr 1fr;grid-template-areas:"table table" "s0 s1" "s2 s3" "s4 s5" "s6 s7"}.meeting-v6-table{border-radius:20px;min-height:230px}.meeting-v6-lower{grid-template-columns:1fr}}
    @media(max-width:520px){#meetingModal .modal-card{width:98vw!important}.meeting-v6-council{grid-template-columns:1fr;grid-template-areas:"table" "s0" "s1" "s2" "s3" "s4" "s5" "s6" "s7"}.meeting-v6-kpis{grid-template-columns:1fr 1fr}.meeting-v6-table{min-height:260px}.meeting-v6-docket-row{grid-template-columns:75px 1fr}.meeting-v6-docket-row em{grid-column:2}.meeting-v6-line{grid-template-columns:1fr}}
  `;
  document.head.appendChild(style);
})();

function meetingV6Array(v){return Array.isArray(v)?v:[]}
function meetingV6Short(v,n=160){const s=String(v??'');return s.length>n?s.slice(0,n-1)+'…':s}
function meetingV6Class(v){return ['ok','warn','bad','info'].includes(String(v))?String(v):'info'}
function meetingV6Library(){
  const engineering=meetingV6Array(window.ENGINEER?.data?.source_library);
  const autonomy=meetingV6Array(window.V4?.autonomy?.source_library);
  const seen=new Set();
  return [...engineering,...autonomy].filter(x=>{const k=String(x?.source_id||x?.canonical_uri||'');if(!k||seen.has(k))return false;seen.add(k);return true;});
}
function meetingV6SourceAssessment(item){
  const wantedHost=typeof meetingV5Hostname==='function'?meetingV5Hostname(item?.uri):'';
  const wantedSource=String(item?.source||'').toLowerCase();
  return meetingV6Library().find(src=>{
    const srcHost=typeof meetingV5Hostname==='function'?meetingV5Hostname(src?.canonical_uri):'';
    const ids=[src?.source_id,src?.provider].map(v=>String(v||'').toLowerCase());
    return (wantedHost&&srcHost===wantedHost)||(wantedSource&&ids.includes(wantedSource));
  })||null;
}
function meetingV6Evidence(item){
  if(!item)return{verified:false,label:'OLAY YOK',tone:'info',trust:null,decisionEligible:false,library:null};
  const lib=meetingV6SourceAssessment(item),assessment=lib?.assessment||{};
  const trust=Number.isFinite(Number(lib?.trust_score))?Number(lib.trust_score):Number.isFinite(Number(assessment.trust_score))?Number(assessment.trust_score):null;
  const trustClass=String(item.sourceTrust||'UNKNOWN');
  const tier=String(lib?.tier||'');
  const decisionEligible=assessment.eligible_for_decision_evidence===true||tier==='VERIFIED_PRIMARY';
  const explicitlyTrusted=/PRIMARY|OFFICIAL|REGULATOR|EXCHANGE|FILING|VERIFIED/i.test(trustClass)&&!/UNVERIFIED/i.test(trustClass);
  if(/UNVERIFIED|DISCOVERY/i.test(trustClass))return{verified:false,label:'DOĞRULAMA BEKLİYOR',tone:'warn',trust,decisionEligible,library:lib};
  if(explicitlyTrusted||decisionEligible||tier==='VERIFIED_PRIMARY'||tier==='VERIFIED_RESEARCH')return{verified:true,label:'KAYNAK DOĞRULANDI',tone:'ok',trust,decisionEligible,library:lib};
  return{verified:false,label:'KAYNAK İNCELENİYOR',tone:'warn',trust,decisionEligible,library:lib};
}
function meetingV6MajorEvents(){
  const rank={CRITICAL:3,HIGH:2,MEDIUM:1};
  return meetingV6Array(typeof news==='function'?news():[]).filter(n=>{
    const sec=typeof ageSec==='function'?ageSec(n.time):null;
    const u=String(n.urgency||'MEDIUM');
    return sec!=null&&((u==='CRITICAL'&&sec<=7200)||(u==='HIGH'&&sec<=3600));
  }).sort((a,b)=>(rank[String(b.urgency)]||0)-(rank[String(a.urgency)]||0)||(Number(b.importance||0)-Number(a.importance||0))||(Date.parse(String(b.time))-Date.parse(String(a.time))));
}
function meetingV6Severity(m){
  if(m?.mode==='technical')return{key:'red',label:'KIRMIZI MASA',icon:'🚨'};
  if(String(m?.event?.urgency)==='CRITICAL')return{key:'red',label:'KRİTİK OLAY',icon:'🔴'};
  if(String(m?.event?.urgency)==='HIGH')return{key:'orange',label:'YÜKSEK ÖNCELİK',icon:'🟠'};
  return{key:'calm',label:'MASA HAZIR',icon:'🟢'};
}
function meetingV6ResearchMember(m){
  const run=meetingV6Array(window.S?.evolution?.runs)[0]||null;
  const hypotheses=meetingV6Array(window.V4?.autonomy?.hypotheses);
  if(m?.mode==='technical')return{icon:'⚗',name:'Araştırma Lab',status:'KÖK NEDEN',tone:'warn',group:'wait',text:'Teknik alarmın tekrar üretilebilir kanıtını ve kök nedenini arıyor.',detail:run?`${String(run.status||'RUN')} · ${typeof age==='function'?age(run.started_at||run.observed_at):'—'} önce`:'Evolution run kanıtı bekleniyor'};
  if(m?.event)return{icon:'⚗',name:'Araştırma Lab',status:'KARŞI KANIT',tone:'info',group:'info',text:'Olayın geçmiş örneklerini, ters senaryoyu ve tezin bozulacağı koşulları topluyor.',detail:`${hypotheses.length} görünür hipotez${run?` · son run ${String(run.status||'—')}`:''}`};
  return{icon:'⚗',name:'Araştırma Lab',status:'HAZIR',tone:'info',group:'info',text:'Yeni büyük olay gelirse geçmiş örnek ve karşı kanıt paketi hazırlayacak.',detail:`${hypotheses.length} görünür hipotez`};
}
function meetingV6Participants(m){
  let base=[];
  try{base=typeof meetingV5Participants==='function'?meetingV5Participants(m):[]}catch{base=[]}
  const research=meetingV6ResearchMember(m);
  const names=new Set(base.map(x=>x.name));
  if(!names.has('Araştırma Lab'))base.splice(Math.min(4,base.length),0,research);
  return base.slice(0,8);
}
function meetingV6Votes(parts){
  const voters=parts.filter(p=>!['Dünya Gezgini','Yazılımcı Brian','Brian','Araştırma Lab'].includes(String(p.name)));
  return{support:voters.filter(p=>p.group==='support').length,wait:voters.filter(p=>p.group==='wait').length,object:voters.filter(p=>p.group==='object').length};
}
function meetingV6EventImpact(event){
  const assets=[event?.asset,...meetingV6Array(event?.entityIds)].filter(Boolean).map(String);
  return [...new Set(assets)].slice(0,5).join(' · ')||'Varlık etkisi henüz çıkarılıyor';
}
function meetingV6Publisher(item){try{return typeof meetingV5Publisher==='function'?meetingV5Publisher(item):String(item?.source||'—')}catch{return String(item?.source||'—')}}
function meetingV6SafeUrl(value){try{return typeof meetingV5SafeUrl==='function'?meetingV5SafeUrl(value):''}catch{return''}}
function meetingV6TechnicalSummary(m){return meetingV6Array(m?.bad).map(x=>x.name).join(' · ')||'—'}
function meetingV6Freshness(event){const s=typeof ageSec==='function'?ageSec(event?.time):null;return s==null?'—':s<60?`${Math.round(s)} sn`:s<3600?`${Math.round(s/60)} dk`:`${Math.round(s/3600)} sa`}
function meetingV6SourceTone(ev){return ev?.verified?'ok':ev?.tone==='bad'?'bad':'warn'}
function meetingV6Decision(m,severity,ev){
  if(m?.mode==='technical')return'GÜVENLİ DURUŞ — piyasa aksiyonu askıda. Teknik bütünlük geri gelmeden yeni risk yok.';
  if(!m?.event)return'İZLE — büyük olay yok. Konsey beklemede, veri akışı ve risk kapıları canlı.';
  if(!ev?.verified)return'BEKLE / DOĞRULA — büyük olay masada ama kaynak kanıtı tamamlanmadan işlem yok.';
  if(m?.alpha?.group!=='support')return'BEKLE / FİYAT KANITI TOPLA — kaynak doğrulandı; ALPHA henüz yön/edge onayı vermedi.';
  if(m?.treasury?.group!=='support')return'İŞLEM YOK — ALPHA koşullu onay verdi fakat Hazine / promotion gate kapalı.';
  return'KOŞULLU SHADOW ONAY — kaynak + ALPHA + Hazine aynı yönde. Yalnız mevcut maliyet/risk kuralları içinde gölge aksiyon.';
}
function meetingV6Docket(m,ev){
  const d=m?.d||null,t=m?.t||null,rows=meetingV6Array(m?.rows),bad=rows.filter(x=>x.state==='bad'),warn=rows.filter(x=>x.state==='warn');
  const eng=window.ENGINEER?.data?.source_library_stats||{};
  return[
    ['Kaynak',ev?.verified?'Doğrulandı':m?.event?'Doğrulama sürüyor':'Olay yok',ev?.verified?'ok':m?.event?'warn':'info'],
    ['Kaynak güveni',ev?.trust!=null?`${Math.round(Number(ev.trust)*100)}%`:'Puan yok',ev?.verified?'ok':'warn'],
    ['Olay tazeliği',m?.event?meetingV6Freshness(m.event):'—',m?.event&&meetingV6Freshness(m.event).includes('sa')?'warn':'ok'],
    ['ALPHA',d?`${act(d.action)} · ${String(d.asset_id||'').replace('crypto:','')}`:'Karar yok',m?.alpha?.tone||'warn'],
    ['ALPHA kanıtı',d&&Number.isFinite(Number(d.evidence_score))?Number(d.evidence_score).toFixed(2):'—',m?.alpha?.tone||'warn'],
    ['Hazine gate',t?.promotion_gate_open===true?'AÇIK':'KAPALI',t?.promotion_gate_open===true?'ok':'warn'],
    ['Sistem bütünlüğü',bad.length?`${bad.length} hata`:warn.length?`${warn.length} uyarı`:'Temiz',bad.length?'bad':warn.length?'warn':'ok'],
    ['Dünya kütüphanesi',eng.total!=null?`${eng.total} kaynak · ${eng.verified_research??0} verified`:'Canlı kütüphane','info']
  ];
}
function meetingV6Transcript(parts){
  return parts.map(p=>`<div class="meeting-v6-line"><b>${esc(p.icon+' '+p.name)}</b><span><strong>${esc(p.status)}</strong> — ${esc(p.text)}</span></div>`).join('');
}
function meetingV6Seat(p,i){
  const tone=meetingV6Class(p?.tone);
  return `<div class="meeting-v6-seat s${i}"><div class="meeting-v6-seat-head"><div class="meeting-v6-person"><span class="meeting-v6-avatar">${esc(p?.icon||'•')}</span><b>${esc(p?.name||'Katılımcı')}</b></div><span class="meeting-v6-stance ${tone}">${esc(p?.status||'İZLİYOR')}</span></div><p>${esc(p?.text||'')}</p>${p?.detail?`<small>${esc(p.detail)}</small>`:''}</div>`;
}
function meetingV6SignalCard(m,severity,decision){
  const card=$('meetingCard');if(!card)return;
  card.classList.remove('meeting-alarm','meeting-review','meeting-v6-red','meeting-v6-orange');
  if(severity.key==='red')card.classList.add('meeting-v6-red');
  if(severity.key==='orange')card.classList.add('meeting-v6-orange');
  const badge=card.querySelector('.badge');
  if(badge){badge.textContent=severity.label;badge.className=`badge ${severity.key==='red'?'bad':severity.key==='orange'?'warn':'info'}`;}
  let strip=$('meetingV6SignalStrip');
  if(!strip){strip=document.createElement('div');strip.id='meetingV6SignalStrip';const btn=$('openMeeting');if(btn)btn.parentNode.insertBefore(strip,btn);else card.appendChild(strip);strip.addEventListener('click',()=>$('meetingModal')?.classList.add('show'));}
  strip.className=`meeting-v6-signal ${severity.key}`;
  const title=m?.event?.title||m?.trigger||'Konsey hazır';
  strip.innerHTML=`<span class="meeting-v6-beacon"></span><span>${esc(severity.label)} · ${esc(meetingV6Short(title,105))}</span>`;
}

renderMeetingV4 = function(){
  injectV4Panels();
  const m=meetingSnapshotV4();
  const events=meetingV6MajorEvents();
  if(events.length&&(!m.event||Date.parse(String(events[0].time))>Date.parse(String(m.event.time||0)))){
    m.event=events[0];m.critical=events[0];m.evidence=meetingV6Evidence(events[0]);m.mode=m.mode==='technical'?'technical':'event';m.active=true;
    m.alpha=typeof meetingV5AlphaStance==='function'?meetingV5AlphaStance(m.d):m.alpha;
    m.treasury=typeof meetingV5TreasuryStance==='function'?meetingV5TreasuryStance(m.t):m.treasury;
  } else if(m.event){m.evidence=meetingV6Evidence(m.event);}
  const severity=meetingV6Severity(m),ev=m.evidence||meetingV6Evidence(m.event),decision=meetingV6Decision(m,severity,ev),parts=meetingV6Participants(m),votes=meetingV6Votes(parts);
  m.decision=decision;
  meetingV6SignalCard(m,severity,decision);

  const modal=$('meetingModal'),modalCard=modal?.querySelector('.modal-card'),modalTitle=modal?.querySelector('.modal-top .section-title'),modalSub=modal?.querySelector('.modal-top .section-sub');
  modalCard?.classList.remove('meeting-v6-modal-red','meeting-v6-modal-orange');
  if(severity.key==='red')modalCard?.classList.add('meeting-v6-modal-red');
  if(severity.key==='orange')modalCard?.classList.add('meeting-v6-modal-orange');
  if(modalTitle)modalTitle.textContent='🧠 Brian Olay Konseyi / Toplantı Odası';
  if(modalSub)modalSub.textContent='Büyük olay → kaynak doğrulama → karşı kanıt → ALPHA → Hazine → ortak karar';

  const brief=$('meetingBrief'),room=modal?.querySelector('.room');if(!brief||!room)return;
  const publisher=meetingV6Publisher(m.event),safeUrl=meetingV6SafeUrl(m.event?.uri),topicTitle=m.mode==='technical'?m.trigger:m.event?.title||'Büyük olay bekleniyor',topicSummary=m.mode==='technical'?'Piyasa değerlendirmesi askıya alındı; önce sistem bütünlüğü masada.':m.event?(m.event.summary||'Olay; kaynak, fiyat etkisi, ters senaryo ve sermaye açısından değerlendiriliyor.'):'Şu anda konsey çağıracak taze CRITICAL/HIGH olay yok.';
  const docket=meetingV6Docket(m,ev),queue=events.filter(x=>x!==m.event).slice(0,4);
  const observed=m.event?.time?`${clock(m.event.time)} · ${meetingV6Freshness(m.event)} önce`:'—';
  const sourceStatus=ev?.verified?'DOĞRULANDI':m.event?'BEKLİYOR':'—';
  const alphaStatus=m.d?`${act(m.d.action)}${m.d.asset_id?' · '+String(m.d.asset_id).replace('crypto:',''):''}`:'—';
  const treasuryStatus=m.t?`${m.t.promotion_gate_open===true?'GATE AÇIK':'GATE KAPALI'} · ${money(m.t.equity_usd)}`:'—';
  const badCount=meetingV6Array(m.bad).length;

  brief.innerHTML=`<div class="meeting-v6-shell"><div class="meeting-v6-alert ${severity.key}"><div class="meeting-v6-alert-icon">${severity.icon}</div><div><div class="meeting-v6-alert-title">${esc(topicTitle)}</div><div class="meeting-v6-alert-sub">${esc(topicSummary)}</div></div><div class="meeting-v6-severity">${esc(severity.label)}</div></div><div class="meeting-v6-kpis"><div class="meeting-v6-kpi"><span>Olay zamanı</span><b>${esc(observed)}</b></div><div class="meeting-v6-kpi"><span>Kaynak</span><b class="${meetingV6SourceTone(ev)}">${esc(sourceStatus)}</b></div><div class="meeting-v6-kpi"><span>Yayıncı</span><b>${safeUrl?`<a href="${esc(safeUrl)}" target="_blank" rel="noopener noreferrer" style="color:#6edbf8;text-decoration:none">${esc(publisher)} ↗</a>`:esc(publisher)}</b></div><div class="meeting-v6-kpi"><span>ALPHA</span><b class="${meetingV6Class(m.alpha?.tone)}">${esc(alphaStatus)}</b></div><div class="meeting-v6-kpi"><span>Hazine</span><b class="${meetingV6Class(m.treasury?.tone)}">${esc(treasuryStatus)}</b></div><div class="meeting-v6-kpi"><span>Teknik alarm</span><b class="${badCount?'bad':'ok'}">${badCount?badCount+' MODÜL':'YOK'}</b></div></div></div>`;

  room.innerHTML=`<div class="meeting-v6-council">${parts.map((p,i)=>meetingV6Seat(p,i)).join('')}<div class="meeting-v6-table ${severity.key}"><div class="meeting-v6-table-label">KONSEY MASASI · GERÇEK ZAMANLI OLAY ANALİZİ</div><div class="meeting-v6-table-topic">${esc(topicTitle)}</div><div class="meeting-v6-table-summary">${esc(m.event?meetingV6EventImpact(m.event):m.mode==='technical'?meetingV6TechnicalSummary(m):'Olay kuyruğu boş')}</div><div class="meeting-v6-decision">${esc(decision)}</div><div class="meeting-v6-votes"><span class="meeting-v6-vote ok">ONAY ${votes.support}</span><span class="meeting-v6-vote warn">BEKLE ${votes.wait}</span><span class="meeting-v6-vote bad">İTİRAZ ${votes.object}</span></div></div></div><div class="meeting-v6-lower"><section class="meeting-v6-panel"><h4>📋 Karar dosyası / kanıt kapıları</h4><div class="meeting-v6-docket">${docket.map(([k,v,t])=>`<div class="meeting-v6-docket-row"><span>${esc(k)}</span><b>${esc(v)}</b><em class="${meetingV6Class(t)}">${t==='ok'?'GEÇTİ':t==='bad'?'BLOKE':t==='warn'?'BEKLİYOR':'BİLGİ'}</em></div>`).join('')}</div><h4 style="margin-top:10px">🗣 Konsey tutanağı</h4><div class="meeting-v6-transcript">${meetingV6Transcript(parts)}</div></section><section class="meeting-v6-panel"><h4>🛰 Masaya gelecek diğer büyük olaylar</h4><div class="meeting-v6-agenda">${queue.length?queue.map((x,i)=>`<div class="meeting-v6-agenda-item"><b>#${i+2} · ${esc(meetingV6Short(x.title,120))}</b><span>${esc(String(x.urgency||'HIGH'))} · ${esc(meetingV6Publisher(x))} · ${esc(meetingV6Freshness(x))} önce</span></div>`).join(''):'<div class="meeting-v6-empty">Aynı anda masaya alınacak ikinci büyük olay yok.</div>'}</div><h4 style="margin-top:10px">🔒 Karar sınırı</h4><div class="meeting-v6-empty">Toplantı kararı kanıta dayalıdır. Haber tek başına işlem değildir. Kaynak, ALPHA, risk/Hazine ve sistem bütünlüğü kapıları ayrı ayrı geçmeden aksiyon yok. SHADOW ONLY · DIP AYRI.</div></section></div>`;
};

const meetingV6AnswerBase=answer;
answer=function(q){
  const text=String(q||'').trim(),l=text.toLocaleLowerCase('tr-TR');
  if(text&&(l.includes('toplantı')||l.includes('konsey')||l.includes('alarm')||l.includes('büyük olay'))){
    const m=meetingSnapshotV4(),events=meetingV6MajorEvents();
    if(events.length&&!m.event){m.event=events[0];m.evidence=meetingV6Evidence(events[0]);m.mode='event';}
    const sev=meetingV6Severity(m),ev=m.evidence||meetingV6Evidence(m.event),parts=meetingV6Participants(m),votes=meetingV6Votes(parts),decision=meetingV6Decision(m,sev,ev);
    const msg=m.event?`${sev.label}: ${m.event.title}. Kaynak ${ev.verified?'doğrulandı':'doğrulama bekliyor'}. ALPHA ${m.alpha?.status||'kanıt bekliyor'}, Hazine ${m.treasury?.status||'okunuyor'}. Mutabakat ${votes.support} onay / ${votes.wait} bekle / ${votes.object} itiraz. Karar: ${decision}`:`${sev.label}: taze CRITICAL/HIGH olay yok. Karar: ${decision}`;
    bubble(text,'user');setTimeout(()=>bubble(msg,'brian'),80);return;
  }
  return meetingV6AnswerBase(q);
};

renderMeetingV4();
