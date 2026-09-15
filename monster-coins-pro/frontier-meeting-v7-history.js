'use strict';

(function installMeetingV7History(){
  if(window.__meetingV7HistoryInstalled) return;
  window.__meetingV7HistoryInstalled=true;

  const STORAGE_KEY='brian-meeting-event-history-v7';
  const MAX_EVENTS=80;
  const RETAIN_MS=7*24*60*60*1000;
  const ACTIVE_WINDOWS={CRITICAL:2*60*60*1000,HIGH:60*60*1000};
  const terminal=new Set(['SONUÇLANDI','REDDEDİLDİ','KAPANDI','İPTAL EDİLDİ']);

  const text=(v,fallback='—')=>String(v??'').trim()||fallback;
  const arr=v=>Array.isArray(v)?v:[];
  const html=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const short=(v,n=120)=>{const s=text(v,'');return s.length>n?s.slice(0,n-1)+'…':s};
  const stamp=v=>{const t=Date.parse(String(v||''));return Number.isFinite(t)?t:0};
  const clock=v=>{const t=stamp(v);return t?new Intl.DateTimeFormat('tr-TR',{hour:'2-digit',minute:'2-digit',day:'2-digit',month:'2-digit'}).format(new Date(t)):'—'};
  const ageLabel=v=>{const t=stamp(v);if(!t)return'—';const s=Math.max(0,(Date.now()-t)/1000);return s<60?`${Math.round(s)} sn`:s<3600?`${Math.round(s/60)} dk`:s<86400?`${Math.round(s/3600)} sa`:`${Math.round(s/86400)} gün`};
  const normalizeAsset=v=>text(v,'').replace(/^crypto:/i,'').replace(/[^A-Z0-9]/gi,'').toUpperCase();
  const urgency=v=>['CRITICAL','HIGH'].includes(String(v||'').toUpperCase())?String(v).toUpperCase():'MEDIUM';

  function canonicalUrl(value){
    try{
      const u=new URL(String(value||''));
      u.hash='';
      ['utm_source','utm_medium','utm_campaign','utm_term','utm_content','fbclid','gclid'].forEach(k=>u.searchParams.delete(k));
      const path=u.pathname.replace(/\/+$/,'')||'/';
      return `${u.protocol}//${u.host.toLowerCase()}${path}${u.search}`;
    }catch{return''}
  }
  function stableToken(value){
    let h=2166136261;
    const s=String(value||'');
    for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}
    return (h>>>0).toString(36);
  }
  function eventKey(e){
    const url=canonicalUrl(e?.uri);
    const accession=(url+' '+text(e?.original,'')).match(/\b\d{10}-\d{2}-\d{6}\b/)?.[0]||'';
    if(accession) return `sec:${accession}`;
    const specific=url && !/\/(rss|feed|api|news)?\/?$/i.test(new URL(url).pathname||'/');
    if(specific) return `url:${url}`;
    const when=text(e?.publishedAt||e?.published_at||e?.time||e?.observed_at,'');
    const claim=text(e?.original||e?.original_claim||e?.title,'');
    return `evt:${stableToken([text(e?.source,''),url,when.slice(0,16),claim].join('|'))}`;
  }
  function safeUrl(value){const u=canonicalUrl(value);return /^https?:\/\//.test(u)?u:''}
  function publisher(e){
    try{
      if(typeof meetingV6Publisher==='function') return meetingV6Publisher(e);
      if(typeof meetingV5Publisher==='function') return meetingV5Publisher(e);
    }catch{}
    try{return new URL(String(e?.uri||'')).hostname.replace(/^www\./,'')}catch{}
    return text(e?.source,'Bilinmeyen kaynak');
  }
  function evidence(e){
    try{
      if(typeof meetingV6Evidence==='function') return meetingV6Evidence(e);
      if(typeof meetingV5Evidence==='function') return meetingV5Evidence(e);
    }catch{}
    const tc=String(e?.sourceTrust||'');
    const verified=/PRIMARY|OFFICIAL|REGULATOR|EXCHANGE|FILING|VERIFIED/i.test(tc)&&!/UNVERIFIED/i.test(tc);
    return {verified,label:verified?'KAYNAK DOĞRULANDI':'DOĞRULAMA BEKLİYOR',tone:verified?'ok':'warn'};
  }
  function load(){
    try{
      const raw=JSON.parse(localStorage.getItem(STORAGE_KEY)||'[]');
      return arr(raw).filter(x=>x&&x.key&&Date.now()-Number(x.lastSeen||x.eventTime||0)<RETAIN_MS);
    }catch{return[]}
  }
  function save(records){
    try{
      const clean=records.sort((a,b)=>Number(b.eventTime||0)-Number(a.eventTime||0)).slice(0,MAX_EVENTS);
      localStorage.setItem(STORAGE_KEY,JSON.stringify(clean));
    }catch{}
  }
  function newsNow(){
    try{return arr(typeof news==='function'?news():[])}catch{return[]}
  }
  function currentMeeting(){
    try{return typeof meetingSnapshotV4==='function'?meetingSnapshotV4():null}catch{return null}
  }
  function sourceFields(e){
    const ev=evidence(e);
    return {
      verified:ev?.verified===true,
      sourceLabel:text(ev?.label,ev?.verified?'KAYNAK DOĞRULANDI':'DOĞRULAMA BEKLİYOR'),
      sourceTrust:text(e?.sourceTrust||e?.source_trust_class,'UNKNOWN'),
      sourceScore:Number.isFinite(Number(ev?.trust))?Number(ev.trust):null
    };
  }
  function assetFor(e,m,old){
    return normalizeAsset(e?.asset||e?.primary_asset||m?.d?.asset_id||old?.alphaAsset||old?.asset);
  }
  function relevantActions(asset,eventTime){
    if(!asset) return [];
    const actions=arr(window.V4?.autonomy?.treasury_actions);
    const start=Number(eventTime||0)-5*60*1000,end=Number(eventTime||0)+12*60*60*1000;
    return actions.filter(a=>{
      const aa=normalizeAsset(a?.asset_id||a?.assetId);
      const at=stamp(a?.observed_at||a?.time);
      return aa===asset && (!at||(at>=start&&at<=end));
    }).sort((a,b)=>stamp(b?.observed_at||b?.time)-stamp(a?.observed_at||a?.time));
  }
  function tradeState(asset,eventTime){
    const actions=relevantActions(asset,eventTime);
    if(!actions.length)return null;
    const exit=actions.find(a=>String(a?.kind||'').toUpperCase()==='EXIT');
    if(exit)return{status:'KAPANDI',summary:`Hazine ${asset} pozisyonunu kapattı.`,action:exit};
    const open=actions.find(a=>String(a?.kind||'').toUpperCase()==='OPEN');
    if(open)return{status:'İŞLEM AÇILDI',summary:`Hazine ${asset} için SHADOW pozisyon açtı.`,action:open};
    return null;
  }
  function activeByAge(record){
    const win=ACTIVE_WINDOWS[record.urgency]||ACTIVE_WINDOWS.HIGH;
    return record.eventTime>0 && Date.now()-record.eventTime<=win;
  }
  function deriveStatus(record,isCurrent,m){
    const trade=tradeState(record.alphaAsset||record.asset,record.eventTime);
    if(trade){record.trade=trade.summary;record.tradeKind=text(trade.action?.kind,'');return trade.status}
    if(isCurrent&&m?.mode==='technical')return'TEKNİK BLOKE';
    if(isCurrent&&String(m?.d?.action||'').toUpperCase()==='VETO')return'REDDEDİLDİ';
    if(isCurrent&&!record.verified)return'DOĞRULANIYOR';
    if(isCurrent&&['OPEN_LONG','OPEN_SHORT'].includes(String(m?.d?.action||''))&&m?.t?.promotion_gate_open===true)return'AKSİYON ADAYI';
    if(isCurrent)return'GÖZLEMLENİYOR';
    if(activeByAge(record)) return record.verified?'SIRADA / İZLENİYOR':'SIRADA / DOĞRULANIYOR';
    if(record.status==='İŞLEM AÇILDI'||record.status==='KAPANDI'||record.status==='REDDEDİLDİ')return record.status;
    return'SONUÇLANDI';
  }
  function sync(){
    const old=load(),byKey=new Map(old.map(x=>[x.key,x]));
    const m=currentMeeting(),currentKey=m?.event?eventKey(m.event):null;
    const feed=newsNow().filter(e=>['CRITICAL','HIGH'].includes(urgency(e?.urgency)));
    const now=Date.now();

    feed.forEach(e=>{
      const key=eventKey(e),prev=byKey.get(key)||{};
      const src=sourceFields(e);
      const rec={
        ...prev,
        key,
        urgency:urgency(e?.urgency),
        importance:Number(e?.importance||prev.importance||0),
        title:text(e?.title,prev.title||'Brian için önemli gelişme'),
        summary:text(e?.summary,prev.summary||''),
        original:text(e?.original,prev.original||''),
        eventKind:text(e?.eventKind,prev.eventKind||''),
        source:text(e?.source,prev.source||''),
        publisher:publisher(e),
        uri:safeUrl(e?.uri)||prev.uri||'',
        sourceTrust:src.sourceTrust,
        verified:src.verified,
        sourceLabel:src.sourceLabel,
        sourceScore:src.sourceScore,
        asset:normalizeAsset(e?.asset)||prev.asset||'',
        eventTime:stamp(e?.time||e?.observed_at)||prev.eventTime||now,
        publishedAt:stamp(e?.publishedAt||e?.published_at)||prev.publishedAt||0,
        firstSeen:prev.firstSeen||now,
        lastSeen:now
      };
      const isCurrent=key===currentKey;
      if(isCurrent&&m){
        rec.lastCouncilAt=now;
        rec.councilDecision=text(m.decision,rec.councilDecision||'');
        rec.alphaAction=text(m?.d?.action,rec.alphaAction||'');
        rec.alphaAsset=assetFor(e,m,rec);
        rec.alphaEvidence=Number.isFinite(Number(m?.d?.evidence_score))?Number(m.d.evidence_score):rec.alphaEvidence??null;
        rec.treasuryGate=m?.t?.promotion_gate_open===true;
        rec.treasuryReason=text(m?.t?.promotion_gate_reason||arr(m?.t?.blocked_reasons).join(' · '),rec.treasuryReason||'');
      }
      rec.status=deriveStatus(rec,isCurrent,m);
      rec.lastStatusAt=now;
      byKey.set(key,rec);
    });

    for(const [key,rec] of byKey){
      const isCurrent=key===currentKey;
      if(!feed.some(e=>eventKey(e)===key)){
        rec.status=deriveStatus(rec,isCurrent,m);
        rec.lastStatusAt=now;
      }
    }

    const records=[...byKey.values()].filter(x=>now-Number(x.lastSeen||x.eventTime||0)<RETAIN_MS);
    save(records);
    return records.sort((a,b)=>Number(b.eventTime||0)-Number(a.eventTime||0));
  }
  function isOpen(rec){return !terminal.has(rec.status)&&rec.status!=='SONUÇLANDI'}
  function counts(records){
    const open=records.filter(isOpen),done=records.filter(x=>!isOpen(x));
    return {
      critical:open.filter(x=>x.urgency==='CRITICAL').length,
      high:open.filter(x=>x.urgency==='HIGH').length,
      open:open.length,
      done:done.length
    };
  }
  function tone(rec){
    if(rec.status==='REDDEDİLDİ'||rec.status==='TEKNİK BLOKE')return'bad';
    if(rec.status==='SONUÇLANDI'||rec.status==='KAPANDI')return'ok';
    if(rec.urgency==='CRITICAL')return'bad';
    return'warn';
  }
  function ensureStyle(){
    if(document.getElementById('frontierMeetingV7HistoryStyle'))return;
    const s=document.createElement('style');s.id='frontierMeetingV7HistoryStyle';
    s.textContent=`
      #meetingCard .meeting-v7-summary{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px;margin-top:10px}
      #meetingCard .meeting-v7-stat{padding:7px 8px;border-radius:9px;background:rgba(3,19,31,.68);border:1px solid rgba(77,168,201,.14)}
      #meetingCard .meeting-v7-stat span{display:block;font-size:7px;color:#718da0;text-transform:uppercase;letter-spacing:.06em}
      #meetingCard .meeting-v7-stat b{display:block;font-size:12px;margin-top:3px;color:#eaf9ff}
      #meetingModal .meeting-v7-ledger{margin:10px 0 12px;padding:12px;border-radius:15px;background:rgba(2,14,25,.88);border:1px solid rgba(77,178,216,.16)}
      .meeting-v7-head{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:9px}.meeting-v7-head h3{margin:0;font-size:12px}.meeting-v7-head span{font-size:8px;color:#7693a7}
      .meeting-v7-counts{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:9px}.meeting-v7-pill{font-size:7px;font-weight:900;padding:4px 7px;border-radius:999px;border:1px solid rgba(87,169,202,.18);color:#a9c0cf}.meeting-v7-pill.bad{color:#ff9bac;border-color:rgba(255,91,116,.30)}.meeting-v7-pill.warn{color:#ffd07c;border-color:rgba(255,185,67,.28)}.meeting-v7-pill.ok{color:#76ebc9;border-color:rgba(67,218,181,.28)}
      .meeting-v7-list{display:grid;gap:6px;max-height:280px;overflow:auto;padding-right:2px}.meeting-v7-row{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:8px;align-items:center;padding:8px 9px;border-radius:10px;background:rgba(4,22,35,.72);border:1px solid rgba(76,158,190,.11);cursor:pointer}.meeting-v7-row:hover{border-color:rgba(89,198,234,.28)}.meeting-v7-sev{font-size:7px;font-weight:950;padding:3px 5px;border-radius:999px}.meeting-v7-sev.bad{color:#ff9bac;background:rgba(123,24,42,.18)}.meeting-v7-sev.warn{color:#ffd07c;background:rgba(121,77,15,.18)}.meeting-v7-row b{display:block;font-size:8.5px;color:#dff4fb;line-height:1.35}.meeting-v7-row small{display:block;margin-top:3px;font-size:7px;color:#718b9e}.meeting-v7-status{font-size:7px;font-weight:950;text-align:right;max-width:105px}.meeting-v7-status.bad{color:#ff90a2}.meeting-v7-status.warn{color:#ffd078}.meeting-v7-status.ok{color:#72ecc9}
      .meeting-v7-detail{margin-top:9px;padding:10px;border-radius:11px;background:rgba(5,29,43,.72);border:1px solid rgba(85,185,219,.16)}.meeting-v7-detail-top{display:flex;gap:8px;justify-content:space-between;align-items:flex-start}.meeting-v7-detail h4{margin:0;font-size:10px;color:#eefbff;line-height:1.4}.meeting-v7-detail-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px;margin-top:8px}.meeting-v7-k{padding:7px;border-radius:8px;background:rgba(1,15,27,.62)}.meeting-v7-k span{display:block;font-size:6.5px;color:#6e8799;text-transform:uppercase}.meeting-v7-k b{display:block;margin-top:3px;font-size:8px;color:#d8edf5;overflow-wrap:anywhere}.meeting-v7-detail p{font-size:8px;line-height:1.5;color:#9db1bf;margin:8px 0 0}.meeting-v7-detail a{color:#6edbf8;text-decoration:none}
      .meeting-v7-section-label{margin:10px 0 5px;font-size:7px;font-weight:950;letter-spacing:.08em;color:#7896a9;text-transform:uppercase}
      @media(max-width:760px){#meetingCard .meeting-v7-summary,.meeting-v7-detail-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.meeting-v7-row{grid-template-columns:auto minmax(0,1fr)}.meeting-v7-status{grid-column:2;text-align:left;max-width:none}}
    `;
    document.head.appendChild(s);
  }
  function renderCard(records){
    const card=document.getElementById('meetingCard');if(!card)return;
    const c=counts(records),badge=card.querySelector('.badge');
    if(badge){
      badge.textContent=`KRİTİK ${c.critical} · YÜKSEK ${c.high}`;
      badge.className=`badge ${c.critical?'bad':c.high?'warn':'info'}`;
    }
    let summary=document.getElementById('meetingV7CardSummary');
    if(!summary){
      summary=document.createElement('div');summary.id='meetingV7CardSummary';summary.className='meeting-v7-summary';
      const btn=document.getElementById('openMeeting');if(btn)btn.parentNode.insertBefore(summary,btn);else card.appendChild(summary);
    }
    summary.innerHTML=`
      <div class="meeting-v7-stat"><span>Kritik</span><b>${c.critical}</b></div>
      <div class="meeting-v7-stat"><span>Yüksek</span><b>${c.high}</b></div>
      <div class="meeting-v7-stat"><span>İzlenen</span><b>${c.open}</b></div>
      <div class="meeting-v7-stat"><span>Sonuçlanan</span><b>${c.done}</b></div>`;
    const strip=document.getElementById('meetingV6SignalStrip');
    if(strip){
      strip.className=`meeting-v6-signal ${c.critical?'red':c.high?'orange':'calm'}`;
      strip.innerHTML=`<span class="meeting-v6-beacon"></span><span>Olay defteri · ${c.open} aktif/izlenen · ${c.done} sonuçlanan</span>`;
    }
  }
  function eventRow(rec){
    const t=tone(rec);
    return `<div class="meeting-v7-row" data-meeting-v7-key="${html(rec.key)}"><span class="meeting-v7-sev ${rec.urgency==='CRITICAL'?'bad':'warn'}">${html(rec.urgency)}</span><div><b>${html(short(rec.title,115))}</b><small>${html(clock(rec.eventTime))} · ${html(rec.publisher||rec.source||'Kaynak')} · ${html(rec.verified?'kaynak doğrulandı':'doğrulama')}</small></div><span class="meeting-v7-status ${t}">${html(rec.status)}</span></div>`;
  }
  function detailHtml(rec){
    if(!rec)return'';
    const src=rec.verified?'DOĞRULANDI':rec.sourceLabel||'DOĞRULANIYOR';
    const alpha=rec.alphaAction?`${rec.alphaAction}${rec.alphaAsset?' · '+rec.alphaAsset:''}`:'Bu olay için saklanmış ALPHA kararı yok';
    const treasury=rec.trade|| (rec.treasuryGate===true?'Gate açıktı':rec.treasuryGate===false?`Gate kapalı${rec.treasuryReason?' · '+short(rec.treasuryReason,70):''}`:'Hazine sonucu kaydedilmedi');
    return `<div class="meeting-v7-detail"><div class="meeting-v7-detail-top"><h4>${html(rec.title)}</h4><span class="meeting-v7-status ${tone(rec)}">${html(rec.status)}</span></div><div class="meeting-v7-detail-grid">
      <div class="meeting-v7-k"><span>Önem</span><b>${html(rec.urgency)} · ${Number(rec.importance||0).toFixed(0)}</b></div>
      <div class="meeting-v7-k"><span>Kaynak</span><b>${html(src)}</b></div>
      <div class="meeting-v7-k"><span>ALPHA son durum</span><b>${html(alpha)}</b></div>
      <div class="meeting-v7-k"><span>Hazine / işlem</span><b>${html(treasury)}</b></div>
      <div class="meeting-v7-k"><span>Olay zamanı</span><b>${html(clock(rec.eventTime))} · ${html(ageLabel(rec.eventTime))} önce</b></div>
      <div class="meeting-v7-k"><span>Yayıncı</span><b>${html(rec.publisher||rec.source||'—')}</b></div>
      <div class="meeting-v7-k"><span>Varlık</span><b>${html(rec.alphaAsset||rec.asset||'Henüz eşlenmedi')}</b></div>
      <div class="meeting-v7-k"><span>Son konsey kaydı</span><b>${rec.lastCouncilAt?html(clock(rec.lastCouncilAt)):'—'}</b></div>
    </div>${rec.summary?`<p>${html(rec.summary)}</p>`:''}${rec.councilDecision?`<p><b style="color:#dff5ff">Konsey son kararı:</b> ${html(rec.councilDecision)}</p>`:''}${rec.original?`<p><b style="color:#dff5ff">Orijinal iddia:</b> ${html(short(rec.original,260))}</p>`:''}${rec.uri?`<p><a href="${html(rec.uri)}" target="_blank" rel="noopener noreferrer">Kaynağı aç ↗</a></p>`:''}</div>`;
  }
  let selectedKey=null;
  function renderLedger(records){
    const modal=document.getElementById('meetingModal'),brief=document.getElementById('meetingBrief');if(!modal||!brief)return;
    let ledger=document.getElementById('meetingV7Ledger');
    if(!ledger){
      ledger=document.createElement('section');ledger.id='meetingV7Ledger';ledger.className='meeting-v7-ledger';
      brief.parentNode.insertBefore(ledger,brief);
      ledger.addEventListener('click',ev=>{
        const row=ev.target.closest('[data-meeting-v7-key]');if(!row)return;
        selectedKey=row.dataset.meetingV7Key||null;
        renderLedger(sync());
      });
    }
    const c=counts(records);
    const open=records.filter(isOpen).slice(0,12),done=records.filter(x=>!isOpen(x)).slice(0,12);
    if(!selectedKey && open[0])selectedKey=open[0].key;
    if(selectedKey && !records.some(x=>x.key===selectedKey))selectedKey=open[0]?.key||done[0]?.key||null;
    const selected=records.find(x=>x.key===selectedKey)||null;
    ledger.innerHTML=`<div class="meeting-v7-head"><div><h3>🗂 Kritik Olay Defteri</h3><span>Yeni haber eskisini silmez; son durum burada kalır.</span></div><span>${records.length} kayıt</span></div>
      <div class="meeting-v7-counts"><span class="meeting-v7-pill bad">KRİTİK ${c.critical}</span><span class="meeting-v7-pill warn">YÜKSEK ${c.high}</span><span class="meeting-v7-pill">İZLENEN ${c.open}</span><span class="meeting-v7-pill ok">SONUÇLANAN ${c.done}</span></div>
      ${detailHtml(selected)}
      <div class="meeting-v7-section-label">Devam eden / gözlenen</div>
      <div class="meeting-v7-list">${open.length?open.map(eventRow).join(''):'<div class="meeting-v7-row"><div><b>Şu anda aktif kritik/yüksek olay yok.</b><small>Yeni olay gelince burada kalıcı kayda alınır.</small></div></div>'}</div>
      <div class="meeting-v7-section-label">Sonuçlanan / arşiv</div>
      <div class="meeting-v7-list">${done.length?done.map(eventRow).join(''):'<div class="meeting-v7-row"><div><b>Henüz sonuçlanan kayıt yok.</b></div></div>'}</div>`;
  }
  function refresh(){
    try{
      ensureStyle();
      const records=sync();
      renderCard(records);
      renderLedger(records);
    }catch(e){console.warn('[meeting-v7-history]',e)}
  }

  const baseRender=typeof renderMeetingV4==='function'?renderMeetingV4:null;
  if(baseRender){
    renderMeetingV4=function(){
      const out=baseRender.apply(this,arguments);
      refresh();
      return out;
    };
  }

  const open=document.getElementById('openMeeting');
  if(open)open.addEventListener('click',()=>setTimeout(refresh,0));
  window.addEventListener('storage',e=>{if(e.key===STORAGE_KEY)refresh()});
  setInterval(()=>{if(!document.hidden)refresh()},15000);
  setTimeout(refresh,0);
})();
