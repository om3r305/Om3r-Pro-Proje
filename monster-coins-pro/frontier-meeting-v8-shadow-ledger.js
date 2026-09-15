'use strict';

(function installMeetingShadowLedgerV8(){
  if(window.__meetingShadowLedgerV8Installed) return;
  window.__meetingShadowLedgerV8Installed=true;

  const ENDPOINT='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-meeting-shadow-ledger';
  const DASHBOARD_KEY='mcp-dashboard-key-v1';
  let lastSignature='';
  let lastSentAt=0;
  let sending=false;

  const text=(v,fallback='')=>String(v??'').trim()||fallback;
  const arr=v=>Array.isArray(v)?v:[];
  const normAsset=v=>text(v).replace(/^crypto:/i,'').replace(/[^A-Z0-9]/gi,'').toUpperCase();
  const finite=v=>Number.isFinite(Number(v))?Number(v):null;
  const iso=v=>{
    const n=typeof v==='number'?v:Date.parse(String(v||''));
    return Number.isFinite(n)?new Date(n).toISOString():null;
  };
  function stableToken(value){
    let h=2166136261;
    const s=String(value||'');
    for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}
    return (h>>>0).toString(36);
  }
  function canonicalUrl(value){
    try{
      const u=new URL(String(value||''));
      u.hash='';
      ['utm_source','utm_medium','utm_campaign','utm_term','utm_content','fbclid','gclid'].forEach(k=>u.searchParams.delete(k));
      return `${u.protocol}//${u.host.toLowerCase()}${u.pathname.replace(/\/+$/,'')||'/'}${u.search}`;
    }catch{return''}
  }
  function eventKey(event,eventId){
    if(eventId) return `event:${eventId}`;
    const url=canonicalUrl(event?.uri);
    const accession=(url+' '+text(event?.original)+' '+text(event?.title)).match(/\b\d{10}-\d{2}-\d{6}\b/)?.[0];
    if(accession) return `sec:${accession}`;
    if(url) return `url:${url}`;
    return `evt:${stableToken([event?.source,event?.time,event?.title].join('|'))}`;
  }
  function snapshot(){
    if(typeof meetingSnapshotV4!=='function'||typeof meetingV6MajorEvents!=='function') return null;
    const m=meetingSnapshotV4();
    const events=arr(meetingV6MajorEvents());
    const event=events[0]||null;
    if(!event) return null;
    const raw=arr(window.S?.news?.items).find(x=>x?.provenance_uri===event.uri&&x?.observed_at===event.time)||null;
    const eventId=raw?.event_id||event?.event_id||null;
    let evidence={};
    try{evidence=typeof meetingV6Evidence==='function'?meetingV6Evidence(event):{}}catch{}
    let view={};
    try{
      view=window.BrianCouncilModel?.project
        ? BrianCouncilModel.project({event,decision:m?.d,treasury:m?.t,evidence,rows:m?.rows})
        : {};
    }catch{}
    const action=text(m?.d?.action).toUpperCase();
    const gate=m?.t?.promotion_gate_open===true;
    let status='GÖZLEMLENİYOR';
    if(m?.mode==='technical') status='TEKNİK BLOKE';
    else if(action==='VETO') status='REDDEDİLDİ';
    else if(['OPEN_LONG','OPEN_SHORT'].includes(action)&&gate) status='AKSİYON ADAYI';
    const publisher=(()=>{
      try{return typeof meetingV6Publisher==='function'?meetingV6Publisher(event):new URL(event.uri).hostname.replace(/^www\./,'')}catch{return text(event.source,'Bilinmeyen kaynak')}
    })();
    const sourceVerified=evidence?.verified===true||/PRIMARY|OFFICIAL|REGULATOR|EXCHANGE|FILING|VERIFIED/i.test(text(event?.sourceTrust))&&!/UNVERIFIED/i.test(text(event?.sourceTrust));
    const councilDecision=[text(view?.title),text(view?.reason),view?.next?`Sonraki koşul: ${text(view.next)}`:''].filter(Boolean).join('. ');
    const eventTime=iso(event?.time||event?.observed_at)||new Date().toISOString();
    return {
      event_key:eventKey(event,eventId),
      event_time:eventTime,
      first_seen_at:eventTime,
      urgency:['CRITICAL','HIGH','MEDIUM','LOW'].includes(text(event?.urgency).toUpperCase())?text(event.urgency).toUpperCase():'HIGH',
      importance:finite(event?.importance),
      title:text(event?.title,'Brian için önemli gelişme'),
      summary:text(event?.summary),
      original_claim:text(event?.original||event?.original_claim),
      event_kind:text(event?.eventKind),
      source:text(event?.source),
      publisher,
      source_uri:canonicalUrl(event?.uri),
      source_trust:text(event?.sourceTrust||event?.source_trust_class),
      source_verified:sourceVerified,
      asset:normAsset(event?.asset||event?.primary_asset),
      alpha_asset:normAsset(m?.d?.asset_id),
      alpha_action:action,
      alpha_evidence:finite(m?.d?.evidence_score),
      council_decision:councilDecision,
      treasury_gate:typeof m?.t?.promotion_gate_open==='boolean'?m.t.promotion_gate_open:null,
      treasury_reason:text(m?.t?.promotion_gate_reason||arr(m?.t?.blocked_reasons).join(' · ')),
      status,
      payload:{
        event_id:eventId,
        observed_at:new Date().toISOString(),
        evidence,
        council:{
          title:text(view?.title),
          reason:text(view?.reason),
          next:text(view?.next),
          linked:view?.linked===true,
          blocked:view?.blocked===true,
          ready:finite(view?.ready),
          gates:arr(view?.gates).map(g=>({id:g?.id,name:g?.name,state:g?.state,text:g?.text,need:g?.need}))
        },
        alpha:m?.d||null,
        treasury:m?.t||null,
        shadow_only:true,
        live_execution:false
      }
    };
  }
  function markUi(ok){
    const modal=document.getElementById('meetingModal');
    if(!modal) return;
    let badge=document.getElementById('meetingShadowLedgerV8Badge');
    if(!badge){
      badge=document.createElement('div');
      badge.id='meetingShadowLedgerV8Badge';
      badge.style.cssText='margin:8px 0 0;font:800 8px/1.3 system-ui;letter-spacing:.06em;color:#7ddfc8;opacity:.82';
      const room=modal.querySelector('.room');
      if(room) room.parentNode.insertBefore(badge,room);
    }
    badge.textContent=ok?'KALICI SHADOW TUTANAĞI · SENKRON':'KALICI SHADOW TUTANAĞI · BAĞLANTI BEKLİYOR';
    badge.style.color=ok?'#7ddfc8':'#f0bd68';
  }
  async function sync(force=false){
    if(sending) return;
    const key=localStorage.getItem(DASHBOARD_KEY)||'';
    if(!key){markUi(false);return}
    const record=snapshot();
    if(!record) return;
    const signature=JSON.stringify([
      record.event_key,record.status,record.alpha_action,record.alpha_asset,
      record.alpha_evidence,record.treasury_gate,record.treasury_reason,record.council_decision
    ]);
    if(!force&&signature===lastSignature&&Date.now()-lastSentAt<60000) return;
    sending=true;
    try{
      const r=await fetch(ENDPOINT,{
        method:'POST',
        headers:{'content-type':'application/json','x-brian-dashboard-key':key},
        body:JSON.stringify({action:'upsert',record}),
        cache:'no-store'
      });
      if(!r.ok) throw new Error(`HTTP ${r.status}`);
      const out=await r.json().catch(()=>({}));
      if(out?.status!=='UPSERTED') throw new Error(text(out?.status,'INVALID_RESPONSE'));
      lastSignature=signature;
      lastSentAt=Date.now();
      markUi(true);
      window.__meetingShadowLedgerLast=out.record||null;
    }catch(e){
      console.warn('[meeting-shadow-ledger-v8]',e);
      markUi(false);
    }finally{sending=false}
  }

  const modal=document.getElementById('meetingModal');
  modal?.addEventListener('click',()=>setTimeout(()=>sync(true),0),{passive:true});
  new MutationObserver(()=>{if(modal?.classList.contains('show'))setTimeout(()=>sync(true),0)})
    .observe(modal||document.body,{attributes:true,attributeFilter:['class']});
  setInterval(()=>{if(!document.hidden)sync(false)},15000);
  setTimeout(()=>sync(true),1200);
})();