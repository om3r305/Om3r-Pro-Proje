/* Brian DIP V7 dashboard status bridge.
   SHADOW ONLY. Presentation-only overlay: execution remains owned by the Supabase V7 worker. */
(function(){
  if(typeof renderDip!=='function') return;
  const baseRenderDip=renderDip;
  const el=id=>document.getElementById(id);
  const fmtMoney=v=>{const n=Number(v);return Number.isFinite(n)?new Intl.NumberFormat('tr-TR',{style:'currency',currency:'USD',minimumFractionDigits:2,maximumFractionDigits:2}).format(n):'—';};
  const fmtAge=sec=>{if(!Number.isFinite(sec))return'—';if(sec<60)return`${Math.max(0,Math.round(sec))} sn`;if(sec<3600)return`${Math.round(sec/60)} dk`;return`${(sec/3600).toFixed(sec<7200?1:0)} sa`;};
  const fmtClock=v=>{if(!v)return'—';try{return new Intl.DateTimeFormat('tr-TR',{timeZone:'Europe/Berlin',hour:'2-digit',minute:'2-digit',second:'2-digit'}).format(new Date(v));}catch{return'—';}};

  renderDip=function(data){
    const d=data?.system?.dip||{};
    const snap=d.snapshot||{};
    const runtime=snap?.state?.serverRuntime||{};
    const authoritative=runtime.authoritative===true||String(runtime.authoritative).toLowerCase()==='true';
    if(!authoritative){baseRenderDip(data);return;}

    const active=Boolean(d.session_id)&&d.status!=='PAUSED';
    const snapAt=d.latest_snapshot_at||snap.observed_at||runtime.generated_at||null;
    const parsed=snapAt?Date.parse(snapAt):NaN;
    const snapshotAge=Number.isFinite(parsed)?Math.max(0,(Date.now()-parsed)/1000):NaN;
    const fresh=active&&Number.isFinite(snapshotAge)&&snapshotAge<180;
    const worker=runtime.worker_version||runtime.workerVersion||'brian-dip-server-v7';

    if(el('dipState')){
      el('dipState').textContent=!active?'DURAKLATILDI':fresh?'SUNUCU AKTİF':'SUNUCU GECİKMİŞ';
      el('dipState').className=`cc-kpi-value ${fresh?'ok':active?'warn':'warn'}`;
    }
    if(el('dipStateMeta')) el('dipStateMeta').textContent=d.session_id?`Session ${d.session_id} · ${worker} · karar sahibi Brian V7`:'Aktif DIP session yok';
    if(el('dipEquity')) el('dipEquity').textContent=snap.equity==null?'—':fmtMoney(snap.equity);
    if(el('dipEquityMeta')) el('dipEquityMeta').textContent=snapAt?`Server snapshot ${fmtClock(snapAt)} · ${fmtAge(snapshotAge)} önce`:'Server snapshot yok';
    if(el('dipHeartbeat')) el('dipHeartbeat').textContent=Number.isFinite(snapshotAge)?fmtAge(snapshotAge):'—';
    if(el('dipHeartbeatMeta')) el('dipHeartbeatMeta').textContent=snapAt?`V7 cloud worker son snapshot ${fmtClock(snapAt)}`:'V7 cloud heartbeat bekleniyor';

    const box=el('dipBackgroundNote');
    if(box){
      box.className=fresh?'cc-safe-box':'cc-warning-box';
      box.innerHTML=fresh
        ?'<b>DIP V7 cloud runner aktif:</b> karar, sizing, giriş/çıkış ve risk lifecycle Supabase sunucusunda çalışıyor. Telefonu veya sayfayı kapatsan da SHADOW taraması devam eder; browser yalnız grafik/radar görüntüler.'
        :'<b>DIP V7 server-authoritative:</b> son server snapshot gecikmiş görünüyor. Yeni SHADOW girişler için cloud worker heartbeatini kontrol et.';
    }
  };
})();
