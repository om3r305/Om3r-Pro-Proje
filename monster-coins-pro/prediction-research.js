'use strict';
(() => {
  const host=document.getElementById('opsRooms');if(!host||document.getElementById('predictionResearch'))return;
  const M=window.BrianPredictionResearchModel;
  const E=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num=v=>M.number(v)===null?'—':Number(v).toLocaleString('tr-TR',{maximumFractionDigits:3});
  const at=v=>Number.isFinite(Date.parse(v))?new Date(v).toLocaleString('tr-TR',{timeZone:'Europe/Berlin'}):'Zaman bilinmiyor';
  const panel=document.createElement('section');panel.id='predictionResearch';panel.className='prediction-research';panel.setAttribute('aria-label','Prediction Intelligence ve NIULAI4 araştırması');
  panel.innerHTML='<header><div><small>ARAŞTIRMA / GERÇEK KAYITLAR</small><h3>Prediction Intelligence</h3></div><button type="button" id="predictionRefresh">Araştırmayı yenile</button></header><p>Polymarket gözlemi, kalibrasyon ve NIULAI4 deneyi ayrı ölçümlerdir.</p><div id="predictionState" role="status" aria-live="polite">Durum henüz alınmadı.</div><div id="predictionContent"></div>';
  host.appendChild(panel);
  let data=null,error='',busy=false,visible=false,lastAttempt=0,unauthorized=false;
  const button=panel.querySelector('#predictionRefresh'),body=panel.querySelector('#predictionContent'),state=panel.querySelector('#predictionState');
  function card(s,title,html){return '<article class="prediction-card" data-state="'+(error?'warn':s.tone)+'"><h4>'+title+'</h4><strong>'+E((error?'Eski kayıt · son bilinen durum: ':'')+s.label)+'</strong>'+html+'</article>';}
  const detail=(title,value)=>'<details><summary>'+title+'</summary><pre>'+E(JSON.stringify(value,null,2))+'</pre></details>';
  function render(){
    button.disabled=busy;button.textContent=busy?'Alınıyor…':'Araştırmayı yenile';
    state.textContent=error+(data?' · Son başarılı yanıt: '+at(data.observed_at)+(error?' · Eski kayıt':''):busy?'Durum alınıyor…':'');
    if(!data){body.textContent=unauthorized?'Dashboard kilidini açıp tekrar dene.':'Henüz doğrulanmış durum kaydı alınmadı; sayı veya sonuç üretilmez.';return;}
    const v=M.project(data);if(!v){body.textContent='Durum şeması doğrulanamadı.';return;}
    const p=v.prediction.raw,c=v.calibration.raw,n=v.pilot.raw,available=s=>s.status!=='UNAVAILABLE';
    const pBody=available(v.prediction)?'<dl><dt>Son tur</dt><dd>'+E(at(p.last_cycle_at))+'</dd><dt>Anlık görüntü</dt><dd>'+num(v.counts.snapshots)+'</dd><dt>Son 24 saat</dt><dd>'+num(v.counts.last24h)+'</dd><dt>Sonuçlanan piyasa</dt><dd>'+num(v.counts.resolutions)+'</dd><dt>Cüzdan takibi</dt><dd>'+(v.walletTracking===null?'Bilinmiyor':v.walletTracking?'Açık':'Kapalı')+'</dd></dl>'+(v.counts.resolutions===0?'<p>Henüz sonuçlanan piyasa yok.</p>':'')+detail('Kaynaklar ve veri sağlığı',p):'<p>Gözlem durumu okunamadı.</p>';
    const cBody=available(v.calibration)?'<p>'+num(v.counts.events)+' bağımsız olay · '+num(v.counts.markets)+' piyasa</p>'+(v.calibration.status==='SUFFICIENT'?'<p>Brier: '+num(c.report?.brier)+'</p>':'<p>Güvenilir çıkarım için yeterli olay bekleniyor.</p>')+'<p>Kalibrasyon, kâr veya işlem sinyali değildir.</p>'+detail('Kalibrasyon yöntemi ve kapsamı',c):'<p>Kalibrasyon verisi okunamadı.</p>';
    const report=n.latest_report;
    const nBody=available(v.pilot)?'<p>Son nabız: '+E(at(n.last_heartbeat?.observed_at))+'</p>'+(n.run?'<p>Kural: '+E(n.run.rules_version||'Bilinmiyor')+'</p>':'')+(n.run?.stop_reason?'<p>Durma nedeni: '+E(n.run.stop_reason)+'</p>':'')+(report?'<p>'+E(report.kind==='INTERIM'?'Ara rapor · avantaj kararı verilemez':report.kind==='FINAL'?'Final rapor · inceleme gerekli':'Rapor türü doğrulanamadı')+'</p><p>d = 0: iyimser üst sınır. Cüzdandan bağımsız zamanlama katkısı (H2) test edilmedi.</p>'+detail('Ölçümler, maliyetler ve ret nedenleri',report.summary):'<p>Deney raporu henüz yok.</p>')+'<p>Başlatma için kullanıcı onayı gerekir. Bu ekran deneyi başlatmaz.</p>': '<p>Pilot durumu okunamadı.</p>';
    // Save disclosures by stable card position before replacing the view.
    const opened=[...body.querySelectorAll('details')].map(d=>d.open);
    body.innerHTML=card(v.prediction,'Polymarket gözlemi',pBody)+card(v.calibration,'Kalibrasyon',cBody)+card(v.pilot,'NIULAI4 gözlem deneyi',nBody);
    [...body.querySelectorAll('details')].forEach((d,i)=>{d.open=opened[i]===true});
  }
  async function refresh(force=false){
    if(busy||document.hidden||(!force&&Date.now()-lastAttempt<60000))return;
    let k='';try{k=(localStorage.getItem('mcp-dashboard-key-v1')||'').trim()}catch{}
    if(!k){data=null;unauthorized=true;error='Araştırma için dashboard kilidini aç.';render();return;}
    busy=true;unauthorized=false;error='';lastAttempt=Date.now();render();
    try{
      const r=await fetch('/api/brian/brian-prediction-status',{method:'POST',headers:{'content-type':'application/json','x-brian-dashboard-key':k},body:'{}',cache:'no-store',signal:AbortSignal.timeout(20000)});
      if(r.status===401){data=null;unauthorized=true;throw Error('Dashboard anahtarı doğrulanamadı.');}
      if(!r.ok)throw Error('Araştırma durumu şu an alınamıyor.');
      const next=await r.json();if(!M.project(next))throw Error('Araştırma veri sözleşmesi doğrulanamadı.');data=next;
    }catch(e){error=['TimeoutError','AbortError'].includes(e?.name)?'Araştırma isteği zaman aşımına uğradı.':e?.message||'Bağlantı hatası';}
    finally{busy=false;render();}
  }
  button.onclick=()=>refresh(true);
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible)refresh();},{threshold:0}).observe(panel);
  setInterval(()=>{if(visible)refresh();},60000);
  document.addEventListener('visibilitychange',()=>{if(!document.hidden&&visible)refresh();});
  document.getElementById('unlockBtn')?.addEventListener('click',()=>{if(visible)refresh(true);});
  render();
})();
