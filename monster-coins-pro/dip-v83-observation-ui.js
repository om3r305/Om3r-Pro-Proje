'use strict';
(function(){
  const reasons={NO_SIGNAL:'Şu anda uygun yapı sinyali yok.',TARGET_BELOW_COST:'Hedef mesafesi, modellenen toplam maliyetin 2,5 katı olan giriş eşiğinin altında.',ECONOMIC_RR_TOO_LOW:'Maliyet sonrası kazanç/risk oranı yeterli değil.',CALIBRATING:'Bu koşullar için yeterli bağımsız sonuç henüz yok.',DIRECTION_REFEREE_REJECT:'Zaman dilimleri ve yön değerlendirmesi uyuşmuyor.',COUNTER_STRUCTURE:'İşlem yönü mevcut yapıyla çatışıyor.',RAW_CONVICTION_LOW:'Yapı puanı giriş için düşük.',OPPOSING_MULTI_FLOW:'Emir akışı önerilen yönü desteklemiyor.',EPISODE_ALREADY_TRADED:'Bu fikir daha önce işleme alındı; tekrar giriş kapalı.',THESIS_LOCKED_NO_NEW_STRUCTURE:'Yeni yapısal teyit bekleniyor.',MIN_NOTIONAL_OR_RISK_CAP:'Borsa minimumu veya risk sınırı nedeniyle boyut ayrılamıyor.',CALIBRATION_NO_EDGE:'Geçmiş sonuçlar bu koşullarda yeterli avantaj göstermiyor.',CALIBRATION_UNAVAILABLE:'Kalibrasyon verisine erişilemiyor.',SESSION_PAUSED:'Session duraklatılmış.',OBSERVATION_ONLY:'Yalnız gözlem modu açık.',CLOSED_THIS_RUN:'Bu turda pozisyon kapandı; aynı turda yeni giriş yapılmıyor.',STALE_DATA:'Piyasa verisinin yenilenmesi bekleniyor.',STRUCTURE_LEVEL_INCOMPLETE:'Giriş, hedef ve iptal seviyeleri yeterince net değil.',LABEL_UNCERTAINTY_HIGH:'Belirsiz sonuçların oranı yüksek.'};
  const labels={UNTOUCHED:'Teyit sonrası dokunulmamış',TOUCHED:'Temas edilmiş',SWEPT:'Fitille aşılmış, mum geri kapanmış',CLOSE_CROSSED:'En az bir mum seviyenin ötesinde kapanmış'};
  const pct=v=>v==null||!Number.isFinite(Number(v))?'—':(Number(v)/100).toFixed(3)+'%';
  function showContext(){
    const host=document.getElementById('decisionContext');if(!host)return;
    const t=thesis(),p=position(),sr=runtime();
    if(!t){host.innerHTML='<b>Karar açıklaması</b><p class="note">Sunucudan ilk karar bekleniyor.</p>';return;}
    const veto=Array.isArray(t.veto)?t.veto:[],obs=t.level_observation;
    const fresh=!model.statusError&&sr?.status==='OK'&&sr?.generated_at&&Date.now()-Date.parse(sr.generated_at)<150000;
    const waiting=veto.some(x=>x!=='CALIBRATING')||t.thesis_state==='WAIT';
    const heading=!fresh?'Son kaydedilen karar · veri yenilenmesi bekleniyor':p?'Açık pozisyon yönetiliyor':waiting?'Brian neden bekliyor?':'Giriş koşulları uygun · sunucu kararı bekleniyor';
    const explanation=p?'Mevcut pozisyonun sabit hedefi ve iptal seviyesi takip ediliyor. Aşağıdaki aday değerlendirmesi pozisyon planını değiştirmez.':veto.length?veto.map(v=>reasons[v]||String(v)).join(' '):'Sunucu yeni kararını işlerken pozisyon ve işlem kaydını takip edin.';
    const levels=!p&&obs?.version==='dip-level-observation-v1'&&Array.isArray(obs.levels)?obs.levels.filter(x=>Number.isFinite(Number(x.price))&&Math.abs(Number(x.price)-Number(t.target_price))<1e-6):[];
    const history=p?'Açık pozisyonun planı yukarıda gösteriliyor.':levels.length?levels.map(x=>`${x.tf} · ${px(x.price)} · ${labels[x.status]||'Durum bilinmiyor'} · ${x.origin==='EQUAL_PAIR'?'eş seviye':'pivot'}`).join(' | '):'Bu hedef için geçmiş gözlemi henüz yok; dokunulmamış olduğu varsayılmaz.';
    host.innerHTML=`<div class="title">${esc(heading)}</div><p class="note" style="margin:8px 0;line-height:1.6">${esc(explanation)}</p><div class="thesis-levels"><span>Hedef mesafesi <b>${pct(t.target_distance_bps)}</b></span><span>Modellenen toplam maliyet <b>${pct(t.cost_bps)}</b></span><span>Hedef mesafesi alt sınırı (2,5× maliyet) <b>${pct(t.cost_bps==null?null:Number(t.cost_bps)*2.5)}</b></span><span>Bağımsız kalibrasyon örneği <b>${Math.max(0,num(t.calibration_samples))}</b></span></div><p class="note" style="margin:8px 0;overflow-wrap:anywhere"><b>Seçili hedefin geçmişi:</b> ${esc(history)}</p><small class="note">Hedefe temas, maliyet sonrası kârlı işlem anlamına gelmez. Komisyon varsayımı gerçek hesap tarifesiyle doğrulanmalıdır. Ham yapı puanı başarı olasılığı değildir. Seviye geçmişi yalnız açıklamadır; kırılmış seviye otomatik olarak geçersiz sayılmaz.</small>`;
  }
  const previous=render;
  render=function(){previous();showContext();};
})();
