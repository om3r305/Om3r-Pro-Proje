'use strict';
((root)=>{
 const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const date=v=>v&&Number.isFinite(Date.parse(v))?new Date(v).toLocaleString('tr-TR',{timeZone:'Europe/Berlin'}):'—';
 const fields=['sampled','mature','not_mature','resolved','missing_entry','invalid_outcome','awaiting_auditor','window_open','missing_terminal','missing_cost','cost_known_directional'];
 function aggregate(rows){
  const buckets=new Map();
  for(const r of rows){
   if(![300,900,3600].includes(r.seconds)||fields.some(k=>!Number.isSafeInteger(r[k])||r[k]<0))throw Error('INVALID_COVERAGE');
   if(r.sampled!==r.mature+r.not_mature||r.mature!==r.resolved+r.missing_entry+r.invalid_outcome+r.awaiting_auditor+r.window_open+r.missing_terminal||r.missing_cost+r.cost_known_directional>r.resolved)throw Error('INVALID_COVERAGE_TOTALS');
   const b=buckets.get(r.seconds)||Object.fromEntries(fields.map(k=>[k,0]));
   for(const k of fields)b[k]+=r[k];b.seconds=r.seconds;buckets.set(r.seconds,b);
  }
  return [...buckets.values()].sort((a,b)=>a.seconds-b.seconds);
 }
 function table(rows,asset=false){return `<div class="sc-net-scroll"><table><thead><tr>${asset?'<th>Varlık</th>':''}<th>Ufuk</th><th>Karar</th><th>Olgun</th><th>Sonuç var</th><th>Fiyat eksik</th><th>Denetçi bekliyor</th><th>Pencere açık</th><th>Giriş eksik</th><th>Geçersiz sonuç</th><th>Maliyet eksik*</th></tr></thead><tbody>${rows.map(r=>`<tr>${asset?`<td>${esc(r.asset_id)}</td>`:''}<td>${r.seconds/60} dk</td>${['sampled','mature','resolved','missing_terminal','awaiting_auditor','window_open','missing_entry','invalid_outcome','missing_cost'].map(k=>`<td>${r[k]}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
 function render(value){
  if(value?.status!=='AVAILABLE'||!Array.isArray(value.rows))return '<section class="sc-net"><h3>ALPHA sonuç kapsamı</h3><p class="sc-warning">Kapsam ölçümü alınamadı; eksik veriler sıfır veya tamamlanmış sayılmıyor.</p></section>';
  try{
   if(value.rows.some(r=>!['rolling','prospective'].includes(r.sample)))throw Error('UNKNOWN_SAMPLE');
   const rolling=value.rows.filter(r=>r.sample==='rolling'),prospective=value.rows.filter(r=>r.sample==='prospective');
   const groups=[{key:'rolling',name:'Son 6 saatin en yeni 300 kararı',rows:rolling,truncated:value.rolling_truncated},
     {key:'prospective',name:'Sabit ileriye dönük karar dönemi',rows:prospective,truncated:value.prospective_truncated}];
   const versions=(Array.isArray(value.compiler_versions)?value.compiler_versions:[]).filter(v=>v.sample==='prospective');
   const versionNote=`<p class="sc-muted">İleri dönem karar sürümleri: ${versions.length?versions.map(v=>esc(v.compiler_version)+': '+esc(v.decisions)+' karar').join(' · '):'Henüz kayıt yok'}. ${versions.length>1?'Birden çok sürüm var; tek bir sabit stratejinin testi sayılmaz.':''} Örtüşen kararlar bağımsız örnek sayılmaz; bu dönem bir kapsam testi, strateji üstünlüğü deneyi değildir.</p>`;
   const p=value.protocol||{},at=Date.parse(value.as_of),start=Date.parse(p.starts_at),end=Date.parse(p.ends_at),final=Date.parse(p.final_horizon_at);
   const phase=!Number.isFinite(at)||!Number.isFinite(start)||!Number.isFinite(end)||!Number.isFinite(final)?'Dönem zamanı doğrulanamadı':at<start?'Başlaması bekleniyor':at<end?'Kararlar birikiyor':at<final?'Sonuç ufukları bekleniyor':'Ölçüm dönemi sona erdi · eksikler ayrıca incelenmeli';
   return `<section class="sc-net"><h3>ALPHA sonuç kapsamı · 5 / 15 / 60 dakika</h3><p>Sonucu bulunmayan kararlar da sayılır. Bu tablo fiyat toplamaz, eksik geçmişi doldurmaz ve kârlılık kanıtlamaz.</p>${groups.map(g=>{const totals=aggregate(g.rows);return `<details class="sc-alpha" ${g.key==='rolling'?'open':''}><summary>${g.name}</summary>${g.key==='prospective'?`<p>${esc(phase)}<br>${esc(date(p.starts_at))} → ${esc(date(p.ends_at))} · Berlin<br>Son 60 dk penceresi en geç ${esc(date(p.final_horizon_at))}. Sürüm ${esc(p.id)}</p>`:'<p>Kayan, rastgele olmayan tanı örneğidir; önceki kayan örnekle başarı artışı karşılaştırılmaz.</p>'}${g.truncated?'<p class="sc-warning">Kayıt sınırına ulaşıldı; dönem kapsamı kısmi.</p>':''}${totals.length?table(totals):'<p>Bu dönemde henüz karar kaydı yok; kapsam oranı hesaplanmıyor.</p>'}${g.rows.length?`<details><summary>Varlık ve ufuk ayrımı</summary>${table(g.rows,true)}</details>`:''}</details>`}).join('')}${versionNote}<p class="sc-muted">Olgun kararlar: sonuç var + fiyat eksik + denetçi bekliyor + pencere açık + giriş eksik + geçersiz sonuç. *Maliyet eksik, sonucu bulunan yönlü kararların alt kümesidir; toplama tekrar eklenmez. Ufuklara aynı karar yeniden girer. Son fiyat için yalnızca hedeften sonraki 120 saniye kabul edilir. Kaydın gözlem ve sisteme geliş zamanı rapor anından ileride olamaz. Sonuç var olması, uygulanabilir işlem veya kâr anlamına gelmez.</p><p class="sc-muted">Ölçüm ${esc(date(value.as_of))} · Otomatik terfi veya gerçek emir yok.</p></section>`;
  }catch{return '<section class="sc-net"><h3>ALPHA sonuç kapsamı</h3><p class="sc-warning">Kapsam verisinin tutarlılığı doğrulanamadı; başarı sonucu gösterilmiyor.</p></section>';}
 }
 root.BrianOutcomeCoverage={render,aggregate};
 if(typeof module==='object'&&module.exports)module.exports=root.BrianOutcomeCoverage;
})(typeof window==='object'?window:globalThis);
