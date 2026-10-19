(function(root){
 'use strict';
 const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 function link(uri,label){try{const u=new URL(uri);if(u.protocol!=='https:'||u.username||u.password||u.port||!(/^[a-z][a-z0-9.-]+\.[a-z]{2,}$/i.test(u.hostname))||/(^|\.)(local|localhost|internal|invalid|test)$/i.test(u.hostname))throw Error();u.search='';u.hash='';return '<a style="color:#89d9dd" href="'+esc(u.toString())+'" target="_blank" rel="noopener noreferrer">'+esc(label)+'</a>';}catch{return esc(label);}}
 const time=v=>v&&Number.isFinite(Date.parse(v))?esc(new Date(v).toLocaleString('tr-TR',{timeZone:'Europe/Berlin'})):'—';
 const delta=d=>d&&d.bps!==null&&Number.isFinite(Number(d.bps))?Number(d.bps).toFixed(2)+' bps ('+time(d.from)+' → '+time(d.to)+')':'Ölçüm yok';
 const labels={PRIMARY_TEXT_MATCH:'Başlık iddiasının metni belgenin içinde bulundu',PRIMARY_TEXT_MATCH_REVIEW_REQUIRED:'Metin eşleşti; düzeltme dili nedeniyle inceleme gerekli',TITLE_MATCH_ONLY:'Yalnız başlık eşleşti',CLAIM_TEXT_NOT_MATCHED:'İddia metni birebir bulunamadı',INDEX_NOT_CLAIM_VERIFICATION:'Yalnız dosya dizini okundu'};
 function render(rows,status){
  if(status!=='AVAILABLE'||!Array.isArray(rows))return '<p>İnceleme verisi alınamadı; tamamlanmış doğrulama sayılmaz.</p>';
  if(!rows.length)return '<p>Henüz kayıtlı inceleme yok.</p>';
  return rows.map(row=>{const r=row.review||{},d=r.document||{},c=r.counter_evidence||{},p=r.market_reaction?.probes||[];
   return '<details class="world-row"><summary class="world-title">'+esc(r.event?.claim||row.event_id)+' · '+(row.status==='READ'?'Belge okundu':row.status==='INDEX_ONLY'?'Dizin okundu':'Okuma başarısız')+'</summary><div class="world-meta"><p>İnceleme: '+time(row.assessed_at)+' · Kayıt: '+time(row.captured_at)+'<br>Brian ilk gördü: '+time(r.event?.first_observed_at)+' · Yayın: '+time(r.event?.published_at)+'</p><p>'+link(row.document_uri,'Birincil belge')+' · '+esc(labels[d.claim_text_status]||r.error||'İddia eşleşmesi bilinmiyor')+'<br>Bağımsız iddia doğrulaması: <b>Yapılmadı</b><br>Yanıt SHA-256: '+esc(row.content_sha256||'yok')+'</p>'+(d.excerpt?'<blockquote>'+esc(d.excerpt)+'</blockquote>':'')+
    '<p><b>Karşı kanıt araması:</b> '+esc(c.searched_records??'—')+' kayıtlı başlık tarandı; '+esc(c.candidates?.length??0)+' aday. İngilizce sözcük ve yön karşılaştırmasıdır; tüm internet taraması veya doğrulanmış çelişki değildir.</p>'+(c.candidates||[]).map(x=>'<p>'+link(x.provenance_uri,x.claim)+' · '+esc(x.reason)+' · Bağımsızlık ve çelişki doğrulanmadı</p>').join('')+
    '<p><b>Fiyatlanmışlık incelemesi:</b> BTC/ETH genel piyasa ölçümleri; olaya nedensel atıf yok. Fiyatlanmışlık olasılığı hesaplanmadı.</p>'+p.map(x=>'<p>'+esc(x.asset_id)+'<br>Yayın öncesi 15 dk: '+delta(x.pre_publication_15m)+'<br>Yayın → Brian: '+delta(x.publication_to_first_seen)+'<br>Brian sonrası 5 dk: '+delta(x.after_first_seen_5m)+'<br>Brian sonrası 15 dk: '+delta(x.after_first_seen_15m)+'</p>').join('')+'<p>Sonradan okunan belge, önceki kararın kanıtı sayılmaz. Bu inceleme işlem sinyali üretmez.</p></div></details>';
  }).join('');
 }
 root.BrianWorldEvidence={render};if(typeof module==='object')module.exports=root.BrianWorldEvidence;
})(globalThis);
