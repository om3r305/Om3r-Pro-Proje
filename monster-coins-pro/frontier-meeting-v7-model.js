/* Read-only council projection. A display assessment never executes a trade. */
(function(root){
 const arr=x=>Array.isArray(x)?x:[];
 const fresh=(t,now,ttl)=>{const ms=Date.parse(t);return Number.isFinite(ms)&&ms<=now&&now-ms<=ttl};
 function project({event=null,decision=null,treasury=null,evidence={},rows=[],now=Date.now()}={}){
  const eventFresh=!!event&&fresh(event.time,now,event.urgency==='CRITICAL'?7200000:3600000);
  const eventId=event?.event_id;
  const refs=arr(decision?.source_event_ids);
  const linked=!!eventId&&refs.includes(eventId)&&!!event?.asset&&event.asset===decision?.asset_id;
  const alphaFresh=fresh(decision?.observed_at||decision?.created_at,now,300000);
  const gates=[
   {id:'world',name:'Dünya',icon:'◎',state:eventFresh?'ok':'wait',text:eventFresh?'Taze olay masada.':event?'Olay güncelliğini yitirdi.':'Taze yüksek öncelikli olay bekleniyor.',need:'Yeni ve zaman damgalı bir olay kaydı.'},
   {id:'source',name:'Kaynak',icon:'◈',state:eventFresh&&evidence.verified&&evidence.decisionEligible?'ok':'wait',text:!event?'İncelenecek olay yok.':evidence.verified?'Kaynak tanınıyor; karar kanıtına uygunluğu ayrıca aranır.':'Kaynak doğrulaması tamamlanmadı.',need:'Karar kanıtına uygun kaynak değerlendirmesi.'},
   {id:'alpha',name:'ALPHA',icon:'α',state:!eventFresh||!linked||!alphaFresh?'wait':decision.action==='VETO'?'blocked':['OPEN_LONG','OPEN_SHORT'].includes(decision.action)?'ok':'wait',text:!decision?'ALPHA kararı yok.':!linked?'Son ALPHA kararı bu olaya bağlanamadı; toplantı onayı sayılmıyor.':!alphaFresh?'İlişkili ALPHA kararı güncel değil.':`İlişkili karar: ${decision.action}.`,need:'Aynı olay kimliği ve varlıkla ilişkili, son 5 dakikaya ait ALPHA kanıtı.'},
   {id:'treasury',name:'Hazine',icon:'◇',state:treasury?.promotion_gate_open===true?'ok':'wait',text:treasury?treasury.promotion_gate_open===true?'Son hazine kaydında kapı açık.':'Son hazine kaydında kapı kapalı.':'Hazine kaydı bekleniyor.',need:'Hazine kapısının açık olduğuna dair güncel sunucu kontrolü.'},
   {id:'integrity',name:'Sistem',icon:'⊞',state:!rows.length?'wait':rows.some(r=>r.state==='bad')?'blocked':rows.some(r=>r.state!=='ok')?'wait':'ok',text:!rows.length?'Sistem kanıtı yok.':rows.some(r=>r.state==='bad')?'Teknik hata var; önce sistem bütünlüğü.':rows.some(r=>r.state!=='ok')?'Sistemde uyarı veya belirsiz durum var.':'Görünür modüllerde hata yok.',need:'Tüm ilgili modüllerin sağlıklı çalışma kanıtı.'}
  ];
  const blocked=gates.find(g=>g.state==='blocked'),pending=gates.filter(g=>g.state!=='ok');
  return {gates,linked,alphaFresh,eventFresh,blocked:!!blocked,title:blocked?'İnceleme bloke':!event?'Yeni olay bekleniyor':pending.length?'Kanıt bekleniyor':'Kanıtlar uyumlu',reason:blocked?.text||pending[0]?.text||'Görünür koşullar uyumlu. Gerçek işlem kararı sunucunun risk ve maliyet kontrollerine aittir.',next:blocked?.need||pending[0]?.need||'Sunucunun işlem sonucu ve gerekçesi bekleniyor.',ready:gates.filter(g=>g.state==='ok').length};
 }
 const api={project,fresh};if(typeof module!=='undefined')module.exports=api;else root.BrianCouncilModel=api;
})(typeof window!=='undefined'?window:globalThis);
