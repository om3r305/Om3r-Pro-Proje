(function(root){
 const arr=x=>Array.isArray(x)?x:[];
 const time=v=>{const n=Date.parse(v);return Number.isFinite(n)?n:null};
 const phases=[['CLAIMED','Görev alındı'],['UNDERSTAND','İnceleme'],['PLAN','Plan'],['CODE','Kod'],['COMPILE','Derleme'],['TEST','Test'],['REPLAY','Geçmiş veri testi'],['STRESS','Stres testi'],['REVIEW','İnceleme onayı'],['PR','PR'],['PREVIEW','Önizleme'],['MEASURE','Ölçüm'],['HUMAN_APPROVAL','GPT onayı'],['DEPLOY','Yayın'],['MONITOR','İzleme'],['COMPLETE','Tamamlandı']];
 const fields={COMPILE:'compile_passed',TEST:'tests_passed',REPLAY:'replay_passed',STRESS:'stress_passed',REVIEW:'review_passed',PREVIEW:'preview_passed',MEASURE:'measurement_passed'};
 function project(data,selected=null,error=null,now=Date.now()){
  const e=data||{},all=[e.current_run,...arr(e.approval_runs),...arr(e.recent_runs)].filter(Boolean),runs=[...new Map(all.map(r=>[r.run_id,r])).values()];
  const run=runs.find(r=>r.run_id===selected)||e.current_run||arr(e.approval_runs)[0]||runs[0]||null;
  const events=arr(run?.recent_events).slice().sort((a,b)=>(time(b.observed_at)||0)-(time(a.observed_at)||0));
  const stamp=time(e.observed_at),updated=time(run?.updated_at),reportFresh=!error&&stamp!==null&&now>=stamp&&now-stamp<300000;
  const recent=updated!==null&&updated<=now&&now-updated<900000;
  const historical=!!run&&run.run_id!==e.current_run?.run_id&&!arr(e.approval_runs).some(r=>r.run_id===run.run_id);
  const blocked=!!run&&['BLOCKED','FAILED','ROLLED_BACK'].includes(run.status);
  const running=run?.status==='RUNNING';
  const phase=phases.find(p=>p[0]===run?.phase)?.[1]||run?.phase||'Kayıt bekleniyor';
  const state=!data?'Veri bekleniyor':!reportFresh?'Güncel bağlantı yok':!run?'Henüz çalışma yok':blocked?'Çalışma engellendi':historical?'Geçmiş çalışma':run.status==='COMPLETE'?'Tamamlandı':run.status==='WAITING'?'Onay / koşul bekliyor':running&&!recent?'Yeni ilerleme kaydı yok':running?'Çalışma kaydı güncel':run.status;
  const checks=Object.entries(fields).map(([p,f])=>{const exact=events.find(x=>x.event_kind===p&&run?.commit_sha&&x.commit_sha===run.commit_sha&&typeof x.passed==='boolean');return {phase:p,label:phases.find(x=>x[0]===p)[1],state:exact?.passed===false?'fail':exact?.passed===true?'pass':'unknown',note:exact?'Aday sürümüne bağlı olay kaydı.':run?.[f]===true?'Run başarılı işaretli; bu aday sürümüne bağlı olay kanıtı görünmüyor.':'Bu aday için doğrulanmış sonuç yok.'}});
  const failure=run?.failure_reason||events.find(x=>x.passed===false)?.payload?.error||events.find(x=>x.passed===false)?.payload?.reason||null;
  const next=!run?'Kuyruktan alınmış bir çalışma kaydı bekleniyor.':blocked?'Hata ayrıntısı incelenip aynı aday için yeni çalışma kanıtı üretilmeli.':running&&!recent?'Çalıştırıcının son durumu ve günlükleri kontrol edilmeli; kayıt sessizliği tek başına işlemin durduğunu kanıtlamaz.':!run.commit_sha?'Kod değişikliği ve aday sürüm kaydı bekleniyor.':checks.some(c=>c.state==='fail')?'Başarısız kontrol düzeltilip aynı sürüm üzerinde yeniden doğrulanmalı.':checks.some(c=>c.state==='unknown')?'Aday sürümüne bağlı test ve inceleme sonuçları bekleniyor.':'Onay, yayın ve ölçüm kayıtları kendi aşamalarında doğrulanmalı.';
  return {runs,run,events,reportFresh,recent,historical,blocked,phase,state,checks,failure,next,minutes:updated!==null&&updated<=now?Math.floor((now-updated)/60000):null,animate:reportFresh&&recent&&running&&!historical&&!blocked};
 }
 const api={project,phases};if(typeof module!=='undefined')module.exports=api;else root.EngineeringRoomModel=api;
})(typeof window!=='undefined'?window:globalThis);
