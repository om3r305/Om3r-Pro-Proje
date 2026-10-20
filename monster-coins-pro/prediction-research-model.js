(function(root){
  'use strict';
  const number=v=>v===null||v===undefined||v===''?null:Number.isFinite(Number(v))?Number(v):null;
  const hardStops=new Set(['ACCESS_DENIED','RATE_LIMITED','WS_UNSTABLE','IDENTITY_UNVERIFIED','DISK_CAP']);
  const labels={OK:'Güncel',DISABLED:'Kapalı',NO_DATA:'Henüz veri yok',STALE:'Gecikmiş',DEGRADED:'Kaynak sorunu',UNAVAILABLE:'Durum alınamadı',INSUFFICIENT_DATA:'Yetersiz veri',SUFFICIENT:'Kalibrasyon ölçüldü',NOT_CONFIGURED:'Kurulmadı · pilot başlatılmadı',NOT_STARTED:'Başlatılmadı · onay bekliyor',RUNNING:'Gözlemde',STOPPED:'Durduruldu',COMPLETED:'Tamamlandı'};
  function section(value){const x=value&&typeof value==='object'?value:{};const status=String(x.status||'UNAVAILABLE');return {raw:x,status,label:labels[status]||'Bilinmeyen durum',tone:['OK','RUNNING'].includes(status)?'ok':['STALE','DEGRADED'].includes(status)?'warn':status==='STOPPED'?(hardStops.has(x.run?.stop_reason)?'bad':'warn'):'neutral'};}
  function project(data){
    if(!data||data.schema_version!=='brian-prediction-status.v1'||data.shadow_only!==true||data.live_execution!==false||data.trade_signal!==false)return null;
    const prediction=section(data.prediction),calibration=section(data.calibration),pilot=section(data.niulai4_pilot);
    return {observedAt:data.observed_at,prediction,calibration,pilot,counts:{snapshots:number(prediction.raw.snapshots?.total),last24h:number(prediction.raw.snapshots?.last_24h),resolutions:number(prediction.raw.resolutions?.total),events:number(calibration.raw.report?.events),markets:number(calibration.raw.report?.markets)},walletTracking:typeof prediction.raw.control?.wallet_tracking_enabled==='boolean'?prediction.raw.control.wallet_tracking_enabled:null};
  }
  const api={project,section,number};
  root.BrianPredictionResearchModel=api;
  if(typeof module==='object'&&module.exports)module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
