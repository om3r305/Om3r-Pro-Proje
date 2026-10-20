(function(root){
 'use strict';
 const finite=v=>typeof v==='number'&&Number.isFinite(v)?v:null;
 const fresh=(at,now,ttl)=>{const t=Date.parse(at||'');return Number.isFinite(t)&&t<=now&&now-t<=ttl;};
 function graph(hb,now=Date.now()){
  if(!hb||hb.status!=='OK'||!hb.observed_at||hb.read_only!==true||hb.shadow_only!==true||hb.live_execution!==false)return{nodes:[],edges:[],fresh:false};
  const healthy=fresh(hb.observed_at,now,120000)&&!hb.__cached&&!hb.transport_degraded;
  const rootId='envelope:'+hb.observed_at;
  const nodes=[{id:rootId,label:'BRIAN',kind:'envelope',at:hb.observed_at,path:'/',state:healthy?'current':'stale',record_id:null,source_kind:'public-heartbeat',raw:{status:hb.status,observed_at:hb.observed_at}}],edges=[];
  const add=(key,label,raw,path,ttl)=>{if(!raw)return;const at=raw.observed_at||raw.finished_at||raw.started_at;const id=key;nodes.push({id,label,kind:key,at:at||null,path,state:!healthy?'stale':raw.status==='FAILED'?'failed':fresh(at,now,ttl)?'current':'stale',record_id:raw.id||raw.run_id||raw.source_decision_id||null,source_kind:'public-heartbeat',raw});edges.push({from:rootId,to:id,type:'contains-record',path,meaning:'Aynı heartbeat yanıtındaki kayıt; nedensellik veya ajan mesajı değildir.'});};
  add('world','Dünya',hb.world_run,'/world_run',1200000);
  add('alpha','ALPHA',hb.alpha,'/alpha',420000);
  add('treasury','Hazine',hb.control?.treasury,'/control/treasury',600000);
  for(const[key,label]of[['brian-evolution-orchestrator-v1','Araştırma'],['brian-evolution-ocean-worker-v1','Okyanus']])add(key,label,hb.collectors?.[key],'/collectors/'+key,2400000);
  if(hb.alpha?.asset_id){const id='asset:'+hb.alpha.asset_id;nodes.push({id,label:hb.alpha.asset_id,kind:'asset',at:hb.alpha.observed_at,path:'/alpha/asset_id',state:healthy&&fresh(hb.alpha.observed_at,now,420000)?'current':'stale',record_id:null,source_kind:'explicit-asset-reference',raw:{asset_id:hb.alpha.asset_id,action:hb.alpha.action}});edges.push({from:'alpha',to:id,type:'explicit-asset-reference',path:'/alpha/asset_id',meaning:'Kararın açık varlık alanı. Fiyat etkisi kanıtı değildir.'});}
  return{nodes,edges,fresh:healthy,equity:finite(hb.control?.treasury?.equity_usd),observed_at:hb.observed_at};
 }
 function bufferSize(w,h,dpr,scale=1,maxPixels=8000000,maxSize=16384){const ratio=Math.min(Math.max(.25,dpr||1),1.5)*scale;const bounded=Math.min(ratio,Math.sqrt(maxPixels/Math.max(1,w*h)),maxSize/Math.max(1,w),maxSize/Math.max(1,h));return{width:Math.max(1,Math.floor(w*bounded)),height:Math.max(1,Math.floor(h*bounded)),ratio:bounded,estimatedBufferBytes:Math.floor(w*bounded)*Math.floor(h*bounded)*8};}
 const api={graph,bufferSize,fresh};if(typeof module==='object'&&module.exports)module.exports=api;else root.CinematicModel=api;
})(typeof window==='object'?window:globalThis);
