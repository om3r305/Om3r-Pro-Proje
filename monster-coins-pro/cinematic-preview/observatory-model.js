(function(root){
 'use strict';
 const number=v=>typeof v==='number'&&Number.isFinite(v)?v:null;
 const stamp=v=>v?.observed_at||v?.finished_at||v?.last_run_at||v?.started_at||null;
 const titles={'brian-world-brain-v1':'Dünya motoru','brian-evolution-sandbox-v1':'Sandbox','brian-evolution-treasury-v1':'Hazine motoru','brian-world-discovery-eye-v1':'Keşif','brian-evolution-researcher-v1':'Araştırmacı','brian-evolution-ocean-worker-v1':'Okyanus','brian-evolution-orchestrator-v1':'Orkestratör','brian-alpha-decision-compiler-v2':'Karar derleyici'};
 function project(h,now=Date.now()){
  const empty={valid:false,fresh:false,nodes:[],edges:[],events:[],collectors:[],metrics:{}};
  if(!h||h.status!=='OK'||h.read_only!==true||h.shadow_only!==true||h.live_execution!==false||!Number.isFinite(Date.parse(h.observed_at||'')))return empty;
  const current=(at,ttl)=>{const t=Date.parse(at||'');return Number.isFinite(t)&&t<=now&&now-t<=ttl;};
  const fresh=current(h.observed_at,120000)&&!h.__cached&&!h.transport_degraded;
  const nodes=[],edges=[];
  const add=(id,label,path,raw,ttl=1200000,group='core')=>{if(!raw||typeof raw!=='object'||Array.isArray(raw))return;const at=stamp(raw);const failed=raw.status==='FAILED'||raw.status==='ERROR';const n={id,label,path,raw,at,group,state:failed?'failed':fresh&&current(at,ttl)?'current':'stale',record_id:raw.id||raw.run_id||raw.source_decision_id||raw.snapshot_id||null};nodes.push(n);if(id!=='brian')edges.push({from:'brian',to:id,type:'contains-record',path});return n;};
  add('brian','BRIAN','/',{status:h.status,observed_at:h.observed_at,source:h.source,read_only:h.read_only,shadow_only:h.shadow_only,live_execution:h.live_execution},120000);
  add('world','Dünya istihbaratı','/world_run',h.world_run);
  add('alpha','ALPHA karar','/alpha',h.alpha,420000);
  add('treasury','Hazine','/control/treasury',h.control?.treasury,600000);
  add('behavior','Davranış bağlamı','/behavior',h.behavior,900000);
  add('news','Haber hattı','/news_pipeline',h.news_pipeline,120000);
  const collectors=[];
  for(const [key,raw]of Object.entries(h.collectors||{})){if(!raw||typeof raw!=='object'||Array.isArray(raw))continue;const node=add('collector:'+key,titles[key]||key,'/collectors/'+key,raw,2400000,'collector');if(node)collectors.push(node);}
  if(h.alpha?.asset_id){const n=add('asset',h.alpha.asset_id,'/alpha/asset_id',{asset_id:h.alpha.asset_id,action:h.alpha.action,observed_at:h.alpha.observed_at},420000,'asset');edges.pop();edges.push({from:'alpha',to:n.id,type:'explicit-asset-reference',path:'/alpha/asset_id'});}
  const events=nodes.filter(n=>n.id!=='brian'&&n.at&&Number.isFinite(Date.parse(n.at))).map(n=>({...n,eventStatus:n.raw.activity_state||n.raw.status||n.raw.action||n.raw.state||'KAYIT',observed:number(n.raw.observed_records),stored:number(n.raw.stored_records)})).sort((a,b)=>Date.parse(b.at)-Date.parse(a.at));
  const t=h.control?.treasury||{},equity=number(t.equity_usd),start=number(t.starting_equity_usd);
  return {valid:true,fresh,at:h.observed_at,nodes,edges,events,collectors,metrics:{equity,cash:number(t.cash_usd),start,change:equity!==null&&start!==null?equity-start:null,changePct:equity!==null&&start>0?(equity/start-1)*100:null,positions:number(t.open_positions),deployment:number(t.deployment_pct),inputs:number(h.world_run?.input_events),frames:number(h.world_run?.event_frames),entities:number(h.world_run?.entity_observations),impacts:number(h.world_run?.asset_impacts),narratives:number(h.world_run?.narrative_snapshots),managed:number(h.control?.managed_jobs_total),active:number(h.control?.managed_jobs_active),evidence:number(h.alpha?.evidence_score),backlog:number(h.news_pipeline?.backlog_events)}};
 }
 const api={project,number,stamp};if(typeof module==='object'&&module.exports)module.exports=api;else root.ObservatoryModel=api;
})(typeof window==='object'?window:globalThis);
