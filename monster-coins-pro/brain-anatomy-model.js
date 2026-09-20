/* Read-only anatomy projection. No synthetic maturity, history or performance. */
(function (root) {
  'use strict';
  const organs = [
    {id:'alpha',name:'ALPHA',subtitle:'Karar merkezi',color:'#c4a0ff',pattern:/alpha-decision-compiler/,ttl:420},
    {id:'sensors',name:'Algı ağı',subtitle:'Gözler · kulaklar · sinirler',color:'#70e4ec',pattern:/eye|capture|sensor|micro-book/,ttl:1200},
    {id:'world',name:'Dünya hafızası',subtitle:'Kaynaklar · bağlam · hafıza',color:'#81bfff',pattern:/world/,ttl:1200},
    {id:'lab',name:'Gelişim laboratuvarı',subtitle:'Deney · kod · öğrenme',color:'#f5a2cd',pattern:/evolution-(research|sandbox|orchestrat|ocean|engineer)/,ttl:2400},
    {id:'treasury',name:'Hazine',subtitle:'Sermaye · uygulama',color:'#edc783',pattern:/treasury/,ttl:600}
  ];
  function number(v){if(v===null||v===undefined||(typeof v==='string'&&!v.trim())||typeof v==='boolean')return null;const n=Number(v);return Number.isFinite(n)?n:null;}
  function pct(v){const n=number(v);return n!==null&&n>=0&&n<=100?n:null;}
  function age(v,now=Date.now()){const t=Date.parse(v);return Number.isFinite(t)&&t<=now?(now-t)/1000:Infinity;}
  function project(data,hb,now=Date.now(),error=null,heartbeatError=null){
    const reportAge=age(data?.observed_at,now), reportFresh=reportAge<=180&&!error;
    const heartbeatFresh=age(hb?.observed_at,now)<=150&&!heartbeatError;
    const runStamp=(r)=>r?.last_success_at||r?.finished_at||r?.started_at||null;
    const runs=Object.entries(hb?.collectors||{}).map(([key,r])=>({...r,collector_id:r?.collector_id||key})).filter(r=>! /dip/i.test(r.collector_id));
    const mapped=organs.map(o=>{
      const c=data?.components?.find(x=>x.id===o.id);
      const evidence=pct(c?.evidence_pct);
      const matching=runs.filter(r=>o.pattern.test(r.collector_id)).sort((a,b)=>age(runStamp(a),now)-age(runStamp(b),now));
      const recent=matching.filter(r=>age(runStamp(r),now)<=o.ttl);
      const healthy=recent.filter(r=>(r.status==='SUCCESS'||r.status==='ONLINE'||age(r.last_success_at,now)<=o.ttl)).length;
      const hardError=recent.some(r=>r.status==='FAILED'&&age(r.last_success_at,now)>o.ttl);
      const softIssue=recent.some(r=>r.status==='FAILED'||r.status==='DEGRADED'||String(r.status||'').startsWith('SKIPPED'));
      const state=!heartbeatFresh||!recent.length?'unknown':hardError?'error':softIssue?'degraded':healthy===recent.length?'live':'degraded';
      return {...o,component:c||null,maturity:evidence>0?pct(c?.maturity_pct):null,quality:evidence>0?pct(c?.quality_pct):null,evidence,samples:number(c?.samples),runs:matching,healthy,state,stamp:runStamp(recent[0]),reportFresh};
    });
    return {organs:mapped,reportFresh,reportAge,heartbeatFresh,runs,overall:data?.overall||null};
  }
  const api={organs,number,pct,age,project};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  else root.BrianAnatomyModel=Object.freeze(api);
})(typeof window==='undefined'?globalThis:window);
