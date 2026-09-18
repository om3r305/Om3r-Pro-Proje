(function(root){
 const modules=[{key:'world',name:'Dünya',icon:'◎',ttl:1200},{key:'behavior',name:'Davranış',icon:'◈',ttl:1200},{key:'alpha',name:'ALPHA',icon:'α',ttl:420},{key:'treasury',name:'Hazine',icon:'◇',ttl:900},{key:'research',name:'Araştırma',icon:'⌬',ttl:2400},{key:'ocean',name:'Okyanus',icon:'≈',ttl:2400}];
 const fresh=(v,now,seconds)=>{const t=Date.parse(v);return Number.isFinite(t)&&now>=t&&now-t<=seconds*1000};
 function project(hb,rows=[],error=null,now=Date.now()){
  const run=id=>hb?.collectors?.[id],stamp=r=>r?.finished_at||r?.started_at;
  const world=stamp(hb?.world_run),alpha=hb?.alpha?.observed_at,treasury=hb?.control?.treasury;
  const research=run('brian-evolution-orchestrator-v1')||run('brian-evolution-sandbox-v1')||run('brian-evolution-researcher-v1');
  const stamps={world,behavior:world&&alpha?(Date.parse(world)<Date.parse(alpha)?world:alpha):null,alpha,treasury:stamp(run('brian-evolution-treasury-v1'))||treasury?.observed_at,research:stamp(research),ocean:stamp(run('brian-evolution-ocean-worker-v1'))};
  const heartbeatFresh=!!hb&&String(hb.status||'').toUpperCase()==='OK'&&!hb.__cached&&fresh(hb.observed_at,now,120),enabled=hb?.control?.system_enabled!==false;
  const result=modules.map(m=>{const r=rows.find(r=>r.key===m.key),recent=fresh(stamps[m.key],now,m.ttl);const state=!heartbeatFresh?'unknown':!enabled?'off':r?.state==='bad'?'error':r?.state==='ok'&&recent?'live':'waiting';return {...m,state,stamp:stamps[m.key]||null,meta:!heartbeatFresh?'Güncel heartbeat doğrulanamadı.':!enabled?'Sistem operatör tarafından durduruldu.':r?.meta||'Modül kanıtı bekleniyor.'}});
  const number=v=>v!==null&&v!==undefined&&v!==''&&Number.isFinite(Number(v))?Number(v):null;
  return {modules:result,heartbeatFresh,enabled,live:result.filter(x=>x.state==='live').length,heartbeatAt:hb?.observed_at||null,equity:number(treasury?.equity_usd),treasuryFresh:heartbeatFresh&&fresh(treasury?.observed_at,now,600),positions:Array.isArray(treasury?.positions)?treasury.positions.length:null,action:hb?.alpha?.action||null,asset:hb?.alpha?.asset_id||null};
 }
 const api={project};if(typeof module!=='undefined')module.exports=api;else root.BrianCommandModel=api;
})(typeof window!=='undefined'?window:globalThis);
