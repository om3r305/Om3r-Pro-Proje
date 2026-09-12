'use strict';

/* Frontier stability layer: one lightweight heartbeat every 15s, one detailed service per cycle.
   Keeps last-known-good evidence during transient DB/API timeouts. DIP is not queried or controlled here. */
const HEARTBEAT_ENDPOINT = `${ROOT}/brian-frontier-heartbeat`;
const STABILITY = { heartbeat:null, error:null, slowCursor:0, lastSlow:0, lastAutonomyFetch:0 };

async function frontierPost(url,body={},timeoutMs=7500){
  const k=key();
  if(!k) throw new Error('UNAUTHORIZED_DASHBOARD');
  const controller=new AbortController();
  const timer=setTimeout(()=>controller.abort(),timeoutMs);
  try{
    const r=await fetch(url,{method:'POST',headers:{'content-type':'application/json','x-brian-dashboard-key':k},body:JSON.stringify(body),cache:'no-store',signal:controller.signal});
    let data={};try{data=await r.json()}catch{}
    if(!r.ok){const e=new Error(data.error||data.status||`HTTP ${r.status}`);e.status=r.status;e.payload=data;throw e}
    return data;
  }catch(e){
    if(e?.name==='AbortError') throw new Error('STATUS_TIMEOUT');
    throw e;
  }finally{clearTimeout(timer)}
}

safe = async function(name,p){
  try{S[name]=await p;delete S.errors[name]}
  catch(e){S.errors[name]=String(e?.message||e)}
};

function heartbeatRun(id){return STABILITY.heartbeat?.collectors?.[id]||null}
function heartbeatAgeSeconds(v){const t=Date.parse(String(v||''));return Number.isFinite(t)?Math.max(0,(Date.now()-t)/1000):999999}
function heartbeatFresh(run,maxAge){return Boolean(run)&&heartbeatAgeSeconds(run.finished_at||run.started_at)<=maxAge}

function applyHeartbeat(hb){
  if(!hb||hb.status!=='OK')return;
  const c=hb.control||{};
  const collectors=Object.values(hb.collectors||{});
  S.systemControl={status:'OK',control:c,last_data:{alpha:hb.alpha||null,world:hb.world_run||null,collectors},dip_touched:false,shadow_only:true,live_execution:false};

  if(hb.alpha){
    const seconds=heartbeatAgeSeconds(hb.alpha.observed_at);
    if(!S.control||S.errors.control){
      S.control={alpha_v2:{online:seconds<=900,decision_age_seconds:seconds,decisions:[hb.alpha]}};
    }
  }

  if(c.treasury&&(!S.treasury||S.errors.treasury)){
    S.treasury={status:'ONLINE',snapshot:c.treasury,summary:c.treasury,runs:[heartbeatRun('brian-evolution-treasury-v1')].filter(Boolean)};
  }

  const wr=heartbeatRun('brian-world-brain-v1'),disc=heartbeatRun('brian-world-discovery-eye-v1');
  if(!S.world||S.errors.world){
    const w=hb.world_run||{};
    S.world={
      status:wr&&String(wr.status)==='SUCCESS'?'ONLINE':'DEGRADED',
      collectors:{world_brain:wr,discovery:disc},
      summary:{unique_entities:Number(w.entity_observations||0),narratives:Number(w.narrative_snapshots||0),asset_impact_candidates:Number(w.asset_impacts||0)},
      narratives:[]
    };
  }

  const evo=heartbeatRun('brian-evolution-orchestrator-v1')||heartbeatRun('brian-evolution-sandbox-v1')||heartbeatRun('brian-evolution-researcher-v1');
  if(!S.evolution||S.errors.evolution){S.evolution={status:evo&&String(evo.status)==='SUCCESS'?'ONLINE':'DEGRADED',runs:evo?[evo]:[],journal:[],gaps:[]}}
  const ocean=heartbeatRun('brian-evolution-ocean-worker-v1');
  if(!S.ocean||S.errors.ocean){S.ocean={status:ocean&&String(ocean.status)==='SUCCESS'?'ONLINE':'IDLE',runs:ocean?[ocean]:[],active_run:null}}
  if(!S.news&&S.errors.news)S.news={items:[]};
}

const moduleRowsBeforeStability=moduleRows;
moduleRows=function(){
  const hb=STABILITY.heartbeat;
  if(!hb||hb.status!=='OK')return moduleRowsBeforeStability();
  const c=hb.control||{},enabled=c.system_enabled!==false;
  if(!enabled)return moduleRowsBeforeStability();
  const total=Number(c.managed_jobs_total||0),active=Number(c.managed_jobs_active||0);
  const wr=heartbeatRun('brian-world-brain-v1'),disc=heartbeatRun('brian-world-discovery-eye-v1');
  const alphaRun=heartbeatRun('brian-alpha-decision-compiler-v2');
  const treasuryRun=heartbeatRun('brian-evolution-treasury-v1');
  const evoRun=heartbeatRun('brian-evolution-orchestrator-v1');
  const sandboxRun=heartbeatRun('brian-evolution-sandbox-v1');
  const researcherRun=heartbeatRun('brian-evolution-researcher-v1');
  const oceanRun=heartbeatRun('brian-evolution-ocean-worker-v1');

  const alphaAge=heartbeatAgeSeconds(hb.alpha?.observed_at);
  let alphaState=alphaAge<=360?'ok':alphaAge<=900?'warn':'bad';
  if(String(alphaRun?.status)==='FAILED'&&alphaAge>360)alphaState='bad';
  const alphaMeta=hb.alpha?`Karar ${Math.round(alphaAge)} sn önce · ${hb.alpha.asset_id||'ALPHA'} ${act(hb.alpha.action)}`:(alphaRun?.error_message||'ALPHA karar kanıtı bekleniyor');

  let worldState=wr&&String(wr.status)==='SUCCESS'&&heartbeatFresh(wr,1200)?'ok':wr&&String(wr.status)==='FAILED'?'bad':'warn';
  const discIssue=disc&&['FAILED','DEGRADED'].includes(String(disc.status));
  const worldMeta=worldState==='ok'?(discIssue?'Dünya çekirdeği canlı · keşif sensörü gecikiyor':'Dünya çekirdeği ve keşif hattı canlı'):(wr?.error_message||'World Brain taze kanıt bekliyor');

  const t=c.treasury||{};const tAge=heartbeatAgeSeconds(t.observed_at);
  let treasuryState=t.equity_usd!=null&&tAge<=1200?'ok':t.equity_usd!=null?'warn':'bad';
  if(String(treasuryRun?.status)==='FAILED'&&tAge>600)treasuryState='warn';
  const treasuryMeta=t.equity_usd!=null?`Gerçek kasa ${money(t.equity_usd)} · ${Math.round(tAge/60)} dk önce`:(treasuryRun?.error_message||'Hazine kanıtı bekleniyor');

  const evoOk=[evoRun,sandboxRun,researcherRun].some(r=>r&&String(r.status)==='SUCCESS'&&heartbeatFresh(r,2400));
  const evoHardFail=[evoRun,sandboxRun].every(r=>r&&String(r.status)==='FAILED'&&heartbeatFresh(r,2400));
  const researchState=evoOk?'ok':evoHardFail?'bad':'warn';
  const researchMeta=evoOk?'Evolution/Lab canlı; taze çalışma kanıtı var':(evoRun?.error_message||sandboxRun?.error_message||researcherRun?.error_message||'Araştırma hattı kanıt bekliyor');

  const oceanState=oceanRun&&String(oceanRun.status)==='SUCCESS'&&heartbeatFresh(oceanRun,1800)?'ok':oceanRun&&String(oceanRun.status)==='FAILED'?'warn':'warn';
  const oceanMeta=oceanState==='ok'?'Ocean worker canlı':(oceanRun?.error_message||'Ocean yeni cycle bekliyor');

  const behaviorState=worldState==='ok'&&alphaState==='ok'?'ok':worldState==='bad'||alphaState==='bad'?'bad':'warn';
  const behaviorMeta=behaviorState==='ok'?'World + ALPHA davranış kanıt zinciri canlı':'Davranış kanıt zinciri tazeleniyor';
  if(total>0&&active<total&&![worldState,alphaState,treasuryState,researchState,oceanState].includes('bad'))worldState=worldState==='ok'?'warn':worldState;

  return[
    {key:'world',name:'Dünya Gezgini',icon:'🌍',state:worldState,meta:worldMeta},
    {key:'behavior',name:'Davranış Motoru',icon:'🧠',state:behaviorState,meta:behaviorMeta},
    {key:'alpha',name:'ALPHA',icon:'α',state:alphaState,meta:alphaMeta},
    {key:'treasury',name:'Hazine / Portföy',icon:'◉',state:treasuryState,meta:treasuryMeta},
    {key:'research',name:'Araştırma Lab',icon:'⚗',state:researchState,meta:researchMeta},
    {key:'ocean',name:'Okyanus',icon:'≈',state:oceanState,meta:oceanMeta}
  ];
};

const slowServices=[
  ['control',CONTROL,{action:'status'}],
  ['world',EP.world,{}],
  ['treasury',EP.treasury,{}],
  ['ocean',EP.ocean,{}],
  ['news',EP.news,{}],
  ['evolution',EP.evolution,{}],
  ['lab',EP.lab,{}],
  ['alphaIntel',EP.alphaIntel,{}],
];

refresh=async function(){
  if(!key()){unlock(true);return}
  $('syncText').textContent='Brian heartbeat doğrulanıyor…';
  try{
    const hb=await frontierPost(HEARTBEAT_ENDPOINT,{},7000);
    STABILITY.heartbeat=hb;STABILITY.error=null;delete S.errors.systemControl;applyHeartbeat(hb);
  }catch(e){
    STABILITY.error=String(e?.message||e);S.errors.systemControl=STABILITY.error;
  }

  const now=Date.now();
  if(now-STABILITY.lastSlow>=12000){
    const [name,url,body]=slowServices[STABILITY.slowCursor++%slowServices.length];
    STABILITY.lastSlow=now;
    await safe(name,frontierPost(url,body,7000));
  }
  S.lastSync=new Date();
  const c=sysControl(),target=num(c.treasury_target_equity_usd)??num(c.treasury?.starting_equity_usd);
  if(!S.amountDirty&&target!=null)S.selectedAmount=target;
  render();
};

/* Expensive autonomy console is evidence analytics, not a 15-second heartbeat. */
const refreshAutonomyCore=refreshAutonomyV4;
refreshAutonomyV4=async function(){
  const now=Date.now();
  if(STABILITY.lastAutonomyFetch&&now-STABILITY.lastAutonomyFetch<180000){renderAutonomyV4();renderTreasuryV4();renderMeetingV4();return}
  STABILITY.lastAutonomyFetch=now;
  return refreshAutonomyCore();
};

/* Re-render immediately from the authoritative lightweight heartbeat. */
if(key())refresh();
