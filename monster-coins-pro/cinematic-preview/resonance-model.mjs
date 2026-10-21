import {healthFor} from './journey-model.mjs';
export function stable(value){if(Array.isArray(value))return '['+value.map(stable).join(',')+']';if(value&&typeof value==='object')return '{'+Object.keys(value).sort().map(k=>JSON.stringify(k)+':'+stable(value[k])).join(',')+'}';return JSON.stringify(value);}
export function createRecordStream({limit=48,queueLimit=8}={}){
 let prior=new Map(),ledger=[],queue=[],seq=0,replayState=false;
 function observe(graph,{replay=false}={}){
  if(!graph?.valid)return [];
  if(replay!==replayState){prior.clear();queue=[];replayState=replay;}
  const first=prior.size===0,added=[];
  for(const n of graph.nodes||[]){if(n.id==='brian'||n.group==='asset'||!Number.isFinite(Date.parse(n.at)))continue;
   const signature=stable(n.raw),old=prior.get(n.id);
   if(old&&(Date.parse(n.at)<Date.parse(old.at)||old.signature===signature))continue;
   prior.set(n.id,{at:n.at,signature});
   const revision=old&&old.at===n.at;
   const record={...n,raw:structuredClone(n.raw),seq:++seq,observedAt:graph.at,mode:replay?'import':first?'baseline':revision?'revision':'live',liveEligible:healthFor(n,graph,{replay}).flow===1,delta:old?Object.keys(n.raw).filter(k=>stable(n.raw[k])!==stable(old.raw?.[k])).slice(0,8):[]};
   prior.set(n.id,{at:n.at,signature,raw:structuredClone(n.raw)});ledger.unshift(record);added.push(record);
  }
  ledger=ledger.slice(0,limit);
  if(!first&&!replay)queue.push(...added.filter(n=>n.liveEligible));
  const newest=new Map();for(const n of queue)newest.set(n.id,n);const priority=n=>['CRITICAL','EMERGENCY'].includes(String(n.raw.severity).toUpperCase())?3:n.id==='alpha'?2:n.id==='news'?1:0;queue=[...newest.values()].sort((a,b)=>priority(b)-priority(a)||Date.parse(b.at)-Date.parse(a.at)).slice(0,queueLimit);return added;
 }
 return {observe,records:()=>ledger.slice(),next:()=>queue.shift()||null,pending:()=>queue.length,clear:()=>{queue=[];}};
}
export function eventTone(record){const r=record?.raw||{},severity=String(r.severity||'').toUpperCase();if(['CRITICAL','EMERGENCY'].includes(severity)||['FAILED','ERROR'].includes(String(r.status).toUpperCase()))return '#ff4c69';if(['WARNING','HIGH'].includes(severity)||['VETO','BLOCKED'].includes(r.action))return '#ffc85b';if(['OPEN_LONG','OPEN_SHORT'].includes(r.action))return '#7de9c4';return ({world:'#6fc7ff',alpha:'#bca0ff',news:'#65e1f0',treasury:'#f2cf7a',behavior:'#ed9bd4'})[record?.id]||'#b6caff';}
export function phaseAt(seconds){return seconds<.8?0:seconds<2.8?1:seconds<3.7?2:seconds<5.1?3:4;}