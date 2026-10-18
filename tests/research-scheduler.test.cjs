const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {stripTypeScriptTypes}=require('node:module');
const src=fs.readFileSync('supabase/functions/brian-realtime-core-scheduler/index.ts','utf8');
const planCode=src.slice(src.indexOf('function coreActionForMinute'),src.indexOf('async function handle'));
const {planned}=vm.runInNewContext(stripTypeScriptTypes(planCode)+';({planned})');
test('research schedule never launches position engines, recovery fanout or DIP',()=>{
 for(let minute=0;minute<60;minute++){
  const actions=planned(minute);
  assert.ok(actions.every(a=>!['treasury','multiasset','dip','recovery','archive','missed_auditor'].includes(a)),`${minute}: ${actions}`);
  assert.equal(actions.length,new Set(actions).size);
 }
});
test('a full hour retains news ingestion, classification, decision evidence and cost measurement',()=>{
 const actions=new Set(Array.from({length:60},(_,m)=>planned(m)).flat());
 for(const action of ['direct_wire','readiness_cost','alpha_sync','official_primary','world','source_observer','source_registry','meeting_sync','watchdog','discovery'])assert.ok(actions.has(action),action);
});

test('decision bridge cadence fits the strict terminal audit window without piling Core lanes',()=>{
 const syncMinutes=[];
 for(let m=0;m<60;m++){
  const core=planned(m).filter(a=>!['direct_wire','readiness_cost'].includes(a));
  assert.equal(core.length,1);
  if(core[0]==='alpha_sync')syncMinutes.push(m);
 }
 assert.equal(syncMinutes.length,30);
 for(let i=0;i<syncMinutes.length;i++)assert.equal(((syncMinutes[(i+1)%syncMinutes.length]-syncMinutes[i])+60)%60,2);
});
