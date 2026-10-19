const{test}=require('node:test'),assert=require('node:assert/strict');const{project,providerExplanation}=require('../monster-coins-pro/engineering-room-model.js');const now=Date.now(),date=n=>new Date(now+n).toISOString();const run={run_id:'active',phase:'PLAN',status:'RUNNING',updated_at:date(-1000),commit_sha:'sha-a',recent_events:[]};const data={observed_at:date(-1000),current_run:run,approval_runs:[{run_id:'older',status:'WAITING'}]};
test('active run wins over older approval',()=>assert.equal(project(data,null,null,now).run.run_id,'active'));
test('no data and no timestamp never look live',()=>{assert.equal(project(null).animate,false);assert.equal(project({current_run:run}).reportFresh,false)});
test('stale progress reports silence and no animation',()=>{const v=project({...data,current_run:{...run,updated_at:date(-3600000)}},null,null,now);assert.equal(v.state,'Yeni ilerleme kaydı yok');assert.equal(v.animate,false)});
test('blocked and failed never animate',()=>{for(const status of ['BLOCKED','FAILED'])assert.equal(project({...data,current_run:{...run,status}},null,null,now).animate,false)});
test('test flag and unrelated candidate cannot turn green',()=>{const v=project({...data,current_run:{...run,tests_passed:true,recent_events:[{event_kind:'TEST',commit_sha:'other',passed:true}]}},null,null,now);assert.equal(v.checks.find(x=>x.phase==='TEST').state,'unknown')});
test('exact candidate test recognizes fail and pass, null unknown',()=>{for(const passed of [true,false,null]){const v=project({...data,current_run:{...run,recent_events:[{event_kind:'TEST',commit_sha:'sha-a',passed}]}},null,null,now);assert.equal(v.checks.find(x=>x.phase==='TEST').state,passed===true?'pass':passed===false?'fail':'unknown')}});
test('transport error disables live presentation',()=>assert.equal(project(data,null,'error',now).animate,false));
test('actual gateway receipt names map to their matching evidence checks',()=>{
 const kinds={COMPILE:'COMPILE',TEST:'UNIT_REGRESSION',REPLAY:'REPLAY',STRESS:'STRESS',REVIEW:'INDEPENDENT_REVIEW',MEASURE:'ENGINEERING_MEASURE'};
 for(const preview of ['VERCEL_PREVIEW','PREVIEW_EQUIVALENT']){
  const events=Object.entries({...kinds,PREVIEW:preview}).map(([phase,event_kind])=>({phase,event_kind,commit_sha:'sha-a',passed:true}));
  const v=project({...data,current_run:{...run,recent_events:events}},null,null,now);
  assert.ok(v.checks.every(c=>c.state==='pass'));
 }
});
test('actual receipt names from another candidate or unrelated phase cannot pass',()=>{
 const events=[{event_kind:'UNIT_REGRESSION',phase:'TEST',commit_sha:'old',passed:true},{event_kind:'BLOCKED',phase:'TEST',commit_sha:'sha-a',passed:true}];
 const v=project({...data,current_run:{...run,tests_passed:true,recent_events:events}},null,null,now);
 assert.equal(v.checks.find(c=>c.phase==='TEST').state,'unknown');
});
test('a later failed receipt overrides older passing evidence for the exact candidate',()=>{
 const events=[{event_kind:'UNIT_REGRESSION',commit_sha:'sha-a',passed:true,observed_at:date(-2000)},
  {event_kind:'UNIT_REGRESSION',commit_sha:'sha-a',passed:false,observed_at:date(-1000)}];
 const v=project({...data,current_run:{...run,recent_events:events}},null,null,now);
 assert.equal(v.checks.find(c=>c.phase==='TEST').state,'fail');
});
test('provider outage explains quota and timeout without inventing test failure',()=>{
 const v=project({...data,current_run:{...run,status:'BLOCKED',commit_sha:null,recent_events:[{event_kind:'BLOCKED',passed:false,payload:{provider_exhausted:true,provider_state:{attempts:[{error_class:'QUOTA_EXHAUSTED'},{exit_code:124}]}}}]}},null,null,now);
 assert.match(v.failure,/kotası dolmuş/);assert.match(v.failure,/yanıt süresini aşmış/);
 assert.match(v.next,/yeniden deneme bütçesi/);assert.ok(v.checks.every(c=>c.state==='unknown'));assert.equal(v.animate,false);
});

test('single hosted provider CODE timeout does not invent a fallback model or quota failure',()=>{
 const payload={provider_exhausted:true,failure_stage:'CODE',provider_state:{attempts:[
  {phase:'analysis',provider:'COPILOT_HOSTED',model:'auto',exit_code:0,error_class:'NONE'},
  {phase:'code',provider:'COPILOT_HOSTED',model:'auto',exit_code:124,error_class:'TIMEOUT',worktree_changed_files:5}
 ]}};
 const now=Date.now(),stamp=new Date(now).toISOString();
 const view=project({observed_at:stamp,current_run:{run_id:'timed-out',status:'BLOCKED',phase:'BLOCKED',updated_at:stamp,commit_sha:null,recent_events:[{event_kind:'BLOCKED',passed:false,payload}]}},null,null,now);
 assert.match(view.failure,/kod üretiminde yanıt süresini aşmış/);
 assert.doesNotMatch(view.failure,/yedek|kotası/);
 assert.equal(view.animate,false);
 assert.ok(view.checks.every(check=>check.state==='unknown'));
});

test('exit 137 alone is a terminated process, not proven timeout or memory exhaustion',()=>{
 const message=providerExplanation({provider_exhausted:true,provider_state:{attempts:[{phase:'code',exit_code:137,error_class:'UNKNOWN'}]}});
 assert.match(message,/sonlandırılmış/);
 assert.doesNotMatch(message,/süresini aşmış|bellek|yedek/);
});

test('explicit timeout classification remains evidence when exit code differs',()=>{
 const message=providerExplanation({provider_exhausted:true,provider_state:{attempts:[{phase:'analysis',exit_code:75,error_class:'TIMEOUT'}]}});
 assert.match(message,/inceleme aşamasında yanıt süresini aşmış/);
 assert.doesNotMatch(message,/kod üretimi tamamlanmadı/);
});

test('successful attempts and malformed empty receipts cannot invent failure causes',()=>{
 const message=providerExplanation({provider_exhausted:true,provider_state:{attempts:[null,{phase:'analysis',exit_code:0,error_class:'QUOTA_EXHAUSTED'}]}});
 assert.doesNotMatch(message,/kotası dolmuş/);
 assert.equal(providerExplanation({provider_exhausted:false}),null);
});
