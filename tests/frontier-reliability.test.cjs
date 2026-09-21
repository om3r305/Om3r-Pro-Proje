const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const recent=new Date().toISOString();
function heartbeat(){return {status:'OK',observed_at:recent,control:{system_enabled:true,treasury:{equity_usd:5000,observed_at:recent}},alpha:{observed_at:recent,asset_id:'crypto:BTCUSDT'},world_run:{status:'SUCCESS',finished_at:recent},behavior:{pipeline_status:'SUCCESS',pipeline_observed_at:recent},news_pipeline:{status:'HEALTHY'},collectors:Object.fromEntries(['brian-world-brain-v1','brian-alpha-decision-compiler-v2','brian-evolution-treasury-v1','brian-evolution-orchestrator-v1','brian-evolution-ocean-worker-v1'].map(id=>[id,{status:'SUCCESS',finished_at:recent}]))};}
function truth(hb){const elements={newsBadge:{},criticalNews:{},ticker:{}};const ctx=vm.createContext({STABILITY:{heartbeat:hb},localStorage:{getItem:()=>null,setItem:()=>{}},document:{getElementById:id=>elements[id],querySelector:()=>elements.ticker},window:{__FRONTIER_COMPOSED_BOOT__:true},renderNews:()=>{},news:()=>[],render:()=>{},setTimeout:()=>{},moduleRows:()=>[]});vm.runInContext(fs.readFileSync('monster-coins-pro/frontier-status-truth.js','utf8'),ctx);return {ctx,elements};}
test('healthy pipeline is green, partial news makes world warning without disabling ALPHA',()=>{let {ctx}=truth(heartbeat());assert.equal(ctx.moduleRows().find(x=>x.key==='world').state,'ok');const hb=heartbeat();hb.news_pipeline={status:'DEGRADED',direct_wire:{status:'SUCCESS',last_run_at:recent}};({ctx}=truth(hb));assert.equal(ctx.moduleRows().find(x=>x.key==='world').state,'warn');assert.equal(ctx.moduleRows().find(x=>x.key==='alpha').state,'ok');});
test('cached transport failure cannot display fully healthy modules',()=>{const hb=heartbeat();hb.transport_degraded=true;const {ctx}=truth(hb);assert.ok(ctx.moduleRows().every(x=>x.state!=='ok'));});
test('missing headlines are distinguished from missing coverage',()=>{const hb=heartbeat();hb.news_pipeline.status='DEGRADED';const {ctx,elements}=truth(hb);ctx.renderNews();assert.equal(elements.newsBadge.textContent,'KAPSAM KISMİ');assert.match(elements.criticalNews.innerHTML,/olmadığı anlamına gelmez/);const healthy=truth(heartbeat());healthy.ctx.renderNews();assert.match(healthy.elements.criticalNews.innerHTML,/yeni kritik gelişme yok/);});
test('operator stop remains stopped',()=>{const hb=heartbeat();hb.control.system_enabled=false;assert.ok(truth(hb).ctx.moduleRows().every(x=>x.state==='info'));});
test('empty partial news preserves ticker so the next render reaches live alerts',()=>{
 const hb=heartbeat();hb.news_pipeline.status='DEGRADED';
 const {ctx,elements}=truth(hb);
 const track={innerHTML:''};elements.tickerTrack=track;
 Object.defineProperty(elements.ticker,'innerHTML',{set(){delete elements.tickerTrack;}});
 ctx.$=id=>elements[id];ctx.ageSec=()=>0;ctx.age=()=> '1 dk';ctx.esc=x=>String(x);
 const source=fs.readFileSync('monster-coins-pro/frontier-v4.js','utf8');
 vm.runInContext(source.slice(source.indexOf('renderNews = function(){'),source.indexOf('function meetingSnapshotV4()')),ctx);
 vm.runInContext(fs.readFileSync('monster-coins-pro/frontier-status-truth.js','utf8'),ctx);
 let alerts=0;ctx.renderAlerts=()=>alerts++;
 ctx.renderNews();assert.equal(elements.tickerTrack,track);
 assert.match(track.innerHTML,/kapsamı kısmi/);
 ctx.news=()=>[{title:'Yeni haber',urgency:'HIGH',time:recent,summary:'Kaynaklı kayıt'}];
 vm.runInContext('renderNews(); renderAlerts(moduleRows());',ctx);
 assert.equal(alerts,1);assert.match(track.innerHTML,/Yeni haber/);
 assert.equal(elements.newsBadge.textContent,'KAPSAM KISMİ');
});
test('overlapping refreshes perform one heartbeat request and release the lock',async()=>{
 let release,calls=0;const elements={syncText:{}};
 const ctx=vm.createContext({window:{__FRONTIER_COMPOSED_BOOT__:true},key:()=>true,unlock:()=>{},$:id=>elements[id],moduleRows:()=>[],refreshAutonomyV4:()=>{},renderAutonomyV4:()=>{},renderTreasuryV4:()=>{},renderMeetingV4:()=>{},CONTROL:'control',EP:{},S:{errors:{}},sysControl:()=>({}),num:x=>x,render:()=>{},AbortController,setTimeout,clearTimeout,fetch:()=>{calls++;return new Promise(r=>{release=()=>r({ok:true,json:async()=>heartbeat()})})}});
 vm.runInContext(fs.readFileSync('monster-coins-pro/frontier-stability.js','utf8'),ctx);
 vm.runInContext('STABILITY.lastSlow=Date.now()',ctx);
 const first=ctx.refresh(),second=ctx.refresh();assert.equal(calls,1);release();await Promise.all([first,second]);assert.equal(vm.runInContext('STABILITY.refreshing',ctx),false);
});
