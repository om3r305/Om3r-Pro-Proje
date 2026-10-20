const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=name=>fs.readFileSync('monster-coins-pro/'+name,'utf8');
function element(){
 const classes=new Set();
 return {style:{},dataset:{},attributes:{},classList:{contains:x=>classes.has(x),remove:x=>classes.delete(x),toggle(x,on){if(on)classes.add(x);else classes.delete(x)}},setAttribute(k,v){this.attributes[k]=v}};
}
function beacon(events){
 const tiles=[element(),element(),element()];let sync;
 const document={hidden:false,getElementById:()=>null,createElement:()=>({}),head:{appendChild(){}},body:{},querySelector:()=>tiles[0],querySelectorAll:()=>tiles,addEventListener(){}};
 const window={meetingV6MajorEvents:()=>events};
 vm.runInNewContext(source('frontier-meeting-alert-beacon.js'),{window,document,MutationObserver:class{constructor(fn){sync=fn}observe(){}},setInterval(){},Date});
 return {tiles,sync};
}
test('a fresh critical meeting lights every entry point, not only the first shortcut',()=>{
 const {tiles}=beacon([{time:new Date().toISOString(),urgency:'CRITICAL'}]);
 for(const tile of tiles){assert.equal(tile.dataset.meetingAlert,'red');assert.equal(tile.classList.contains('meeting-alert-red'),true);assert.match(tile.attributes['aria-description'],/Kritik/)}
});
test('expired, future and malformed events never raise an alarm; high is amber',()=>{
 const events=[{time:new Date(Date.now()-3*3600000).toISOString(),urgency:'CRITICAL'},{time:new Date(Date.now()+60000).toISOString(),urgency:'CRITICAL'},{time:'unknown',urgency:'CRITICAL'}];
 const {tiles,sync}=beacon(events);
 assert.equal(tiles[0].dataset.meetingAlert,'calm');
 events.push({time:new Date().toISOString(),urgency:'HIGH'});sync();
 for(const tile of tiles){assert.equal(tile.dataset.meetingAlert,'orange');assert.equal(tile.classList.contains('meeting-alert-red'),false)}
 events.length=0;sync();for(const tile of tiles)assert.equal(tile.dataset.meetingAlert,'calm');
});
test('mobile rescue preserves the scroll lock while accounting or engineering is open',()=>{
 for(const active of ['#treasuryAccountingRoom.show','#brianEngineeringRoomV2.show']){
  const body=element(),root=element();body.style.overflow='hidden';root.style.overflow='hidden';
  const document={body,documentElement:root,readyState:'complete',visibilityState:'visible',querySelector:s=>s===active?element():null,querySelectorAll:()=>[],addEventListener(){}};
  vm.runInNewContext(source('frontier-v4-mobile-rescue.js'),{document,window:{addEventListener(){}},requestAnimationFrame:()=>{},setTimeout(){},MutationObserver:class{observe(){}}});
  assert.equal(body.style.overflow,'hidden');assert.equal(root.style.overflow,'hidden');assert.equal(body.classList.contains('frontier-overlay-open'),true);
 }
});
test('mobile rescue restores scrolling when every room is closed',()=>{
 const body=element(),root=element();body.style.overflow='hidden';root.style.overflow='hidden';
 const document={body,documentElement:root,readyState:'complete',querySelector:()=>null,querySelectorAll:()=>[],addEventListener(){}};
 vm.runInNewContext(source('frontier-v4-mobile-rescue.js'),{document,window:{addEventListener(){}},requestAnimationFrame:()=>{},setTimeout(){},MutationObserver:class{observe(){}}});
 assert.equal(body.style.overflow,'');assert.equal(root.style.overflow,'');
});
test('missing system state is not rendered as running or made actionable',()=>{
 const script=source('frontier-v3.js');
 const controls=script.slice(script.indexOf('function renderControls(){'),script.indexOf('function renderAlerts('));
 const nodes=new Map();const $=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id)};
 const amounts=[element(),element()];
 vm.runInNewContext(controls+';renderControls();',{$,$$:()=>amounts,sysControl:()=>({})});
 for(const id of ['startSystem','restartSystem','stopSystem'])assert.equal($(id).disabled,true);
 assert.equal($('startSystem').textContent,'DURUM BEKLENİYOR');
 assert.doesNotMatch($('controlFeedback').textContent,/Brian açık/);
 assert.ok(amounts.every(b=>b.disabled));
});
test('overlapping engineering refreshes share one request and a failed request can be retried',async()=>{
 const script=source('frontier-engineer-console.js');
 const refresh=script.slice(script.indexOf('let engineerRefreshInFlight='),script.indexOf('function renderEngineerConsole(){'));
 let resolve,reject,calls=0;
 const ENGINEER={data:null,error:null};
 const ctx=vm.createContext({document:{hidden:false},key:()=>true,ENGINEERING_STATUS_ENDPOINT:'read-only-status',ENGINEER,renderEngineerConsole(){},post:()=>{calls++;return new Promise((yes,no)=>{resolve=yes;reject=no})}});
 vm.runInContext(refresh,ctx);
 const first=ctx.refreshEngineerConsole(),second=ctx.refreshEngineerConsole();assert.equal(calls,1);assert.equal(first,second);
 reject(new Error('timeout'));await first;assert.equal(ENGINEER.error,'timeout');
 const retry=ctx.refreshEngineerConsole();assert.equal(calls,2);resolve({status:'OK'});await retry;assert.equal(ENGINEER.error,null);assert.equal(ENGINEER.data.status,'OK');
});
