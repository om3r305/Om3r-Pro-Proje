// Read-only Brian V8 audit reproductions. Node 24+, no network/DB credentials.
// Executes the repository's own source in VM contexts with in-memory DB/exchange fixtures.
// Executor cases substitute structure() with a fixed structural signal to isolate lifecycle logic.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { stripTypeScriptTypes } = require('node:module');
const { webcrypto } = require('node:crypto');
const { execFileSync } = require('node:child_process');
const ROOT = path.resolve(process.argv[2] || './brian-v8-audit');
const sourceCommit = execFileSync('git',['-C',ROOT,'rev-parse','HEAD'],{encoding:'utf8'}).trim();
assert.equal(sourceCommit,'0c1b1e88e758a71e7cf9732300b77f8f6e2f597d','Use the audited commit for these defect reproductions.');
const NOW = Date.parse('2026-09-07T12:00:20Z');
const cp = x => JSON.parse(JSON.stringify(x));
const results = [];
class MemoryDB {
  constructor() { this.tables = {}; this.fail = null; this.calls = []; }
  from(table) {
    const db = this; const q = {table, action:'select', filters:[], orders:[], max:Infinity, single:false};
    const b = {
      select() {return b;},
      eq(k,v) {q.filters.push(x=>x[k]===v);return b;},
      is(k,v) {q.filters.push(x=>v===null?x[k]==null:x[k]===v);return b;},
      not(k,op,v) {q.filters.push(x=>v===null?x[k]!=null:x[k]!==v);return b;},
      order(k,o={}) {q.orders.push([k,o.ascending!==false]);return b;},
      limit(n) {q.max=n;return b;},
      maybeSingle() {q.single=true;return b;},
      single() {q.single=true;return b;},
      insert(row) {q.action='insert';q.row=row;return b;},
      update(row) {q.action='update';q.row=row;return b;},
      then(resolve,reject) {return Promise.resolve().then(()=>{
        db.calls.push({table,action:q.action});
        if(db.fail?.(q)) return {data:null,error:{message:'INJECTED_READ_OR_WRITE_FAILURE'}};
        const all = db.tables[table] ||= [];
        if(q.action==='insert') {all.push(cp(q.row));return {data:null,error:null};}
        let rows = all.filter(x=>q.filters.every(f=>f(x)));
        if(q.action==='update') {for(const row of rows)Object.assign(row,cp(q.row));return {data:null,error:null};}
        rows.sort((a,b)=>{for(const [k,asc] of q.orders)if(a[k]!==b[k])return (a[k]<b[k]?-1:1)*(asc?1:-1);return 0;});
        rows=rows.slice(0,q.max);
        return {data:cp(q.single?(rows[0]||null):rows),error:null};
      }).then(resolve,reject);}
    }; return b;
  }
  async rpc() {return {data:true,error:null};}
}
function load(relative, db=new MemoryDB(), epoch=NOW) {
  class AuditDate extends Date {constructor(...a){super(...(a.length?a:[epoch]));}static now(){return epoch;}}
  const context={console,Date:AuditDate,crypto:webcrypto,TextEncoder,TextDecoder,Uint8Array,AbortSignal,
    Request,Response,setInterval:()=>1,clearInterval:()=>{},setTimeout,clearTimeout,
    createClient:()=>db,Deno:{env:{get:()=> 'audit-placeholder'},serve:fn=>{context.handler=fn;}},
    fetch:()=>{throw Error('Unexpected network request: audit is offline');}};
  vm.createContext(context);
  const source=fs.readFileSync(path.join(ROOT,relative),'utf8').replace(/^import .*;\s*$/mg,'');
  vm.runInContext(stripTypeScriptTypes(source,{mode:'transform'}),context,{filename:relative});
  return context;
}
function struct(tf, changes={}) {
  return {tf,lastClose:100,atr:1,pivots:[],lastHigh:{i:9,t:NOW-900000,p:103,kind:'H',label:'HH'},
    lastLow:{i:10,t:NOW-600000,p:99,kind:'L',label:'HL'},trend:'UP',bos:null,choch:null,
    sweep:tf==='1m'?'BULL':null,failedBreak:null,equalHigh:null,equalLow:null,fingerprint:tf+'-fixed',...changes};
}
function worker(opts={}) {
  const db=new MemoryDB();
  db.tables.brian_dip_session_events=[{session_id:'audit-session',event_kind:'START',requested_at:new Date(NOW-10000000).toISOString(),starting_equity:opts.cash??1000,
    config:{engine_version:'brian-dip-chart-reader-v8',symbols:['ETHUSDT'],fee_bps:10,slippage_bps:1,allow_shadow_short:false}}];
  const c=load('supabase/functions/brian-dip-shadow-worker/index.ts',db);
  let mid=opts.mid??100;
  const marketCalls=[];
  c.fetchJson=async route=>{
    marketCalls.push(route);
    if(route.includes('/depth'))return {bids:[[String(mid-.01),'1']],asks:[[String(mid+.01),'1']]};
    if(route.includes('/aggTrades'))return [{p:'100',q:'1',m:true,T:NOW}];
    return Array.from({length:25},(_,i)=>[NOW-(24-i)*300000-20000,100,opts.high??101,opts.low??99.5,100,10]);
  };
  c.structure=async tf=>struct(tf,opts.structure?.(tf)||{});
  return {c,db,marketCalls,setMid:p=>{mid=p;}};
}
function position(overrides={}) {
  return {side:'LONG',thesis_id:'existing-thesis',setup:'SWEEP_RECLAIM',entry:100,qty:.8,notional:80,
    target:103,stop:99,opened_at:new Date(NOW-120000).toISOString(),fees_open:.088,venue:'SPOT',...overrides};
}
function seedSnapshot(db,pos,overrides={}) {
  db.tables.brian_dip_snapshots=[{snapshot_id:'seed',session_id:'audit-session',observed_at:new Date(NOW-60000).toISOString(),cash:1000,
    realized_pnl:0,trade_count:0,win_count:0,loss_count:0,state:{start:1000,v8:{pos,locks:{},latestThesis:null}},...overrides}];
}
function calRows(n,hits=0) {return Array.from({length:n},(_,i)=>({id:'cal-'+i,setup:'SWEEP_RECLAIM',direction:'UP',venue:'SPOT',metric_version:'target-before-invalidation-v8',hit:i<hits,resolved_at:new Date(NOW-1000000+i).toISOString()}));}
async function test(name,fn) {try{const evidence=await fn();results.push({name,verified:true,evidence});console.log(name+': '+JSON.stringify(evidence));}catch(e){results.push({name,verified:false,error:e.stack});console.error(name+': FAILED '+e.stack);}}
(async()=>{
await test('legacy_ui_undefined_function_and_focus_override',async()=>{
  const c={console,params:()=>({config:{}}),restore:()=>{},note:()=>'',v4UiPatch:()=>{},
    addEventListener:()=>{},document:{addEventListener:()=>{}},v4DiscoverUniverse:()=>{},
    v4Ensure:()=>{},v4Universe:['ETHUSDT'],selected:'ETHUSDT',sid:'audit-session',book:{cfg:{}},
    session:{config:{}},states:{},start:async()=>{},renderRadar:()=>{},renderKpi:()=>{},draw:()=>{}};
  vm.createContext(c);
  vm.runInContext(fs.readFileSync(path.join(ROOT,'monster-coins-pro/dip-foresight-v7.js'),'utf8'),c);
  let error='';
  try{vm.runInContext(fs.readFileSync(path.join(ROOT,'monster-coins-pro/dip-focus3-stability-v7.js'),'utf8'),c);}catch(e){error=e.message;}
  c.restore({});
  assert.match(error,/v7Foresight is not defined/);
  assert.equal(c.v4Universe.length,3);
  return {error,universe_after_restore:c.v4Universe};
});
await test('runtime_read_error_silently_resets_state',async()=>{
  const {c,db}=worker();seedSnapshot(db,position(),{cash:700});
  db.fail=q=>q.table==='brian_dip_snapshots'&&q.action==='select';
  const rt=await c.latestRuntime('audit-session',1000);
  assert.equal(rt.cash,1000);assert.equal(rt.pos,null);
  return {expected_existing_cash:700,returned_cash:rt.cash,returned_position:rt.pos};
});
await test('same_cycle_expiry_reopens_locked_thesis',async()=>{
  const {c,db}=worker();const id=await c.hash('ETHUSDT|SWEEP_RECLAIM|UP|99.000000|103.000000');
  seedSnapshot(db,position({thesis_id:id,opened_at:new Date(NOW-91*60000).toISOString()}));
  const r=await c.run();const events=db.tables.brian_dip_events;
  assert.deepEqual(events.map(e=>e.event_kind),['SELL','BUY']);assert.equal(r.position.thesis_id,id);
  return {events:events.map(e=>e.event_kind),exit_reason:events[0].metadata.exit_reason,same_thesis_reopened:true};
});
await test('long_entry_accepts_stop_above_fill',async()=>{
  const {c}=worker({structure:tf=>({lastClose:101,lastLow:{t:NOW-600000,p:100.5,kind:'L',i:10}})});
  const r=await c.run();assert.equal(r.position.side,'LONG');assert.ok(r.position.stop>r.position.entry);
  return {entry:r.position.entry,stop:r.position.stop,target:r.position.target,rr:r.thesis.rr};
});
await test('worker_ignores_intrabar_target',async()=>{
  const {c,db}=worker({high:104,structure:()=>({sweep:null})});seedSnapshot(db,position());
  const r=await c.run();assert.ok(r.position);assert.equal((db.tables.brian_dip_events||[]).length,0);
  return {bar_high:104,position_target:103,current_mid:100,position_still_open:!!r.position};
});
await test('forty_losses_still_enable_calibrated_entry',async()=>{
  const {c,db}=worker();db.tables.brian_dip_foresight=calRows(40,0);const r=await c.run();
  assert.equal(r.thesis.calibrated_probability,0);assert.equal(r.position.notional,50);
  return {samples:r.thesis.calibration_samples,hit_rate:r.thesis.calibrated_probability,notional:r.position.notional,cash:1000};
});
await test('minimum_notional_breaks_eight_percent_cap',async()=>{
  const {c}=worker({cash:50});const r=await c.run();assert.equal(r.position.notional,10);
  return {cash:50,notional:r.position.notional,actual_fraction:r.position.notional/50,stated_cap:.08};
});
await test('sweep_enters_despite_opposing_flow',async()=>{
  const {c}=worker();const r=await c.run();assert.equal(r.thesis.flow.ofi,-1);assert.ok(r.position);
  return {ofi:r.thesis.flow.ofi,raw_conviction:r.thesis.raw_conviction,position:r.position.side};
});
await test('same_thesis_persistence_freezes_calibration_and_entry',async()=>{
  const {c,db,setMid}=worker();db.tables.brian_dip_foresight=calRows(39,39);await c.run();
  db.tables.brian_dip_foresight=calRows(40,40);setMid(100.1);const r=await c.run();
  const persisted=db.tables.brian_dip_theses;
  assert.equal(persisted.length,1);assert.equal(persisted[0].calibrated_probability,null);assert.equal(r.thesis.calibrated_probability,1);
  return {persisted_rows:1,persisted_probability:persisted[0].calibrated_probability,worker_probability:r.thesis.calibrated_probability,persisted_entry_low:persisted[0].entry_low,worker_entry_low:r.thesis.entry_low};
});
await test('event_survives_snapshot_failure_and_duplicates_on_retry',async()=>{
  const {c,db}=worker();let injected=0;
  db.fail=q=>q.table==='brian_dip_snapshots'&&q.action==='insert'&&injected++===0;
  try{await c.run();}catch{}
  await c.run();const buys=db.tables.brian_dip_events.filter(e=>e.event_kind==='BUY');
  assert.equal(buys.length,2);assert.notEqual(buys[0].event_id,buys[1].event_id);
  return {buy_events:buys.length,committed_snapshots:db.tables.brian_dip_snapshots.length,unique_event_ids:true};
});
await test('short_uses_spot_data_and_ignores_disabled_short_setting',async()=>{
  const {c,marketCalls}=worker({structure:tf=>({lastLow:{p:97,t:1,kind:'L',i:1},lastHigh:{p:101,t:2,kind:'H',i:2},trend:'DOWN',sweep:tf==='1m'?'BEAR':null})});
  const r=await c.run();assert.equal(r.position.side,'SHORT');assert.equal(r.position.venue,'PERP');
  assert.ok(marketCalls.every(p=>p.startsWith('/api/v3/')));
  return {allow_shadow_short:false,opened_side:r.position.side,labeled_venue:r.position.venue,all_market_routes:marketCalls};
});
function resolverRow(overrides={}){return {id:'forecast',symbol:'ETHUSDT',session_id:'audit-session',created_at:new Date(NOW-80000).toISOString(),due_at:new Date(NOW+90*60000).toISOString(),direction:'UP',target_price:103,invalidation_price:99,entry_price:100,predicted_close:103,metric_version:'target-before-invalidation-v8',resolved_at:null,...overrides};}
await test('resolver_skips_first_prediction_minute',async()=>{
  const db=new MemoryDB();const created=Math.floor(NOW/60000)*60000;
  db.tables.brian_dip_foresight=[resolverRow({created_at:new Date(created).toISOString()})];
  const c=load('supabase/functions/brian-dip-foresight/index.ts',db);let requests=0;
  c.klines=async()=>{requests++;return [{t:created,o:100,h:104,l:100,c:100}];};
  const resolved=await c.resolveDue();assert.equal(resolved,0);assert.equal(requests,0);
  return {created_at:new Date(created).toISOString(),target_hit_in_first_20_seconds:true,scan_requests:requests,resolved};
});
await test('resolver_can_record_future_resolved_at_from_open_candle',async()=>{
  const db=new MemoryDB();db.tables.brian_dip_foresight=[resolverRow()];
  const c=load('supabase/functions/brian-dip-foresight/index.ts',db);
  c.klines=async()=>[{t:Math.floor(NOW/60000)*60000,o:100,h:104,l:100,c:100}];
  await c.resolveDue();const r=db.tables.brian_dip_foresight[0];assert.ok(Date.parse(r.resolved_at)>NOW);
  return {now:new Date(NOW).toISOString(),resolved_at:r.resolved_at,reason:r.resolution_reason};
});
await test('resolver_includes_bar_remainder_after_deadline',async()=>{
  const db=new MemoryDB(),due=Date.parse('2026-09-07T12:00:20Z'),epoch=due+60000;
  db.tables.brian_dip_foresight=[resolverRow({created_at:new Date(due-90*60000).toISOString(),due_at:new Date(due).toISOString()})];
  const c=load('supabase/functions/brian-dip-foresight/index.ts',db,epoch);
  c.klines=async()=>[{t:Math.floor(due/60000)*60000,o:100,h:104,l:100,c:104}];
  await c.resolveDue();const r=db.tables.brian_dip_foresight[0];assert.equal(r.hit,true);assert.ok(Date.parse(r.resolved_at)>due);
  return {due_at:r.due_at,resolved_at:r.resolved_at,hit:r.hit,note:'OHLC alone cannot establish whether the touch occurred before the 20-second cutoff'};
});
await test('parallel_forecast_persist_duplicates_same_thesis',async()=>{
  const db=new MemoryDB();const c=load('supabase/functions/brian-dip-foresight/index.ts',db);
  const t={thesis_state:'CONFIRMED',direction:'UP',target_price:103,invalidation_price:99,thesis_id:'same-thesis',entry_low:99.9,entry_high:100.1};
  await Promise.all([c.persist('audit-session',t),c.persist('audit-session',t)]);
  assert.equal(db.tables.brian_dip_foresight.length,2);
  return {rows:2,session:'audit-session',thesis_id:'same-thesis',note:'Schema has only random id primary key; no unique session/thesis/metric constraint'};
});
await test('positive_worker_rejects_v7_session',async()=>{
  const {c,db,marketCalls}=worker();db.tables.brian_dip_session_events[0].config.engine_version='v7';
  const r=await c.run();assert.equal(r.status,'WAIT_V8_CLEAN_RESTART');assert.equal(marketCalls.length,0);
  return {status:r.status,market_requests:0};
});
await test('positive_pivot_confirmation_and_open_bar_exclusion',async()=>{
  const c=load('supabase/functions/brian-dip-shadow-worker/index.ts');
  const rows=[1,2,3,6,3,2,1].map((h,i)=>({t:i*60000,o:1,h,l:.5,c:1,v:1}));
  assert.equal(c.classifyPivots(rows.slice(0,6)).filter(p=>p.kind==='H').length,0);
  assert.equal(c.classifyPivots(rows).filter(p=>p.kind==='H').length,1);
  const a=await c.structure('1m',[...rows,{t:420000,o:1,h:100,l:0,c:90,v:1}]);
  const b=await c.structure('1m',[...rows,{t:420000,o:1,h:2,l:0,c:1,v:1}]);
  assert.equal(a.fingerprint,b.fingerprint);
  return {pivot_waits_for_three_right_bars:true,forming_bar_does_not_change_fingerprint:true};
});
fs.writeFileSync(path.join(path.dirname(__filename),'audit-repro-results.json'),JSON.stringify({source_commit:sourceCommit,offline:true,results},null,2));
if(results.some(r=>!r.verified))process.exitCode=1;
})();
