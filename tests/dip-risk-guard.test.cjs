const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {stripTypeScriptTypes}=require('node:module');
const path=require('node:path');
const root=path.resolve(__dirname,'../supabase/functions');

function load(name,extra={}){
  const source=fs.readFileSync(path.join(root,name,'index.ts'),'utf8').replace(/^import .*;\s*$/mg,'');
  const context=vm.createContext({Deno:{env:{get:()=>''},serve:()=>{}},console,Date,Math,Map,Set,crypto:require('node:crypto').webcrypto,Response,Request,AbortSignal,...extra});
  vm.runInContext(stripTypeScriptTypes(source),context);
  return context;
}
function sqlMock(){
  const calls=[];
  const sql=async(strings,...values)=>{calls.push({query:strings.join('?'),values});return[];};
  sql.json=x=>x;sql.calls=calls;return sql;
}
function arenaState(overrides={}){
  return {enabled:true,run_until:new Date(Date.now()+3600000).toISOString(),starting_equity:1000,cash:1000,positions:{},cooldowns:{},last_scan:{},...overrides};
}
function candidate(explosion=.96){
  const forecast={continuation:.95,expected_5m_bps:100,expected_15m_bps:150,expected_30m_bps:200,expected_60m_bps:200};
  return {c:{symbol:'TESTUSDT'},m:{bid:100,ask:100.02,mid:100.01,spreadBps:2,atrPct:1,lastBarCloseTime:Date.now()-1000,forecast},ev:{forecast,explosionScore:explosion,brainUtility:.95,forecastUtility:.95,opp:.9,net:100,cost:25,gates:Object.fromEntries(['spread','day_move','forecast_consistency','horizon_alignment','regime_guard','shock_memory','post_pump_reset','long_extension','economics'].map(k=>[k,true]))}};
}
async function arenaRun(overrides={},explosion=.96,previousAge=60000){
  const w=load('brian-dip-multiasset-worker-v860'),x=candidate(explosion),sql=sqlMock();
  const state=arenaState({last_scan:{candidate_memory:{TESTUSDT:{last_seen:Date.now()-previousAge,last_evidence_bar_close:x.m.lastBarCloseTime-60000,streak:1}}},...overrides});
  const result=await w.processArena(sql,state,[x],new Map([['TESTUSDT',x.m]]),{riskOff:false},Date.now());
  return {result,sql};
}

test('even ultra-monster allocation obeys cash, notional and modeled risk caps',async()=>{
  const {result}=await arenaRun();
  assert.equal(result.positions.length,1);
  const p=result.positions[0];
  assert.ok(p.notional<=100);
  assert.ok(p.notional*(.05+.0025)<=5);
  assert.ok(result.cash>=900);
  assert.equal(result.recent_events[0].reason,'ARENA_MONSTER_RISK_CAPPED');
});
test('regular candidates pass the same risk budget',async()=>{
  const {result}=await arenaRun({},.85);
  assert.equal(result.positions.length,1);
  assert.ok(result.positions[0].notional<=100);
});
test('existing 3.06% arena loss blocks new entries without erasing history',async()=>{
  const {result}=await arenaRun({cash:969.433984893621,realized_pnl:-30.566015106379496,trade_count:25,win_count:14,loss_count:11});
  assert.equal(result.positions.length,0);assert.equal(result.trade_count,25);
  assert.equal(result.realized_pnl,-30.566015106379496);
});
test('expired or paused arena never buys',async()=>{
  for(const overrides of [{enabled:false},{run_until:'2026-01-01T00:00:00Z'},{run_until:'invalid'}]){
    const {result}=await arenaRun(overrides);assert.equal(result.positions.length,0);
  }
});
test('budget handles exhausted exposure, larger stop gaps, and invalid inputs',()=>{
  const w=load('brian-dip-multiasset-worker-v860');
  assert.equal(w.arenaRiskBudget(1000,1000,1000,300,995,.05,25),0);
  assert.ok(w.arenaRiskBudget(1000,1000,1000,0,995,.2,25)<=5/.2025);
  assert.equal(w.arenaRiskBudget(1000,1000,1000,0,995,NaN,25),0);
});
test('ended empty session stops both engines and keeps scan evidence',async()=>{
  const w=load('brian-dip-multiasset-worker-v860'),sql=sqlMock();
  w.state=arenaState({run_until:'2026-01-01T00:00:00Z'});w.arena=arenaState();
  vm.runInContext('loadState=async()=>state;loadArenaState=async()=>arena;loadUniverse=async()=>{throw Error("unexpected market scan")}',w);
  const result=await w.run(sql);assert.equal(result.status,'WINDOW_COMPLETE');
  assert.equal(sql.calls.length,1);assert.match(sql.calls[0].query,/last_scan=last_scan\|\|/);
  assert.ok(sql.calls[0].values.includes('dip-multiasset-v1'));
  assert.ok(sql.calls[0].values.includes('dip-aggressive-arena-v1'));
});
test('expired session with exposure reaches management instead of abandoning it',async()=>{
  const w=load('brian-dip-multiasset-worker-v860');
  w.state=arenaState({run_until:'2026-01-01T00:00:00Z',positions:{TESTUSDT:{}}});w.arena=arenaState();
  vm.runInContext('loadState=async()=>state;loadArenaState=async()=>arena;loadUniverse=async()=>{throw Error("MANAGEMENT_REACHED")}',w);
  await assert.rejects(()=>w.run(sqlMock()),/MANAGEMENT_REACHED/);
  assert.equal(w.arena.enabled,false);
});

async function guardianTick({bid=102,trail=103,harvest=false,enabled=true}={}){
  const events=[],updates=[];
  const state={enabled,cash:899.9,realized_pnl:0,trade_count:0,win_count:0,loss_count:0,cooldowns:{},last_scan:{},positions:{TESTUSDT:{symbol:'TESTUSDT',entry:100,qty:1,cost_basis:100.1,stop:95,target:110,max_price:104,trail,profit_lock_active:true,harvest_armed:harvest,opened_at:new Date(Date.now()-600000).toISOString(),last_thesis_at:Date.now(),last_forecast_utility:.95,entry_forecast_utility:.95,peak_forecast_utility:.95,last_continuation:.95,last_expected_15m_bps:100,last_expected_30m_bps:150}}};
  const db={from:()=>({select:()=>({eq:()=>({maybeSingle:async()=>({data:state})})}),update:patch=>({eq:async()=>{updates.push(patch);return{};}})})};
  const w=load('brian-dip-position-guardian',{createClient:()=>db});
  w.testBid=bid;w.record=e=>events.push(e);
  vm.runInContext('acquireLease=async()=>({});releaseLease=async()=>{};processArenaGuardian=async()=>({});book=async()=>({bid:testBid,ask:testBid+.02,spreadBps:2});insertEvent=async(e)=>record(e)',w);
  await w.tick(false);return {events,updates};
}
test('strong forecast cannot veto a previously armed profit floor',async()=>{
  const {events}=await guardianTick();assert.equal(events.length,1);
  assert.equal(events[0].reason,'PROFIT_RATCHET');assert.equal(events[0].metadata.brain_strong,true);
  assert.ok(events[0].price<=102); // Never invent a fill at the missed 103 trail.
});
test('strong runner above its protected floor remains open',async()=>{
  const {events,updates}=await guardianTick({bid:104,trail:102});
  assert.equal(events.length,0);assert.ok(updates[0].positions.TESTUSDT);
});
test('harvest floor and paused-position management still close at observed prices',async()=>{
  const {events}=await guardianTick({harvest:true,enabled:false});
  assert.equal(events.length,1);assert.equal(events[0].reason,'HARVEST_TRAIL');assert.ok(events[0].price<=102);
});

test('status tolerates the real three-minute scan cadence but reports frozen and expired arena',async()=>{
  const w=load('brian-dip-multiasset-status');
  const now=Date.now();
  const main={engine_id:'dip-multiasset-v1',...arenaState(),started_at:new Date(now-3600000).toISOString(),updated_at:new Date(now-180000).toISOString(),last_scan:{status:'RUNNING'}};
  const arena={...main,engine_id:'dip-aggressive-arena-v1',last_scan:{status:'RISK_FROZEN',risk_reason:'SESSION_DRAWDOWN'}};
  const sql=async(strings)=>strings.join('').includes('positions,last_scan')?[main,arena]:[];
  let result=await w.statusPayload(sql);assert.equal(result.status,'RUNNING');assert.equal(result.arena.status,'RISK_FROZEN');
  arena.run_until=new Date(now-1000).toISOString();result=await w.statusPayload(sql);assert.equal(result.arena.status,'STOPPED');
});

test('arena confirmation survives a three-minute cycle but not a missed cycle',async()=>{
  assert.equal((await arenaRun({},.85,190000)).result.positions.length,1);
  assert.equal((await arenaRun({},.85,370000)).result.positions.length,0);
});

test('worker gap exits never fill above the observed bid',()=>{
  const w=load('brian-dip-multiasset-worker-v860');
  for(const [reason,p,bid] of [
    ['STOP',{stop:95},90],
    ['PROFIT_RATCHET',{trail:103},101],
    ['HARVEST_TRAIL',{trail:103},101],
    ['PROTECT_TRAIL',{trail:103},101]
  ]) assert.ok(w.triggerFill(reason,p,{bid},2)<=bid);
  assert.equal(w.triggerFill('STOP',{stop:95},{bid:90},2),90*.9996);
});
