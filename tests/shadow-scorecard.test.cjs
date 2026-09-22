const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {stripTypeScriptTypes}=require('node:module');
const code=stripTypeScriptTypes(fs.readFileSync('supabase/functions/brian-shadow-scorecard/model.ts','utf8')).replace(/export /g,'');
const {summarize,alphaSummary}=vm.runInNewContext(code+';({summarize,alphaSummary})');
const start=Date.parse('2026-09-22T10:00:00Z'),iso=n=>new Date(start+n).toISOString();
const protocol={engine_id:'test',protocol_id:'test-v1',starts_at:iso(0),ends_at:iso(7*86400000),rules:{minimum_closed_trades:100,minimum_days:7,review_drawdown_limit:.02}};
const point=(n,equity=1000)=>({captured_at:iso(n),source_at:iso(n),session_id:'session1',policy_version:'v1',equity,open_positions:0,shadow_only:true,live_execution:false});
const trade=n=>({net:n,policy_version:'v1',shadow_only:true,live_execution:false});
test('net ledger costs are not subtracted twice; zero loss does not invent infinity',()=>{
 const r=summarize(protocol,[point(0),point(300000,1002)],[trade(2)],false,start+300000);
 assert.equal(r.closed_trade_net,2);assert.equal(r.net_equity_change,2);assert.equal(r.profit_factor,null);assert.equal(r.verdict,'COLLECTING');assert.equal(r.automatic_promotion,false);
});
test('session resets cannot appear as profits',()=>{
 const p=point(300000,2000);p.session_id='reset';const r=summarize(protocol,[point(0),p],[],false,start+300000);
 assert.equal(r.net_equity_change,null);assert.equal(r.verdict,'INCOMPLETE');assert.ok(r.issues.includes('SESSION_CHANGED'));
});
test('unknown cost/entry is not silently counted as zero',()=>{
 const r=summarize(protocol,[point(0)],[trade(null),trade(-2)],false,start);
 assert.equal(r.closed_trade_net,null);assert.equal(r.known_trade_net,-2);assert.equal(r.unknown_trades,1);
});
test('stale heartbeat, changed policy and bounded reports block conclusion',()=>{
 const p=point(900000);p.source_at=iso(0);p.policy_version='v2';
 const r=summarize(protocol,[point(0),p],[trade(100)],true,start+900000);
 assert.equal(r.verdict,'INCOMPLETE');assert.equal(r.net_equity_change,null);assert.ok(r.issues.includes('STALE_SOURCE'));assert.ok(r.issues.includes('BOUNDED_REPORT_TRUNCATED'));assert.ok(r.issues.includes('OBSERVATION_GAP'));
});
test('drawdown is measured from running peak, not just initial capital',()=>{
 const r=summarize(protocol,[point(0),point(300000,1100),point(600000,1050)],[],false,start+600000);
 assert.ok(Math.abs(r.sampled_drawdown-50/1100)<1e-10);assert.equal(r.verdict,'RISK_REVIEW');
});
test('empty and expired small tests never claim a positive edge',()=>{
 assert.equal(summarize(protocol,[],[],false,start).verdict,'INCOMPLETE');
 const r=summarize(protocol,[point(0),point(300000,1200)],[trade(200)],false,start+7*86400000);
 assert.notEqual(r.verdict,'REVIEW_POSITIVE');
});
test('a complete positive sample still requires human review and never promotes',()=>{
 const points=Array.from({length:2017},(_,i)=>point(i*300000,1000+i/10));
 const r=summarize(protocol,points,Array.from({length:100},()=>trade(1)),false,start+7*86400000);
 assert.equal(r.verdict,'REVIEW_POSITIVE');assert.equal(r.automatic_promotion,false);
});
test('ALPHA uses decision-time costs once, excludes WAIT and exposes unresolved predictions',()=>{
 const row={action:'OPEN_LONG',compiler_version:'v1',observed_at:iso(0),direction_adjusted_return:.01,estimated_round_trip_cost_bps:20};
 const a=alphaSummary([row,{...row,action:'WAIT'},{...row,direction_adjusted_return:null},{...row,estimated_round_trip_cost_bps:null}],start+7200000);
 assert.equal(a.mean_net_bps,80);assert.equal(a.resolved,1);assert.equal(a.pending,1);assert.equal(a.overdue,1);assert.equal(a.wait_or_veto,1);assert.equal(a.missing_cost,1);
 assert.equal(a.recent[0].result_status,'MISSING_COST');assert.equal(a.recent[1].result_status,'PENDING');assert.equal(a.recent[2].result_status,'NO_TRADE');assert.equal(a.recent[3].resolved_net_bps,80);
});
test('non-shadow trades and observations fail closed',()=>{
 const p=point(0);p.live_execution=true;const r=summarize(protocol,[p],[{...trade(100),shadow_only:false}],false,start);
 assert.equal(r.verdict,'INCOMPLETE');assert.equal(r.net_equity_change,null);assert.equal(r.closed_trade_net,null);
});
