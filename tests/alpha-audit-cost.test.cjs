const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const vm=require('node:vm');const {stripTypeScriptTypes}=require('node:module');
const source=stripTypeScriptTypes(fs.readFileSync('supabase/functions/_shared/alpha_audit_cost.ts','utf8')).replace(/export /g,'');
const select=vm.runInNewContext(source+';selectDecisionAuditCost');
const at='2026-10-18T08:05:00Z';
const quote={role:'AUDIT_ONLY',quality:'L2_OBSERVED',policy:'max_buy_sell_same_snapshot_v1',captured_at:'2026-10-18T08:04:59Z',source_id:'binance_public_rest_depth:BTCUSDT:1',requested_notional_usd:3,estimated_round_trip_cost_bps:21};
test('WAIT/VETO accept causal observed cost without supplying execution cost',()=>{
 for(const action of ['WAIT','VETO']){const r=select(at,action,null,{audit_cost:quote});assert.equal(r.costBps,21);assert.equal(r.source,'observed_audit_only');}
 assert.equal(select(at,'OPEN_LONG',null,{audit_cost:quote}).costBps,null);
});
test('execution cost remains authoritative',()=>{assert.equal(select(at,'WAIT',25,{audit_cost:quote}).costBps,25);});
test('missing, future, stale, degraded or malformed cost remains unresolved',()=>{
 for(const q of [null,{...quote,captured_at:'2026-10-18T08:05:01Z'},{...quote,captured_at:'2026-10-18T08:02:59Z'},{...quote,quality:'DEGRADED_TOP_OF_BOOK'},{...quote,estimated_round_trip_cost_bps:null},{...quote,estimated_round_trip_cost_bps:-1},{...quote,source_id:'guessed'}]) assert.equal(select(at,'WAIT',null,{audit_cost:q}).costBps,null);
});
