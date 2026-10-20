const {test}=require('node:test');
const assert=require('node:assert/strict');
const M=require('../monster-coins-pro/prediction-research-model.js');
const payload=()=>({schema_version:'brian-prediction-status.v1',shadow_only:true,live_execution:false,trade_signal:false,prediction:{status:'OK',snapshots:{total:0,last_24h:null},control:{wallet_tracking_enabled:false}},calibration:{status:'INSUFFICIENT_DATA',report:{events:0,markets:null}},niulai4_pilot:{status:'NOT_CONFIGURED'}});
test('zero observed records remain zero; unknown or missing values are never converted to zero',()=>{
 const v=M.project(payload());assert.equal(v.counts.snapshots,0);assert.equal(v.counts.last24h,null);assert.equal(v.counts.events,0);assert.equal(v.counts.markets,null);assert.equal(v.walletTracking,false);
 assert.equal(M.number(''),null);assert.equal(M.number('not a number'),null);
});
test('incompatible schemas and unsafe execution payloads fail closed',()=>{
 for(const override of [{schema_version:'v2'},{shadow_only:false},{live_execution:true},{trade_signal:true}])assert.equal(M.project({...payload(),...override}),null);
});
test('each unavailable section stays unavailable without suppressing healthy neighbors',()=>{
 const p=payload();p.calibration={status:'UNAVAILABLE'};const v=M.project(p);assert.equal(v.prediction.tone,'ok');assert.equal(v.calibration.status,'UNAVAILABLE');assert.equal(v.counts.events,null);assert.equal(v.pilot.tone,'neutral');
});
test('a stale or missing pilot never appears running; hard stops are red, operator stops amber',()=>{
 for(const status of ['STALE','NOT_STARTED','NOT_CONFIGURED','UNAVAILABLE'])assert.notEqual(M.section({status}).tone,'ok');
 assert.equal(M.section({status:'STOPPED',run:{stop_reason:'ACCESS_DENIED'}}).tone,'bad');
 assert.equal(M.section({status:'STOPPED',run:{stop_reason:'OPERATOR'}}).tone,'warn');
 assert.equal(M.section({status:'RUNNING'}).label,'Gözlemde');
});
