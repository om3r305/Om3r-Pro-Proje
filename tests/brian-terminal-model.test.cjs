const{test}=require('node:test');
const assert=require('node:assert/strict');
const{radar}=require('../monster-coins-pro/brian-terminal-model.js');
const now=Date.parse('2026-10-20T08:30:00Z');
const record={status:'ONLINE',observed_at:new Date(now-60000).toISOString(),hot:[{symbol:'BTCUSDT',price_change_pct:2.4}]};
test('radar preserves actual percentage units and negative returns',()=>{
 assert.equal(radar(record,now).rows[0].change,2.4);
 assert.equal(radar({...record,hot:[{symbol:'ETHUSDT',price_change_pct:'-1.25'}]},now).rows[0].change,-1.25);
 assert.equal(radar(record,now).current,true);
});
test('missing and invalid returns remain unknown, including boolean and empty values',()=>{
 for(const value of [null,undefined,'',false,true,NaN,Infinity,{}])assert.equal(radar({...record,hot:[{price_change_pct:value}]},now).rows[0].change,null);
 assert.deepEqual(radar(null,now).rows,[]);assert.equal(radar(null,now).current,false);
});
test('stale, future, missing and expired timestamps cannot claim current market data',()=>{
 for(const change of [{status:'STALE'},{stale:true},{observed_at:null},{observed_at:new Date(now+1).toISOString()},{observed_at:new Date(now-900001).toISOString()}])assert.equal(radar({...record,...change},now).current,false);
});
test('malformed rows are ignored, results bounded, source never mutated',()=>{
 const input={...record,hot:[],candidates:[null,'bad',[],...Array.from({length:12},(_,i)=>({symbol:'A'+i,price_change_pct:i}))]};
 const before=JSON.stringify(input),result=radar(input,now);
 assert.equal(result.rows.length,8);assert.equal(result.rows[0].symbol,'A0');assert.equal(JSON.stringify(input),before);
});
