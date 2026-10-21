import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {recordSummary} from './record-details.mjs';
import {createEventHistory} from './event-history.mjs';
const require=createRequire(import.meta.url),{project}=require('./observatory-model.js');
test('core and asset summaries expose actual nonnumeric fields without inventing metrics',()=>{
 assert.deepEqual(recordSummary({status:'OK',read_only:true,live_execution:false}),[['status','OK'],['read_only',true],['live_execution',false]]);
 assert.deepEqual(recordSummary({asset_id:'crypto:BTCUSDT',action:'WAIT'}),[['asset_id','crypto:BTCUSDT'],['action','WAIT']]);
 assert.deepEqual(recordSummary({evidence_score:0,reason:'real',direction:1}),[['evidence_score',0]]);
 assert.deepEqual(recordSummary({missing:null,bad:NaN,nested:{a:1}}),[]);
});
test('core history stays with core and actual decision identity survives projection',()=>{
 const at='2026-10-21T15:00:00Z',g=project({status:'OK',read_only:true,shadow_only:true,live_execution:false,observed_at:at,alpha:{decision_id:'decision-actual',observed_at:at,asset_id:'crypto:BTCUSDT',action:'WAIT'}},Date.parse(at));
 assert.equal(g.nodes.find(n=>n.id==='alpha').record_id,'decision-actual');
 const h=createEventHistory();h.observe(g);
 assert.equal(h.list('brian')[0].label,'BRIAN');assert.equal(h.list('brian')[0].path,'/');assert.equal(h.list('brian')[0].arrivalAt,null);
 assert.equal(h.list('alpha')[0].recordId,'decision-actual');assert.equal(h.list('asset')[0].refs[0].raw.decision_id,'decision-actual');
});
