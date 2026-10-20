import test from 'node:test';
import assert from 'node:assert/strict';
import {atmosphereFor} from './atmosphere-model.mjs';
const graph=(nodes=[],fresh=true)=>({valid:true,fresh,nodes});
const alpha={id:'alpha',state:'current',path:'/alpha',raw:{},at:new Date().toISOString()};
test('opening action is a candidate, never a strong opportunity by itself',()=>{
 assert.equal(atmosphereFor({alpha:{action:'OPEN_LONG',evidence_score:99}},graph([alpha])).key,'candidate');
});
test('strong coloring requires the explicit source strength and opening action',()=>{
 assert.equal(atmosphereFor({alpha:{action:'OPEN_LONG',opportunity_strength:'STRONG'}},graph([alpha])).key,'opportunity');
 assert.equal(atmosphereFor({alpha:{action:'VETO',opportunity_strength:'STRONG'}},graph([alpha])).key,'caution');
});
test('stale and replay sources cannot produce live opportunity colors',()=>{
 assert.equal(atmosphereFor({},graph([alpha],false)).key,'unverified');
 assert.equal(atmosphereFor({},graph([alpha]),true).key,'replay');
});
test('critical source severity overrides an opening candidate',()=>{
 const node={...alpha,id:'news',label:'Haber hattı',raw:{severity:'CRITICAL'}};
 assert.equal(atmosphereFor({alpha:{action:'OPEN_LONG'}},graph([alpha,node])).key,'critical');
});
test('module failure is labeled as a system fault and expires',()=>{
 const node={...alpha,id:'collector',label:'Toplayıcı',state:'failed',raw:{status:'FAILED'}};
 assert.equal(atmosphereFor({},graph([node])).key,'fault');
 assert.equal(atmosphereFor({},graph([{...node,at:'2020-01-01T00:00:00Z'}])).key,'neutral');
});
test('a missing alpha field is safe without an alpha node',()=>{
 assert.equal(atmosphereFor({},graph()).key,'neutral');
});
