import test from 'node:test';import assert from 'node:assert/strict';
import {organicPolicy,organicEventAllowed} from './organic-motion-model.mjs';
test('mobile and adaptive quality bound added carriers without new render targets',()=>{
 assert.equal(organicPolicy({fresh:true}).paths,24);assert.equal(organicPolicy({fresh:true,mobile:true}).paths,8);
 const low=organicPolicy({fresh:true,quality:.55});assert.equal(low.paths,4);assert.equal(low.breathing,false);
});
test('pause, reduced motion and offscreen stop the new layer; stale and replay stop inward traffic',()=>{
 for(const input of [{paused:true},{visible:false}]){const p=organicPolicy({fresh:true,...input});assert.equal(p.flowing,false);assert.equal(p.breathing,false);}
 assert.equal(organicPolicy({fresh:false}).flowing,false);assert.equal(organicPolicy({fresh:true,replay:true}).flowing,false);
});
test('only current observed live or revised records can pulse the environment',()=>{
 const node={state:'current'},g={valid:true,fresh:true};
 for(const mode of ['live','revision'])assert.equal(organicEventAllowed(mode,node,g),true);
 for(const mode of ['baseline','import','replay'])assert.equal(organicEventAllowed(mode,node,g),false);
 assert.equal(organicEventAllowed('live',node,{...g,fresh:false}),false);
 assert.equal(organicEventAllowed('live',{state:'stale'},g),false);
 assert.equal(organicEventAllowed('live',node,g,{replay:true}),false);
});
