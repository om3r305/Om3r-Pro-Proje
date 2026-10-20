import test from 'node:test';import assert from 'node:assert/strict';import {healthFor,journeyFor} from './journey-model.mjs';
const node={id:'news',label:'Haber',path:'/news_pipeline',state:'current',at:'2026-10-20T20:00:00Z',raw:{},record_id:'news-1'};
const graph={valid:true,fresh:true,nodes:[node],edges:[{from:'brian',to:'news',path:'/news_pipeline',type:'contains-record'}]};
test('current source animates; stale module stops',()=>{assert.equal(healthFor(node,graph).flow,1);assert.equal(healthFor({...node,state:'stale'},graph).flow,0)});
test('transport failure stops all flows even with a current node',()=>{assert.equal(healthFor(node,graph,{transportDegraded:true}).key,'offline');assert.equal(healthFor(node,graph,{transportDegraded:true}).flow,0)});
test('stale envelope, replay and failed module never animate',()=>{assert.equal(healthFor(node,{...graph,fresh:false}).flow,0);assert.equal(healthFor(node,graph,{replay:true}).flow,0);assert.equal(healthFor({...node,state:'failed'},graph).flow,0)});
test('membership is shown without inventing a news-to-decision causal chain',()=>{const j=journeyFor(graph,'news');assert.equal(j.relations[0].kind,'YANIT ÜYELİĞİ');assert.equal(j.causalChain,false);assert.equal(j.relations.length,1)});
test('only unique exact record IDs create a reference link',()=>{const a={...node,raw:{source_decision_id:'decision-1'}};const b={...node,id:'alpha',record_id:'decision-1'};const g={...graph,nodes:[a,b]};assert.equal(journeyFor(g,'news').relations.length,2);assert.equal(journeyFor({...g,nodes:[a,b,{...b,id:'other'}]},'news').relations.length,1)});
test('empty graph has no fabricated journey',()=>assert.equal(journeyFor({nodes:[],edges:[]},'brian'),null));
