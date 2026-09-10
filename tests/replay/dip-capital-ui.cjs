const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const nodes=new Map();function el(id){if(!nodes.has(id))nodes.set(id,{value:'500',addEventListener(k,f){this[k]=f;}});return nodes.get(id);}
const c=vm.createContext({document:{getElementById:el},window:{addEventListener(){}},localStorage:{getItem(){return null}},console});
vm.runInContext(fs.readFileSync('monster-coins-pro/dip-v83.js','utf8'),c);
vm.runInContext(`renderHealth=renderThesis=renderKpis=renderLog=renderChart=()=>{};position=()=>null;model.session={session_id:'old',status:'RUNNING',starting_equity:500};bind();render();`,c);
assert.equal(el('capitalInput').disabled,false);el('capitalInput').value='1200';el('capitalInput').input();
vm.runInContext('render();render();',c);assert.equal(el('capitalInput').value,'1200');
vm.runInContext(`engineToken=()=>'';toast=()=>{};loadStatus=async()=>{};trader=async(action,body)=>{globalThis.sent={action,body};};`,c);
(async()=>{await vm.runInContext('start(true)',c);assert.equal(c.sent.action,'restart');assert.equal(c.sent.body.starting_equity,1200);assert.equal(c.sent.body.trade_notional,1200);assert.equal(c.sent.body.config.live_execution,false);vm.runInContext('position=()=>({side:"LONG"});render()',c);assert.equal(el('capitalInput').disabled,true);console.log('capital draft survives polling; requested amount sent; open-position input locked');})();
