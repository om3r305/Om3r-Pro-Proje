const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const script=fs.readFileSync('monster-coins-pro/shadow-scorecard.js','utf8');
const report=engine_id=>({engine_id,verdict:'COLLECTING',rules:{minimum_closed_trades:100,minimum_days:7,review_drawdown_limit:.02},observations:[],issues:[],recent_trades:[]});
async function page(pathname){
 const body={innerHTML:'',querySelectorAll:()=>[]};
 const button={};
 const panel={querySelector:s=>s==='#scContent'?body:button,getBoundingClientRect:()=>({top:0,bottom:100})};
 const payload={reports:['treasury','dip-multiasset-v1','dip-aggressive-arena-v1','unknown-experiment'].map(report),shadow_only:true,net_scoreboard:[{verdict:'EDGE',horizon_seconds:300,action:'OPEN_LONG'}],trend_shadow:[{nav:10000,target_weights:{},day:'2026-10-10'}],alpha:{compiler_versions:[],recent:[]}};
 const document={hidden:false,querySelector:s=>s==='main'?{prepend:()=>{}}:null,createElement:tag=>tag==='section'?panel:{},head:{appendChild:()=>{}},getElementById:()=>null,addEventListener:()=>{}};
 const ctx={document,location:{pathname},localStorage:{getItem:()=> 'test-key'},fetch:async()=>({ok:true,json:async()=>payload}),AbortSignal,setInterval:()=>{}};
 vm.runInNewContext(script,ctx);
 await new Promise(resolve=>setImmediate(resolve));
 return {html:body.innerHTML,heading:panel.innerHTML};
}
test('main dashboard renders only Brian reports, excluding DIP and unknown engines',async()=>{
 const {html,heading}=await page('/frontier-v4.html');
 assert.match(heading,/BRIAN \/ İLERİYE/);assert.doesNotMatch(heading,/BRIAN \+ DIP/);
 assert.match(html,/Brian · Hazine/);assert.doesNotMatch(html,/DIP ·|unknown-experiment/);
 assert.match(html,/Trend motoru/);assert.match(html,/ALPHA ne bekledi/);
 assert.match(html,/varsayımsal karar sonuçları/);assert.doesNotMatch(html,/Komisyon sonrası kârlı · anlamlı/);
});
test('all supported DIP URLs exclude Brian treasury, ALPHA and trend results',async()=>{
 for(const path of ['/dip','/dip.html','/dip/']){
  const {html,heading}=await page(path);
  assert.match(heading,/DIP \/ İLERİYE/);assert.match(html,/DIP · Ana motor/);assert.match(html,/DIP · Arena/);
  assert.doesNotMatch(html,/Brian · Hazine|Trend motoru|ALPHA ne bekledi|varsayımsal karar sonuçları|unknown-experiment/);
 }
});
test('root and other pages default to Brian results',async()=>{
 for(const path of ['/','/dip-other']){
  const {html}=await page(path);assert.match(html,/Brian · Hazine/);assert.doesNotMatch(html,/DIP ·/);
 }
});
