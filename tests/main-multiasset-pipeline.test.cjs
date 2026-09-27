const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const {stripTypeScriptTypes}=require('node:module');
const {webcrypto,createHash}=require('node:crypto');
const root=path.resolve(__dirname,'../supabase/functions');
const {assessMarketSession}=require('../supabase/functions/_shared/multiasset_market_session.ts');
const now=Date.parse('2026-09-25T15:00:00Z');
const start=now/1000-3600,end=now/1000+3600;
const iso=ms=>new Date(ms).toISOString();
test('regular session is verified without an optional marketState field',()=>{
  assert.equal(assessMarketSession(iso(now-60000),start,end,now).sessionState,'REGULAR');
});
test('cached marks age at decision time even if stored latency was low',()=>{
  assert.equal(assessMarketSession(iso(now-901000),start,end,now).reason,'STALE_PROVIDER_PRICE');
});
test('pre/post market and exact closing boundary are closed',()=>{
  for(const t of [start*1000-1,end*1000,end*1000+60000])
    assert.equal(assessMarketSession(iso(t-1000),start,end,t).eligible,false);
});
test('missing, reversed, future and invalid evidence fails closed',()=>{
  for(const args of [[iso(now),null,end,now],[iso(now),end,start,now],['bad',start,end,now],[iso(now+6000),start,end,now],[iso(start*1000-1000),start,end,start*1000+1000]])
    assert.equal(assessMarketSession(...args).eligible,false);
});

function load(name,extras={}){
  let handler;
  const source=fs.readFileSync(path.join(root,name,'index.ts'),'utf8').replace(/^import .*;\s*$/mg,'');
  class Clock extends Date { constructor(...args){super(...(args.length?args:[now]));} static now(){return now;} }
  const ctx=vm.createContext({Deno:{env:{get:()=>''},serve:fn=>handler=fn},createClient:()=>({}),assessMarketSession:(p,s,e)=>assessMarketSession(p,s,e,now),Date:Clock,Response,Request,AbortSignal,TextEncoder,crypto:webcrypto,console,...extras});
  vm.runInContext(stripTypeScriptTypes(source),ctx);
  return {ctx,handle:req=>handler(req)};
}
test('collector derives REGULAR from actual provider session, not OPEN fallback',async()=>{
  const {ctx}=load('brian-realtime-multiasset-market-eye',{fetch:async()=>new Response(JSON.stringify({chart:{result:[{meta:{regularMarketPrice:100,regularMarketTime:now/1000-5,currentTradingPeriod:{regular:{start,end}}}}]}}))});
  const mark=await ctx.fetchOne({asset_id:'commodity:GOLD',asset_class:'commodity',symbol:'GC=F',themes:[],priority:100});
  assert.equal(mark.session_state,'REGULAR');
  assert.equal(mark.metadata.regular_session_end,end);
  assert.equal(mark.live_execution,false);
});
async function runEngine({age=5,sessionEnd=end,claim='Gold bullion reserves expanded',duplicate=false,reaction=.01}={}){
  const writes={};
  const events=[{event_id:'news-1',observed_at:iso(now-60000),published_at:iso(now-60000),source_id:'official:test',claim,primary_asset:null}];
  if(duplicate)events.push({...events[0]});
  const db={from:table=>{
    const q={select:()=>q,eq:()=>q,gte:()=>q,order:()=>q,limit:()=>q,single:()=>q,
      insert:async rows=>{writes[table]=rows;return {error:null}},
      then:resolve=>resolve({data:table==='brian_dashboard_auth'?{cron_key_sha256:createHash('sha256').update('test-key').digest('hex')}:events,error:null})};
    return q;
  }};
  const mark={asset_id:'commodity:GOLD',asset_class:'commodity',provider_time:iso(now-age*1000),price:100,return_5m:reaction,return_1h:reaction,session_state:'OPEN',data_latency_seconds:1,provider_quality:'PUBLIC_UNOFFICIAL_SHADOW_ONLY',metadata:{themes:['MONETARY','FINANCIAL','FX','GEOPOLITICAL'],regular_session_start:start,regular_session_end:sessionEnd}};
  const {handle}=load('brian-multiasset-opportunity-engine',{createClient:()=>db,fetch:async()=>new Response(JSON.stringify({status:'SUCCESS',marks:[mark],crowd:[]}))});
  const response=await handle(new Request('https://test.local',{method:'POST',headers:{'x-brian-cron-key':'test-key'}}));
  assert.equal(response.status,200);
  return writes.brian_multiasset_alpha_decisions[0];
}
test('gold reserves headline reaches a bounded shadow decision',async()=>{
  const d=await runEngine();
  assert.equal(d.action,'OPEN_LONG');assert.equal(d.session_state,'REGULAR');
  assert.ok(d.requested_virtual_notional_usd<=20);assert.equal(d.shadow_only,true);assert.equal(d.live_execution,false);
  assert.equal(d.linked_event_ids[0],'news-1');
});
test('same event through multiple themes or duplicate frames contributes once',async()=>{
  const one=await runEngine({claim:'Gold reserves dollar bank inflation'});
  const repeated=await runEngine({claim:'Gold reserves dollar bank inflation',duplicate:true});
  assert.equal(one.metadata.event_strength,repeated.metadata.event_strength);
  assert.ok(one.metadata.event_strength<.46);
});
test('stale cached OPEN mark cannot generate an order',async()=>{
  const d=await runEngine({age:901});
  assert.equal(d.action,'WAIT');assert.equal(d.metadata.data_quality_reason,'STALE_PROVIDER_PRICE');
  assert.equal(d.data_latency_seconds,901);
});
test('closed session and unconfirmed market reaction both wait',async()=>{
  assert.equal((await runEngine({sessionEnd:now/1000})).action,'WAIT');
  assert.equal((await runEngine({reaction:0})).veto_reason,'REACTION_NOT_CONFIRMED');
});
// Existing real treasury planner contract: small shadow entries, costs, exits and closed promotion gate.
global.Deno={test};
require('../supabase/functions/_shared/evolution_treasury_multiasset_shadow.test.ts');
