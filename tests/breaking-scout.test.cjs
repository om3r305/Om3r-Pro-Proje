const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm');
const {stripTypeScriptTypes}=require('node:module');
const source=stripTypeScriptTypes(fs.readFileSync('supabase/functions/brian-breaking-scout/index.ts','utf8').replace(/^import .*;\n/gm,''));
const lane={id:'discovery:test',theme:'geopolitics',priority:'CRITICAL',googleQ:'attack when:1h',fallbackQ:'attack',compactQ:'attack'};
const date=new Date().toISOString();
const xml=(published=date)=>`<rss><channel><item><title>News</title><link>https://example.org/news</link>${published?`<pubDate>${published}</pubDate>`:''}</item></channel></rss>`;
function runtime(fetch){return vm.createContext({Deno:{env:{get:()=>''},serve:()=>{}},createClient:()=>({}),fetch,URL,URLSearchParams,AbortSignal,Response,TextEncoder,crypto:require('node:crypto').webcrypto});}
function setup(fetch){const ctx=runtime(fetch);vm.runInContext(source,ctx);return ctx;}
test('slow valid Google response remains usable after stale Bing response',async()=>{
 const ctx=setup(async url=>({ok:true,text:async()=>xml(url.includes('bing.com')?'2000-01-01T00:00:00Z':date)}));
 const r=await ctx.fetchDiscovery(lane);assert.equal(r.covered,true);assert.equal(r.articles.length,1);assert.equal(r.provider,'google_news_rss');
 assert.match(r.providerErrors[0],/NO_FRESH/);
});
test('undated, stale and future articles cannot establish current coverage',async()=>{
 for(const stamp of [null,'2000-01-01T00:00:00Z',new Date(Date.now()+3600000).toISOString()]){
  const ctx=setup(async()=>({ok:true,text:async()=>xml(stamp)}));const r=await ctx.fetchDiscovery(lane);assert.equal(r.covered,false);assert.equal(r.articles.length,0);
 }
});
test('valid empty Google feed differs from failed providers',async()=>{
 const ctx=setup(async url=>{if(url.includes('google'))return {ok:true,text:async()=>'<rss><channel></channel></rss>'};throw Error('offline');});
 assert.equal((await ctx.fetchDiscovery(lane)).covered,true);
 const failed=setup(async()=>{throw Error('offline')});assert.equal((await failed.fetchDiscovery(lane)).covered,false);
});
test('normal lane also falls back from stale Bing to current Google',async()=>{
 const ctx=setup(async url=>({ok:true,text:async()=>xml(url.includes('bing')?'2000-01-01T00:00:00Z':date)}));
 assert.equal((await ctx.fetchDiscovery({...lane,priority:'NORMAL'})).articles.length,1);
});
test('provider deadline permits an eight-second response while bounding each request',async()=>{
 const budgets=[];const ctx=setup(async(url,init)=>{if(url.includes('google')&&init.signal.budget<8500)throw Error('timeout');return {ok:true,text:async()=>xml(url.includes('google')?date:null)};});
 ctx.AbortSignal={timeout:budget=>{budgets.push(budget);return {budget}}};
 assert.equal((await ctx.fetchDiscovery(lane)).covered,true);assert.ok(budgets.every(x=>x<=12000));
});
