const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const vm=require('node:vm');const {stripTypeScriptTypes}=require('node:module');
const source=fs.readFileSync('supabase/functions/brian-direct-wire-eye/index.ts','utf8');
const fn=stripTypeScriptTypes(source.slice(source.indexOf('async function fetchFeed('),source.indexOf('async function recordRun(')));
const spec={id:'test',url:'https://primary.invalid/rss',fallbackUrls:['https://fallback.invalid/rss']};
function make(fetch){return vm.runInNewContext(fn+';fetchFeed',{fetch,AbortSignal,parseFeed:s=>s==='feed'?[{title:'Real item'}]:[],err:e=>String(e)});}
test('failed primary uses the configured fallback and records original failure',async()=>{
 const urls=[];const run=make(async url=>{urls.push(url);return url===spec.url?{ok:false,status:404}:{ok:true,text:async()=> 'feed'}});
 const r=await run(spec);assert.deepEqual(urls,[spec.url,spec.fallbackUrls[0]]);assert.equal(r.fallbackUsed,true);assert.equal(r.url,spec.fallbackUrls[0]);assert.match(r.primaryErrors[0],/404/);
});
test('primary success never calls fallback',async()=>{
 let count=0;const r=await make(async()=>{count++;return {ok:true,text:async()=> 'feed'}})(spec);assert.equal(count,1);assert.equal(r.fallbackUsed,false);
});
test('HTML/empty responses cannot be reported as healthy RSS',async()=>{
 await assert.rejects(make(async()=>({ok:true,text:async()=>'<html>error</html>'}))(spec),/NO_FEED_ITEMS/);
});
test('fallback articles retain the actual publisher instead of the failed primary publisher',async()=>{
 const fetch=async url=>url===spec.url?{ok:false,status:404}:{ok:true,text:async()=> 'feed'};
 const run=vm.runInNewContext(fn+';fetchFeed',{fetch,AbortSignal,parseFeed:(_body,s)=>[{publisher:s.publisher}],err:e=>String(e)});
 const r=await run({...spec,publisher:'Yahoo Finance',fallbackPublisher:'Yahoo News'});
 assert.equal(r.items[0].publisher,'Yahoo News');
});

test('Yahoo benchmark RSS uses its honest limited scope without relabeling Yahoo News',()=>{
 const config=stripTypeScriptTypes(source.slice(source.indexOf('const FEEDS:'),source.indexOf('const MARKET_RE=')));
 const feeds=vm.runInNewContext(config+';FEEDS');const yahoo=feeds.find(f=>f.id==='wire:yahoo-finance');
 assert.equal(new URL(yahoo.url).hostname,'feeds.finance.yahoo.com');
 assert.equal(new URL(yahoo.url).searchParams.get('s'),'^GSPC,^IXIC');
 assert.equal(yahoo.publisher,'Yahoo Finance');assert.equal(yahoo.fallbackPublisher,undefined);
 assert.equal(yahoo.coverageScope,'SP500_NASDAQ_BENCHMARK_NEWS_NOT_ALL_YAHOO');
});
