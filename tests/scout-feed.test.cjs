const {test}=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),crypto=require('node:crypto');
const ctx=vm.createContext({require,module:{exports:{}},fetch,Buffer,URLSearchParams,AbortSignal});
vm.runInContext(fs.readFileSync('api/scout-feed.js','utf8'),ctx);
const digest=crypto.createHash('sha256').update('test-key').digest('hex');
async function run(key,lane,fetcher){const res={setHeader(){},status(n){this.code=n;return this},json(x){this.body=x;return this},send(x){this.body=x;return this}};await ctx.createHandler(fetcher,digest)({method:'GET',headers:{'x-brian-internal-key':key},query:{lane}},res);return res;}
test('unauthorized callers and arbitrary destinations never reach upstream',async()=>{let calls=0;const f=async()=>{calls++};assert.equal((await run('bad','discovery:energy',f)).code,401);assert.equal((await run('test-key','https://internal.example',f)).code,400);assert.equal(calls,0)});
test('authenticated lane returns only validated RSS from fixed Google origin',async()=>{const r=await run('test-key','discovery:energy',async url=>{assert.equal(new URL(url).hostname,'news.google.com');return{ok:true,text:async()=>'<rss><channel/></rss>'}});assert.equal(r.code,200)});
test('upstream failure or HTML cannot be reported as successful feed',async()=>{for(const f of [async()=>({ok:false,status:503}),async()=>({ok:true,text:async()=>'<html>Error</html>'})])assert.equal((await run('test-key','discovery:energy',f)).code,502)});
