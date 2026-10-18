const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const vm=require('node:vm');const {stripTypeScriptTypes}=require('node:module');
const source=stripTypeScriptTypes(fs.readFileSync('supabase/functions/_shared/derivatives_universe.ts','utf8')).replace(/export /g,'');
const select=vm.runInNewContext(source+';selectDerivativesUniverse');
const active={symbol:'BTCUSDT',status:'TRADING',contractType:'PERPETUAL',quoteAsset:'USDT'};
test('spot-only, delisted and delivery contracts are disclosed and never queried as perpetuals',()=>{
 const result=select(['BTCUSDT','SOXLBUSDT','OLDUSDT','BTCUSDT','NEXTUSDT'],{symbols:[active,{...active,symbol:'OLDUSDT',status:'SETTLING'},{...active,symbol:'NEXTUSDT',contractType:'CURRENT_QUARTER'}]});
 assert.equal(JSON.stringify(result.symbols),'["BTCUSDT"]');assert.equal(JSON.stringify(result.skipped),'["SOXLBUSDT","OLDUSDT","NEXTUSDT"]');
});
test('missing or malformed contract catalog fails closed instead of claiming all symbols unavailable',()=>{
 for(const value of [null,{}, {symbols:[]},{symbols:[{symbol:'BTCUSDT'}]}])assert.throws(()=>select(['BTCUSDT'],value),/CATALOG/);
});
