const {test}=require('node:test');
const assert=require('node:assert/strict');
const {render,aggregate}=require('../monster-coins-pro/outcome-coverage.js');
const protocol={id:'fixed',starts_at:'2026-10-19T13:00:00Z',ends_at:'2026-10-19T14:00:00Z',final_horizon_at:'2026-10-19T15:02:00Z'};
const row=(extra={})=>({sample:'rolling',asset_id:'BTC',seconds:300,sampled:5,mature:4,not_mature:1,resolved:1,missing_entry:0,invalid_outcome:0,awaiting_auditor:1,window_open:1,missing_terminal:1,missing_cost:1,cost_known_directional:0,...extra});
const data=(extra={})=>({status:'AVAILABLE',as_of:'2026-10-19T12:00:00Z',protocol,rows:[row()],...extra});
test('the exact aggregate conserves missing outcomes and cost subset',()=>{
 const r=aggregate([row(),row({asset_id:'ETH'})])[0];
 assert.equal(r.sampled,10);assert.equal(r.resolved,2);assert.equal(r.missing_terminal,2);assert.equal(r.missing_cost,2);assert.equal(r.mature,8);
});
test('inconsistent totals or null counts cannot look complete',()=>{
 assert.match(render(data({rows:[row({mature:0})]})),/tutarlılığı doğrulanamadı/);
 assert.match(render(data({rows:[row({missing_cost:null})]})),/tutarlılığı doğrulanamadı/);
 assert.match(render(data({rows:[row({missing_cost:2})]})),/tutarlılığı doğrulanamadı/);
});
test('missing legacy API and failed queries do not fabricate zeros',()=>{
 assert.match(render(undefined),/alınamadı/);assert.match(render({status:'UNAVAILABLE',rows:null}),/alınamadı/);
});
test('a future empty cohort is pending, never 100 percent',()=>{
 const h=render(data());assert.match(h,/Başlaması bekleniyor/);assert.match(h,/henüz karar kaydı yok/);assert.doesNotMatch(h,/100%|%100/);
});
test('fixed phases use the observation time, including the final tolerance',()=>{
 for(const [at,label] of [['13:30:00','Kararlar birikiyor'],['14:30:00','Sonuç ufukları bekleniyor'],['15:02:00','Ölçüm dönemi sona erdi']])assert.ok(render(data({as_of:`2026-10-19T${at}Z`})).includes(label));
});
test('bounded data is visibly partial, and asset text cannot inject HTML',()=>{
 const h=render(data({rolling_truncated:true,rows:[row({asset_id:'<img src=x onerror=alert(1)>'})]}));
 assert.match(h,/kapsamı kısmi/);assert.ok(h.includes('&lt;img'));assert.ok(!h.includes('<img'));
});
test('unrecognized groups and horizons fail closed',()=>{
 assert.match(render(data({rows:[row({sample:'profit'})]})),/tutarlılığı doğrulanamadı/);
 assert.match(render(data({rows:[row({seconds:301})]})),/tutarlılığı doğrulanamadı/);
});
test('window boundaries are fixed and no promotion or return is promised',()=>{
 const h=render(data());assert.match(h,/120 saniye/);assert.match(h,/kârlılık kanıtlamaz/);assert.match(h,/Otoma/);assert.match(h,/Maliyet eksik/);
});
