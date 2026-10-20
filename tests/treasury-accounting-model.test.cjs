const{test}=require('node:test'),a=require('node:assert/strict'),{project,num}=require('../monster-coins-pro/treasury-accounting-model.js');const t='2026-09-16T06:00:00Z';const open={action_id:'o',position_id:'p',kind:'OPEN',asset_id:'BTC',direction:1,capital_usd:1000,reference_price:100,cost_usd:1,observed_at:'2026-09-15T06:00:00Z'};const exit={...open,action_id:'e',kind:'EXIT',reference_price:110,cost_usd:2,observed_at:t};const data={snapshot:{observed_at:t,starting_equity_usd:1000,equity_usd:1097,realized_pnl_usd:97},actions:[exit,open],history:[]};
test('long gain and fees match ledger, no double deduction',()=>{const v=project(data);a(Math.abs(v.knownRealized-97)<1e-9);a(Math.abs(v.rows[0].net-97)<1e-9);a.equal(v.cost,3);a.equal(v.total,97)});
test('coverage labels use the actual server limits and never invent 1000 records',()=>{
 const {coverageNotes}=require('../monster-coins-pro/treasury-accounting-model.js');
 const notes=coverageNotes({actions_truncated:true,history_truncated:true,action_limit:300,history_limit:360});
 a.match(notes.actions,/300/);a.match(notes.history,/360/);a.doesNotMatch(JSON.stringify(notes),/1000/);
 a.doesNotMatch(JSON.stringify(coverageNotes({actions_truncated:true,history_truncated:true})),/\d/);
});
test('a server-marked stale accounting response is not fresh even when its snapshot is recent',()=>{
 const {snapshotFresh}=require('../monster-coins-pro/treasury-accounting-model.js');
 const now=Date.now(),recent={snapshot:{observed_at:new Date(now-1000).toISOString()}};
 a.equal(snapshotFresh(recent,now),true);a.equal(snapshotFresh({...recent,stale:true},now),false);
 a.equal(snapshotFresh({...recent,status:'STALE'},now),false);a.equal(snapshotFresh(null,now),false);
 a.equal(snapshotFresh({snapshot:{observed_at:new Date(now+1000).toISOString()}},now),false);
});
test('short loss uses signed return',()=>{const v=project({...data,actions:[{...open,direction:-1},{...exit,direction:-1}]});a(Math.abs(v.knownRealized+103)<1e-9)});
test('REALTIME requires an explicit live quote, valid price and recent nonfuture source timestamp',()=>{
 const {markFresh}=require('../monster-coins-pro/treasury-accounting-model.js');
 const now=Date.now(),mark={live:true,price:57.46,observed_at:new Date(now-1000).toISOString()};
 a.equal(markFresh(mark,now),true);
 for(const patch of [{live:false},{live:undefined},{stale:true},{price:null},{price:true},{price:false},{price:0},{price:-1},{observed_at:null},{observed_at:'bad'},{observed_at:new Date(now+1).toISOString()},{observed_at:new Date(now-120001).toISOString()}])a.equal(markFresh({...mark,...patch},now),false);
});
test('missing or ambiguous opening stays unknown',()=>{for(const actions of [[exit],[exit,open,{...open,action_id:'o2'}]])a.equal(project({...data,actions}).rows[0].net,null)});
test('period exit uses older matched entry but charges only exit-period ledger cost',()=>{const v=project({...data,actions:[exit],opening_actions:[{...open,observed_at:'2026-09-10T00:00:00Z'}]},1);a(Math.abs(v.knownRealized-98)<1e-9);a(Math.abs(v.rows[0].net-97)<1e-9);a.equal(v.cost,2)});
test('missing data is not numeric zero',()=>{a.equal(num(null),null);a.equal(project({}).equity,null);a.equal(project({...data,actions:[{...exit,cost_usd:null},open]}).rows[0].realized,null)});
test('future records and duplicates excluded',()=>a.equal(project({...data,actions:[open,open,{...exit,observed_at:'2027-01-01'}]}).rows.length,1));
test('wrong asset or partial capital cannot be falsely paired',()=>{for(const patch of [{asset_id:'ETH'},{capital_usd:500}])a.equal(project({...data,actions:[open,{...exit,...patch}]}).rows[0].net,null)});
test('open net contribution is fee only, not invested capital',()=>a.equal(project({...data,actions:[open]}).knownRealized,-1));
