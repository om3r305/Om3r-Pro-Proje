(function(root){
 const array=x=>Array.isArray(x)?x:[];
 const num=x=>x===null||x===undefined||x===''||typeof x==='boolean'?null:Number.isFinite(Number(x))?Number(x):null;
 const validTime=x=>Number.isFinite(Date.parse(x));
 function project(data,days=0){
  const snapshot=data?.snapshot||null,cutoff=Date.parse(snapshot?.observed_at),since=days&&Number.isFinite(cutoff)?cutoff-days*86400000:-Infinity;
  const all=[...new Map(array(data?.actions).filter(a=>a.action_id).map(a=>[a.action_id,a])).values()];
  const opens=new Map();for(const a of [...all,...array(data?.opening_actions)])if(a.kind==='OPEN'&&a.position_id){const list=opens.get(a.position_id)||[];if(!list.some(x=>x.action_id===a.action_id))list.push(a);opens.set(a.position_id,list)}
  const rows=all.filter(a=>validTime(a.observed_at)&&Date.parse(a.observed_at)>=since&&Date.parse(a.observed_at)<=cutoff).sort((a,b)=>Date.parse(b.observed_at)-Date.parse(a.observed_at)).map(a=>{
   const cost=num(a.cost_usd),capital=num(a.capital_usd),price=num(a.reference_price),direction=num(a.direction),o=(opens.get(a.position_id)||[]),open=o.length===1?o[0]:null;
   const matched=a.kind==='EXIT'&&open&&open.asset_id===a.asset_id&&num(open.direction)===direction&&[1,-1].includes(direction)&&num(open.capital_usd)===capital&&capital>0&&price>0&&num(open.reference_price)>0&&Date.parse(open.observed_at)<=Date.parse(a.observed_at);
   const gross=matched?direction*(price/num(open.reference_price)-1)*capital:null;
   const validCost=cost!==null&&cost>=0?cost:null;
   const realized=a.kind==='OPEN'&&validCost!==null?-validCost:a.kind==='EXIT'&&gross!==null&&validCost!==null?gross-validCost:null;
   const net=gross!==null&&validCost!==null&&num(open?.cost_usd)!==null&&num(open.cost_usd)>=0?gross-validCost-num(open.cost_usd):null;
   return {...a,cost:validCost,gross,realized,net,open:matched?open:null};
  });
  const known=rows.filter(r=>r.realized!==null),unknown=rows.length-known.length;
  const sum=(rs,key)=>rs.reduce((s,r)=>s+(r[key]??0),0);
  const daily=new Map(),assets=new Map();for(const r of rows){const key=r.observed_at.slice(0,10);if(!daily.has(key))daily.set(key,{date:key,realized:0,cost:0,unknown:0});const d=daily.get(key);d.realized+=r.realized??0;d.cost+=r.cost??0;d.unknown+=r.realized===null?1:0;
   if(r.kind==='EXIT'){if(!assets.has(r.asset_id))assets.set(r.asset_id,{asset:r.asset_id,net:0,count:0,unknown:0});const a=assets.get(r.asset_id);if(r.net===null)a.unknown++;else{a.net+=r.net;a.count++}}}
  const n=k=>num(snapshot?.[k]),equity=n('equity_usd'),start=n('starting_equity_usd'),realized=n('realized_pnl_usd');
  return {snapshot,rows,history:array(data?.history).filter(h=>validTime(h.observed_at)&&Date.parse(h.observed_at)>=since&&Date.parse(h.observed_at)<=cutoff&&num(h.equity_usd)!==null),daily:[...daily.values()].sort((a,b)=>a.date.localeCompare(b.date)),assets:[...assets.values()].sort((a,b)=>a.net-b.net),unknown,costUnknown:rows.filter(r=>r.cost===null).length,cost:sum(rows,'cost'),knownRealized:sum(known,'realized'),grossProfit:rows.reduce((s,r)=>s+Math.max(0,r.gross??0),0),grossLoss:rows.reduce((s,r)=>s+Math.min(0,r.gross??0),0),equity,start,realized,total:equity!==null&&start!==null?equity-start:null,unrealized:equity!==null&&start!==null&&realized!==null?equity-start-realized:null,cash:n('cash_usd'),positions:array(snapshot?.positions)};
 }
 const api={project,num};if(typeof module!=='undefined')module.exports=api;else root.TreasuryAccountingModel=api;
})(typeof window!=='undefined'?window:globalThis);
