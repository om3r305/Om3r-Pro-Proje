async function v4FetchJson(host,path){
  const r=await fetch(host+path,{cache:'no-store'});if(!r.ok)throw Error(`HTTP ${r.status} ${path}`);return r.json();
}
async function v4Spot(path){
  let last;for(const h of ['https://api.binance.com','https://api1.binance.com','https://api3.binance.com'])try{return await v4FetchJson(h,path)}catch(e){last=e}throw last||Error('Binance spot unavailable');
}
async function v4Perp(path){return v4FetchJson('https://fapi.binance.com',path);}
function v4MapKlines(a){return a.map(x=>({t:Number(x[0]),o:Number(x[1]),h:Number(x[2]),l:Number(x[3]),c:Number(x[4]),v:Number(x[5]),closed:true}));}
async function v4LoadKlines(sym,tf,limit,venue='SPOT'){
  const path=`${venue==='USDM_PERP'?'/fapi/v1/klines':'/api/v3/klines'}?symbol=${encodeURIComponent(sym)}&interval=${tf}&limit=${limit}`;
  const a=venue==='USDM_PERP'?await v4Perp(path):await v4Spot(path);return v4MapKlines(a);
}
async function v4Batch(items,n,fn){for(let i=0;i<items.length;i+=n)await Promise.all(items.slice(i,i+n).map(fn));}
function v4ValidTicker(t){
  const s=String(t.symbol||'');if(!s.endsWith('USDT'))return false;const base=s.slice(0,-4);if(V4_EXCLUDED_BASE.has(base)||/(UP|DOWN|BULL|BEAR)$/.test(base))return false;
  const q=Number(t.quoteVolume||0),bid=Number(t.bidPrice||0),ask=Number(t.askPrice||0),mid=(bid+ask)/2;if(q<V4_MIN_QUOTE_VOL||!(bid>0&&ask>=bid&&mid>0))return false;
  const spread=(ask-bid)/mid*10000;return spread<=V4_MAX_DISCOVERY_SPREAD_BPS;
}
function v4RankCandidate(t,rows5,btc5){
  const f=v4Feature(rows5),b=v4Feature(btc5);if(!f||!b)return null;
  const bid=Number(t.bidPrice),ask=Number(t.askPrice),mid=(bid+ask)/2,spread=(ask-bid)/mid*10000,fee=Number(book.cfg?.fee_bps??10),cost=2*fee+spread+4;
  const atrBps=v4Bps(f.atrPct),tradeability=cost?atrBps/cost:0,volScore=v4Clamp(Math.log10(Math.max(1,Number(t.quoteVolume))/1e6)/2.5,0,1),overheat=v4Clamp(Math.abs(Number(t.priceChangePercent||0))/18,0,1);
  const score=100*v4Clamp(.48*v4Clamp(tradeability/3.2,0,1)+.24*volScore+.20*v4Clamp(f.volRel/2,0,1)+.08*v4Clamp(Math.abs(f.trend),0,1)-.18*overheat,0,1);
  return{symbol:t.symbol,score,tradeability,atrBps,spreadBps:spread,quoteVolume:Number(t.quoteVolume||0),change:Number(t.priceChangePercent||0)};
}
async function v4DiscoverUniverse(force=false){
  if(!force&&v4Universe.length&&v4Now()-v4UniverseUpdatedAt<V4_UNIVERSE_REFRESH_MS)return v4Universe;
  const [tickers,perpInfo,premiums]=await Promise.all([v4Spot('/api/v3/ticker/24hr'),v4Perp('/fapi/v1/exchangeInfo').catch(()=>({symbols:[]})),v4Perp('/fapi/v1/premiumIndex').catch(()=>[])]);
  v4FuturesSymbols.clear();for(const x of (perpInfo.symbols||[]))if(x.status==='TRADING'&&x.contractType==='PERPETUAL'&&x.quoteAsset==='USDT')v4FuturesSymbols.add(x.symbol);
  for(const x of (Array.isArray(premiums)?premiums:[]))v4Funding[x.symbol]=Number(x.lastFundingRate||0);
  const btc5=await v4LoadKlines('BTCUSDT','5m',90);
  const pre=tickers.filter(v4ValidTicker).sort((a,b)=>Number(b.quoteVolume)-Number(a.quoteVolume)).slice(0,V4_SCAN_LIMIT);
  const ranks=[];
  await v4Batch(pre,6,async t=>{try{const r=await v4LoadKlines(t.symbol,'5m',70);const q=v4RankCandidate(t,r,btc5);if(q)ranks.push(q)}catch{}});
  ranks.sort((a,b)=>b.score-a.score);v4Ranks=Object.fromEntries(ranks.map(x=>[x.symbol,x]));
  const open=Object.keys(states).filter(s=>states[s]?.pos),next=[];
  for(const s of [...open,...ranks.filter(x=>x.tradeability>=1.55).map(x=>x.symbol),...ranks.map(x=>x.symbol)])if(!next.includes(s)&&next.length<V4_UNIVERSE_SIZE)next.push(s);
  v4Universe=next;v4UniverseUpdatedAt=v4Now();v4Universe.forEach(v4Ensure);return v4Universe;
}
async function v4LoadHistory(){
  if($('chartOverlay'))$('chartOverlay').classList.add('show');
  await v4DiscoverUniverse(true);
  const all=[...new Set(['BTCUSDT',...v4Universe])];
  await v4Batch(all,3,async sym=>{
    v4Ensure(sym);
    const [m1,m5,m15,h1]=await Promise.all([
      v4LoadKlines(sym,'1m',720),v4LoadKlines(sym,'5m',180),v4LoadKlines(sym,'15m',120),v4LoadKlines(sym,'1h',72)
    ]);
    v4Bars[sym]={'1m':m1,'5m':m5,'15m':m15,'1h':h1};candles[sym]=m1.slice(-180);live[sym]=m1.at(-1)?.c;
    if(sym!=='BTCUSDT'&&v4FuturesSymbols.has(sym)){
      try{const [p5,p15,p1h]=await Promise.all([v4LoadKlines(sym,'5m',120,'USDM_PERP'),v4LoadKlines(sym,'15m',96,'USDM_PERP'),v4LoadKlines(sym,'1h',72,'USDM_PERP')]);v4PerpBars[sym]={'5m':p5,'15m':p15,'1h':p1h};}catch{}
    }
  });
  if($('chartOverlay'))$('chartOverlay').classList.remove('show');
  if(!selected||!v4Universe.includes(selected))selected=v4Universe[0]||'ETHUSDT';
  return true;
}
historyLoad=v4LoadHistory;

function v4SeedState(sym){
  const st=v4Ensure(sym),f5=v4Feature(v4Closed(sym,'5m')),p=Number(live[sym]||f5?.last||0);
  st.v4={phase:'WATCH',armSide:null,armLow:p||null,armHigh:p||null,armAt:0,lastVeto:'WAIT_SETUP',lastDecisionAt:0,lastSignal:null};
  st.dip=null;st.armed=false;st.lastAction='WATCH';st.peak=p||null;st.last=p||st.last;return st;
}
