/* Brian Aggressive Dip Expert V4 — microstructure / multi-timeframe shadow engine.
   IMPORTANT: SHADOW ONLY. No live Binance order endpoint. Phase 3.7 stays untouched.
   V4 fixes V3's cost/payoff asymmetry, dead 15m layer, weak dip gating, browser reload state bloat,
   spot-short mismatch, and missing portfolio/pair guards. */

const DIP_EXPERT_V4 = 'aggressive-dip-expert-v4';
const V4_SCAN_LIMIT = 36;
const V4_UNIVERSE_SIZE = 12;
const V4_MAX_OPEN = 2;
const V4_MAX_HEAT = 0.32;
const V4_MAX_POSITION = 0.18;
const V4_RISK_PER_TRADE = 0.0045;
const V4_UNIVERSE_REFRESH_MS = 15 * 60 * 1000;
const V4_MIN_QUOTE_VOL = 25_000_000;
const V4_MAX_DISCOVERY_SPREAD_BPS = 12;
const V4_MAX_ENTRY_SPREAD_BPS = 8;
const V4_COST_EDGE_MULT = 2.5;
const V4_MIN_RR = 1.35;
const V4_LONG_HOLD_MS = 18 * 60 * 1000;
const V4_SHORT_HOLD_MS = 22 * 60 * 1000;
const V4_PAIR_LOCK_MS = 30 * 60 * 1000;
const V4_GLOBAL_LOCK_MS = 10 * 60 * 1000;
const V4_DD_LOCK_MS = 30 * 60 * 1000;
const V4_EXCLUDED_BASE = new Set([
  'BTC','USDC','FDUSD','TUSD','USDP','DAI','EUR','TRY','BRL','GBP','AUD','BIDR','IDRT','UAH','PLN','RON','ARS','AEUR','USD1','USDE','USDS'
]);

let v4Universe = [];
let v4Ranks = {};
let v4UniverseUpdatedAt = 0;
let v4NeedsRestart = false;
let v4Booting = false;
let v4CloudFault = false;
let v4SpotWs = null;
let v4PerpWs = null;
let v4WsGeneration = 0;
let v4ReconnectTimer = null;
let v4UniverseTimer = null;
let v4RenderAt = 0;
let v4LastFunnelEmit = 0;
let v4SessionLossLockUntil = 0;

const v4Bars = {};       // symbol -> {1m:[],5m:[],15m:[],1h:[]}
const v4SpotBook = {};   // symbol -> {bid,ask,bids,asks,ts}
const v4PerpBook = {};   // symbol -> same
const v4PerpBars = {};   // symbol -> {5m,15m,1h}; used for SHORT regime parity
const v4Funding = {};    // symbol -> latest funding rate
const v4TapeSpot = {};   // symbol -> [{t,buy,sell,p}]
const v4TapePerp = {};
const v4FuturesSymbols = new Set();
const v4PairGuard = {};
const v4ClosedOutcomes = [];
const v4EvalAt = {};
const v4Funnel = {evaluated:0,entered:0,rejected:{}};

const _v4PrevRender = render;
const _v4PrevRenderRadar = renderRadar;
const _v4PrevRenderKpi = renderKpi;
const _v4PrevDraw = draw;
const _v4PrevNote = note;
const _v4PrevEngineErr = engineErr;

const v4Clamp = (v,a,b) => Math.max(a, Math.min(b, Number(v) || 0));
const v4Pct = (a,b) => b ? ((a / b) - 1) * 100 : 0;
const v4Bps = pct => Number(pct || 0) * 100;
const v4Sum = a => a.reduce((s,x)=>s+Number(x||0),0);
const v4Mean = a => a.length ? v4Sum(a)/a.length : 0;
const v4Sd = a => { if(a.length<2) return 0; const m=v4Mean(a); return Math.sqrt(v4Mean(a.map(x=>(x-m)**2))); };
const v4Esc = s => String(s ?? '').replace(/[&<>\"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[m]));
const v4Now = () => Date.now();

function v4Reject(reason){
  v4Funnel.rejected[reason]=(v4Funnel.rejected[reason]||0)+1;
  return false;
}
function v4ResetFunnel(){v4Funnel.evaluated=0;v4Funnel.entered=0;v4Funnel.rejected={};}
function v4Ensure(sym){
  if(!states[sym]) states[sym]=fresh(sym);
  const st=states[sym];
  if(!st.v4) st.v4={phase:'WATCH',armSide:null,armLow:null,armHigh:null,armAt:0,lastVeto:'WAIT_DATA',lastDecisionAt:0,lastSignal:null};
  if(!Number.isFinite(Number(st.realized))) st.realized=0;
  if(!v4Bars[sym]) v4Bars[sym]={'1m':[],'5m':[],'15m':[],'1h':[]};
  if(!v4PerpBars[sym]) v4PerpBars[sym]={'5m':[],'15m':[],'1h':[]};
  return st;
}
function v4AllSymbols(){
  const open=Object.keys(states).filter(s=>states[s]?.pos);
  return [...new Set([...v4Universe,...open])];
}
function v4OpenPositions(){return Object.values(states).filter(s=>s?.pos);}
function v4OpenCount(){return v4OpenPositions().length;}
function v4Heat(){
  const eq=Math.max(1,metrics().equity);
  return v4OpenPositions().reduce((s,st)=>s+Number(st.pos?.margin||st.pos?.notional||0),0)/eq;
}

function v4Ema(values,n){
  if(!values.length) return 0;
  const k=2/(n+1); let x=Number(values[0]);
  for(let i=1;i<values.length;i++) x=Number(values[i])*k+x*(1-k);
  return x;
}
function v4Rsi(values,n=14){
  if(values.length<n+1) return 50;
  let g=0,l=0;
  for(let i=values.length-n;i<values.length;i++){
    const d=Number(values[i])-Number(values[i-1]); if(d>=0)g+=d;else l-=d;
  }
  if(!l) return 100; const rs=(g/n)/(l/n); return 100-(100/(1+rs));
}
function v4Atr(rows,n=14){
  if(rows.length<2) return 0;
  const xs=[]; for(let i=Math.max(1,rows.length-n);i<rows.length;i++){
    const r=rows[i],pc=rows[i-1].c; xs.push(Math.max(r.h-r.l,Math.abs(r.h-pc),Math.abs(r.l-pc)));
  }
  return v4Mean(xs);
}
function v4Returns(rows,n=60){
  const out=[]; const s=Math.max(1,rows.length-n);
  for(let i=s;i<rows.length;i++) if(rows[i-1].c>0&&rows[i].c>0) out.push(Math.log(rows[i].c/rows[i-1].c));
  return out;
}
function v4Beta(rows,btcRows,n=60){
  const a=v4Returns(rows,n),b=v4Returns(btcRows,n),m=Math.min(a.length,b.length);
  if(m<20) return 1;
  const aa=a.slice(-m),bb=b.slice(-m),ma=v4Mean(aa),mb=v4Mean(bb);
  let cov=0,vr=0;for(let i=0;i<m;i++){cov+=(aa[i]-ma)*(bb[i]-mb);vr+=(bb[i]-mb)**2;}
  return vr?cov/vr:1;
}
function v4Swing(rows,look=18){
  const a=rows.slice(-look); if(!a.length)return{low:null,high:null};
  return{low:Math.min(...a.map(x=>x.l)),high:Math.max(...a.map(x=>x.h))};
}
function v4Feature(rows){
  if(!rows||rows.length<30) return null;
  const closes=rows.map(x=>Number(x.c)),last=rows.at(-1),atr=v4Atr(rows,14),atrPct=last.c?atr/last.c*100:0;
  const ema9=v4Ema(closes.slice(-40),9),ema21=v4Ema(closes.slice(-70),21),ema50=v4Ema(closes.slice(-100),50);
  const rsi=v4Rsi(closes,14),swing=v4Swing(rows,18),mean=v4Ema(closes.slice(-40),20),z=atr?((last.c-mean)/atr):0;
  const vols=rows.slice(-21,-1).map(x=>Number(x.v||0)),volRel=v4Mean(vols)>0?Number(last.v||0)/v4Mean(vols):1;
  const slope21=ema21?((last.c/ema21)-1)*100:0;
  let trend=0;
  if(last.c>ema9)trend+=.22;else trend-=.22;
  if(ema9>ema21)trend+=.28;else trend-=.28;
  if(ema21>ema50)trend+=.30;else trend-=.30;
  trend+=v4Clamp(slope21*1.5,-.20,.20);
  return{last:last.c,open:last.o,high:last.h,low:last.l,atr,atrPct,ema9,ema21,ema50,rsi,z,volRel,swingLow:swing.low,swingHigh:swing.high,trend:v4Clamp(trend,-1,1)};
}
function v4Closed(sym,tf){return (v4Bars[sym]?.[tf]||[]).filter(x=>x.closed!==false);}
function v4Context(sym,venue='SPOT'){
  const shortVenue=venue==='USDM_PERP', f1=v4Feature(v4Closed(sym,'1m')),
        f5=v4Feature(shortVenue?(v4PerpBars[sym]?.['5m']||[]):v4Closed(sym,'5m')),
        f15=v4Feature(shortVenue?(v4PerpBars[sym]?.['15m']||[]):v4Closed(sym,'15m')),
        f1h=v4Feature(shortVenue?(v4PerpBars[sym]?.['1h']||[]):v4Closed(sym,'1h'));
  const b1=v4Feature(v4Closed('BTCUSDT','1m')),b15=v4Feature(v4Closed('BTCUSDT','15m')),b1h=v4Feature(v4Closed('BTCUSDT','1h'));
  if(!f1||!f5||!f15||!f1h||!b1||!b15||!b1h)return null;
  const btc1=v4Closed('BTCUSDT','1m'),s1=v4Closed(sym,'1m'),beta=v4Beta(s1,btc1,60),idioZ=f1.z-beta*b1.z;
  const flow=v4Flow(sym,venue,8_000),bk=v4BookMetrics(sym,venue);
  const feeSideBps=venue==='USDM_PERP'?5:Number(book.cfg?.fee_bps??10);
  const impact=v4ImpactBps(sym,venue,Math.max(50,metrics().equity*V4_MAX_POSITION));
  const roundTripCostBps=2*feeSideBps+Math.max(0,bk.spreadBps)+2*impact;
  const expectedMoveBps=Math.max(v4Bps(f5.atrPct)*1.55,v4Bps(f15.atrPct)*.55);
  const htfLong=.58*f15.trend+.42*f1h.trend,htfShort=-htfLong;
  const btcLongRisk=.55*b15.trend+.45*b1h.trend;
  return{sym,venue,f1,f5,f15,f1h,b1,b15,b1h,beta,idioZ,flow,bk,feeSideBps,impact,roundTripCostBps,expectedMoveBps,edgeRatio:roundTripCostBps?expectedMoveBps/roundTripCostBps:0,htfLong,htfShort,btcLongRisk,fundingRate:Number(v4Funding[sym]||0)};
}

function v4TapePush(store,sym,m){
  const a=store[sym]||(store[sym]=[]),p=Number(m.p),q=Number(m.q),notional=p*q,t=v4Now();
  // Binance aggTrade: m=true => buyer is maker => aggressive seller. m=false => aggressive buyer.
  a.push({t,buy:m.m?0:notional,sell:m.m?notional:0,p});
  while(a.length&&t-a[0].t>30_000)a.shift();
}
function v4Flow(sym,venue='SPOT',ms=8_000){
  const a=(venue==='USDM_PERP'?v4TapePerp[sym]:v4TapeSpot[sym])||[],cut=v4Now()-ms,x=a.filter(q=>q.t>=cut);
  const buy=v4Sum(x.map(q=>q.buy)),sell=v4Sum(x.map(q=>q.sell)),tot=buy+sell;
  return{buy,sell,ofi:tot?(buy-sell)/tot:0,ratio:(sell>0?buy/sell:(buy>0?9:1)),prints:x.length};
}
function v4BookUpdate(store,sym,m){
  const x=store[sym]||(store[sym]={bid:null,ask:null,bids:[],asks:[],ts:0});
  if(m.b!=null&&m.a!=null&&!Array.isArray(m.b)){x.bid=Number(m.b);x.ask=Number(m.a);}
  const bids=m.bids||m.b,asks=m.asks||m.a;
  if(Array.isArray(bids))x.bids=bids.slice(0,5).map(z=>[Number(z[0]),Number(z[1])]);
  if(Array.isArray(asks))x.asks=asks.slice(0,5).map(z=>[Number(z[0]),Number(z[1])]);
  if(x.bids.length&&!x.bid)x.bid=x.bids[0][0]; if(x.asks.length&&!x.ask)x.ask=x.asks[0][0];x.ts=v4Now();
}
function v4BookMetrics(sym,venue='SPOT'){
  const x=(venue==='USDM_PERP'?v4PerpBook[sym]:v4SpotBook[sym])||{};
  const bid=Number(x.bid||x.bids?.[0]?.[0]),ask=Number(x.ask||x.asks?.[0]?.[0]),mid=bid>0&&ask>0?(bid+ask)/2:Number(live[sym]||0);
  const spreadBps=bid>0&&ask>=bid?(ask-bid)/mid*10000:999;
  const bidN=v4Sum((x.bids||[]).map(([p,q])=>p*q)),askN=v4Sum((x.asks||[]).map(([p,q])=>p*q));
  return{bid,ask,mid,spreadBps,pressure:askN>0?bidN/askN:1,bidN,askN,ageMs:x.ts?v4Now()-x.ts:999999};
}
function v4ImpactBps(sym,venue='SPOT',notional=100){
  const x=(venue==='USDM_PERP'?v4PerpBook[sym]:v4SpotBook[sym])||{},levels=(x.asks||[]),mid=v4BookMetrics(sym,venue).mid;
  if(!mid||!levels.length)return 2;
  let rem=Math.max(1,notional),spent=0,qty=0;
  for(const [p,q] of levels){const cap=p*q,take=Math.min(rem,cap);spent+=take;qty+=take/p;rem-=take;if(rem<=0)break;}
  if(rem>0)return Math.min(50,8+rem/notional*25);
  const vwap=qty?spent/qty:mid;return Math.max(0,(vwap-mid)/mid*10000);
}
