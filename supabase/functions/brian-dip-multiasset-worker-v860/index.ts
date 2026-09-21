// @ts-ignore Supabase Edge remote import
import postgres from 'https://deno.land/x/postgresjs@v3.4.5/mod.js';
declare const Deno: any;

const DB_URL=Deno.env.get('SUPABASE_DB_URL')!;
const ENGINE_ID='dip-multiasset-v1';
const ENGINE_VERSION='V8.8.1';
const POLICY_VERSION='dip-v881-monster-conviction-scale-20260921.1';
const RISK_ENGINE='V881_MONSTER_CONVICTION_SCALE';
const MAX_DEEP_SCAN=32,CORE_SCAN_SLOTS=24,INTERRUPT_SLOTS=4,EXPLORER_SLOTS=4,LANE_TOP=6,MEMORY_MAX=48,MEMORY_TTL_MS=60*60_000;
const MAX_POSITIONS=3,FEE_BPS=10,MIN_SHADOW_NOTIONAL=8,MAX_TOTAL_GROSS_PCT=.30,LOSS_STREAK_PAUSE_MS=60*60_000;
const HOSTS=['https://api.binance.com','https://api1.binance.com','https://api2.binance.com'];
const EXCLUDED=new Set(['ETHUSDT','USDTUSDT','USDCUSDT','FDUSDUSDT','TUSDUSDT','DAIUSDT','BUSDUSDT','EURUSDT','TRYUSDT','USD1USDT']);

type J=Record<string,unknown>;
type Candidate={symbol:string;radar_score:number;price_change_pct:number;spread_bps:number;quote_volume:number;trades_24h:number;momentum_score:number;volatility_score:number;short_momentum_bps?:number;lane?:string;selection_score?:number};
type Bar={h:number;l:number;c:number;v:number;closeTime:number};
type Book={bid:number;ask:number;spreadBps:number};
type HorizonForecast={continuation:number;reversal:number;ret1_bps:number;ret3_bps:number;ret5_bps:number;ret15_bps:number;ret30_bps:number;ret60_bps:number;trend_bps:number;expected_5m_bps:number;expected_15m_bps:number;expected_30m_bps:number;expected_60m_bps:number};
type Market={symbol:string;bid:number;ask:number;mid:number;spreadBps:number;atr:number;atrPct:number;ema8:number;ema21:number;recentLow:number;recentHigh:number;pullbackPct:number;bouncePct:number;lastClose:number;prevClose:number;recovery:boolean;trendOk:boolean;forecast:HorizonForecast;shock_up_15m_bps:number;shock_age_min:number};
type Position={symbol:string;entry:number;qty:number;opened_at:string;stop:number;target:number;trail:number|null;max_price:number;cost_basis:number;radar_score:number;estimated_cost_bps:number;policy_version:string;entry_reason:string;entry_style?:string;opportunity_tier?:string;entry_signal?:number;entry_opportunity?:number;entry_forecast_net_bps?:number;entry_continuation?:number;capital_fraction?:number;harvest_armed?:boolean;harvest_at?:string|null;runner_mode?:boolean;runner_trail?:number|null;last_continuation?:number;initial_risk_usd?:number;winner_expansion?:boolean;profit_lock_active?:boolean;profit_lock_ratio?:number;profit_lock_max_net_pnl?:number;profit_lock_armed_at?:string|null;last_expected_15m_bps?:number;last_expected_30m_bps?:number;entry_explosion_score?:number;peak_continuation?:number;peak_expected_15m_bps?:number;peak_expected_30m_bps?:number;runner_regime?:string;reentry_type?:string;reentry_attempt?:number;wave_profit_bank?:number;wave_started_at?:number;prior_exit_reason?:string;prior_exit_price?:number;scale_stage?:number;scale_count?:number;last_scale_at?:number;last_scale_price?:number;scale_total_added?:number;scale_peak_explosion?:number;scale_peak_continuation?:number;scale_target_fraction?:number};
type State={engine_id:string;started_at:string;run_until:string;starting_equity:number;cash:number;realized_pnl:number;trade_count:number;win_count:number;loss_count:number;positions:Record<string,Position>;cooldowns:Record<string,any>;last_scan:J;last_eval_minute:string|null;enabled:boolean};
type Eval={coreRaw:boolean;winnerRaw:boolean;scoutRaw:boolean;hunterRaw:boolean;surgeRaw?:boolean;emergencyRaw?:boolean;score:number;opp:number;cost:number;gross:number;net:number;tier:string;capitalScore:number;explosionScore:number;forecast:HorizonForecast;reason:string;gates:Record<string,boolean>};
type MemoryItem={symbol:string;until:number;last_seen:number;score:number;opp:number;cont:number;lane:string};

const num=(v:unknown,f=0)=>Number.isFinite(Number(v))?Number(v):f;
const clip=(v:number,lo=0,hi=1)=>Math.max(lo,Math.min(hi,v));
const iso=()=>new Date().toISOString();
const pct=(a:number,b:number)=>b?((a/b)-1)*100:0;
function db(){return postgres(DB_URL,{prepare:false,max:1,idle_timeout:1,connect_timeout:8,max_lifetime:30});}
async function sha256Hex(v:string){const d=new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(v)));return[...d].map(b=>b.toString(16).padStart(2,'0')).join('');}
function same(a:string,b:string){if(a.length!==b.length)return false;let d=0;for(let i=0;i<a.length;i++)d|=a.charCodeAt(i)^b.charCodeAt(i);return d===0;}
async function requireCron(sql:any,req:Request){const supplied=(req.headers.get('x-brian-cron-key')||'').trim();if(!supplied)throw Error('UNAUTHORIZED_CRON');const rows=await sql`select cron_key_sha256 from public.brian_dashboard_auth where auth_id='control-v3' limit 1`;const expected=String(rows[0]?.cron_key_sha256||'');if(!expected)throw Error('CRON_AUTH_UNAVAILABLE');if(!same(await sha256Hex(supplied),expected))throw Error('UNAUTHORIZED_CRON');}
async function acquireLease(sql:any){const owner=crypto.randomUUID();const rows=await sql`select * from public.brian_dip_v84_acquire_lease('brian-dip-multiasset-worker-v1',${owner},45)`;return rows[0]?.acquired?{owner,generation:Number(rows[0].lease_generation)}:null;}
async function releaseLease(sql:any,l:any){if(!l)return;try{await sql`select public.brian_dip_v84_release_lease('brian-dip-multiasset-worker-v1',${l.owner},${l.generation})`;}catch{}}
async function marketJson(path:string,timeout=5000){let last='MARKET_UNAVAILABLE';for(const host of HOSTS){try{const r=await fetch(host+path,{headers:{accept:'application/json','user-agent':'Brian-DIP-V873/1.0'},signal:AbortSignal.timeout(timeout)});if(!r.ok){last=`HTTP_${r.status}`;if(r.status===418||r.status===429)break;continue;}return await r.json();}catch(e){last=e instanceof Error?e.message:String(e);}}throw Error(`BINANCE:${last}`);}
function ema(xs:number[],p:number){const k=2/(p+1);let out=xs[0]??0;for(let i=1;i<xs.length;i++)out=xs[i]*k+out*(1-k);return out;}
function atr(b:Bar[],p=14){if(b.length<p+2)return 0;const xs:number[]=[];for(let i=1;i<b.length;i++)xs.push(Math.max(b[i].h-b[i].l,Math.abs(b[i].h-b[i-1].c),Math.abs(b[i].l-b[i-1].c)));return xs.slice(-p).reduce((a,x)=>a+x,0)/p;}
function retBps(closes:number[],n:number){const last=closes.at(-1)||0,base=closes[Math.max(0,closes.length-1-n)]||last;return base>0?(last/base-1)*10000:0;}
function buildForecast(closes:number[],atrPct:number,e8:number,e21:number,recovery:boolean,trendOk:boolean,bouncePct:number,pullbackPct:number):HorizonForecast{const r1=retBps(closes,1),r3=retBps(closes,3),r5=retBps(closes,5),r15=retBps(closes,15),r30=retBps(closes,30),r60=retBps(closes,60),trend=e21>0?(e8/e21-1)*10000:0,atrBps=atrPct*100;let cont=.50+clip(r1/120,-1,1)*.05+clip(r3/180,-1,1)*.08+clip(r5/300,-1,1)*.09+clip(r15/650,-1,1)*.06+clip(trend/100,-1,1)*.13+(recovery?.08:-.07)+(trendOk?.07:-.09)+clip(bouncePct/3)*.04-clip(pullbackPct/8)*.02;cont=clip(cont,.05,.95);const bias=(cont-.5)*2,e5=bias*atrBps*.72+r3*.16+r1*.06,e15=bias*atrBps*1.18+r5*.18+r3*.10,e30=bias*atrBps*1.72+r15*.17+r5*.14,e60=bias*atrBps*2.25+r15*.24+r5*.16;return{continuation:cont,reversal:1-cont,ret1_bps:r1,ret3_bps:r3,ret5_bps:r5,ret15_bps:r15,ret30_bps:r30,ret60_bps:r60,trend_bps:trend,expected_5m_bps:e5,expected_15m_bps:e15,expected_30m_bps:e30,expected_60m_bps:e60};}
async function loadMarket(symbol:string,book?:Book):Promise<Market>{const rawBars=await marketJson(`/api/v3/klines?symbol=${encodeURIComponent(symbol)}&interval=1m&limit=90`,3500);if(!Array.isArray(rawBars)||rawBars.length<40)throw Error('INSUFFICIENT_BARS');let b=book;if(!b){const rawBook=await marketJson(`/api/v3/ticker/bookTicker?symbol=${encodeURIComponent(symbol)}`,3000) as J;const bid=num(rawBook.bidPrice),ask=num(rawBook.askPrice);if(!(bid>0&&ask>bid))throw Error('BAD_BOOK');b={bid,ask,spreadBps:(ask-bid)/((ask+bid)/2)*10000};}if(!(b.bid>0&&b.ask>b.bid))throw Error('BAD_BOOK');if(b.spreadBps>35)throw Error('WIDE_BOOK');const now=Date.now(),bars:Bar[]=rawBars.map((x:any)=>({h:num(x[2]),l:num(x[3]),c:num(x[4]),v:num(x[5]),closeTime:num(x[6])})).filter(x=>x.closeTime<now-250).slice(-75);if(bars.length<35)throw Error('INSUFFICIENT_CLOSED_BARS');const mid=(b.bid+b.ask)/2,closes=bars.map(x=>x.c),a=atr(bars);if(!(a>0))throw Error('BAD_ATR');const r20=bars.slice(-20),r12=bars.slice(-12),recentHigh=Math.max(...r20.map(x=>x.h)),recentLow=Math.min(...r12.map(x=>x.l)),lastClose=bars.at(-1)!.c,prevClose=bars.at(-2)!.c,e8=ema(closes.slice(-40),8),e21=ema(closes.slice(-55),21),atrPct=a/mid*100,recovery=mid>=lastClose*.9988&&lastClose>=prevClose*.9970,trendOk=e8>=e21*.9905,pullbackPct=(recentHigh-mid)/recentHigh*100,bouncePct=(mid-recentLow)/recentLow*100,forecast=buildForecast(closes,atrPct,e8,e21,recovery,trendOk,bouncePct,pullbackPct);const shockCloses=closes.slice(-16);let shockUp15=0,shockAge=99;for(let i=1;i<shockCloses.length;i++){const r=shockCloses[i-1]>0?(shockCloses[i]/shockCloses[i-1]-1)*10000:0;if(r>shockUp15){shockUp15=r;shockAge=shockCloses.length-1-i;}}return{symbol,bid:b.bid,ask:b.ask,mid,spreadBps:b.spreadBps,atr:a,atrPct,ema8:e8,ema21:e21,recentLow,recentHigh,pullbackPct,bouncePct,lastClose,prevClose,recovery,trendOk,forecast,shock_up_15m_bps:shockUp15,shock_age_min:shockAge};}

type EntryPulse={ret5_bps:number;ret15_bps:number;buy_ratio:number;trade_count:number};
async function loadEntryPulse(symbol:string):Promise<EntryPulse>{
  const raw=await marketJson(`/api/v3/aggTrades?symbol=${encodeURIComponent(symbol)}&limit=200`,1600);
  if(!Array.isArray(raw)||raw.length<2)throw Error('ENTRY_PULSE_EMPTY');
  const now=Date.now(),rows=raw.map((x:any)=>({p:num(x.p),q:num(x.q),t:num(x.T),buy:!Boolean(x.m)})).filter((x:any)=>x.p>0&&x.q>0&&x.t>0);
  if(rows.length<2)throw Error('ENTRY_PULSE_BAD');
  const last=rows.at(-1)!.p,base=(ms:number)=>{const cutoff=now-ms;const r=rows.find((x:any)=>x.t>=cutoff);return r?.p||rows[0].p;};
  const recent=rows.filter((x:any)=>x.t>=now-15000);let buy=0,total=0;
  for(const x of recent){const v=x.p*x.q;total+=v;if(x.buy)buy+=v;}
  return{ret5_bps:(last/base(5000)-1)*10000,ret15_bps:(last/base(15000)-1)*10000,buy_ratio:total>0?buy/total:.5,trade_count:recent.length};
}
async function loadUniverse(state:State){
  const[raw24,rawBooks]=await Promise.all([marketJson('/api/v3/ticker/24hr',7000),marketJson('/api/v3/ticker/bookTicker',5000)]);
  if(!Array.isArray(raw24)||!Array.isArray(rawBooks))throw Error('GLOBAL_RADAR_BAD_PAYLOAD');
  const books=new Map<string,Book>();
  for(const v of rawBooks as any[]){
    const symbol=String(v?.symbol||'').toUpperCase(),bid=num(v?.bidPrice),ask=num(v?.askPrice);
    if(bid>0&&ask>bid){const mid=(bid+ask)/2;books.set(symbol,{bid,ask,spreadBps:(ask-bid)/mid*10000});}
  }
  const prevPrices=(((state.last_scan as any)?.universe_prices||{}) as Record<string,number>),priceSnapshot:Record<string,number>={},scored:any[]=[];
  for(const v of raw24 as any[]){
    const symbol=String(v?.symbol||'').toUpperCase();
    if(!/^[A-Z0-9]{2,20}USDT$/.test(symbol)||EXCLUDED.has(symbol))continue;
    const last=num(v?.lastPrice),high=num(v?.highPrice),low=num(v?.lowPrice),qv=num(v?.quoteVolume),count=num(v?.count),chg=num(v?.priceChangePercent),book=books.get(symbol);
    if(!(last>0)||!book||book.spreadBps>35||qv<3_000_000||count<10_000||chg<-45||chg>40)continue;
    priceSnapshot[symbol]=last;
    const prev=num(prevPrices[symbol]),shortBps=prev>0?(last/prev-1)*10000:0,range=(high-low)/last*100,
      liq=clip((Math.log10(Math.max(qv,1))-6.3)/3.7),act=clip((Math.log10(Math.max(count,1))-3.8)/2.5),
      vol=clip(range/18),dipBias=clip((8-chg)/28),mom24=clip((chg+8)/24),burstUp=clip((shortBps+4)/55),burstAbs=clip(Math.abs(shortBps)/70),
      radar=clip(.50+liq*.19+act*.11+vol*.13+dipBias*.07,.45,.98),momentum=clip(mom24*.45+burstUp*.55);
    const c:Candidate={symbol,radar_score:radar,price_change_pct:chg,spread_bps:book.spreadBps,quote_volume:qv,trades_24h:count,momentum_score:momentum,volatility_score:vol,short_momentum_bps:shortBps};
    const global=radar*.55+liq*.20+act*.10+vol*.15,
      burst=burstUp*.45+burstAbs*.15+act*.15+vol*.15+liq*.10,
      dip=clip((-chg+6)/24)*.40+vol*.22+liq*.18+act*.10+radar*.10,
      activity=act*.34+vol*.28+liq*.20+radar*.13+burstAbs*.05,
      interrupt=burst*.48+clip((shortBps-6)/90)*.27+activity*.10+global*.10+clip((20-book.spreadBps)/20)*.05;
    scored.push({c,global,burst,dip,activity,interrupt});
  }
  if(!scored.length)throw Error('GLOBAL_RADAR_EMPTY');
  const bySymbol=new Map<string,Candidate>(scored.map(x=>[x.c.symbol,x.c])),chosen=new Map<string,Candidate>();
  const add=(x:any,lane:string,key:string)=>{if(chosen.size>=MAX_DEEP_SCAN||chosen.has(x.c.symbol))return false;chosen.set(x.c.symbol,{...(x.c as Candidate),lane,selection_score:num(x[key])});return true;};
  const addLane=(arr:any[],lane:string,key:string)=>{let used=0;for(const x of arr){if(chosen.size>=CORE_SCAN_SLOTS||used>=LANE_TOP)break;if(add(x,lane,key))used++;}};
  addLane([...scored].sort((a,b)=>b.global-a.global),'GLOBAL','global');
  addLane([...scored].sort((a,b)=>b.burst-a.burst),'BURST','burst');
  addLane([...scored].sort((a,b)=>b.dip-a.dip),'DIP','dip');
  addLane([...scored].sort((a,b)=>b.activity-a.activity),'ACTIVITY','activity');
  const prevMem=Array.isArray((state.last_scan as any)?.radar_memory)?((state.last_scan as any).radar_memory as MemoryItem[]):[],
    alive=prevMem.filter(x=>num(x.until)>Date.now()).sort((a,b)=>(num(b.score)+num(b.opp)+num(b.cont))-(num(a.score)+num(a.opp)+num(a.cont)));
  for(const m of alive){
    if(chosen.size>=CORE_SCAN_SLOTS)break;
    const cur=bySymbol.get(String(m.symbol));
    if(cur&&!chosen.has(cur.symbol))chosen.set(cur.symbol,{...cur,lane:'MEMORY',selection_score:num(m.score)});
  }
  if(chosen.size<CORE_SCAN_SLOTS){
    for(const x of [...scored].sort((a,b)=>Math.max(b.global,b.burst,b.dip,b.activity)-Math.max(a.global,a.burst,a.dip,a.activity))){
      if(chosen.size>=CORE_SCAN_SLOTS)break;
      add(x,'CORE_FILL','global');
    }
  }
  const coreCount=chosen.size;
  let interruptCount=0;
  for(const x of [...scored].filter(x=>!chosen.has(x.c.symbol)).sort((a,b)=>b.interrupt-a.interrupt)){
    if(interruptCount>=INTERRUPT_SLOTS)break;
    if(add(x,'INTERRUPT','interrupt'))interruptCount++;
  }
  const explorerPool=[...scored].filter(x=>!chosen.has(x.c.symbol)).sort((a,b)=>a.c.symbol.localeCompare(b.c.symbol));
  let explorerCount=0;
  if(explorerPool.length){
    const cursor=(Math.floor(Date.now()/60000)*EXPLORER_SLOTS)%explorerPool.length;
    for(let i=0;i<explorerPool.length&&explorerCount<EXPLORER_SLOTS;i++){
      const x=explorerPool[(cursor+i)%explorerPool.length];
      if(add(x,'EXPLORER','activity'))explorerCount++;
    }
  }
  if(chosen.size<MAX_DEEP_SCAN){
    for(const x of [...scored].sort((a,b)=>Math.max(b.global,b.burst,b.dip,b.activity)-Math.max(a.global,a.burst,a.dip,a.activity))){
      if(chosen.size>=MAX_DEEP_SCAN)break;
      add(x,'COVERAGE_FILL','global');
    }
  }
  const rows=[...chosen.values()].slice(0,MAX_DEEP_SCAN),laneCounts=rows.reduce((a:any,c:any)=>(a[c.lane||'UNKNOWN']=(a[c.lane||'UNKNOWN']||0)+1,a),{});
  const lightSnapshot=scored.map(x=>({symbol:x.c.symbol,price:priceSnapshot[x.c.symbol],radar_score:x.c.radar_score,short_momentum_bps:x.c.short_momentum_bps??0,
    spread_bps:x.c.spread_bps,price_change_pct:x.c.price_change_pct,global:x.global,burst:x.burst,dip:x.dip,activity:x.activity,interrupt:x.interrupt}));
  return{rows,books,source:'MULTI_RADAR_UNIVERSE',ageSec:0,stale:false,observedAt:iso(),universeCount:scored.length,laneCounts,prevMemory:alive,priceSnapshot,lightSnapshot,
    selectionPlan:{max_deep_scan:MAX_DEEP_SCAN,core_slots:CORE_SCAN_SLOTS,interrupt_slots:INTERRUPT_SLOTS,explorer_slots:EXPLORER_SLOTS,core_count:coreCount,interrupt_count:interruptCount,explorer_count:explorerCount}};
}
async function loadState(sql:any):Promise<State>{const rows=await sql`select * from public.brian_dip_multiasset_state where engine_id=${ENGINE_ID} limit 1`;if(rows.length)return rows[0] as State;const now=new Date(),until=new Date(now.getTime()+18*3600_000),scan={status:'BOOT',policy_version:POLICY_VERSION,engine_version:ENGINE_VERSION};const out=await sql`insert into public.brian_dip_multiasset_state (engine_id,started_at,run_until,starting_equity,cash,realized_pnl,trade_count,win_count,loss_count,positions,cooldowns,last_scan,last_eval_minute,updated_at,enabled,shadow_only,live_execution) values (${ENGINE_ID},${now},${until},1000,1000,0,0,0,0,${sql.json({})},${sql.json({})},${sql.json(scan)},null,${now},true,true,false) returning *`;return out[0] as State;}
function equity(cash:number,positions:Record<string,Position>,markets:Map<string,Market>){let e=cash;for(const p of Object.values(positions))e+=p.qty*(markets.get(p.symbol)?.bid??p.entry);return e;}
function riskMode(state:State,current:number,radarStale:boolean){const start=Math.max(1,num(state.starting_equity,1000)),dd=(start-current)/start,closed=num(state.win_count)+num(state.loss_count),wr=closed?num(state.win_count)/closed:0,streak=num((state.last_scan as any)?.loss_streak),lastLoss=num((state.last_scan as any)?.last_loss_at);if(dd>=.025||(streak>=3&&Date.now()-lastLoss<LOSS_STREAK_PAUSE_MS))return{name:'FROZEN',gross:0,risk:0,minScore:.99,minOpp:.99,frozen:true,reason:dd>=.025?'SESSION_DRAWDOWN':'LOSS_STREAK',dd,wr};if(radarStale)return{name:'DEFENSIVE',gross:.035,risk:.0020,minScore:.80,minOpp:.78,frozen:false,reason:'RADAR_STALE_FALLBACK',dd,wr};if(closed<12||wr<.40||num(state.realized_pnl)<0)return{name:'DEFENSIVE',gross:.045,risk:.0025,minScore:.78,minOpp:.76,frozen:false,reason:'EARN_SCALING_RIGHTS',dd,wr};if(closed<30)return{name:'COLD',gross:.065,risk:.0035,minScore:.75,minOpp:.73,frozen:false,reason:'CALIBRATING',dd,wr};return{name:'NORMAL',gross:.09,risk:.0045,minScore:.72,minOpp:.71,frozen:false,reason:'WARM',dd,wr};}
function tierOf(score:number,opp:number,net:number,cont:number){if(score>=.91&&opp>=.87&&net>=65&&cont>=.78)return'A+';if(score>=.86&&opp>=.81&&net>=32&&cont>=.72)return'A';if(score>=.81&&opp>=.76&&net>=14&&cont>=.66)return'B+';return'B';}
function evaluate(c:Candidate,m:Market,mode:any):Eval{
  const atrBps=m.atrPct*100,slip=Math.max(2,m.spreadBps/2),cost=2*FEE_BPS+m.spreadBps+2*slip,
    reqPull=Math.max(.45,m.atrPct*.60),reqBounce=Math.max(.12,m.atrPct*.13),
    struct=Math.max(0,(m.recentHigh-m.ask)/m.ask*10000),
    quality=clip(.58+(c.radar_score-.64)*.72+clip(c.momentum_score-.48,0,.52)*.18+m.forecast.continuation*.10,.54,1.02),
    gross=Math.min(Math.max(struct,atrBps*.42),atrBps*1.90)*quality,unc=Math.max(7,atrBps*.15,cost*.30),net=gross-cost-unc,
    score=clip(c.radar_score*.27+clip(m.pullbackPct/Math.max(reqPull*1.8,.58))*.16+clip(m.bouncePct/Math.max(reqBounce*2.8,.22))*.10+(m.recovery?.08:0)+(m.trendOk?.07:0)+m.forecast.continuation*.17+clip(net/120)*.12),
    opp=clip(score*.50+c.radar_score*.18+clip(c.momentum_score)*.06+m.forecast.continuation*.15+clip(net/120)*.09),
    explosionScore=clip(score*.28+opp*.24+m.forecast.continuation*.16+clip(net/140)*.16+clip(gross/Math.max(cost*2,1))*.08+clip(c.volatility_score)*.04+clip(c.momentum_score)*.04),
    impulse5=Math.max(180,atrBps*1.50),impulse3=Math.max(125,atrBps*1.05),impulse1=Math.max(85,atrBps*.75),f=m.forecast,
    rawConflict=(f.ret15_bps<-150&&f.expected_15m_bps>15)||(f.ret5_bps<-70&&f.ret3_bps<-45)||(f.ret3_bps<-80&&f.ret5_bps<20),
    regimeConflict=f.ret15_bps<-80&&f.ret30_bps<-100&&f.ret60_bps<-180&&f.trend_bps<-20,
    shockLimit=Math.max(240,atrBps*1.35),shockMemoryBlock=m.shock_up_15m_bps>=shockLimit&&m.shock_age_min<=12&&m.pullbackPct<Math.max(1.5,m.atrPct*.80),
    pullbackOverride=m.pullbackPct>=Math.max(1.5,m.atrPct*.85),long30=Math.max(500,atrBps*4.0),long60=Math.max(800,atrBps*6.5);

  const radarFloor=mode.name==='DEFENSIVE'?.72:mode.name==='FROZEN'?.72:.68;
  const gates={
    radar:c.radar_score>=radarFloor,
    spread:m.spreadBps<=(mode.name==='DEFENSIVE'||mode.name==='FROZEN'?10:14),
    day_move:Math.abs(c.price_change_pct)<=40,
    pullback:m.pullbackPct>=reqPull,bounce:m.bouncePct>=reqBounce,recovery:m.recovery,trend:m.trendOk,
    not_chasing:m.mid<=m.recentHigh*(1-Math.max(.0015,m.atrPct*.10/100)),
    forecast:f.continuation>=.60&&f.expected_15m_bps>=4&&f.expected_30m_bps>=0,
    forecast_consistency:!rawConflict,regime_guard:!regimeConflict,shock_memory:!shockMemoryBlock,
    impulse:f.ret5_bps<=impulse5&&f.ret3_bps<=impulse3&&f.ret1_bps<=impulse1,
    extension:m.bouncePct<=Math.max(3.0,m.atrPct*2.0),
    long_extension:(f.ret30_bps<=long30||pullbackOverride)&&(f.ret60_bps<=long60||pullbackOverride),
    economics:net>=Math.max(10,cost*.40)&&gross>=cost*1.60
  };

  const failed=Object.entries(gates).filter(([,v])=>!v).map(([k])=>k),allGates=failed.length===0,
    coreRaw=!mode.frozen&&score>=mode.minScore&&opp>=mode.minOpp&&allGates,
    winnerRaw=!mode.frozen&&!coreRaw&&allGates&&score>=Math.max(.79,mode.minScore-.01)&&opp>=Math.max(.73,mode.minOpp-.012)&&f.continuation>=.63&&f.expected_15m_bps>=8&&f.expected_30m_bps>=5&&net>=Math.max(18,cost*.55),
    critical=['spread','day_move','bounce','recovery','trend','forecast','forecast_consistency','regime_guard','shock_memory','extension','long_extension'],
    criticalOk=critical.every(k=>gates[k]),
    hunterAllowed=failed.every(k=>['radar','economics','pullback','not_chasing','impulse'].includes(k)),
    hunterRaw=!mode.frozen&&!coreRaw&&!winnerRaw&&criticalOk&&hunterAllowed&&score>=.80&&opp>=.76&&explosionScore>=.72&&f.continuation>=.72&&net>=14&&cost<=42&&gross>=cost*1.60&&
      (!failed.includes('radar')||score>=.84&&opp>=.80&&net>=22)&&(!failed.includes('economics')||net>=22)&&
      (!failed.includes('pullback')||f.expected_15m_bps>=18&&f.expected_30m_bps>=18)&&(!failed.includes('not_chasing')||f.expected_15m_bps>=25&&f.expected_30m_bps>=25)&&
      (!failed.includes('impulse')||score>=.84&&opp>=.80&&explosionScore>=.80&&f.continuation>=.76&&net>=28),
    scoutRaw=!mode.frozen&&!coreRaw&&!winnerRaw&&!hunterRaw&&(
      (failed.length===1&&failed[0]==='economics'&&score>=.81&&opp>=.75&&f.continuation>=.67&&net>=5&&cost<=40&&gross>=cost*1.20)||
      (failed.length===0&&score>=.74&&opp>=.64&&explosionScore>=.62&&f.continuation>=.66&&net>=20&&cost<=38&&gross>=cost*1.45)
    );

  const safetyExceptRadar=['spread','day_move','bounce','recovery','trend','not_chasing','forecast','forecast_consistency','regime_guard','shock_memory','impulse','extension','long_extension','economics']
    .every(k=>gates[k]);
  const surgeRaw=!mode.frozen&&failed.length===1&&failed[0]==='radar'&&(c.lane==='BURST'||c.lane==='INTERRUPT')&&safetyExceptRadar&&
    c.radar_score>=.64&&score>=.75&&opp>=.68&&explosionScore>=.67&&f.continuation>=.76&&net>=25&&cost<=42;
  const emergencyStrength=(score>=.86&&opp>=.81&&f.continuation>=.78&&net>=55)||(score>=.80&&opp>=.75&&f.continuation>=.67&&net>=80&&explosionScore>=.60);
  const emergencyRaw=mode.frozen&&mode.reason==='LOSS_STREAK'&&safetyExceptRadar&&c.radar_score>=.64&&emergencyStrength&&cost<=45;

  const tier=tierOf(score,opp,net,f.continuation),capitalScore=clip(score*.34+opp*.30+f.continuation*.20+clip(net/100)*.16);
  const reason=emergencyRaw?'V880_EMERGENCY_SCOUT_CONFIRM':surgeRaw?'V880_SURGE_SCOUT_CONFIRM':coreRaw?'V873_CORE_CONFIRM':winnerRaw?'V873_WINNER_SCOUT_CONFIRM':mode.frozen?`RISK_FROZEN_${mode.reason}`:`WAIT_${failed.join('+')||'SCORE'}`;
  return{coreRaw,winnerRaw,scoutRaw,hunterRaw,surgeRaw,emergencyRaw,score,opp,cost,gross,net,tier,capitalScore,explosionScore,forecast:f,reason,gates};
}
function capitalProfile(mode:any,ev:Eval,style:string){
  let gross=mode.gross,risk=mode.risk;
  if(mode.name==='DEFENSIVE'){gross=ev.tier==='A+'?.060:ev.tier==='A'?.055:ev.tier==='B+'?.050:.045;risk=ev.tier==='A+'?.0032:ev.tier==='A'?.0030:ev.tier==='B+'?.0028:.0025;}
  else if(mode.name==='COLD'){gross=ev.tier==='A+'?.11:ev.tier==='A'?.09:ev.tier==='B+'?.075:.065;risk=ev.tier==='A+'?.0045:ev.tier==='A'?.0042:ev.tier==='B+'?.0038:.0035;}
  else if(mode.name==='NORMAL'){gross=ev.tier==='A+'?.16:ev.tier==='A'?.13:ev.tier==='B+'?.10:.09;risk=ev.tier==='A+'?.0060:ev.tier==='A'?.0055:ev.tier==='B+'?.0050:.0045;}
  if(style==='EMERGENCY_SCOUT')return{gross:.015,risk:.0006};
  if(style==='SURGE_SCOUT')return{gross:Math.min(.025,Math.max(.018,gross*.45)),risk:Math.min(.0010,Math.max(.0007,risk*.40))};
  if(style==='RECOVERY_SCOUT'||style==='EARLY_SCOUT'){gross*=.45;risk*=.45;}
  else if(style==='HUNTER'){gross*=.80;risk*=.80;}
  else if(style==='WINNER_SCOUT'){gross*=.52;risk*=.52;}
  if(ev.explosionScore>=.90)gross=Math.max(gross,mode.name==='DEFENSIVE'?.13:.16);
  else if(ev.explosionScore>=.86)gross=Math.max(gross,mode.name==='DEFENSIVE'?.10:.13);
  else if(ev.explosionScore>=.82)gross=Math.max(gross,mode.name==='DEFENSIVE'?.075:.10);
  return{gross:Math.min(.18,gross),risk:Math.min(.006,risk)};
}

function reentryContext(guard:any,m:Market,ev:Eval,now:number){
  if(!guard||num(guard.exit_at)<=0||now-num(guard.exit_at)>120*60_000)return{active:false,eligible:false,type:'FRESH',reason:'FRESH',guard};
  const exitAt=num(guard.exit_at),exitPrice=num(guard.exit_price),age=now-exitAt;
  guard.post_exit_low=Math.min(num(guard.post_exit_low,exitPrice||m.mid),m.mid);
  guard.post_exit_high=Math.max(num(guard.post_exit_high,exitPrice||m.mid),m.mid);
  const low=num(guard.post_exit_low,m.mid),lossStreak=num(guard.symbol_loss_streak),attempts=num(guard.reentry_count),pnl=num(guard.pnl),
    exhaustedUntil=num(guard.wave_exhausted_until),reason=String(guard.reason||''),
    critical=Boolean(ev.gates.spread&&ev.gates.forecast_consistency&&ev.gates.regime_guard&&ev.gates.shock_memory&&ev.gates.extension&&ev.gates.long_extension&&ev.gates.economics);

  if(lossStreak>=2&&num(guard.until)>now)return{active:true,eligible:false,type:'CIRCUIT',reason:`CIRCUIT_BREAKER_${lossStreak}`,guard};

  if(attempts>=1&&age>=12*60_000){
    const resetFrac=Math.max(.008,Math.min(.025,m.atrPct*.70/100)),reclaimFrac=Math.max(.0035,Math.min(.012,m.atrPct*.28/100)),
      resetSeen=low<=exitPrice*(1-resetFrac),reclaimed=m.mid>=low*(1+reclaimFrac),
      noChase=m.mid<=exitPrice*(1+Math.max(.006,Math.min(.018,m.atrPct*.55/100))),
      thesis=critical&&ev.score>=.79&&ev.opp>=.71&&ev.explosionScore>=.70&&ev.forecast.continuation>=.72&&ev.net>=30&&
        ev.forecast.expected_15m_bps>=25&&ev.forecast.expected_30m_bps>=30&&ev.forecast.ret1_bps<=90&&ev.forecast.ret3_bps<=170;
    if(resetSeen&&reclaimed&&noChase&&thesis){
      guard.reentry_count=0;guard.wave_started_at=now;guard.wave_exhausted_until=0;guard.wave_profit_bank=0;
      return{active:true,eligible:true,type:'NEW_WAVE_RESET',reason:'V880_NEW_WAVE_RESET',guard};
    }
  }

  if(exhaustedUntil>now)return{active:true,eligible:false,type:'EXHAUSTED',reason:'WAIT_WAVE_EXHAUSTED',guard};
  if(attempts>=1)return{active:true,eligible:false,type:'LIMIT',reason:'WAIT_WAVE_REENTRY_LIMIT',guard};

  if(pnl<-.01&&reason==='STOP'){
    if(age<2*60_000)return{active:true,eligible:false,type:'STOP_RECLAIM',reason:'WAIT_STOP_RESET',guard};
    const reclaimFrac=Math.max(.0025,Math.min(.010,m.atrPct*.20/100)),reclaimed=m.mid>=low*(1+reclaimFrac),
      anchor=Math.max(exitPrice,num(guard.prior_entry,exitPrice)),maxAbove=Math.max(.004,Math.min(.012,m.atrPct*.35/100)),
      noChase=m.mid<=anchor*(1+maxAbove),
      thesis=critical&&ev.score>=.82&&ev.opp>=.74&&ev.explosionScore>=.74&&ev.forecast.continuation>=.72&&ev.net>=35&&
        ev.forecast.expected_15m_bps>=30&&ev.forecast.expected_30m_bps>=35&&ev.forecast.ret1_bps<=90&&ev.forecast.ret3_bps<=160;
    const eligible=reclaimed&&noChase&&thesis;
    return{active:true,eligible,type:'STOP_RECLAIM',reason:eligible?'V880_STOP_RECLAIM':!reclaimed?'WAIT_STOP_RECLAIM':!noChase?'WAIT_REENTRY_NO_CHASE':'WAIT_REENTRY_THESIS',guard};
  }

  if(pnl<-.01){
    const hardUntil=Math.max(exitAt+12*60_000,num(guard.until));
    return{active:true,eligible:false,type:'LOSS_RESET',reason:hardUntil>now?'WAIT_LOSS_RESET':'WAIT_NEW_WAVE',guard};
  }

  if(pnl>.01){
    if(age<5*60_000)return{active:true,eligible:false,type:'PROFIT_RESET',reason:'WAIT_PROFIT_RESET',guard};
    const resetFrac=Math.max(.004,Math.min(.015,m.atrPct*.45/100)),reclaimFrac=Math.max(.0025,Math.min(.009,m.atrPct*.22/100)),
      resetSeen=low<=exitPrice*(1-resetFrac),reclaimed=m.mid>=low*(1+reclaimFrac),
      maxAbove=Math.max(.0045,Math.min(.012,m.atrPct*.45/100)),noChase=m.mid<=exitPrice*(1+maxAbove),
      bank=num(guard.wave_profit_bank,pnl),
      thesis=critical&&ev.score>=.84&&ev.opp>=.79&&ev.explosionScore>=.78&&ev.forecast.continuation>=.76&&ev.net>=40&&
        ev.forecast.expected_15m_bps>=35&&ev.forecast.expected_30m_bps>=45&&ev.forecast.ret1_bps<=80&&ev.forecast.ret3_bps<=150;
    const eligible=bank>0&&resetSeen&&reclaimed&&noChase&&thesis;
    return{active:true,eligible,type:'PROFIT_RESET',reason:eligible?'V880_PROFIT_RESET':!resetSeen?'WAIT_PROFIT_DIP':!reclaimed?'WAIT_PROFIT_RECLAIM':!noChase?'WAIT_PROFIT_NO_CHASE':bank<=0?'WAIT_PROFIT_BANK':'WAIT_PROFIT_THESIS',guard};
  }

  if(num(guard.until)>now)return{active:true,eligible:false,type:'COOLDOWN',reason:`COOLDOWN_${reason||'EXIT'}`,guard};
  return{active:false,eligible:false,type:'FRESH',reason:'FRESH',guard};
}
function pulseOk(p:EntryPulse,reentryType:string,grossFrac:number,style:string){
  if(style==='EMERGENCY_SCOUT')return p.buy_ratio>=.58&&p.ret5_bps>=-5&&p.ret15_bps>=0&&p.ret5_bps<=45&&p.ret15_bps<=90;
  if(style==='SURGE_SCOUT')return p.buy_ratio>=.55&&p.ret5_bps>=-5&&p.ret15_bps>=-10&&p.ret5_bps<=60&&p.ret15_bps<=120;
  if(grossFrac>=.08)return p.buy_ratio>=.58&&p.ret5_bps>=-5&&p.ret15_bps>=0&&p.ret5_bps<=50&&p.ret15_bps<=110;
  if(grossFrac>=.05)return p.buy_ratio>=.54&&p.ret5_bps>=-10&&p.ret15_bps>=-12&&p.ret5_bps<=65&&p.ret15_bps<=130;
  if(reentryType==='STOP_RECLAIM')return p.buy_ratio>=.52&&p.ret5_bps>=-10&&p.ret15_bps>=-20&&p.ret5_bps<=55&&p.ret15_bps<=110;
  if(reentryType==='PROFIT_RESET'||reentryType==='NEW_WAVE_RESET')return p.buy_ratio>=.54&&p.ret5_bps>=-8&&p.ret15_bps>=-18&&p.ret5_bps<=55&&p.ret15_bps<=120;
  return p.buy_ratio>=.46&&p.ret5_bps>=-22&&p.ret15_bps>=-40&&p.ret5_bps<=85&&p.ret15_bps<=160;
}

function monsterScaleTarget(modeName:string,stage:number){
  if(stage===1)return modeName==='NORMAL'?.15:modeName==='COLD'?.12:.10;
  if(stage===2)return modeName==='NORMAL'?.22:modeName==='COLD'?.20:.18;
  return modeName==='NORMAL'?.30:modeName==='COLD'?.28:.25;
}
function monsterScaleSignal(p:Position,m:Market,ev:Eval,mode:any){
  const stage=Math.max(0,Math.min(3,num(p.scale_stage))),next=stage+1;if(next>3||mode.frozen||p.harvest_armed)return null;
  if(['EMERGENCY_SCOUT','SURGE_SCOUT'].includes(String(p.entry_style||'')))return null;
  const held=Date.now()-Date.parse(p.opened_at),lastScaleAt=num(p.last_scale_at),lastScalePrice=num(p.last_scale_price,p.entry),
    cost=num(p.estimated_cost_bps,25),winnerBps=p.entry>0?(m.bid/p.entry-1)*10000:0,
    stepBps=lastScalePrice>0?(m.bid/lastScalePrice-1)*10000:0,
    critical=Boolean(ev.gates.spread&&ev.gates.day_move&&ev.gates.recovery&&ev.gates.trend&&ev.gates.forecast&&ev.gates.forecast_consistency&&ev.gates.regime_guard&&ev.gates.shock_memory&&ev.gates.long_extension&&ev.gates.economics),
    notVertical=ev.forecast.ret1_bps<=65&&ev.forecast.ret3_bps<=140,
    timeOk=held>=55_000&&(!lastScaleAt||Date.now()-lastScaleAt>=50_000),
    proofOk=winnerBps>=Math.max(45,cost+12)&&(stage===0||stepBps>=28);
  if(!critical||!notVertical||!timeOk||!proofOk)return null;
  const f=ev.forecast;
  const ok1=ev.score>=.84&&ev.opp>=.78&&ev.explosionScore>=.82&&f.continuation>=.80&&ev.net>=55&&f.expected_15m_bps>=45&&f.expected_30m_bps>=75;
  const ok2=ev.score>=.87&&ev.opp>=.82&&ev.explosionScore>=.88&&f.continuation>=.84&&ev.net>=75&&f.expected_15m_bps>=65&&f.expected_30m_bps>=100&&f.ret1_bps<=60&&f.ret3_bps<=130;
  const ok3=ev.score>=.89&&ev.opp>=.85&&ev.explosionScore>=.92&&f.continuation>=.88&&ev.net>=95&&f.expected_15m_bps>=85&&f.expected_30m_bps>=140&&f.ret1_bps<=55&&f.ret3_bps<=120;
  const ok=next===1?ok1:next===2?ok2:ok3;if(!ok)return null;
  return{stage:next,targetFraction:monsterScaleTarget(mode.name,next),winnerBps,stepBps};
}
function monsterScalePulseOk(p:EntryPulse,stage:number){
  if(stage===1)return p.buy_ratio>=.55&&p.ret5_bps>=-5&&p.ret15_bps>=-10&&p.ret5_bps<=45&&p.ret15_bps<=90;
  if(stage===2)return p.buy_ratio>=.58&&p.ret5_bps>=0&&p.ret15_bps>=0&&p.ret5_bps<=50&&p.ret15_bps<=100;
  return p.buy_ratio>=.62&&p.ret5_bps>=0&&p.ret15_bps>=5&&p.ret5_bps<=45&&p.ret15_bps<=90;
}
async function insertEvent(sql:any,row:any){await sql`insert into public.brian_dip_multiasset_events (engine_id,observed_at,symbol,action,price,qty,notional,pnl,reason,metadata) values (${ENGINE_ID},${new Date(row.observed_at)},${row.symbol},${row.action},${row.price},${row.qty},${row.notional},${row.pnl},${row.reason},${sql.json(row.metadata||{})})`;}
async function learnMissed(sql:any,state:State,now:number){const prev=(state.last_scan as any)?.learning_summary??{},lastAt=Date.parse(String((state.last_scan as any)?.learning_at||''));if(Number.isFinite(lastAt)&&now-lastAt<5*60_000)return{at:String((state.last_scan as any)?.learning_at||iso()),summary:prev};try{const from=new Date(now-80*60_000),to=new Date(now-62*60_000),rows=await sql`select distinct on (symbol) evaluation_id,observed_at,symbol,price,signal_score,reason,metadata from public.brian_dip_multiasset_evaluations where engine_id=${ENGINE_ID} and action='WAIT' and signal_score>=0.70 and observed_at>=${from} and observed_at<=${to} and coalesce(metadata->>'outcome_checked','false')<>'true' order by symbol,signal_score desc limit 4`;for(const r of rows){try{const start=Date.parse(String(r.observed_at)),end=start+60*60_000,raw=await marketJson(`/api/v3/klines?symbol=${encodeURIComponent(String(r.symbol))}&interval=1m&startTime=${start}&endTime=${end}&limit=65`,4500);if(!Array.isArray(raw)||!raw.length)continue;const entry=num(r.price),hi=Math.max(...raw.map((x:any)=>num(x[2]))),lo=Math.min(...raw.map((x:any)=>num(x[3]))),up=pct(hi,entry),down=pct(lo,entry),label=up>=5?'MISSED_A_PLUS':up>=2.5?'MISSED_WINNER':down<=-2.5?'GOOD_REJECT':'NEUTRAL',meta={...((r.metadata||{}) as J),outcome_checked:true,outcome_label:label,future_max_60m_pct:up,future_min_60m_pct:down,outcome_checked_at:iso()};await sql`update public.brian_dip_multiasset_evaluations set metadata=${sql.json(meta)} where evaluation_id=${r.evaluation_id}`;}catch{}}const since=new Date(now-24*60*60_000),stats=await sql`select count(*) filter(where metadata->>'outcome_label'='MISSED_A_PLUS')::int as missed_a_plus,count(*) filter(where metadata->>'outcome_label'='MISSED_WINNER')::int as missed_winner,count(*) filter(where metadata->>'outcome_label'='GOOD_REJECT')::int as good_reject,count(*) filter(where metadata->>'outcome_checked'='true')::int as checked from public.brian_dip_multiasset_evaluations where engine_id=${ENGINE_ID} and observed_at>=${since}`;return{at:iso(),summary:{...(stats[0]||prev),window_hours:24}};}catch{return{at:iso(),summary:prev};}}
function triggerFill(reason:string,p:Position,m:Market,slip:number){const extra=slip+2;if(reason==='STOP')return p.stop*(1-extra/10000);if((reason==='HARVEST_TRAIL'||reason==='PROTECT_TRAIL'||reason==='PROFIT_RATCHET')&&p.trail)return p.trail*(1-extra/10000);return m.bid*(1-slip/10000);}
function cooldownMs(reason:string,pnl:number){if(pnl<0){if(reason==='THESIS_BREAK')return 12*60_000;return 30*60_000;}if(reason==='PROFIT_RATCHET'||reason==='PROTECT_TRAIL')return 3*60_000;if(reason==='HARVEST_TRAIL'||reason==='PEAK_REVERSAL'||reason==='FORECAST_REVERSAL')return 7*60_000;return 10*60_000;}
async function run(sql:any){const state=await loadState(sql),now=Date.now();if(!state.enabled||now>=Date.parse(state.run_until)){const scan={status:'WINDOW_COMPLETE',policy_version:POLICY_VERSION,engine_version:ENGINE_VERSION};await sql`update public.brian_dip_multiasset_state set enabled=false,updated_at=now(),last_scan=${sql.json(scan)} where engine_id=${ENGINE_ID}`;return{status:'WINDOW_COMPLETE',engine_version:ENGINE_VERSION};}const radar=await loadUniverse(state),symbols=[...new Set([...Object.keys(state.positions||{}),...radar.rows.map((x:Candidate)=>x.symbol)])].slice(0,MAX_DEEP_SCAN+MAX_POSITIONS),pairs=await Promise.all(symbols.map(async s=>{try{return[s,await loadMarket(s,radar.books.get(s))] as const;}catch(e){return[s,{error:e instanceof Error?e.message:String(e)}] as const;}})),markets=new Map<string,Market>(),marketErrors:J={};for(const[s,v]of pairs){if('error'in v)marketErrors[s]=v.error;else markets.set(s,v);}const positions={...(state.positions||{})},cooldowns={...(state.cooldowns||{})};let cash=num(state.cash),realized=num(state.realized_pnl),trades=num(state.trade_count),wins=num(state.win_count),losses=num(state.loss_count),lossStreak=num((state.last_scan as any)?.loss_streak),lastLossAt=num((state.last_scan as any)?.last_loss_at);const actions:any[]=[];
for(const[s,p]of Object.entries(positions)){const m=markets.get(s);if(!m)continue;const held=now-Date.parse(p.opened_at),slip=Math.max(2,m.spreadBps/2),cost=num(p.estimated_cost_bps,25),cont=m.forecast.continuation,prevCont=num(p.last_continuation,cont),prevExp15=num(p.last_expected_15m_bps,m.forecast.expected_15m_bps),prevExp30=num(p.last_expected_30m_bps,m.forecast.expected_30m_bps);p.last_continuation=cont;p.last_expected_15m_bps=m.forecast.expected_15m_bps;p.last_expected_30m_bps=m.forecast.expected_30m_bps;p.peak_continuation=Math.max(num(p.peak_continuation,cont),cont);p.peak_expected_15m_bps=Math.max(num(p.peak_expected_15m_bps,m.forecast.expected_15m_bps),m.forecast.expected_15m_bps);p.peak_expected_30m_bps=Math.max(num(p.peak_expected_30m_bps,m.forecast.expected_30m_bps),m.forecast.expected_30m_bps);p.max_price=Math.max(num(p.max_price,p.entry),m.bid);const atrBps=m.atrPct*100,mfe=(p.max_price-p.entry)/p.entry*10000,peakDrawBps=p.max_price>0?(p.max_price-m.bid)/p.max_price*10000:0,targetDist=Math.max(1e-12,p.target-p.entry),progress=clip((p.max_price-p.entry)/targetDist,0,2);const maxGrossNow=p.qty*p.max_price,maxFeeNow=maxGrossNow*FEE_BPS/10000,maxNetPnlNow=maxGrossNow-maxFeeNow-p.cost_basis;if(!p.harvest_armed&&m.bid>=p.target){p.harvest_armed=true;p.runner_mode=true;p.harvest_at=iso();p.winner_expansion=true;}if(p.harvest_armed){const exp15=m.forecast.expected_15m_bps,exp30=m.forecast.expected_30m_bps,explosive=cont>=.82&&exp15>=55&&exp30>=70&&m.forecast.ret3_bps>-25&&m.forecast.trend_bps>-15,strong=cont>=.72&&exp15>=25&&exp30>=35&&m.forecast.ret3_bps>-40,decay=(num(p.peak_continuation)>=.80&&(cont<=num(p.peak_continuation)-.18||exp15<=num(p.peak_expected_15m_bps)*.48)||prevCont>=.80&&cont<=prevCont-.16||prevExp15>=60&&exp15<=prevExp15*.45)&&m.forecast.ret1_bps<-25;p.runner_regime=explosive?'EXPLOSIVE_RUNNER':strong?'STRONG_RUNNER':decay?'FORECAST_DECAY':'PROTECT_RUNNER';const gap=explosive?Math.max(m.atr*1.45,p.max_price*.0115):strong?Math.max(m.atr*1.08,p.max_price*.0075):decay?Math.max(m.atr*.38,p.max_price*.0028):Math.max(m.atr*.62,p.max_price*.0045),explosiveProtect=explosive&&peakDrawBps>=28,keep=explosive?(explosiveProtect?.72:.48):strong?.75:decay?.90:.84,fillFactor=(1-(slip+2)/10000)*(1-FEE_BPS/10000),profitFloor=(p.cost_basis+Math.max(0,maxNetPnlNow)*keep)/Math.max(1e-12,p.qty*fillFactor),costFloor=p.entry*(1+(cost+8)/10000),noRetroCap=m.bid*(1-Math.max(2,m.spreadBps/2)/10000),candidate=Math.min(Math.max(costFloor,p.max_price-gap,profitFloor),noRetroCap);p.runner_trail=Math.max(num(p.runner_trail),candidate);p.trail=Math.max(num(p.trail),num(p.runner_trail));}else{const ratchetEligible=(progress>=.30||maxNetPnlNow>=p.cost_basis*.006)&&maxNetPnlNow>=Math.max(.12,p.cost_basis*.0025);if(ratchetEligible){const strong=cont>=.78&&m.forecast.expected_15m_bps>=25&&m.forecast.expected_30m_bps>=35,healthy=cont>=.68&&m.forecast.expected_15m_bps>=8&&m.forecast.expected_30m_bps>=8,lockRatio=strong?.22:healthy?.36:(cont>=.58&&m.forecast.expected_15m_bps>=0)?.55:.72,desiredPnl=maxNetPnlNow*lockRatio,fillFactor=(1-(slip+2)/10000)*(1-FEE_BPS/10000),rawTrail=(p.cost_basis+desiredPnl)/Math.max(1e-12,p.qty*fillFactor),noRetroCap=m.bid*(1-Math.max(2,m.spreadBps/2)/10000),candidate=Math.min(rawTrail,noRetroCap);if(candidate>num(p.trail)){p.trail=candidate;}if(!p.profit_lock_active)p.profit_lock_armed_at=iso();p.profit_lock_active=true;p.profit_lock_ratio=lockRatio;p.profit_lock_max_net_pnl=maxNetPnlNow;p.winner_expansion=true;}if(progress>=.72&&mfe>=Math.max(cost+45,atrBps*1.05)){p.winner_expansion=true;const floor=p.entry*(1+(cost+3)/10000),candidate=Math.max(floor,p.max_price-Math.max(m.atr*1.18,p.entry*.0090));p.trail=Math.max(num(p.trail),candidate);}}const peakCont=num(p.peak_continuation,cont),peakExp15=num(p.peak_expected_15m_bps,m.forecast.expected_15m_bps),forecastCollapse=Boolean(p.harvest_armed&&peakCont>=.80&&((cont<=peakCont-.18&&m.forecast.ret1_bps<-35)||(peakExp15>=60&&m.forecast.expected_15m_bps<=peakExp15*.45&&m.forecast.ret1_bps<-30))&&peakDrawBps>=Math.max(18,atrBps*.18)),momentumBreak=!m.trendOk&&m.forecast.ret3_bps<-18&&cont<.44,harvestTrail=Boolean(p.harvest_armed&&p.trail&&m.bid<=p.trail&&cont<.64),forecastReversal=Boolean(p.harvest_armed&&((cont<.39&&peakDrawBps>=Math.max(22,atrBps*.26))||forecastCollapse)),profitRatchetTrail=Boolean(!p.harvest_armed&&p.profit_lock_active&&p.trail&&m.bid<=p.trail),preHarvestTrail=Boolean(!p.harvest_armed&&!p.profit_lock_active&&p.trail&&m.bid<=p.trail&&cont<.43&&m.forecast.expected_15m_bps<0&&peakDrawBps>=Math.max(40,atrBps*.45)),thesisBreak=Boolean(!p.harvest_armed&&held>=3*60_000&&cont<.36&&m.forecast.expected_15m_bps<-8&&m.forecast.expected_30m_bps<-4&&m.bid<p.entry*(1-Math.max(12,cost*.35)/10000));let reason='';if(m.bid<=p.stop)reason='STOP';else if(thesisBreak)reason='THESIS_BREAK';else if(profitRatchetTrail)reason='PROFIT_RATCHET';else if(forecastReversal)reason='FORECAST_REVERSAL';else if(harvestTrail)reason='HARVEST_TRAIL';else if(p.harvest_armed&&momentumBreak&&peakDrawBps>=16)reason='PEAK_REVERSAL';else if(preHarvestTrail)reason='PROTECT_TRAIL';else if(!p.harvest_armed&&held>18*60_000&&cont<.34&&m.bid<p.entry*(1+cost/10000))reason='EDGE_DECAY';else if(p.harvest_armed&&held>=150*60_000)reason='HARVEST_MAX_HOLD';else if(!p.harvest_armed&&held>=80*60_000&&cont<.54)reason='MAX_HOLD';if(!reason)continue;const exit=triggerFill(reason,p,m,slip),gross=p.qty*exit,fee=gross*FEE_BPS/10000,pnl=gross-fee-p.cost_basis,maxGross=p.qty*p.max_price,maxFee=maxGross*FEE_BPS/10000,maxPnl=maxGross-maxFee-p.cost_basis,peakCapture=maxPnl>0?clip(Math.max(0,pnl)/maxPnl,0,1):null;cash+=gross-fee;realized+=pnl;trades++;if(pnl>.01){wins++;lossStreak=0;}else if(pnl<-.01){losses++;lossStreak++;lastLossAt=now;}delete positions[s];const prevGuard:any=cooldowns[s]||{},prevLossAt=num(prevGuard.last_loss_at),sameWindow=pnl<-.01&&prevLossAt>0&&now-prevLossAt<6*60*60_000,
symbolLossStreak=pnl<-.01?(sameWindow?Math.max(1,num(prevGuard.symbol_loss_streak))+1:1):0,reentryCount=num(p.reentry_attempt),
waveProfitBank=Math.max(0,num(p.wave_profit_bank)+pnl),waveStartedAt=num(p.wave_started_at,Date.parse(p.opened_at));
let until=now+cooldownMs(reason,pnl);
if(pnl<-.01){
  until=now+(reason==='STOP'?2*60_000:12*60_000);
  if(symbolLossStreak>=2)until=now+(symbolLossStreak>=3?8*60*60_000:4*60*60_000);
}else if(pnl>.01){
  until=now+5*60_000;
}
const waveExhaustedUntil=reentryCount>=1?now+45*60_000:0;
cooldowns[s]={until,reason,exit_price:exit,exit_at:now,pnl,symbol_loss_streak:symbolLossStreak,last_loss_at:pnl<-.01?now:prevLossAt,
  circuit_breaker:symbolLossStreak>=2,prior_entry:p.entry,post_exit_low:exit,post_exit_high:exit,reentry_count:reentryCount,
  wave_profit_bank:waveProfitBank,wave_started_at:waveStartedAt,wave_exhausted_until:waveExhaustedUntil};const row={observed_at:iso(),symbol:s,action:'SELL',price:exit,qty:p.qty,notional:gross,pnl,reason,metadata:{policy_version:POLICY_VERSION,engine_version:ENGINE_VERSION,risk_engine:RISK_ENGINE,entry_style:p.entry_style||'CORE',opportunity_tier:p.opportunity_tier||'B',entry:p.entry,harvest_trigger:p.target,harvest_armed:p.harvest_armed===true,winner_expansion:p.winner_expansion===true,profit_lock_active:p.profit_lock_active===true,profit_lock_ratio:p.profit_lock_ratio??null,profit_lock_max_net_pnl:p.profit_lock_max_net_pnl??null,profit_lock_armed_at:p.profit_lock_armed_at??null,harvest_at:p.harvest_at??null,stop:p.stop,trail:p.trail,runner_trail:p.runner_trail??null,max_price:p.max_price,target_progress:progress,peak_draw_bps:peakDrawBps,peak_capture_ratio:peakCapture,continuation_at_exit:cont,peak_continuation:p.peak_continuation??null,peak_expected_15m_bps:p.peak_expected_15m_bps??null,peak_expected_30m_bps:p.peak_expected_30m_bps??null,runner_regime:p.runner_regime??null,forecast:m.forecast,estimated_cost_bps:cost,symbol_loss_streak: symbolLossStreak,circuit_breaker_until:symbolLossStreak>=2?new Date(until).toISOString():null,shadow_fill_model:'TRIGGER_PLUS_BOUNDED_SLIPPAGE',shadow_only:true,live_execution:false}};await insertEvent(sql,row);actions.push(row);}
const mode=riskMode({...state,realized_pnl:realized,win_count:wins,loss_count:losses,last_scan:{...state.last_scan,loss_streak:lossStreak,last_loss_at:lastLossAt}} as State,equity(cash,positions,markets),radar.stale),
  evals:any[]=[],ready:any[]=[],openCandidates:any[]=[],
  prevCore=(((state.last_scan as any)?.core_streaks||{}) as Record<string,number>),
  prevWinner=(((state.last_scan as any)?.winner_streaks||{}) as Record<string,number>),
  prevHunter=(((state.last_scan as any)?.hunter_streaks||{}) as Record<string,number>),
  prevScout=(((state.last_scan as any)?.scout_streaks||{}) as Record<string,number>),
  prevSurge=(((state.last_scan as any)?.surge_streaks||{}) as Record<string,number>),
  coreStreaks:Record<string,number>={},winnerStreaks:Record<string,number>={},hunterStreaks:Record<string,number>={},scoutStreaks:Record<string,number>={},surgeStreaks:Record<string,number>={};
for(const c of radar.rows){
  const m=markets.get(c.symbol);if(!m)continue;
  const ev=evaluate(c,m,mode),open=!!positions[c.symbol],guard:any=cooldowns[c.symbol],reentry=reentryContext(guard,m,ev,now);
  const coreStreak=ev.coreRaw?Math.min(3,num(prevCore[c.symbol])+1):0,winnerStreak=ev.winnerRaw?Math.min(4,num(prevWinner[c.symbol])+1):0,
    hunterStreak=ev.hunterRaw?Math.min(3,num(prevHunter[c.symbol])+1):0,scoutStreak=ev.scoutRaw?Math.min(3,num(prevScout[c.symbol])+1):0,
    surgeRaw=Boolean(ev.surgeRaw||ev.emergencyRaw),surgeStreak=surgeRaw?Math.min(3,num(prevSurge[c.symbol])+1):0;
  coreStreaks[c.symbol]=coreStreak;winnerStreaks[c.symbol]=winnerStreak;hunterStreaks[c.symbol]=hunterStreak;scoutStreaks[c.symbol]=scoutStreak;surgeStreaks[c.symbol]=surgeStreak;

  const lateCore=ev.coreRaw&&(m.pullbackPct<.10||ev.forecast.ret1_bps>95||ev.forecast.ret3_bps>180)&&ev.forecast.expected_15m_bps<95,
    coreReady=ev.coreRaw&&coreStreak>=2&&!lateCore,winnerReady=ev.winnerRaw&&winnerStreak>=3,hunterReady=ev.hunterRaw&&hunterStreak>=2,scoutReady=ev.scoutRaw&&scoutStreak>=2,
    emergencyReady=Boolean(ev.emergencyRaw&&surgeStreak>=1),surgeReady=Boolean(ev.surgeRaw&&surgeStreak>=1),
    normalReady=!reentry.active&&(emergencyReady||surgeReady||coreReady||winnerReady||hunterReady||scoutReady),reentryReady=reentry.active&&reentry.eligible,
    isReady=!open&&(normalReady||reentryReady),
    style=reentryReady?(reentry.type==='STOP_RECLAIM'?'REENTRY_STOP':reentry.type==='NEW_WAVE_RESET'?'NEW_WAVE_RESET':'REENTRY_RESET'):
      emergencyReady?'EMERGENCY_SCOUT':surgeReady?'SURGE_SCOUT':coreReady?'CORE':winnerReady?'WINNER_SCOUT':hunterReady?'HUNTER':scoutReady?'EARLY_SCOUT':'WAIT',
    reason=open?'POSITION_OPEN':reentry.active&&!reentry.eligible?reentry.reason:reentryReady?reentry.reason:
      emergencyReady?'V880_EMERGENCY_SCOUT':surgeReady?'V880_SURGE_SCOUT':coreReady?'V875_CORE_EDGE':winnerReady?'V875_WINNER_SCOUT':hunterReady?'V875_HUNTER_EDGE':scoutReady?'V875_EARLY_SCOUT':
      lateCore?'WAIT_LATE_ENTRY':ev.coreRaw?'WAIT_CORE_CONFIRM':ev.winnerRaw?'WAIT_WINNER_CONFIRM':ev.hunterRaw?'WAIT_HUNTER_CONFIRM':ev.scoutRaw?'WAIT_SCOUT_CONFIRM':ev.reason,
    rec:any={symbol:c.symbol,price:m.mid,radar_score:c.radar_score,radar_lane:c.lane||'UNKNOWN',signal_score:ev.score,opportunity_score:ev.opp,opportunity_tier:ev.tier,
      capital_score:ev.capitalScore,explosion_score:ev.explosionScore,shock_up_15m_bps:m.shock_up_15m_bps,shock_age_min:m.shock_age_min,estimated_cost_bps:ev.cost,
      forecast_gross_bps:ev.gross,forecast_net_bps:ev.net,continuation_prob:ev.forecast.continuation,forecast:ev.forecast,pullback_pct:m.pullbackPct,bounce_pct:m.bouncePct,
      spread_bps:m.spreadBps,ready:isReady,entry_style:style,reentry_type:reentry.type,reentry_reason:reentry.reason,reentry_attempt:num(guard?.reentry_count),
      post_exit_low:guard?.post_exit_low??null,post_exit_high:guard?.post_exit_high??null,wave_profit_bank:guard?.wave_profit_bank??0,
      core_streak:coreStreak,winner_streak:winnerStreak,hunter_streak:hunterStreak,scout_streak:scoutStreak,surge_streak:surgeStreak,
      emergency_scout:emergencyReady,surge_scout:surgeReady,reason,gates:ev.gates,policy_version:POLICY_VERSION,engine_version:ENGINE_VERSION,risk_mode:mode.name};
  evals.push(rec);if(open)openCandidates.push({c,m,ev,rec});if(isReady)ready.push({c,m,ev,style,reentry,rec});
}

for(const x of openCandidates){
  const p=positions[x.c.symbol];if(!p)continue;
  const sig=monsterScaleSignal(p,x.m,x.ev,mode);if(!sig)continue;
  let pulse:EntryPulse;
  try{pulse=await loadEntryPulse(x.c.symbol);}catch(e){x.rec.scale_reason='WAIT_SCALE_PULSE_UNAVAILABLE';continue;}
  x.rec.scale_pulse=pulse;
  if(!monsterScalePulseOk(pulse,sig.stage)){x.rec.scale_reason='WAIT_MONSTER_SCALE_PULSE';continue;}

  const liveEq=Math.max(0,equity(cash,positions,markets)),currentGross=p.qty*p.entry,
    openGross=Object.values(positions).reduce((s,z)=>s+z.qty*z.entry,0),remaining=Math.max(0,liveEq*MAX_TOTAL_GROSS_PCT-openGross),
    targetGross=liveEq*sig.targetFraction,desired=Math.max(0,targetGross-currentGross),
    slip=Math.max(2,x.m.spreadBps/2),scaleEntry=x.m.ask*(1+slip/10000),
    allowedLoss=Math.max(.08,num(p.initial_risk_usd,liveEq*.001)),
    stopBufferBps=Math.max(10,x.m.spreadBps*1.5+4),stopCap=x.m.bid*(1-stopBufferBps/10000),
    exitFactor=1-(FEE_BPS+slip+2)/10000,feeFactor=1+FEE_BPS/10000,
    rhs=allowedLoss-p.cost_basis+p.qty*stopCap*exitFactor,
    coeff=feeFactor-(stopCap/scaleEntry)*exitFactor,
    maxByRisk=coeff>0?Math.max(0,rhs/coeff):0,
    addGross=Math.min(desired,remaining,cash*.90,maxByRisk);

  if(addGross<MIN_SHADOW_NOTIONAL){x.rec.scale_reason='WAIT_MONSTER_SCALE_RISK_BUFFER';x.rec.scale_max_by_risk=maxByRisk;continue;}

  const addQty=addGross/scaleEntry,addFee=addGross*FEE_BPS/10000,newQty=p.qty+addQty,newCost=p.cost_basis+addGross+addFee,
    newAvg=(p.qty*p.entry+addQty*scaleEntry)/newQty,
    safeStop=(newCost-allowedLoss)/Math.max(1e-12,newQty*exitFactor),
    newStop=Math.max(p.stop,safeStop),
    targetGap=Math.max(x.m.atr*(sig.stage===1?1.25:1.40),newAvg*(Math.max(x.ev.cost*1.65,35)/10000)),
    oldQty=p.qty,oldEntry=p.entry,oldCost=p.cost_basis,oldStop=p.stop;

  if(newStop>=x.m.bid*(1-8/10000)){x.rec.scale_reason='WAIT_MONSTER_SCALE_STOP_TOO_CLOSE';continue;}

  p.qty=newQty;p.cost_basis=newCost;p.entry=newAvg;p.stop=newStop;p.target=Math.max(p.target,newAvg+targetGap);
  p.trail=p.trail?Math.max(p.trail,newStop):p.trail;p.max_price=Math.max(num(p.max_price,newAvg),x.m.bid);
  p.scale_stage=sig.stage;p.scale_count=num(p.scale_count)+1;p.last_scale_at=now;p.last_scale_price=scaleEntry;
  p.scale_total_added=num(p.scale_total_added)+addGross;p.scale_peak_explosion=Math.max(num(p.scale_peak_explosion),x.ev.explosionScore);
  p.scale_peak_continuation=Math.max(num(p.scale_peak_continuation),x.ev.forecast.continuation);p.scale_target_fraction=sig.targetFraction;
  p.capital_fraction=(p.qty*p.entry)/Math.max(1e-12,liveEq);p.winner_expansion=true;cash-=addGross+addFee;
  x.rec.scale_reason=`V881_MONSTER_SCALE_${sig.stage}`;x.rec.scale_stage=sig.stage;x.rec.scale_added=addGross;x.rec.scale_total_fraction=p.capital_fraction;

  const row={observed_at:iso(),symbol:x.c.symbol,action:'BUY',price:scaleEntry,qty:addQty,notional:addGross,pnl:null,reason:`V881_MONSTER_SCALE_${sig.stage}`,metadata:{
    policy_version:POLICY_VERSION,engine_version:ENGINE_VERSION,risk_engine:RISK_ENGINE,scale_add:true,scale_stage:sig.stage,scale_count:p.scale_count,
    target_total_fraction:sig.targetFraction,capital_fraction_after:p.capital_fraction,previous_entry:oldEntry,average_entry_after:newAvg,previous_qty:oldQty,qty_after:newQty,
    previous_cost_basis:oldCost,cost_basis_after:newCost,allowed_loss_usd:allowedLoss,risk_neutral_stop:newStop,previous_stop:oldStop,
    winner_bps:sig.winnerBps,step_bps:sig.stepBps,signal_score:x.ev.score,opportunity_score:x.ev.opp,explosion_score:x.ev.explosionScore,
    continuation_prob:x.ev.forecast.continuation,forecast_net_bps:x.ev.net,forecast:x.ev.forecast,entry_pulse:pulse,estimated_cost_bps:x.ev.cost,
    shadow_only:true,live_execution:false}};
  await insertEvent(sql,row);actions.push(row);
}

ready.sort((a,b)=>(b.style==='EMERGENCY_SCOUT'?1:0)-(a.style==='EMERGENCY_SCOUT'?1:0)||b.ev.capitalScore-a.ev.capitalScore||b.ev.opp-a.ev.opp||b.ev.net-a.ev.net);
for(const x of ready){
  if(Object.keys(positions).length>=MAX_POSITIONS)break;
  if(mode.frozen&&x.style!=='EMERGENCY_SCOUT')continue;

  let profile=capitalProfile(mode,x.ev,x.style);
  if(x.reentry.type==='STOP_RECLAIM'||x.reentry.type==='NEW_WAVE_RESET')profile={gross:profile.gross*.55,risk:profile.risk*.55};
  else if(x.reentry.type==='PROFIT_RESET')profile={gross:profile.gross*.45,risk:profile.risk*.45};

  const liveEq=Math.max(0,equity(cash,positions,markets)),openGross=Object.values(positions).reduce((s,p)=>s+p.qty*p.entry,0),
    remaining=Math.max(0,liveEq*MAX_TOTAL_GROSS_PCT-openGross);
  let riskUsd=liveEq*profile.risk;
  const profitBank=num(x.reentry.guard?.wave_profit_bank);
  if(x.reentry.type==='PROFIT_RESET')riskUsd=Math.min(riskUsd,profitBank*.35);

  const baseStopDist=Math.min(x.m.ask*.020,Math.max(x.m.atr*1.35,x.m.ask*.0070)),
    priorLow=num(x.reentry.guard?.post_exit_low),structuralDist=priorLow>0?Math.max(0,x.m.ask-priorLow*(1-.0015)):0,
    resetEntry=x.reentry.type==='STOP_RECLAIM'||x.reentry.type==='PROFIT_RESET'||x.reentry.type==='NEW_WAVE_RESET',
    stopDist=resetEntry?Math.min(x.m.ask*.025,Math.max(baseStopDist,structuralDist)):baseStopDist,
    riskPerDollar=stopDist/x.m.ask+x.ev.cost/10000,
    budget=Math.min(cash*.94,liveEq*profile.gross,remaining,riskUsd/Math.max(.0001,riskPerDollar));

  if(budget<MIN_SHADOW_NOTIONAL){
    x.rec.ready=false;x.rec.reason=x.reentry.type==='PROFIT_RESET'?'WAIT_PROFIT_PROTECTION_BUDGET':'WAIT_RISK_BUDGET';continue;
  }

  let pulse:EntryPulse;
  try{pulse=await loadEntryPulse(x.c.symbol);}catch(e){
    x.rec.ready=false;x.rec.reason='WAIT_ENTRY_PULSE_UNAVAILABLE';x.rec.entry_pulse_error=e instanceof Error?e.message:String(e);continue;
  }
  x.rec.entry_pulse=pulse;
  const grossFrac=liveEq>0?budget/liveEq:0;
  if(!pulseOk(pulse,x.reentry.type,grossFrac,x.style)){
    x.rec.ready=false;x.rec.reason=grossFrac>=.08?'WAIT_ENTRY_PULSE_LARGE_SIZE':'WAIT_ENTRY_PULSE';continue;
  }

  const slip=Math.max(2,x.m.spreadBps/2),entry=x.m.ask*(1+slip/10000),gross=budget/(1+FEE_BPS/10000),qty=gross/entry,openFee=gross*FEE_BPS/10000,costBasis=gross+openFee;
  const mult=x.ev.tier==='A+'?1.45:x.ev.tier==='A'?1.35:x.ev.tier==='B+'?1.25:1.20;
  let harvestDist=Math.max(x.m.atr*mult,entry*(Math.max(x.ev.cost*1.95,18)/10000),stopDist*(x.ev.tier==='A+'?1.35:1.22));harvestDist=Math.min(harvestDist,entry*.035);

  const isReentry=['STOP_RECLAIM','PROFIT_RESET','NEW_WAVE_RESET'].includes(x.reentry.type),
    attempt=isReentry?num(x.reentry.guard?.reentry_count)+1:0,
    bank=x.reentry.type==='NEW_WAVE_RESET'?0:isReentry?num(x.reentry.guard?.wave_profit_bank):0,
    waveStarted=x.reentry.type==='NEW_WAVE_RESET'?now:isReentry?num(x.reentry.guard?.wave_started_at,now):now,
    entryReason=x.style==='EMERGENCY_SCOUT'?'V880_EMERGENCY_SCOUT':x.style==='SURGE_SCOUT'?'V880_SURGE_SCOUT':
      x.reentry.type==='STOP_RECLAIM'?'V880_STOP_RECLAIM':x.reentry.type==='PROFIT_RESET'?'V880_PROFIT_RESET':x.reentry.type==='NEW_WAVE_RESET'?'V880_NEW_WAVE_RESET':
      x.style==='WINNER_SCOUT'?'V873_WINNER_SCOUT':x.style==='HUNTER'?'V873_HUNTER_EDGE':x.style==='RECOVERY_SCOUT'?'V873_RECOVERY_SCOUT':'V873_CORE_EDGE';

  const p:Position={symbol:x.c.symbol,entry,qty,opened_at:iso(),stop:entry-stopDist,target:entry+harvestDist,trail:null,max_price:entry,cost_basis:costBasis,
    radar_score:x.c.radar_score,estimated_cost_bps:x.ev.cost,policy_version:POLICY_VERSION,entry_reason:entryReason,entry_style:x.style,opportunity_tier:x.ev.tier,
    entry_signal:x.ev.score,entry_opportunity:x.ev.opp,entry_forecast_net_bps:x.ev.net,entry_continuation:x.ev.forecast.continuation,capital_fraction:gross/liveEq,
    harvest_armed:false,harvest_at:null,runner_mode:false,runner_trail:null,last_continuation:x.ev.forecast.continuation,initial_risk_usd:gross*riskPerDollar,
    winner_expansion:false,profit_lock_active:false,profit_lock_ratio:0,profit_lock_max_net_pnl:0,profit_lock_armed_at:null,
    last_expected_15m_bps:x.ev.forecast.expected_15m_bps,last_expected_30m_bps:x.ev.forecast.expected_30m_bps,entry_explosion_score:x.ev.explosionScore,
    reentry_type:x.reentry.type,reentry_attempt:attempt,wave_profit_bank:bank,wave_started_at:waveStarted,
    prior_exit_reason:isReentry?String(x.reentry.guard?.reason||''):undefined,prior_exit_price:isReentry?num(x.reentry.guard?.exit_price):undefined};
  positions[x.c.symbol]=p;cash-=costBasis;

  const row={observed_at:iso(),symbol:x.c.symbol,action:'BUY',price:entry,qty,notional:gross,pnl:null,reason:p.entry_reason,metadata:{
    policy_version:POLICY_VERSION,engine_version:ENGINE_VERSION,risk_engine:RISK_ENGINE,radar_lane:x.c.lane||'UNKNOWN',entry_style:x.style,opportunity_tier:x.ev.tier,
    risk_mode:mode.name,capital_score:x.ev.capitalScore,explosion_score:x.ev.explosionScore,conviction_sizing:true,capital_fraction:p.capital_fraction,
    gross_cap_pct:profile.gross,risk_cap_pct:profile.risk,initial_risk_usd:p.initial_risk_usd,signal_score:x.ev.score,opportunity_score:x.ev.opp,
    forecast_net_bps:x.ev.net,forecast:x.ev.forecast,estimated_cost_bps:x.ev.cost,harvest_trigger:p.target,stop:p.stop,radar_source:radar.source,
    entry_guard_v872:true,profit_ratchet:true,entry_pulse:pulse,entry_pulse_size_aware:true,reentry_type:p.reentry_type,reentry_attempt:p.reentry_attempt,wave_profit_bank:p.wave_profit_bank,
    prior_exit_reason:p.prior_exit_reason??null,prior_exit_price:p.prior_exit_price??null,shadow_only:true,live_execution:false}};
  await insertEvent(sql,row);actions.push(row);
}
const memMap=new Map<string,MemoryItem>();for(const m of radar.prevMemory){if(num(m.until)>now)memMap.set(String(m.symbol),m);}for(const e of evals){if(e.ready||e.signal_score>=.60||e.continuation_prob>=.60||e.radar_score>=.72){const strength=e.signal_score*.34+e.opportunity_score*.30+e.continuation_prob*.24+e.radar_score*.12;memMap.set(e.symbol,{symbol:e.symbol,until:now+MEMORY_TTL_MS,last_seen:now,score:strength,opp:e.opportunity_score,cont:e.continuation_prob,lane:e.radar_lane});}}const radarMemory=[...memMap.values()].sort((a,b)=>(b.score+b.opp*.3+b.cont*.2)-(a.score+a.opp*.3+a.cont*.2)).slice(0,MEMORY_MAX),learning=await learnMissed(sql,state,now),minute=new Date(Math.floor(now/60000)*60000).toISOString();
if(state.last_eval_minute!==minute){
  if(evals.length){for(const e of evals){try{await sql`insert into public.brian_dip_multiasset_evaluations (engine_id,observed_at,symbol,price,radar_score,signal_score,action,reason,metadata) values (${ENGINE_ID},now(),${e.symbol},${e.price},${e.radar_score},${e.signal_score},${e.ready?'READY':'WAIT'},${e.reason},${sql.json(e)})`;}catch{}}}
  try{
    await sql`insert into public.brian_dip_light_radar_history (engine_id,observed_minute,universe_count,snapshot) values (${ENGINE_ID},${new Date(minute)},${radar.universeCount},${sql.json((radar as any).lightSnapshot||[])}) on conflict (engine_id,observed_minute) do update set universe_count=excluded.universe_count,snapshot=excluded.snapshot`;
    if(new Date(now).getUTCMinutes()%15===0)await sql`delete from public.brian_dip_light_radar_history where observed_minute<now()-interval '7 days'`;
  }catch{}
}const finalEq=equity(cash,positions,markets),scan={status:'RUNNING',policy_version:POLICY_VERSION,engine_version:ENGINE_VERSION,risk_engine:RISK_ENGINE,risk_mode:mode.name,risk_reason:mode.reason,drawdown_pct:mode.dd,win_rate:mode.wr,loss_streak:lossStreak,last_loss_at:lastLossAt,radar_source:radar.source,radar_observed_at:radar.observedAt,radar_age_seconds:radar.ageSec,radar_soft_stale:radar.stale,universe_watch_count:radar.universeCount,deep_scan_count:evals.length,radar_lane_counts:radar.laneCounts,deep_scan_selection:(radar as any).selectionPlan,radar_memory_count:radarMemory.length,radar_memory:radarMemory,universe_prices:radar.priceSnapshot,position_guardian:(state.last_scan as any)?.position_guardian??null,core_streaks:coreStreaks,winner_streaks:winnerStreaks,hunter_streaks:hunterStreaks,scout_streaks:scoutStreaks,surge_streaks:surgeStreaks,learning_at:learning.at,learning_summary:learning.summary,hardening:{all_market_light_watch:true,multi_radar_universe:true,radar_memory_60m:true,deep_scan_32:true,opportunity_interrupt_slots:true,rotating_explorer_slots:true,two_cycle_core_confirm:true,forecast_consensus_gate:true,forecast_raw_consistency:true,multi_horizon_regime_guard:true,shock_memory_15m:true,reentry_chase_guard:true,reentry_reset_state_machine:true,stop_reclaim_reentry:true,profit_reset_reentry:true,new_wave_reset_reentry:true,entry_pulse_15s:true,size_aware_entry_pulse:true,wave_profit_protection:true,frozen_emergency_scout:true,radar_surge_override:true,light_radar_history_7d:true,explosion_score:true,conviction_sizing:true,long_horizon_extension_veto:true,per_symbol_circuit_breaker:true,impulse_chase_veto:true,earned_scaling:true,monster_conviction_scale_v1:true,winner_only_scaling:true,risk_neutral_pyramiding:true,max_scale_stage:3,trigger_fill_model:true,winner_expansion:true,profit_ratchet:true,mfe_profit_lock:true,forecast_adaptive_trail:true,no_retroactive_fill:true,near_miss_three_cycle:true,adaptive_profit_cooldown:true,thesis_break:true},scanned:evals,ranked_ready:ready.slice(0,8).map(x=>({symbol:x.c.symbol,radar_lane:x.c.lane||'UNKNOWN',entry_style:x.style,opportunity_tier:x.ev.tier,capital_score:x.ev.capitalScore,explosion_score:x.ev.explosionScore,continuation_prob:x.ev.forecast.continuation,opportunity_score:x.ev.opp,forecast_net_bps:x.ev.net,radar_score:x.c.radar_score})),actions,market_errors:marketErrors,equity:finalEq,position_count:Object.keys(positions).length,max_positions:MAX_POSITIONS,max_total_gross_pct:MAX_TOTAL_GROSS_PCT,gross_cap_pct:mode.gross,risk_cap_pct:mode.risk,heartbeat_at:iso()};await sql`update public.brian_dip_multiasset_state set cash=${cash},realized_pnl=${realized},trade_count=${trades},win_count=${wins},loss_count=${losses},positions=${sql.json(positions)},cooldowns=${sql.json(cooldowns)},last_eval_minute=${new Date(minute)},last_scan=${sql.json(scan)},updated_at=now(),enabled=true,shadow_only=true,live_execution=false where engine_id=${ENGINE_ID}`;return{status:'RUNNING',engine_id:ENGINE_ID,engine_version:ENGINE_VERSION,policy_version:POLICY_VERSION,risk_engine:RISK_ENGINE,risk_mode:mode.name,risk_reason:mode.reason,radar_source:radar.source,universe_watch_count:radar.universeCount,deep_scan_count:evals.length,radar_memory_count:radarMemory.length,equity:finalEq,cash,realized_pnl:realized,trade_count:trades,positions:Object.keys(positions),learning_summary:learning.summary,actions,shadow_only:true,live_execution:false};}
Deno.serve(async(req:Request)=>{if(req.method!=='POST')return new Response('method',{status:405});const sql=db();let lease:any=null;try{await requireCron(sql,req);lease=await acquireLease(sql);if(!lease)return Response.json({status:'WAIT_LEASE',engine_id:ENGINE_ID,engine_version:ENGINE_VERSION,policy_version:POLICY_VERSION,shadow_only:true,live_execution:false});return Response.json(await run(sql),{headers:{'cache-control':'no-store'}});}catch(e){const message=e instanceof Error?e.message:String(e);try{const rows=await sql`select last_scan from public.brian_dip_multiasset_state where engine_id=${ENGINE_ID} limit 1`;const scan={...((rows[0]?.last_scan||{}) as J),status:'FAILED_CLOSED',error:message,error_at:iso(),policy_version:POLICY_VERSION,engine_version:ENGINE_VERSION,risk_engine:RISK_ENGINE};await sql`update public.brian_dip_multiasset_state set last_scan=${sql.json(scan)},updated_at=now() where engine_id=${ENGINE_ID}`;}catch{}return Response.json({status:'FAILED_CLOSED',engine_id:ENGINE_ID,engine_version:ENGINE_VERSION,policy_version:POLICY_VERSION,error:message,shadow_only:true,live_execution:false},{status:message.includes('UNAUTHORIZED')?401:500});}finally{await releaseLease(sql,lease);try{await sql.end({timeout:1});}catch{}}});