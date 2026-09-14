import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });
const ENGINE_ID = "dip-multiasset-v1";
const RUN_HOURS = 8;
const MAX_POSITIONS = 3;
const MAX_SCAN = 7;
const FEE_BPS = 10;
const HOSTS = ["https://api.binance.com", "https://api1.binance.com", "https://api2.binance.com"];
const EXCLUDED = new Set(["ETHUSDT","USDTUSDT","USDCUSDT","FDUSDUSDT","TUSDUSDT","DAIUSDT","BUSDUSDT","EURUSDT","TRYUSDT","USD1USDT"]);

type J = Record<string, unknown>;
type Position = {
  symbol: string; entry: number; qty: number; opened_at: string; stop: number; target: number;
  trail: number | null; max_price: number; open_fee: number; cost_basis: number; atr: number;
  radar_score: number; entry_spread_bps: number; entry_reason: string;
};
type State = {
  engine_id: string; started_at: string; run_until: string; starting_equity: number; cash: number;
  realized_pnl: number; trade_count: number; win_count: number; loss_count: number;
  positions: Record<string, Position>; cooldowns: Record<string, number>; last_scan: J;
  last_eval_minute: string | null; updated_at: string; enabled: boolean; shadow_only: boolean; live_execution: boolean;
};
type Candidate = {
  symbol: string; base_asset: string; radar_score: number; price_change_pct: number; range_pct: number;
  spread_bps: number; quote_volume: number; trades_24h: number; momentum_score: number; volatility_score: number;
};
type Bar = { o:number; h:number; l:number; c:number; v:number; openTime:number; closeTime:number };
type Market = {
  symbol:string; bid:number; ask:number; mid:number; spreadBps:number; bars:Bar[]; atr:number; atrPct:number;
  ema8:number; ema21:number; recentLow:number; recentHigh:number; pullbackPct:number; bouncePct:number;
  lastClose:number; prevClose:number; recovery:boolean; trendOk:boolean;
};

function num(v: unknown, fallback = 0): number { const n = Number(v); return Number.isFinite(n) ? n : fallback; }
function clip(v:number, lo=0, hi=1):number { return Math.max(lo, Math.min(hi, v)); }
async function sha256Hex(value:string):Promise<string>{
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)));
  return [...digest].map((b)=>b.toString(16).padStart(2,"0")).join("");
}
function same(a:string,b:string):boolean { if(a.length!==b.length)return false; let d=0; for(let i=0;i<a.length;i++)d|=a.charCodeAt(i)^b.charCodeAt(i); return d===0; }
async function requireCron(req:Request){
  const supplied=(req.headers.get("x-brian-cron-key")||"").trim(); if(!supplied) throw Error("UNAUTHORIZED_CRON");
  const q=await db.from("brian_dashboard_auth").select("cron_key_sha256").eq("auth_id","control-v3").single();
  if(q.error||!q.data)throw Error("CRON_AUTH_UNAVAILABLE");
  if(!same(await sha256Hex(supplied), String(q.data.cron_key_sha256||"")))throw Error("UNAUTHORIZED_CRON");
}
async function marketJson(path:string):Promise<unknown>{
  let last="MARKET_UNAVAILABLE";
  for(const host of HOSTS){
    try{
      const r=await fetch(host+path,{headers:{accept:"application/json","user-agent":"Brian-DIP-MultiAsset-Shadow/1.0"},signal:AbortSignal.timeout(5000)});
      if(!r.ok){last=`HTTP_${r.status}`; if(r.status===418||r.status===429)break; continue;}
      return await r.json();
    }catch(e){last=e instanceof Error?e.message:String(e);}
  }
  throw Error(`BINANCE_SPOT:${last}`);
}
function ema(values:number[], period:number):number{
  const k=2/(period+1); let out=values[0]??0; for(let i=1;i<values.length;i++)out=values[i]*k+out*(1-k); return out;
}
function atr(bars:Bar[], period=14):number{
  if(bars.length<period+2)return 0; const xs:number[]=[];
  for(let i=1;i<bars.length;i++){const b=bars[i],p=bars[i-1].c;xs.push(Math.max(b.h-b.l,Math.abs(b.h-p),Math.abs(b.l-p)));}
  return xs.slice(-period).reduce((a,b)=>a+b,0)/period;
}
async function loadMarket(symbol:string):Promise<Market>{
  const [rawBars, rawBook]=await Promise.all([
    marketJson(`/api/v3/klines?symbol=${encodeURIComponent(symbol)}&interval=1m&limit=70`),
    marketJson(`/api/v3/ticker/bookTicker?symbol=${encodeURIComponent(symbol)}`),
  ]);
  if(!Array.isArray(rawBars)||rawBars.length<35)throw Error("INSUFFICIENT_BARS");
  const now=Date.now();
  const parsed:Bar[]=rawBars.map((x)=>{if(!Array.isArray(x)||x.length<7)throw Error("BAD_BAR");return{o:num(x[1]),h:num(x[2]),l:num(x[3]),c:num(x[4]),v:num(x[5]),openTime:num(x[0]),closeTime:num(x[6])};});
  const bars=parsed.filter((b)=>b.closeTime<now-250).slice(-60); if(bars.length<30)throw Error("INSUFFICIENT_CLOSED_BARS");
  const book=rawBook as J, bid=num(book.bidPrice), ask=num(book.askPrice); if(!(bid>0&&ask>bid))throw Error("BAD_BOOK");
  const mid=(bid+ask)/2, spreadBps=(ask-bid)/mid*10000; if(!Number.isFinite(spreadBps)||spreadBps>40)throw Error("WIDE_BOOK");
  const closes=bars.map((b)=>b.c), a=atr(bars,14); if(!(a>0))throw Error("BAD_ATR");
  const recent20=bars.slice(-20), recent12=bars.slice(-12); const recentHigh=Math.max(...recent20.map(b=>b.h)),recentLow=Math.min(...recent12.map(b=>b.l));
  const lastClose=bars.at(-1)!.c,prevClose=bars.at(-2)!.c,ema8=ema(closes.slice(-35),8),ema21=ema(closes.slice(-45),21);
  const pullbackPct=(recentHigh-mid)/recentHigh*100,bouncePct=(mid-recentLow)/recentLow*100,atrPct=a/mid*100;
  const recovery=mid>=lastClose*.9985 && lastClose>=prevClose*.995;
  const trendOk=ema8>=ema21*.985;
  return{symbol,bid,ask,mid,spreadBps,bars,atr:a,atrPct,ema8,ema21,recentLow,recentHigh,pullbackPct,bouncePct,lastClose,prevClose,recovery,trendOk};
}
async function acquireLease():Promise<{owner:string;generation:number}|null>{
  const owner=crypto.randomUUID();
  const q=await db.rpc("brian_dip_v84_acquire_lease",{p_lease_key:"brian-dip-multiasset-worker-v1",p_owner_id:owner,p_lease_seconds:45});
  if(q.error)throw Error("LEASE:"+q.error.message); const row=Array.isArray(q.data)?q.data[0]:q.data; if(!row?.acquired)return null;
  return{owner,generation:Number(row.lease_generation)};
}
async function releaseLease(lease:{owner:string;generation:number}|null){
  if(!lease)return; try{await db.rpc("brian_dip_v84_release_lease",{p_lease_key:"brian-dip-multiasset-worker-v1",p_owner_id:lease.owner,p_lease_generation:lease.generation});}catch{}
}
async function loadOrCreateState():Promise<State>{
  const q=await db.from("brian_dip_multiasset_state").select("*").eq("engine_id",ENGINE_ID).maybeSingle(); if(q.error)throw q.error;
  if(q.data)return q.data as State;
  const now=new Date(),runUntil=new Date(now.getTime()+RUN_HOURS*3600_000);
  const row={engine_id:ENGINE_ID,started_at:now.toISOString(),run_until:runUntil.toISOString(),starting_equity:1000,cash:1000,realized_pnl:0,trade_count:0,win_count:0,loss_count:0,positions:{},cooldowns:{},last_scan:{status:"BOOT"},last_eval_minute:null,updated_at:now.toISOString(),enabled:true,shadow_only:true,live_execution:false};
  const ins=await db.from("brian_dip_multiasset_state").insert(row).select("*").single(); if(ins.error)throw ins.error; return ins.data as State;
}
async function latestCandidates():Promise<{observedAt:string;rows:Candidate[]}> {
  const q=await db.from("brian_universe_snapshots").select("observed_at,candidates").order("observed_at",{ascending:false}).limit(1).maybeSingle(); if(q.error||!q.data)throw Error("RADAR_UNAVAILABLE");
  const observedAt=String(q.data.observed_at),age=Date.now()-Date.parse(observedAt); if(!Number.isFinite(age)||age>12*60_000)throw Error("RADAR_STALE");
  const payload=(q.data.candidates??{}) as J,raw=Array.isArray(payload.candidates)?payload.candidates:[];
  const rows:Candidate[]=raw.map((v)=>{const r=(v??{}) as J;return{symbol:String(r.symbol||"").toUpperCase(),base_asset:String(r.base_asset||"").toUpperCase(),radar_score:num(r.radar_score),price_change_pct:num(r.price_change_pct),range_pct:num(r.range_pct),spread_bps:num(r.spread_bps,999),quote_volume:num(r.quote_volume),trades_24h:num(r.trades_24h),momentum_score:num(r.momentum_score),volatility_score:num(r.volatility_score)}})
    .filter((r)=>/^[A-Z0-9]{2,20}USDT$/.test(r.symbol)&&!EXCLUDED.has(r.symbol)&&r.radar_score>=.66&&r.quote_volume>=5_000_000&&r.trades_24h>=25_000&&r.spread_bps<=35&&Math.abs(r.price_change_pct)<=190)
    .sort((a,b)=>b.radar_score-a.radar_score).slice(0,MAX_SCAN);
  return{observedAt,rows};
}
function equity(state:State,markets:Map<string,Market>):number{
  let e=state.cash; for(const p of Object.values(state.positions||{})){const m=markets.get(p.symbol);e+=p.qty*(m?.bid??p.entry);} return e;
}
function evaluateEntry(c:Candidate,m:Market){
  const huge=Math.abs(c.price_change_pct)>=80;
  const requiredPull=Math.max(huge?1.35:.55,m.atrPct*(huge?.9:.55));
  const requiredBounce=Math.max(.18,m.atrPct*.14);
  const costBps=2*FEE_BPS+m.spreadBps+Math.max(4,m.spreadBps);
  const score=clip(c.radar_score*.42 + clip(m.pullbackPct/Math.max(requiredPull*1.8,.5))*.20 + clip(m.bouncePct/Math.max(requiredBounce*3,.2))*.16 + (m.recovery?.11:0) + (m.trendOk?.11:0));
  const gates={radar:c.radar_score>=.72,spread:m.spreadBps<=28,pullback:m.pullbackPct>=requiredPull,bounce:m.bouncePct>=requiredBounce,recovery:m.recovery,trend:m.trendOk,not_chasing:m.mid<=m.recentHigh*(1-Math.max(.0015,m.atrPct*.10/100)),economics:Math.max(1.2,m.atrPct*1.6)>=costBps/100};
  const ok=score>=.66&&Object.values(gates).every(Boolean);
  const failed=Object.entries(gates).filter(([,v])=>!v).map(([k])=>k);
  return{ok,score,costBps,requiredPull,requiredBounce,gates,reason:ok?"DIP_RECLAIM_READY":`WAIT_${failed.join("+")||"SCORE"}`};
}
async function insertEvent(row:J){const q=await db.from("brian_dip_multiasset_events").insert(row);if(q.error)throw q.error;}

async function run():Promise<J>{
  const state=await loadOrCreateState(),now=Date.now();
  if(!state.enabled||now>=Date.parse(state.run_until)){
    await db.from("brian_dip_multiasset_state").update({enabled:false,updated_at:new Date().toISOString(),last_scan:{status:"WINDOW_COMPLETE",run_until:state.run_until}}).eq("engine_id",ENGINE_ID);
    return{status:"WINDOW_COMPLETE",engine_id:ENGINE_ID,run_until:state.run_until,shadow_only:true,live_execution:false};
  }
  const radar=await latestCandidates();
  const symbols=[...new Set([...Object.keys(state.positions||{}),...radar.rows.map(r=>r.symbol)])].slice(0,10);
  const marketPairs=await Promise.all(symbols.map(async(s)=>{try{return[s,await loadMarket(s)] as const;}catch(e){return[s,{error:e instanceof Error?e.message:String(e)}] as const;}}));
  const markets=new Map<string,Market>(); const errors:J={};
  for(const [s,v] of marketPairs){if("error" in v)errors[s]=v.error;else markets.set(s,v);}
  const positions={...(state.positions||{})},cooldowns={...(state.cooldowns||{})}; let cash=num(state.cash),realized=num(state.realized_pnl),tradeCount=num(state.trade_count),wins=num(state.win_count),losses=num(state.loss_count);
  const actions:J[]=[];

  for(const [symbol,p] of Object.entries(positions)){
    const m=markets.get(symbol); if(!m)continue;
    const heldMs=now-Date.parse(p.opened_at),gainPct=(m.bid-p.entry)/p.entry*100; p.max_price=Math.max(num(p.max_price,p.entry),m.bid);
    if(gainPct>=Math.max(.9,m.atrPct*.75)){const candidate=m.bid-Math.max(m.atr*.75,p.entry*.006);p.trail=Math.max(num(p.trail,0),candidate);}
    const momentumBreak=heldMs>5*60_000&&m.bid<m.ema8*.994&&m.lastClose<m.prevClose*.997;
    let reason=""; if(m.bid<=p.stop)reason="STOP"; else if(p.trail&&m.bid<=p.trail)reason="TRAIL"; else if(m.bid>=p.target)reason="TARGET"; else if(momentumBreak)reason="MOMENTUM_BREAK"; else if(heldMs>=35*60_000)reason="MAX_HOLD";
    if(!reason)continue;
    const slipBps=Math.max(2,m.spreadBps/2),exitPrice=m.bid*(1-slipBps/10000),gross=p.qty*exitPrice,exitFee=gross*FEE_BPS/10000,pnl=gross-exitFee-p.cost_basis;
    cash+=gross-exitFee;realized+=pnl;tradeCount++;if(pnl>=0)wins++;else losses++;delete positions[symbol];cooldowns[symbol]=now+8*60_000;
    const ev={engine_id:ENGINE_ID,observed_at:new Date().toISOString(),symbol,action:"SELL",price:exitPrice,qty:p.qty,notional:gross,pnl,reason,metadata:{entry:p.entry,opened_at:p.opened_at,hold_seconds:Math.round(heldMs/1000),target:p.target,stop:p.stop,trail:p.trail,max_price:p.max_price,spread_bps:m.spreadBps,shadow_only:true,live_execution:false}};await insertEvent(ev);actions.push(ev);
  }

  const evals:J[]=[];
  const candidatesWithMarkets=radar.rows.map((c)=>({c,m:markets.get(c.symbol)})).filter((x):x is {c:Candidate;m:Market}=>!!x.m);
  const currentEquity=equity({...state,cash,positions} as State,markets);
  for(const {c,m} of candidatesWithMarkets){
    const ev=evaluateEntry(c,m);const onCooldown=num(cooldowns[c.symbol],0)>now;const already=!!positions[c.symbol];
    evals.push({symbol:c.symbol,price:m.mid,radar_score:c.radar_score,change_pct:c.price_change_pct,range_pct:c.range_pct,spread_bps:m.spreadBps,atr_pct:m.atrPct,pullback_pct:m.pullbackPct,bounce_pct:m.bouncePct,signal_score:ev.score,ready:ev.ok&&!onCooldown&&!already,reason:already?"POSITION_OPEN":onCooldown?"COOLDOWN":ev.reason,gates:ev.gates});
    if(!ev.ok||onCooldown||already||Object.keys(positions).length>=MAX_POSITIONS)continue;
    const eq=Math.max(0,currentEquity),budget=Math.min(cash*.34,eq*.22); if(budget<20)continue;
    const slipBps=Math.max(2,m.spreadBps/2),entry=m.ask*(1+slipBps/10000),grossBudget=budget/(1+FEE_BPS/10000),qty=grossBudget/entry,openFee=grossBudget*FEE_BPS/10000,costBasis=grossBudget+openFee;
    const stopDistance=Math.min(entry*.045,Math.max(m.atr*1.25,entry*.010)); let targetDistance=Math.max(m.atr*1.75,entry*.012,entry*(ev.costBps*3.0/10000)); if(targetDistance/stopDistance<1.35)targetDistance=stopDistance*1.35;
    const p:Position={symbol:c.symbol,entry,qty,opened_at:new Date().toISOString(),stop:entry-stopDistance,target:entry+targetDistance,trail:null,max_price:entry,open_fee:openFee,cost_basis:costBasis,atr:m.atr,radar_score:c.radar_score,entry_spread_bps:m.spreadBps,entry_reason:ev.reason};
    positions[c.symbol]=p;cash-=costBasis;
    const action={engine_id:ENGINE_ID,observed_at:new Date().toISOString(),symbol:c.symbol,action:"BUY",price:entry,qty,notional:grossBudget,pnl:null,reason:ev.reason,metadata:{signal_score:ev.score,radar_score:c.radar_score,change_pct:c.price_change_pct,range_pct:c.range_pct,atr_pct:m.atrPct,pullback_pct:m.pullbackPct,bounce_pct:m.bouncePct,spread_bps:m.spreadBps,target:p.target,stop:p.stop,fee_bps:FEE_BPS,slippage_bps:slipBps,shadow_only:true,live_execution:false}};await insertEvent(action);actions.push(action);
  }

  const minute=new Date(Math.floor(now/60000)*60000).toISOString();
  if(state.last_eval_minute!==minute&&evals.length){
    const rows=evals.map((e)=>({engine_id:ENGINE_ID,observed_at:new Date().toISOString(),symbol:e.symbol,price:e.price,radar_score:e.radar_score,signal_score:e.signal_score,action:e.ready?"READY":"WAIT",reason:e.reason,metadata:e}));
    const q=await db.from("brian_dip_multiasset_evaluations").insert(rows);if(q.error)throw q.error;
  }
  const finalState={cash,realized_pnl:realized,trade_count:tradeCount,win_count:wins,loss_count:losses,positions,cooldowns,last_eval_minute:minute,last_scan:{status:"RUNNING",radar_observed_at:radar.observedAt,scanned:evals.slice(0,MAX_SCAN),actions,market_errors:errors,equity:equity({...state,cash,positions} as State,markets),position_count:Object.keys(positions).length},updated_at:new Date().toISOString(),enabled:true,shadow_only:true,live_execution:false};
  const uq=await db.from("brian_dip_multiasset_state").update(finalState).eq("engine_id",ENGINE_ID);if(uq.error)throw uq.error;
  return{status:"RUNNING",engine_id:ENGINE_ID,run_until:state.run_until,equity:(finalState.last_scan as J).equity,cash,realized_pnl:realized,trade_count:tradeCount,positions:Object.keys(positions),scanned:evals.slice(0,MAX_SCAN),actions,shadow_only:true,live_execution:false};
}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return new Response("method",{status:405});
  let lease:{owner:string;generation:number}|null=null;
  try{
    await requireCron(req);lease=await acquireLease();if(!lease)return Response.json({status:"WAIT_LEASE",engine_id:ENGINE_ID,shadow_only:true,live_execution:false});
    const out=await run();return Response.json(out,{headers:{"cache-control":"no-store"}});
  }catch(e){const message=e instanceof Error?e.message:String(e);console.error("dip-multiasset",message);return Response.json({status:"FAILED_CLOSED",engine_id:ENGINE_ID,error:message,shadow_only:true,live_execution:false},{status:message.includes("UNAUTHORIZED")?401:500,headers:{"cache-control":"no-store"}});}
  finally{await releaseLease(lease);}
});
