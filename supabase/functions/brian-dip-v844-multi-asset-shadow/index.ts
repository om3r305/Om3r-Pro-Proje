import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });

const ENGINE = "dip-v844-multi-asset-shadow-20260914.1";
const LEASE_KEY = "brian-dip-v844-multi-asset-shadow";
const MAX_POSITIONS = 3;
const FEE_BPS = 10;
const SLIPPAGE_BPS = 2;
const MAX_CANDIDATES = 8;
const MAX_HOLD_MS = 60 * 60 * 1000;
const COOLDOWN_MS = 20 * 60 * 1000;
const BINANCE_BASES = ["https://api.binance.com", "https://data-api.binance.vision"];

type J = Record<string, unknown>;
type RuntimeRow = { session_id:string; source_started_at:string; starting_equity:number|string; cash:number|string; realized_pnl:number|string; fees_paid:number|string; trade_count:number; win_count:number; loss_count:number; status:string; last_run_at:string|null; config:J };
type PositionRow = { session_id:string; symbol:string; qty:number|string; entry_price:number|string; entry_at:string; notional:number|string; entry_fee:number|string; cost_basis_total:number|string; stop_price:number|string; target_price:number|string; peak_price:number|string; mark_price:number|string|null; unrealized_pnl:number|string; fee_bps:number|string; slippage_bps:number|string; radar_score:number|string|null; entry_metrics:J };
type RadarCandidate = { symbol:string; base_asset:string; radar_score:number; price_change_pct:number; range_pct:number; spread_bps:number|null; quote_volume:number|null; trades_24h:number|null; momentum_score:number|null; volatility_score:number|null };
type Market = { symbol:string; bid:number; ask:number; mid:number; spreadBps:number; price:number; high15:number; high30:number; low5:number; low15:number; ema9:number; ema20:number; atrPct:number; volumeBoost:number; m1Pct:number; m3Pct:number; dipDepthPct:number; reclaimPct:number; rangePos:number; barsAt:string };
type Scored = { radar:RadarCandidate; market:Market; score:number; canEnter:boolean; reason:string; depthNeed:number; reclaimNeed:number; netEdgeBps:number; targetPrice:number; stopPrice:number };

function num(v: unknown, fallback = 0): number { const n = Number(v); return Number.isFinite(n) ? n : fallback; }
function clamp(v:number,lo:number,hi:number){ return Math.max(lo,Math.min(hi,v)); }
function iso(ms=Date.now()){ return new Date(ms).toISOString(); }
function safeSymbol(v:unknown){ const s=String(v??"").toUpperCase(); return /^[A-Z0-9]{2,20}USDT$/.test(s)?s:""; }
function ema(values:number[],period:number){ if(!values.length)return 0; const k=2/(period+1); let out=values[0]; for(let i=1;i<values.length;i++) out=values[i]*k+out*(1-k); return out; }
async function sha256Hex(value:string){ const d=await crypto.subtle.digest("SHA-256",new TextEncoder().encode(value)); return [...new Uint8Array(d)].map(b=>b.toString(16).padStart(2,"0")).join(""); }
function same(a:string,b:string){ if(a.length!==b.length)return false; let d=0; for(let i=0;i<a.length;i++)d|=a.charCodeAt(i)^b.charCodeAt(i); return d===0; }
async function requireAuth(req:Request,kind:"cron"|"dashboard"){
  const header=kind==="cron"?"x-brian-cron-key":"x-brian-dashboard-key";
  const supplied=(req.headers.get(header)??"").trim(); if(!supplied)throw Error(kind==="cron"?"UNAUTHORIZED_CRON":"UNAUTHORIZED_DASHBOARD");
  const q=await db.from("brian_dashboard_auth").select("cron_key_sha256,dashboard_key_sha256").eq("auth_id","control-v3").single();
  if(q.error||!q.data)throw Error("AUTH_UNAVAILABLE");
  const expected=kind==="cron"?String(q.data.cron_key_sha256??""):String(q.data.dashboard_key_sha256??"");
  if(!same(await sha256Hex(supplied),expected))throw Error(kind==="cron"?"UNAUTHORIZED_CRON":"UNAUTHORIZED_DASHBOARD");
}
async function binanceJson(path:string):Promise<unknown>{
  let last="BINANCE_UNAVAILABLE";
  for(const base of BINANCE_BASES){ try{ const r=await fetch(`${base}${path}`,{signal:AbortSignal.timeout(6500),headers:{accept:"application/json"}}); if(!r.ok){last=`HTTP_${r.status}`;continue;} return await r.json(); }catch(e){last=e instanceof Error?e.message:String(e);} }
  throw Error(last);
}
async function latestDipSession(){
  const s=await db.from("brian_dip_v84_session_events").select("session_id,requested_at,starting_equity,trade_notional,config").eq("event_kind","START").order("requested_at",{ascending:false}).order("event_id",{ascending:false}).limit(1).maybeSingle();
  if(s.error)throw s.error; if(!s.data)return null; const sid=String(s.data.session_id);
  const l=await db.from("brian_dip_v84_session_events").select("event_kind,requested_at").eq("session_id",sid).order("requested_at",{ascending:false}).order("event_id",{ascending:false}).limit(1).maybeSingle();
  if(l.error)throw l.error;
  return {sessionId:sid,startedAt:String(s.data.requested_at),startingEquity:num(s.data.starting_equity,1000),active:String(l.data?.event_kind??"PAUSE")==="START"};
}
async function ensureRuntime(session:NonNullable<Awaited<ReturnType<typeof latestDipSession>>>):Promise<RuntimeRow>{
  const q=await db.from("brian_dip_multi_runtime").select("*").eq("session_id",session.sessionId).maybeSingle(); if(q.error)throw q.error; if(q.data)return q.data as RuntimeRow;
  const row={session_id:session.sessionId,source_started_at:session.startedAt,starting_equity:session.startingEquity,cash:session.startingEquity,realized_pnl:0,fees_paid:0,trade_count:0,win_count:0,loss_count:0,status:session.active?"RUNNING":"PAUSED",config:{engine:ENGINE,shadow_only:true,live_execution:false,long_only:true,max_positions:MAX_POSITIONS,allocation_fraction:0.25,fee_bps:FEE_BPS,slippage_bps:SLIPPAGE_BPS,decision_cadence_seconds:20,source:"BINANCE_SPOT_1M_PLUS_BOOK",isolation:"DIP_ONLY"}};
  const ins=await db.from("brian_dip_multi_runtime").insert(row).select("*").single(); if(ins.error)throw ins.error; return ins.data as RuntimeRow;
}
async function latestRadar():Promise<RadarCandidate[]>{
  const q=await db.from("brian_universe_snapshots").select("observed_at,candidates").order("observed_at",{ascending:false}).limit(1).maybeSingle(); if(q.error)throw q.error;
  const payload=(q.data?.candidates??{}) as J, rows=Array.isArray(payload.candidates)?payload.candidates:[];
  return rows.map(raw=>{ const r=(raw??{}) as J, symbol=safeSymbol(r.symbol); return {symbol,base_asset:String(r.base_asset??symbol.replace(/USDT$/i,"")).toUpperCase(),radar_score:num(r.radar_score),price_change_pct:num(r.price_change_pct),range_pct:num(r.range_pct),spread_bps:Number.isFinite(Number(r.spread_bps))?num(r.spread_bps):null,quote_volume:Number.isFinite(Number(r.quote_volume))?num(r.quote_volume):null,trades_24h:Number.isFinite(Number(r.trades_24h))?num(r.trades_24h):null,momentum_score:Number.isFinite(Number(r.momentum_score))?num(r.momentum_score):null,volatility_score:Number.isFinite(Number(r.volatility_score))?num(r.volatility_score):null}; }).filter(r=>r.symbol&&r.radar_score>=0.66&&(r.quote_volume??0)>=5_000_000).sort((a,b)=>b.radar_score-a.radar_score);
}
async function marketFor(symbol:string):Promise<Market>{
  const [kr,br]=await Promise.all([binanceJson(`/api/v3/klines?symbol=${encodeURIComponent(symbol)}&interval=1m&limit=60`),binanceJson(`/api/v3/ticker/bookTicker?symbol=${encodeURIComponent(symbol)}`)]);
  const k=Array.isArray(kr)?kr:[]; if(k.length<20)throw Error("KLINES_SHORT");
  const bars=k.map(row=>{const a=Array.isArray(row)?row:[];return{o:num(a[1]),h:num(a[2]),l:num(a[3]),c:num(a[4]),v:num(a[5]),ct:num(a[6])};}).filter(b=>b.c>0&&b.h>0&&b.l>0); if(bars.length<20)throw Error("KLINES_INVALID");
  const book=(br??{}) as J,bid=num(book.bidPrice),ask=num(book.askPrice); if(!(bid>0&&ask>bid))throw Error("BOOK_INVALID");
  const mid=(bid+ask)/2,spreadBps=(ask-bid)/mid*10000,closes=bars.map(b=>b.c),last=bars.at(-1)!;
  const high15=Math.max(...bars.slice(-15).map(b=>b.h)),high30=Math.max(...bars.slice(-30).map(b=>b.h)),low5=Math.min(...bars.slice(-5).map(b=>b.l)),low15=Math.min(...bars.slice(-15).map(b=>b.l));
  const trs:number[]=[]; for(let i=Math.max(1,bars.length-14);i<bars.length;i++){const b=bars[i],pc=bars[i-1].c;trs.push(Math.max(b.h-b.l,Math.abs(b.h-pc),Math.abs(b.l-pc)));}
  const atr=trs.reduce((a,b)=>a+b,0)/Math.max(1,trs.length),atrPct=atr/last.c*100,prior=bars.slice(-11,-1).map(b=>b.v),avg=prior.reduce((a,b)=>a+b,0)/Math.max(1,prior.length),volumeBoost=avg>0?last.v/avg:1;
  return {symbol,bid,ask,mid,spreadBps,price:last.c,high15,high30,low5,low15,ema9:ema(closes.slice(-30),9),ema20:ema(closes.slice(-40),20),atrPct,volumeBoost,m1Pct:(last.c/closes.at(-2)!-1)*100,m3Pct:(last.c/closes.at(-4)!-1)*100,dipDepthPct:(high15-last.c)/high15*100,reclaimPct:(last.c-low5)/low5*100,rangePos:high15>low15?clamp((last.c-low15)/(high15-low15),0,1):0.5,barsAt:iso(last.ct)};
}
function scoreEntry(radar:RadarCandidate,m:Market):Scored{
  const entryFill=m.ask*(1+SLIPPAGE_BPS/10000),cost=m.spreadBps+FEE_BPS*2+SLIPPAGE_BPS*2,netEdgeBps=(m.high15/entryFill-1)*10000-cost;
  const depthNeed=clamp(0.35+m.atrPct*0.55,0.45,1.55),reclaimNeed=clamp(0.10+m.atrPct*0.14,0.12,0.50),targetPct=clamp(Math.max(0.85,m.atrPct*1.20),0.85,2.50),stopPct=clamp(Math.max(0.60,m.atrPct*0.95),0.60,1.90);
  const targetPrice=Math.min(m.high15*0.998,entryFill*(1+targetPct/100)),stopPrice=Math.max(m.low5*0.9975,entryFill*(1-stopPct/100));
  const checks:[[boolean,string],[boolean,string],[boolean,string],[boolean,string],[boolean,string],[boolean,string],[boolean,string],[boolean,string],[boolean,string],[boolean,string],[boolean,string],[boolean,string],[boolean,string]]=[
    [radar.radar_score>=0.72,"RADAR_LOW"],[m.spreadBps<=18,"SPREAD_WIDE"],[(radar.quote_volume??0)>=7_000_000,"LIQUIDITY_LOW"],[m.dipDepthPct>=depthNeed,"NO_DIP"],[m.reclaimPct>=reclaimNeed,"NO_RECLAIM"],[m.m1Pct>=0.02,"M1_NOT_UP"],[m.m3Pct>=-0.08,"M3_TOO_WEAK"],[m.rangePos<=0.82,"CHASE_ZONE"],[m.price>=m.ema9*0.994,"BELOW_RECLAIM_BAND"],[m.volumeBoost>=0.55,"VOLUME_THIN"],[netEdgeBps>=55,"EDGE_BELOW_COST"],[targetPrice>entryFill*1.0045,"TARGET_TOO_CLOSE"],[stopPrice<entryFill,"STOP_INVALID"]];
  const failed=checks.filter(([ok])=>!ok).map(([,r])=>r),depthScore=clamp(m.dipDepthPct/Math.max(depthNeed,0.01),0,2),reclaimScore=clamp(m.reclaimPct/Math.max(reclaimNeed,0.01),0,2),edgeScore=clamp(netEdgeBps/160,0,2),momentumScore=clamp((m.m1Pct+0.10)/0.45,0,1.5);
  const score=radar.radar_score*42+depthScore*16+reclaimScore*13+edgeScore*12+momentumScore*10+clamp(m.volumeBoost,0,2)*3.5;
  return {radar,market:m,score,canEnter:failed.length===0,reason:failed[0]??"DIP_RECLAIM_EDGE",depthNeed,reclaimNeed,netEdgeBps,targetPrice,stopPrice};
}
async function acquireLease(){const owner=crypto.randomUUID(),q=await db.rpc("brian_dip_v84_acquire_lease",{p_lease_key:LEASE_KEY,p_owner_id:owner,p_lease_seconds:55});if(q.error)throw q.error;const row=Array.isArray(q.data)?q.data[0]:q.data;if(!row?.acquired)return null;return{owner,generation:Number(row.lease_generation)};}
async function releaseLease(l:{owner:string;generation:number}|null){if(!l)return;try{await db.rpc("brian_dip_v84_release_lease",{p_lease_key:LEASE_KEY,p_owner_id:l.owner,p_lease_generation:l.generation});}catch{}}
async function loadPositions(sid:string):Promise<PositionRow[]>{const q=await db.from("brian_dip_multi_positions").select("*").eq("session_id",sid).order("entry_at",{ascending:true});if(q.error)throw q.error;return(q.data??[]) as PositionRow[];}
async function loadRecentExitMap(sid:string){const q=await db.from("brian_dip_multi_trades").select("symbol,exit_at").eq("session_id",sid).gte("exit_at",iso(Date.now()-COOLDOWN_MS)).order("exit_at",{ascending:false});if(q.error)throw q.error;const m=new Map<string,number>();for(const r of q.data??[])if(!m.has(String(r.symbol)))m.set(String(r.symbol),Date.parse(String(r.exit_at)));return m;}
async function closePosition(runtime:RuntimeRow,p:PositionRow,m:Market,reason:string){
  const qty=num(p.qty),exitFill=m.bid*(1-SLIPPAGE_BPS/10000),gross=qty*exitFill,exitFee=gross*FEE_BPS/10000,proceeds=gross-exitFee,pnl=proceeds-num(p.cost_basis_total),ret=num(p.cost_basis_total)>0?pnl/num(p.cost_basis_total)*100:0;
  const upd=await db.from("brian_dip_multi_runtime").update({cash:num(runtime.cash)+proceeds,realized_pnl:num(runtime.realized_pnl)+pnl,fees_paid:num(runtime.fees_paid)+exitFee,trade_count:runtime.trade_count+1,win_count:runtime.win_count+(pnl>0?1:0),loss_count:runtime.loss_count+(pnl<=0?1:0),updated_at:iso()}).eq("session_id",runtime.session_id).select("*").single(); if(upd.error)throw upd.error;
  const tr=await db.from("brian_dip_multi_trades").insert({session_id:runtime.session_id,symbol:p.symbol,entry_at:p.entry_at,exit_at:iso(),entry_price:num(p.entry_price),exit_price:exitFill,qty,notional:num(p.notional),entry_fee:num(p.entry_fee),exit_fee:exitFee,net_pnl:pnl,return_pct:ret,exit_reason:reason,entry_metrics:p.entry_metrics??{},exit_metrics:{engine:ENGINE,mark:m.price,bid:m.bid,spread_bps:m.spreadBps,atr_pct:m.atrPct,m1_pct:m.m1Pct,m3_pct:m.m3Pct,peak_price:num(p.peak_price)}}); if(tr.error)throw tr.error;
  const del=await db.from("brian_dip_multi_positions").delete().eq("session_id",runtime.session_id).eq("symbol",p.symbol); if(del.error)throw del.error;
  return {runtime:upd.data as RuntimeRow,pnl};
}
async function openPosition(runtime:RuntimeRow,s:Scored):Promise<RuntimeRow>{
  const cash=num(runtime.cash),start=num(runtime.starting_equity),feeRate=FEE_BPS/10000,notional=Math.min(start*0.25,cash/(1+feeRate)); if(notional<Math.max(40,start*0.05))throw Error("CASH_TOO_LOW");
  const fill=s.market.ask*(1+SLIPPAGE_BPS/10000),qty=notional/fill,entryFee=notional*feeRate,total=notional+entryFee;
  const ins=await db.from("brian_dip_multi_positions").insert({session_id:runtime.session_id,symbol:s.radar.symbol,qty,entry_price:fill,entry_at:iso(),notional,entry_fee:entryFee,cost_basis_total:total,stop_price:s.stopPrice,target_price:s.targetPrice,peak_price:fill,mark_price:s.market.price,unrealized_pnl:-entryFee,fee_bps:FEE_BPS,slippage_bps:SLIPPAGE_BPS,radar_score:s.radar.radar_score,entry_metrics:{engine:ENGINE,score:s.score,radar_score:s.radar.radar_score,price_change_pct_24h:s.radar.price_change_pct,range_pct_24h:s.radar.range_pct,live_spread_bps:s.market.spreadBps,atr_pct:s.market.atrPct,dip_depth_pct:s.market.dipDepthPct,depth_need_pct:s.depthNeed,reclaim_pct:s.market.reclaimPct,reclaim_need_pct:s.reclaimNeed,m1_pct:s.market.m1Pct,m3_pct:s.market.m3Pct,volume_boost:s.market.volumeBoost,range_pos:s.market.rangePos,net_edge_bps:s.netEdgeBps,high15:s.market.high15,low5:s.market.low5,long_only:true,shadow_only:true,live_execution:false}}); if(ins.error)throw ins.error;
  const upd=await db.from("brian_dip_multi_runtime").update({cash:cash-total,fees_paid:num(runtime.fees_paid)+entryFee,updated_at:iso()}).eq("session_id",runtime.session_id).select("*").single(); if(upd.error)throw upd.error; return upd.data as RuntimeRow;
}
async function recordDecisions(sid:string,rows:J[]){if(!rows.length)return;const q=await db.from("brian_dip_multi_decisions").insert(rows.map(r=>({session_id:sid,observed_at:iso(),...r})));if(q.error)console.error("decision-log",q.error.message);}
async function runEngine(){
  const lease=await acquireLease(); if(!lease)return{status:"WAIT_LEASE",engine:ENGINE,shadow_only:true,live_execution:false};
  try{
    const session=await latestDipSession(); if(!session)return{status:"NO_DIP_SESSION",engine:ENGINE,shadow_only:true,live_execution:false};
    let runtime=await ensureRuntime(session),positions=await loadPositions(session.sessionId); const recentExits=await loadRecentExitMap(session.sessionId),radar=await latestRadar();
    const symbols=[...new Set([...positions.map(p=>p.symbol),...radar.slice(0,MAX_CANDIDATES).map(r=>r.symbol)])].slice(0,MAX_CANDIDATES+MAX_POSITIONS);
    const mr=await Promise.allSettled(symbols.map(async symbol=>[symbol,await marketFor(symbol)] as const)),markets=new Map<string,Market>(); let marketErrors=0;
    for(const r of mr){if(r.status==="fulfilled")markets.set(r.value[0],r.value[1]);else marketErrors++;}
    const decisions:J[]=[],exits:J[]=[];
    for(const p of positions){
      const m=markets.get(p.symbol); if(!m){decisions.push({symbol:p.symbol,action:"HOLD",score:0,price:null,radar_score:num(p.radar_score),reason:"MARKET_UNAVAILABLE",metrics:{}});continue;}
      const qty=num(p.qty),markFill=m.bid*(1-SLIPPAGE_BPS/10000),gross=qty*markFill,exitFee=gross*FEE_BPS/10000,unrealized=gross-exitFee-num(p.cost_basis_total),entry=num(p.entry_price),peak=Math.max(num(p.peak_price,entry),m.price),profitPct=(markFill/entry-1)*100,trailGap=clamp(Math.max(0.32,m.atrPct*0.65),0.32,1.10),trailStop=profitPct>=0.55?peak*(1-trailGap/100):num(p.stop_price),stop=Math.max(num(p.stop_price),trailStop),held=Date.now()-Date.parse(p.entry_at); let reason="";
      if(markFill<=stop)reason=profitPct>0?"TRAIL_STOP":"HARD_STOP";else if(markFill>=num(p.target_price))reason="TARGET";else if(held>=MAX_HOLD_MS)reason="MAX_HOLD";else if(profitPct>=0.30&&m.m1Pct<-0.18&&m.m3Pct<-0.20)reason="MOMENTUM_FADE";
      if(reason){const c=await closePosition(runtime,p,m,reason);runtime=c.runtime;exits.push({symbol:p.symbol,reason,pnl:c.pnl});decisions.push({symbol:p.symbol,action:"SELL",score:100,price:markFill,radar_score:num(p.radar_score),reason,metrics:{profit_pct:profitPct,stop,target:num(p.target_price),peak,unrealized_pnl:unrealized}});}else{const u=await db.from("brian_dip_multi_positions").update({peak_price:peak,stop_price:stop,mark_price:m.price,unrealized_pnl:unrealized,updated_at:iso()}).eq("session_id",session.sessionId).eq("symbol",p.symbol);if(u.error)throw u.error;decisions.push({symbol:p.symbol,action:"HOLD",score:60,price:m.price,radar_score:num(p.radar_score),reason:profitPct>=0.55?"TRAIL_ACTIVE":"POSITION_OPEN",metrics:{profit_pct:profitPct,stop,target:num(p.target_price),peak,unrealized_pnl:unrealized,m1_pct:m.m1Pct,m3_pct:m.m3Pct}});}
    }
    positions=await loadPositions(session.sessionId); const open=new Set(positions.map(p=>p.symbol)),scored:Scored[]=[]; for(const r of radar.slice(0,MAX_CANDIDATES)){if(open.has(r.symbol))continue;const m=markets.get(r.symbol);if(m)scored.push(scoreEntry(r,m));} scored.sort((a,b)=>b.score-a.score);
    let slots=Math.max(0,MAX_POSITIONS-positions.length); const buys:J[]=[];
    for(const s of scored){if(slots<=0)break;const cooled=recentExits.has(s.radar.symbol)&&Date.now()-(recentExits.get(s.radar.symbol)??0)<COOLDOWN_MS,allow=session.active&&s.canEnter&&!cooled,reason=!session.active?"SESSION_PAUSED":cooled?"COOLDOWN":s.reason; decisions.push({symbol:s.radar.symbol,action:allow?"BUY":"WAIT",score:s.score,price:s.market.price,radar_score:s.radar.radar_score,reason,metrics:{dip_depth_pct:s.market.dipDepthPct,depth_need_pct:s.depthNeed,reclaim_pct:s.market.reclaimPct,reclaim_need_pct:s.reclaimNeed,m1_pct:s.market.m1Pct,m3_pct:s.market.m3Pct,atr_pct:s.market.atrPct,range_pos:s.market.rangePos,volume_boost:s.market.volumeBoost,live_spread_bps:s.market.spreadBps,net_edge_bps:s.netEdgeBps,target_price:s.targetPrice,stop_price:s.stopPrice,change_pct_24h:s.radar.price_change_pct,range_pct_24h:s.radar.range_pct}}); if(!allow)continue; runtime=await openPosition(runtime,s);buys.push({symbol:s.radar.symbol,price:s.market.ask,score:s.score,target:s.targetPrice,stop:s.stopPrice});slots--;}
    await recordDecisions(session.sessionId,decisions.slice(0,24)); const finalPositions=await loadPositions(session.sessionId),cash=num(runtime.cash),markValue=finalPositions.reduce((sum,p)=>sum+Math.max(0,num(p.cost_basis_total)+num(p.unrealized_pnl)),0),equity=cash+markValue,status=session.active?"RUNNING":"PAUSED";
    const ru=await db.from("brian_dip_multi_runtime").update({status,last_run_at:iso(),updated_at:iso()}).eq("session_id",session.sessionId).select("*").single();if(ru.error)throw ru.error;runtime=ru.data as RuntimeRow;
    return{status,engine:ENGINE,session_id:session.sessionId,shadow_only:true,live_execution:false,long_only:true,candidates_scanned:symbols.length,market_errors:marketErrors,buys,exits,open_positions:finalPositions.map(p=>({symbol:p.symbol,entry_price:num(p.entry_price),mark_price:num(p.mark_price),unrealized_pnl:num(p.unrealized_pnl),stop_price:num(p.stop_price),target_price:num(p.target_price),entry_at:p.entry_at})),cash,equity,realized_pnl:num(runtime.realized_pnl),trade_count:runtime.trade_count,wins:runtime.win_count,losses:runtime.loss_count};
  }finally{await releaseLease(lease);}
}
async function statusPayload(){
  const session=await latestDipSession();if(!session)return{status:"NO_DIP_SESSION",engine:ENGINE,shadow_only:true,live_execution:false};
  const [r,p,t,d]=await Promise.all([db.from("brian_dip_multi_runtime").select("*").eq("session_id",session.sessionId).maybeSingle(),db.from("brian_dip_multi_positions").select("*").eq("session_id",session.sessionId).order("entry_at",{ascending:false}),db.from("brian_dip_multi_trades").select("trade_id,symbol,entry_at,exit_at,entry_price,exit_price,net_pnl,return_pct,exit_reason").eq("session_id",session.sessionId).order("exit_at",{ascending:false}).limit(20),db.from("brian_dip_multi_decisions").select("observed_at,symbol,action,score,price,radar_score,reason,metrics").eq("session_id",session.sessionId).order("observed_at",{ascending:false}).limit(40)]); if(r.error)throw r.error;if(p.error)throw p.error;if(t.error)throw t.error;if(d.error)throw d.error;
  const runtime=r.data as RuntimeRow|null,positions=(p.data??[]) as PositionRow[],cash=runtime?num(runtime.cash):session.startingEquity,markValue=positions.reduce((sum,x)=>sum+Math.max(0,num(x.cost_basis_total)+num(x.unrealized_pnl)),0);
  return{status:runtime?.status??(session.active?"STARTING":"PAUSED"),engine:ENGINE,session_id:session.sessionId,source_session_active:session.active,shadow_only:true,live_execution:false,long_only:true,last_run_at:runtime?.last_run_at??null,starting_equity:runtime?num(runtime.starting_equity):session.startingEquity,cash,equity:cash+markValue,realized_pnl:runtime?num(runtime.realized_pnl):0,fees_paid:runtime?num(runtime.fees_paid):0,trade_count:runtime?.trade_count??0,wins:runtime?.win_count??0,losses:runtime?.loss_count??0,positions:positions.map(x=>({symbol:x.symbol,entry_price:num(x.entry_price),mark_price:num(x.mark_price),unrealized_pnl:num(x.unrealized_pnl),stop_price:num(x.stop_price),target_price:num(x.target_price),entry_at:x.entry_at,radar_score:num(x.radar_score)})),recent_trades:t.data??[],recent_decisions:d.data??[]};
}
Deno.serve(async(req:Request)=>{if(req.method!=="POST")return new Response("method",{status:405});let action="run";try{action=String(((await req.clone().json()) as J)?.action??"run").toLowerCase();}catch{}try{if(action==="status"){await requireAuth(req,"dashboard");return Response.json(await statusPayload(),{headers:{"cache-control":"no-store"}});}await requireAuth(req,"cron");return Response.json(await runEngine(),{headers:{"cache-control":"no-store"}});}catch(e){const message=e instanceof Error?e.message:String(e);console.error("dip-multi-shadow",message);return Response.json({status:"FAILED_CLOSED",engine:ENGINE,error:message,shadow_only:true,live_execution:false,long_only:true},{status:message.includes("UNAUTHORIZED")?401:500,headers:{"cache-control":"no-store"}});}});
