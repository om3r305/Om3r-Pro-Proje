import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const SUPABASE_URL=Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db=createClient(SUPABASE_URL,SERVICE_ROLE,{auth:{persistSession:false,autoRefreshToken:false}});

const ENGINE_ID="dip-multiasset-v1";
const ARENA_ID="dip-aggressive-arena-v1";
const GUARDIAN_VERSION="dip-position-guardian-v8-confirmed-turn-wave-memory";
const LEASE_KEY="brian-dip-multiasset-worker-v1";
const FEE_BPS=10;
const HOSTS=["https://api.binance.com","https://api1.binance.com","https://api2.binance.com"];
const OFFSETS_MS=[0,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000];
const HEAVY_TICKS=new Set([2,5,8]);

type Pos=Record<string,any>;
const num=(v:any,f=0)=>Number.isFinite(Number(v))?Number(v):f;
const clip=(v:number,lo=0,hi=1)=>Math.max(lo,Math.min(hi,v));
const iso=()=>new Date().toISOString();
const sleep=(ms:number)=>new Promise(r=>setTimeout(r,ms));

async function sha256Hex(v:string){
  const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(v)));
  return [...d].map(b=>b.toString(16).padStart(2,"0")).join("");
}
function same(a:string,b:string){if(a.length!==b.length)return false;let d=0;for(let i=0;i<a.length;i++)d|=a.charCodeAt(i)^b.charCodeAt(i);return d===0;}
async function requireCron(req:Request){
  const supplied=(req.headers.get("x-brian-cron-key")||"").trim();
  if(!supplied)throw new Error("UNAUTHORIZED_CRON");
  const q=await db.from("brian_dashboard_auth").select("cron_key_sha256").eq("auth_id","control-v3").maybeSingle();
  if(q.error||!q.data?.cron_key_sha256)throw new Error("CRON_AUTH_UNAVAILABLE");
  if(!same(await sha256Hex(supplied),String(q.data.cron_key_sha256)))throw new Error("UNAUTHORIZED_CRON");
}
async function acquireLease(){
  const owner=crypto.randomUUID();
  const q=await db.rpc("brian_dip_v84_acquire_lease",{p_lease_key:LEASE_KEY,p_owner_id:owner,p_lease_seconds:6});
  if(q.error)throw q.error;
  const row=Array.isArray(q.data)?q.data[0]:q.data;
  return row?.acquired?{owner,generation:Number(row.lease_generation)}:null;
}
async function releaseLease(l:any){
  if(!l)return;
  try{await db.rpc("brian_dip_v84_release_lease",{p_lease_key:LEASE_KEY,p_owner_id:l.owner,p_lease_generation:l.generation});}catch{}
}
async function marketJson(path:string,timeout=1400){
  let last="MARKET_UNAVAILABLE";
  for(const host of HOSTS){
    try{
      const r=await fetch(host+path,{headers:{accept:"application/json","user-agent":"Brian-DIP-Guardian/1.0"},signal:AbortSignal.timeout(timeout)});
      if(!r.ok){last="HTTP_"+r.status;if(r.status===418||r.status===429)break;continue;}
      return await r.json();
    }catch(e){last=e instanceof Error?e.message:String(e);}
  }
  throw new Error("BINANCE:"+last);
}
async function book(symbol:string){
  const v:any=await marketJson("/api/v3/ticker/bookTicker?symbol="+encodeURIComponent(symbol),1200);
  const bid=num(v?.bidPrice),ask=num(v?.askPrice);
  if(!(bid>0&&ask>bid))throw new Error("BAD_BOOK");
  const mid=(bid+ask)/2;
  return{bid,ask,spreadBps:(ask-bid)/mid*10000};
}
async function micro(symbol:string){
  const raw:any=await marketJson("/api/v3/aggTrades?symbol="+encodeURIComponent(symbol)+"&limit=200",1400);
  if(!Array.isArray(raw)||raw.length<2)return{ret5:0,ret15:0,buyRatio:.5,trades:0};
  const now=Date.now();
  const rows=raw.map((x:any)=>({p:num(x.p),q:num(x.q),t:num(x.T),buy:!Boolean(x.m)})).filter((x:any)=>x.p>0&&x.q>0&&x.t>0);
  if(rows.length<2)return{ret5:0,ret15:0,buyRatio:.5,trades:rows.length};
  const last=rows.at(-1)!.p;
  const base=(ms:number)=>{
    const cutoff=now-ms;
    const r=rows.find((x:any)=>x.t>=cutoff);
    return r?.p||rows[0].p;
  };
  const r5=(last/base(5000)-1)*10000,r15=(last/base(15000)-1)*10000;
  const recent=rows.filter((x:any)=>x.t>=now-15000);
  let buy=0,total=0;
  for(const x of recent){const v=x.p*x.q;total+=v;if(x.buy)buy+=v;}
  return{ret5:r5,ret15:r15,buyRatio:total>0?buy/total:.5,trades:recent.length};
}
function cooldownMs(reason:string,pnl:number){
  if(pnl<0)return reason==="STOP"?2*60_000:12*60_000;
  if(pnl>0)return 5*60_000;
  return 10*60_000;
}
function inferRegime(p:Pos){
  const cont=num(p.last_continuation,p.entry_continuation),e15=num(p.last_expected_15m_bps),e30=num(p.last_expected_30m_bps);
  if(cont>=.82&&e15>=55&&e30>=70)return"EXPLOSIVE_RUNNER";
  if(cont>=.72&&e15>=25&&e30>=35)return"STRONG_RUNNER";
  return"PROTECT_RUNNER";
}
function fillPrice(reason:string,p:Pos,bid:number,spreadBps:number){
  const extra=Math.max(2,spreadBps/2)+2;
  let trigger=bid;
  if(reason==="STOP")trigger=Math.min(bid,num(p.stop,bid));
  else if((reason==="HARVEST_TRAIL"||reason==="PROFIT_RATCHET")&&num(p.trail)>0)trigger=Math.min(bid,num(p.trail));
  return trigger*(1-extra/10000);
}
async function insertEventFor(engineId:string,row:any){
  const q=await db.from("brian_dip_multiasset_events").insert({
    engine_id:engineId,observed_at:row.observed_at,symbol:row.symbol,action:row.action,price:row.price,qty:row.qty,
    notional:row.notional,pnl:row.pnl,reason:row.reason,metadata:row.metadata
  });
  if(q.error)throw q.error;
}
async function insertEvent(row:any){return insertEventFor(ENGINE_ID,row);}

async function processArenaGuardian(heavy:boolean){
  const q=await db.from("brian_dip_multiasset_state").select("*").eq("engine_id",ARENA_ID).maybeSingle();
  if(q.error)throw q.error;
  const state:any=q.data;
  if(!state||!state.enabled)return{status:"ARENA_IDLE_DISABLED",checked:0,actions:0};
  const positions:{[k:string]:Pos}={...(state.positions||{})},cooldowns:any={...(state.cooldowns||{})},symbols=Object.keys(positions);
  if(!symbols.length){
    const scan={...(state.last_scan||{}),arena_guardian:{status:"IDLE_NO_POSITIONS",version:GUARDIAN_VERSION,heartbeat_at:iso(),heavy_tick:heavy,checked_symbols:[],telemetry:[],actions:[]}};
    const u=await db.from("brian_dip_multiasset_state").update({last_scan:scan,updated_at:iso(),shadow_only:true,live_execution:false}).eq("engine_id",ARENA_ID);
    if(u.error)throw u.error;
    return{status:"ARENA_IDLE_NO_POSITIONS",checked:0,actions:0};
  }

  const rows=await Promise.all(symbols.map(async symbol=>{
    try{return{symbol,b:await book(symbol),m:heavy?await micro(symbol):null};}
    catch(e){return{symbol,error:e instanceof Error?e.message:String(e)};}
  }));

  let cash=num(state.cash),realized=num(state.realized_pnl),trades=num(state.trade_count),wins=num(state.win_count),losses=num(state.loss_count);
  const actions:any[]=[],telemetry:any[]=[],quotes:Record<string,any>={};
  let recent=Array.isArray(state.last_scan?.recent_events)?[...state.last_scan.recent_events]:[];

  for(const row of rows){
    const p=positions[row.symbol];
    if(!p||row.error||!row.b){telemetry.push({symbol:row.symbol,status:"MARKET_ERROR",error:row.error});continue;}
    const now=Date.now(),bid=row.b.bid,spreadBps=row.b.spreadBps,slip=Math.max(2,spreadBps/2),qty=num(p.qty),entry=num(p.entry),costBasis=num(p.cost_basis);
    p.max_price=Math.max(num(p.max_price,entry),bid);
    const maxGross=qty*p.max_price,maxFee=maxGross*FEE_BPS/10000,maxNet=Math.max(0,maxGross-maxFee-costBasis),
      curGross=qty*bid,curFee=curGross*FEE_BPS/10000,currentNet=curGross-curFee-costBasis,
      peakCapture=maxNet>0?clip(currentNet/maxNet,0,1.25):0,peakDrawBps=p.max_price>0?(p.max_price-bid)/p.max_price*10000:0,
      meaningfulPeak=maxNet>=Math.max(.30,costBasis*.0045),
      brainU=num(p.last_forecast_utility,p.entry_forecast_utility),entryU=num(p.entry_forecast_utility,brainU),peakU=Math.max(num(p.peak_forecast_utility,brainU),brainU),
      brainWeak=brainU<=entryU-.10||brainU<=peakU-.16||num(p.thesis_decay_streak)>=1,
      vote=num(p.last_vote),age=Math.max(0,now-Date.parse(String(p.opened_at||iso())));

    let softMicro=false,hardMicro=false;
    if(heavy&&row.m){
      softMicro=(row.m.ret5<0&&row.m.ret15<0&&row.m.buyRatio<.48)||row.m.buyRatio<.40;
      hardMicro=row.m.ret5<=-12||(row.m.ret5<=-6&&row.m.ret15<=-10&&row.m.buyRatio<.44);
      p.arena_last_micro_at=now;
      p.arena_micro_soft=softMicro;
      p.arena_micro_hard=hardMicro;
      p.arena_last_micro=row.m;
    }
    const microFresh=now-num(p.arena_last_micro_at)<20_000,
      microSoftFresh=microFresh&&p.arena_micro_soft===true,
      microHardFresh=microFresh&&p.arena_micro_hard===true,
      softStreak=microSoftFresh?Math.min(3,num(p.arena_soft_streak)+1):0,
      peakGiveback=maxNet>0?1-clip(currentNet/maxNet,0,1):0;
    p.arena_soft_streak=softStreak;
    const actualTurn=Boolean(meaningfulPeak&&currentNet>0&&(
        (microHardFresh&&(peakDrawBps>=5||peakGiveback>=.07))||
        (softStreak>=2&&(peakDrawBps>=8||peakGiveback>=.14))||
        (brainWeak&&(peakDrawBps>=7||peakGiveback>=.12))||
        (vote<0&&(peakDrawBps>=7||peakGiveback>=.12))
      )),
      rapidTurn=Boolean(age>=25_000&&meaningfulPeak&&currentNet>0&&microHardFresh&&peakGiveback>=.05);

    let reason="";
    if(rapidTurn||actualTurn)reason="ARENA_PEAK_REVERSAL";

    telemetry.push({symbol:row.symbol,bid,spread_bps:spreadBps,max_price:p.max_price,current_net_pnl:currentNet,peak_net_pnl:maxNet,
      peak_capture_now:peakCapture,peak_draw_bps:peakDrawBps,peak_giveback_ratio:peakGiveback,meaningful_peak:meaningfulPeak,
      brain_utility:brainU,entry_brain_utility:entryU,peak_brain_utility:peakU,brain_weak:brainWeak,forecast_vote_bps:vote,
      micro_fresh:microFresh,micro_soft:microSoftFresh,micro_hard:microHardFresh,micro:row.m??p.arena_last_micro??null,decision:reason||"HOLD"});
    quotes[row.symbol]={bid,spreadBps,currentNet};

    if(!reason)continue;

    const exit=bid*(1-(slip+2)/10000),gross=qty*exit,fee=gross*FEE_BPS/10000,pnl=gross-fee-costBasis;
    cash+=gross-fee;realized+=pnl;trades++;if(pnl>.01)wins++;else if(pnl<-.01)losses++;
    delete positions[row.symbol];
    cooldowns[row.symbol]={exit_at:now,exit_price:exit,reason,pnl,post_exit_low:exit,post_exit_high:exit,weak_seen:brainWeak||vote<0,
      exit_forecast_utility:brainU,exit_vote:vote,prior_entry:entry,wave_reset_consumed:false};

    const event={observed_at:iso(),symbol:row.symbol,action:"SELL",price:exit,qty,notional:gross,pnl,reason,metadata:{
      arena:true,arena_guardian_fast_exit:true,guardian_version:GUARDIAN_VERSION,guardian_interval_target_seconds:5,
      engine_version:state.last_scan?.engine_version||null,policy_version:state.last_scan?.policy_version||null,
      entry,max_price:p.max_price,peak_net_pnl:maxNet,peak_capture_ratio:maxNet>0?clip(Math.max(0,pnl)/maxNet,0,1):null,
      peak_capture_before_exit:peakCapture,peak_draw_bps:peakDrawBps,peak_giveback_ratio:peakGiveback,
      brain_utility:brainU,entry_brain_utility:entryU,peak_brain_utility:peakU,brain_weak:brainWeak,forecast_vote_bps:vote,
      micro:row.m??p.arena_last_micro??null,spread_bps:spreadBps,shadow_fill_model:"ARENA_GUARDIAN_CURRENT_BID_SLIPPAGE",shadow_only:true,live_execution:false
    }};
    await insertEventFor(ARENA_ID,event);actions.push(event);recent.unshift(event);recent=recent.slice(0,20);
  }

  let equity=cash;
  const openPositions:any[]=[];
  for(const p of Object.values(positions)){
    const q=quotes[String(p.symbol)],mark=q?.bid||num(p.last_mark,p.entry),gross=num(p.qty)*mark,fee=gross*FEE_BPS/10000,unrealized=gross-fee-num(p.cost_basis);
    equity+=gross-fee;
    p.last_mark=mark;
    openPositions.push({symbol:p.symbol,entry:num(p.entry),mark,qty:num(p.qty),notional:num(p.qty)*num(p.entry),allocation_fraction:num(p.allocation_fraction),
      slot_kind:p.slot_kind||"REGULAR",brain_utility:num(p.last_forecast_utility,p.entry_brain_utility),entry_brain_utility:num(p.entry_brain_utility),
      forecast_utility:num(p.last_forecast_utility),explosion_score:num(p.entry_explosion_score),unrealized_pnl:unrealized,max_price:num(p.max_price),opened_at:p.opened_at});
  }

  const scan={...(state.last_scan||{}),equity,open_positions:openPositions,recent_events:recent,arena_guardian:{
    status:"RUNNING",version:GUARDIAN_VERSION,heartbeat_at:iso(),heavy_tick:heavy,open_positions:openPositions.length,checked_symbols:symbols,
    strategy:"PEAK_PROFIT + FORECAST/MICRO TURN DETECTION",telemetry,actions:actions.map(a=>({symbol:a.symbol,reason:a.reason,pnl:a.pnl,price:a.price}))
  }};
  const u=await db.from("brian_dip_multiasset_state").update({
    cash,realized_pnl:realized,trade_count:trades,win_count:wins,loss_count:losses,positions,cooldowns,last_scan:scan,updated_at:iso(),shadow_only:true,live_execution:false
  }).eq("engine_id",ARENA_ID);
  if(u.error)throw u.error;
  return{status:"ARENA_RUNNING",checked:symbols.length,heavy,actions:actions.length};
}

async function tick(heavy:boolean){
  const lease=await acquireLease();
  if(!lease)return{status:"WAIT_LEASE"};
  try{
    const arena=await processArenaGuardian(heavy);
    const q=await db.from("brian_dip_multiasset_state").select("*").eq("engine_id",ENGINE_ID).maybeSingle();
    if(q.error)throw q.error;
    const state:any=q.data;
    if(!state||!state.enabled)return{status:"IDLE_DISABLED",arena};
    const positions:{[k:string]:Pos}={...(state.positions||{})};
    const symbols=Object.keys(positions);
    if(!symbols.length)return{status:"IDLE_NO_POSITIONS",arena};

    const marketRows=await Promise.all(symbols.map(async symbol=>{
      try{
        const b=await book(symbol);
        const m=heavy?await micro(symbol):null;
        return{symbol,b,m};
      }catch(e){return{symbol,error:e instanceof Error?e.message:String(e)};}
    }));

    let cash=num(state.cash),realized=num(state.realized_pnl),trades=num(state.trade_count),wins=num(state.win_count),losses=num(state.loss_count);
    const cooldowns:any={...(state.cooldowns||{})};
    let lossStreak=num(state.last_scan?.loss_streak),lastLossAt=num(state.last_scan?.last_loss_at);
    const actions:any[]=[];
    const telemetry:any[]=[];

    for(const row of marketRows){
      const p=positions[row.symbol];
      if(!p||row.error||!row.b){telemetry.push({symbol:row.symbol,status:"MARKET_ERROR",error:row.error});continue;}
      const bid=row.b.bid,spreadBps=row.b.spreadBps,slip=Math.max(2,spreadBps/2),qty=num(p.qty),entry=num(p.entry),costBasis=num(p.cost_basis);
      p.max_price=Math.max(num(p.max_price,entry),bid);
      const maxGross=qty*p.max_price,maxFee=maxGross*FEE_BPS/10000,maxNet=Math.max(0,maxGross-maxFee-costBasis);
      const currentGross=qty*bid,currentFee=currentGross*FEE_BPS/10000,currentNet=currentGross-currentFee-costBasis;
      const peakDrawBps=p.max_price>0?(p.max_price-bid)/p.max_price*10000:0;
      const targetDist=Math.max(1e-12,num(p.target)-entry),progress=clip((p.max_price-entry)/targetDist,0,2);

      if(!p.harvest_armed&&bid>=num(p.target)){
        p.harvest_armed=true;p.runner_mode=true;p.harvest_at=p.harvest_at||iso();p.winner_expansion=true;
      }

      const regime=inferRegime(p);
      p.runner_regime=regime;
      const cont=num(p.last_continuation,p.entry_continuation),e15=num(p.last_expected_15m_bps),e30=num(p.last_expected_30m_bps),
        fallbackU=clip(.24+cont*.46+Math.tanh(e15/60)*.14+Math.tanh(e30/95)*.16),
        brainU=num(p.last_forecast_utility,fallbackU),entryU=num(p.entry_forecast_utility,brainU),peakU=Math.max(num(p.peak_forecast_utility,brainU),brainU),
        brainFresh=num(p.last_thesis_at)>0&&Date.now()-num(p.last_thesis_at)<=95_000,
        brainStrong=Boolean(brainFresh&&brainU>=Math.max(.56,entryU-.05)&&brainU>=peakU-.16&&num(p.thesis_decay_streak)<2),
        brainWeak=Boolean(!brainFresh||brainU<=entryU-.12||brainU<=peakU-.20||num(p.thesis_decay_streak)>=2),
        ageMs=Math.max(0,Date.now()-Date.parse(String(p.opened_at||iso())));
      const microSoft=Boolean(heavy&&row.m&&((row.m.ret5<=-6&&row.m.ret15<=-8&&row.m.buyRatio<.46)||row.m.buyRatio<.38)),
        microHard=Boolean(heavy&&row.m&&((row.m.ret5<=-12&&row.m.ret15<=-16&&row.m.buyRatio<.42)||row.m.buyRatio<.32));
      const keep=brainStrong?(regime==="EXPLOSIVE_RUNNER"?.45:.62):brainWeak?.86:.76;
      const fillFactor=(1-(slip+2)/10000)*(1-FEE_BPS/10000);
      const noRetroCap=bid*(1-slip/10000);
      const peakCaptureNow=maxNet>0?clip(currentNet/maxNet,0,1.25):0;
      const meaningfulPeak=maxNet>=Math.max(.12,costBasis*.0040);

      if(p.harvest_armed&&maxNet>0){
        const desired=maxNet*keep;
        const raw=(costBasis+desired)/Math.max(1e-12,qty*fillFactor);
        const candidate=Math.min(raw,noRetroCap);
        if(candidate>num(p.trail)){p.trail=candidate;p.runner_trail=Math.max(num(p.runner_trail),candidate);}
      }else if(!p.harvest_armed&&maxNet>=Math.max(.12,costBasis*.0025)&&progress>=.30){
        const baseKeep=brainStrong?.34:brainWeak?.80:.58,
          drawKeep=brainStrong?(peakDrawBps>=26?.58:baseKeep):brainWeak?(peakDrawBps>=10?.88:.82):(peakDrawBps>=18?.78:baseKeep),
          microKeep=meaningfulPeak&&microSoft?.84:baseKeep,earlyKeep=Math.max(baseKeep,drawKeep,microKeep),
          raw=(costBasis+maxNet*earlyKeep)/Math.max(1e-12,qty*fillFactor),candidate=Math.min(raw,noRetroCap);
        if(candidate>num(p.trail))p.trail=candidate;
        p.profit_lock_active=true;p.profit_lock_ratio=earlyKeep;p.profit_lock_max_net_pnl=Math.max(num(p.profit_lock_max_net_pnl),maxNet);p.profit_lock_armed_at=p.profit_lock_armed_at||iso();
      }

      const rawPeakGiveback=Boolean(!p.harvest_armed&&meaningfulPeak&&currentNet>0&&peakDrawBps>=12&&peakCaptureNow<=.84),
        prevGivebackStreak=num(p.peak_giveback_streak),peakGivebackStreak=rawPeakGiveback?Math.min(3,prevGivebackStreak+1):0;
      p.peak_giveback_streak=peakGivebackStreak;
      const forecastVeto=Boolean(brainStrong&&!microHard);
      const peakGiveback=Boolean(rawPeakGiveback&&!forecastVeto&&(brainWeak||microHard||(peakGivebackStreak>=2&&microSoft)));
      const ratchetTouched=Boolean(!p.harvest_armed&&p.profit_lock_active&&num(p.trail)>0&&bid<=num(p.trail)),
        harvestTouched=Boolean(p.harvest_armed&&num(p.trail)>0&&bid<=num(p.trail)),
        ratchetExit=Boolean(ratchetTouched&&(brainWeak||microHard||(!brainStrong&&microSoft))),
        harvestExit=Boolean(harvestTouched&&(brainWeak||microHard||(!brainStrong&&microSoft))),
        noMfe=Boolean(num(p.max_price,entry)<=entry*1.0015),
        earlyMicroFailure=Boolean(heavy&&row.m&&!p.harvest_armed&&ageMs<=120_000&&currentNet<0&&noMfe&&row.m.ret5<=-18&&row.m.ret15<=-28&&row.m.buyRatio<.40);

      let reason="";
      if(bid<=num(p.stop))reason="STOP";
      else if(earlyMicroFailure)reason="MICRO_THESIS_BREAK";
      else if(harvestExit)reason="HARVEST_TRAIL";
      else if(peakGiveback)reason="PEAK_PROFIT_LOCK";
      else if(ratchetExit)reason="PROFIT_RATCHET";

      let microBreak=false;
      if(!reason&&heavy&&p.harvest_armed&&row.m&&currentNet>0){
        const m=row.m;
        microBreak=peakDrawBps>=18&&m.ret5<=-12&&m.ret15<=-18&&m.buyRatio<.45;
        if(microBreak)reason="PEAK_REVERSAL";
      }

      telemetry.push({symbol:row.symbol,bid,spread_bps:spreadBps,max_price:p.max_price,trail:p.trail??null,harvest_armed:p.harvest_armed===true,runner_regime:regime,peak_draw_bps:peakDrawBps,current_net_pnl:currentNet,peak_net_pnl:maxNet,peak_capture_now:peakCaptureNow,meaningful_peak:meaningfulPeak,profit_keep_ratio:p.profit_lock_ratio??keep,micro_softening:microSoft,micro_hard:microHard,brain_fresh:brainFresh,brain_utility:brainU,entry_brain_utility:entryU,peak_brain_utility:peakU,brain_strong:brainStrong,brain_weak:brainWeak,peak_giveback_raw:rawPeakGiveback,peak_giveback_streak:peakGivebackStreak,forecast_veto:forecastVeto,ratchet_touched:ratchetTouched,harvest_touched:harvestTouched,early_micro_failure:earlyMicroFailure,peak_giveback_exit:peakGiveback,micro:row.m??null,decision:reason||"HOLD"});

      if(!reason)continue;

      const exit=fillPrice(reason,p,bid,spreadBps),gross=qty*exit,fee=gross*FEE_BPS/10000,pnl=gross-fee-costBasis;
      const peakCapture=maxNet>0?clip(Math.max(0,pnl)/maxNet,0,1):null;
      cash+=gross-fee;realized+=pnl;trades++;
      if(pnl>.01){wins++;lossStreak=0;}else if(pnl<-.01){losses++;lossStreak++;lastLossAt=Date.now();}
      delete positions[row.symbol];
      const exitNow=Date.now(),prev:any=cooldowns[row.symbol]||{},prevLossAt=num(prev.last_loss_at),sameWindow=pnl<-.01&&prevLossAt>0&&exitNow-prevLossAt<6*60*60_000;
      const symbolLossStreak=pnl<-.01?(sameWindow?Math.max(1,num(prev.symbol_loss_streak))+1:1):0,reentryCount=num(p.reentry_attempt),
        waveProfitBank=Math.max(0,num(p.wave_profit_bank)+pnl),waveStartedAt=num(p.wave_started_at,Date.parse(String(p.opened_at||"")));
      let until=exitNow+cooldownMs(reason,pnl);
      if(pnl<-.01&&symbolLossStreak>=2)until=exitNow+(symbolLossStreak>=3?8*60*60_000:4*60*60_000);
      const waveExhaustedUntil=reentryCount>=1?exitNow+45*60_000:0;
      cooldowns[row.symbol]={until,reason,exit_price:exit,exit_at:exitNow,pnl,symbol_loss_streak:symbolLossStreak,last_loss_at:pnl<-.01?exitNow:prevLossAt,
        circuit_breaker:symbolLossStreak>=2,prior_entry:entry,post_exit_low:exit,post_exit_high:exit,reentry_count:reentryCount,
        wave_profit_bank:waveProfitBank,wave_started_at:waveStartedAt,wave_exhausted_until:waveExhaustedUntil,
        exit_continuation:cont,exit_expected_15m_bps:e15,exit_expected_30m_bps:e30,exit_runner_regime:regime,exit_peak_capture:peakCapture,
        exit_forecast_utility:brainU,exit_brain_strong:brainStrong,exit_brain_weak:brainWeak};

      const event={observed_at:iso(),symbol:row.symbol,action:"SELL",price:exit,qty,notional:gross,pnl,reason,metadata:{
        guardian_version:GUARDIAN_VERSION,engine_version:state.last_scan?.engine_version||null,policy_version:state.last_scan?.policy_version||null,
        guardian_fast_exit:true,guardian_interval_target_seconds:5,entry,stop:p.stop,harvest_trigger:p.target,trail:p.trail??null,max_price:p.max_price,
        harvest_armed:p.harvest_armed===true,runner_regime:regime,peak_draw_bps:peakDrawBps,peak_capture_ratio:peakCapture,peak_capture_before_exit:peakCaptureNow,meaningful_peak:meaningfulPeak,profit_keep_ratio:p.profit_lock_ratio??keep,peak_giveback_exit:peakGiveback,peak_giveback_streak:peakGivebackStreak,forecast_veto:forecastVeto,brain_fresh:brainFresh,brain_utility:brainU,entry_brain_utility:entryU,peak_brain_utility:peakU,brain_strong:brainStrong,brain_weak:brainWeak,early_micro_failure:earlyMicroFailure,
        micro:row.m??null,micro_softening:microSoft,micro_hard:microHard,spread_bps:spreadBps,reentry_type:p.reentry_type??"FRESH",reentry_attempt:num(p.reentry_attempt),wave_profit_bank:Math.max(0,num(p.wave_profit_bank)+pnl),
        shadow_fill_model:"GUARDIAN_CURRENT_OR_TRIGGER_WORSE",shadow_only:true,live_execution:false
      }};
      await insertEvent(event);actions.push(event);
    }

    const scan={...(state.last_scan||{}),loss_streak:lossStreak,last_loss_at:lastLossAt,position_guardian:{
      status:"RUNNING",version:GUARDIAN_VERSION,heartbeat_at:iso(),heavy_tick:heavy,open_positions:Object.keys(positions).length,checked_symbols:symbols,telemetry,actions:actions.map(a=>({symbol:a.symbol,reason:a.reason,pnl:a.pnl,price:a.price}))
    }};
    const u=await db.from("brian_dip_multiasset_state").update({
      cash,realized_pnl:realized,trade_count:trades,win_count:wins,loss_count:losses,positions,cooldowns,last_scan:scan,updated_at:iso(),shadow_only:true,live_execution:false
    }).eq("engine_id",ENGINE_ID);
    if(u.error)throw u.error;
    return{status:"RUNNING",checked:symbols.length,heavy,actions:actions.length,arena};
  }finally{await releaseLease(lease);}
}

Deno.serve(async(req:Request)=>{
  if(req.method==="GET")return Response.json({status:"OK",version:GUARDIAN_VERSION,interval_target_seconds:5,shadow_only:true,live_execution:false});
  if(req.method!=="POST")return new Response("method",{status:405});
  try{
    await requireCron(req);
    const started=Date.now(),results:any[]=[];
    for(let i=0;i<OFFSETS_MS.length;i++){
      const wait=Math.max(0,started+OFFSETS_MS[i]-Date.now());
      if(wait>0)await sleep(wait);
      results.push(await tick(HEAVY_TICKS.has(i)));
    }
    return Response.json({status:"COMPLETE",version:GUARDIAN_VERSION,ticks:results.length,results,elapsed_ms:Date.now()-started,shadow_only:true,live_execution:false},{headers:{"cache-control":"no-store"}});
  }catch(e){
    const message=e instanceof Error?e.message:String(e);
    return Response.json({status:"FAILED_CLOSED",version:GUARDIAN_VERSION,error:message,shadow_only:true,live_execution:false},{status:message.includes("UNAUTHORIZED")?401:500});
  }
});
