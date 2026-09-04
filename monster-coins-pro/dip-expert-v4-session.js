cfgUi=function(){return{symbols:[...v4Universe].slice(0,V4_UNIVERSE_SIZE),interval:'1m',step_pct:Number($('stepPctInput')?.value)||.15,dip_trigger_steps:Number($('dipTriggerInput')?.value)||2,rebound_steps:Number($('reboundInput')?.value)||1,take_profit_steps:Number($('takeProfitInput')?.value)||3,max_chase_steps:Number($('maxChaseInput')?.value)||5,fee_bps:Number($('feeInput')?.value)||10,slippage_bps:Number($('slippageInput')?.value)||1,expert_mode:true,auto_universe:true,universe_size:V4_UNIVERSE_SIZE,allow_shadow_short:true,max_shadow_leverage:1,engine_version:DIP_EXPERT_V4,v4_max_open:V4_MAX_OPEN,v4_cost_edge_mult:V4_COST_EDGE_MULT};};
params=function(){const eq=Number($('capitalInput')?.value);if(!(eq>0))throw Error('Kasa geçersiz.');if($('tradeInput'))$('tradeInput').value=String(eq);return{starting_equity:eq,trade_notional:eq,config:cfgUi(),engine_token:token()};};

function v4CompactState(){
  const sy={};for(const s of v4AllSymbols()){const st=v4Ensure(s);sy[s]={symbol:s,pos:st.pos||null,last:Number(st.last||live[s]||0),lastAction:st.lastAction||'WATCH',realized:Number(st.realized||0),dip:st.dip??null,armed:Boolean(st.armed),v4:st.v4};}
  return{v4:true,startingEquity:book.start,cash:book.cash,realizedPnl:book.realized,tradeCount:book.trades,winCount:book.wins,lossCount:book.losses,config:book.cfg,v4Universe:[...v4Universe],symbols:sy,pairGuard:v4PairGuard,closedOutcomes:v4ClosedOutcomes.slice(-20),sessionLossLockUntil:v4SessionLossLockUntil};
}
snapshot=async function(){
  if(!running||!sid||v4NeedsRestart)return;const m=metrics(),state=v4CompactState();try{await api('snapshot',{session_id:sid,engine_token:token(),observed_at:iso(),cash:book.cash,equity:m.equity,realized_pnl:book.realized,unrealized_pnl:m.unrealized,trade_count:book.trades,win_count:book.wins,loss_count:book.losses,state});v4CloudFault=false;if($('cloudState')){$('cloudState').textContent='V4 SYNC';$('cloudState').className='pos'}if($('cloudMeta'))$('cloudMeta').textContent=`Compact snapshot ${clock(new Date())}`}catch(e){engineErr(e)}
};
restore=function(d){
  session=d.session;history=d.events||[];if(!session){sid=null;running=false;v4NeedsRestart=false;book={start:Number($('capitalInput')?.value)||500,cash:Number($('capitalInput')?.value)||500,realized:0,trades:0,wins:0,losses:0,cfg:null};render();return;}
  sid=session.session_id;const isV4=session.config?.engine_version===DIP_EXPERT_V4;v4NeedsRestart=!isV4;
  if(!isV4){running=false;book.cfg=session.config||null;render();return;}
  const cfg={...session.config,engine_version:DIP_EXPERT_V4,max_shadow_leverage:1};book={start:Number(session.starting_equity),cash:Number(session.starting_equity),realized:0,trades:0,wins:0,losses:0,cfg};
  if($('capitalInput'))$('capitalInput').value=String(session.starting_equity);if($('tradeInput'))$('tradeInput').value=String(session.trade_notional);
  const z=d.snapshot,state=z?.state;if(state?.v4){book.cash=Number(z.cash);book.realized=Number(z.realized_pnl);book.trades=Number(z.trade_count);book.wins=Number(z.win_count);book.losses=Number(z.loss_count);v4Universe=[...(state.v4Universe||cfg.symbols||[])].slice(0,V4_UNIVERSE_SIZE);for(const k of Object.keys(v4PairGuard))delete v4PairGuard[k];Object.assign(v4PairGuard,state.pairGuard||{});v4ClosedOutcomes.splice(0,v4ClosedOutcomes.length,...(state.closedOutcomes||[]));v4SessionLossLockUntil=Number(state.sessionLossLockUntil||0);for(const [s,x] of Object.entries(state.symbols||{})){states[s]={...fresh(s),...x,symbol:s};v4Ensure(s)}}else{v4Universe=[...(cfg.symbols||[])].slice(0,V4_UNIVERSE_SIZE);v4Universe.forEach(v4SeedState)}
  v4Universe.forEach(v4Ensure);render();
};
status=async function(initial=false){
  try{const d=await api('status');restore(d);if(session?.status==='RUNNING'&&!v4NeedsRestart){try{await api('engine_check',{session_id:sid,engine_token:token()});running=true}catch{running=false}}else running=false;v4UiPatch();render();if(initial)toast(v4NeedsRestart?'V3 session bulundu · V4 için Yeni Kasa ile Restart.':'Dip Expert V4 hazır.')}catch(e){engineErr(e);if(initial)toast(String(e.message||e))}
};
start=async function(restart=false){
  v4Booting=true;try{await v4LoadHistory();const p=params(),d=await api(restart?'restart':'start',p),cfg={...p.config,...(d.config||{}),symbols:[...v4Universe],engine_version:DIP_EXPERT_V4,allow_shadow_short:true,max_shadow_leverage:1};session={session_id:d.session_id,status:'RUNNING',started_at:iso(),starting_equity:d.starting_equity,trade_notional:d.trade_notional,config:cfg};sid=d.session_id;book={start:Number(d.starting_equity),cash:Number(d.starting_equity),realized:0,trades:0,wins:0,losses:0,cfg};history=[];v4NeedsRestart=false;v4CloudFault=false;v4SessionLossLockUntil=0;v4ClosedOutcomes.splice(0);for(const k of Object.keys(v4PairGuard))delete v4PairGuard[k];for(const s of v4Universe)v4SeedState(s);running=true;v4ResetFunnel();event('ENGINE_START',null,null,{metadata:{expert_v4:true,starting_equity:d.starting_equity,universe_size:v4Universe.length,max_open:V4_MAX_OPEN,max_leverage:1,data:'native 1m+5m+15m+1h + aggTrade + bookTicker + depth5',entry:'cost gate + BTC regime + OFI + book pressure',risk:'risk sizing + heat + pair/global guards'}});await snapshot();connect();v4ScheduleUniverse();v4UiPatch();render();toast(restart?'Expert V4 temiz kasa ile başladı.':'Expert V4 başladı.')}catch(e){toast(String(e.message||e))}finally{v4Booting=false;render()}
};

engineErr=function(e){const t=String(e?.message||e);v4CloudFault=true;if(/UNAUTHORIZED_DIP_ENGINE|STALE_DIP_SESSION|DIP_SESSION_PAUSED/.test(t)){running=false;toast('V4 motor yetkisi durdu.')}else{if($('cloudState')){$('cloudState').textContent='PERSIST FAULT';$('cloudState').className='neg'}if($('cloudMeta'))$('cloudMeta').textContent=t.slice(0,100);toast('Cloud persist hatası: yeni girişler durduruldu.')}render();};

function v4EmitFunnel(){
  if(!running||!sid||v4NeedsRestart)return;const now=v4Now();if(now-v4LastFunnelEmit<60_000)return;v4LastFunnelEmit=now;const top=Object.entries(v4Funnel.rejected).sort((a,b)=>b[1]-a[1]).slice(0,8);event('INFO',null,null,{metadata:{expert_v4:true,info:'ENTRY_FUNNEL',evaluated:v4Funnel.evaluated,entered:v4Funnel.entered,rejected:Object.fromEntries(top),open:v4OpenCount(),heat:v4Heat(),equity:metrics().equity}});v4ResetFunnel();
}
function v4ScheduleUniverse(){
  clearInterval(v4UniverseTimer);v4UniverseTimer=setInterval(async()=>{try{const before=v4Universe.join(','),next=(await v4DiscoverUniverse(true)).join(',');if(before!==next){await v4LoadHistory();event('INFO',null,null,{metadata:{expert_v4:true,info:'UNIVERSE_REFRESH',symbols:[...v4Universe]}});connect();render()}}catch(e){console.warn('v4 universe',e)}},V4_UNIVERSE_REFRESH_MS);
}
setInterval(v4EmitFunnel,15_000);
