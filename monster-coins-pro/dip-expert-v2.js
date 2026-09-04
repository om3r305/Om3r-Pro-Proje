/* Brian Aggressive Dip — Chart Expert V2 live adapter.
   Additive/shadow-only: normal Brian Phase 3.7 is not mutated. */
const DIP_EXPERT_V2='aggressive-dip-expert-v2';
let expertHistoryReady=false,expertBooting=false;
const _baseReset=reset,_baseRestore=restore,_baseHistoryLoad=historyLoad,_baseRender=render,_baseRenderKpi=renderKpi,_baseNote=note;

function expertHydrate(st){
  if(st.startPrice===undefined)st.startPrice=null;if(st.anchorHigh===undefined)st.anchorHigh=null;
  if(st.postStartLow===undefined)st.postStartLow=null;if(st.referenceLow===undefined)st.referenceLow=null;
  if(st.baselineLow===undefined)st.baselineLow=null;if(st.baselineHigh===undefined)st.baselineHigh=null;
  if(st.warmupReady===undefined)st.warmupReady=false;if(st.startClosedAt===undefined)st.startClosedAt=null;
  if(st.freshClosedSeen===undefined)st.freshClosedSeen=false;if(st.expert===undefined)st.expert=null;
  if(st.expertUpdatedAt===undefined)st.expertUpdatedAt=0;if(st.maxSinceEntry===undefined)st.maxSinceEntry=null;
  return st;
}
reset=function(start,cfg){_baseReset(start,cfg);COINS.forEach(s=>expertHydrate(states[s]));};
restore=function(d){
  _baseRestore(d);COINS.forEach(s=>expertHydrate(states[s]));
  const z=d?.snapshot?.state;if(z?.config)book.cfg={...book.cfg,...z.config,expert_mode:true,engine_version:DIP_EXPERT_V2};
  if(expertHistoryReady)seedExperts(false);
};

const xMean=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:0;
const xSd=a=>{if(a.length<2)return 0;let m=xMean(a);return Math.sqrt(xMean(a.map(x=>(x-m)**2)))};
const xClamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function xEma(a,n){if(!a.length)return[];let k=2/(n+1),o=[a[0]];for(let i=1;i<a.length;i++)o.push(o.at(-1)+k*(a[i]-o.at(-1)));return o}
function xRsi(a,n=14){if(a.length<n+1)return 50;let g=0,l=0;for(let i=a.length-n;i<a.length;i++){let d=a[i]-a[i-1];g+=Math.max(d,0);l+=Math.max(-d,0)}g/=n;l/=n;return l===0?(g?100:50):100-100/(1+g/l)}
function xAtr(r,n=14){if(r.length<n+1)return r.length?r.at(-1).h-r.at(-1).l:0;let a=[];for(let i=r.length-n;i<r.length;i++){let pc=r[i-1]?.c??r[i].c;a.push(Math.max(r[i].h-r[i].l,Math.abs(r[i].h-pc),Math.abs(r[i].l-pc)))}return xMean(a)}
function xClosed(sym){let now=Date.now();return(candles[sym]||[]).filter(x=>x.t+60000<=now).slice(-160)}
function xAgg(rows,min){let ms=min*60000,m=new Map;for(let r of rows){let k=Math.floor(r.t/ms)*ms,z=m.get(k);if(!z)m.set(k,{t:k,o:r.o,h:r.h,l:r.l,c:r.c,v:r.v});else{z.h=Math.max(z.h,r.h);z.l=Math.min(z.l,r.l);z.c=r.c;z.v+=r.v}}return[...m.values()].sort((a,b)=>a.t-b.t).filter(x=>x.t+ms<=Date.now())}
function xSwings(r,L=3,R=3){let hi=[],lo=[];for(let i=L;i<r.length-R;i++){let w=r.slice(i-L,i+R+1),x=r[i];if(x.h===Math.max(...w.map(q=>q.h))&&w.filter(q=>q.h===x.h).length===1)hi.push(x.h);if(x.l===Math.min(...w.map(q=>q.l))&&w.filter(q=>q.l===x.l).length===1)lo.push(x.l)}return{hi,lo}}
function xState(r){let s=xSwings(r);if(s.hi.length<2||s.lo.length<2)return 0;let hu=s.hi.at(-1)>s.hi.at(-2),lu=s.lo.at(-1)>s.lo.at(-2);return hu&&lu?1:!hu&&!lu?-1:0}
function xFeat(r){
  if(r.length<25)return null;let c=r.map(x=>x.c),v=r.map(x=>x.v),x=r.at(-1),p=r.at(-2),A=xAtr(r),ap=A/x.c*100,s=xSwings(r),sh=s.hi.at(-1)??null,sl=s.lo.at(-1)??null,state=xState(r),ef=xEma(c,9),es=xEma(c,21),slope=ef.length>3?(ef.at(-1)/ef.at(-4)-1)*100:0,spread=(ef.at(-1)-es.at(-1))/x.c*100,R=xRsi(c),samp=c.slice(-20),mu=xMean(samp),sd=xSd(samp),z=sd?(x.c-mu)/sd:0,bb=sd?xClamp((x.c-(mu-2*sd))/(4*sd),0,1):.5,pv=v.slice(-21,-1),vm=xMean(pv),vs=xSd(pv),vz=vs?(x.v-vm)/vs:0,rel=vm?x.v/vm:1,r1=p?(x.c/p.c-1)*100:0,r0=r.at(-3)?(p.c/r.at(-3).c-1)*100:0,acc=r1-r0,decel=Math.abs(r1)<Math.abs(r0),span=Math.max(x.h-x.l,1e-12),lower=(Math.min(x.o,x.c)-x.l)/span,upper=(x.h-Math.max(x.o,x.c))/span,b=A*.15,bs=!!(sl&&x.l<sl-b&&x.c>sl),br=!!(sh&&x.h>sh+b&&x.c<sh),fd=!!(sl&&p.c<sl-b&&x.c>sl),fu=!!(sh&&p.c>sh+b&&x.c<sh),ds=sl&&A?(x.c-sl)/A:null,dr=sh&&A?(sh-x.c)/A:null,nearS=ds!=null&&ds>=0&&ds<=.75,nearR=dr!=null&&dr>=0&&dr<=.75,pull=rel<.8&&((state===1&&x.c<x.o)||(state===-1&&x.c>x.o));
  let dp=[state===1?1:0,nearS?1:sl?0:null,lower,(bs||fd)?1:0,xClamp((50-R)/30,0,1),decel?1:0,(r0<0&&r1>0)?1:0,pull?1:0].filter(q=>q!=null),rp=[state===-1?1:0,nearR?1:sh?0:null,upper,(br||fu)?1:0,xClamp((R-50)/30,0,1),decel?1:0,(r0>0&&r1<0)?1:0,pull?1:0].filter(q=>q!=null);
  return{state,atrPct:ap,sh,sl,spread,slope,rsi:R,z,bb,vz,rel,r1,acc,decel,lower,upper,bs,br,fd,fu,ds,dr,dip:xMean(dp),rally:xMean(rp)};
}
function chartExpert(sym){
  let rows=xClosed(sym),f=xFeat(rows);if(!f)return null;let f5=xFeat(xAgg(rows,5)),f15=xFeat(xAgg(rows,15));
  let structure=.45*((f.bs||f.fd)?1:0)-.45*((f.br||f.fu)?1:0)+.20*f.state;if(f5&&f15){if(f5.state===f15.state&&f5.state)structure+=.15*f5.state;else if(f5.state*f15.state===-1)structure*=.65}structure=xClamp(structure,-1,1);
  let trend=Math.tanh(f.spread*2.2+f.slope*1.5)+.12*(f5?.state||0)+.18*(f15?.state||0);trend=xClamp(trend,-1,1);
  let momentum=.50*(f.dip-f.rally)+xClamp((f.rsi-50)/200,-.15,.15)+(f.acc>0?.12:f.acc<0?-.12:0)+(f.r1>0?.08:f.r1<0?-.08:0);momentum=xClamp(momentum,-1,1);
  let vi=Math.min(.20,Math.max(0,f.rel-1)*.10),volume=(f.r1>0?vi:f.r1<0?-vi:0)+(f.vz>=0?1:-1)*Math.min(.12,Math.abs(f.vz)*.04);if(f.rel>1.5&&f.lower>.5)volume+=.18;if(f.rel>1.5&&f.upper>.5)volume-=.18;volume=xClamp(volume,-1,1);
  let mr=xClamp(-f.z*.12,-.35,.35)+xClamp((.5-f.bb)*.30,-.18,.18);if(f.ds!=null&&f.ds>=0&&f.ds<=.5)mr+=.16;if(f.dr!=null&&f.dr>=0&&f.dr<=.5)mr-=.16;if(f.state)mr*=.65;mr=xClamp(mr,-1,1);
  let edge=xClamp(.32*structure+.24*trend+.18*momentum+.12*volume+.14*mr,-1,1),conf=xClamp(.42+.32*f.dip+.20*Math.max(0,edge)+.06*Math.max(0,1-Math.abs(f.z)/3),.35,.96),cfg=book.cfg||cfgUi(),cost=2*(N(cfg.fee_bps||0)+N(cfg.slippage_bps||0))/100,trigger=xClamp(f.atrPct*(.52+.14*Math.max(0,-edge)-.10*Math.max(0,edge)),.10,1.40),rebound=xClamp(f.atrPct*(.20+.12*(1-f.dip)),.04,.55),tp=xClamp(Math.max(cost*1.55,f.atrPct*(.70+.55*conf+.22*Math.max(0,edge))),.25,2.20),chase=xClamp(Math.max(rebound*3.2,f.atrPct*1.05),.22,2.80),m=metrics(),frac=xClamp(.04+.18*Math.pow(conf,1.7)+.035*Math.max(0,edge),.04,.28),fee=N(cfg.fee_bps||0)/1e4,not=Math.min(Math.max(0,book.cash/(1+fee)),Math.max(1,m.equity*frac));
  return{edge,confidence:conf,structure,trend,momentum,volume,meanRev:mr,dipScore:f.dip,rallyScore:f.rally,atrPct:f.atrPct,triggerPct:trigger,reboundPct:rebound,tpPct:tp,chasePct:chase,notional:not,rsi:f.rsi,referenceLow:f.sl,closedAt:rows.at(-1)?.t||0};
}
function seedExpert(sym,newSession){
  let st=expertHydrate(states[sym]),r=xClosed(sym),e=chartExpert(sym);if(!r.length||!e)return false;let w=r.slice(-120),low=Math.min(...w.map(x=>x.l)),high=Math.max(...w.map(x=>x.h)),sp=N(live[sym]||r.at(-1).c);
  st.expert=e;st.expertUpdatedAt=Date.now();st.baselineLow=low;st.baselineHigh=high;st.referenceLow=e.referenceLow||low;st.warmupReady=r.length>=60;
  if(newSession||st.startPrice==null){st.startPrice=sp;st.anchorHigh=Math.max(sp,...r.slice(-20).map(x=>x.h));st.postStartLow=sp;st.startClosedAt=e.closedAt;st.freshClosedSeen=false;st.dip=null;st.armed=false;st.lastDipLog=null;st.lastAction='WAIT_NEW_LOW'}
  return st.warmupReady;
}
function seedExperts(newSession){COINS.forEach(s=>seedExpert(s,newSession));}
function refreshExpert(sym){let e=chartExpert(sym);if(e){states[sym].expert=e;states[sym].referenceLow=e.referenceLow||states[sym].referenceLow;states[sym].expertUpdatedAt=Date.now()}return e}

cfgUi=function(){let sy=[...document.querySelectorAll('.symbolToggle.active')].map(x=>x.dataset.symbol);if(!sy.length)throw Error('En az bir coin seç.');return{symbols:sy,interval:'1m',step_pct:N($('stepPctInput').value)||.15,dip_trigger_steps:N($('dipTriggerInput').value)||2,rebound_steps:N($('reboundInput').value)||1,take_profit_steps:N($('takeProfitInput').value)||3,max_chase_steps:N($('maxChaseInput').value)||5,fee_bps:N($('feeInput').value)||10,slippage_bps:N($('slippageInput').value)||1,expert_mode:true,engine_version:DIP_EXPERT_V2};};
params=function(){let eq=N($('capitalInput').value);if(!(eq>0))throw Error('Kasa geçersiz.');$('tradeInput').value=String(eq);return{starting_equity:eq,trade_notional:eq,config:cfgUi(),engine_token:token()};};
score=function(st){let e=st?.expert;if(!e)return st?.warmupReady?12:0;return Math.round(xClamp(25+45*e.confidence+25*Math.max(0,e.edge)+10*e.dipScore,0,100));};

buy=function(st,p){
  let e=refreshExpert(st.symbol)||st.expert,c=book.cfg||cfgUi();if(!e)return;let fee=N(c.fee_bps||0)/1e4,slip=N(c.slippage_bps||0)/1e4,not=Math.min(e.notional,book.cash/(1+fee));if(not<.5){event('NO_CASH',st.symbol,p,{metadata:{expert_v2:true,available:book.cash}});return}let fill=p*(1+slip),entryFee=not*fee,qty=not/fill;book.cash-=not+entryFee;st.pos={entry:fill,qty,notional:not,entryFee,at:iso(),dip:st.dip,target:fill*(1+e.tpPct/100),targetPct:e.tpPct,maxSinceEntry:p};st.maxSinceEntry=p;st.lastAction='LONG';event('BUY',st.symbol,p,{entry_price:fill,quantity:qty,notional:not,fees:entryFee,metadata:{expert_v2:true,adaptive_notional:not,adaptive_target_pct:e.tpPct,adaptive_rebound_pct:e.reboundPct,expert_edge:e.edge,expert_confidence:e.confidence,dip_score:e.dipScore,reference_low:st.referenceLow}});render();
};
sell=function(st,p,reason='EXPERT_TARGET'){
  let q=st.pos;if(!q)return;let c=book.cfg||cfgUi(),fee=N(c.fee_bps||0)/1e4,slip=N(c.slippage_bps||0)/1e4,fill=p*(1-slip),gross=q.qty*fill,exitFee=gross*fee,net=gross-exitFee,r=net-(q.notional+q.entryFee);book.cash+=net;book.realized+=r;book.trades++;r>=0?book.wins++:book.losses++;st.realized+=r;st.pos=null;st.armed=false;st.dip=null;st.postStartLow=p;st.anchorHigh=p;st.startPrice=p;st.startClosedAt=xClosed(st.symbol).at(-1)?.t||Date.now();st.freshClosedSeen=false;st.lastAction='WAIT_NEW_LOW';event('SELL',st.symbol,p,{entry_price:q.entry,exit_price:fill,quantity:q.qty,notional:q.notional,fees:q.entryFee+exitFee,realized_pnl:r,metadata:{expert_v2:true,exit_reason:reason,target_pct:q.targetPct}});render();
};
tick=function(sym,p){
  p=N(p);if(!(p>0))return;live[sym]=p;let st=expertHydrate(states[sym]);st.last=p;if(!running||expertBooting||!sid||!book.cfg.symbols.includes(sym))return;let e=st.expert;if(!e||Date.now()-st.expertUpdatedAt>65000)e=refreshExpert(sym);if(!e||!st.warmupReady)return;
  let latestClosed=xClosed(sym).at(-1)?.t||0;if(st.startClosedAt!=null&&latestClosed>st.startClosedAt)st.freshClosedSeen=true;
  if(st.pos){st.maxSinceEntry=Math.max(st.maxSinceEntry||p,p);let gain=(p/st.pos.entry-1)*100,goal=st.pos.targetPct||e.tpPct,progress=goal>0?gain/goal:0,pull=(st.maxSinceEntry-p)/st.maxSinceEntry*100;if(p>=st.pos.target)return sell(st,p,'EXPERT_TARGET');if(progress>=.60&&e.edge<-.08&&pull>=Math.max(.05,e.atrPct*.18))return sell(st,p,'EXPERT_MOMENTUM_FADE');return}
  st.anchorHigh=Math.max(st.anchorHigh||p,p);if(st.postStartLow==null)st.postStartLow=p;
  if(p<st.postStartLow)st.postStartLow=p;
  if(!st.freshClosedSeen){st.lastAction='WAIT_NEW_CANDLE';return}
  if(!st.armed){let drop=(st.anchorHigh-p)/st.anchorHigh*100,belowStart=p<(st.startPrice||p),nearRef=st.referenceLow? p<=st.referenceLow*(1+Math.min(.003,e.atrPct/100*.35)):false;if(belowStart&&(drop>=e.triggerPct||nearRef)&&p<=st.postStartLow){st.armed=true;st.dip=p;st.lastDipLog=p;st.lastAction='DIP_HUNT';event('DIP_ARMED',sym,p,{metadata:{expert_v2:true,anchor_high:st.anchorHigh,reference_low:st.referenceLow,adaptive_trigger_pct:e.triggerPct,expert_edge:e.edge,expert_confidence:e.confidence,dip_score:e.dipScore}})}return}
  if(p<st.dip){let prev=st.dip;st.dip=p;st.postStartLow=p;if((prev-p)/prev*100>=Math.max(.01,e.atrPct*.05)){event('DIP_NEW_LOW',sym,p,{metadata:{expert_v2:true,expert_edge:e.edge,expert_confidence:e.confidence,dip_score:e.dipScore}})}return}
  let reb=(p/st.dip-1)*100;if(reb>=e.chasePct){event('SKIP_CHASE',sym,p,{metadata:{expert_v2:true,rebound_pct:reb,chase_pct:e.chasePct}});st.armed=false;st.dip=null;st.startPrice=p;st.anchorHigh=p;st.postStartLow=p;st.startClosedAt=latestClosed;st.freshClosedSeen=false;st.lastAction='WAIT_NEW_LOW';return}if(reb>=e.reboundPct&&e.confidence>=.42&&e.dipScore>=e.rallyScore-.12)buy(st,p);
};

historyLoad=async function(){await _baseHistoryLoad();expertHistoryReady=COINS.every(s=>xClosed(s).length>=25);if(expertHistoryReady)seedExperts(false);render();return expertHistoryReady;};
start=async function(restart=false){
  expertBooting=true;try{if(!expertHistoryReady)await historyLoad();if(!expertHistoryReady)throw Error('Binance geçmişi henüz hazır değil.');let p=params(),d=await api(restart?'restart':'start',p),cfg={...p.config,...(d.config||{}),expert_mode:true,engine_version:DIP_EXPERT_V2};session={session_id:d.session_id,status:'RUNNING',started_at:iso(),starting_equity:d.starting_equity,trade_notional:d.trade_notional,config:cfg};sid=d.session_id;reset(d.starting_equity,cfg);history=[];seedExperts(true);running=true;event('ENGINE_START',null,null,{metadata:{expert_v2:true,starting_equity:d.starting_equity,baseline_bars:120,rule:'wait_new_closed_candle_then_fresh_low'}});await snapshot();render();toast(restart?'Expert V2 temiz kasa ile başladı.':'Chart Expert V2 başladı; yeni mum + yeni dip bekliyor.')}catch(e){toast(String(e.message||e))}finally{expertBooting=false;render()}
};

note=function(e){let m=e.metadata||{};if(m.expert_v2){if(e.event_kind==='ENGINE_START')return'Expert V2 · 120m baseline · yeni kapalı mum bekler';if(e.event_kind==='DIP_ARMED')return`AI dip · trigger ${N(m.adaptive_trigger_pct||0).toFixed(2)}% · conf ${Math.round(N(m.expert_confidence||0)*100)}%`;if(e.event_kind==='DIP_NEW_LOW')return`Yeni dip · edge ${N(m.expert_edge||0).toFixed(2)}`;if(e.event_kind==='BUY')return`AI ${cash(m.adaptive_notional)} · hedef +${N(m.adaptive_target_pct||0).toFixed(2)}%`;if(e.event_kind==='SELL')return`${m.exit_reason||'EXPERT SELL'} · cost dahil`;if(e.event_kind==='SKIP_CHASE')return`Sıçrama ${N(m.rebound_pct||0).toFixed(2)}% · kovalanmadı`}return _baseNote(e);};
renderRadar=function(){let c=book.cfg||cfgUi();$('coinRadar').innerHTML=COINS.map(s=>{let st=expertHydrate(states[s]),e=st.expert,state=st.pos?'LONG':st.armed?'DIP HUNT':!st.warmupReady?'WARMUP':st.lastAction==='WAIT_NEW_CANDLE'?'WAIT CANDLE':'WATCH LOW',cl=st.pos?'long':st.armed?'hunt':state==='WARMUP'?'warm':'';let detail=e?`Ref ${price(st.referenceLow)} · ATR ${e.atrPct.toFixed(2)}%<br>AI dip ${e.triggerPct.toFixed(2)}% · al +${e.reboundPct.toFixed(2)}% · sat +${e.tpPct.toFixed(2)}% · ${cash(e.notional)}`:'Binance geçmişi hazırlanıyor';return`<div class="coinCard ${s===selected?'selected':''}" data-s="${s}"><div><div class="coinTop"><span class="coinSymbol">${s.replace('USDT','')}</span><span class="coinState ${cl}">${state}</span></div><div class="coinMeta">${st.pos?`Dip ${price(st.pos.dip)} · Buy ${price(st.pos.entry)}<br>`:''}${detail}</div><div class="scoreBar"><i style="width:${score(st)}%"></i></div><div class="scoreLabel">Chart Expert ${score(st)}/100 · edge ${e?e.edge.toFixed(2):'—'} · RSI ${e?e.rsi.toFixed(0):'—'}</div></div><div class="coinRight"><div class="coinPrice">${price(live[s])}</div>${st.realized?`<div class="coinMove ${st.realized>=0?'pos':'neg'}">${pnl(st.realized)}</div>`:''}</div></div>`}).join('');document.querySelectorAll('[data-s]').forEach(x=>x.onclick=()=>{selected=x.dataset.s;renderRadar();draw()})};
renderKpi=function(){_baseRenderKpi();if(running){$('kpiEngine').textContent='EXPERT V2';$('kpiEngine').className='value pos';$('kpiEngineMeta').textContent='Adaptive giriş · çıkış · bütçe'}};
render=function(){_baseRender();if($('startBtn'))$('startBtn').disabled=Boolean(session?.status==='RUNNING')||!expertHistoryReady;if($('radarStatus'))$('radarStatus').textContent=running?'EXPERT V2':'WAIT';if($('ruleRebound'))$('ruleRebound').textContent='AI';if($('ruleTp'))$('ruleTp').textContent='AI';if(session&&$('sessionBanner'))$('sessionBanner').textContent=`${session.status} · ${cash(session.starting_equity)} · AI bütçe yönetimi`;renderKpi();};

// Existing websocket function calls tick dynamically. Refresh the expert on every completed 1m bar.
const _baseConnect=connect;
connect=function(){_baseConnect();let timer=setInterval(()=>{if(!ws||ws.readyState>1){clearInterval(timer);return}COINS.forEach(s=>{let st=expertHydrate(states[s]),last=xClosed(s).at(-1)?.t||0;if(last&&last>(st.expert?.closedAt||0)){refreshExpert(s);if(!st.warmupReady)seedExpert(s,false)}})},3000)};

// Make the AI-owned budget field clearly read-only as soon as this overlay loads.
if($('tradeInput')){$('tradeInput').disabled=true;$('tradeInput').setAttribute('aria-label','Chart Expert V2 kullanılabilir bütçe üst sınırı')}
