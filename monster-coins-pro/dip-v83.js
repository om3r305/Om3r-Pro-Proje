'use strict';

const TRADER_API='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-dip-trader';
const CHART_API='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-dip-chart';
const KEY_NAME='mcp-dashboard-key-v1';
const TOKEN_NAME='mcp-dip-engine-token-v1';
const $=id=>document.getElementById(id);
const esc=v=>String(v??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
const num=(v,f=0)=>Number.isFinite(Number(v))?Number(v):f;
const money=v=>`$${num(v).toFixed(2)}`;
const pnl=v=>`${num(v)>=0?'+':''}$${num(v).toFixed(3)}`;
const px=v=>{const n=Number(v);return n>0?n.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}):'—'};
const berlinTime=v=>v?new Intl.DateTimeFormat('de-DE',{timeZone:'Europe/Berlin',hour:'2-digit',minute:'2-digit',second:'2-digit'}).format(new Date(v)):'—';

let model={session:null,snapshot:null,events:[],statusError:null,chart:null,chartError:null,statusAt:0,chartAt:0};
let statusBusy=false,chartBusy=false;

function dashboardKey(){return localStorage.getItem(KEY_NAME)||'';}
function engineToken(){let t=localStorage.getItem(TOKEN_NAME);if(!t){t=`dip-${crypto.randomUUID()}-${crypto.randomUUID()}`;localStorage.setItem(TOKEN_NAME,t);}return t;}
function toast(text,kind='ok'){const el=$('toast');el.textContent=text;el.className=`toast show ${kind}`;clearTimeout(el._t);el._t=setTimeout(()=>el.className='toast',3200);}
function showLock(show){$('unlock').classList.toggle('show',show);}

async function trader(action,body={}){
  const key=dashboardKey();if(!key){showLock(true);throw new Error('Dashboard anahtarı gerekli.');}
  const r=await fetch(TRADER_API,{method:'POST',cache:'no-store',headers:{'content-type':'application/json','x-brian-dashboard-key':key},body:JSON.stringify({action,...body})});
  const d=await r.json().catch(()=>({}));
  if(r.status===401){localStorage.removeItem(KEY_NAME);showLock(true);}
  if(!r.ok)throw new Error(d.error||d.status||`HTTP ${r.status}`);
  return d;
}

async function loadStatus(){
  if(statusBusy||document.hidden)return;statusBusy=true;
  try{
    const d=await trader('status');
    model.session=d.session||null;model.snapshot=d.snapshot||null;model.events=Array.isArray(d.events)?d.events:[];model.statusError=null;model.statusAt=Date.now();
  }catch(e){model.statusError=String(e.message||e);}finally{statusBusy=false;render();}
}

async function loadChart(){
  if(chartBusy||document.hidden)return;chartBusy=true;
  try{
    const r=await fetch(`${CHART_API}?limit=180`,{cache:'no-store'});const d=await r.json().catch(()=>({}));if(!r.ok)throw new Error(d.error||`HTTP ${r.status}`);
    if(!Array.isArray(d.candles)||d.candles.length<20)throw new Error('USD-M mum verisi yetersiz');
    model.chart=d;model.chartError=null;model.chartAt=Date.now();
  }catch(e){model.chartError=String(e.message||e);}finally{chartBusy=false;renderChart();renderHealth();}
}

function thesis(){const z=model.snapshot;return z?.state?.symbols?.ETHUSDT?.thesis||z?.state?.v8?.latestThesis||null;}
function runtime(){return model.snapshot?.state?.serverRuntime||null;}
function position(){return model.snapshot?.state?.v8?.pos||model.snapshot?.state?.symbols?.ETHUSDT?.pos||null;}
function workerFresh(){const sr=runtime();const t=Date.parse(sr?.generated_at||model.snapshot?.observed_at||0);return Boolean(model.session?.status==='RUNNING'&&sr?.status==='OK'&&sr?.authoritative===true&&sr?.market_source==='BINANCE_USDM_PERP'&&!sr?.market_error&&t&&Date.now()-t<150000);}
function chartFresh(){return Boolean(model.chart&&!model.chartError&&Date.now()-model.chartAt<12000);}

function renderHealth(){
  const wf=workerFresh(),cf=chartFresh(),sr=runtime();
  $('sessionState').textContent=model.session?.status||'IDLE';$('sessionState').className=model.session?.status==='RUNNING'?'good':'warn';
  $('sessionMeta').textContent=model.session?`Başlangıç ${berlinTime(model.session.started_at)} · ${model.session.session_id}`:'Session yok';
  $('feedState').textContent=wf&&cf?'SERVER LIVE':wf?'CHART WAIT':'SERVER WAIT';$('feedState').className=wf&&cf?'good':'warn';
  $('feedMeta').textContent=wf&&cf?`USD-M PERP · chart ${Math.round((Date.now()-model.chartAt)/1000)} sn · karar ${num(sr?.decision_cadence_seconds,60)} sn`:(model.chartError||sr?.market_error||'heartbeat bekleniyor');
  $('cloudState').textContent=wf?'BULUT V8.3':'WAIT';$('cloudState').className=wf?'good':'warn';
  $('cloudMeta').textContent=sr?.worker_version||model.statusError||'server-authoritative';
  $('topStatus').textContent=wf?'V8.3 RUNNING':'V8.3 WAIT';$('topStatus').className=`pill ${wf?'good':'warn'}`;
}

function renderThesis(){
  const t=thesis(),host=$('thesisBox');
  if(!t){host.innerHTML='<div class="thesis-main"><b>BRIAN V8.3 · SERVER THESIS</b><span class="dir wait">WAIT</span><span>Server henüz ilk kararını üretmedi.</span></div>';return;}
  const dir=t.direction==='UP'?'YUKARI':t.direction==='DOWN'?'AŞAĞI':'WAIT';
  const dc=t.direction==='UP'?'up':t.direction==='DOWN'?'down':'wait';
  const raw=t.raw_conviction==null?'—':`${Math.round(num(t.raw_conviction)*100)}/100`;
  const cal=t.calibrated_probability==null?`CALIBRATING · n=${num(t.calibration_samples||t.calibration?.samples)}`:`${Math.round(num(t.calibrated_probability)*100)}% · n=${num(t.calibration_samples)}`;
  const veto=Array.isArray(t.veto)&&t.veto.length?t.veto.join(' · '):'YOK';
  const why=Array.isArray(t.why)&&t.why.length?t.why.join(' · '):'—';
  const target=t.target??t.target_price,stop=t.invalidation??t.invalidation_price,rr=t.economic_rr??t.rr;
  host.innerHTML=`<div class="thesis-main"><b>BRIAN V8.3 · SERVER THESIS</b><span class="dir ${dc}">${dir}</span><span>Setup <strong>${esc(t.setup||'NONE')}</strong></span><span>Rejim <strong>${esc(t.regime||'—')}</strong></span><span>Ham görüş <strong>${raw}</strong></span><span>Kalibrasyon <strong>${esc(cal)}</strong></span></div><div class="thesis-levels"><span>Entry <b>${px(t.entry_low)} – ${px(t.entry_high)}</b></span><span>Hedef <b>${px(target)}</b></span><span>İptal <b>${px(stop)}</b></span><span>Econ R <b>${num(rr).toFixed(2)}</b></span><small>Veto: ${esc(veto)}<br>Yapı: ${esc(why)}</small></div>`;
}

function renderKpis(){
  const z=model.snapshot||{},rt=z.state?.v8||{},cashVal=num(z.cash??rt.cash??model.session?.starting_equity??500),real=num(z.realized_pnl??rt.realized),trades=num(z.trade_count??rt.trades),wins=num(z.win_count??rt.wins),losses=num(z.loss_count??rt.losses),closed=wins+losses,pos=position();
  $('kpiEquity').textContent=money(cashVal);$('kpiPnl').textContent=pnl(real);$('kpiPnl').className=`value ${real>0?'pos':real<0?'neg':''}`;$('kpiOpen').textContent=pos?1:0;$('kpiWin').textContent=closed?`${(wins/closed*100).toFixed(1)}%`:'—';$('kpiTrades').textContent=String(trades);$('kpiEngine').textContent=workerFresh()?'BRIAN V8.3 DUAL':'V8.3 WAIT';$('kpiEngine').className=`value ${workerFresh()?'pos':'amber'}`;
  $('kpiEquityMeta').textContent=`Başlangıç ${money(model.session?.starting_equity??cashVal)} · kullanılabilir üst sınır ${money(model.session?.trade_notional??cashVal)}`;
  $('kpiPnlMeta').textContent='fee + slippage + spread/funding dahil';$('kpiOpenMeta').textContent=pos?`${pos.side} · ${num(pos.leverage,1)}x · notional ${money(pos.notional)}`:'Pozisyon yok · risk ≤ %0.5/trade';$('kpiWinMeta').textContent=`${wins} win / ${losses} loss`;$('kpiTradesMeta').textContent='kapalı round trips';$('kpiEngineMeta').textContent=`USD-M PERP · ${num(runtime()?.decision_cadence_seconds,60)} sn karar`;
}

function eventKind(e){return String(e?.event_kind||'—');}
function renderLog(){
  const rows=model.events||[];const host=$('eventRows');
  if(!rows.length){host.innerHTML='<tr><td colspan="8">Bu V8.3 session’da henüz LONG / SHORT açılış-kapanış olayı yok.</td></tr>';return;}
  host.innerHTML=rows.slice(0,80).map(e=>{const k=eventKind(e),rp=e.realized_pnl,side=k.includes('SHORT')?'short':k==='BUY'?'buy':k==='SELL'?'sell':'neutral',m=e.metadata||{};return `<tr><td>${berlinTime(e.observed_at)}</td><td><span class="tag ${side}">${esc(k)}</span></td><td>ETH</td><td>${px(e.price)}</td><td>${px(e.entry_price)}</td><td>${px(e.exit_price)}</td><td class="${num(rp)>0?'pos':num(rp)<0?'neg':''}">${rp==null?'—':pnl(rp)}</td><td>${esc(m.setup||m.exit_reason||'')}</td></tr>`;}).join('');
}

function drawLine(ctx,y,L,R,w,label,value,color,dash=[5,4],right=false){const v=Number(value);if(!(v>0))return;const yy=y(v);if(!Number.isFinite(yy))return;ctx.save();ctx.setLineDash(dash);ctx.strokeStyle=color;ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.setLineDash([]);ctx.font='700 10px system-ui';const text=`${label} ${px(v)}`,tw=ctx.measureText(text).width,x=right?Math.max(L+4,w-R-tw-8):L+6;ctx.fillStyle='rgba(5,10,18,.9)';ctx.fillRect(x-3,yy-15,tw+7,15);ctx.fillStyle=color;ctx.fillText(text,x,yy-4);ctx.restore();}
function pivot(s,k){const x=s?.[k];return Number(x?.p)>0?Number(x.p):null;}

function renderChart(){
  const cv=$('candleCanvas'),box=$('chartWrap');if(!cv||!box)return;const ctx=cv.getContext('2d'),dpr=window.devicePixelRatio||1,w=Math.max(500,box.clientWidth),h=Math.max(360,box.clientHeight);cv.width=w*dpr;cv.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#07101a';ctx.fillRect(0,0,w,h);
  const raw=model.chart?.candles||[];const a=raw.slice(-110).map(c=>({t:num(c.t),o:num(c.o),h:num(c.h),l:num(c.l),c:num(c.c),v:num(c.v)}));if(!a.length){ctx.fillStyle='#91a2b8';ctx.font='14px system-ui';ctx.fillText(model.chartError||'USD-M mumları bekleniyor…',20,35);return;}
  const t=thesis()||{},S=t.structure||{},s1=S.s1||S['1m']||{},s5=S.s5||S['5m']||{},pos=position();
  const target=pos?.target??t.target??t.target_price,stop=pos?.stop??t.invalidation??t.invalidation_price,entry=pos?.entry??t.entry_low;
  const levels=[target,stop,entry,pivot(s1,'lastHigh'),pivot(s1,'lastLow'),pivot(s5,'lastHigh'),pivot(s5,'lastLow')].map(Number).filter(v=>v>0);
  const vals=a.flatMap(x=>[x.l,x.h]).concat(levels);let lo=Math.min(...vals),hi=Math.max(...vals),pad=(hi-lo)*.08||1;lo-=pad;hi+=pad;
  const L=15,R=92,T=25,B=38,plotW=w-L-R,plotH=h-T-B,xw=plotW/a.length,y=v=>T+(hi-v)/(hi-lo)*plotH;
  ctx.font='10px system-ui';for(let i=0;i<=5;i++){const yy=T+plotH*i/5,v=hi-(hi-lo)*i/5;ctx.strokeStyle='#172334';ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.fillStyle='#718198';ctx.fillText(px(v),w-R+8,yy+3);}
  a.forEach((c,i)=>{const x=L+xw*i+xw/2,up=c.c>=c.o,col=up?'#12d996':'#ff5068';ctx.strokeStyle=col;ctx.fillStyle=col;ctx.beginPath();ctx.moveTo(x,y(c.h));ctx.lineTo(x,y(c.l));ctx.stroke();const top=y(Math.max(c.o,c.c)),bot=y(Math.min(c.o,c.c));ctx.fillRect(x-Math.max(1,xw*.28),top,Math.max(2,xw*.56),Math.max(1,bot-top));});
  if(Number(t.entry_low)>0&&Number(t.entry_high)>0){drawLine(ctx,y,L,R,w,'ENTRY',t.entry_low,'#6da3ff',[2,3]);if(Math.abs(num(t.entry_high)-num(t.entry_low))>.001)drawLine(ctx,y,L,R,w,'ENTRY HIGH',t.entry_high,'#6da3ff',[2,3]);}
  if(target)drawLine(ctx,y,L,R,w,pos?'POZİSYON HEDEF':'ADAY HEDEF',target,'#19d69a',[6,4],true);if(stop)drawLine(ctx,y,L,R,w,pos?'POZİSYON İPTAL':'ADAY İPTAL',stop,'#ff6379',[6,4],true);
  const h1=pivot(s1,'lastHigh'),l1=pivot(s1,'lastLow'),h5=pivot(s5,'lastHigh'),l5=pivot(s5,'lastLow');if(h1)drawLine(ctx,y,L,R,w,`1m ${s1.lastHigh?.label||'H'}`,h1,'#92a7c8',[1,5]);if(l1)drawLine(ctx,y,L,R,w,`1m ${s1.lastLow?.label||'L'}`,l1,'#92a7c8',[1,5]);if(h5)drawLine(ctx,y,L,R,w,`5m ${s5.lastHigh?.label||'H'}`,h5,'#ae89ff',[3,6]);if(l5)drawLine(ctx,y,L,R,w,`5m ${s5.lastLow?.label||'L'}`,l5,'#ae89ff',[3,6]);
  const last=num(model.chart?.last_price||a.at(-1).c);if(last>0){const yy=Math.max(T+10,Math.min(h-B-10,y(last))),up=a.at(-1).c>=a.at(-1).o,col=up?'#12d996':'#ff5068';ctx.setLineDash([3,3]);ctx.strokeStyle=col;ctx.globalAlpha=.65;ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.globalAlpha=1;ctx.setLineDash([]);ctx.fillStyle=col;ctx.fillRect(w-R+3,yy-11,R-6,22);ctx.fillStyle='#fff';ctx.font='700 10px system-ui';ctx.fillText(px(last),w-R+9,yy+4);}
  ctx.fillStyle='#a8b6ca';ctx.font='700 10px system-ui';ctx.fillText(`1m ${s1.trend||'—'} · BOS ${s1.bos||'—'} · CHOCH ${s1.choch||'—'} · SWEEP ${s1.sweep||'—'}  |  5m ${s5.trend||'—'}`,L+4,15);
  for(let i=0;i<a.length;i+=22){const x=L+xw*i+xw/2;ctx.fillStyle='#66778e';ctx.font='9px system-ui';ctx.fillText(berlinTime(a[i].t),x-18,h-12);}
  $('lastPrice').textContent=px(last);$('chartSource').textContent='SERVER USD-M';
}

function render(){renderHealth();renderThesis();renderKpis();renderLog();renderChart();const running=model.session?.status==='RUNNING';if(model.session?.starting_equity>0)$('capitalInput').value=String(model.session.starting_equity);$('startBtn').disabled=running;$('pauseBtn').disabled=!running;$('capitalInput').disabled=running;}

async function start(restart=false){
  try{const capital=num($('capitalInput').value,500);if(!(capital>=10&&capital<=1e6))throw new Error('Kasa 10–1.000.000 USDT arasında olmalı.');const config={symbols:['ETHUSDT'],interval:'1m',fee_bps:10,slippage_bps:1,execution_mode:'SHADOW_PAPER',shadow_only:true,live_execution:false,browser_execution:false,server_authoritative:true,allow_shadow_short:true,max_shadow_leverage:2,market_source:'BINANCE_USDM_PERP',decision_cadence_seconds:60};await trader(restart?'restart':'start',{starting_equity:capital,trade_notional:capital,config,engine_token:engineToken()});toast(restart?'Temiz V8.3 session açıldı.':'V8.3 session başladı.');await loadStatus();}catch(e){toast(String(e.message||e),'err');}
}
async function pause(){try{await trader('pause');toast('V8.3 session pause edildi.');await loadStatus();}catch(e){toast(String(e.message||e),'err');}}

function bind(){
  $('unlockBtn').onclick=async()=>{const k=$('unlockKey').value.trim();if(!k)return;localStorage.setItem(KEY_NAME,k);$('unlockKey').value='';showLock(false);await loadStatus();await loadChart();};
  $('startBtn').onclick=()=>start(false);$('restartBtn').onclick=()=>start(true);$('pauseBtn').onclick=pause;window.addEventListener('resize',renderChart);
}

async function init(){
  bind();if(!dashboardKey()){showLock(true);}else showLock(false);
  await Promise.allSettled([loadStatus(),loadChart()]);
  setInterval(loadChart,2000);setInterval(loadStatus,5000);setInterval(renderHealth,1000);
}
window.addEventListener('load',init);
