/* Brian DIP V7 Focus-3 + Foresight UI.
   SHADOW ONLY. This file is visualization/selection only; it never owns execution. */
const V7_FOCUS_UNIVERSE=['XRPUSDT','ETHUSDT','DOGEUSDT'];
const V7_FORESIGHT_API='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-dip-foresight';
let v7ForesightBySymbol={};
let v7ForesightTimer=null;
let v7ForesightBusy=false;

v4DiscoverUniverse=async function(){
  v4Universe=[...V7_FOCUS_UNIVERSE];
  v4UniverseUpdatedAt=Date.now();
  v4Universe.forEach(v4Ensure);
  return v4Universe;
};

async function v7FetchForesight(){
  if(v7ForesightBusy||document.visibilityState==='hidden')return;
  const key=dashboardKey();if(!key)return;
  v7ForesightBusy=true;
  try{
    const r=await fetch(V7_FORESIGHT_API,{method:'POST',headers:{'content-type':'application/json','x-brian-dashboard-key':key},body:JSON.stringify({session_id:sid||null})});
    const d=await r.json().catch(()=>({}));
    if(!r.ok)throw Error(d.error||d.status||`HTTP ${r.status}`);
    v7ForesightBySymbol=d.forecasts||{};
    for(const sym of V7_FOCUS_UNIVERSE){
      const f=v7ForesightBySymbol[sym];
      if(f){const st=v4Ensure(sym);st.v4.foresight=f;}
    }
    renderForesightBar();draw();
  }catch(e){
    const el=$('v7ForesightStatus');if(el){el.textContent='FORECAST WAIT';el.className='v7ForesightStatus wait';}
  }finally{v7ForesightBusy=false;}
}

function v7Foresight(sym=selected){return states?.[sym]?.v4?.foresight||v7ForesightBySymbol?.[sym]||null;}
function v7DirText(f){return f?.direction==='UP'?'YUKARI':f?.direction==='DOWN'?'AŞAĞI':'YATAY';}
function v7ForesightStrength(f){
  const x=Number(f?.confidence||0)*100;
  return x>=75?'YÜKSEK':x>=58?'ORTA':'DÜŞÜK';
}
function renderForesightBar(){
  const bar=$('v7ForesightBar');if(!bar)return;
  const f=v7Foresight();
  if(!f){bar.innerHTML='<div><b>BRIAN İLERİ GÖRÜŞ</b><span>İlk server tahmini bekleniyor…</span></div><span id="v7ForesightStatus" class="v7ForesightStatus wait">WAIT</span>';return;}
  const conf=Math.round(Number(f.confidence||0)*100),acc=f.accuracy==null?'—':`${Math.round(Number(f.accuracy)*100)}%`,n=Number(f.samples||0);
  const dir=v7DirText(f),strength=v7ForesightStrength(f),cls=f.direction==='UP'?'up':f.direction==='DOWN'?'down':'flat';
  bar.innerHTML=`<div class="v7ForesightMain"><b>BRIAN İLERİ GÖRÜŞ · ${v4Esc(selected.replace('USDT',''))}</b><span class="${cls}">${dir}</span><span>Güven <strong>${conf}% · ${strength}</strong></span><span>İsabet <strong>${acc}${n?` (${n})`:''}</strong></span><span>Ufuk <strong>${Number(f.horizon_min||8)} dk</strong></span></div><div class="v7ForesightLevels"><span>Muhtemel tepe <b>${price(f.peak)}</b></span><span>Muhtemel dip <b>${price(f.trough)}</b></span><small>Tahmin alanı; garanti fiyat değildir.</small></div>`;
}

function v7DrawLabel(ctx,x,y,text,color){
  ctx.save();ctx.font='700 9px system-ui';const tw=ctx.measureText(text).width;
  ctx.fillStyle='rgba(4,9,16,.90)';ctx.fillRect(x-3,y-10,tw+7,14);ctx.fillStyle=color;ctx.fillText(text,x,y);ctx.restore();
}

const _v7ForesightBaseDraw=draw;
draw=function(){
  const cv=$('candleCanvas'),box=$('chartWrap');if(!cv||!box)return;
  const ctx=cv.getContext('2d'),dpr=devicePixelRatio||1,w=Math.max(320,box.clientWidth),h=Math.max(280,box.clientHeight);
  cv.width=w*dpr;cv.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#080e17';ctx.fillRect(0,0,w,h);
  const f=v7Foresight(),future=Array.isArray(f?.candles)?f.candles.slice(0,10):[],a=(candles[selected]||[]).slice(-88);if(!a.length)return;
  const st=states[selected]||{},q=st.pos||null,dipInfo=typeof v7LatestDip==='function'?v7LatestDip(selected):null;
  const values=a.flatMap(c=>[Number(c.l),Number(c.h)]).filter(Number.isFinite),lp=Number(live[selected]||a.at(-1)?.c||0),levels=[];
  if(dipInfo?.price>0&&(!lp||Math.abs(dipInfo.price/lp-1)<=.12))levels.push(dipInfo.price);
  if(q)for(const v of [q.entry,q.stop,q.target])if(Number(v)>0)levels.push(Number(v));
  for(const c of future)for(const v of [c.l,c.h])if(Number(v)>0)levels.push(Number(v));
  for(const v of [f?.peak,f?.trough])if(Number(v)>0)levels.push(Number(v));
  let lo=Math.min(...values,...levels),hi=Math.max(...values,...levels),pad=(hi-lo)*.075||1;lo-=pad;hi+=pad;
  const L=12,R=82,T=12,B=28,totalSlots=a.length+Math.max(4,future.length),plotW=w-L-R,plotH=h-T-B,xw=plotW/totalSlots,y=v=>T+(hi-v)/(hi-lo)*plotH;
  ctx.lineWidth=1;ctx.font='9px system-ui';
  for(let i=0;i<=5;i++){const yy=T+plotH*i/5,v=hi-(hi-lo)*i/5;ctx.strokeStyle='#182235';ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.fillStyle='#728096';ctx.fillText(price(v),w-R+8,yy+3);}
  a.forEach((c,i)=>{const x=L+xw*i+xw/2,up=Number(c.c)>=Number(c.o),col=up?'#0ecb81':'#f6465d';ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,y(Number(c.h)));ctx.lineTo(x,y(Number(c.l)));ctx.stroke();const top=y(Math.max(Number(c.o),Number(c.c))),bot=y(Math.min(Number(c.o),Number(c.c)));ctx.fillRect(x-Math.max(1,xw*.27),top,Math.max(2,xw*.54),Math.max(1,bot-top));});
  const futureStart=L+xw*a.length;
  if(future.length){
    ctx.fillStyle='rgba(86,122,255,.045)';ctx.fillRect(futureStart,T,Math.max(0,w-R-futureStart),plotH);
    ctx.setLineDash([4,4]);ctx.strokeStyle='rgba(128,155,255,.55)';ctx.beginPath();ctx.moveTo(futureStart,T);ctx.lineTo(futureStart,h-B);ctx.stroke();ctx.setLineDash([]);v7DrawLabel(ctx,futureStart+6,T+14,'BRAIN TAHMİN ALANI','#8aa6ff');
    future.forEach((c,j)=>{const i=a.length+j,x=L+xw*i+xw/2,up=Number(c.c)>=Number(c.o),col=up?'#55dca8':'#ff788c';ctx.globalAlpha=.52;ctx.strokeStyle=col;ctx.fillStyle=col;ctx.beginPath();ctx.moveTo(x,y(Number(c.h)));ctx.lineTo(x,y(Number(c.l)));ctx.stroke();const top=y(Math.max(Number(c.o),Number(c.c))),bot=y(Math.min(Number(c.o),Number(c.c)));ctx.fillRect(x-Math.max(1,xw*.22),top,Math.max(2,xw*.44),Math.max(1,bot-top));ctx.globalAlpha=1;});
    if(Number(f?.peak)>0){ctx.setLineDash([2,4]);ctx.strokeStyle='#55dca8';ctx.beginPath();ctx.moveTo(futureStart,y(f.peak));ctx.lineTo(w-R,y(f.peak));ctx.stroke();ctx.setLineDash([]);v7DrawLabel(ctx,futureStart+8,y(f.peak)-4,`TAHMİN TEPE ${price(f.peak)}`,'#55dca8');}
    if(Number(f?.trough)>0){ctx.setLineDash([2,4]);ctx.strokeStyle='#ff788c';ctx.beginPath();ctx.moveTo(futureStart,y(f.trough));ctx.lineTo(w-R,y(f.trough));ctx.stroke();ctx.setLineDash([]);v7DrawLabel(ctx,futureStart+8,y(f.trough)-4,`TAHMİN DİP ${price(f.trough)}`,'#ff788c');}
  }
  if(dipInfo?.price>0&&typeof v7ChartLevel==='function')v7ChartLevel(ctx,y,L,R,w,dipInfo.active?'AKTİF DİP':'SON DİP',dipInfo.price,'#f0b90b',[4,4],'left');
  if(q&&typeof v7ChartLevel==='function'){const side=String(q.side||'LONG').toUpperCase();v7ChartLevel(ctx,y,L,R,w,side==='SHORT'?'SHORT GİRİŞ':'ALIM GİRİŞ',q.entry,side==='SHORT'?'#f6465d':'#0ecb81',[2,2],'left');v7ChartLevel(ctx,y,L,R,w,'TP',q.target,'#2af0a3',[6,4],'right');v7ChartLevel(ctx,y,L,R,w,'SL',q.stop,'#ff6b7a',[6,4],'right');}
  if(lp>0){const yy=Math.max(T+9,Math.min(h-B-9,y(lp))),up=Number(a.at(-1)?.c)>=Number(a.at(-1)?.o),col=up?'#0ecb81':'#f6465d';ctx.setLineDash([3,3]);ctx.strokeStyle=col;ctx.globalAlpha=.65;ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.globalAlpha=1;ctx.setLineDash([]);ctx.fillStyle=col;ctx.fillRect(w-R+3,yy-10,R-6,20);ctx.fillStyle='#fff';ctx.font='600 9px system-ui';ctx.fillText(price(lp),w-R+8,yy+3);}
  $('chartSymbol').textContent=selected;$('lastPrice').textContent=price(lp);$('chartSub').textContent=`Binance Spot · 1m · Focus 3 · gerçek mum + Brian tahmin alanı`;
  renderForesightBar();
};

const _v7ForesightUiPatch=v4UiPatch;
v4UiPatch=function(){
  _v7ForesightUiPatch();
  const p=document.querySelector('.symbolPicker .pickerTitle');if(p)p.innerHTML='<b>Brian Focus Lab · 3 Coin</b><span>XRP · ETH · DOGE — grafik okuma ve ileri görüş gelişimi</span>';
  const buttons=$('symbolButtons');if(buttons)buttons.innerHTML=V7_FOCUS_UNIVERSE.map(s=>`<button class="symbolToggle active" data-symbol="${s}">${s.replace('USDT','')}</button>`).join('');
  const expert=document.querySelector('.expertModeCard small');if(expert)expert.textContent='Focus 3: XRP · ETH · DOGE · multi-timeframe grafik uzmanı · foresight telemetry · SHADOW ONLY';
};

function v7InstallForesightUi(){
  if(!$('v7ForesightBar')){const panel=$('chartPanel'),head=panel?.querySelector('.chartHead');if(panel&&head){const d=document.createElement('div');d.id='v7ForesightBar';d.className='v7ForesightBar';head.insertAdjacentElement('afterend',d);}}
  if(!document.getElementById('v7-foresight-style')){const s=document.createElement('style');s.id='v7-foresight-style';s.textContent=`
  .v7ForesightBar{display:flex;justify-content:space-between;gap:12px;align-items:center;margin:0 12px 9px;padding:9px 11px;border:1px solid rgba(92,130,255,.24);background:linear-gradient(90deg,rgba(63,87,180,.10),rgba(16,24,39,.55));border-radius:10px;font-size:11px;color:#9baac0}.v7ForesightMain,.v7ForesightLevels{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.v7ForesightMain b{color:#dfe8ff}.v7ForesightMain .up{color:#28d99b}.v7ForesightMain .down{color:#ff647b}.v7ForesightMain .flat{color:#f0b90b}.v7ForesightLevels b{color:#e8eefb}.v7ForesightLevels small{color:#66758c}.v7ForesightStatus.wait{color:#f0b90b}@media(max-width:760px){.v7ForesightBar{align-items:flex-start;flex-direction:column}.v7ForesightMain,.v7ForesightLevels{gap:7px 11px}}
  `;document.head.appendChild(s);}
  v4UiPatch();renderForesightBar();
}

addEventListener('load',()=>{v7InstallForesightUi();v4Universe=[...V7_FOCUS_UNIVERSE];if(!V7_FOCUS_UNIVERSE.includes(selected))selected='XRPUSDT';setTimeout(()=>{v4LoadHistory().then(()=>{connect();render();v7FetchForesight();}).catch(()=>{});},250);if(v7ForesightTimer)clearInterval(v7ForesightTimer);v7ForesightTimer=setInterval(v7FetchForesight,15000);});
document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible')v7FetchForesight();});
