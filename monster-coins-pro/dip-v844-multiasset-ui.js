'use strict';

(()=>{
  const API='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-dip-multiasset-status';
  const KEY='mcp-dashboard-key-v1';
  const REST_HOSTS=['https://api.binance.com','https://data-api.binance.vision'];
  const el=id=>document.getElementById(id);
  const n=(v,f=0)=>Number.isFinite(Number(v))?Number(v):f;
  const esc=s=>String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
  const money=v=>`${n(v)>=0?'+':''}$${Math.abs(n(v)).toFixed(2)}`;
  const usd=v=>`$${n(v).toFixed(2)}`;
  const price=v=>{const x=n(v);if(!(x>0))return '—';const d=x<.001?8:x<1?6:x<100?4:2;return x.toLocaleString('en-US',{maximumFractionDigits:d,minimumFractionDigits:Math.min(2,d)});};
  const pct=v=>`${(n(v)*100).toFixed(1)}%`;
  const time=v=>{const d=new Date(String(v||''));return Number.isFinite(d.getTime())?d.toLocaleTimeString('tr-TR',{hour:'2-digit',minute:'2-digit',second:'2-digit'}):'—';};
  const safeSymbol=v=>{const s=String(v||'').toUpperCase();return /^[A-Z0-9]{2,20}USDT$/.test(s)?s:'';};
  const setText=(node,value)=>{if(node&&node.textContent!==value)node.textContent=value;};
  let latest=null,selected='',chart={symbol:'',candles:[],last:0,source:'WAIT',ws:null,reconnect:null,seq:0},summaryQueued=false,summaryApplying=false;

  function style(){
    if(el('multiDipStyle'))return;
    const s=document.createElement('style');s.id='multiDipStyle';s.textContent=`
      #multiDipPanel{margin:14px 0;overflow-anchor:none;padding:14px}
      #multiDipPanel .md-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:10px}
      #multiDipPanel .md-title{font-size:18px;font-weight:900;color:#eef6ff}
      #multiDipPanel .md-sub{font-size:12px;color:#8497ab;line-height:1.45;margin-top:3px}
      #multiDipPanel .md-live{font-size:11px;font-weight:900;color:#35f0ae;white-space:nowrap}
      #multiDipPanel .md-session{font-size:10px;color:#71869c;margin:0 0 10px;word-break:break-all}
      #multiDipPanel .md-tabs{display:flex;gap:6px;overflow:auto;padding:2px 0 9px;scrollbar-width:thin}
      #multiDipPanel .md-chip{appearance:none;border:1px solid #1d3b51;background:#08131f;color:#b8c9d9;border-radius:999px;padding:7px 10px;font:800 11px system-ui;white-space:nowrap;cursor:pointer}
      #multiDipPanel .md-chip.open{border-color:#217b59;color:#35f0ae}.md-chip.watch{border-color:#31516a;color:#71c9ff}.md-chip.active{background:#12324a;border-color:#62c7ff;color:#fff}
      #multiDipPanel .md-workspace{display:grid;grid-template-columns:minmax(0,1.75fr) minmax(250px,.8fr);gap:10px;margin-bottom:10px}
      #multiDipPanel .md-box{border:1px solid #193248;border-radius:11px;padding:10px;background:rgba(7,17,28,.55);min-width:0}
      #multiDipPanel .md-chart-head{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;margin-bottom:7px}
      #multiDipPanel .md-chart-title{font-size:14px;font-weight:900;color:#eef6ff}.md-chart-price{font-size:20px;font-weight:900;color:#eef6ff;font-variant-numeric:tabular-nums;text-align:right}
      #multiDipPanel .md-chart-state{font-size:9px;color:#35f0ae;text-align:right;margin-top:2px}
      #multiDipChartWrap{height:330px;min-height:280px;position:relative;border:1px solid #142b3d;border-radius:9px;overflow:hidden;background:#07101a}
      #multiDipCanvas{display:block;width:100%;height:100%}
      #multiDipPanel .md-plan{display:grid;gap:8px}
      #multiDipPanel .md-plan-row{padding:8px;border:1px solid #173247;border-radius:8px;background:#08131f}
      #multiDipPanel .md-label{font-size:9px;letter-spacing:.08em;text-transform:uppercase;color:#8193a8}.md-value{font-size:12px;font-weight:850;color:#eaf4ff;margin-top:3px;line-height:1.35}
      #multiDipPanel .md-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
      #multiDipPanel .md-box-title{font-size:11px;font-weight:900;letter-spacing:.06em;color:#9ab0c5;text-transform:uppercase;margin-bottom:8px}
      #multiDipPanel .md-list{display:grid;gap:6px;max-height:270px;overflow:auto}
      #multiDipPanel .md-row{width:100%;appearance:none;text-align:left;display:flex;align-items:center;justify-content:space-between;gap:8px;padding:8px 9px;border:1px solid #193248;border-radius:8px;background:#08131f;cursor:pointer}
      #multiDipPanel .md-row:hover,#multiDipPanel .md-row.active{border-color:#3a6c8d;background:#0b1b29}
      #multiDipPanel .md-main{font-size:12px;font-weight:850;color:#edf6ff}.md-meta{font-size:10px;color:#8193a8;margin-top:2px;line-height:1.35}.md-side{font-size:11px;font-weight:900;white-space:nowrap}.md-side.pos{color:#35f0ae}.md-side.neg{color:#ff6278}.md-side.wait{color:#f3c969}.md-side.info{color:#62c7ff}
      #multiDipPanel .md-table{width:100%;border-collapse:collapse;font-size:10px}#multiDipPanel .md-table th,#multiDipPanel .md-table td{padding:7px 5px;border-bottom:1px solid #152b3e;text-align:left;white-space:nowrap}#multiDipPanel .md-table th{color:#8193a8;font-size:9px;text-transform:uppercase}#multiDipPanel .md-table td{color:#dce9f6}
      #multiDipPanel .md-symbol-btn{appearance:none;border:0;background:none;padding:0;color:#71c9ff;font:850 10px system-ui;cursor:pointer;text-decoration:underline;text-decoration-color:#31516a;text-underline-offset:2px}
      #multiDipPanel .md-scroll{overflow:auto;max-height:300px}
      #multiDipPanel .md-note{font-size:10px;color:#71869c;line-height:1.45;margin-top:8px}
      @media(max-width:900px){#multiDipPanel .md-workspace{grid-template-columns:1fr}#multiDipPanel .md-grid{grid-template-columns:1fr}}
      @media(max-width:760px){#multiDipPanel{padding:11px}#multiDipPanel .md-head{align-items:flex-start}#multiDipPanel .md-title{font-size:17px}#multiDipChartWrap{height:315px}#multiDipPanel .md-chart-price{font-size:18px}}
    `;document.head.appendChild(s);
  }

  function ensure(){
    style();if(el('multiDipPanel'))return el('multiDipPanel');
    const overview=el('overview');if(!overview)return null;
    const panel=document.createElement('section');panel.id='multiDipPanel';panel.className='card';
    panel.innerHTML=`
      <div class="md-head"><div><div class="md-title">⚡ DIP Coin Çalışma Alanı</div><div class="md-sub">Tek DIP session: ETH çekirdeği + altcoin DIP taraması. Coin'e dokun; canlı 1s grafiği, Brian giriş/stop/hedef çizgileri ve karar nedeni açılır.</div></div><div id="multiDipState" class="md-live">BAĞLANIYOR</div></div>
      <div id="multiDipSession" class="md-session">Session bağlantısı kontrol ediliyor…</div>
      <div id="multiDipTabs" class="md-tabs"><button class="md-chip active" data-symbol="ETHUSDT">ETH · çekirdek grafik</button></div>
      <div class="md-workspace">
        <div class="md-box"><div class="md-chart-head"><div><div id="mdChartTitle" class="md-chart-title">Altcoin seç</div><div id="mdChartMeta" class="md-meta">Açık pozisyon veya izlenen coin'e dokun.</div></div><div><div id="mdChartPrice" class="md-chart-price">—</div><div id="mdChartState" class="md-chart-state">WAIT</div></div></div><div id="multiDipChartWrap"><canvas id="multiDipCanvas"></canvas></div></div>
        <div class="md-box"><div class="md-box-title">Brian planı / canlı durum</div><div id="mdPlan" class="md-plan"><div class="md-plan-row"><div class="md-label">Coin</div><div class="md-value">Altcoin seç</div></div></div><div class="md-note">ETH'ye dokunursan mevcut detaylı V8.4.4 grafiğine gider. Altcoin grafik burada Binance Spot 1s ile açılır.</div></div>
      </div>
      <div class="md-grid"><div class="md-box"><div class="md-box-title">Açık DIP pozisyonları</div><div id="mdPositions" class="md-list"><div class="md-row"><div><div class="md-main">Yükleniyor…</div></div><span class="md-side wait">WAIT</span></div></div></div><div class="md-box"><div class="md-box-title">Brian şu coinleri inceliyor</div><div id="mdWatching" class="md-list"><div class="md-row"><div><div class="md-main">Tarama yükleniyor…</div></div><span class="md-side wait">WAIT</span></div></div></div></div>
      <div class="md-box"><div class="md-box-title">DIP AL / SAT hareketleri</div><div class="md-scroll"><table class="md-table"><thead><tr><th>Saat</th><th>Coin</th><th>Olay</th><th>Fiyat</th><th>P&L</th><th>Neden</th></tr></thead><tbody id="mdEvents"><tr><td colspan="6">Yükleniyor…</td></tr></tbody></table></div></div>`;
    overview.insertAdjacentElement('afterend',panel);
    const nav=document.querySelector('.side .nav');if(nav&&!nav.querySelector('[data-multi-dip-link]')){const a=document.createElement('a');a.href='#multiDipPanel';a.dataset.multiDipLink='1';a.innerHTML='<span>⚡</span><span>DIP Coinler</span>';const radar=nav.querySelector('[data-alpha-radar-link]');if(radar)nav.insertBefore(a,radar);else nav.appendChild(a);}
    panel.addEventListener('click',e=>{const target=e.target.closest('[data-symbol]');if(!target)return;choose(String(target.dataset.symbol||''));});
    return panel;
  }

  function core(){
    try{
      const z=typeof model!=='undefined'?(model.snapshot||{}):{},session=typeof model!=='undefined'?(model.session||null):null,rt=typeof state84==='function'?(state84()||{}):{},pos=typeof position==='function'?position():null;
      const start=n(session?.starting_equity,latest?.starting_equity||1000),equity=n(z.equity??rt.cash,start),realized=n(z.realized_pnl??rt.realized),trades=n(z.trade_count??rt.trades),wins=n(z.win_count??rt.wins),losses=n(z.loss_count??rt.losses);
      return{session,start,equity,realized,trades,wins,losses,pos};
    }catch{return{session:null,start:n(latest?.starting_equity,1000),equity:n(latest?.starting_equity,1000),realized:0,trades:0,wins:0,losses:0,pos:null};}
  }

  function applyUnifiedSummary(){
    if(summaryApplying||!latest)return;summaryApplying=true;
    try{
      const c=core(),linked=!latest.source_session_id||!c.session?.session_id||String(latest.source_session_id)===String(c.session.session_id),multiDelta=linked?n(latest.equity)-n(latest.starting_equity):0,combinedEq=c.equity+multiDelta,combinedReal=c.realized+(linked?n(latest.realized_pnl):0),multiPos=linked&&Array.isArray(latest.positions)?latest.positions:[],symbols=[...(c.pos?['ETHUSDT']:[]),...multiPos.map(p=>safeSymbol(p.symbol)).filter(Boolean)],wins=c.wins+(linked?n(latest.win_count):0),losses=c.losses+(linked?n(latest.loss_count):0),trades=c.trades+(linked?n(latest.trade_count):0),closed=wins+losses;
      const labels=el('overview')?.querySelectorAll('.kpi .label');if(labels?.length>=5){setText(labels[0],'DIP Equity');setText(labels[1],'DIP Realized P&L');setText(labels[2],'Açık Pozisyon');setText(labels[3],'DIP Win Rate');setText(labels[4],'DIP İşlem');}
      const eq=el('kpiEquity');if(eq)setText(eq,usd(combinedEq));const pnl=el('kpiPnl');if(pnl){setText(pnl,money(combinedReal));pnl.className=`value ${combinedReal>0?'pos':combinedReal<0?'neg':''}`;}
      setText(el('kpiOpen'),String(symbols.length));setText(el('kpiWin'),closed?`${(wins/closed*100).toFixed(1)}%`:'—');setText(el('kpiTrades'),String(trades));
      setText(el('kpiEquityMeta'),linked?`Tek DIP session · başlangıç ${usd(c.start)} · ETH + altcoin sonuçları`:`ETH ${usd(c.equity)} · altcoin session eşleşmesi bekleniyor`);
      setText(el('kpiPnlMeta'),'ETH + altcoin SHADOW gerçekleşen toplam sonuç');setText(el('kpiOpenMeta'),symbols.length?symbols.join(' · '):'Pozisyon yok · DIP fırsat taranıyor');setText(el('kpiWinMeta'),`${wins} win / ${losses} loss`);setText(el('kpiTradesMeta'),`ETH ${c.trades} + altcoin ${linked?n(latest.trade_count):0} kapalı round trip`);
    }finally{summaryApplying=false;}
  }

  function queueSummary(){if(summaryQueued)return;summaryQueued=true;queueMicrotask(()=>{summaryQueued=false;applyUnifiedSummary();});}
  function guardOverview(){const o=el('overview');if(!o)return;new MutationObserver(()=>{if(!summaryApplying)queueSummary();}).observe(o,{subtree:true,childList:true,characterData:true});}

  function latestUniqueEvaluations(rows){const map=new Map();for(const r of rows||[]){const s=safeSymbol(r.symbol);if(s&&!map.has(s))map.set(s,r);}return[...map.values()];}
  function latestEvent(symbol){return (latest?.recent_events||[]).find(e=>safeSymbol(e.symbol)===symbol)||null;}
  function latestEval(symbol){return (latest?.recent_evaluations||[]).find(e=>safeSymbol(e.symbol)===symbol)||null;}
  function openPosition(symbol){return (latest?.positions||[]).find(p=>safeSymbol(p.symbol)===symbol)||null;}

  function candidateSymbols(){
    const out=[],seen=new Set(),push=(s,kind)=>{s=safeSymbol(s);if(!s||seen.has(s)||s==='ETHUSDT')return;seen.add(s);out.push({symbol:s,kind});};
    for(const p of latest?.positions||[])push(p.symbol,'open');
    for(const e of latest?.recent_events||[])push(e.symbol,'event');
    for(const e of latestUniqueEvaluations(latest?.recent_evaluations||[]))push(e.symbol,'watch');
    return out.slice(0,12);
  }

  function renderTabs(){const host=el('multiDipTabs');if(!host)return;const rows=candidateSymbols();host.innerHTML=`<button class="md-chip ${selected==='ETHUSDT'?'active':''}" data-symbol="ETHUSDT">ETH · çekirdek</button>`+rows.map(r=>`<button class="md-chip ${r.kind==='open'?'open':'watch'} ${selected===r.symbol?'active':''}" data-symbol="${esc(r.symbol)}">${esc(r.symbol.replace(/USDT$/,''))}${r.kind==='open'?' · AÇIK':''}</button>`).join('');}

  function renderLists(){
    const positions=Array.isArray(latest?.positions)?latest.positions:[],posHost=el('mdPositions');if(posHost){posHost.innerHTML=positions.length?positions.map(p=>{const s=safeSymbol(p.symbol),up=n(p.mark)>=n(p.entry);return `<button class="md-row ${selected===s?'active':''}" data-symbol="${esc(s)}"><div><div class="md-main">${esc(s)} · LONG</div><div class="md-meta">Entry ${price(p.entry)} · Son ${price(p.mark)} · Stop ${price(p.stop)} · Hedef ${price(p.target)}</div></div><span class="md-side ${up?'pos':'neg'}">${money(p.unrealized_pnl)}</span></button>`;}).join(''):'<div class="md-row"><div><div class="md-main">Açık pozisyon yok</div><div class="md-meta">Brian coin taramaya devam ediyor.</div></div><span class="md-side wait">0</span></div>';}
    const watching=latestUniqueEvaluations(latest?.recent_evaluations||[]).filter(r=>!positions.some(p=>safeSymbol(p.symbol)===safeSymbol(r.symbol))).slice(0,8),watchHost=el('mdWatching');if(watchHost){watchHost.innerHTML=watching.length?watching.map(r=>{const s=safeSymbol(r.symbol),score=Math.round(n(r.signal_score)*100),reason=String(r.reason||r.action||'WAIT');return `<button class="md-row ${selected===s?'active':''}" data-symbol="${esc(s)}"><div><div class="md-main">${esc(s)}</div><div class="md-meta">${esc(reason)} · Radar ${Math.round(n(r.radar_score)*100)}/100</div></div><span class="md-side info">${score}/100</span></button>`;}).join(''):'<div class="md-row"><div><div class="md-main">Tarama bekleniyor</div></div><span class="md-side wait">WAIT</span></div>';}
    const events=Array.isArray(latest?.recent_events)?latest.recent_events:[],body=el('mdEvents');if(body){body.innerHTML=events.length?events.slice(0,40).map(e=>{const s=safeSymbol(e.symbol),pv=e.pnl==null?null:n(e.pnl);return `<tr><td>${esc(time(e.observed_at))}</td><td><button class="md-symbol-btn" data-symbol="${esc(s)}">${esc(s)}</button></td><td>${esc(e.action)}</td><td>${esc(price(e.price))}</td><td class="${pv==null?'':pv>=0?'pos':'neg'}">${pv==null?'—':esc(money(pv))}</td><td>${esc(e.reason)}</td></tr>`;}).join(''):'<tr><td colspan="6">Bu session’da henüz altcoin DIP işlemi yok.</td></tr>';}
  }

  function planLevels(symbol){
    const p=openPosition(symbol),ev=latestEvent(symbol),meta=(ev?.metadata||{}),evaluation=latestEval(symbol),em=(evaluation?.metadata||{});
    return{p,ev,evaluation,entry:n(p?.entry||meta.entry||((ev?.action==='BUY')?ev.price:0)),stop:n(p?.stop||meta.stop),target:n(p?.target||meta.target),trail:n(p?.trail||meta.trail),reason:String(p?.entry_reason||evaluation?.reason||ev?.reason||'WAIT'),action:p?'HOLD / LONG':String(evaluation?.action||ev?.action||'WAIT'),radar:n(p?.radar_score||evaluation?.radar_score||em.radar_score),signal:n(evaluation?.signal_score||em.signal_score),pullback:n(em.pullback_pct),bounce:n(em.bounce_pct)};
  }

  function renderPlan(){const host=el('mdPlan');if(!host)return;if(!selected||selected==='ETHUSDT'){host.innerHTML='<div class="md-plan-row"><div class="md-label">ETH çekirdeği</div><div class="md-value">Mevcut V8.4.4 grafiğinde detaylı thesis, entry, stop ve hedef çizgileri var.</div></div>';return;}const x=planLevels(selected);host.innerHTML=`<div class="md-plan-row"><div class="md-label">Coin / Brian</div><div class="md-value">${esc(selected)} · ${esc(x.action)} · ${esc(x.reason)}</div></div><div class="md-plan-row"><div class="md-label">Seviyeler</div><div class="md-value">Entry ${price(x.entry)} · Stop ${price(x.stop)} · Hedef ${price(x.target)}${x.trail>0?` · Trail ${price(x.trail)}`:''}</div></div><div class="md-plan-row"><div class="md-label">Evidence</div><div class="md-value">Radar ${Math.round(x.radar*100)}/100 · Signal ${Math.round(x.signal*100)}/100${x.pullback?` · Pullback ${x.pullback.toFixed(2)}%`:''}${x.bounce?` · Bounce ${x.bounce.toFixed(2)}%`:''}</div></div>`;}

  function drawLine(ctx,y,L,R,w,label,value,color){const v=n(value);if(!(v>0))return;const yy=y(v);if(!Number.isFinite(yy))return;ctx.save();ctx.setLineDash([6,4]);ctx.strokeStyle=color;ctx.lineWidth=1.25;ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.setLineDash([]);ctx.font='700 10px system-ui';const text=`${label} ${price(v)}`,tw=ctx.measureText(text).width,x=Math.max(L+4,Math.min(w-R-tw-5,L+6));ctx.fillStyle='rgba(5,10,18,.92)';ctx.fillRect(x-3,yy-14,tw+6,14);ctx.fillStyle=color;ctx.fillText(text,x,yy-3);ctx.restore();}

  function drawChart(){
    const cv=el('multiDipCanvas'),wrap=el('multiDipChartWrap');if(!cv||!wrap)return;const ctx=cv.getContext('2d'),dpr=window.devicePixelRatio||1,w=Math.max(300,wrap.clientWidth),h=Math.max(260,wrap.clientHeight);cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#07101a';ctx.fillRect(0,0,w,h);
    const rows=chart.candles.slice(-180);if(!selected||selected==='ETHUSDT'){ctx.fillStyle='#91a2b8';ctx.font='13px system-ui';ctx.fillText('ETH için aşağıdaki V8.4.4 çekirdek grafiğini kullan.',18,34);return;}if(rows.length<2){ctx.fillStyle='#91a2b8';ctx.font='13px system-ui';ctx.fillText('Binance Spot 1s mumları yükleniyor…',18,34);return;}
    const lev=planLevels(selected),levels=[lev.entry,lev.stop,lev.target,lev.trail,chart.last].filter(v=>n(v)>0),vals=rows.flatMap(c=>[c.l,c.h]).concat(levels);let lo=Math.min(...vals),hi=Math.max(...vals),pad=(hi-lo)*.09||hi*.001||.001;lo-=pad;hi+=pad;const L=9,R=78,T=18,B=28,pw=w-L-R,ph=h-T-B,cw=pw/rows.length,y=v=>T+(hi-v)/(hi-lo)*ph;
    for(let i=0;i<=4;i++){const yy=T+ph*i/4,v=hi-(hi-lo)*i/4;ctx.strokeStyle='#152436';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.fillStyle='#718198';ctx.font='9px system-ui';ctx.fillText(price(v),w-R+6,yy+3);}
    rows.forEach((c,i)=>{const x=L+cw*i+cw/2,up=c.c>=c.o,col=up?'#12d996':'#ff5068',body=Math.max(1,Math.abs(y(c.o)-y(c.c))),top=Math.min(y(c.o),y(c.c)),ww=Math.max(1,Math.min(5,cw*.72));ctx.strokeStyle=col;ctx.fillStyle=col;ctx.beginPath();ctx.moveTo(x,y(c.h));ctx.lineTo(x,y(c.l));ctx.stroke();ctx.fillRect(x-ww/2,top,ww,body);});
    drawLine(ctx,y,L,R,w,'ENTRY',lev.entry,'#62a9ff');drawLine(ctx,y,L,R,w,'STOP',lev.stop,'#ff6278');drawLine(ctx,y,L,R,w,'HEDEF',lev.target,'#35f0ae');if(lev.trail>0)drawLine(ctx,y,L,R,w,'TRAIL',lev.trail,'#f3c969');drawLine(ctx,y,L,R,w,'CANLI',chart.last,'#e8f4ff');
  }

  async function seedChart(symbol,seq){
    let lastErr=null;
    for(const base of REST_HOSTS){try{const r=await fetch(`${base}/api/v3/klines?symbol=${encodeURIComponent(symbol)}&interval=1s&limit=180`,{cache:'no-store'});if(!r.ok)throw Error(`HTTP ${r.status}`);const raw=await r.json();if(!Array.isArray(raw)||raw.length<2)throw Error('MUM YOK');if(seq!==chart.seq)return;chart.candles=raw.map(a=>({t:n(a[0]),o:n(a[1]),h:n(a[2]),l:n(a[3]),c:n(a[4])})).filter(c=>c.t>0&&c.c>0);chart.last=chart.candles.at(-1)?.c||0;chart.source='REST + WS';drawChart();renderChartHead();return;}catch(e){lastErr=e;}}
    if(seq===chart.seq){chart.source='REST HATA';renderChartHead();drawChart();console.warn('multi dip chart seed',lastErr);}
  }

  function stopSocket(){if(chart.reconnect){clearTimeout(chart.reconnect);chart.reconnect=null;}if(chart.ws){try{chart.ws.onclose=null;chart.ws.close();}catch{}chart.ws=null;}}
  function connectSocket(symbol,seq){stopSocket();if(!symbol||symbol==='ETHUSDT'||document.hidden)return;const url=`wss://stream.binance.com:9443/ws/${symbol.toLowerCase()}@kline_1s`;try{const ws=new WebSocket(url);chart.ws=ws;ws.onopen=()=>{if(seq!==chart.seq)return;chart.source='BINANCE 1s LIVE';renderChartHead();};ws.onmessage=ev=>{if(seq!==chart.seq)return;try{const d=JSON.parse(ev.data),k=d?.k;if(!k)return;const c={t:n(k.t),o:n(k.o),h:n(k.h),l:n(k.l),c:n(k.c)};if(!(c.t>0&&c.c>0))return;const last=chart.candles.at(-1);if(last?.t===c.t)chart.candles[chart.candles.length-1]=c;else{chart.candles.push(c);if(chart.candles.length>180)chart.candles.splice(0,chart.candles.length-180);}chart.last=c.c;renderChartHead();drawChart();}catch{}};ws.onerror=()=>{chart.source='WS RETRY';renderChartHead();};ws.onclose=()=>{if(seq!==chart.seq||document.hidden)return;chart.reconnect=setTimeout(()=>connectSocket(symbol,seq),1800);};}catch{chart.reconnect=setTimeout(()=>connectSocket(symbol,seq),2200);}}

  function renderChartHead(){setText(el('mdChartTitle'),selected&&selected!=='ETHUSDT'?`${selected} · Binance Spot · 1s LIVE`:'ETHUSDT · V8.4.4 çekirdek');setText(el('mdChartPrice'),selected==='ETHUSDT'?'ETH ↓':price(chart.last));setText(el('mdChartState'),selected==='ETHUSDT'?'ÇEKİRDEK GRAFİK':chart.source);const ev=selected&&selected!=='ETHUSDT'?latestEval(selected):null;setText(el('mdChartMeta'),selected==='ETHUSDT'?'Detaylı Brian çizgileri mevcut ETH grafiğinde.':ev?`${ev.action||'WAIT'} · ${ev.reason||'—'} · signal ${Math.round(n(ev.signal_score)*100)}/100`:'Brian planı / son işlem seviyeleri');}

  function choose(symbol){symbol=safeSymbol(symbol);if(!symbol)return;if(symbol==='ETHUSDT'){selected='ETHUSDT';chart.seq++;stopSocket();renderTabs();renderLists();renderPlan();renderChartHead();drawChart();el('chartPanel')?.scrollIntoView({behavior:'smooth',block:'start'});return;}if(selected===symbol&&chart.symbol===symbol){el('multiDipChartWrap')?.scrollIntoView({behavior:'smooth',block:'center'});return;}selected=symbol;chart.symbol=symbol;chart.candles=[];chart.last=0;chart.source='YÜKLENİYOR';const seq=++chart.seq;renderTabs();renderLists();renderPlan();renderChartHead();drawChart();seedChart(symbol,seq);connectSocket(symbol,seq);el('multiDipChartWrap')?.scrollIntoView({behavior:'smooth',block:'center'});}

  function render(data){
    latest=data;ensure();const c=core(),linked=!data?.source_session_id||!c.session?.session_id||String(data.source_session_id)===String(c.session.session_id),state=el('multiDipState'),running=String(data?.status||'')==='RUNNING';if(state){state.textContent=running?`DIP CANLI · ${Math.round(n(data.age_seconds))} sn`:String(data?.status||'WAIT');state.style.color=running?'#35f0ae':'#f3c969';}
    setText(el('multiDipSession'),linked?`Tek session bağlı · ${String(data?.source_session_id||c.session?.session_id||'—')} · SHADOW ONLY · gerçek emir KAPALI`:`SESSION EŞLEŞMİYOR · ETH ${c.session?.session_id||'—'} · altcoin ${data?.source_session_id||'—'} · Yeni SHADOW session ile eşleştir.`);
    if(!selected){const first=(data?.positions||[])[0]?.symbol||(data?.recent_events||[])[0]?.symbol||(data?.recent_evaluations||[])[0]?.symbol;selected=safeSymbol(first)||'ETHUSDT';if(selected!=='ETHUSDT'){chart.symbol=selected;const seq=++chart.seq;seedChart(selected,seq);connectSocket(selected,seq);}}
    applyUnifiedSummary();renderTabs();renderLists();renderPlan();renderChartHead();drawChart();
  }

  async function refresh(){ensure();const key=localStorage.getItem(KEY)||'';if(!key){setText(el('multiDipState'),'KİLİTLİ');return;}try{const r=await fetch(API,{method:'POST',headers:{'content-type':'application/json','x-brian-dashboard-key':key},body:JSON.stringify({action:'status'}),cache:'no-store'});const d=await r.json().catch(()=>({}));if(!r.ok)throw Error(d.error||`HTTP ${r.status}`);render(d);}catch(e){const s=el('multiDipState');if(s){s.textContent='VERİ HATASI';s.style.color='#ff6278';}console.warn('multi dip status',e);}}

  window.addEventListener('load',()=>{ensure();guardOverview();refresh();setInterval(()=>{if(!document.hidden)refresh();},6000);const wrap=el('multiDipChartWrap');if(wrap&&window.ResizeObserver)new ResizeObserver(()=>drawChart()).observe(wrap);});
  window.addEventListener('dip:session-restarted',()=>{latest=null;selected='';chart.seq++;stopSocket();});
  document.addEventListener('visibilitychange',()=>{if(document.hidden){stopSocket();return;}refresh();if(selected&&selected!=='ETHUSDT')connectSocket(selected,chart.seq);});
})();
