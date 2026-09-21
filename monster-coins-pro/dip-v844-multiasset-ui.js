'use strict';

(()=>{
  const API='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-dip-multiasset-status';
  const KEY='mcp-dashboard-key-v1';
  const REST_HOSTS=['https://api.binance.com','https://data-api.binance.vision'];
  const el=id=>document.getElementById(id);
  const n=(v,f=0)=>Number.isFinite(Number(v))?Number(v):f;
  const esc=s=>String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
  const money=v=>{const x=n(v);return `${x>=0?'+':'-'}$${Math.abs(x).toFixed(2)}`;};
  const usd=v=>`$${n(v).toFixed(2)}`;
  const price=v=>{const x=n(v);if(!(x>0))return '—';const d=x<.001?8:x<1?6:x<100?4:2;return x.toLocaleString('en-US',{maximumFractionDigits:d,minimumFractionDigits:Math.min(2,d)});};
  const time=v=>{const d=new Date(String(v||''));return Number.isFinite(d.getTime())?d.toLocaleTimeString('tr-TR',{hour:'2-digit',minute:'2-digit',second:'2-digit'}):'—';};
  const safeSymbol=v=>{const s=String(v||'').toUpperCase();return /^[A-Z0-9]{2,20}USDT$/.test(s)?s:'';};
  const setText=(node,value)=>{if(node&&node.textContent!==value)node.textContent=value;};
  let latest=null,selected='',chart={symbol:'',candles:[],micro:[],last:0,source:'WAIT',ws:null,reconnect:null,seq:0},positionTicks=new Map(),summaryQueued=false,summaryApplying=false;

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
      #multiDipPanel .md-arena{margin:0 0 10px;border-color:#5d4520;background:linear-gradient(180deg,rgba(42,28,8,.55),rgba(7,17,28,.66))}
      #multiDipPanel .md-arena-head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:9px}
      #multiDipPanel .md-arena-title{font-size:13px;font-weight:950;color:#ffd98a}.md-arena-badge{font-size:9px;font-weight:900;color:#35f0ae;white-space:nowrap}
      #multiDipPanel .md-arena-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px;margin-bottom:8px}
      #multiDipPanel .md-arena-kpi{padding:8px;border:1px solid #47391e;border-radius:8px;background:rgba(8,19,31,.72)}
      #multiDipPanel .md-arena-kpi b{display:block;font-size:14px;color:#eef6ff;margin-top:2px}.md-arena-kpi span{font-size:8px;color:#8ea0b2;text-transform:uppercase;letter-spacing:.06em}
      #multiDipPanel .md-arena-grid{display:grid;grid-template-columns:1.2fr .8fr;gap:8px}.md-arena-mini{display:grid;gap:5px}
      @media(max-width:760px){#multiDipPanel .md-arena-kpis{grid-template-columns:1fr 1fr}#multiDipPanel .md-arena-grid{grid-template-columns:1fr}}
      @media(max-width:900px){#multiDipPanel .md-workspace{grid-template-columns:1fr}#multiDipPanel .md-grid{grid-template-columns:1fr}}
      @media(max-width:760px){#multiDipPanel{padding:11px}#multiDipPanel .md-head{align-items:flex-start}#multiDipPanel .md-title{font-size:17px}#multiDipChartWrap{height:315px}#multiDipPanel .md-chart-price{font-size:18px}}
    `;document.head.appendChild(s);
  }

  function ensure(){
    style();if(el('multiDipPanel'))return el('multiDipPanel');
    const overview=el('overview');if(!overview)return null;
    const panel=document.createElement('section');panel.id='multiDipPanel';panel.className='card';
    panel.innerHTML=`
      <div class="md-head"><div><div class="md-title">⚡ DIP Coin Çalışma Alanı</div><div class="md-sub">Tek DIP session: ETH çekirdeği + altcoin DIP taraması. Coin'e dokun; 30dk canlı bağlam, Binance 1s gerçek son dip/tepe, Brian giriş/stop/hedef/trail çizgileri ve karar nedeni açılır.</div></div><div id="multiDipState" class="md-live">BAĞLANIYOR</div></div>
      <div id="multiDipSession" class="md-session">Session bağlantısı kontrol ediliyor…</div>
      <div id="multiDipTabs" class="md-tabs"><button class="md-chip active" data-symbol="ETHUSDT">ETH · çekirdek grafik</button></div>
      <div class="md-box md-arena">
        <div class="md-arena-head"><div><div class="md-arena-title">🧨 AGGRESSIVE ARENA · $1000 paralel SHADOW kasa</div><div class="md-meta">Aynı Brian gözleri ve aynı tarama. EXPLOSION-FIRST ADAPTIVE: Arena ana Brian’ın FROZEN durumundan bağımsızdır; o anki piyasanın en güçlü patlama adaylarını göreli olarak sıralar, iki tur teyit ister. MONSTER/ULTRA gelirse %85–99.5 kasa test edilebilir.</div></div><div id="mdArenaBadge" class="md-arena-badge">BAĞLANIYOR</div></div>
        <div id="mdArenaKpis" class="md-arena-kpis"></div>
        <div class="md-arena-grid"><div><div class="md-box-title">Arena açık pozisyonları</div><div id="mdArenaPositions" class="md-arena-mini"><div class="md-row"><div><div class="md-main">Yükleniyor…</div></div><span class="md-side wait">WAIT</span></div></div></div><div><div class="md-box-title">Son Arena hareketleri</div><div id="mdArenaEvents" class="md-arena-mini"><div class="md-row"><div><div class="md-main">Henüz hareket yok</div></div></div></div></div></div>
        <div class="md-box-title" style="margin-top:9px">Patlama adayları</div><div id="mdArenaWatch" class="md-arena-mini"><div class="md-row"><div><div class="md-main">Adaylar yükleniyor…</div></div></div></div>
        <div class="md-note">Dinamik seçim: patlama eşiği piyasanın o anki üst dilimine göre ayarlanır. Teyit aynı mumda tekrar sayılmaz; ikinci bağımsız kapanmış 1dk mum gerekir. Rebound adayları önce testte bekler. Normal Arena sermayesi conviction'a göre yaklaşık %22–70 arasında değişir; MONSTER/ULTRA ayrı kurallıdır. Yük koruması: ikinci piyasa taraması yok.</div>
      </div>
      <div class="md-workspace">
        <div class="md-box"><div class="md-chart-head"><div><div id="mdChartTitle" class="md-chart-title">Altcoin seç</div><div id="mdChartMeta" class="md-meta">Açık pozisyon veya izlenen coin'e dokun.</div></div><div><div id="mdChartPrice" class="md-chart-price">—</div><div id="mdChartState" class="md-chart-state">WAIT</div></div></div><div id="multiDipChartWrap"><canvas id="multiDipCanvas"></canvas></div></div>
        <div class="md-box"><div class="md-box-title">Brian planı / canlı durum</div><div id="mdPlan" class="md-plan"><div class="md-plan-row"><div class="md-label">Coin</div><div class="md-value">Altcoin seç</div></div></div><div class="md-note">ETH'ye dokunursan mevcut detaylı V8.4.4 grafiğine gider. Altcoin grafik 30 dakikalık mum bağlamını gösterir; son 3 dakikanın gerçek Binance 1s dip/tepesi işaretlenir ve fiyat 1 saniyelik akışla canlı güncellenir.</div></div>
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
    for(const p of latest?.positions||[])push(p.symbol,'open');for(const p of latest?.arena?.positions||[])push(p.symbol,'arena');for(const e of latest?.recent_events||[])push(e.symbol,'event');for(const e of latest?.arena?.recent_events||[])push(e.symbol,'arena');for(const e of latestUniqueEvaluations(latest?.recent_evaluations||[]))push(e.symbol,'watch');return out.slice(0,14);
  }
  function renderTabs(){const host=el('multiDipTabs');if(!host)return;const rows=candidateSymbols();host.innerHTML=`<button class="md-chip ${selected==='ETHUSDT'?'active':''}" data-symbol="ETHUSDT">ETH · çekirdek</button>`+rows.map(r=>`<button class="md-chip ${r.kind==='open'||r.kind==='arena'?'open':'watch'} ${selected===r.symbol?'active':''}" data-symbol="${esc(r.symbol)}">${esc(r.symbol.replace(/USDT$/,''))}${r.kind==='open'?' · AÇIK':r.kind==='arena'?' · ARENA':''}</button>`).join('');}

  function livePosition(p){const s=safeSymbol(p.symbol),tick=positionTicks.get(s),mark=n(tick?.price,n(p.mark,n(p.entry))),entry=n(p.entry),qty=n(p.qty),cost=n(p.cost_basis),gross=(mark-entry)*qty,feeExit=mark*qty*.001,pnl=gross-cost-feeExit,pct=entry>0?(mark/entry-1)*100:0;return{mark,pnl,pct,age:tick?Math.max(0,(Date.now()-tick.at)/1000):null};}
  function renderPositionRows(){const positions=Array.isArray(latest?.positions)?latest.positions:[],posHost=el('mdPositions');if(!posHost)return;posHost.innerHTML=positions.length?positions.map(p=>{const s=safeSymbol(p.symbol),v=livePosition(p),up=v.pnl>=0,fresh=v.age!==null&&v.age<4;return `<button class="md-row ${selected===s?'active':''}" data-symbol="${esc(s)}"><div><div class="md-main">${esc(s)} · LONG</div><div class="md-meta">Entry ${price(p.entry)} · CANLI ${price(v.mark)} · ${v.pct>=0?'+':''}${v.pct.toFixed(2)}% · Stop ${price(p.stop)} · Hedef ${price(p.target)}${fresh?' · 1s':''}</div></div><span class="md-side ${up?'pos':'neg'}">${money(v.pnl)}</span></button>`;}).join(''):'<div class="md-row"><div><div class="md-main">Açık pozisyon yok</div><div class="md-meta">Brian coin taramaya devam ediyor.</div></div><span class="md-side wait">0</span></div>';}
  function renderArena(){
    const a=latest?.arena,badge=el('mdArenaBadge'),kpis=el('mdArenaKpis'),ph=el('mdArenaPositions'),eh=el('mdArenaEvents'),wh=el('mdArenaWatch');
    if(!a){if(badge)setText(badge,'BEKLENİYOR');if(kpis)kpis.innerHTML='<div class="md-arena-kpi"><span>Durum</span><b>Bağlanıyor</b></div>';return;}
    const running=String(a.status||'')==='RUNNING',eq=n(a.equity,n(a.cash,1000)),start=n(a.starting_equity,1000),pnl=eq-start,real=n(a.realized_pnl),positions=Array.isArray(a.positions)?a.positions:[],events=Array.isArray(a.recent_events)?a.recent_events:[];
    if(badge){const th=n(a.adaptive_explosion_threshold);badge.textContent=running?`CANLI · ADAPTIVE · EŞİK ${th?Math.round(th*100):'—'}`:String(a.status||'WAIT');badge.style.color=running?'#35f0ae':'#f3c969';}
    if(kpis)kpis.innerHTML=`<div class="md-arena-kpi"><span>Equity</span><b>${usd(eq)}</b></div><div class="md-arena-kpi"><span>Toplam Δ</span><b class="${pnl>=0?'pos':'neg'}">${money(pnl)}</b></div><div class="md-arena-kpi"><span>Realized</span><b class="${real>=0?'pos':'neg'}">${money(real)}</b></div><div class="md-arena-kpi"><span>Açık / İşlem</span><b>${positions.length} / ${n(a.trade_count)}</b></div>`;
    if(ph)ph.innerHTML=positions.length?positions.map(p=>{const s=safeSymbol(p.symbol),alloc=n(p.allocation_fraction)*100,up=n(p.unrealized_pnl)>=0,kind=String(p.slot_kind||'REGULAR'),u=Math.round(n(p.brain_utility||p.forecast_utility)*100),ex=Math.round(n(p.explosion_score)*100);return `<button class="md-row ${selected===s?'active':''}" data-symbol="${esc(s)}"><div><div class="md-main">${esc(s)} · ${esc(kind)}</div><div class="md-meta">Kasa %${alloc.toFixed(1)} · Entry ${price(p.entry)} · Mark ${price(p.mark)} · Brain ${u}/100 · Patlama ${ex}/100</div></div><span class="md-side ${up?'pos':'neg'}">${money(p.unrealized_pnl)}</span></button>`;}).join(''):'<div class="md-row"><div><div class="md-main">Arena pozisyon bekliyor</div><div class="md-meta">En güçlü iki fırsat için sermaye hazır.</div></div><span class="md-side wait">0</span></div>';
    if(eh)eh.innerHTML=events.length?events.slice(0,5).map(e=>{const s=safeSymbol(e.symbol),pv=e.pnl==null?null:n(e.pnl);return `<button class="md-row" data-symbol="${esc(s)}"><div><div class="md-main">${esc(s)} · ${esc(e.action)} · ${esc(e.reason)}</div><div class="md-meta">${time(e.observed_at)} · ${price(e.price)}</div></div><span class="md-side ${pv==null?'info':pv>=0?'pos':'neg'}">${pv==null?usd(e.notional):money(pv)}</span></button>`;}).join(''):'<div class="md-row"><div><div class="md-main">Henüz Arena BUY/SELL yok</div><div class="md-meta">Ana tarama sinyal üretince burada görünecek.</div></div></div>';
    const watch=Array.isArray(a.arena_watch)?a.arena_watch:[];
    if(wh)wh.innerHTML=watch.length?watch.slice(0,6).map(x=>{const s=safeSymbol(x.symbol),ok=x.qualified===true,ex=Math.round(n(x.explosion_score)*100),bu=Math.round(n(x.brain_utility)*100),co=Math.round(n(x.continuation)*100),op=Math.round(n(x.opportunity_score)*100),st=Math.round(n(x.confirm_streak)),rebound=x.rebound_test===true,newBar=x.new_evidence_bar===true;const state=!ok?'BEKLE':st>=2?'TEYİTLİ':rebound?'REBOUND TEST':newBar?'1/2 · YENİ MUM':'1/2 · YENİ MUM BEKLE';return `<button class="md-row ${selected===s?'active':''}" data-symbol="${esc(s)}"><div><div class="md-main">${esc(s)} · ${state}</div><div class="md-meta">Patlama ${ex}/100 · Brain ${bu}/100 · Opp ${op}/100 · Devam ${co}/100 · Bağımsız teyit ${st}/2</div></div><span class="md-side ${ok?'pos':'wait'}">${ok?(st>=2?'ARENA':'1/2'):'WAIT'}</span></button>`;}).join(''):'<div class="md-row"><div><div class="md-main">Şu an Arena kalitesinde aday yok</div><div class="md-meta">Kasa boşta bekler; sırf slot doldurmak için işlem açılmaz.</div></div><span class="md-side wait">WAIT</span></div>';
  }

  function renderLists(){
    const positions=Array.isArray(latest?.positions)?latest.positions:[];renderPositionRows();
    const watching=latestUniqueEvaluations(latest?.recent_evaluations||[]).filter(r=>!positions.some(p=>safeSymbol(p.symbol)===safeSymbol(r.symbol))).slice(0,8),watchHost=el('mdWatching');if(watchHost){watchHost.innerHTML=watching.length?watching.map(r=>{const s=safeSymbol(r.symbol),score=Math.round(n(r.signal_score)*100),reason=String(r.reason||r.action||'WAIT');return `<button class="md-row ${selected===s?'active':''}" data-symbol="${esc(s)}"><div><div class="md-main">${esc(s)}</div><div class="md-meta">${esc(reason)} · Radar ${Math.round(n(r.radar_score)*100)}/100</div></div><span class="md-side info">${score}/100</span></button>`;}).join(''):'<div class="md-row"><div><div class="md-main">Tarama bekleniyor</div></div><span class="md-side wait">WAIT</span></div>';}
    const events=Array.isArray(latest?.recent_events)?latest.recent_events:[],body=el('mdEvents');if(body){body.innerHTML=events.length?events.slice(0,40).map(e=>{const s=safeSymbol(e.symbol),pv=e.pnl==null?null:n(e.pnl);return `<tr><td>${esc(time(e.observed_at))}</td><td><button class="md-symbol-btn" data-symbol="${esc(s)}">${esc(s)}</button></td><td>${esc(e.action)}</td><td>${esc(price(e.price))}</td><td class="${pv==null?'':pv>=0?'pos':'neg'}">${pv==null?'—':esc(money(pv))}</td><td>${esc(e.reason)}</td></tr>`;}).join(''):'<tr><td colspan="6">Bu session’da henüz altcoin DIP işlemi yok.</td></tr>';}
  }

  function planLevels(symbol){const p=openPosition(symbol),ev=latestEvent(symbol),meta=(ev?.metadata||{}),evaluation=latestEval(symbol),em=(evaluation?.metadata||{});return{p,ev,evaluation,entry:n(p?.entry||meta.entry||((ev?.action==='BUY')?ev.price:0)),stop:n(p?.stop||meta.stop),target:n(p?.target||meta.target),trail:n(p?.trail||meta.trail),reason:String(p?.entry_reason||evaluation?.reason||ev?.reason||'WAIT'),action:p?'HOLD / LONG':String(evaluation?.action||ev?.action||'WAIT'),radar:n(p?.radar_score||evaluation?.radar_score||em.radar_score),signal:n(evaluation?.signal_score||em.signal_score),pullback:n(em.pullback_pct),bounce:n(em.bounce_pct)};}
  function renderPlan(){const host=el('mdPlan');if(!host)return;if(!selected||selected==='ETHUSDT'){host.innerHTML='<div class="md-plan-row"><div class="md-label">ETH çekirdeği</div><div class="md-value">Mevcut V8.4.4 grafiğinde detaylı thesis, entry, stop ve hedef çizgileri var.</div></div>';return;}const x=planLevels(selected);host.innerHTML=`<div class="md-plan-row"><div class="md-label">Coin / Brian</div><div class="md-value">${esc(selected)} · ${esc(x.action)} · ${esc(x.reason)}</div></div><div class="md-plan-row"><div class="md-label">Seviyeler</div><div class="md-value">Entry ${price(x.entry)} · Stop ${price(x.stop)} · Hedef ${price(x.target)}${x.trail>0?` · Trail ${price(x.trail)}`:''}</div></div><div class="md-plan-row"><div class="md-label">Evidence</div><div class="md-value">Radar ${Math.round(x.radar*100)}/100 · Signal ${Math.round(x.signal*100)}/100${x.pullback?` · Pullback ${x.pullback.toFixed(2)}%`:''}${x.bounce?` · Bounce ${x.bounce.toFixed(2)}%`:''}</div></div>`;}

  function drawLine(ctx,y,L,R,w,label,value,color){const v=n(value);if(!(v>0))return;const yy=y(v);if(!Number.isFinite(yy))return;ctx.save();ctx.setLineDash([6,4]);ctx.strokeStyle=color;ctx.lineWidth=1.25;ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.setLineDash([]);ctx.font='700 10px system-ui';const text=`${label} ${price(v)}`,tw=ctx.measureText(text).width,x=Math.max(L+4,Math.min(w-R-tw-5,L+6));ctx.fillStyle='rgba(5,10,18,.92)';ctx.fillRect(x-3,yy-14,tw+6,14);ctx.fillStyle=color;ctx.fillText(text,x,yy-3);ctx.restore();}
  function minuteBucket(t){return Math.floor(n(t)/60000)*60000;}
  function fillMinuteGaps(rows){
    const out=[];for(const raw of rows||[]){const c={t:minuteBucket(raw.t),o:n(raw.o),h:n(raw.h),l:n(raw.l),c:n(raw.c)};if(!(c.t>0&&c.c>0))continue;
      const prev=out.at(-1);if(prev&&c.t>prev.t+60000){for(let t=prev.t+60000;t<c.t;t+=60000)out.push({t,o:prev.c,h:prev.c,l:prev.c,c:prev.c,synthetic:true});}
      if(prev&&prev.t===c.t)out[out.length-1]=c;else out.push(c);
    }return out.slice(-40);
  }
  function recentSwing(rows,kind='low'){
    if(!rows?.length)return null;const arr=rows.slice(-24),key=kind==='low'?'l':'h';
    for(let i=arr.length-3;i>=2;i--){const v=n(arr[i][key]);if(!(v>0))continue;let ok=true;for(let j=i-2;j<=i+2;j++){if(j===i)continue;const x=n(arr[j]?.[key]);if(!(x>0))continue;if(kind==='low'?v>x:v<x){ok=false;break;}}if(ok)return{value:v,t:arr[i].t};}
    const scope=arr.slice(-12);if(!scope.length)return null;let best=scope[0];for(const x of scope){if(kind==='low'?n(x.l)<n(best.l):n(x.h)>n(best.h))best=x;}return{value:n(best[key]),t:best.t};
  }
  function microExtreme(kind='low'){
    const rows=(chart.micro||[]).slice(-180).filter(x=>n(x.t)>0&&n(x.c)>0);
    if(!rows.length)return null;const key=kind==='low'?'l':'h';let best=rows[0];
    for(const x of rows){if(kind==='low'?n(x[key])<n(best[key]):n(x[key])>n(best[key]))best=x;}
    return{value:n(best[key]),t:n(best.t)};
  }
  function drawTag(ctx,y,L,R,w,label,value,color){
    const v=n(value);if(!(v>0))return;const yy=y(v);if(!Number.isFinite(yy))return;ctx.save();ctx.setLineDash([3,4]);ctx.strokeStyle=color;ctx.globalAlpha=.72;ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.globalAlpha=1;ctx.setLineDash([]);ctx.font='800 9px system-ui';const text=`${label} ${price(v)}`,tw=ctx.measureText(text).width;ctx.fillStyle='rgba(5,10,18,.9)';ctx.fillRect(L+5,yy-13,tw+7,13);ctx.fillStyle=color;ctx.fillText(text,L+8,yy-3);ctx.restore();
  }
  function drawChart(){const cv=el('multiDipCanvas'),wrap=el('multiDipChartWrap');if(!cv||!wrap)return;const ctx=cv.getContext('2d'),dpr=window.devicePixelRatio||1,w=Math.max(300,wrap.clientWidth),h=Math.max(260,wrap.clientHeight);cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#07101a';ctx.fillRect(0,0,w,h);const rows=fillMinuteGaps(chart.candles).slice(-30);if(!selected||selected==='ETHUSDT'){ctx.fillStyle='#91a2b8';ctx.font='13px system-ui';ctx.fillText('ETH için aşağıdaki V8.4.4 çekirdek grafiğini kullan.',18,34);return;}if(rows.length<2){ctx.fillStyle='#91a2b8';ctx.font='13px system-ui';ctx.fillText('30 dakikalık Binance bağlamı yükleniyor…',18,34);return;}const lev=planLevels(selected),dip=microExtreme('low')||recentSwing(rows,'low'),peak=microExtreme('high')||recentSwing(rows,'high'),levels=[lev.entry,lev.stop,lev.target,lev.trail,chart.last,dip?.value,peak?.value].filter(v=>n(v)>0),vals=rows.flatMap(c=>[c.l,c.h]).concat(levels);let lo=Math.min(...vals),hi=Math.max(...vals),pad=(hi-lo)*.08||hi*.001||.001;lo-=pad;hi+=pad;const L=10,R=82,T=18,B=34,pw=w-L-R,ph=h-T-B,cw=pw/rows.length,y=v=>T+(hi-v)/(hi-lo)*ph;
    for(let i=0;i<=4;i++){const yy=T+ph*i/4,v=hi-(hi-lo)*i/4;ctx.strokeStyle='#152436';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(w-R,yy);ctx.stroke();ctx.fillStyle='#718198';ctx.font='9px system-ui';ctx.fillText(price(v),w-R+6,yy+3);}
    rows.forEach((c,i)=>{const x=L+cw*i+cw/2,up=c.c>=c.o,col=c.synthetic?'#5b7185':up?'#12d996':'#ff5068',body=Math.max(2,Math.abs(y(c.o)-y(c.c))),top=Math.min(y(c.o),y(c.c)),ww=Math.max(3,Math.min(10,cw*.62));ctx.strokeStyle=col;ctx.fillStyle=col;ctx.globalAlpha=c.synthetic?.38:1;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,y(c.h));ctx.lineTo(x,y(c.l));ctx.stroke();ctx.fillRect(x-ww/2,top,ww,body);ctx.globalAlpha=1;});
    const step=Math.max(1,Math.round(rows.length/6));ctx.font='9px system-ui';ctx.fillStyle='#627890';for(let i=0;i<rows.length;i+=step){const d=new Date(rows[i].t),txt=d.toLocaleTimeString('tr-TR',{hour:'2-digit',minute:'2-digit'}),x=L+cw*i+cw/2;ctx.fillText(txt,Math.max(L,Math.min(w-R-28,x-14)),h-10);}
    if(dip?.value>0)drawTag(ctx,y,L,R,w,'SON DİP 1s',dip.value,'#9b8cff');if(peak?.value>0)drawTag(ctx,y,L,R,w,'SON TEPE 1s',peak.value,'#54bfff');
    drawLine(ctx,y,L,R,w,'ENTRY',lev.entry,'#62a9ff');drawLine(ctx,y,L,R,w,'STOP',lev.stop,'#ff6278');drawLine(ctx,y,L,R,w,'HEDEF',lev.target,'#35f0ae');if(lev.trail>0)drawLine(ctx,y,L,R,w,'TRAIL',lev.trail,'#f3c969');drawLine(ctx,y,L,R,w,'CANLI',chart.last,'#e8f4ff');}

  async function seedChart(symbol,seq){
    let lastErr=null;
    for(const base of REST_HOSTS){
      try{
        const [r1m,r1s]=await Promise.all([
          fetch(`${base}/api/v3/klines?symbol=${encodeURIComponent(symbol)}&interval=1m&limit=30`,{cache:'no-store'}),
          fetch(`${base}/api/v3/klines?symbol=${encodeURIComponent(symbol)}&interval=1s&limit=180`,{cache:'no-store'})
        ]);
        if(!r1m.ok)throw Error(`1m HTTP ${r1m.status}`);
        const raw1m=await r1m.json();
        if(!Array.isArray(raw1m)||raw1m.length<2)throw Error('1m MUM YOK');
        let raw1s=[];
        if(r1s.ok){const z=await r1s.json();if(Array.isArray(z))raw1s=z;}
        if(seq!==chart.seq)return;
        chart.candles=fillMinuteGaps(raw1m.map(a=>({t:n(a[0]),o:n(a[1]),h:n(a[2]),l:n(a[3]),c:n(a[4])})));
        chart.micro=raw1s.map(a=>({t:n(a[0]),o:n(a[1]),h:n(a[2]),l:n(a[3]),c:n(a[4])})).filter(x=>x.t>0&&x.c>0).slice(-180);
        chart.last=chart.micro.at(-1)?.c||chart.candles.at(-1)?.c||0;
        chart.source=chart.micro.length?'30DK + 3DK 1s DİP + LIVE':'30DK BAĞLAM + 1s LIVE';
        drawChart();renderChartHead();return;
      }catch(e){lastErr=e;}
    }
    if(seq===chart.seq){chart.source='REST HATA';renderChartHead();drawChart();console.warn('multi dip chart seed',lastErr);}
  }
  function stopSocket(){if(chart.reconnect){clearTimeout(chart.reconnect);chart.reconnect=null;}if(chart.ws){try{chart.ws.onclose=null;chart.ws.close();}catch{}chart.ws=null;}}
  function connectSocket(symbol,seq){stopSocket();if(!symbol||symbol==='ETHUSDT'||document.hidden)return;const url=`wss://stream.binance.com:9443/ws/${symbol.toLowerCase()}@kline_1s`;try{const ws=new WebSocket(url);chart.ws=ws;ws.onopen=()=>{if(seq!==chart.seq)return;chart.source='30DK + 3DK 1s DİP + LIVE';renderChartHead();};ws.onmessage=ev=>{if(seq!==chart.seq)return;try{const d=JSON.parse(ev.data),k=d?.k;if(!k)return;const mt=n(k.t),px=n(k.c),hi=n(k.h),lo=n(k.l),op=n(k.o),bucket=minuteBucket(mt);if(!(mt>0&&bucket>0&&px>0))return;
      const mc={t:mt,o:op||px,h:hi||px,l:lo||px,c:px},mlast=chart.micro.at(-1);
      if(mlast?.t===mc.t)chart.micro[chart.micro.length-1]=mc;else if(!mlast||mc.t>mlast.t)chart.micro.push(mc);
      if(chart.micro.length>180)chart.micro.splice(0,chart.micro.length-180);
      let last=chart.candles.at(-1);if(!last||last.t<bucket){const prev=n(last?.c,px);if(last&&bucket>last.t+60000){for(let t=last.t+60000;t<bucket;t+=60000)chart.candles.push({t,o:prev,h:prev,l:prev,c:prev,synthetic:true});}chart.candles.push({t:bucket,o:prev,h:Math.max(prev,hi,px),l:Math.min(prev,lo||px,px),c:px});}else if(last.t===bucket){last.h=Math.max(n(last.h,px),hi,px);last.l=Math.min(n(last.l,px),lo||px,px);last.c=px;}else{return;}if(chart.candles.length>40)chart.candles.splice(0,chart.candles.length-40);chart.last=px;positionTicks.set(symbol,{price:px,at:Date.now()});renderChartHead();renderPositionRows();drawChart();}catch{}};ws.onerror=()=>{chart.source='WS RETRY';renderChartHead();};ws.onclose=()=>{if(seq!==chart.seq||document.hidden)return;chart.reconnect=setTimeout(()=>connectSocket(symbol,seq),1800);};}catch{chart.reconnect=setTimeout(()=>connectSocket(symbol,seq),2200);}}
  function renderChartHead(){setText(el('mdChartTitle'),selected&&selected!=='ETHUSDT'?`${selected} · Binance Spot · 30dk bağlam / 1s LIVE`:'ETHUSDT · V8.4.4 çekirdek');setText(el('mdChartPrice'),selected==='ETHUSDT'?'ETH ↓':price(chart.last));setText(el('mdChartState'),selected==='ETHUSDT'?'ÇEKİRDEK GRAFİK':chart.source);const ev=selected&&selected!=='ETHUSDT'?latestEval(selected):null;setText(el('mdChartMeta'),selected==='ETHUSDT'?'Detaylı Brian çizgileri mevcut ETH grafiğinde.':ev?`${ev.action||'WAIT'} · ${ev.reason||'—'} · signal ${Math.round(n(ev.signal_score)*100)}/100`:'Brian planı / son işlem seviyeleri');}
  function choose(symbol){symbol=safeSymbol(symbol);if(!symbol)return;if(symbol==='ETHUSDT'){selected='ETHUSDT';chart.seq++;stopSocket();renderTabs();renderLists();renderPlan();renderChartHead();drawChart();el('chartPanel')?.scrollIntoView({behavior:'smooth',block:'start'});return;}if(selected===symbol&&chart.symbol===symbol){el('multiDipChartWrap')?.scrollIntoView({behavior:'smooth',block:'center'});return;}selected=symbol;chart.symbol=symbol;chart.candles=[];chart.micro=[];chart.last=0;chart.source='YÜKLENİYOR';const seq=++chart.seq;renderTabs();renderLists();renderPlan();renderChartHead();drawChart();seedChart(symbol,seq);connectSocket(symbol,seq);el('multiDipChartWrap')?.scrollIntoView({behavior:'smooth',block:'center'});}

  function render(data){latest=data;for(const p of data?.positions||[]){const s=safeSymbol(p.symbol);if(s&&!positionTicks.has(s))positionTicks.set(s,{price:n(p.mark,n(p.entry)),at:0});}ensure();const c=core(),linked=!data?.source_session_id||!c.session?.session_id||String(data.source_session_id)===String(c.session.session_id),state=el('multiDipState'),running=String(data?.status||'')==='RUNNING';if(state){state.textContent=running?`DIP CANLI · ${Math.round(n(data.age_seconds))} sn`:String(data?.status||'WAIT');state.style.color=running?'#35f0ae':'#f3c969';}setText(el('multiDipSession'),linked?`Tek session bağlı · ${String(data?.source_session_id||c.session?.session_id||'—')} · SHADOW ONLY · gerçek emir KAPALI`:`SESSION EŞLEŞMİYOR · ETH ${c.session?.session_id||'—'} · altcoin ${data?.source_session_id||'—'} · Yeni SHADOW session ile eşleştir.`);if(!selected){const first=(data?.positions||[])[0]?.symbol||(data?.recent_events||[])[0]?.symbol||(data?.recent_evaluations||[])[0]?.symbol;selected=safeSymbol(first)||'ETHUSDT';if(selected!=='ETHUSDT'){chart.symbol=selected;const seq=++chart.seq;seedChart(selected,seq);connectSocket(selected,seq);}}applyUnifiedSummary();renderTabs();renderArena();renderLists();renderPlan();renderChartHead();drawChart();}
  async function refresh(){ensure();const key=localStorage.getItem(KEY)||'';if(!key){setText(el('multiDipState'),'KİLİTLİ');return;}try{const r=await fetch(API,{method:'POST',headers:{'content-type':'application/json','x-brian-dashboard-key':key},body:JSON.stringify({action:'status'}),cache:'no-store'});const d=await r.json().catch(()=>({}));if(!r.ok)throw Error(d.error||`HTTP ${r.status}`);render(d);}catch(e){const s=el('multiDipState');if(s){s.textContent='VERİ HATASI';s.style.color='#ff6278';}console.warn('multi dip status',e);}}
  window.addEventListener('load',()=>{ensure();guardOverview();refresh();setInterval(()=>{if(!document.hidden)refresh();},6000);const wrap=el('multiDipChartWrap');if(wrap&&window.ResizeObserver)new ResizeObserver(()=>drawChart()).observe(wrap);});
  window.addEventListener('dip:session-restarted',()=>{latest=null;selected='';chart.seq++;stopSocket();});
  document.addEventListener('visibilitychange',()=>{if(document.hidden){stopSocket();return;}refresh();if(selected&&selected!=='ETHUSDT')connectSocket(selected,chart.seq);});
})();
