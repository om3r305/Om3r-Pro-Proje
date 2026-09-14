'use strict';

(()=>{
  const API='https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-dip-multiasset-status';
  const KEY='mcp-dashboard-key-v1';
  const el=id=>document.getElementById(id);
  const n=(v,f=0)=>Number.isFinite(Number(v))?Number(v):f;
  const money=v=>`${n(v)>=0?'+':''}$${n(v).toFixed(2)}`;
  const price=v=>{const x=n(v);if(!(x>0))return '—';const d=x<.01?6:x<1?5:x<100?3:2;return x.toLocaleString('en-US',{maximumFractionDigits:d,minimumFractionDigits:Math.min(2,d)});};
  const esc=s=>String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
  const time=v=>{const d=new Date(String(v||''));return Number.isFinite(d.getTime())?d.toLocaleTimeString('tr-TR',{hour:'2-digit',minute:'2-digit',second:'2-digit'}):'—';};

  function style(){
    if(el('multiDipStyle'))return;
    const s=document.createElement('style');s.id='multiDipStyle';s.textContent=`
      #multiDipPanel{margin:14px 0;overflow-anchor:none}
      #multiDipPanel .md-head{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;margin-bottom:12px}
      #multiDipPanel .md-title{font-size:18px;font-weight:900;color:#eef6ff}
      #multiDipPanel .md-sub{font-size:12px;color:#8497ab;line-height:1.45;margin-top:3px}
      #multiDipPanel .md-live{font-size:11px;font-weight:900;color:#35f0ae;white-space:nowrap}
      #multiDipPanel .md-kpis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-bottom:12px}
      #multiDipPanel .md-kpi{padding:10px;border:1px solid #193248;border-radius:10px;background:rgba(7,17,28,.62)}
      #multiDipPanel .md-label{font-size:9px;letter-spacing:.08em;text-transform:uppercase;color:#8193a8}
      #multiDipPanel .md-val{font-size:18px;font-weight:900;color:#edf6ff;margin-top:4px;font-variant-numeric:tabular-nums}
      #multiDipPanel .md-val.pos{color:#35f0ae}#multiDipPanel .md-val.neg{color:#ff6278}
      #multiDipPanel .md-cols{display:grid;grid-template-columns:1fr 1.35fr;gap:10px}
      #multiDipPanel .md-box{border:1px solid #193248;border-radius:10px;padding:10px;background:rgba(7,17,28,.52)}
      #multiDipPanel .md-box-title{font-size:11px;font-weight:900;letter-spacing:.06em;color:#9ab0c5;text-transform:uppercase;margin-bottom:8px}
      #multiDipPanel .md-list{display:grid;gap:6px}
      #multiDipPanel .md-row{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:8px 9px;border:1px solid #193248;border-radius:8px;background:#08131f}
      #multiDipPanel .md-main{font-size:12px;font-weight:850;color:#edf6ff}
      #multiDipPanel .md-meta{font-size:10px;color:#8193a8;margin-top:2px;line-height:1.35}
      #multiDipPanel .md-side{font-size:11px;font-weight:900;white-space:nowrap}.md-side.pos{color:#35f0ae}.md-side.neg{color:#ff6278}.md-side.wait{color:#f3c969}
      #multiDipPanel .md-table{width:100%;border-collapse:collapse;font-size:10px}#multiDipPanel .md-table th,#multiDipPanel .md-table td{padding:7px 5px;border-bottom:1px solid #152b3e;text-align:left;white-space:nowrap}#multiDipPanel .md-table th{color:#8193a8;font-size:9px;text-transform:uppercase}#multiDipPanel .md-table td{color:#dce9f6}
      #multiDipPanel .md-scroll{overflow:auto;max-height:285px}
      @media(max-width:760px){#multiDipPanel .md-kpis{grid-template-columns:repeat(2,minmax(0,1fr))}#multiDipPanel .md-cols{grid-template-columns:1fr}#multiDipPanel .md-title{font-size:17px}#multiDipPanel .md-scroll{max-height:330px}}
    `;document.head.appendChild(s);
  }

  function ensure(){
    style();
    if(el('multiDipPanel'))return el('multiDipPanel');
    const overview=el('overview');if(!overview)return null;
    const panel=document.createElement('section');panel.id='multiDipPanel';panel.className='card';
    panel.innerHTML=`<div class="md-head"><div><div class="md-title">⚡ DIP Çoklu Coin SHADOW</div><div class="md-sub">ETH çekirdek grafiğinden ayrı DIP altcoin motoru. Gerçek emir yok; SHADOW BUY / SELL kayıtları canlı gösterilir.</div></div><div id="multiDipState" class="md-live">BAĞLANIYOR</div></div><div class="md-kpis"><div class="md-kpi"><div class="md-label">Altcoin Equity</div><div id="mdEquity" class="md-val">—</div></div><div class="md-kpi"><div class="md-label">Realized P&L</div><div id="mdPnl" class="md-val">—</div></div><div class="md-kpi"><div class="md-label">Açık Pozisyon</div><div id="mdOpen" class="md-val">—</div></div><div class="md-kpi"><div class="md-label">Win Rate</div><div id="mdWin" class="md-val">—</div></div><div class="md-kpi"><div class="md-label">Kapalı İşlem</div><div id="mdTrades" class="md-val">—</div></div></div><div class="md-cols"><div class="md-box"><div class="md-box-title">Açık pozisyonlar</div><div id="mdPositions" class="md-list"><div class="md-row"><div><div class="md-main">Yükleniyor…</div></div><span class="md-side wait">WAIT</span></div></div></div><div class="md-box"><div class="md-box-title">Son AL / SAT işlemleri</div><div class="md-scroll"><table class="md-table"><thead><tr><th>Saat</th><th>Coin</th><th>Olay</th><th>Fiyat</th><th>P&L</th><th>Neden</th></tr></thead><tbody id="mdEvents"><tr><td colspan="6">Yükleniyor…</td></tr></tbody></table></div></div></div>`;
    overview.insertAdjacentElement('afterend',panel);
    const nav=document.querySelector('.side .nav');if(nav&&!nav.querySelector('[data-multi-dip-link]')){const a=document.createElement('a');a.href='#multiDipPanel';a.dataset.multiDipLink='1';a.innerHTML='<span>⚡</span><span>DIP İşlemleri</span>';const radar=nav.querySelector('[data-alpha-radar-link]');if(radar)nav.insertBefore(a,radar);else nav.appendChild(a);}
    return panel;
  }

  function markEthOverview(){
    const overview=el('overview');if(!overview||overview.dataset.ethCoreMarked)return;overview.dataset.ethCoreMarked='1';
    const first=overview.querySelector('.kpi .label');if(first&&!first.textContent.includes('ETH'))first.textContent='ETH Çekirdek Equity';
    const trade=overview.querySelectorAll('.kpi .label')[4];if(trade&&!trade.textContent.includes('ETH'))trade.textContent='ETH İşlem';
  }

  function render(data){
    ensure();markEthOverview();
    const state=el('multiDipState');const running=String(data?.status||'')==='RUNNING';
    if(state){state.textContent=running?`CANLI · ${Math.round(n(data.age_seconds))} sn`:String(data?.status||'WAIT');state.style.color=running?'#35f0ae':'#f3c969';}
    const eq=n(data?.equity),pnl=n(data?.realized_pnl),trades=n(data?.trade_count),wins=n(data?.win_count),losses=n(data?.loss_count),wr=trades>0?wins/trades:0,positions=Array.isArray(data?.positions)?data.positions:[];
    const set=(id,value,cls)=>{const x=el(id);if(!x)return;x.textContent=value;if(cls)x.className=`md-val ${cls}`;};
    set('mdEquity',`$${eq.toFixed(2)}`,eq>=n(data?.starting_equity,1000)?'pos':'neg');set('mdPnl',money(pnl),pnl>=0?'pos':'neg');set('mdOpen',String(positions.length));set('mdWin',trades?`${(wr*100).toFixed(1)}%`:'—');set('mdTrades',String(trades));
    const posBox=el('mdPositions');if(posBox){posBox.innerHTML=positions.length?positions.map(p=>{const up=n(p.mark)>=n(p.entry);return `<div class="md-row"><div><div class="md-main">${esc(p.symbol)}</div><div class="md-meta">Entry ${price(p.entry)} · Son ${price(p.mark)} · Stop ${price(p.stop)} · Hedef ${price(p.target)}</div></div><span class="md-side ${up?'pos':'neg'}">${money(n(p.unrealized_pnl))}</span></div>`;}).join(''):'<div class="md-row"><div><div class="md-main">Açık altcoin pozisyonu yok</div><div class="md-meta">Radar DIP setup bekliyor.</div></div><span class="md-side wait">0</span></div>';}
    const events=Array.isArray(data?.recent_events)?data.recent_events:[],body=el('mdEvents');if(body){body.innerHTML=events.length?events.slice(0,24).map(e=>{const sell=String(e.action)==='SELL',pv=e.pnl==null?null:n(e.pnl);return `<tr><td>${esc(time(e.observed_at))}</td><td><b>${esc(e.symbol)}</b></td><td>${esc(e.action)}</td><td>${esc(price(e.price))}</td><td class="${pv==null?'':pv>=0?'pos':'neg'}">${pv==null?'—':esc(money(pv))}</td><td>${esc(e.reason)}</td></tr>`;}).join(''):'<tr><td colspan="6">Henüz altcoin DIP işlemi yok.</td></tr>';}
  }

  async function refresh(){
    ensure();markEthOverview();const key=localStorage.getItem(KEY)||'';if(!key){const s=el('multiDipState');if(s)s.textContent='KİLİTLİ';return;}
    try{const r=await fetch(API,{method:'POST',headers:{'content-type':'application/json','x-brian-dashboard-key':key},body:'{}',cache:'no-store'});if(!r.ok)throw Error(`HTTP ${r.status}`);render(await r.json());}catch(e){const s=el('multiDipState');if(s){s.textContent='VERİ HATASI';s.style.color='#ff6278';}}
  }

  window.addEventListener('load',()=>{ensure();markEthOverview();refresh();setInterval(()=>{if(!document.hidden)refresh();},8000);});
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh();});
})();
