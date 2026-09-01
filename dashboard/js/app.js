
import {parseCSV} from './csv.js';
import {fetchJSONL} from './jsonl.js';

const WS_PORT = 8765; // config.ui.socket_port default
const FALLBACK_POLL_MS = 5000;

const el = (q)=>document.querySelector(q);
const els = (q)=>Array.from(document.querySelectorAll(q));

function fmt(n, d=2){ n = Number(n); if (isNaN(n)) return '-'; return n.toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}); }
function pct(n){ if(n===null||n===undefined) return '-'; return (Number(n)*100).toFixed(1)+'%'; }

const state = {
  metrics: { cash: 0, pf: 0, wr: 0, maxdd: 0, open: 0, pnl24h: 0 },
  slots: { pred: 0, dip: 0, news: 0, ob: 0 },
  trades: [],
  events: [],
  telegram: [],
  overrides: []
};

async function tryWebSocket(){
  return new Promise((resolve)=>{
    try{
      const ws = new WebSocket(`ws://localhost:${WS_PORT}`);
      ws.onopen = ()=>resolve(ws);
      ws.onerror = ()=>resolve(null);
    }catch(e){ resolve(null); }
  });
}

function renderKPI(){
  el('#kpi-cash').textContent = '$'+fmt(state.metrics.cash,2);
  el('#kpi-pf').textContent   = fmt(state.metrics.pf,2);
  el('#kpi-wr').textContent   = fmt(state.metrics.wr*100,1)+'%';
  el('#kpi-dd').textContent   = fmt(state.metrics.maxdd,2);
  el('#kpi-open').textContent = state.metrics.open ?? 0;
  el('#kpi-24h').textContent  = fmt(state.metrics.pnl24h,2);
}

function renderSlots(){
  const total = state.slots.pred + state.slots.dip + state.slots.news + state.slots.ob;
  const pred = total? state.slots.pred/total : 0;
  const dip  = total? state.slots.dip/total  : 0;
  const news = total? state.slots.news/total : 0;
  const ob   = total? state.slots.ob/total   : 0;
  const canvas = el('#slotChart');
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.clientWidth;
  const H = canvas.height= canvas.clientHeight;
  ctx.clearRect(0,0,W,H);
  const vals = [pred,dip,news,ob];
  const colors = ['#7c5cff','#9fb0ff','#ffcc00','#00d98b'];
  const labels = ['PRED','DIP','NEWS','OB'];
  // simple donut
  let start= -Math.PI/2;
  const cx=W/2, cy=H/2, r=Math.min(W,H)/2 - 10, inner=r*0.6;
  vals.forEach((v,i)=>{
    const ang = v*2*Math.PI;
    ctx.beginPath();
    ctx.moveTo(cx,cy);
    ctx.arc(cx,cy,r,start,start+ang);
    ctx.closePath();
    ctx.fillStyle = colors[i];
    ctx.globalAlpha = .9;
    ctx.fill();
    start+=ang;
  });
  // inner hole
  ctx.globalCompositeOperation='destination-out';
  ctx.beginPath(); ctx.arc(cx,cy,inner,0,Math.PI*2); ctx.fill();
  ctx.globalCompositeOperation='source-over';
  // legend
  const legend=el('#slotLegend'); legend.innerHTML='';
  vals.forEach((v,i)=>{
    const div=document.createElement('div'); div.className='tag';
    div.innerHTML=`<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${colors[i]};margin-right:6px"></span>${labels[i]} ${Math.round(v*100)}%`;
    legend.appendChild(div);
  });
}

function renderTrades(){
  const tbody = el('#trades-body'); tbody.innerHTML='';
  const q = el('#search').value.toUpperCase();
  const rows = state.trades.filter(t=> (t.symbol||'').toUpperCase().includes(q));
  rows.slice(-200).reverse().forEach(t=>{
    const tr = document.createElement('tr');
    const side = (t.side||'').toUpperCase();
    const pnl  = Number(t.pnl||0);
    tr.innerHTML = `
      <td>${t.time||''}</td>
      <td>${t.symbol||''}</td>
      <td><span class="badge ${side==='BUY'?'good':'bad'}">${side||'-'}</span></td>
      <td>${t.slot||t.reason||'-'}</td>
      <td>${t.qty||'-'}</td>
      <td>${t.price||'-'}</td>
      <td class="${pnl>=0?'good':'bad'}">${fmt(pnl,2)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderEvents(){
  const box = el('#events'); box.innerHTML='';
  state.events.slice(-100).reverse().forEach(ev=>{
    const div=document.createElement('div'); div.className='flex wrap small';
    const t = ev.ts || ev.time || '';
    const m = ev.msg || ev.event || JSON.stringify(ev);
    div.innerHTML = `<span class="tag">${t}</span><span>${m}</span>`;
    box.appendChild(div);
  });
}

function renderOverrides(){
  const box = el('#overrides'); box.innerHTML='';
  state.overrides.slice(-100).reverse().forEach(o=>{
    const set = o.set || o.change || o;
    const keys = set? Object.keys(set) : [];
    const msg = keys.map(k=>`${k}: ${set[k]}`).join(', ');
    const t = o.ts || '';
    const src = o.source || 'override';
    const div=document.createElement('div'); div.className='flex wrap small';
    div.innerHTML = `<span class="tag">${src}</span><span class="tag">${t}</span><span>${msg}</span>`;
    box.appendChild(div);
  });
}

function drawLine(canvasId, series){
  const canvas = el(canvasId);
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.clientWidth;
  const H = canvas.height= canvas.clientHeight;
  ctx.clearRect(0,0,W,H);
  if(!series.length) return;
  const xs = series.map((_,i)=>i);
  const ys = series;
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad=10;
  function xTo(i){ return pad + (W-2*pad)*(i/(xs.length-1||1)); }
  function yTo(v){ return H-pad - (H-2*pad)*((v-minY)/((maxY-minY)||1)); }
  ctx.beginPath();
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#9fb0ff';
  ys.forEach((v,i)=>{
    const x=xTo(i), y=yTo(v);
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.stroke();
}

async function pollFiles(){
  try{
    // CSVs
    const tradesCSV = await fetch('..//logs/trades_full_log.csv?ts=' + Date.now(), {cache:'no-store'});
    const eventsCSV = await fetch('..//logs/events.csv?ts=' + Date.now(), {cache:'no-store'});
    if(tradesCSV.ok){
      const text = await tradesCSV.text();
      const rows = parseCSV(text);
      state.trades = rows.map(r=>({
        time: r.time || r.ts || r.timestamp || '',
        symbol: r.symbol || r.sym,
        side: r.side,
        slot: r.slot || r.reason || r.strategy || '',
        qty: r.qty || r.quantity,
        price: r.price || r.fill_price || r.entry,
        pnl: r.pnl || r.p_l || r.profit
      }));
      // derive PnL series (last 150)
      const pnlSeries = state.trades.slice(-150).map(t=> Number(t.pnl)||0);
      drawLine('#pnlChart', pnlSeries);
      // quick metrics approximation
      const wins = state.trades.filter(t=>Number(t.pnl||0)>0).length;
      const wr = state.trades.length? wins/state.trades.length : 0;
      const pf = (()=>{
        let g=0,l=0; state.trades.forEach(t=>{ const p=Number(t.pnl||0); if(p>=0) g+=p; else l+=-p; }); return l? g/l : 0;
      })();
      state.metrics.wr = wr;
      state.metrics.pf = pf;
    }
    if(eventsCSV.ok){
      const text = await eventsCSV.text();
      const rows = parseCSV(text);
      state.events = rows.map(r=>({ts:r.ts||r.time,msg:r.event||r.msg||r.message||JSON.stringify(r)}));
    }
    // Telegram JSONL (pick any latest)
    // naive: try today's file; fallback ignore
    const d = new Date(); const yyyy=d.getFullYear(); const mm=String(d.getMonth()+1).padStart(2,'0'); const dd=String(d.getDate()).padStart(2,'0');
    const teleUrl = `..//logs/telegram_out/${yyyy}-${mm}-${dd}.jsonl`;
    try{
      state.telegram = await fetchJSONL(teleUrl);
    }catch{ /* ignore */ }
    // Overrides
    try{
      state.overrides = await fetchJSONL('..//runtime/runtime_overrides.jsonl');
    }catch{ /* ignore */ }

    // Guess slot weights from opening reasons (rough)
    const slotCounts = {pred:0,dip:0,news:0,ob:0};
    state.trades.slice(-300).forEach(t=>{
      const r=(t.slot||'').toLowerCase();
      if(r.includes('pred')) slotCounts.pred++;
      else if(r.includes('news')) slotCounts.news++;
      else if(r.includes('dip')) slotCounts.dip++;
      else if(r.includes('ob') || r.includes('order')) slotCounts.ob++;
    });
    state.slots = slotCounts;
    renderKPI(); renderSlots(); renderTrades(); renderEvents(); renderOverrides();
  }catch(e){
    console.warn('poll error', e);
  }
}

async function boot(){
  // Login gate (simple front-only)
  const login = el('#login'); const gate = el('#login-screen');
  login.addEventListener('submit', (ev)=>{
    ev.preventDefault();
    gate.style.display='none';
  });

  // Try WS
  const ws = await tryWebSocket();
  if(ws){
    console.log('WS connected');
    ws.onmessage = (ev)=>{
      try{
        const msg = JSON.parse(ev.data);
        if(msg.type==='metrics') Object.assign(state.metrics, msg.data||{});
        if(msg.type==='trades')  state.trades = (msg.data||[]);
        if(msg.type==='events')  state.events = (msg.data||[]);
        if(msg.type==='slots')   state.slots  = (msg.data||state.slots);
        if(msg.type==='overrides') state.overrides = (msg.data||[]);
        renderKPI(); renderSlots(); renderTrades(); renderEvents(); renderOverrides();
      }catch(e){ /* ignore */ }
    };
  }else{
    // fallback polling
    await pollFiles();
    setInterval(pollFiles, FALLBACK_POLL_MS);
  }
}

window.addEventListener('load', boot);
