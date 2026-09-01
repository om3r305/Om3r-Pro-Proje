# -*- coding: utf-8 -*-
"""
Kurulum: Proje kökünde çalıştır.
Oluşturur:
- Proje1/dashboard/ (index.html, style.css, js/*, assets/)
- Proje1/core/dashboard_api.py (Start/Stop + WS köprüsü)
- Proje1/scripts/runner.bat / runner.sh
Not: logo.png ve avatar.png dosyalarını sonradan assets altına kopyalayın.
"""

import os, json, textwrap
from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware


ROOT = Path(__file__).resolve().parent
PROJ = ROOT / "Proje1"
DASH = PROJ / "dashboard"
ASSETS = DASH / "assets"
JS = DASH / "js"
CORE = PROJ / "core"
SCRIPTS = PROJ / "scripts"

for p in [DASH, ASSETS, JS, CORE, SCRIPTS]:
    p.mkdir(parents=True, exist_ok=True)

# README
(DASH / "README_DASHBOARD.md").write_text(textwrap.dedent("""
# Monster Coins Pro — Dashboard (Pro)
- Neon siyah-yeşil tema, glow/pulse animasyonlar
- Tek şifreli giriş (varsayılan: Monster) — Settings'ten localStorage ile değişir
- Start/Stop (gerçek): `python -m Proje1.main --config Proje1/config_live.json`
- Durum ışıkları: RUNNING(yeşil) / STOPPED(kırmızı)
- Canlı veri: `ws://localhost:8767/ws`; yoksa `logs/*` dosyalarından polling
- Trade akışı, PnL çizgisi, Slot donut, Events, Overrides

## Çalıştırma
pip install fastapi uvicorn psutil
python Proje1/core/dashboard_api.py
python -m http.server 8766
Aç: http://localhost:8766/Proje1/dashboard/  — Şifre: Monster
""").strip(), encoding="utf-8")

# CSS (black/green flashy)
(DASH / "style.css").write_text("""
:root{
  --bg:#050a0a; --panel:#0b1313; --glass:rgba(255,255,255,.06);
  --txt:#e6ffe6; --muted:#93d1a8; --accent:#00ff88; --danger:#ff3366; --warn:#ffd166;
  --glow: 0 14px 40px rgba(0,255,136,.18);
}
*{box-sizing:border-box}
html,body{height:100%;margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto}
body{background:
 radial-gradient(900px 500px at 10% -10%, rgba(0,255,136,.12), transparent 50%),
 radial-gradient(900px 500px at 110% 10%, rgba(0,128,64,.12), transparent 50%),
 #000; color:var(--txt)}
.container{max-width:1440px;margin:0 auto;padding:24px}
.header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
.brand{display:flex;gap:10px;align-items:center}
.brand img{width:36px;height:36px;filter:drop-shadow(0 0 10px rgba(0,255,136,.5))}
h1{margin:0;font-size:20px;letter-spacing:.4px}
.controls{display:flex;gap:8px;align-items:center}
input,button,select{background:var(--glass);color:var(--txt);border:1px solid rgba(255,255,255,.1);padding:10px 12px;border-radius:12px;outline:none}
button{cursor:pointer;transition:.2s;box-shadow:var(--glow)}
button:hover{transform:translateY(-1px)}
.tag{font-size:11px;padding:4px 10px;border-radius:999px;border:1px solid rgba(255,255,255,.15);color:var(--muted);background:rgba(0,0,0,.3)}
.grid{display:grid;gap:16px}
.grid-4{grid-template-columns:repeat(4, minmax(0,1fr))}
.grid-3{grid-template-columns:repeat(3, minmax(0,1fr))}
.grid-2{grid-template-columns:repeat(2, minmax(0,1fr))}
.card{background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.08);
      border-radius:18px;padding:18px;box-shadow:var(--glow);position:relative;overflow:hidden}
.card h3{margin:0 0 6px 0;color:var(--muted);font-size:13px;letter-spacing:.35px}
.kpi{font-size:28px;font-weight:800}
.good{color:var(--accent)} .bad{color:var(--danger)} .warn{color:var(--warn)}
.status{display:flex;gap:8px;align-items:center}
.dot{width:10px;height:10px;border-radius:50%;box-shadow:0 0 12px rgba(0,0,0,.4)}
.dot.g{background:#00ff88; box-shadow:0 0 16px #00ff88}
.dot.r{background:#ff3366; box-shadow:0 0 16px #ff3366}
.pulse{animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{0%{opacity:.6}50%{opacity:1}100%{opacity:.6}}
.hero{display:grid;grid-template-columns: 1.2fr .8fr;gap:16px}
canvas.chart{width:100%;height:220px;display:block}
.section-title{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.legend{display:flex;gap:6px;flex-wrap:wrap}
.legend .tag{background:rgba(0,255,136,.08)}
table{width:100%;border-collapse:separate;border-spacing:0 8px}
thead th{font-size:12px;color:var(--muted);text-align:left;padding:8px 10px}
tbody td{background:rgba(255,255,255,.03);padding:10px;border-top:1px solid rgba(255,255,255,.08);border-bottom:1px solid rgba(255,255,255,.08)}
tbody tr td:first-child{border-left:1px solid rgba(255,255,255,.08);border-top-left-radius:12px;border-bottom-left-radius:12px}
tbody tr td:last-child{border-right:1px solid rgba(255,255,255,.08);border-top-right-radius:12px;border-bottom-right-radius:12px}
.footer{margin-top:16px;display:flex;gap:8px;align-items:center;color:var(--muted);font-size:12px}
.login{position:fixed;inset:0;display:grid;place-items:center;background:rgba(0,0,0,.65);backdrop-filter:blur(10px);z-index:50}
.login-card{width:min(920px,94vw);display:grid;grid-template-columns:1fr 1fr;background:#0b1313;border:1px solid rgba(255,255,255,.1);
  border-radius:18px;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.45)}
.login-side{padding:24px}
.login-side h2{margin:0 0 6px 0}
.login-side p{margin:0 0 16px 0;color:var(--muted)}
.login-img{position:relative;background:#001a0f;display:grid;place-items:center}
.login-img img{width:100%;height:100%;object-fit:cover;opacity:.9;filter:drop-shadow(0 0 24px rgba(0,255,136,.35))}
.toast{position:fixed;right:16px;bottom:16px;background:#001a0f;border:1px solid rgba(0,255,136,.25);color:var(--txt);padding:12px 14px;border-radius:12px;box-shadow:var(--glow);opacity:0;transform:translateY(10px);transition:.25s}
.toast.show{opacity:1;transform:none}
.small{font-size:12px;color:var(--muted)}
""", encoding="utf-8")

# Tiny helpers
(JS / "csv.js").write_text("export function parseCSV(t){const l=t.trim().split(/\\r?\\n/);const h=l.shift()?.split(',')||[];return l.filter(Boolean).map(r=>{const p=r.split(',');const o={};h.forEach((x,i)=>o[x.trim()]=(p[i]||'').trim());return o;});}", encoding="utf-8")
(JS / "jsonl.js").write_text("export async function fetchJSONL(u,m=10000){const r=await fetch(u,{cache:'no-store'});if(!r.ok)throw new Error('fetch '+u+' '+r.status);const t=await r.text();return t.split(/\\r?\\n/).filter(Boolean).slice(-m).map(l=>{try{return JSON.parse(l)}catch{return null}}).filter(Boolean);}", encoding="utf-8")

# Main app.js (auth+start/stop+status+charts)
(JS / "app.js").write_text("""
import {parseCSV} from './csv.js';
import {fetchJSONL} from './jsonl.js';
const API = 'http://localhost:8767';
const WS  = 'ws://localhost:8767/ws';
const POLL = 5000;
const $ = (q)=>document.querySelector(q);
const fmt = (n,d=2)=>{n=Number(n); if(isNaN(n)) return '-'; return n.toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});};
let token = localStorage.getItem('dash_token')||'';
let password = localStorage.getItem('dash_password')||'Monster';
function toast(msg, kind='ok'){const t=$('#toast');t.textContent=msg;t.classList.add('show');t.style.borderColor=kind==='err'?'rgba(255,51,102,.35)':'rgba(0,255,136,.35)';setTimeout(()=>t.classList.remove('show'),2200);}
async function auth(pw){const res=await fetch(API+'/auth',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw})}); if(!res.ok) throw new Error('auth'); const j=await res.json(); token=j.token; localStorage.setItem('dash_token',token); localStorage.setItem('dash_password',pw);}
async function status(){const r=await fetch(API+'/status'); return r.json();}
async function start(){const r=await fetch(API+'/start',{method:'POST',headers:{'Authorization':'Bearer '+token}}); if(!r.ok) throw new Error('start'); return r.json();}
async function stop(){const r=await fetch(API+'/stop',{method:'POST',headers:{'Authorization':'Bearer '+token}}); if(!r.ok) throw new Error('stop'); return r.json();}
function setRunLight(r){const d=$('#run-dot'),t=$('#run-text'); if(r){d.className='dot g pulse'; t.textContent='RUNNING';} else {d.className='dot r pulse'; t.textContent='STOPPED';}}
function renderKPI(m){$('#kpi-cash').textContent='$'+fmt(m.cash||0,2);$('#kpi-pf').textContent=fmt(m.pf||0,2);$('#kpi-wr').textContent=((m.wr||0)*100).toFixed(1)+'%';$('#kpi-dd').textContent=fmt(m.maxdd||0,2);}
function donut(id,vals,labels,colors){const c=$(id),x=c.getContext('2d');const W=c.width=c.clientWidth,H=c.height=c.clientHeight;const tot=vals.reduce((a,b)=>a+b,0)||1;let s=-Math.PI/2;const r=Math.min(W,H)/2-10,cx=W/2,cy=H/2,inner=r*.62;x.clearRect(0,0,W,H);vals.forEach((v,i)=>{const a=v/tot*2*Math.PI;x.beginPath();x.moveTo(cx,cy);x.arc(cx,cy,r,s,s+a);x.closePath();x.fillStyle=colors[i];x.globalAlpha=.92;x.fill();s+=a;});x.globalCompositeOperation='destination-out';x.beginPath();x.arc(cx,cy,inner,0,Math.PI*2);x.fill();x.globalCompositeOperation='source-over';const leg=$(id+'-legend');leg.innerHTML='';labels.forEach((lb,i)=>{const d=document.createElement('div');d.className='tag';d.innerHTML=`<span style="display:inline-block;width:10px;height:10px;background:${colors[i]};margin-right:6px"></span>${lb} ${(vals[i]/tot*100).toFixed(0)}%`;leg.appendChild(d);});}
function line(id,s){const c=$(id),x=c.getContext('2d');const W=c.width=c.clientWidth,H=c.height=c.clientHeight;x.clearRect(0,0,W,H);if(!s.length)return;const mn=Math.min(...s),mx=Math.max(...s),p=8;const X=i=>p+(W-2*p)*(i/(s.length-1||1));const Y=v=>H-p-(H-2*p)*((v-mn)/((mx-mn)||1));x.beginPath();x.strokeStyle='#00ff88';x.lineWidth=2;s.forEach((v,i)=>{const XX=X(i),YY=Y(v);if(i===0)x.moveTo(XX,YY);else x.lineTo(XX,YY);});x.stroke();}
async function pollFiles(){try{let trades=[];const rt=await fetch('../logs/trades_full_log.csv?ts='+Date.now(),{cache:'no-store'});if(rt.ok){trades=parseCSV(await rt.text());}const series=trades.slice(-150).map(r=>Number(r.pnl||r.PnL||0)||0);line('#pnlChart',series);const c={PRED:0,DIP:0,NEWS:0,OB:0};trades.slice(-300).forEach(r=>{const s=(r.slot||r.reason||'').toUpperCase();if(s.includes('PRED'))c.PRED++;else if(s.includes('DIP'))c.DIP++;else if(s.includes('NEWS'))c.NEWS++;else if(s.includes('OB')||s.includes('ORDER'))c.OB++;});donut('#slotChart',[c.PRED,c.DIP,c.NEWS,c.OB],['PRED','DIP','NEWS','OB'],['#00ff88','#a7ffb3','#ffd166','#26d07c']);const re=await fetch('../logs/events.csv?ts='+Date.now(),{cache:'no-store'});if(re.ok){const evs=parseCSV(await re.text());$('#events').innerHTML=evs.slice(-120).reverse().map(e=>`<div class="tag">${e.ts||e.time}</div> <span class="small">${e.event||e.msg||''}</span>`).join('<br>');}try{const ov=await fetchJSONL('../runtime/runtime_overrides.jsonl');$('#overrides').innerHTML=ov.slice(-80).reverse().map(o=>`<div class="tag">${o.source||'override'}</div> <span class="small">${o.ts||''}</span> <span>${JSON.stringify(o.set||o)}</span>`).join('<br>');}catch(e){}}catch(e){}}
function bindUI(){$('#login-form').addEventListener('submit',async ev=>{ev.preventDefault();const pw=$('#pw').value||'Monster';try{await auth(pw);document.querySelector('.login').style.display='none';toast('Giriş başarılı');}catch(e){toast('Şifre hatalı','err');}});$('#logout').addEventListener('click',()=>{localStorage.removeItem('dash_token');document.querySelector('.login').style.display='grid';});$('#saveSettings').addEventListener('click',()=>{const np=$('#set-password').value.trim();if(np.length<3){toast('En az 3 karakter','err');return;}localStorage.setItem('dash_password',np);toast('Şifre güncellendi');});$('#btn-start').addEventListener('click',async()=>{try{await auth(localStorage.getItem('dash_password')||'Monster');await start();toast('Sistem başlatıldı');refreshStatus();}catch{toast('Başlatılamadı','err');}});$('#btn-stop').addEventListener('click',async()=>{try{await auth(localStorage.getItem('dash_password')||'Monster');await stop();toast('Sistem durduruldu');refreshStatus();}catch{toast('Durdurulamadı','err');}});}
async function refreshStatus(){try{const st=await status();setRunLight(st.running);}catch{setRunLight(false);}}
function bootWS(){try{const ws=new WebSocket(WS);ws.onmessage=(ev)=>{try{const m=JSON.parse(ev.data);if(m.type==='status')setRunLight(m.running);if(m.type==='metrics')renderKPI(m.data||{});}catch{}};}catch{}}
async function boot(){bindUI();setInterval(refreshStatus,3000);setInterval(pollFiles, POLL);bootWS();pollFiles();refreshStatus();try{await auth(localStorage.getItem('dash_password')||'Monster');document.querySelector('.login').style.display='none';}catch{}}
window.addEventListener('load', boot);
""", encoding="utf-8")

# HTML
(DASH / "index.html").write_text("""
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Monster Coins Pro • Dashboard</title>
  <link rel="stylesheet" href="./style.css">
</head>
<body>
  <div class="container">
    <header class="header">
      <div class="brand">
        <img src="./assets/logo.png" alt="logo">
        <h1>Monster Coins Pro — Dashboard</h1>
      </div>
      <div class="controls">
        <div class="status"><div id="run-dot" class="dot r pulse"></div><span id="run-text">STOPPED</span></div>
        <button id="btn-start">Başlat</button>
        <button id="btn-stop">Durdur</button>
        <button id="logout">Çıkış</button>
      </div>
    </header>

    <section class="grid grid-4">
      <div class="card"><h3>Kasa</h3><div class="kpi" id="kpi-cash">$0.00</div></div>
      <div class="card"><h3>Profit Factor</h3><div class="kpi" id="kpi-pf">0.00</div></div>
      <div class="card"><h3>Win Rate</h3><div class="kpi" id="kpi-wr">0.0%</div></div>
      <div class="card"><h3>Max Drawdown</h3><div class="kpi" id="kpi-dd">0.00</div></div>
    </section>

    <section class="hero">
      <div class="card">
        <div class="section-title"><h3>PnL Zaman Serisi</h3></div>
        <canvas id="pnlChart" class="chart" height="220"></canvas>
      </div>
      <div class="card">
        <div class="section-title"><h3>Slot Dağılımı</h3></div>
        <canvas id="slotChart" class="chart" height="220"></canvas>
        <div id="slotChart-legend" class="legend" style="margin-top:8px"></div>
      </div>
    </section>

    <section class="grid grid-3">
      <div class="card">
        <div class="section-title"><h3>Trade Akışı</h3><span class="small">Son 200</span></div>
        <table>
          <thead><tr><th>Zaman</th><th>Sembol</th><th>Yön</th><th>Slot</th><th>Adet</th><th>Fiyat</th><th>PnL</th></tr></thead>
          <tbody id="trades-body"></tbody>
        </table>
      </div>
      <div class="card">
        <div class="section-title"><h3>Olaylar</h3></div>
        <div id="events" style="max-height:380px; overflow:auto"></div>
      </div>
      <div class="card">
        <div class="section-title"><h3>Runtime Overrides</h3></div>
        <div id="overrides" style="max-height:380px; overflow:auto"></div>
      </div>
    </section>

    <footer class="footer">
      <span class="tag">API: http://localhost:8767</span>
      <span class="tag">WS: ws://localhost:8767/ws</span>
      <span class="tag">Static: http://localhost:8766</span>
    </footer>
  </div>

  <div class="login">
    <div class="login-card">
      <div class="login-side">
        <h2>Giriş</h2>
        <p>Tek şifre ile giriş yapın (varsayılan: <b>Monster</b>)</p>
        <form id="login-form">
          <div class="grid grid-2" style="margin-bottom:10px">
            <input id="pw" placeholder="Şifre" type="password" required>
            <button type="submit">Giriş</button>
          </div>
        </form>
        <div class="card" style="margin-top:10px">
          <h3>Ayarlar</h3>
          <div class="grid grid-2">
            <input id="set-password" placeholder="Yeni şifre">
            <button id="saveSettings">Kaydet</button>
          </div>
          <div class="small" style="margin-top:6px">Şifre localStorage'da tutulur.</div>
        </div>
      </div>
      <div class="login-img">
        <img src="./assets/avatar.png" alt="avatar">
      </div>
    </div>
  </div>

  <div id="toast" class="toast">Mesaj</div>
  <script type="module" src="./js/app.js"></script>
</body>
</html>
""", encoding="utf-8")

# FastAPI bridge
(CORE / "dashboard_api.py").write_text("""
# -*- coding: utf-8 -*-
import os, json, subprocess
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException
from fastapi.responses import JSONResponse
import uvicorn, psutil, asyncio

ROOT = Path(__file__).resolve().parents[1]   # .../Proje1
PID_FILE = ROOT / "runtime" / "bot.pid"
RUN_CMD = ["python", "-m", "Proje1.main", "--config", "Proje1/config_live.json"]
API_PORT = int(os.environ.get("DASH_API_PORT", "8767"))
TOKEN = "Monster"   # basit tek şifre

app = FastAPI(title="Dashboard API", docs_url=None, redoc_url=None)

def read_pid()->Optional[int]:
    try:
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text().strip())
            if psutil.pid_exists(pid):
                return pid
    except Exception:
        pass
    return None

def write_pid(pid:int):
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(pid), encoding="utf-8")

def clear_pid():
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except Exception:
        pass

def is_running()->bool:
    pid = read_pid()
    return pid is not None and psutil.pid_exists(pid)

def start_bot():
    if is_running():
        return {"ok": True, "already": True, "pid": read_pid()}
    proc = subprocess.Popen(RUN_CMD, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    write_pid(proc.pid)
    return {"ok": True, "pid": proc.pid}

def stop_bot(timeout=10):
    pid = read_pid()
    if not pid or not psutil.pid_exists(pid):
        clear_pid()
        return {"ok": True, "already": True}
    p = psutil.Process(pid)
    p.terminate()
    try:
        p.wait(timeout=timeout)
    except psutil.TimeoutExpired:
        p.kill()
    clear_pid()
    return {"ok": True}

@app.post("/auth")
async def auth(payload: dict):
    pw = payload.get("password","")
    if pw != TOKEN:
        raise HTTPException(status_code=401, detail="invalid password")
    return {"token": "ok"}

def check_auth(auth: Optional[str]):
    return bool(auth)

@app.get("/status")
async def status():
    pid = read_pid()
    return {"running": is_running(), "pid": pid}

@app.post("/start")
async def api_start(authorization: Optional[str] = Header(default=None)):
    if not check_auth(authorization):
        raise HTTPException(status_code=401, detail="unauthorized")
    return JSONResponse(start_bot())

@app.post("/stop")
async def api_stop(authorization: Optional[str] = Header(default=None)):
    if not check_auth(authorization):
        raise HTTPException(status_code=401, detail="unauthorized")
    return JSONResponse(stop_bot())

def derive_metrics():
    metrics = {"cash": 0, "wr": 0, "pf": 0, "maxdd": 0}
    try:
        tsv = ROOT / "logs" / "trades_full_log.csv"
        if tsv.exists():
            import csv
            wins=0; tot=0; g=0; l=0
            with tsv.open("r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    tot += 1
                    pnl = float(r.get("pnl") or r.get("PnL") or 0)
                    if pnl >= 0: g += pnl; wins += 1
                    else: l += -pnl
            metrics["wr"] = wins/tot if tot else 0
            metrics["pf"] = (g/l) if l else (g and 99 or 0)
    except Exception:
        pass
    try:
        st = ROOT / "runtime" / "state.json"
        if st.exists():
            j = json.loads(st.read_text())
            metrics["cash"] = j.get("cash", metrics["cash"])
            metrics["maxdd"] = j.get("maxdd", metrics["maxdd"])
    except Exception:
        pass
    return metrics

@app.get("/metrics")
async def metrics():
    return {"ok": True, "data": derive_metrics()}

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json({"type":"status","running": is_running()})
            await ws.send_json({"type":"metrics","data": derive_metrics()})
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        return
    except Exception:
        return

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
""", encoding="utf-8")

# Scripts
(SCRIPTS / "runner.bat").write_text(r"""@echo off
pip install fastapi uvicorn psutil
python Proje1\core\dashboard_api.py
""", encoding="utf-8")
(SCRIPTS / "runner.sh").write_text("""#!/usr/bin/env bash
pip install fastapi uvicorn psutil
python Proje1/core/dashboard_api.py
""", encoding="utf-8")

print("OK – Dashboard ve API dosyaları yazıldı.")
