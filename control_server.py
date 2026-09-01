# control_server.py — Dashboard + process kontrol + chat
import os, json, asyncio, subprocess, signal
from typing import List, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from brain_engine import Brian

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BRAIN = Brian()
PROC = None
EVENTS: List[Dict] = []      # kısa buffer
WS_CLIENTS: List[WebSocket] = []

HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Monster Coins Pro — Control Panel</title>
<style>
  html,body{margin:0;height:100%;background:#0b0f14;color:#e9eef5;font-family:Inter,system-ui,Segoe UI,Roboto;}
  .wrap{display:grid;grid-template-columns: 420px 1fr; height:100vh;}
  .left{border-right:1px solid #1b2533; display:flex; flex-direction:column;}
  .brand{padding:14px 16px; font-weight:700; letter-spacing:.5px; background:#0f141b;}
  .controls{padding:10px; display:flex; gap:8px; border-bottom:1px solid #1b2533;}
  button{background:#1f6feb;border:none;color:#fff;padding:10px 14px;border-radius:10px;cursor:pointer}
  button.stop{background:#d04a4a}
  .chat{flex:1; display:flex; flex-direction:column; padding:10px; gap:8px; overflow:auto}
  .msg{background:#111823; border:1px solid #1b2533; border-radius:10px; padding:10px}
  .you{background:#131d2a}
  .send{display:flex; gap:8px; border-top:1px solid #1b2533; padding:10px}
  .send input{flex:1; padding:12px; border-radius:10px; border:1px solid #263446; background:#0d141d; color:#e9eef5}
  .send button{padding:10px 16px}
  .right{display:grid; grid-template-rows: 280px 1fr; }
  .panel{padding:10px; overflow:auto}
  .panel h3{margin:6px 0 10px 0}
  .box{background:#0d141d; border:1px solid #1b2533; border-radius:10px; padding:10px}
  .list{display:flex; flex-direction:column; gap:6px}
  .row{display:flex; justify-content:space-between; gap:10px; padding:6px 8px; border-radius:8px; background:#0f1824}
  .tag{font-size:12px; opacity:.8}
</style>
</head>
<body>
<div class="wrap">
  <div class="left">
    <div class="brand">Monster Coins Pro — Boss & Brian</div>
    <div class="controls">
      <button onclick="start()">Start</button>
      <button class="stop" onclick="stop()">Stop</button>
      <div id="status" style="padding:10px 6px;opacity:.85">status: ...</div>
    </div>
    <div id="chat" class="chat"></div>
    <div class="send">
      <input id="in" placeholder="Brian'a yaz... örn: Telegram niye gelmiyor?"/>
      <button onclick="sendChat()">Gönder</button>
    </div>
  </div>
  <div class="right">
    <div class="panel">
      <h3>Akış (Events & Trades)</h3>
      <div id="events" class="box list"></div>
    </div>
    <div class="panel">
      <h3>Özet</h3>
      <div id="summary" class="box">Canlı özet burada görünecek…</div>
    </div>
  </div>
</div>
<script>
const chatDiv = document.getElementById('chat');
const evDiv = document.getElementById('events');
const statusDiv = document.getElementById('status');

function pushMsg(who, text){
  const d=document.createElement('div');
  d.className='msg '+(who==='you'?'you':'');
  d.innerText = (who==='you'?'Boss: ':'Brian: ')+text;
  chatDiv.appendChild(d);
  chatDiv.scrollTop = chatDiv.scrollHeight;
}
async function start(){
  const r=await fetch('/start'); const j=await r.json();
  statusDiv.innerText = 'status: '+j.status;
}
async function stop(){
  const r=await fetch('/stop'); const j=await r.json();
  statusDiv.innerText = 'status: '+j.status;
}
async function stat(){
  const r=await fetch('/status'); const j=await r.json();
  statusDiv.innerText = 'status: '+(j.running?'running':'stopped');
}
async function sendChat(){
  const inp=document.getElementById('in'); const msg=inp.value.trim();
  if(!msg) return;
  pushMsg('you', msg); inp.value='';
  const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});
  const j=await r.json();
  pushMsg('ai', j.reply||'(boş)');
  if(j.intent && j.intent!=='none'){
    const d=document.createElement('div');
    d.className='row';
    d.innerHTML='<div>'+j.intent+'</div><div class="tag">'+JSON.stringify(j.args||{})+'</div>';
    evDiv.prepend(d);
  }
}
let ws=null;
function wsInit(){
  ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');
  ws.onmessage = (e)=>{
    const j=JSON.parse(e.data);
    if(j.type==='event'||j.type==='trade'){
      const d=document.createElement('div'); d.className='row';
      d.innerHTML='<div>'+j.type.toUpperCase()+'</div><div class="tag">'+JSON.stringify(j.data)+'</div>';
      evDiv.prepend(d);
    }else if(j.type==='status'){
      statusDiv.innerText = 'status: '+j.data;
    }
  };
  ws.onclose = ()=> setTimeout(wsInit, 1500);
}
wsInit(); stat();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def idx():
    return HTML

@app.get("/status")
async def status():
    return {"running": PROC is not None and PROC.poll() is None}

@app.get("/start")
async def start():
    global PROC
    if PROC and PROC.poll() is None:
        return {"status":"already-running"}
    # Fast trader’ı başlat
    PROC = subprocess.Popen(
        ["python", "main.py", "--config", "config_live.json"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    asyncio.create_task(pipe_logs(PROC))
    await broadcast({"type":"status","data":"running"})
    return {"status":"started"}

@app.get("/stop")
async def stop():
    global PROC
    if PROC and PROC.poll() is None:
        PROC.terminate()
        try: PROC.wait(timeout=5)
        except: PROC.kill()
    PROC = None
    await broadcast({"type":"status","data":"stopped"})
    return {"status":"stopped"}

@app.post("/chat")
async def chat(req: Request):
    j = await req.json()
    msg = j.get("message","")
    out = BRAIN.chat(msg)
    return JSONResponse(out)

@app.post("/event")
async def accept_event(req: Request):
    j = await req.json()
    EVENTS.append(j)
    if len(EVENTS)>400: EVENTS[:] = EVENTS[-400:]
    await broadcast(j)
    return {"ok": True}

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    WS_CLIENTS.append(ws)
    try:
        # son 40 olayı gönder
        for e in EVENTS[-40:]:
            await ws.send_text(json.dumps(e, ensure_ascii=False))
        while True:
            await ws.receive_text()  # ping/pong
    except WebSocketDisconnect:
        pass
    finally:
        if ws in WS_CLIENTS: WS_CLIENTS.remove(ws)

async def broadcast(msg: dict):
    dead=[]
    for c in WS_CLIENTS:
        try:
            await c.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:
            dead.append(c)
    for d in dead:
        try: WS_CLIENTS.remove(d)
        except: pass

async def pipe_logs(proc: subprocess.Popen):
    loop = asyncio.get_event_loop()
    while proc and proc.poll() is None:
        line = await loop.run_in_executor(None, proc.stdout.readline)
        if not line: await asyncio.sleep(0.05); continue
        EVENTS.append({"type":"event","data":{"ts":"", "kind":"stdout", "msg": line.strip()}})
        if len(EVENTS)>400: EVENTS[:] = EVENTS[-400:]
        await broadcast({"type":"event","data":{"kind":"stdout","msg": line.strip()}})

# uvicorn control_server:app --host 0.0.0.0 --port 8000
