# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, subprocess, time
from pathlib import Path
PID=Path("runtime/bot.pid")
RUN_CMD=["python","-m","Proje1.main","--config","Proje1/config_live.json"]
def read_pid():
    try:
        if PID.exists():
            pid=int(PID.read_text().strip())
            return pid if pid>0 else None
    except Exception: return None
def is_running()->bool:
    pid=read_pid()
    if not pid: return False
    try:
        os.kill(pid,0); return True
    except Exception: return False
def start()->dict:
    if is_running(): return {"ok":True,"already":True,"pid":read_pid()}
    proc=subprocess.Popen(RUN_CMD)
    PID.parent.mkdir(parents=True, exist_ok=True); PID.write_text(str(proc.pid))
    return {"ok":True,"pid":proc.pid}
def stop()->dict:
    pid=read_pid()
    if not pid: return {"ok":True,"already":True}
    try:
        os.kill(pid,15); time.sleep(1.0)
    except Exception: pass
    try:
        os.kill(pid,9)
    except Exception: pass
    try: PID.unlink()
    except Exception: pass
    return {"ok":True}
