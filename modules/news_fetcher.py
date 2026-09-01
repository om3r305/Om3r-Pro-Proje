# -*- coding: utf-8 -*-
from __future__ import annotations
import time, json
from pathlib import Path
try:
    import requests
except Exception:
    requests=None  # type: ignore
from Proje1.core.web_audit import enable as _audit_enable
from Proje1.core.http_policy import new_session
from Proje1.core.http_cache import get_headers_for, update_headers, write_body
_audit_enable()

INTEL=Path("logs/market_intel.jsonl")
SRC=[
    {"name":"BinanceBlog","url":"https://www.binance.com/blog/rss","type":"rss"},
    {"name":"CoinDesk","url":"https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml","type":"rss"},
]
def _w(obj):
    try:
        INTEL.parent.mkdir(parents=True, exist_ok=True)
        INTEL.open("a",encoding="utf-8").write(json.dumps(obj,ensure_ascii=False)+"\n")
    except Exception: pass

def fetch_once(timeout:int=8,max_bytes:int=2_000_000):
    if requests is None: return 0
    s=new_session(); 
    if s is None: return 0
    c=0
    for src in SRC:
        url=src["url"]
        try:
            r=s.get(url,headers=get_headers_for(url),timeout=timeout)
            if r.status_code==304: continue
            if r.ok and r.content and len(r.content)<max_bytes:
                update_headers(url,r)
                path=write_body(url,r.content,".xml")
                _w({"ts":time.time(),"source":src["name"],"url":url,"saved":str(path)}); c+=1
            else:
                _w({"ts":time.time(),"source":src["name"],"url":url,"status":r.status_code})
        except Exception as e:
            _w({"ts":time.time(),"source":src["name"],"url":url,"error":str(e)})
    return c
