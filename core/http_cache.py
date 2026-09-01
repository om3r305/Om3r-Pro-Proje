# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time, hashlib
from pathlib import Path
CACHE_DIR=Path("runtime/downloads"); META=CACHE_DIR/"_meta.json"
def _load(): 
    try:
        if META.exists(): return json.loads(META.read_text(encoding="utf-8"))
    except Exception: pass
    return {}
def _save(m):
    try:
        META.parent.mkdir(parents=True, exist_ok=True)
        META.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding="utf-8")
    except Exception: pass
def key_for(url:str)->str: return hashlib.sha1(url.encode("utf-8")).hexdigest()
def get_headers_for(url:str):
    m=_load(); rec=m.get(key_for(url)) or {}; h={}
    if "etag" in rec: h["If-None-Match"]=rec["etag"]
    if "last_modified" in rec: h["If-Modified-Since"]=rec["last_modified"]
    return h
def update_headers(url:str, resp):
    try:
        m=_load(); k=key_for(url); rec=m.get(k) or {}
        et=resp.headers.get("ETag"); lm=resp.headers.get("Last-Modified")
        if et: rec["etag"]=et
        if lm: rec["last_modified"]=lm
        rec["ts"]=time.time(); m[k]=rec; _save(m)
    except Exception: pass
def write_body(url:str, content:bytes, ext:str=".bin")->Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p=CACHE_DIR/f"{key_for(url)}{ext}"; p.write_bytes(content); return p
