# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time
from pathlib import Path
try:
    import requests
except Exception:
    requests = None  # type: ignore

LOG = Path("logs/web_audit.jsonl")
if requests is not None:
    _ORIG = requests.Session.request  # type: ignore

def _w(obj):
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.open("a", encoding="utf-8").write(json.dumps(obj, ensure_ascii=False)+"\n")
    except Exception:
        pass

def enable():
    if requests is None:
        return
    def wrap(self, method, url, *a, **kw):
        t0 = time.time()
        host = None
        try:
            if isinstance(url, str) and "://" in url:
                host = url.split("/", 3)[2]
            r = _ORIG(self, method, url, *a, **kw)
            _w({"ts": time.time(),"method":method,"url":url,"host":host,
                "status": getattr(r, "status_code", None),
                "bytes": len(getattr(r, "content", b"") or b""),
                "ms": int((time.time()-t0)*1000)})
            return r
        except Exception as e:
            _w({"ts": time.time(),"method":method,"url":url,"host":host,"error":str(e),
                "ms": int((time.time()-t0)*1000)})
            raise
    if getattr(requests.Session.request, "__name__", "") != "wrap":
        requests.Session.request = wrap  # type: ignore
