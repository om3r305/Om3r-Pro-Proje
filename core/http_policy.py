# -*- coding: utf-8 -*-
from __future__ import annotations
import platform
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except Exception:
    requests=None; HTTPAdapter=object; Retry=object  # type: ignore

UA = f"MonsterCoinsPro/1.0 Python/{platform.python_version()}"
def new_session(backoff_factor:float=0.5,total:int=3):
    if requests is None: return None
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    try:
        kwargs=dict(total=total,backoff_factor=backoff_factor,status_forcelist=[429,500,502,503,504],raise_on_status=False)
        try:
            r=Retry(allowed_methods=frozenset({"GET","POST"}),**kwargs)  # type: ignore
        except TypeError:
            r=Retry(method_whitelist=frozenset({"GET","POST"}),**kwargs)  # type: ignore
        ad=HTTPAdapter(max_retries=r); s.mount("https://",ad); s.mount("http://",ad)
    except Exception:
        pass
    return s
