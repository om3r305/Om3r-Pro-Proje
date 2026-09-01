# -*- coding: utf-8 -*-
from __future__ import annotations
# Auto-created module: Proje1.knowledge.utils
# created_at: 1757975314
# id: 75d780549c

from __future__ import annotations
import time
from typing import Any, Dict

__all__ = ['now_ts','reason_tag','reason_tag_safe','ping']

def now_ts() -> int: return int(time.time())

def reason_tag(x: str|None) -> str:
    return (x or '').upper()

def reason_tag_safe(x: Any) -> str:
    try:
        return str(x).upper()
    except Exception:
        return str(x)

def ping() -> str: return 'utils_ok'
