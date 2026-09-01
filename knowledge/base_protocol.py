# -*- coding: utf-8 -*-
from __future__ import annotations
# Auto-created module: Proje1.knowledge.base_protocol
# created_at: 1757975314
# id: 75d780549c

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Quote:
    px: float
    ts: int

class ProtocolError(RuntimeError): ...
