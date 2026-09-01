# -*- coding: utf-8 -*-
from __future__ import annotations
# Auto-created module: Proje1.knowledge.client
# created_at: 1758275571
# id: 4f4e03321b

from __future__ import annotations
from typing import Any, Dict, Optional

class Client:
    '''Lightweight HTTP client stub.'''
    def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {'url': url, 'params': params, 'ok': True}
