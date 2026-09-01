# research/agent.py
from __future__ import annotations
from typing import List
from knowledge.vector_store import add_finding
from research.sources import fetch_mock_news
from knowledge.schema import Finding

def run_once(symbols: List[str] | None = None) -> int:
    """Tek seferlik araştırma; mock kaynaklardan bulgu yaz."""
    items: List[Finding] = fetch_mock_news()
    if symbols:
        items = [f for f in items if any(s in f["symbols"] for s in symbols)]
    for it in items:
        add_finding(it)
    return len(items)
