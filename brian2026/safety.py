from __future__ import annotations

from typing import Any, Mapping


def shadow_workflow_guard(config: Mapping[str, Any] | None) -> bool:
    """Suppress code-writing services whenever Brian shadow mode is enabled."""
    section = dict((config or {}).get("brian2026", {}) or {})
    return bool(section.get("shadow_enabled", False))
