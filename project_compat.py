"""Compatibility for legacy imports that still use the ``Proje1`` name."""
from __future__ import annotations

from pathlib import Path
import sys
import types


def install_proje1_alias() -> None:
    """Expose this checkout as ``Proje1`` without depending on its folder name."""
    if "Proje1" in sys.modules:
        return
    module = types.ModuleType("Proje1")
    module.__package__ = "Proje1"
    module.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules["Proje1"] = module
