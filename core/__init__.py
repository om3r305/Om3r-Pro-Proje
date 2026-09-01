"""Core package bootstrap and historical package-name compatibility."""
from __future__ import annotations

import sys

from project_compat import install_proje1_alias

install_proje1_alias()
sys.modules.setdefault("Proje1.core", sys.modules[__name__])
