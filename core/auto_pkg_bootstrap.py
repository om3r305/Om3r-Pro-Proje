# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Auto package bootstrapper (stable)
- 'knowledge' alias'ını kurar (knowledge -> Proje1.knowledge).
- İzinli bare modülleri Proje1/knowledge altına OLUŞTURUR ve alias'lar.
- Zengin stub içerikleri: utils/impl/abc/client/base_protocol/schema/vector_store/agent/research
- Eski kısa/bozuk iskeletleri tespit edip otomatik RESEED eder.
- TG’ye emojili bildirim atar.
"""
import importlib, sys, time, types, hashlib
from pathlib import Path
from typing import Iterable, Tuple, Dict

# ---- TG fallback ------------------------------------------------------------
try:
    from Proje1.core.utils_io import tg_send  # type: ignore
except Exception:
    def tg_send(*a, **k):  # no-op
        pass

# ---- Ayarlar ----------------------------------------------------------------
PKG_ROOT         = Path(__file__).resolve().parents[1]     # .../Proje1
KNOW_DIR         = PKG_ROOT / "knowledge"                  # Proje1/knowledge
KNOW_PKG_DOTTED  = "Proje1.knowledge"
BARE_PKG_ALIAS   = "knowledge"                             # top-level alias adı

# Brian’ın çıplak import ile isteyebileceği modüller (izinli liste)
BARE_ALLOW = {
    # mevcut/çekirdek
    "metrics_utils", "brain_engine", "patch_applier", "feature_registry",
    "strategy_fabric", "control_server", "tools_runtime", "brain_supervisor",
    "research",
    # seed/yardımcılar
    "schema", "vector_store", "agent", "utils", "impl", "abc", "client", "base_protocol",
}
BARE_ALWAYS = set(BARE_ALLOW)   # açılışta garanti oluştur

# ---------- küçük yardımcılar ----------
def _stamp() -> tuple[int, str]:
    ts  = int(time.time())
    hid = hashlib.sha1(f"{ts}".encode("utf-8")).hexdigest()[:10]
    return ts, hid

def _header_lines(dotted: str, ts: int, hid: str) -> str:
    return (
        "# -*- coding: utf-8 -*-\n"
        "from __future__ import annotations\n"
        f"# Auto-created module: {dotted}\n"
        f"# created_at: {ts}\n"
        f"# id: {hid}\n\n"
    )

STUBS: Dict[str, str] = {}

def _reg(name: str, body: str) -> None:
    STUBS[name] = body

# ---- STUB İÇERİKLERİ (hepsinde from __future__ zaten başta) ----
_reg("utils",
    "from __future__ import annotations\n"
    "import time\n"
    "from typing import Any, Dict\n\n"
    "__all__ = ['now_ts','reason_tag','reason_tag_safe','ping']\n\n"
    "def now_ts() -> int: return int(time.time())\n\n"
    "def reason_tag(x: str|None) -> str:\n"
    "    return (x or '').upper()\n\n"
    "def reason_tag_safe(x: Any) -> str:\n"
    "    try:\n"
    "        return str(x).upper()\n"
    "    except Exception:\n"
    "        return str(x)\n\n"
    "def ping() -> str: return 'utils_ok'\n"
)

_reg("abc",
    "from __future__ import annotations\n"
    "from abc import ABC, abstractmethod\n\n"
    "class BaseStrategy(ABC):\n"
    "    NAME = 'BaseStrategy'\n"
    "    @abstractmethod\n"
    "    def update(self, price: float) -> None: ...\n"
    "    @abstractmethod\n"
    "    def signal(self) -> dict: ...\n"
)

_reg("base_protocol",
    "from __future__ import annotations\n"
    "from dataclasses import dataclass\n\n"
    "@dataclass\n"
    "class Quote:\n"
    "    px: float\n"
    "    ts: int\n\n"
    "class ProtocolError(RuntimeError): ...\n"
)

_reg("client",
    "from __future__ import annotations\n"
    "from typing import Any, Dict, Optional\n\n"
    "class Client:\n"
    "    '''Lightweight HTTP client stub.'''\n"
    "    def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:\n"
    "        return {'url': url, 'params': params, 'ok': True}\n"
)

_reg("impl",
    "from __future__ import annotations\n"
    "from typing import Dict, Any\n\n"
    "def do_work(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:\n"
    "    return {'ok': True, 'cfg': bool(cfg)}\n"
)

_reg("schema",
    "from __future__ import annotations\n"
    "from dataclasses import dataclass\n\n"
    "@dataclass\n"
    "class TradeRow:\n"
    "    symbol: str\n"
    "    side: str\n"
    "    qty: float\n"
    "    price: float\n"
    "    ts: int\n"
)

_reg("vector_store",
    "from __future__ import annotations\n"
    "from typing import Dict, Any, List\n\n"
    "class VectorStore:\n"
    "    def __init__(self):\n"
    "        self._mem: List[Dict[str, Any]] = []\n"
    "    def add(self, item: Dict[str, Any]) -> None:\n"
    "        self._mem.append(dict(item))\n"
    "    def search(self, key: str, value: Any) -> List[Dict[str, Any]]:\n"
    "        return [r for r in self._mem if r.get(key) == value]\n"
)

_reg("agent",
    "from __future__ import annotations\n\n"
    "class Agent:\n"
    "    def run(self) -> str:\n"
    "        return 'agent_running'\n"
)

_reg("research",
    "from __future__ import annotations\n"
    "# boş sandbox; gerekirse kod burada büyür.\n"
    "def ping() -> str: return 'research_ok'\n"
)

_reg("metrics_utils",
    "from __future__ import annotations\n"
    "def record(metric: str, value: float) -> None:\n"
    "    pass\n"
)

_reg("patch_applier",
    "from __future__ import annotations\n"
    "def apply_patch(diff_text: str) -> bool:\n"
    "    return True\n"
)

_reg("strategy_fabric",
    "from __future__ import annotations\n"
    "def list_strategies(): return []\n"
)

_reg("control_server",
    "from __future__ import annotations\n"
    "def start(): return True\n"
)

_reg("brain_engine",
    "from __future__ import annotations\n"
    "def tick(): return True\n"
)

_reg("brain_supervisor",
    "from __future__ import annotations\n"
    "def health(): return {'ok': True}\n"
)

_reg("tools_runtime",
    "from __future__ import annotations\n"
    "def env_ok() -> bool: return True\n"
)

def _template_for(name: str, dotted: str) -> str:
    ts, hid = _stamp()
    header = _header_lines(dotted, ts, hid)
    body = STUBS.get(name)
    if body:
        # body’lerin hepsi kendi içinde from __future__ ile başlıyor; header + body birleşir
        return header + body
    # generic (çok nadir gerekir)
    return (
        header +
        "from typing import Any, Dict\n\n"
        "__all__ = ['bootstrap_ok','ping','describe']\n\n"
        "def bootstrap_ok() -> bool:\n"
        "    return True\n\n"
        f"def ping() -> str:\n"
        f"    return 'pong:{hid}'\n\n"
        "def describe() -> Dict[str, Any]:\n"
        "    return {\n"
        f"        'module': '{dotted}',\n"
        f"        'created_at': {ts},\n"
        f"        'hash': '{hid}'\n"
        "    }\n"
    )

# ---------- paket yardımcıları ----------
def _ensure_pkg() -> None:
    KNOW_DIR.mkdir(parents=True, exist_ok=True)
    init_py = KNOW_DIR / "__init__.py"
    if not init_py.exists():
        init_py.write_text(
            "from __future__ import annotations\n# package: Proje1.knowledge\n",
            encoding="utf-8"
        )

def _alias_package_root() -> None:
    try:
        pkg = importlib.import_module(KNOW_PKG_DOTTED)
        sys.modules.setdefault(BARE_PKG_ALIAS, pkg)
    except Exception as e:
        try: tg_send(f"🟡 <b>AutoPkg</b> root alias hatası: {e}", parse_mode="HTML")
        except Exception: pass

def _should_reseed_existing(txt: str) -> bool:
    # Çok kısa/bozuk iskeletler → reseed
    lines = [ln for ln in (txt or "").splitlines() if ln.strip() != ""]
    if len(lines) < 12:            # aşırı kısa ise
        return True
    if "invalid syntax" in txt:     # güvenli değil ama yine de işaret
        return True
    if "from __future__ import annotations" not in txt.splitlines()[:3]:
        return True
    return False

def _ensure_module_file(bare_name: str) -> Tuple[Path, str, bool]:
    """Bare modül için fiziksel dosyayı oluşturur. (path, dotted, created_or_reseeded) döner."""
    _ensure_pkg()
    p = KNOW_DIR / f"{bare_name}.py"
    dotted = f"{KNOW_PKG_DOTTED}.{bare_name}"
    created = False
    if not p.exists():
        p.write_text(_template_for(bare_name, dotted), encoding="utf-8")
        created = True
        try: tg_send(f"🧩 <b>AutoPkg</b>: <code>{dotted}</code> oluşturuldu.", parse_mode="HTML")
        except Exception: pass
    else:
        try:
            cur = p.read_text(encoding="utf-8", errors="ignore")
            if _should_reseed_existing(cur):
                p.write_text(_template_for(bare_name, dotted), encoding="utf-8")
                created = True
                try: tg_send(f"🧩 <b>AutoPkg</b>: <code>{dotted}</code> reseed edildi (zengin stub).", parse_mode="HTML")
                except Exception: pass
        except Exception:
            pass
    return p, dotted, created

def _alias_bare_to_knowledge(bare_name: str, dotted: str) -> None:
    real_mod = importlib.import_module(dotted)
    sys.modules.setdefault(bare_name, real_mod)

# ---------- ANA API ----------
def preload_missing_modules(allow: Iterable[str] | None = None) -> None:
    """
    Başlangıçta:
      - BARE_ALWAYS modüllerinin dosyasını garanti eder (gerekirse reseed),
      - 'knowledge' paket kökü alias’ını yazar,
      - İzinli bare isimleri alias’lar.
    """
    names = set(allow or BARE_ALLOW)

    _ensure_pkg()
    _alias_package_root()

    created = 0
    aliased = 0

    # Dosyaları garanti et
    for bare in sorted(BARE_ALWAYS):
        _, dotted, did = _ensure_module_file(bare)
        if did:
            created += 1

    if names:
        for bare in sorted(names):
            try:
                _, dotted, did2 = _ensure_module_file(bare)
                if did2:
                    created += 1
                _alias_bare_to_knowledge(bare, dotted)
                aliased += 1
            except Exception as e:
                try: tg_send(f"🔴 <b>AutoPkg</b> alias fail <code>{bare}</code>: {e}", parse_mode="HTML")
                except Exception: pass

    try:
        tg_send(f"🧠 <b>AutoPkg</b> hazır • 📦 stubs:<b>{created}</b> • 🔗 aliased:<b>{aliased}</b>", parse_mode="HTML")
    except Exception:
        pass

# Importing this module must be side-effect free. Legacy callers may explicitly
# call ``preload_missing_modules`` outside the Brian shadow workflow.

# Dışarıya: tek modül garantisi
def ensure_bare(name: str) -> types.ModuleType:
    _, dotted, _ = _ensure_module_file(name)
    _alias_package_root()
    _alias_bare_to_knowledge(name, dotted)
    return sys.modules[name]
