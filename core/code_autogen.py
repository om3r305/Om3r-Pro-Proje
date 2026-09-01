# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, time, ast, importlib.util, requests
from pathlib import Path

def _ollama_host(cfg: dict) -> str:
    host = os.environ.get(cfg.get("llm",{}).get("host_env","OLLAMA_HOST"), "http://localhost:11434")
    return host.rstrip("/")

def _ollama_model(cfg: dict) -> str:
    return os.environ.get(cfg.get("llm",{}).get("model_env","OLLAMA_MODEL"), "qwen2.5-coder:latest")

def _stream_ollama(prompt: str, cfg: dict) -> str:
    url = _ollama_host(cfg) + "/api/generate"
    model = _ollama_model(cfg)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": cfg.get("file_watcher",{}).get("autogen",{}).get("temperature", 0.2)}
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response","")

def _build_prompt(orig_code: str, path: str, contract: list[str]) -> str:
    goal = (
        f"Dosya: {path}\n"
        f"HEDEF: Aşağıdaki Python modülünü *üretim seviyesinde* tamamla/refactor et. "
        f"Bu modül {contract} fonksiyonlarını **eksiksiz** sağlayacak. "
        "Uygulama minimal bağımlılıkla çalışmalı, dosya/IO pathleri 'logs/' altını kullanmalı, "
        "UTF-8 yazmalı. Kod tek bir dosyada, import edilebilir olmalı. Gerektiğinde docstring ekle."
    )
    return goal + "\n\n# ORİJİNAL KOD:\n" + orig_code + "\n\n# YENİ TAM KOD:\n"

def try_llm_refactor(target: Path, contract: list[str], cfg: dict) -> tuple[bool,str,str]:
    """Return (ok, status, new_file_path)"""
    try:
        orig = target.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return False, f"read_fail:{e}", ""

    prompt = _build_prompt(orig, target.as_posix(), contract)
    try:
        code = _stream_ollama(prompt, cfg)
    except Exception as e:
        return False, f"llm_fail:{e}", ""

    # Kod blokları geldiyse ayıkla
    if "```" in code:
        parts = []
        keep = False
        for line in code.splitlines():
            if line.strip().startswith("```"):
                keep = not keep
                continue
            if keep:
                parts.append(line)
        code = "\n".join(parts) if parts else code

    # Söz dizimi kontrol
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax_bad:{e}", ""

    # Geçici dosya
    out_dir = Path(".auto_runs") / "builds" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    new_path = out_dir / target.name
    new_path.write_text(code, encoding="utf-8")

    # Import & kontrat kontrol
    try:
        spec = importlib.util.spec_from_file_location(f"_fw_new_{target.stem}", str(new_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # noqa
        for fn in contract:
            if not hasattr(mod, fn):
                return False, f"contract_missing:{fn}", ""
    except Exception as e:
        return False, f"import_fail:{e}", ""

    return True, "refactor_ok", new_path.as_posix()
