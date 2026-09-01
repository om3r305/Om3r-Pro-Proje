# llm_engine.py
# Kanka build: 2025-09-06
# LLM sağlayıcı köprüsü (Ollama / OpenAI için basit şablon; şu an Ollama aktif)

from __future__ import annotations
import os, json, time, typing, traceback
from typing import Optional, Dict, Any, List

# ---- .env opsiyonel ----
try:
    from dotenv import load_dotenv
    load_dotenv()  # proje kökünde .env varsa yükler
except Exception:
    pass

import requests

DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
# Öncelik: .env -> program argümanı (Brain geçebilir) -> default
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "").strip()

# Bizim önerdiğimiz fallback sırası (VRAM'e göre mantıklı dizili)
FALLBACK_MODELS: List[str] = [
    os.getenv("OLLAMA_MODEL", "").strip() or "",   # .env’de varsa en önce dene
    "hf.co/unsloth/Qwen2.5-Coder-14B-Instruct-128K-GGUF:Q5_K_M",  # güçlü, coder türevi
    "qwen2.5:14b-instruct",                        # Qwen 14B instruct
    "llama3.1:8b-instruct",                        # 8B hafif seçenek
    "llama2:70b",                                  # sistemin kaldırdığı durumlar için
]

def _clean_models(ms: List[str]) -> List[str]:
    out = []
    for m in ms:
        m = (m or "").strip()
        if m and m not in out:
            out.append(m)
    return out

class LLMEngine:
    """
    Ollama LLM köprüsü.
    ask(prompt, temperature=..., top_p=..., max_tokens=...) destekler.
    Model hatalarında otomatik retry + sıradaki modele fallback yapar.
    """

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None, timeout: int = 120):
        self.host = host or DEFAULT_HOST
        self.session = requests.Session()
        self.timeout = timeout

        seq = _clean_models([model or DEFAULT_MODEL] + FALLBACK_MODELS)
        self.models = seq if seq else ["llama3.1:8b-instruct"]
        # Otomatik keep-alive
        self.session.headers.update({"Connection": "keep-alive"})

    # ---- düşük seviye ----
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.host}{path}"
        r = self.session.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ---- sağlık ----
    def health(self) -> bool:
        try:
            # basit bir HEAD/ping yerine /api/tags çağrısı ile test edelim
            r = self.session.get(f"{self.host}/api/tags", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    # ---- ana çağrı ----
    def ask(
        self,
        prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 700,
        retries: int = 2,
        retries_per_model: int = 1,
        system: Optional[str] = None
    ) -> str:
        """
        prompt -> string cevap döner.
        temperature/top_p/max_tokens destekli.
        Her model için retries_per_model kadar dener, sonra sıradakine geçer.
        """
        last_err: Optional[Exception] = None
        for model in self.models:
            if not model:
                continue
            for attempt in range(1, retries_per_model + 1):
                try:
                    payload = {
                        "model": model,
                        # Ollama /api/generate tek-stepli (chat değil) endpoint
                        "prompt": self._compose_prompt(prompt, system=system),
                        "stream": False,
                        "options": {
                            "temperature": float(temperature),
                            "top_p": float(top_p),
                            # Ollama "num_predict" = max token out; -1 sınırsız
                            "num_predict": int(max_tokens) if max_tokens and max_tokens > 0 else -1,
                        },
                    }
                    data = self._post("/api/generate", payload)
                    txt = (data.get("response") or "").strip()
                    if not txt:
                        raise RuntimeError("Boş yanıt döndü.")
                    return txt
                except Exception as e:
                    last_err = e
                    # Hata mesajı kısa log
                    print(f"[LLMEngine] Model='{model}' deneme#{attempt} hata: {type(e).__name__}: {e}")
                    # küçük bekleme, sonra tekrar veya fallback
                    time.sleep(1.0)
            # sıradaki modele geçmeden kısa bekleme
            time.sleep(0.5)

        # buraya geldiysek tüm denemeler bitti
        raise RuntimeError(f"Tüm modeller hata verdi. Son hata: {last_err!r}")

    # prompt birleştirme (system desteği basit)
    def _compose_prompt(self, user: str, system: Optional[str] = None) -> str:
        if not system:
            return user
        # Çoğu instruct model düz prompt bekler; system'i başa zayıf bir yönerge olarak ekleyelim
        return f"{system.strip()}\n\nKullanıcı: {user.strip()}\nCevap:"

# --------------- CLI ---------------
def _print_exc():
    traceback.print_exc()

def main():
    import argparse
    p = argparse.ArgumentParser("llm_engine")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("health")
    sayp = sub.add_parser("say")
    sayp.add_argument("text", nargs="+")
    sayp.add_argument("--temperature", type=float, default=0.2)
    sayp.add_argument("--top_p", type=float, default=0.9)
    sayp.add_argument("--max_tokens", type=int, default=400)
    sayp.add_argument("--model", type=str, default=None)
    sayp.add_argument("--host", type=str, default=None)

    args = p.parse_args()

    if args.cmd == "health":
        eng = LLMEngine()
        ok = eng.health()
        print("OK" if ok else "FAIL")
        return

    if args.cmd == "say":
        text = " ".join(args.text)
        eng = LLMEngine(model=args.model, host=args.host)
        try:
            out = eng.ask(
                text,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            print(out)
        except Exception:
            _print_exc()
        return

    # varsayılan: küçük demo
    eng = LLMEngine()
    try:
        print(eng.ask("Sadece tek cümlelik Türkçe selam ver."))
    except Exception:
        _print_exc()

if __name__ == "__main__":
    main()
