# brain_chat.py — MetaBrain v2 (proje farkındalığı + hızlı komutlar)
import os, time, glob, json, textwrap, datetime as dt
from dotenv import load_dotenv
from openai import OpenAI

# == Ayarlar ==
DEFAULT_MODEL = "gpt-4o-mini"
# Tek seferde LLM'e vereceğimiz her dosya için karakter üst sınırı
MAX_CHARS_PER_FILE = 12000
# Başlangıçta okunacak dosya kalıpları (PROJECT_ROOT altında)
DEFAULT_PATTERNS = [
    "main.py",
    "autopilot.py", "config_live.json",
    "adaptive.py", "ai_predictor.py", "optimizer.py",
    "candles.py", "dip_tracker.py", "regime.py",
    "logger_utils.py", "report_logger.py", "metrics_utils.py",
    "watchlist_manager.py", "telegram_utils.py",
    "config_live.json", "requirements.txt",
    "patch_rules.yaml", "code_map.json",
    "backtest.py", "rl_tuner.py",
    "logs/*.txt", "logs/*.log", "trades.csv"
]

SYSTEM_PROMPT = """You are MetaBrain, an expert trading co-pilot embedded in a crypto bot project.
You can read the project's key files and recent logs loaded at startup.
Behavior:
- Be concise, propose concrete fixes with diff-like code blocks when appropriate.
- Prefer minimal, safe changes first; note potential risks.
- When numbers can be tuned, suggest exact new values and the rationale.
- If context looks stale, ask the user to /reload.
- Turkish responses are OK. Keep variable/file names in English.
"""

def now_iso():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _safe_read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        if len(txt) > MAX_CHARS_PER_FILE:
            head = txt[:MAX_CHARS_PER_FILE//2]
            tail = txt[-MAX_CHARS_PER_FILE//2:]
            txt = head + "\n...\n# [truncated]\n...\n" + tail
        return txt
    except Exception as e:
        return f"<<could not read {path}: {e}>>"

def _list_files(project_root: str, patterns):
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(os.path.join(project_root, pat))))
    # yinelenenleri temizle
    seen, out = set(), []
    for p in files:
        if p not in seen and os.path.isfile(p):
            seen.add(p); out.append(p)
    return out

def _bootstrap_context(project_root: str, patterns):
    file_list = _list_files(project_root, patterns)
    summaries = []
    contents = []
    for p in file_list:
        rel = os.path.relpath(p, project_root)
        txt = _safe_read(p)
        summaries.append(f"- {rel} ({len(txt)} chars)")
        contents.append({"role":"system",
                         "content": f"[FILE {rel} @ {now_iso()}]\n{txt}"})
    header = ("[PROJECT CONTEXT]\n"
              f"root: {project_root}\n"
              "files:\n" + "\n".join(summaries))
    return header, contents, file_list

def _env_or_fail(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

def chat_loop():
    load_dotenv()
    api_key = _env_or_fail("OPENAI_API_KEY")
    model = os.getenv("BRAIN_MODEL", DEFAULT_MODEL)
    project_root = os.getenv("PROJECT_ROOT", os.getcwd())

    client = OpenAI(api_key=api_key)

    # Proje içeriğini yükle
    header, file_messages, file_list = _bootstrap_context(project_root, DEFAULT_PATTERNS)

    messages = [{"role":"system", "content": SYSTEM_PROMPT},
                {"role":"system", "content": header}]
    messages.extend(file_messages)

    print(f"🧠 MetaBrain ready (model: {model})")
    print(f"📂 Project root: {project_root}")
    print("💡 Komutlar: /files, /open path, /reload, /where, /help, exit")

    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue
            cmd = user.lower().strip()

            if cmd in {"exit", "quit", "q"}:
                print("🧠 Görüşürüz!"); break

            if cmd == "/help":
                print("Komutlar:\n"
                      "  /files            -> yüklü dosyaları listele\n"
                      "  /open <relpath>   -> dosyayı göster (kısaltılmış)\n"
                      "  /reload           -> proje dosyalarını yeniden oku\n"
                      "  /where            -> current PROJECT_ROOT\n"
                      "  exit              -> çıkış")
                continue

            if cmd == "/files":
                for p in file_list:
                    print(" -", os.path.relpath(p, project_root))
                continue

            if cmd.startswith("/open "):
                rel = user.split(" ", 1)[1].strip()
                path = os.path.join(project_root, rel)
                print(f"----- {rel} -----")
                print(_safe_read(path))
                print("----- end -----")
                continue

            if cmd == "/reload":
                header, file_messages, file_list = _bootstrap_context(project_root, DEFAULT_PATTERNS)
                messages = [{"role":"system","content": SYSTEM_PROMPT},
                            {"role":"system","content": header}] + file_messages
                print("🔄 Proje içeriği tazelendi.")
                continue

            if cmd == "/where":
                print("PROJECT_ROOT =", project_root); continue

            # Normal sohbet isteği
            messages.append({"role":"user", "content": user})
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=800,
            )
            text = resp.choices[0].message.content or ""
            print("🤖 Brain:", text)
            messages.append({"role":"assistant","content":text})

        except KeyboardInterrupt:
            print("\n🧠 İptal edildi."); break
        except Exception as e:
            print("❗ Hata:", e); time.sleep(1)

if __name__ == "__main__":
    chat_loop()
