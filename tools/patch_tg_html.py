# tools/patch_tg_html.py
from __future__ import annotations
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # proje kökü (tools/ altında çalıştırıyoruz)

# taranacak klasörler
DIRS = ["core", "live", "scripts"]

# basit markdown -> html dönüştürücü (çatallaşmayı önlemek için temeller yeterli)
MD_PATTERNS = [
    (re.compile(r"\*(?!\s)([^*\n]+?)\*(?!\S)"), r"<b>\1</b>"),          # *bold*
    (re.compile(r"_(?!\s)([^_\n]+?)_(?!\S)"), r"<i>\1</i>"),            # _italic_
    (re.compile(r"`([^`\n]+?)`"), r"<code>\1</code>"),                  # `code`
]

# tg_send(...) parse_mode düzeltmeleri
PMARK = re.compile(r'parse_mode\s*=\s*"(?:Markdown|markdown|MarkdownV2)"')
CALL_TG = re.compile(r'(\btg_send\s*\()')  # tg_send( açılışını yakala

def md_to_html(s: str) -> str:
    for pat, repl in MD_PATTERNS:
        s = pat.sub(repl, s)
    return s

def ensure_parse_mode_html(src: str) -> str:
    # 1) parse_mode="Markdown" -> "HTML"
    src2 = PMARK.sub('parse_mode="HTML"', src)

    # 2) tg_send( ... ) içinde hiç parse_mode yoksa, kapanış parantezinden önce parse_mode="HTML" enjekte et
    def add_pm(m):
        start = m.start(1)
        # m.group(1) sadece "tg_send(" kısmı
        # kapanışı bul: parantez dengeleme (kaba ama iş görür)
        i = m.end(1)
        depth, n = 1, len(src2)
        while i < n:
            c = src2[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    call = src2[m.start(1):i+1]
                    if "parse_mode=" not in call:
                        # virgül var mı bak; argüman varsa , parse_mode=... ekle; yoksa doğrudan ekle
                        insert_at = i
                        # kapatmadan önce ekle
                        new_call = call[:-1]
                        # eğer sadece tek argüman yoksa düzgün virgül ekle
                        inner = call[len("tg_send("):-1].strip()
                        sep = ", " if inner else ""
                        new_call += f'{sep}parse_mode="HTML")'
                        return src2[:m.start(1)] + new_call + src2[i+1:]
                    return src2  # zaten varmış
            i += 1
        return src2  # eşleşme başarısızsa dokunma

    # tüm çağrıları sıralı işle (soldan sağa)
    pos = 0
    while True:
        m = CALL_TG.search(src2, pos)
        if not m:
            break
        before = src2
        src2 = add_pm(m)
        pos = m.end(1) + 1
        if src2 is before:
            pos += 1
    return src2

def process_file(p: Path) -> bool:
    txt = p.read_text(encoding="utf-8")
    orig = txt

    # 0) sadece tg_send içeren satırlarda basit markdown'ı html'e çevir (çok agresif olmamak için)
    if "tg_send(" in txt:
        txt = md_to_html(txt)
        txt = ensure_parse_mode_html(txt)

    if txt != orig:
        bak = p.with_suffix(p.suffix + ".bak")
        if not bak.exists():
            bak.write_text(orig, encoding="utf-8")
        p.write_text(txt, encoding="utf-8")
        return True
    return False

def main():
    changed = 0
    for d in DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            # yedek dosyaları, venv, cache vb. atla
            s = str(p)
            if any(skip in s for skip in ["/.venv/", "\\.venv\\", "__pycache__", ".bak", ".patch_backups"]):
                continue
            if process_file(p):
                print(f"[patch] {p.relative_to(ROOT)}")
                changed += 1
    print(f"done. patched files: {changed}")

if __name__ == "__main__":
    sys.exit(main())
