# fast_trader.py — uyumluluk dosyası
import sys
import subprocess

if __name__ == "__main__":
    # config parametrelerini forward et
    args = ["python", "main.py"] + sys.argv[1:]
    try:
        subprocess.run(args, check=True)
    except Exception as e:
        print(f"[fast_trader redirect hata]: {e}")
