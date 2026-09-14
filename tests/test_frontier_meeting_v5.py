from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "monster-coins-pro" / "frontier-v4.html"
MEETING = ROOT / "monster-coins-pro" / "frontier-meeting-v5.js"


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_meeting_v5_is_loaded_by_frontier_home():
    html = text(HTML)
    assert "/frontier-meeting-v5.js" in html


def test_meeting_v5_preserves_source_provenance_and_truthful_stances():
    src = text(MEETING)
    for phrase in [
        "Haberi getiren",
        "Yayınlayan",
        "Kaynak durumu",
        "DOĞRULAMA BEKLİYOR",
        "İŞLEMİ ONAYLAMADI",
        "OTURUM DIŞI",
        "Toplantı sonucu",
        "BEKLE / KAYNAĞI DOĞRULA",
    ]:
        assert phrase in src
    assert "source_trust_class" in src
    assert "provenance_uri" in src
    assert "eligible_for_decision_evidence" in src
    assert "meeting-alarm" in src
    assert "meeting-review" in src


def test_meeting_v5_javascript_parses_when_node_is_available():
    node = shutil.which("node")
    if node is None:
        return
    result = subprocess.run([node, "--check", str(MEETING)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
