from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "monster-coins-pro" / "index.html"
CSS = ROOT / "monster-coins-pro" / "dashboard.css"
JS = ROOT / "monster-coins-pro" / "dashboard.js"
SW = ROOT / "monster-coins-pro" / "sw.js"
CONTROL = ROOT / "supabase" / "functions" / "brian-control-center" / "index.ts"


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_dashboard_is_turkish_mobile_first_and_uses_unified_views():
    html = text(INDEX)
    css = text(CSS)
    assert '<html lang="tr">' in html
    assert 'viewport-fit=cover' in html
    assert '/dashboard.css' in html and '/dashboard.js' in html
    for label in ["Genel Bakış", "ALPHA Kararları", "Öğrenme", "DIP Laboratuvarı", "Sistem Sağlığı"]:
        assert label in html
    assert 'data-view="overview"' in html
    assert 'data-view="learning"' in html
    assert 'data-view="system"' in html
    assert 'min-height:100dvh' in css
    assert 'env(safe-area-inset-bottom' in css
    assert '@media(min-width:1080px)' in css
    assert '.cc-bottom' in css


def test_dashboard_explains_server_background_semantics_instead_of_browser_magic():
    html = text(INDEX)
    js = text(JS)
    assert "sayfayı veya telefonu kapatsan da MAIN / ALPHA" in js
    assert "Bu oturum yalnız performans penceresidir" in html
    assert "Takibi Başlat" in html
    assert "MAIN / ALPHA arka planda çalışmaya devam ediyor" in js
    assert "browser_engine_required" in js
    assert "DIP taraması durur; MAIN / ALPHA ise durmaz" in js


def test_control_center_is_connected_to_current_auditor_and_learning_chain():
    src = text(CONTROL)
    assert '"brian-missed-opportunity-auditor-v3"' in src
    assert '"brian-missed-opportunity-auditor-v2"' not in src
    assert 'brian_sensor_reliability_shadow_snapshots' in src
    assert 'brian_alpha_reliability_shadow_features' in src
    assert 'brian_sensor_reliability_prospective_calibration' in src
    assert 'schema_version: "brian.control-center.status.v3"' in src
    assert 'continues_when_page_closed: true' in src
    assert 'main_alpha_browser_independent: true' in src
    assert 'dip_browser_independent: false' in src
    assert 'update public.brian_sensor_observations' not in src.lower()


def test_dashboard_keeps_existing_control_actions_and_shadow_boundary():
    js = text(JS)
    src = text(CONTROL)
    for action in ["status", "start", "restart", "pause", "report_now"]:
        assert f"api('{action}'" in js
    assert 'shadow_only: true' in src
    assert 'live_execution: false' in src
    assert "tr-TR" in js
    assert "Europe/Berlin" in js


def test_service_worker_caches_new_control_center_shell():
    sw = text(SW)
    assert "monster-coins-pro-shell-v6" in sw
    assert "'/dashboard.css'" in sw
    assert "'/dashboard.js'" in sw


def test_dashboard_javascript_parses_when_node_is_available():
    node = shutil.which("node")
    if node is None:
        return
    result = subprocess.run([node, "--check", str(JS)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
