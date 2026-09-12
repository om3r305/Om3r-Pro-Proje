from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
# The previous operational dashboard remains available as a hidden recovery/reference
# shell, while the product-facing world is now the unified Brian command center.
INDEX = ROOT / "monster-coins-pro" / "classic.html"
ROOT_INDEX = ROOT / "monster-coins-pro" / "index.html"
FRONTIER_INDEX = ROOT / "monster-coins-pro" / "frontier-v3.html"
FRONTIER_CSS = ROOT / "monster-coins-pro" / "frontier-v3.css"
FRONTIER_JS = ROOT / "monster-coins-pro" / "frontier-v3.js"
CSS = ROOT / "monster-coins-pro" / "dashboard.css"
JS = ROOT / "monster-coins-pro" / "dashboard.js"
SW = ROOT / "monster-coins-pro" / "sw.js"
CONTROL = ROOT / "supabase" / "functions" / "brian-control-center" / "index.ts"
CONTROL_CORE = ROOT / "supabase" / "functions" / "brian-control-center-core" / "index.ts"


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


def test_frontier_v3_is_the_turkish_mobile_product_home_and_root_routes_to_it():
    root = text(ROOT_INDEX)
    frontier = text(FRONTIER_INDEX)
    css = text(FRONTIER_CSS)
    assert "location.replace('/frontier-v3.html'" in root
    assert '<html lang="tr">' in frontier
    assert 'viewport-fit=cover' in frontier
    assert '/frontier-v3.css' in frontier and '/frontier-v3.js' in frontier
    assert 'Brian ile Konuş' in frontier
    assert 'Brian Toplantı Odası' in frontier
    assert 'SİSTEMİ BAŞLAT' in frontier
    assert 'YENİDEN BAŞLAT' in frontier
    assert 'DURDUR' in frontier
    assert 'SHADOW HAZİNE TUTARI' in frontier
    assert 'href="/dip"' in frontier
    assert '/classic.html' not in frontier
    assert '.bottom-nav' in css
    assert 'env(safe-area-inset-bottom' in css


def test_dashboard_explains_server_background_semantics_instead_of_browser_magic():
    html = text(INDEX)
    js = text(JS)
    assert "sayfayı veya telefonu kapatsan da MAIN / ALPHA" in js
    assert "Bu oturum yalnız performans penceresidir" in html
    assert "Takibi Başlat" in html
    assert "MAIN / ALPHA arka planda çalışmaya devam ediyor" in js
    assert "browser_engine_required" in js
    assert "DIP taraması durur; MAIN / ALPHA ise durmaz" in js


def test_control_center_wraps_pinned_core_and_adds_v8_server_authoritative_dip():
    src = text(CONTROL)
    core = text(CONTROL_CORE)
    assert 'brian-control-center-core' in src
    assert 'v8DipSummary' in src
    assert 'brian_dip_session_events' in src
    assert 'brian_dip_v8_runtime' in src
    assert 'brian_dip_v8_ledger' not in src
    assert 'dip_browser_independent: true' in src
    assert 'dip_server_authoritative: true' in src
    assert 'raw.githubusercontent.com/om3r305/Om3r-Pro-Proje/' in core
    assert '/supabase/functions/brian-control-center/index.ts' in core
    assert 'update public.brian_sensor_observations' not in src.lower()


def test_control_center_status_overlay_preserves_core_response_and_fail_safe_boundary():
    src = text(CONTROL)
    assert 'action !== "status"' in src
    assert 'return new Response(coreText' in src
    assert 'system.dip = dip' in src
    assert 'x-brian-dip-overlay' in src
    assert 'cache-control' in src
    assert 'no-store' in src
    assert 'console.error("control-center-v8-overlay"' in src


def test_dashboard_keeps_existing_control_actions_and_shadow_boundary():
    js = text(JS)
    src = text(CONTROL)
    for action in ["status", "start", "restart", "pause", "report_now"]:
        assert f"api('{action}'" in js
    assert 'shadow_only: runtimeQ.data.shadow_only !== false' in src
    assert 'live_execution: runtimeQ.data.live_execution === true' in src
    assert "tr-TR" in js
    assert "Europe/Berlin" in js


def test_service_worker_caches_new_control_center_shell():
    sw = text(SW)
    assert "monster-coins-pro-shell-v12" in sw
    assert "'/dashboard.css'" in sw
    assert "'/dashboard.js'" in sw
    assert "'/dip-expert-v4-runtime-guard.js'" in sw
    assert "'/dip-server-authoritative-v7.js'" in sw


def test_dashboard_javascript_parses_when_node_is_available():
    node = shutil.which("node")
    if node is None:
        return
    for path in (JS, FRONTIER_JS):
        result = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
