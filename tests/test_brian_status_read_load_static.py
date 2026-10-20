"""Static guard for the engineering-status load fix (20 Oct 2026): shared cache, single flight and
an explicitly marked STALE fallback must stay in place."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "supabase/functions/brian-frontier-engineering-status/index.ts").read_text(encoding="utf-8")


def test_engineering_status_has_shared_cache_and_single_flight():
    assert "const CACHE_MS=45_000;" in SRC
    assert "inflight??=build().finally(()=>{inflight=null});" in SRC
    assert 'status:"STALE",stale:true' in SRC
    assert "const STALE_MAX_MS=15*60_000;" in SRC


def test_build_has_no_request_scoped_state():
    build = SRC[SRC.index("async function build()"):SRC.index("Deno.serve(")]
    import re
    assert "origin" not in build and not re.search(r"out\(", build) and not re.search(r"req", build)
