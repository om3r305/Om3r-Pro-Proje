from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from brian2026.state_fingerprint import portable_state_fingerprint


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "supabase/functions/brian-live-shadow/checkpoint.json"
SOURCE = ROOT / "supabase/functions/brian-live-shadow/index.ts"
MIGRATION = ROOT / "supabase/migrations/202609030003_brian_phase37_live_shadow.sql"
RAW_STATE_ID = "de90c35af3525d591f17e2489e64e9c5ebd84f8124e344927d7c829623688d36"
PORTABLE_FINGERPRINT = "b534b611543fcf449a371faad208be20ccf7782343996d08b2bd554ed7f720b9"


def _canonical_hash(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def test_phase37_pins_exact_phase35_frozen_checkpoint() -> None:
    state = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    assert _canonical_hash(state) == RAW_STATE_ID
    assert portable_state_fingerprint(state) == PORTABLE_FINGERPRINT
    assert state["episodes_learned"] == 1000
    assert state["transitions_learned"] == 635000
    assert tuple(sorted(state["models"])) == tuple(sorted(("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")))


def test_phase37_runtime_is_forward_only_and_shadow_only() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert 'PROSPECTIVE_DEVELOPMENT_SHADOW' in source
    assert 'historical_backfill: false' in source
    assert 'learning_enabled: false' in source
    assert 'live_execution: false' in source
    assert '/api/v3/klines' in source
    assert '/api/v3/ticker/bookTicker' in source
    assert 'starting_equity: 500.0' in source
    for forbidden in ('/api/v3/order', 'create_order', 'place_order', 'submit_order', 'apiKey', 'secretKey'):
        assert forbidden not in source


def test_phase37_storage_is_append_only_private_and_five_minute() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "enable row level security" in sql
    assert "brian_reject_mutation" in sql
    assert "revoke all on public.brian_live_shadow_ticks from anon, authenticated" in sql
    assert "grant select, insert on public.brian_live_shadow_ticks to service_role" in sql
    assert "'1-59/5 * * * *'" in sql
    assert "brian-live-shadow-5m" in sql
    assert "historical backfill and live exchange execution are forbidden" in sql
