"""Static guards for the treasury scorecard v2 period (20 Oct 2026)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "supabase/functions/brian-shadow-scorecard/index.ts").read_text(encoding="utf-8")
MIG = (ROOT / "supabase/migrations/20261020082424_brian_shadow_scorecard_treasury_v2.sql").read_text(encoding="utf-8")


def test_scorecard_reads_newest_protocol_per_engine():
    assert "select distinct on (engine_id) * from public.brian_shadow_protocols order by engine_id, starts_at desc, protocol_id desc limit 3" in INDEX
    assert "order by engine_id limit 3`" not in INDEX


def test_v2_migration_is_treasury_only_same_rules_and_server_side():
    assert "'forward-20261020-v2-treasury', 'treasury', now(), now() + interval '7 days', v1.rules" in MIG
    assert "where v1.protocol_id = 'forward-20260922-v1-treasury'" in MIG
    sql = "\n".join(l for l in MIG.splitlines() if not l.lstrip().startswith("--")).lower()
    for forbidden in ("delete ", "update ", "truncate", "brian_dip_multiasset_state", "net.http_post"):
        assert forbidden not in sql
    assert "cron.alter_job(j, schedule := '12,22,32,42,52 * * * *', active := true)" in MIG
