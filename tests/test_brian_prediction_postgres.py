"""Real-Postgres tests for the Prediction Intelligence storage migration (Phase 1b).

Applies supabase/migrations/20261019195019_brian_prediction_intelligence.sql to a real Postgres
database and proves the doctrine the migration claims: everything is OFF by default, evidence is
append-only and idempotent, missing probabilities stay NULL, only a VERIFIED wallet's trades are
stored, and anon/authenticated/service_role cannot do more than they are granted.

Skipped unless psycopg2 is installed and BRIAN_TEST_DATABASE_URL is set (the dedicated CI job
`verify-prediction-storage` in .github/workflows/brian-prediction-ci.yml sets both).
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 is only installed in the Postgres CI job")

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase" / "migrations" / "20261019195019_brian_prediction_intelligence.sql"
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="BRIAN_TEST_DATABASE_URL not set")

_BOOTSTRAP_ROLES = """
do $$ begin
  if not exists (select 1 from pg_roles where rolname = 'anon') then create role anon nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'authenticated') then create role authenticated nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'service_role') then create role service_role nologin; end if;
end $$;
"""

WALLET = "0x44832d0d2ec11187c1e77d786feb15f6a50254c6"
OTHER_WALLET = "0x3ef47f3553283b1857c023637d35636fcb9ffb21"


def _connect():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    return conn


@pytest.fixture(scope="module", autouse=True)
def _apply_migration():
    sql = MIGRATION.read_text(encoding="utf-8")
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_BOOTSTRAP_ROLES)
            cur.execute(sql)
    finally:
        conn.close()


@pytest.fixture()
def cur():
    conn = _connect()
    c = conn.cursor()
    c.execute("update public.brian_prediction_control set module_enabled = false, wallet_tracking_enabled = false")
    yield c
    c.execute("reset role")
    c.execute("update public.brian_prediction_control set module_enabled = false, wallet_tracking_enabled = false")
    conn.close()


def _uid() -> str:
    return uuid.uuid4().hex[:12]


def _market(uid: str, **over):
    base = {
        "market_key": f"polymarket:{uid}",
        "source": "polymarket-gamma",
        "market_id": uid,
        "condition_id": "0x" + uid.ljust(64, "a"),
        "event_id": "51456",
        "event_title": "How many Fed rate cuts in 2026?",
        "question": f"Will the Fed cut? {uid}",
        "slug": f"fed-cut-{uid}",
        "end_date": "2026-10-29T03:59:00Z",
        "outcomes": [{"label": "Yes", "clob_token_id": "1"}, {"label": "No", "clob_token_id": "2"}],
        "first_seen_at": "2026-10-19T19:30:00Z",
    }
    base.update(over)
    return base


def _snapshot(uid: str, **over):
    base = {
        "snapshot_id": f"polymarket:{uid}|2026-10-19T19:18:46.966Z",
        "market_key": f"polymarket:{uid}",
        "source_updated_at": "2026-10-19T19:18:46.966Z",
        "received_at": "2026-10-19T19:30:00Z",
        "outcomes": [
            {"label": "Yes", "implied_probability": 0.0005, "clob_token_id": "1"},
            {"label": "No", "implied_probability": None, "clob_token_id": "2"},
        ],
        "best_bid": None,
        "best_ask": 0.001,
        "spread": 0.001,
        "liquidity": 1188890.47,
        "volume_24h": 1249317.17,
        "active": True,
        "closed": False,
        "source_freshness_seconds": 673.0,
        "model_version": "brian.prediction-market.v0",
    }
    base.update(over)
    return base


def _trade(uid: str, wallet: str = WALLET, **over):
    base = {
        "idempotency_key": f"{wallet}:0x{uid.ljust(64, 'b')}:asset:BUY:0.07:10:1792436684",
        "wallet": wallet,
        "condition_id": "0xd4ca8d49b356d404cd01aef74285132ea95012e31dc4c92d863704c035058d22",
        "asset_id": "45780744039970711522644952415639884331285827806483628311008451591801847762772",
        "market_slug": "btc-updown-5m-1792436400",
        "title": "Bitcoin Up or Down - October 19, 3:00PM-3:05PM ET",
        "outcome": "Up",
        "side": "BUY",
        "price": 0.07,
        "size": 10,
        "usdc_size": 0.7,
        "transaction_hash": "0x" + uid.ljust(64, "b"),
        "traded_at": "2026-10-19T19:04:44Z",
        "received_at": "2026-10-19T19:30:00Z",
        "model_version": "brian.prediction-market.v0",
    }
    base.update(over)
    return base


def _health(uid: str, **over):
    base = {
        "health_id": f"gamma:{uid}",
        "source": "polymarket-gamma",
        "status": "OK",
        "http_status": 200,
        "attempts": 1,
        "latency_ms": 120,
        "records_seen": 2,
        "records_stored": 2,
        "detail": None,
        "observed_at": "2026-10-19T19:30:00Z",
        "metadata": {},
    }
    base.update(over)
    return base


def _record(cur, markets=(), snapshots=(), trades=(), health=()):
    cur.execute(
        "select public.brian_record_prediction_batch_v1(%s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)",
        (json.dumps(list(markets)), json.dumps(list(snapshots)), json.dumps(list(trades)), json.dumps(list(health))),
    )
    return cur.fetchone()[0]


def _enable(cur, wallets: bool = False):
    cur.execute("update public.brian_prediction_control set module_enabled = true, wallet_tracking_enabled = %s", (wallets,))


def _verify_wallet(cur, wallet: str = WALLET, status: str = "VERIFIED", watermark=None):
    cur.execute(
        "select public.brian_upsert_prediction_wallet_v1(%s, 'NIULAI4', %s, '[\"test\"]'::jsonb, %s)",
        (wallet, status, watermark),
    )


def _count(cur, table: str, where: str = "true", args=()):
    cur.execute(f"select count(*) from public.{table} where {where}", args)
    return cur.fetchone()[0]


def test_migration_is_idempotent_and_defaults_everything_off():
    conn = _connect()
    try:
        with conn.cursor() as c:
            c.execute(MIGRATION.read_text(encoding="utf-8"))
            c.execute("select module_enabled, wallet_tracking_enabled, shadow_only, live_execution from public.brian_prediction_control")
            assert c.fetchall() == [(False, False, True, False)]
            c.execute("select count(*) from public.brian_prediction_control")
            assert c.fetchone()[0] == 1
    finally:
        conn.close()


def test_disabled_module_writes_nothing(cur):
    uid = _uid()
    result = _record(cur, [_market(uid)], [_snapshot(uid)], [], [_health(uid)])
    assert result == {"status": "OPERATOR_STOPPED"}
    assert _count(cur, "brian_prediction_markets", "market_id = %s", (uid,)) == 0
    assert _count(cur, "brian_prediction_source_health", "health_id = %s", (f"gamma:{uid}",)) == 0


def test_record_is_idempotent_and_keeps_missing_values_null(cur):
    _enable(cur)
    uid = _uid()
    first = _record(cur, [_market(uid)], [_snapshot(uid)], [], [_health(uid)])
    assert first["status"] == "SUCCESS"
    assert first["markets"] == {"received": 1, "stored": 1}
    assert first["snapshots"] == {"received": 1, "stored": 1}
    assert first["health"] == {"received": 1, "stored": 1}
    again = _record(cur, [_market(uid)], [_snapshot(uid)], [], [_health(uid)])
    assert again["markets"]["stored"] == 0 and again["snapshots"]["stored"] == 0 and again["health"]["stored"] == 0
    assert _count(cur, "brian_prediction_market_snapshots", "market_key = %s", (f"polymarket:{uid}",)) == 1
    cur.execute("select best_bid, outcomes from public.brian_prediction_market_snapshots where market_key = %s", (f"polymarket:{uid}",))
    best_bid, outcomes = cur.fetchone()
    assert best_bid is None, "a missing bid must stay NULL, never 0"
    assert outcomes[1]["implied_probability"] is None


def test_snapshot_constraints_reject_impossible_values_and_orphans(cur):
    _enable(cur)
    uid = _uid()
    _record(cur, [_market(uid)])
    for bad in ({"best_ask": 1.5}, {"best_bid": -0.1}, {"spread": -1}, {"liquidity": -5}):
        with pytest.raises(psycopg2.errors.CheckViolation):
            _record(cur, [], [_snapshot(uid, snapshot_id=f"bad:{_uid()}", **bad)])
    with pytest.raises(psycopg2.errors.ForeignKeyViolation):
        _record(cur, [], [_snapshot(uid, snapshot_id=f"orphan:{_uid()}", market_key="polymarket:does-not-exist")])


def test_batch_size_is_bounded_and_sections_must_be_arrays(cur):
    _enable(cur)
    uid = _uid()
    with pytest.raises(psycopg2.errors.RaiseException):
        _record(cur, [_market(f"{uid}{i}") for i in range(301)])
    with pytest.raises(psycopg2.errors.RaiseException):
        cur.execute("select public.brian_record_prediction_batch_v1('{}'::jsonb, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb)")


def test_evidence_is_append_only_for_postgres_and_unwritable_for_service_role(cur):
    _enable(cur)
    uid = _uid()
    _record(cur, [_market(uid)], [_snapshot(uid)], [], [_health(uid)])
    cur.execute("reset role")
    for table in (
        "brian_prediction_markets",
        "brian_prediction_market_snapshots",
        "brian_prediction_source_health",
        "brian_prediction_wallet_trades",
    ):
        cur.execute("set role service_role")
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            cur.execute(f"update public.{table} set evidence_class = 'X'")
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            cur.execute(f"delete from public.{table}")
        cur.execute("reset role")
    # The trigger also blocks a non-superuser table owner path: postgres is the only exempt role.
    cur.execute("select array_agg(tgname::text order by tgname) from pg_trigger where tgname like 'brian_prediction_%_append_only' and not tgisinternal")
    names = set(cur.fetchone()[0])
    # Later phases may add more append-only tables; these four must always be guarded.
    assert {
        "brian_prediction_markets_append_only",
        "brian_prediction_market_snapshots_append_only",
        "brian_prediction_wallet_trades_append_only",
        "brian_prediction_source_health_append_only",
    } <= names


def test_anon_and_authenticated_have_no_access(cur):
    for role in ("anon", "authenticated"):
        cur.execute(f"set role {role}")
        for table in (
            "brian_prediction_control",
            "brian_prediction_wallets",
            "brian_prediction_markets",
            "brian_prediction_market_snapshots",
            "brian_prediction_wallet_trades",
            "brian_prediction_source_health",
        ):
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute(f"select * from public.{table}")
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            cur.execute("select public.brian_record_prediction_batch_v1('[]','[]','[]','[]')")
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            cur.execute("select public.brian_upsert_prediction_wallet_v1('%s','x','UNVERIFIED','[]')" % WALLET)
        cur.execute("reset role")


def test_service_role_can_call_writers_but_not_rewrite_control_shape(cur):
    cur.execute("set role service_role")
    cur.execute("select public.brian_record_prediction_batch_v1('[]','[]','[]','[]')")
    assert cur.fetchone()[0] == {"status": "OPERATOR_STOPPED"}
    cur.execute("update public.brian_prediction_control set module_enabled = false")
    with pytest.raises(psycopg2.errors.InsufficientPrivilege):
        cur.execute("delete from public.brian_prediction_control")
    with pytest.raises(psycopg2.errors.InsufficientPrivilege):
        cur.execute("insert into public.brian_prediction_control (control_id) values ('other')")


def test_control_row_cannot_enable_live_execution_or_clear_shadow(cur):
    with pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute("update public.brian_prediction_control set live_execution = true")
    with pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute("update public.brian_prediction_control set shadow_only = false")
    with pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute("update public.brian_prediction_control set max_markets = 0")


def test_wallet_registry_never_accepts_a_short_or_guessed_address(cur):
    for bad in ("0x4483...54c6", "0x44832d0d", "", "44832d0d2ec11187c1e77d786feb15f6a50254c6"):
        with pytest.raises(psycopg2.errors.RaiseException):
            _verify_wallet(cur, bad)
    with pytest.raises(psycopg2.errors.RaiseException):
        _verify_wallet(cur, WALLET, "TRUSTED")
    with pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute(
            "insert into public.brian_prediction_wallets (wallet, display_name, verification_status) values (%s, 'x', 'VERIFIED')",
            ("0x" + "c" * 40,),
        )


def test_wallet_watermark_only_moves_forward_and_failed_reverification_downgrades(cur):
    wallet = "0x" + _uid().ljust(40, "d")
    _verify_wallet(cur, wallet, "VERIFIED", "2026-10-19T19:00:00Z")
    _verify_wallet(cur, wallet, "VERIFIED", "2026-10-19T18:00:00Z")
    cur.execute("select trade_watermark from public.brian_prediction_wallets where wallet = %s", (wallet,))
    assert cur.fetchone()[0].isoformat().startswith("2026-10-19T19:00:00")
    _verify_wallet(cur, wallet, "VERIFIED", None)
    cur.execute("select trade_watermark from public.brian_prediction_wallets where wallet = %s", (wallet,))
    assert cur.fetchone()[0].isoformat().startswith("2026-10-19T19:00:00"), "a NULL watermark never erases progress"
    _verify_wallet(cur, wallet, "AMBIGUOUS")
    cur.execute("select verification_status, verified_at from public.brian_prediction_wallets where wallet = %s", (wallet,))
    assert cur.fetchone() == ("AMBIGUOUS", None)


def test_trades_only_stored_for_verified_wallet_while_tracking_is_enabled(cur):
    uid = _uid()
    verified = "0x" + _uid().ljust(40, "e")
    unverified = "0x" + _uid().ljust(40, "f")
    _verify_wallet(cur, verified, "VERIFIED")
    _verify_wallet(cur, unverified, "UNVERIFIED")
    trades = [
        _trade(uid, verified),
        _trade(uid + "1", verified, price=0.06, size=150),
        _trade(uid + "2", unverified),
    ]

    _enable(cur, wallets=False)
    off = _record(cur, [], [], trades)
    assert off["trades"] == {"received": 3, "stored": 0, "tracking_enabled": False}

    _enable(cur, wallets=True)
    on = _record(cur, [], [], trades)
    assert on["trades"] == {"received": 3, "stored": 2, "tracking_enabled": True}, "unverified wallet trades are dropped"
    again = _record(cur, [], [], trades)
    assert again["trades"]["stored"] == 0, "re-ingest is a no-op"
    assert _count(cur, "brian_prediction_wallet_trades", "wallet = %s", (unverified,)) == 0

    cur.execute("select side, price, size from public.brian_prediction_wallet_trades where wallet = %s order by price", (verified,))
    assert [(s, float(p), float(z)) for s, p, z in cur.fetchall()] == [("BUY", 0.06, 150.0), ("BUY", 0.07, 10.0)]


def test_trade_constraints_reject_impossible_rows(cur):
    wallet = "0x" + _uid().ljust(40, "9")
    _verify_wallet(cur, wallet, "VERIFIED")
    _enable(cur, wallets=True)
    for bad in ({"price": 1.2}, {"size": 0}, {"side": "HOLD"}, {"transaction_hash": "0xabc"}):
        with pytest.raises(psycopg2.errors.CheckViolation):
            _record(cur, [], [], [_trade(_uid(), wallet, **bad)])
