"""Real-Postgres integration checks for Brian's deterministic real-L2 capture lineage.

The dedicated CI job applies the intelligence base migration, PR #37's normalized L2 source-event
migration, then migration 012 from this PR. The vanilla postgres:16 service container does not
ship Supabase's platform roles, so the bootstrap mirrors the hosted `service_role` property that
matters to these tests: BYPASSRLS. Table grants/revokes and append-only protections remain under
test independently.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip(
    "psycopg2",
    reason="psycopg2 is only installed in the dedicated real-Postgres CI job",
)

ROOT = Path(__file__).resolve().parents[1]
BASE_MIGRATION = ROOT / "supabase" / "migrations" / "202609030001_brian_intelligence_memory.sql"
L2_FOUNDATION_MIGRATION = ROOT / "supabase" / "migrations" / "202609040010_brian_l2_book_events.sql"
L2_CAPTURE_MIGRATION = ROOT / "supabase" / "migrations" / "202609040012_brian_l2_capture_ordering.sql"
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="BRIAN_TEST_DATABASE_URL not set; real-Postgres tests only run in dedicated CI",
)

_BOOTSTRAP_SUPABASE_ROLES_SQL = """
do $$
begin
  if not exists (select 1 from pg_roles where rolname = 'anon') then
    create role anon nologin;
  end if;
  if not exists (select 1 from pg_roles where rolname = 'authenticated') then
    create role authenticated nologin;
  end if;
  if not exists (select 1 from pg_roles where rolname = 'service_role') then
    create role service_role nologin;
  end if;
end
$$;
alter role service_role bypassrls;
"""


def _connect():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    return conn


@pytest.fixture(scope="module", autouse=True)
def _apply_migrations():
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_BOOTSTRAP_SUPABASE_ROLES_SQL)
            cur.execute(BASE_MIGRATION.read_text(encoding="utf-8"))
            cur.execute(L2_FOUNDATION_MIGRATION.read_text(encoding="utf-8"))
            cur.execute(L2_CAPTURE_MIGRATION.read_text(encoding="utf-8"))
    finally:
        conn.close()


def _raw_capture(conn, *, capture_id: str | None = None) -> str:
    capture_id = capture_id or f"pytest-l2-raw-{uuid.uuid4().hex}"
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_raw_captures
              (capture_id, provider, record_type, observed_at, captured_at,
               provenance_uri, payload_hash, payload)
            values
              (%s, 'binance_public', 'l2_raw_segment',
               '2026-09-04T10:00:00Z', '2026-09-04T10:00:01Z',
               'wss://stream.binance.com', %s, %s::jsonb)
            """,
            (
                capture_id,
                f"hash-{capture_id}",
                json.dumps({"storage_bucket": "brian-intelligence-raw", "storage_path": f"pytest/{capture_id}.ndjson.gz"}),
            ),
        )
    return capture_id


def _raw_segment(
    conn,
    *,
    session: str = "pytest-session-a",
    first: int = 1,
    last: int = 2,
    count: int = 2,
    segment_id: str | None = None,
    capture_id: str | None = None,
) -> str:
    capture_id = capture_id or _raw_capture(conn)
    segment_id = segment_id or f"pytest-segment-{uuid.uuid4().hex}"
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_l2_raw_segments
              (segment_id, raw_capture_id, collector_session_id,
               first_arrival_seq, last_arrival_seq, message_count,
               observed_at, captured_at)
            values (%s,%s,%s,%s,%s,%s,'2026-09-04T10:00:00Z','2026-09-04T10:00:01Z')
            """,
            (segment_id, capture_id, session, first, last, count),
        )
    return segment_id


def _source_event(
    conn,
    *,
    segment_id: str,
    session: str = "pytest-session-a",
    arrival_seq: int = 1,
    raw_message_index: int = 0,
    kind: str = "depth_diff",
    transport: str = "binance_spot_diff_depth_ws",
    connection_generation: int = 1,
    sync_generation: int = 1,
    event_id: str | None = None,
) -> str:
    event_id = event_id or f"pytest-l2-event-{uuid.uuid4().hex}"
    payload = (
        {
            "firstUpdateId": "101",
            "finalUpdateId": "102",
            "bidMutations": [{"price": "100.00", "size": "1.2"}],
            "askMutations": [],
        }
        if kind == "depth_diff"
        else {
            "lastUpdateId": "102",
            "bids": [{"price": "100.00", "size": "1.2"}],
            "asks": [{"price": "101.00", "size": "1.1"}],
        }
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into public.brian_l2_source_events (
              event_id, kind, venue, symbol, exchange_event_at,
              collector_received_at, ingest_at, age_ms, clock_skew_ms,
              payload, source_lineage, evidence_class, shadow_only, live_execution,
              collector_session_id, arrival_seq, connection_generation, sync_generation,
              transport, raw_segment_id, raw_message_index
            ) values (
              %s,%s,'binance','BTCUSDT','2026-09-04T10:00:00Z',
              '2026-09-04T10:00:00Z','2026-09-04T10:00:00Z',0,0,
              %s::jsonb,'{}'::jsonb,'PROSPECTIVE_DEVELOPMENT_SHADOW',true,false,
              %s,%s,%s,%s,%s,%s,%s
            )
            """,
            (
                event_id,
                kind,
                json.dumps(payload),
                session,
                arrival_seq,
                connection_generation,
                sync_generation,
                transport,
                segment_id,
                raw_message_index,
            ),
        )
    return event_id


def test_service_role_matches_supabase_bypassrls_and_can_insert_select_capture_rows():
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("select rolbypassrls from pg_roles where rolname='service_role'")
            assert cur.fetchone() == (True,)
        capture = _raw_capture(conn)
        segment = _raw_segment(conn, capture_id=capture)
        event = _source_event(conn, segment_id=segment)
        with conn.cursor() as cur:
            cur.execute("set role service_role")
            cur.execute(
                "select collector_session_id, arrival_seq, sync_generation from public.brian_l2_source_events where event_id=%s",
                (event,),
            )
            assert cur.fetchone() == ("pytest-session-a", 1, 1)
            cur.execute("reset role")
    finally:
        conn.close()


def test_service_role_can_insert_but_cannot_update_or_delete_append_only_l2_rows():
    conn = _connect()
    try:
        capture = _raw_capture(conn)
        with conn.cursor() as cur:
            cur.execute("set role service_role")
            segment = _raw_segment(conn, capture_id=capture, session="pytest-service-role")
            event = _source_event(conn, segment_id=segment, session="pytest-service-role")
            with pytest.raises(psycopg2.Error):
                cur.execute("update public.brian_l2_source_events set age_ms=1 where event_id=%s", (event,))
            with pytest.raises(psycopg2.Error):
                cur.execute("delete from public.brian_l2_raw_segments where segment_id=%s", (segment,))
            cur.execute("reset role")
            cur.execute("select count(*) from public.brian_l2_source_events where event_id=%s", (event,))
            assert cur.fetchone()[0] == 1
    finally:
        conn.close()


def test_session_arrival_sequence_is_unique_across_segments():
    conn = _connect()
    try:
        session = f"pytest-session-{uuid.uuid4().hex}"
        first_segment = _raw_segment(conn, session=session, first=1, last=1, count=1)
        _source_event(conn, segment_id=first_segment, session=session, arrival_seq=1, raw_message_index=0)
        second_segment = _raw_segment(conn, session=session, first=2, last=2, count=1)
        with pytest.raises(psycopg2.Error):
            _source_event(conn, segment_id=second_segment, session=session, arrival_seq=1, raw_message_index=0)
    finally:
        conn.close()


def test_raw_message_index_is_unique_inside_a_segment():
    conn = _connect()
    try:
        session = f"pytest-session-{uuid.uuid4().hex}"
        segment = _raw_segment(conn, session=session, first=1, last=2, count=2)
        _source_event(conn, segment_id=segment, session=session, arrival_seq=1, raw_message_index=0)
        with pytest.raises(psycopg2.Error):
            _source_event(conn, segment_id=segment, session=session, arrival_seq=2, raw_message_index=0)
    finally:
        conn.close()


def test_source_event_cannot_reference_a_raw_segment_from_another_session():
    conn = _connect()
    try:
        segment = _raw_segment(conn, session="pytest-owner-session", first=1, last=1, count=1)
        with pytest.raises(psycopg2.Error):
            _source_event(conn, segment_id=segment, session="pytest-wrong-session")
    finally:
        conn.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arrival_seq": 0},
        {"connection_generation": 0},
        {"sync_generation": 0},
        {"kind": "depth_diff", "transport": "binance_spot_rest_depth_snapshot"},
        {"kind": "depth_snapshot", "transport": "binance_spot_diff_depth_ws"},
    ],
)
def test_source_event_capture_constraints_fail_closed(kwargs):
    conn = _connect()
    try:
        session = f"pytest-session-{uuid.uuid4().hex}"
        segment = _raw_segment(conn, session=session, first=1, last=1, count=1)
        with pytest.raises(psycopg2.Error):
            _source_event(conn, segment_id=segment, session=session, **kwargs)
    finally:
        conn.close()


def test_raw_segment_requires_contiguous_sequence_count_and_valid_shadow_flags():
    conn = _connect()
    try:
        capture = _raw_capture(conn)
        with pytest.raises(psycopg2.Error):
            _raw_segment(conn, capture_id=capture, first=10, last=12, count=2)
        capture2 = _raw_capture(conn)
        with conn.cursor() as cur:
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    """
                    insert into public.brian_l2_raw_segments
                      (segment_id,raw_capture_id,collector_session_id,first_arrival_seq,last_arrival_seq,
                       message_count,observed_at,captured_at,shadow_only,live_execution)
                    values (%s,%s,'pytest-shadow',1,1,1,
                            '2026-09-04T10:00:00Z','2026-09-04T10:00:01Z',false,false)
                    """,
                    (f"pytest-segment-{uuid.uuid4().hex}", capture2),
                )
        capture3 = _raw_capture(conn)
        with conn.cursor() as cur:
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    """
                    insert into public.brian_l2_raw_segments
                      (segment_id,raw_capture_id,collector_session_id,first_arrival_seq,last_arrival_seq,
                       message_count,observed_at,captured_at,shadow_only,live_execution)
                    values (%s,%s,'pytest-live',1,1,1,
                            '2026-09-04T10:00:00Z','2026-09-04T10:00:01Z',true,true)
                    """,
                    (f"pytest-segment-{uuid.uuid4().hex}", capture3),
                )
    finally:
        conn.close()


def test_raw_segments_form_one_gapless_nonoverlapping_chain_per_session():
    conn = _connect()
    try:
        session = f"pytest-chain-{uuid.uuid4().hex}"
        _raw_segment(conn, session=session, first=1, last=2, count=2)
        with pytest.raises(psycopg2.Error):
            _raw_segment(conn, session=session, first=2, last=3, count=2)
        with pytest.raises(psycopg2.Error):
            _raw_segment(conn, session=session, first=4, last=4, count=1)
        accepted = _raw_segment(conn, session=session, first=3, last=4, count=2)
        with conn.cursor() as cur:
            cur.execute(
                "select first_arrival_seq,last_arrival_seq from public.brian_l2_raw_segments where segment_id=%s",
                (accepted,),
            )
            assert cur.fetchone() == (3, 4)
    finally:
        conn.close()


def test_source_event_pointer_must_match_exact_raw_message_position():
    conn = _connect()
    try:
        session = f"pytest-pointer-{uuid.uuid4().hex}"
        segment = _raw_segment(conn, session=session, first=1, last=3, count=3)
        with pytest.raises(psycopg2.Error):
            _source_event(conn, segment_id=segment, session=session, arrival_seq=2, raw_message_index=0)
        with pytest.raises(psycopg2.Error):
            _source_event(conn, segment_id=segment, session=session, arrival_seq=4, raw_message_index=3)
        event = _source_event(conn, segment_id=segment, session=session, arrival_seq=2, raw_message_index=1)
        with conn.cursor() as cur:
            cur.execute(
                "select arrival_seq,raw_message_index from public.brian_l2_source_events where event_id=%s",
                (event,),
            )
            assert cur.fetchone() == (2, 1)
    finally:
        conn.close()


def test_capture_session_event_constraints_and_append_only_behavior():
    conn = _connect()
    try:
        event_id = f"pytest-session-event-{uuid.uuid4().hex}"
        with conn.cursor() as cur:
            cur.execute("set role service_role")
            cur.execute(
                """
                insert into public.brian_l2_capture_session_events
                  (session_event_id,collector_session_id,event_kind,venue,symbol,
                   connection_generation,sync_generation,arrival_seq_boundary,observed_at)
                values (%s,'pytest-session-life','SYNCED','binance','BTCUSDT',1,1,12,'2026-09-04T10:00:00Z')
                """,
                (event_id,),
            )
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    "update public.brian_l2_capture_session_events set event_kind='STOPPED' where session_event_id=%s",
                    (event_id,),
                )
            cur.execute("reset role")
        with conn.cursor() as cur:
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    """
                    insert into public.brian_l2_capture_session_events
                      (session_event_id,collector_session_id,event_kind,venue,connection_generation,observed_at)
                    values (%s,'pytest-session-life','MADE_UP_EVENT','binance',1,'2026-09-04T10:00:00Z')
                    """,
                    (f"pytest-session-event-{uuid.uuid4().hex}",),
                )
    finally:
        conn.close()
