import os
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2")

DB_URL = os.getenv("BRIAN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="BRIAN_TEST_DATABASE_URL not configured")
ROOT = Path(__file__).resolve().parents[1]


def test_latest_micro_state_rpc_returns_exactly_one_latest_row_per_eye():
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            for role in ("anon", "authenticated", "service_role"):
                cur.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role,))
                if cur.fetchone() is None:
                    cur.execute(f'CREATE ROLE "{role}" NOLOGIN')
            cur.execute("DROP FUNCTION IF EXISTS public.brian_latest_micro_book_ticks(text[])")
            cur.execute("DROP TABLE IF EXISTS public.brian_micro_book_ticks CASCADE")
            cur.execute("""
                CREATE TABLE public.brian_micro_book_ticks (
                    tick_id text PRIMARY KEY,
                    eye_id text NOT NULL,
                    starting_equity numeric NOT NULL,
                    equity_after numeric NOT NULL,
                    peak_equity_after numeric NOT NULL,
                    max_drawdown_pct_after double precision NOT NULL,
                    target_direction smallint NOT NULL,
                    observed_mid_price numeric NOT NULL,
                    observed_at timestamptz NOT NULL
                )
            """)
            cur.execute("CREATE INDEX brian_micro_book_tick_eye_time_idx ON public.brian_micro_book_ticks (eye_id, observed_at DESC)")
            cur.execute("""
                INSERT INTO public.brian_micro_book_ticks
                (tick_id,eye_id,starting_equity,equity_after,peak_equity_after,max_drawdown_pct_after,target_direction,observed_mid_price,observed_at)
                VALUES
                ('a-old','eye-a',5,5.1,5.2,1.0,1,100,'2026-09-06T07:00:00Z'),
                ('a-new','eye-a',5,5.3,5.3,1.0,-1,101,'2026-09-06T07:01:00Z'),
                ('b-old','eye-b',3,2.9,3.0,3.3,0,50,'2026-09-06T07:00:30Z'),
                ('b-new','eye-b',3,3.1,3.1,3.3,1,51,'2026-09-06T07:02:00Z')
            """)
            cur.execute((ROOT / "supabase/migrations/202609060001_brian_latest_micro_book_state_rpc.sql").read_text())
            cur.execute("SELECT eye_id,target_direction,observed_mid_price FROM public.brian_latest_micro_book_ticks(ARRAY['eye-a','eye-b']::text[]) ORDER BY eye_id")
            rows = cur.fetchall()
            assert len(rows) == 2
            assert rows[0] == ("eye-a", -1, 101)
            assert rows[1] == ("eye-b", 1, 51)
