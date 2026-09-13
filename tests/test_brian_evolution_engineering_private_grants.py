import os

import pytest

psycopg2 = pytest.importorskip("psycopg2")

DB_URL = os.getenv("BRIAN_ENGINEER_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="BRIAN_ENGINEER_TEST_DATABASE_URL is not configured")

PRIVATE_SIGNATURES = (
    "brian_private.claim_engineering_task(text,text,text)",
    "brian_private.record_engineering_event(uuid,text,text,boolean,text,jsonb)",
    "brian_private.measure_engineering_run(uuid,text,jsonb)",
    "brian_private.approve_engineering_run(uuid,text,text)",
)


def test_untrusted_roles_cannot_execute_private_engineering_rpcs():
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            for role in ("anon", "authenticated"):
                cur.execute("select has_schema_privilege(%s,'brian_private','USAGE')", (role,))
                assert cur.fetchone()[0] is False
                for signature in PRIVATE_SIGNATURES:
                    cur.execute("select has_function_privilege(%s,%s,'EXECUTE')", (role, signature))
                    assert cur.fetchone()[0] is False, (role, signature)
