"""Real-Postgres tests for the Research Lab ledger (append-only papers/claims/assumptions/results).

Proves: the repository ledger file loads as-is; re-loading identical content is idempotent; a
changed claim or paper under an existing id is refused; every result (PASS and FAIL) is kept and
the latest one is reported; rows can be neither updated nor deleted by service_role; client
roles cannot read anything; trade-signal results are refused.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 is only installed in the Postgres CI job")

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "supabase" / "migrations"
ORDER = [
    MIGRATIONS / "20261019195019_brian_prediction_intelligence.sql",  # defines brian_reject_mutation()
    MIGRATIONS / "20261020211802_brian_research_ledger.sql",
    MIGRATIONS / "20261020211823_brian_research_ledger_writer_grants.sql",
]
LEDGER_PATH = "research/research_lab/ledger/arxiv-2602.19520.json"
LEDGER_TEXT = (ROOT / LEDGER_PATH).read_text(encoding="utf-8")
LEDGER = json.loads(LEDGER_TEXT)
LEDGER_SHA = hashlib.sha256(LEDGER_TEXT.encode("utf-8")).hexdigest()
DATABASE_URL = os.environ.get("BRIAN_TEST_DATABASE_URL")
RUN = uuid.uuid4().hex[:10]  # results are append-only, so every test run uses fresh result ids
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="BRIAN_TEST_DATABASE_URL not set")

_BOOTSTRAP = """
do $$ begin
  if not exists (select 1 from pg_roles where rolname = 'anon') then create role anon nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'authenticated') then create role authenticated nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'service_role') then create role service_role nologin; end if;
end $$;
"""


@pytest.fixture(scope="module")
def conn():
    c = psycopg2.connect(DATABASE_URL)
    c.autocommit = True
    with c.cursor() as cur:
        cur.execute(_BOOTSTRAP)
        # Simulate Supabase: new functions get EXECUTE for service_role by default. The writer
        # grants migration must remove it again (this is exactly what CORE showed on 2026-10-20).
        cur.execute("alter default privileges in schema public grant execute on functions to service_role")
        for path in ORDER:
            cur.execute(path.read_text(encoding="utf-8"))
    yield c
    with c.cursor() as cur:
        cur.execute("alter default privileges in schema public revoke execute on functions from service_role")
    c.close()


@pytest.fixture()
def cur(conn):
    c = conn.cursor()
    yield c
    c.execute("reset role")


def load(cur, ledger=LEDGER, sha=LEDGER_SHA):
    cur.execute("select public.brian_record_research_ledger_v1(%s::jsonb, %s, %s)", (json.dumps(ledger), LEDGER_PATH, sha))
    return cur.fetchone()[0]


def result(cur, **over):
    r = {"result_id": "pm-political-slope-oos-v1:test-1-"+RUN, "claim_id": "arxiv:2602.19520#C-01",
         "experiment_id": "pm-political-slope-oos-v1", "status": "UNVERIFIED", "status_reason": "interval includes 1",
         "observed_at": "2026-10-20T19:30:00Z", "code_commit": "c56aff8", "run_url": None,
         "dataset_sha256": "b" * 64, "evidence_class": "OUT_OF_SAMPLE_ADAPTED_REPLICATION",
         "metrics": {"slope": 1.1, "lower": 0.9, "upper": 1.3}, "shadow_only": True, "trade_signal": False}
    r.update(over)
    cur.execute("select public.brian_record_research_result_v1(%s::jsonb)", (json.dumps(r),))
    return cur.fetchone()[0]


def status(cur):
    cur.execute("select public.brian_research_status_v1(now() + interval '1 day')")
    return cur.fetchone()[0]


def test_repository_ledger_loads_and_is_idempotent(cur):
    first = load(cur)
    assert first["status"] == "STORED" and first["paper_id"] == "arxiv:2602.19520"
    again = load(cur)
    assert again == {"status": "STORED", "paper_id": "arxiv:2602.19520", "new_claims": 0, "new_assumptions": 0}
    cur.execute("select count(*) from public.brian_research_claims where paper_id = 'arxiv:2602.19520'")
    assert cur.fetchone()[0] == len(LEDGER["claims"])
    cur.execute("select count(*) from public.brian_research_assumptions where paper_id = 'arxiv:2602.19520'")
    assert cur.fetchone()[0] == len(LEDGER["assumptions"])


def test_changed_content_under_same_id_is_refused(cur):
    load(cur)
    changed = copy.deepcopy(LEDGER)
    changed["claims"][0]["statement"] = "Polymarket political markets are perfectly calibrated (rewritten)."
    with pytest.raises(psycopg2.errors.RaiseException):
        load(cur, changed)
    renamed = copy.deepcopy(LEDGER)
    renamed["paper"]["title"] = "Another title"
    assert load(cur, renamed)["reason"] == "PAPER_ID_CONTENT_CHANGED"
    bad = copy.deepcopy(LEDGER)
    bad["schema"] = "v0"
    assert load(cur, bad)["reason"] == "UNKNOWN_SCHEMA"


def test_untested_claim_is_not_started_and_all_results_are_kept(cur):
    load(cur)
    s = status(cur)
    by_id = {c["claim_id"]: c for c in s["claims"]}
    assert by_id["arxiv:2602.19520#C-02"]["status"] == "NOT_STARTED"
    assert result(cur, result_id="pm-political-slope-oos-v1:fail-1-"+RUN, status="FAIL", status_reason="interval below 1",
                  observed_at="2026-10-20T19:00:00Z")["status"] == "STORED"
    assert result(cur, result_id="pm-political-slope-oos-v1:pass-2-"+RUN, status="PASS", status_reason="interval above 1",
                  observed_at="2026-10-20T20:00:00Z")["status"] == "STORED"
    assert result(cur, result_id="pm-political-slope-oos-v1:pass-2-"+RUN)["status"] == "DUPLICATE"
    s = status(cur)
    c01 = {c["claim_id"]: c for c in s["claims"]}["arxiv:2602.19520#C-01"]
    assert c01["status"] == "PASS" and c01["runs"] >= 2  # the earlier FAIL is kept, latest is shown
    assert s["trade_signal"] is False and s["counts"]["NOT_STARTED"] >= 1
    assert s["papers"][0]["assumptions"][0]["assumption_id"] == "A-01"


def test_trade_signal_and_bad_status_refused(cur):
    load(cur)
    assert result(cur, result_id="x-trade-signal-1-"+RUN, trade_signal=True)["reason"] == "NOT_SHADOW_ONLY"
    with pytest.raises(psycopg2.errors.CheckViolation):
        result(cur, result_id="x-bad-status-1-"+RUN, status="PROFITABLE")
    with pytest.raises(psycopg2.errors.ForeignKeyViolation):
        result(cur, result_id="x-unknown-claim-1-"+RUN, claim_id="arxiv:0000.00000#C-01")


def test_append_only_and_role_isolation(cur):
    load(cur)
    result(cur, result_id="pm-political-slope-oos-v1:iso-1-"+RUN)
    cur.execute("set role service_role")
    # SELECT is granted (on Supabase service_role also bypasses RLS; the local role does not).
    cur.execute("select count(*) from public.brian_research_claim_results")
    assert cur.fetchone()[0] >= 0
    for stmt in ("update public.brian_research_claim_results set status = 'PASS'",
                 "delete from public.brian_research_claims",
                 "insert into public.brian_research_papers (paper_id, title, url, ledger_path, ledger_sha256) values ('arxiv:1', 'x', 'https://x', 'research/research_lab/ledger/x.json', repeat('a', 64))"):
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            cur.execute(stmt)
    with pytest.raises(psycopg2.errors.InsufficientPrivilege):
        cur.execute("select public.brian_record_research_result_v1('{}'::jsonb)")
    cur.execute("reset role")
    cur.execute("select has_function_privilege('service_role', 'public.brian_record_research_ledger_v1(jsonb, text, text)', 'execute'), "
                "has_function_privilege('service_role', 'public.brian_research_status_v1(timestamptz)', 'execute')")
    assert cur.fetchone() == (False, True)
    # Even the owner path refuses edits through the append-only trigger for non-postgres roles.
    for role in ("anon", "authenticated"):
        cur.execute(f"set role {role}")
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            cur.execute("select 1 from public.brian_research_claims limit 1")
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            cur.execute("select public.brian_research_status_v1(now())")
        cur.execute("reset role")
