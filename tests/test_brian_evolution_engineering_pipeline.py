import json
import os
import uuid

import pytest

psycopg2 = pytest.importorskip("psycopg2")

DB_URL = os.getenv("BRIAN_ENGINEER_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="BRIAN_ENGINEER_TEST_DATABASE_URL is not configured")

BASE_SHA = "a" * 40
CANDIDATE_SHA = "b" * 40
MERGED_SHA = "c" * 40
ROLLBACK_SHA = "d" * 40


def connect():
    return psycopg2.connect(DB_URL)


def insert_request(cur, request_id: str, hypothesis_id: str):
    cur.execute(
        """
        insert into public.brian_evolution_codegen_requests(
          request_id,candidate_id,hypothesis_id,requested_at,parent_commit,branch_name,changed_paths,
          objective,constraints,success_criteria,evidence_refs,contamination_declaration,
          external_generator_required,required_human_review,metadata,evidence_class,
          shadow_only,live_execution,autonomous_apply_allowed,created_at
        ) values (
          %s,%s,%s,now(),%s,%s,%s,%s,%s,%s,%s,%s,true,true,%s,'POINT_IN_TIME',true,false,false,now()
        )
        """,
        (
            request_id,
            f"candidate-{request_id}",
            hypothesis_id,
            "9" * 40,
            f"evolution-candidate/{request_id}",
            [f"supabase/functions/_shared/evolution_candidates/{request_id}.ts"],
            "bounded engineering objective",
            ["shadow only", "DIP excluded"],
            ["deterministic evidence passes"],
            ["fixture:evidence"],
            "point-in-time fixture",
            json.dumps({"priority": 0.9, "hypothesis_kind": "CAPABILITY_GAP"}),
        ),
    )


def claim(cur, request_id: str):
    cur.execute(
        "select brian_private.claim_engineering_task(%s,%s,%s)",
        ("pytest-worker", BASE_SHA, request_id),
    )
    result = cur.fetchone()[0]
    assert result is not None
    return result


def record(cur, run_id, kind, phase, sha=CANDIDATE_SHA, payload=None):
    cur.execute(
        "select brian_private.record_engineering_event(%s,%s,%s,true,%s,%s::jsonb)",
        (run_id, kind, phase, sha, json.dumps(payload or {})),
    )


def advance_to_preview(cur, run_id):
    record(cur, run_id, "UNDERSTAND", "UNDERSTAND", None, {"evidence": "repo inspected"})
    record(cur, run_id, "PLAN", "PLAN", None, {"evidence": "bounded plan"})
    record(cur, run_id, "CODE", "CODE", CANDIDATE_SHA, {"evidence": "local exact commit"})
    record(cur, run_id, "COMPILE", "COMPILE", CANDIDATE_SHA, {"typecheck": True})
    record(cur, run_id, "UNIT_REGRESSION", "TEST", CANDIDATE_SHA, {"tests": True})
    record(cur, run_id, "REPLAY", "REPLAY", CANDIDATE_SHA, {"replay": True})
    record(cur, run_id, "STRESS", "REPLAY", CANDIDATE_SHA, {"stress": True})
    record(cur, run_id, "INDEPENDENT_REVIEW", "REVIEW", CANDIDATE_SHA, {"verdict": "PASS"})
    record(cur, run_id, "PR_CREATED", "PR", CANDIDATE_SHA, {"pr_url": "https://example.invalid/pr/1", "pr_number": 1})
    record(cur, run_id, "PREVIEW_EQUIVALENT", "PREVIEW", CANDIDATE_SHA, {"preview_url": "https://example.invalid/preview"})


def measure(cur, run_id):
    payload = {
        "measurement_kind": "EXACT_COMMIT_ENGINEERING",
        "exact_commit_sha": CANDIDATE_SHA,
        "patch_bytes": 2048,
        "changed_files": 3,
        "source_files": 1,
        "test_files": 2,
        "replay_files": 1,
        "stress_files": 1,
        "protected_scope_clear": True,
        "indirect_dip_dependency_clear": True,
        "canonical_behavior_promotion": False,
        "shadow_only": True,
    }
    cur.execute(
        "select brian_private.measure_engineering_run(%s,%s,%s::jsonb)",
        (run_id, CANDIDATE_SHA, json.dumps(payload)),
    )
    return cur.fetchone()[0]


def test_pipeline_rejects_stage_skips_and_requires_exact_commit_owner_approval():
    request_id = f"pytest-{uuid.uuid4().hex}"
    hypothesis_id = f"hypothesis-{uuid.uuid4().hex}"
    with connect() as conn:
        with conn.cursor() as cur:
            insert_request(cur, request_id, hypothesis_id)
            claimed = claim(cur, request_id)
            run_id = claimed["run_id"]
            assert claimed["base_sha"] == BASE_SHA
            assert claimed["task"]["source_parent_sha"] == "9" * 40

        conn.commit()

        with conn.cursor() as cur:
            with pytest.raises(psycopg2.Error):
                record(cur, run_id, "VERCEL_PREVIEW", "PREVIEW", CANDIDATE_SHA)
            conn.rollback()

        with conn.cursor() as cur:
            advance_to_preview(cur, run_id)
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    "select brian_private.approve_engineering_run(%s,%s,%s)",
                    (run_id, CANDIDATE_SHA, "om3r305"),
                )
            conn.rollback()

        with conn.cursor() as cur:
            measured = measure(cur, run_id)
            assert measured["status"] == "WAITING_HUMAN_APPROVAL"
            with pytest.raises(psycopg2.Error):
                cur.execute(
                    "select brian_private.approve_engineering_run(%s,%s,%s)",
                    (run_id, "e" * 40, "om3r305"),
                )
            conn.rollback()

        with conn.cursor() as cur:
            cur.execute(
                "select brian_private.approve_engineering_run(%s,%s,%s)",
                (run_id, CANDIDATE_SHA, "om3r305"),
            )
            assert cur.fetchone()[0]["status"] == "APPROVED"
            record(cur, run_id, "DEPLOYED", "DEPLOY", MERGED_SHA, {"previous_good_sha": BASE_SHA})
            record(cur, run_id, "MONITOR_HEALTHY", "MONITOR", MERGED_SHA, {"status": "HEALTHY"})
            record(cur, run_id, "COMPLETE", "COMPLETE", MERGED_SHA, {"status": "COMPLETE"})
            cur.execute(
                "select phase,status,human_approval_status,human_approved_by,deployed_sha,monitor_status from public.brian_evolution_engineering_runs where run_id=%s",
                (run_id,),
            )
            row = cur.fetchone()
            assert row == ("COMPLETE", "COMPLETE", "APPROVED", "om3r305", MERGED_SHA, "HEALTHY")
            cur.execute("select event_kind from public.brian_evolution_engineering_events where run_id=%s order by event_id", (run_id,))
            events = [r[0] for r in cur.fetchall()]
            assert events == [
                "TASK_CLAIMED",
                "UNDERSTAND",
                "PLAN",
                "CODE",
                "COMPILE",
                "UNIT_REGRESSION",
                "REPLAY",
                "STRESS",
                "INDEPENDENT_REVIEW",
                "PR_CREATED",
                "PREVIEW_EQUIVALENT",
                "ENGINEERING_MEASURE",
                "HUMAN_APPROVAL_REQUIRED",
                "HUMAN_APPROVED",
                "DEPLOYED",
                "MONITOR_HEALTHY",
                "COMPLETE",
            ]


def test_pipeline_can_rollback_only_after_human_approval_stage():
    request_id = f"pytest-{uuid.uuid4().hex}"
    hypothesis_id = f"hypothesis-{uuid.uuid4().hex}"
    with connect() as conn:
        with conn.cursor() as cur:
            insert_request(cur, request_id, hypothesis_id)
            run_id = claim(cur, request_id)["run_id"]
            advance_to_preview(cur, run_id)
            measure(cur, run_id)
            cur.execute(
                "select brian_private.approve_engineering_run(%s,%s,%s)",
                (run_id, CANDIDATE_SHA, "om3r305"),
            )
            record(cur, run_id, "ROLLBACK", "ROLLBACK", ROLLBACK_SHA, {"previous_good_sha": BASE_SHA})
            cur.execute(
                "select phase,status,rollback_sha from public.brian_evolution_engineering_runs where run_id=%s",
                (run_id,),
            )
            assert cur.fetchone() == ("ROLLBACK", "ROLLED_BACK", ROLLBACK_SHA)


def test_autonomous_claim_is_dormant_by_default():
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("select autonomous_claim_enabled,require_human_approval,max_concurrent_runs from public.brian_evolution_engineering_control where control_id='default'")
            assert cur.fetchone() == (False, True, 1)
            cur.execute("select brian_private.claim_engineering_task(%s,%s,null)", ("pytest-worker", BASE_SHA))
            assert cur.fetchone()[0] is None
