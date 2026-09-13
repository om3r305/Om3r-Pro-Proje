import json
import os
import uuid

import pytest

psycopg2 = pytest.importorskip("psycopg2")

DB_URL = os.getenv("BRIAN_ENGINEER_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="BRIAN_ENGINEER_TEST_DATABASE_URL is not configured")
BASE_SHA = "a" * 40


def test_equal_timestamp_requests_choose_one_deterministic_latest_candidate():
    hypothesis_id = f"hypothesis-tie-{uuid.uuid4().hex}"
    prefix = f"pytest-tie-{uuid.uuid4().hex}"
    older_id = f"{prefix}-a"
    preferred_id = f"{prefix}-b"
    tied_at = "2026-09-13T12:00:00+00:00"

    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            for request_id in (older_id, preferred_id):
                cur.execute(
                    """
                    insert into public.brian_evolution_codegen_requests(
                      request_id,candidate_id,hypothesis_id,requested_at,parent_commit,branch_name,changed_paths,
                      objective,constraints,success_criteria,evidence_refs,contamination_declaration,
                      external_generator_required,required_human_review,metadata,evidence_class,
                      shadow_only,live_execution,autonomous_apply_allowed,created_at
                    ) values (
                      %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,true,true,%s,'POINT_IN_TIME',true,false,false,%s
                    )
                    """,
                    (
                        request_id,
                        f"candidate-{request_id}",
                        hypothesis_id,
                        tied_at,
                        "9" * 40,
                        f"evolution-candidate/{request_id}",
                        [f"supabase/functions/_shared/evolution_candidates/{request_id}.ts"],
                        "dedupe fixture",
                        ["shadow only"],
                        ["one request claimed"],
                        ["fixture:queue-dedupe"],
                        "point-in-time fixture",
                        json.dumps({"priority": 0.9, "hypothesis_kind": "CAPABILITY_GAP"}),
                        tied_at,
                    ),
                )
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                "select brian_private.claim_engineering_task(%s,%s,%s)",
                ("pytest-worker", BASE_SHA, older_id),
            )
            assert cur.fetchone()[0] is None
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                "select brian_private.claim_engineering_task(%s,%s,%s)",
                ("pytest-worker", BASE_SHA, preferred_id),
            )
            claimed = cur.fetchone()[0]
            assert claimed is not None
            assert claimed["task"]["request_id"] == preferred_id
            run_id = claimed["run_id"]
            cur.execute(
                "select brian_private.record_engineering_event(%s,'BLOCKED','BLOCKED',false,%s,%s::jsonb)",
                (run_id, BASE_SHA, json.dumps({"error": "test cleanup"})),
            )
        conn.commit()
