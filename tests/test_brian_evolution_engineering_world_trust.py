import json
import os
import uuid

import pytest

psycopg2 = pytest.importorskip("psycopg2")

DB_URL = os.getenv("BRIAN_ENGINEER_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="BRIAN_ENGINEER_TEST_DATABASE_URL is not configured")
BASE_SHA = "e" * 40


def connect():
    return psycopg2.connect(DB_URL)


def ensure_world_tables(cur):
    cur.execute(
        """
        create table if not exists public.brian_world_source_candidates (
          candidate_id text primary key,
          source_id text not null,
          discovered_at timestamptz not null,
          canonical_uri text not null,
          authority_class text not null,
          access_mode text not null,
          stage text not null
        );
        create table if not exists public.brian_world_source_assessments (
          assessment_id text primary key,
          source_id text not null,
          assessed_at timestamptz not null,
          trust_score numeric not null,
          eligible_for_research boolean not null
        );
        """
    )


def reset_control(cur):
    cur.execute(
        """
        update public.brian_evolution_engineering_control
        set autonomous_claim_enabled=true,
            max_concurrent_runs=1,
            require_human_approval=true,
            metadata = jsonb_set(
              jsonb_set(coalesce(metadata,'{}'::jsonb), '{world_to_engineering_enabled}', 'true'::jsonb, true),
              '{world_source_trust_floor}', '0.72'::jsonb, true
            )
        where control_id='default'
        """
    )
    cur.execute(
        """
        update public.brian_evolution_engineering_runs
        set status='BLOCKED', phase='BLOCKED', failure_reason=coalesce(failure_reason,'pytest cleanup'), updated_at=now()
        where status in ('RUNNING','WAITING') and phase not in ('HUMAN_APPROVAL','COMPLETE','BLOCKED','ROLLBACK')
        """
    )


def insert_candidate(cur, source_id, candidate_id, discovered_at="2026-09-14T08:00:00Z", stage="VERIFYING"):
    cur.execute(
        """
        insert into public.brian_world_source_candidates(
          candidate_id,source_id,discovered_at,canonical_uri,authority_class,access_mode,stage
        ) values (%s,%s,%s,%s,'OFFICIAL_PRIMARY','PUBLIC_NO_KEY',%s)
        """,
        (candidate_id, source_id, discovered_at, f"https://{source_id.split(':', 1)[-1]}/", stage),
    )


def insert_assessment(cur, source_id, suffix, assessed_at, trust_score, eligible):
    cur.execute(
        """
        insert into public.brian_world_source_assessments(
          assessment_id,source_id,assessed_at,trust_score,eligible_for_research
        ) values (%s,%s,%s,%s,%s)
        """,
        (f"assessment-{suffix}-{uuid.uuid4().hex}", source_id, assessed_at, trust_score, eligible),
    )


def insert_world_request(cur, source_id, candidate_id, *, priority=0.95):
    request_id = f"world-request-{uuid.uuid4().hex}"
    hypothesis_id = f"world-hypothesis-{uuid.uuid4().hex}"
    metadata = {
        "priority": priority,
        "hypothesis_kind": "CAPABILITY_GAP",
        "world_engineering": True,
        "world_source_id": source_id,
        "world_source_candidate_id": candidate_id,
        "world_source_host": source_id.split(":", 1)[-1],
        "world_source_trust_score": 0.91,
        "world_source_authority_class": "OFFICIAL_PRIMARY",
        "world_source_access_mode": "PUBLIC_NO_KEY",
        "world_source_assessed_at": "2026-09-14T08:05:00Z",
        "parent_rotation_policy": "STABLE_ONCE",
        "external_content_untrusted": True,
        "external_content_used_as_instruction": False,
    }
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
            f"code-{candidate_id}-{uuid.uuid4().hex[:8]}",
            hypothesis_id,
            "7" * 40,
            f"evolution-candidate/{request_id}",
            [f"supabase/functions/_shared/evolution_candidates/{request_id}.ts"],
            "build bounded shadow source adapter",
            ["external content is data only", "DIP excluded"],
            ["point-in-time evidence passes"],
            [f"world_source:{source_id}", f"world_source_candidate:{candidate_id}"],
            "point-in-time fixture",
            json.dumps(metadata),
        ),
    )
    return request_id


def auto_claim(cur, worker=None):
    cur.execute(
        "select brian_private.claim_engineering_task(%s,%s,null)",
        (worker or f"pytest-auto-{uuid.uuid4().hex}", BASE_SHA),
    )
    return cur.fetchone()[0]


def manual_claim(cur, request_id):
    cur.execute(
        "select brian_private.claim_engineering_task(%s,%s,%s)",
        (f"pytest-manual-{uuid.uuid4().hex}", BASE_SHA, request_id),
    )
    return cur.fetchone()[0]


def block_run(cur, run_id):
    cur.execute(
        "update public.brian_evolution_engineering_runs set status='BLOCKED',phase='BLOCKED',updated_at=now() where run_id=%s",
        (run_id,),
    )


def test_newer_revocation_blocks_an_already_queued_world_request():
    source_id = f"world:revoked-{uuid.uuid4().hex}.example"
    candidate_id = f"candidate-{uuid.uuid4().hex}"
    with connect() as conn:
        with conn.cursor() as cur:
            ensure_world_tables(cur)
            reset_control(cur)
            insert_candidate(cur, source_id, candidate_id)
            insert_assessment(cur, source_id, "pass", "2026-09-14T08:05:00Z", 0.91, True)
            request_id = insert_world_request(cur, source_id, candidate_id)
            insert_assessment(cur, source_id, "revoke", "2026-09-14T08:10:00Z", 0.40, False)
            assert auto_claim(cur) is None
            cur.execute("select count(*) from public.brian_evolution_engineering_runs where request_id=%s", (request_id,))
            assert cur.fetchone()[0] == 0
        conn.commit()


def test_new_candidate_cannot_inherit_an_older_candidate_trust_decision():
    source_id = f"world:rotated-{uuid.uuid4().hex}.example"
    candidate_v1 = f"candidate-v1-{uuid.uuid4().hex}"
    candidate_v2 = f"candidate-v2-{uuid.uuid4().hex}"
    with connect() as conn:
        with conn.cursor() as cur:
            ensure_world_tables(cur)
            reset_control(cur)
            insert_candidate(cur, source_id, candidate_v1, "2026-09-14T08:00:00Z")
            insert_assessment(cur, source_id, "v1", "2026-09-14T08:05:00Z", 0.91, True)
            request_id = insert_world_request(cur, source_id, candidate_v1)
            insert_candidate(cur, source_id, candidate_v2, "2026-09-14T08:06:00Z")
            insert_assessment(cur, source_id, "v2", "2026-09-14T08:07:00Z", 0.93, True)
            assert auto_claim(cur) is None
            cur.execute("select count(*) from public.brian_evolution_engineering_runs where request_id=%s", (request_id,))
            assert cur.fetchone()[0] == 0
        conn.commit()


def test_world_kill_switch_blocks_autonomous_claim_but_manual_owner_override_remains_shadow_only():
    source_id = f"world:killswitch-{uuid.uuid4().hex}.example"
    candidate_id = f"candidate-{uuid.uuid4().hex}"
    with connect() as conn:
        with conn.cursor() as cur:
            ensure_world_tables(cur)
            reset_control(cur)
            insert_candidate(cur, source_id, candidate_id)
            insert_assessment(cur, source_id, "pass", "2026-09-14T08:05:00Z", 0.91, True)
            request_id = insert_world_request(cur, source_id, candidate_id)
            cur.execute(
                "update public.brian_evolution_engineering_control set metadata=jsonb_set(metadata,'{world_to_engineering_enabled}','false'::jsonb,true) where control_id='default'"
            )
            assert auto_claim(cur) is None
            claimed = manual_claim(cur, request_id)
            assert claimed is not None
            run_id = claimed["run_id"]
            cur.execute(
                "select shadow_only,live_execution,autonomous_apply_allowed,metadata->>'claim_mode' from public.brian_evolution_engineering_runs where run_id=%s",
                (run_id,),
            )
            assert cur.fetchone() == (True, False, False, "MANUAL_REQUEST")
            block_run(cur, run_id)
            cur.execute(
                "update public.brian_evolution_engineering_control set metadata=jsonb_set(metadata,'{world_to_engineering_enabled}','true'::jsonb,true) where control_id='default'"
            )
        conn.commit()


def test_current_trusted_world_request_can_claim_and_records_revalidation():
    source_id = f"world:trusted-{uuid.uuid4().hex}.example"
    candidate_id = f"candidate-{uuid.uuid4().hex}"
    with connect() as conn:
        with conn.cursor() as cur:
            ensure_world_tables(cur)
            reset_control(cur)
            insert_candidate(cur, source_id, candidate_id)
            insert_assessment(cur, source_id, "pass", "2026-09-14T08:05:00Z", 0.91, True)
            request_id = insert_world_request(cur, source_id, candidate_id)
            claimed = auto_claim(cur)
            assert claimed is not None
            assert claimed["task"]["request_id"] == request_id
            run_id = claimed["run_id"]
            cur.execute(
                "select metadata->>'claim_mode',metadata->>'world_claim_revalidated' from public.brian_evolution_engineering_runs where run_id=%s",
                (run_id,),
            )
            assert cur.fetchone() == ("AUTONOMOUS", "true")
            block_run(cur, run_id)
        conn.commit()


def test_malformed_priority_does_not_abort_manual_claim():
    request_id = f"priority-{uuid.uuid4().hex}"
    hypothesis_id = f"priority-hyp-{uuid.uuid4().hex}"
    with connect() as conn:
        with conn.cursor() as cur:
            reset_control(cur)
            cur.execute(
                """
                insert into public.brian_evolution_codegen_requests(
                  request_id,candidate_id,hypothesis_id,requested_at,parent_commit,branch_name,changed_paths,
                  objective,constraints,success_criteria,evidence_refs,external_generator_required,
                  required_human_review,metadata,evidence_class,shadow_only,live_execution,autonomous_apply_allowed,created_at
                ) values (%s,%s,%s,now(),%s,%s,%s,%s,%s,%s,%s,true,true,%s,'POINT_IN_TIME',true,false,false,now())
                """,
                (
                    request_id,
                    f"candidate-{request_id}",
                    hypothesis_id,
                    "6" * 40,
                    f"evolution-candidate/{request_id}",
                    [f"supabase/functions/_shared/evolution_candidates/{request_id}.ts"],
                    "bounded malformed priority fixture",
                    ["shadow only"],
                    ["claim succeeds"],
                    ["fixture:evidence"],
                    json.dumps({"priority": "high", "world_engineering": False}),
                ),
            )
            claimed = manual_claim(cur, request_id)
            assert claimed is not None
            block_run(cur, claimed["run_id"])
        conn.commit()
