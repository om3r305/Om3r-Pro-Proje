from __future__ import annotations

import pytest

from brian2026.phase118_runtime_rollout_manifest import (
    RUNTIME_ROLLOUT_REQUIREMENTS,
    RuntimeCapabilityRequirement,
    assert_probe_is_read_only,
    build_read_only_postgres_probe_sql,
    evaluate_runtime_rollout,
)


def _all(kind: str) -> tuple[str, ...]:
    return tuple(
        row.name
        for row in RUNTIME_ROLLOUT_REQUIREMENTS
        if row.kind == kind
    )


def test_manifest_covers_phase70_and_every_sql_phase73_to_87() -> None:
    phases = {row.phase for row in RUNTIME_ROLLOUT_REQUIREMENTS}
    assert phases == {70, *range(73, 88)}
    assert any(
        row.phase == 70 and row.name == "brian_shadow_runtime_heads"
        for row in RUNTIME_ROLLOUT_REQUIREMENTS
    )
    assert any(
        row.phase == 73 and row.name == "brian_operational_risk_heads"
        for row in RUNTIME_ROLLOUT_REQUIREMENTS
    )
    assert any(
        row.phase == 86
        and row.name == "brian_read_shadow_recovery_admission"
        for row in RUNTIME_ROLLOUT_REQUIREMENTS
    )
    assert any(
        row.phase == 87
        and row.name == "brian_read_next_shadow_recovery_work"
        for row in RUNTIME_ROLLOUT_REQUIREMENTS
    )


def test_current_branch_lineage_blocks_scheduler_even_if_capabilities_exist() -> None:
    report = evaluate_runtime_rollout(
        relations=_all("RELATION"),
        functions=_all("FUNCTION"),
    )

    assert report.capabilities_complete is True
    assert report.official_migration_lineage_complete is False
    assert report.safe_to_schedule_phase117 is False
    assert report.missing_capabilities == ()
    assert len(report.draft_sources) == 9
    assert all(
        path.startswith("brian2026/sql/phase")
        for path in report.draft_sources
    )
    assert len(report.report_id) == 64
    assert report.read_only is True
    assert report.shadow_only is True
    assert report.live_execution is False


def test_empty_database_reports_every_required_capability_missing() -> None:
    report = evaluate_runtime_rollout(
        relations=(),
        functions=(),
    )

    assert report.capabilities_complete is False
    assert report.safe_to_schedule_phase117 is False
    assert set(report.missing_capabilities) == {
        row.name for row in RUNTIME_ROLLOUT_REQUIREMENTS
    }


def test_partial_database_identifies_only_missing_names() -> None:
    required_relations = _all("RELATION")
    required_functions = _all("FUNCTION")
    report = evaluate_runtime_rollout(
        relations=required_relations[:-1],
        functions=required_functions[:-1],
    )

    assert report.capabilities_complete is False
    assert set(report.missing_capabilities) == {
        required_relations[-1],
        required_functions[-1],
    }


def test_public_prefix_and_function_signatures_are_normalized() -> None:
    relation = _all("RELATION")[0]
    function = _all("FUNCTION")[0]
    requirement = (
        RuntimeCapabilityRequirement(
            phase=70,
            kind="RELATION",
            name=relation,
            lineage="MIGRATION",
            source_path=(
                "supabase/migrations/"
                "202609230720_brian_phase70_durable_runtime_store.sql"
            ),
        ),
        RuntimeCapabilityRequirement(
            phase=70,
            kind="FUNCTION",
            name=function,
            lineage="MIGRATION",
            source_path=(
                "supabase/migrations/"
                "202609230720_brian_phase70_durable_runtime_store.sql"
            ),
        ),
    )
    report = evaluate_runtime_rollout(
        relations=(f"public.{relation}",),
        functions=(f"public.{function}(text,uuid)",),
        requirements=requirement,
    )
    assert report.capabilities_complete is True
    assert report.official_migration_lineage_complete is True
    assert report.safe_to_schedule_phase117 is True


def test_generated_postgres_probe_is_select_only_and_contains_key_capabilities() -> None:
    sql = build_read_only_postgres_probe_sql()
    assert_probe_is_read_only(sql)

    assert "brian_shadow_runtime_heads" in sql
    assert "brian_operational_risk_heads" in sql
    assert "brian_read_shadow_recovery_admission" in sql
    assert "brian_read_next_shadow_recovery_work" in sql
    assert "to_regclass" in sql
    assert "pg_catalog.pg_proc" in sql


@pytest.mark.parametrize(
    "sql",
    [
        "create table bad(x int)",
        "with x as (select 1) delete from bad",
        "with x as (select 1) update bad set x=1",
        "with x as (select 1) grant all on bad to public",
    ],
)
def test_read_only_probe_guard_rejects_mutation_or_ddl(sql) -> None:
    with pytest.raises(ValueError):
        assert_probe_is_read_only(sql)


def test_requirement_rejects_draft_sql_misclassified_as_migration() -> None:
    with pytest.raises(ValueError, match="supabase/migrations"):
        RuntimeCapabilityRequirement(
            phase=79,
            kind="FUNCTION",
            name="brian_mark_shadow_execution_started",
            lineage="MIGRATION",
            source_path="brian2026/sql/phase79_atomic_execution_start.sql",
        )
