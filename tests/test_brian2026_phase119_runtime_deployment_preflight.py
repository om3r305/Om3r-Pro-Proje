from __future__ import annotations

import pytest

from brian2026.phase118_runtime_rollout_manifest import (
    RUNTIME_ROLLOUT_REQUIREMENTS,
)
from brian2026.phase119_runtime_deployment_preflight import (
    RUNTIME_MIGRATION_REQUIREMENTS,
    RuntimeMigrationRequirement,
    assert_deployment_probe_is_read_only,
    build_read_only_deployment_probe_sql,
    evaluate_runtime_deployment,
)


def _relations():
    return tuple(
        row.name
        for row in RUNTIME_ROLLOUT_REQUIREMENTS
        if row.kind == "RELATION"
    )


def _functions():
    return tuple(
        row.name
        for row in RUNTIME_ROLLOUT_REQUIREMENTS
        if row.kind == "FUNCTION"
    )


def _versions():
    return tuple(row.version for row in RUNTIME_MIGRATION_REQUIREMENTS)


def test_runtime_migration_manifest_is_ordered_and_covers_phase70_73_to_87() -> None:
    phases = tuple(row.phase for row in RUNTIME_MIGRATION_REQUIREMENTS)
    assert phases == (70, *range(73, 88))
    assert list(phases) == sorted(phases)
    assert len({row.version for row in RUNTIME_MIGRATION_REQUIREMENTS}) == len(
        RUNTIME_MIGRATION_REQUIREMENTS
    )
    assert all(
        row.source_path.startswith("supabase/migrations/")
        for row in RUNTIME_MIGRATION_REQUIREMENTS
    )


def test_empty_target_is_safe_clean_install_not_schedule_ready() -> None:
    report = evaluate_runtime_deployment(
        relations=(),
        functions=(),
        applied_migration_versions=(),
        target_label="runtime-empty",
    )

    assert report.state == "SAFE_CLEAN_INSTALL"
    assert report.safe_to_apply_migrations is True
    assert report.safe_to_schedule_phase117 is False
    assert report.present_capabilities == ()
    assert set(report.missing_capabilities) == {
        row.name for row in RUNTIME_ROLLOUT_REQUIREMENTS
    }
    assert report.applied_migration_versions == ()
    assert report.missing_migration_versions == tuple(sorted(_versions()))
    assert len(report.report_id) == 64


def test_fully_deployed_target_is_schedule_ready_and_not_reapply_ready() -> None:
    report = evaluate_runtime_deployment(
        relations=_relations(),
        functions=_functions(),
        applied_migration_versions=_versions(),
        target_label="runtime-complete",
    )

    assert report.state == "ALREADY_DEPLOYED"
    assert report.safe_to_apply_migrations is False
    assert report.safe_to_schedule_phase117 is True
    assert report.missing_capabilities == ()
    assert report.present_capabilities == tuple(
        sorted({
            row.name for row in RUNTIME_ROLLOUT_REQUIREMENTS
        })
    )
    assert report.missing_migration_versions == ()


@pytest.mark.parametrize(
    "relations,functions,versions",
    [
        ((_relations()[0],), (), ()),
        ((), (), (_versions()[0],)),
        (_relations(), _functions(), ()),
        ((), (), _versions()),
        (_relations()[:-1], _functions(), _versions()),
    ],
)
def test_any_partial_or_ledger_drift_state_blocks_apply_and_schedule(
    relations,
    functions,
    versions,
) -> None:
    report = evaluate_runtime_deployment(
        relations=relations,
        functions=functions,
        applied_migration_versions=versions,
        target_label="runtime-partial",
    )

    assert report.state == "BLOCKED_PARTIAL"
    assert report.safe_to_apply_migrations is False
    assert report.safe_to_schedule_phase117 is False


def test_unrelated_database_objects_and_migrations_do_not_affect_preflight() -> None:
    report = evaluate_runtime_deployment(
        relations=("public.unrelated_table",),
        functions=("public.unrelated_function(text)",),
        applied_migration_versions=("199901010000",),
        target_label="runtime-clean-with-unrelated-state",
    )

    assert report.state == "SAFE_CLEAN_INSTALL"
    assert report.safe_to_apply_migrations is True


def test_generated_probe_is_read_only_and_checks_capabilities_plus_ledger() -> None:
    sql = build_read_only_deployment_probe_sql()
    assert_deployment_probe_is_read_only(sql)

    assert "brian_shadow_runtime_heads" in sql
    assert "brian_read_shadow_recovery_admission" in sql
    assert "supabase_migrations.schema_migrations" in sql
    for version in _versions():
        assert version in sql


@pytest.mark.parametrize(
    "sql",
    [
        "create table bad(x int)",
        "with x as (select 1) delete from bad",
        "with x as (select 1) alter table bad add column y int",
        "with x as (select 1) grant all on bad to public",
        "with x as (select 1) merge into bad using src on true when matched then update set x=1",
    ],
)
def test_probe_guard_rejects_mutation_ddl_and_merge(sql) -> None:
    with pytest.raises(ValueError):
        assert_deployment_probe_is_read_only(sql)


def test_migration_requirement_rejects_invalid_version_or_source() -> None:
    with pytest.raises(ValueError, match="12-14 digit"):
        RuntimeMigrationRequirement(
            version="abc",
            name="brian_phase79_x",
            source_path="supabase/migrations/x.sql",
            phase=79,
        )

    with pytest.raises(ValueError, match="supabase/migrations"):
        RuntimeMigrationRequirement(
            version="20260924110000",
            name="brian_phase79_atomic_execution_start",
            source_path="brian2026/sql/phase79_atomic_execution_start.sql",
            phase=79,
        )


def test_report_identity_changes_with_target_or_state() -> None:
    clean_a = evaluate_runtime_deployment(
        relations=(),
        functions=(),
        applied_migration_versions=(),
        target_label="runtime-a",
    )
    clean_b = evaluate_runtime_deployment(
        relations=(),
        functions=(),
        applied_migration_versions=(),
        target_label="runtime-b",
    )
    deployed = evaluate_runtime_deployment(
        relations=_relations(),
        functions=_functions(),
        applied_migration_versions=_versions(),
        target_label="runtime-a",
    )

    assert clean_a.report_id != clean_b.report_id
    assert clean_a.report_id != deployed.report_id

def test_migration_name_can_prove_lineage_when_deployer_generates_new_version() -> None:
    names = tuple(row.name for row in RUNTIME_MIGRATION_REQUIREMENTS)
    report = evaluate_runtime_deployment(
        relations=_relations(),
        functions=_functions(),
        applied_migration_versions=(),
        applied_migration_names=names,
        target_label="runtime-name-based-deploy",
    )

    assert report.state == "ALREADY_DEPLOYED"
    assert report.safe_to_schedule_phase117 is True
    assert report.missing_migration_names == ()
    assert report.missing_migration_versions == ()
    assert report.applied_migration_names == tuple(sorted(names))


def test_partial_migration_name_set_is_blocked() -> None:
    names = tuple(row.name for row in RUNTIME_MIGRATION_REQUIREMENTS)
    report = evaluate_runtime_deployment(
        relations=_relations(),
        functions=_functions(),
        applied_migration_versions=(),
        applied_migration_names=names[:-1],
        target_label="runtime-name-partial",
    )

    assert report.state == "BLOCKED_PARTIAL"
    assert report.safe_to_apply_migrations is False
    assert report.safe_to_schedule_phase117 is False
    assert report.missing_migration_names == (names[-1],)


def test_probe_accepts_matching_migration_name_or_repo_version() -> None:
    sql = build_read_only_deployment_probe_sql()

    assert "required_migrations(version,name)" in sql
    assert "m.version=v.version or m.name=v.name" in sql
    assert RUNTIME_MIGRATION_REQUIREMENTS[0].name in sql

