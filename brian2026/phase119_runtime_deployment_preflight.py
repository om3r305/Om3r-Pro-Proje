from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Literal

from .phase118_runtime_rollout_manifest import (
    RUNTIME_ROLLOUT_REQUIREMENTS,
    RuntimeCapabilityRequirement,
)

PHASE119_SCHEMA_VERSION = "brian.phase119-runtime-deployment-preflight.v1"

DeploymentState = Literal[
    "SAFE_CLEAN_INSTALL",
    "ALREADY_DEPLOYED",
    "BLOCKED_PARTIAL",
]


@dataclass(frozen=True, slots=True)
class RuntimeMigrationRequirement:
    version: str
    name: str
    source_path: str
    phase: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"\d{12,14}", self.version):
            raise ValueError("migration version must be a 12-14 digit prefix")
        if not re.fullmatch(r"[a-z0-9_]+", self.name):
            raise ValueError("migration name must be snake_case")
        if not self.source_path.startswith("supabase/migrations/"):
            raise ValueError("runtime migration must live under supabase/migrations")
        if self.phase < 70 or self.phase > 87:
            raise ValueError("runtime migration phase must be in [70,87]")


def _migration_requirement(path: str, phase: int) -> RuntimeMigrationRequirement:
    filename = PurePosixPath(path).name
    match = re.match(r"^(\d{12,14})_", filename)
    if match is None:
        raise ValueError(f"migration path has no timestamp prefix: {path}")
    name = filename[match.end():]
    if name.endswith(".sql"):
        name = name[:-4]
    return RuntimeMigrationRequirement(
        version=match.group(1),
        name=name,
        source_path=path,
        phase=phase,
    )


def _required_migrations(
    requirements: Sequence[RuntimeCapabilityRequirement] =
        RUNTIME_ROLLOUT_REQUIREMENTS,
) -> tuple[RuntimeMigrationRequirement, ...]:
    by_path: dict[str, int] = {}
    for row in requirements:
        if row.lineage != "MIGRATION":
            raise ValueError(
                "Phase119 requires Phase118 runtime lineage to be fully promoted"
            )
        prior = by_path.get(row.source_path)
        if prior is not None and prior != row.phase:
            raise ValueError("migration path cannot serve multiple phases")
        by_path[row.source_path] = row.phase
    migrations = tuple(sorted(
        (
            _migration_requirement(path, phase)
            for path, phase in by_path.items()
        ),
        key=lambda row: (row.version, row.phase, row.source_path),
    ))
    versions = [row.version for row in migrations]
    names = [row.name for row in migrations]
    if len(versions) != len(set(versions)):
        raise ValueError("runtime migration versions must be unique")
    if len(names) != len(set(names)):
        raise ValueError("runtime migration names must be unique")
    phases = [row.phase for row in migrations]
    if phases != sorted(phases):
        raise ValueError(
            "runtime migration timestamps must preserve phase dependency order"
        )
    return migrations


RUNTIME_MIGRATION_REQUIREMENTS = _required_migrations()


@dataclass(frozen=True, slots=True)
class RuntimeDeploymentPreflightReport:
    state: DeploymentState
    safe_to_apply_migrations: bool
    safe_to_schedule_phase117: bool
    missing_capabilities: tuple[str, ...]
    present_capabilities: tuple[str, ...]
    missing_migration_versions: tuple[str, ...]
    applied_migration_versions: tuple[str, ...]
    required_migration_versions: tuple[str, ...]
    missing_migration_names: tuple[str, ...]
    applied_migration_names: tuple[str, ...]
    required_migration_names: tuple[str, ...]
    target_label: str
    report_id: str = field(init=False)
    schema_version: str = PHASE119_SCHEMA_VERSION
    read_only: bool = True
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if self.state not in (
            "SAFE_CLEAN_INSTALL",
            "ALREADY_DEPLOYED",
            "BLOCKED_PARTIAL",
        ):
            raise ValueError("invalid Phase119 deployment state")
        if not self.target_label.strip():
            raise ValueError("target_label is required")
        if tuple(sorted(set(self.missing_capabilities))) != self.missing_capabilities:
            raise ValueError("missing_capabilities must be unique/sorted")
        if tuple(sorted(set(self.present_capabilities))) != self.present_capabilities:
            raise ValueError("present_capabilities must be unique/sorted")
        if tuple(sorted(set(self.missing_migration_versions))) != (
            self.missing_migration_versions
        ):
            raise ValueError("missing migration versions must be unique/sorted")
        if tuple(sorted(set(self.applied_migration_versions))) != (
            self.applied_migration_versions
        ):
            raise ValueError("applied migration versions must be unique/sorted")
        if tuple(sorted(set(self.required_migration_versions))) != (
            self.required_migration_versions
        ):
            raise ValueError("required migration versions must be unique/sorted")
        for label, values in (
            ("missing migration names", self.missing_migration_names),
            ("applied migration names", self.applied_migration_names),
            ("required migration names", self.required_migration_names),
        ):
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{label} must be unique/sorted")

        all_caps_missing = not self.present_capabilities
        no_migrations_applied = (
            not self.applied_migration_versions
            and not self.applied_migration_names
        )
        all_caps_present = not self.missing_capabilities
        all_migrations_applied = (
            not self.missing_migration_versions
            and not self.missing_migration_names
        )
        expected_state: DeploymentState
        if all_caps_missing and no_migrations_applied:
            expected_state = "SAFE_CLEAN_INSTALL"
        elif all_caps_present and all_migrations_applied:
            expected_state = "ALREADY_DEPLOYED"
        else:
            expected_state = "BLOCKED_PARTIAL"

        if self.state != expected_state:
            raise ValueError("state disagrees with capability/migration snapshot")
        if self.safe_to_apply_migrations != (
            expected_state == "SAFE_CLEAN_INSTALL"
        ):
            raise ValueError("safe_to_apply_migrations disagrees with state")
        if self.safe_to_schedule_phase117 != (
            expected_state == "ALREADY_DEPLOYED"
        ):
            raise ValueError("safe_to_schedule_phase117 disagrees with state")
        if not self.read_only or not self.shadow_only or self.live_execution:
            raise ValueError("Phase119 report must remain read-only shadow-only")

        object.__setattr__(
            self,
            "report_id",
            hashlib.sha256(
                json.dumps(
                    self.identity_payload(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest(),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "state": self.state,
            "safe_to_apply_migrations": self.safe_to_apply_migrations,
            "safe_to_schedule_phase117": self.safe_to_schedule_phase117,
            "missing_capabilities": list(self.missing_capabilities),
            "present_capabilities": list(self.present_capabilities),
            "missing_migration_versions": list(
                self.missing_migration_versions
            ),
            "applied_migration_versions": list(
                self.applied_migration_versions
            ),
            "required_migration_versions": list(
                self.required_migration_versions
            ),
            "missing_migration_names": list(self.missing_migration_names),
            "applied_migration_names": list(self.applied_migration_names),
            "required_migration_names": list(self.required_migration_names),
            "target_label": self.target_label,
            "schema_version": self.schema_version,
            "read_only": self.read_only,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
        }


def evaluate_runtime_deployment(
    *,
    relations: Iterable[str],
    functions: Iterable[str],
    applied_migration_versions: Iterable[str],
    target_label: str,
    applied_migration_names: Iterable[str] = (),
    requirements: Sequence[RuntimeCapabilityRequirement] =
        RUNTIME_ROLLOUT_REQUIREMENTS,
    migrations: Sequence[RuntimeMigrationRequirement] =
        RUNTIME_MIGRATION_REQUIREMENTS,
) -> RuntimeDeploymentPreflightReport:
    relation_names = {
        str(value).strip().removeprefix("public.")
        for value in relations
        if str(value).strip()
    }
    function_names = {
        str(value).strip().removeprefix("public.").split("(", 1)[0]
        for value in functions
        if str(value).strip()
    }
    required_capabilities = {
        row.name for row in requirements
    }
    present = {
        row.name
        for row in requirements
        if (
            row.name in relation_names
            if row.kind == "RELATION"
            else row.name in function_names
        )
    }
    missing = required_capabilities - present

    required_versions = {row.version for row in migrations}
    required_names = {row.name for row in migrations}
    supplied_versions = {
        str(value).strip()
        for value in applied_migration_versions
        if str(value).strip()
    }
    supplied_names = {
        str(value).strip()
        for value in applied_migration_names
        if str(value).strip()
    }
    applied_rows = {
        row
        for row in migrations
        if row.version in supplied_versions or row.name in supplied_names
    }
    applied_versions = {row.version for row in applied_rows}
    applied_names = {row.name for row in applied_rows}
    missing_rows = set(migrations) - applied_rows
    missing_versions = {row.version for row in missing_rows}
    missing_names = {row.name for row in missing_rows}

    if not present and not applied_rows:
        state: DeploymentState = "SAFE_CLEAN_INSTALL"
    elif not missing and not missing_rows:
        state = "ALREADY_DEPLOYED"
    else:
        state = "BLOCKED_PARTIAL"

    return RuntimeDeploymentPreflightReport(
        state=state,
        safe_to_apply_migrations=(state == "SAFE_CLEAN_INSTALL"),
        safe_to_schedule_phase117=(state == "ALREADY_DEPLOYED"),
        missing_capabilities=tuple(sorted(missing)),
        present_capabilities=tuple(sorted(present)),
        missing_migration_versions=tuple(sorted(missing_versions)),
        applied_migration_versions=tuple(sorted(applied_versions)),
        required_migration_versions=tuple(sorted(required_versions)),
        missing_migration_names=tuple(sorted(missing_names)),
        applied_migration_names=tuple(sorted(applied_names)),
        required_migration_names=tuple(sorted(required_names)),
        target_label=str(target_label).strip(),
    )


def build_read_only_deployment_probe_sql(
    requirements: Sequence[RuntimeCapabilityRequirement] =
        RUNTIME_ROLLOUT_REQUIREMENTS,
    migrations: Sequence[RuntimeMigrationRequirement] =
        RUNTIME_MIGRATION_REQUIREMENTS,
) -> str:
    relations = tuple(sorted({
        row.name for row in requirements if row.kind == "RELATION"
    }))
    functions = tuple(sorted({
        row.name for row in requirements if row.kind == "FUNCTION"
    }))
    migration_rows = tuple(sorted(
        ((row.version, row.name) for row in migrations),
        key=lambda row: (row[0], row[1]),
    ))

    rel_values = ",".join(f"('{name}')" for name in relations) or "('')"
    fn_values = ",".join(f"('{name}')" for name in functions) or "('')"
    migration_values = ",".join(
        f"('{version}','{name}')"
        for version, name in migration_rows
    ) or "('','')"

    return (
        "with required_relations(name) as (values "
        + rel_values
        + "), required_functions(name) as (values "
        + fn_values
        + "), required_migrations(version,name) as (values "
        + migration_values
        + "), capability_rows as ("
        + "select 'RELATION'::text as kind,r.name,"
        + "(to_regclass('public.'||r.name) is not null) as present "
        + "from required_relations r union all "
        + "select 'FUNCTION'::text as kind,f.name,exists("
        + "select 1 from pg_catalog.pg_proc p "
        + "join pg_catalog.pg_namespace n on n.oid=p.pronamespace "
        + "where n.nspname='public' and p.proname=f.name"
        + ") as present from required_functions f), migration_rows as ("
        + "select 'MIGRATION'::text as kind,"
        + "(v.name || '@' || v.version) as name,exists("
        + "select 1 from supabase_migrations.schema_migrations m "
        + "where m.version=v.version or m.name=v.name"
        + ") as present from required_migrations v) "
        + "select kind,name,present from capability_rows "
        + "union all select kind,name,present from migration_rows "
        + "order by kind,name;"
    )


def assert_deployment_probe_is_read_only(sql: str) -> None:
    normalized = re.sub(r"\s+", " ", str(sql).strip()).lower()
    if not normalized.startswith("with "):
        raise ValueError("Phase119 probe must start with a CTE/SELECT")
    padded = " " + normalized + " "
    forbidden = (
        " insert ",
        " update ",
        " delete ",
        " truncate ",
        " alter ",
        " drop ",
        " create ",
        " grant ",
        " revoke ",
        " call ",
        " copy ",
        " merge ",
    )
    if any(token in padded for token in forbidden):
        raise ValueError("Phase119 probe contains mutation/DDL token")
