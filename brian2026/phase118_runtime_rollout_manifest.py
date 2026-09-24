from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

PHASE118_SCHEMA_VERSION = "brian.phase118-runtime-rollout-manifest.v1"

CapabilityKind = Literal["RELATION", "FUNCTION"]
LineageState = Literal["MIGRATION", "DRAFT_SQL"]


@dataclass(frozen=True, slots=True)
class RuntimeCapabilityRequirement:
    phase: int
    kind: CapabilityKind
    name: str
    lineage: LineageState
    source_path: str

    def __post_init__(self) -> None:
        if self.phase < 70 or self.phase > 87:
            raise ValueError("runtime rollout capability phase must be in [70,87]")
        if self.kind not in ("RELATION", "FUNCTION"):
            raise ValueError("invalid capability kind")
        if not re.fullmatch(r"brian_[a-z0-9_]+", self.name):
            raise ValueError("capability name must be a Brian SQL identifier")
        if self.lineage == "MIGRATION":
            if not self.source_path.startswith("supabase/migrations/"):
                raise ValueError("migration capability must reference supabase/migrations")
        elif self.lineage == "DRAFT_SQL":
            if not self.source_path.startswith("brian2026/sql/"):
                raise ValueError("draft capability must reference brian2026/sql")
        else:
            raise ValueError("invalid lineage state")


def _requirements(
    phase: int,
    lineage: LineageState,
    source_path: str,
    *,
    relations: Sequence[str] = (),
    functions: Sequence[str] = (),
) -> tuple[RuntimeCapabilityRequirement, ...]:
    return tuple(
        RuntimeCapabilityRequirement(
            phase=phase,
            kind="RELATION",
            name=name,
            lineage=lineage,
            source_path=source_path,
        )
        for name in relations
    ) + tuple(
        RuntimeCapabilityRequirement(
            phase=phase,
            kind="FUNCTION",
            name=name,
            lineage=lineage,
            source_path=source_path,
        )
        for name in functions
    )


RUNTIME_ROLLOUT_REQUIREMENTS: tuple[RuntimeCapabilityRequirement, ...] = (
    *_requirements(
        70,
        "MIGRATION",
        "supabase/migrations/202609230720_brian_phase70_durable_runtime_store.sql",
        relations=(
            "brian_shadow_runtime_heads",
            "brian_shadow_runtime_checkpoints",
            "brian_shadow_runtime_cycles",
            "brian_shadow_runtime_journal_entries",
            "brian_shadow_runtime_events",
        ),
        functions=(
            "brian_acquire_shadow_runtime_lease",
            "brian_renew_shadow_runtime_lease",
            "brian_release_shadow_runtime_lease",
            "brian_commit_shadow_runtime_checkpoint",
            "brian_read_shadow_runtime_checkpoint",
        ),
    ),
    *_requirements(
        73,
        "MIGRATION",
        "supabase/migrations/202609230745_brian_phase73_operational_risk_store.sql",
        relations=(
            "brian_operational_risk_heads",
            "brian_operational_risk_snapshots",
            "brian_operational_risk_entries",
            "brian_operational_risk_events",
        ),
        functions=(
            "brian_commit_operational_risk_ledger",
            "brian_read_operational_risk_ledger",
        ),
    ),
    *_requirements(
        74,
        "MIGRATION",
        "supabase/migrations/202609230815_brian_phase74_governed_cycle_binding.sql",
        relations=(
            "brian_governed_cycle_bindings",
            "brian_governed_cycle_binding_events",
        ),
        functions=(
            "brian_bind_governed_shadow_cycle",
            "brian_read_governed_cycle_binding",
        ),
    ),
    *_requirements(
        75,
        "MIGRATION",
        "supabase/migrations/202609230845_brian_phase75_atomic_governed_writeahead.sql",
        relations=(
            "brian_governed_cycle_authorizations",
            "brian_governed_cycle_authorization_events",
        ),
        functions=(
            "brian_authorize_and_persist_governed_cycle",
            "brian_read_governed_cycle_authorization",
        ),
    ),
    *_requirements(
        76,
        "MIGRATION",
        "supabase/migrations/202609230915_brian_phase76_shadow_execution_outbox.sql",
        relations=(
            "brian_shadow_execution_dispatches",
            "brian_shadow_execution_dispatch_events",
        ),
        functions=(
            "brian_submit_shadow_execution_dispatch",
            "brian_read_shadow_execution_dispatch",
        ),
    ),
    *_requirements(
        77,
        "MIGRATION",
        "supabase/migrations/202609230945_brian_phase77_execution_claim_lifecycle.sql",
        relations=(
            "brian_shadow_execution_claims",
            "brian_shadow_execution_claim_events",
        ),
        functions=(
            "brian_claim_shadow_execution_dispatch",
            "brian_renew_shadow_execution_claim",
            "brian_complete_shadow_execution_claim",
            "brian_read_shadow_execution_claim",
        ),
    ),
    *_requirements(
        78,
        "MIGRATION",
        "supabase/migrations/202609231015_brian_phase78_execution_kill_switch.sql",
        relations=(
            "brian_shadow_execution_cancel_requests",
            "brian_shadow_execution_kill_events",
        ),
        functions=(
            "brian_check_shadow_execution_kill_switch",
            "brian_read_shadow_execution_cancel_requests",
        ),
    ),
    *_requirements(
        79,
        "MIGRATION",
        "supabase/migrations/20260924110000_brian_phase79_atomic_execution_start.sql",
        relations=(
            "brian_shadow_execution_starts",
            "brian_shadow_execution_start_events",
        ),
        functions=(
            "brian_mark_shadow_execution_started",
            "brian_read_shadow_execution_start",
        ),
    ),
    *_requirements(
        80,
        "MIGRATION",
        "supabase/migrations/20260924110500_brian_phase80_claim_fenced_checkpoint_commit.sql",
        relations=("brian_shadow_claim_commit_events",),
        functions=("brian_commit_claimed_shadow_runtime_checkpoint",),
    ),
    *_requirements(
        81,
        "MIGRATION",
        "supabase/migrations/20260924111000_brian_phase81_cancel_recovery_directive.sql",
        relations=(
            "brian_shadow_cancel_recovery_directives",
            "brian_shadow_cancel_recovery_events",
        ),
        functions=(
            "brian_prepare_shadow_cancel_recovery",
            "brian_read_shadow_cancel_recovery",
        ),
    ),
    *_requirements(
        82,
        "MIGRATION",
        "supabase/migrations/20260924111500_brian_phase82_recovery_claim_fencing.sql",
        relations=(
            "brian_shadow_recovery_claims",
            "brian_shadow_recovery_claim_events",
        ),
        functions=(
            "brian_claim_shadow_cancel_recovery",
            "brian_renew_shadow_cancel_recovery_claim",
        ),
    ),
    *_requirements(
        83,
        "MIGRATION",
        "supabase/migrations/20260924112000_brian_phase83_atomic_recovery_start.sql",
        relations=(
            "brian_shadow_recovery_starts",
            "brian_shadow_recovery_start_events",
        ),
        functions=(
            "brian_mark_shadow_recovery_started",
            "brian_read_shadow_recovery_start",
        ),
    ),
    *_requirements(
        84,
        "MIGRATION",
        "supabase/migrations/20260924112500_brian_phase84_recovery_execution_checkpoint.sql",
        relations=("brian_shadow_recovery_commit_events",),
        functions=("brian_commit_shadow_recovery_checkpoint",),
    ),
    *_requirements(
        85,
        "MIGRATION",
        "supabase/migrations/20260924113000_brian_phase85_recovery_completion_audit.sql",
        relations=(
            "brian_shadow_recovery_completion_certificates",
            "brian_shadow_recovery_completion_events",
        ),
        functions=(
            "brian_certify_shadow_recovery_completion",
            "brian_read_shadow_recovery_completion",
        ),
    ),
    *_requirements(
        86,
        "MIGRATION",
        "supabase/migrations/20260924113500_brian_phase86_recovery_admission_interlock.sql",
        relations=("brian_shadow_recovery_admission_events",),
        functions=("brian_read_shadow_recovery_admission",),
    ),
    *_requirements(
        87,
        "MIGRATION",
        "supabase/migrations/20260924114000_brian_phase87_recovery_restart_resume.sql",
        functions=("brian_read_next_shadow_recovery_work",),
    ),
)


@dataclass(frozen=True, slots=True)
class RuntimeCapabilityResult:
    phase: int
    kind: CapabilityKind
    name: str
    lineage: LineageState
    source_path: str
    present: bool


@dataclass(frozen=True, slots=True)
class RuntimeRolloutReport:
    results: tuple[RuntimeCapabilityResult, ...]
    capabilities_complete: bool
    official_migration_lineage_complete: bool
    safe_to_schedule_phase117: bool
    missing_capabilities: tuple[str, ...]
    draft_sources: tuple[str, ...]
    report_id: str = field(init=False)
    schema_version: str = PHASE118_SCHEMA_VERSION
    read_only: bool = True
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        expected_caps = all(row.present for row in self.results)
        expected_drafts = tuple(sorted({
            row.source_path
            for row in self.results
            if row.lineage == "DRAFT_SQL"
        }))
        expected_lineage = not expected_drafts
        expected_missing = tuple(sorted(
            row.name for row in self.results if not row.present
        ))
        if self.capabilities_complete != expected_caps:
            raise ValueError("capabilities_complete disagrees with results")
        if self.official_migration_lineage_complete != expected_lineage:
            raise ValueError(
                "official_migration_lineage_complete disagrees with draft sources"
            )
        if self.missing_capabilities != expected_missing:
            raise ValueError("missing_capabilities disagrees with results")
        if self.draft_sources != expected_drafts:
            raise ValueError("draft_sources disagrees with results")
        if self.safe_to_schedule_phase117 != (
            expected_caps and expected_lineage
        ):
            raise ValueError("safe_to_schedule_phase117 disagrees with rollout state")
        if not self.read_only or not self.shadow_only or self.live_execution:
            raise ValueError("Phase118 report must remain read-only shadow-only")
        payload = self.identity_payload()
        object.__setattr__(
            self,
            "report_id",
            hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest(),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "results": [asdict(row) for row in self.results],
            "capabilities_complete": self.capabilities_complete,
            "official_migration_lineage_complete":
                self.official_migration_lineage_complete,
            "safe_to_schedule_phase117": self.safe_to_schedule_phase117,
            "missing_capabilities": list(self.missing_capabilities),
            "draft_sources": list(self.draft_sources),
            "read_only": self.read_only,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
        }


def evaluate_runtime_rollout(
    *,
    relations: Iterable[str],
    functions: Iterable[str],
    requirements: Sequence[RuntimeCapabilityRequirement] =
        RUNTIME_ROLLOUT_REQUIREMENTS,
) -> RuntimeRolloutReport:
    existing_relations = {
        str(value).strip().removeprefix("public.")
        for value in relations
        if str(value).strip()
    }
    existing_functions = {
        str(value).strip().removeprefix("public.").split("(", 1)[0]
        for value in functions
        if str(value).strip()
    }
    results = tuple(
        RuntimeCapabilityResult(
            phase=row.phase,
            kind=row.kind,
            name=row.name,
            lineage=row.lineage,
            source_path=row.source_path,
            present=(
                row.name in existing_relations
                if row.kind == "RELATION"
                else row.name in existing_functions
            ),
        )
        for row in requirements
    )
    missing = tuple(sorted(
        row.name for row in results if not row.present
    ))
    drafts = tuple(sorted({
        row.source_path
        for row in results
        if row.lineage == "DRAFT_SQL"
    }))
    complete = not missing
    lineage_complete = not drafts
    return RuntimeRolloutReport(
        results=results,
        capabilities_complete=complete,
        official_migration_lineage_complete=lineage_complete,
        safe_to_schedule_phase117=complete and lineage_complete,
        missing_capabilities=missing,
        draft_sources=drafts,
    )


def build_read_only_postgres_probe_sql(
    requirements: Sequence[RuntimeCapabilityRequirement] =
        RUNTIME_ROLLOUT_REQUIREMENTS,
) -> str:
    """Return a SELECT-only capability probe suitable for a DBA/Supabase console.

    Function matching is by proname on purpose: several rollout functions have
    long typed signatures and Phase118 is checking capability presence, not
    dispatching them. Runtime code still validates exact RPC behavior later.
    """
    relations = tuple(sorted({
        row.name for row in requirements if row.kind == "RELATION"
    }))
    functions = tuple(sorted({
        row.name for row in requirements if row.kind == "FUNCTION"
    }))
    relation_values = ",".join(
        f"('{name}')" for name in relations
    ) or "('')"
    function_values = ",".join(
        f"('{name}')" for name in functions
    ) or "('')"
    return (
        "with required_relations(name) as (values "
        + relation_values
        + "), required_functions(name) as (values "
        + function_values
        + ") "
        + "select 'RELATION'::text as kind, r.name, "
        + "(to_regclass('public.' || r.name) is not null) as present "
        + "from required_relations r "
        + "union all "
        + "select 'FUNCTION'::text as kind, f.name, exists("
        + "select 1 from pg_catalog.pg_proc p "
        + "join pg_catalog.pg_namespace n on n.oid=p.pronamespace "
        + "where n.nspname='public' and p.proname=f.name"
        + ") as present from required_functions f "
        + "order by kind,name;"
    )


def assert_probe_is_read_only(sql: str) -> None:
    normalized = re.sub(r"\s+", " ", str(sql).strip()).lower()
    if not normalized.startswith("with "):
        raise ValueError("Phase118 probe must start with a CTE/SELECT")
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
    )
    padded = " " + normalized + " "
    if any(token in padded for token in forbidden):
        raise ValueError("Phase118 probe contains mutation/DDL token")
