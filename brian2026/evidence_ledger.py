from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any, Literal, Mapping, Sequence
import json
import math

from .portfolio import DEVELOPMENT_CUTOFF

SCHEMA_VERSION = "brian.evidence-ledger.v1"
EvidenceScope = Literal["development", "validation", "locked_test", "final_holdout"]
Decision = Literal["REJECT", "INSUFFICIENT_EVIDENCE", "KEEP_CHALLENGER", "SHADOW_CANDIDATE"]


class EvidenceConflictError(ValueError):
    pass


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("evidence payload cannot contain NaN or infinity")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported evidence value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    phase: str
    logical_experiment_id: str
    dataset_id: str
    code_commit: str
    scope: EvidenceScope
    max_data_timestamp: float
    metrics: Mapping[str, Any]
    gates: Mapping[str, Any]
    decision: Decision = "INSUFFICIENT_EVIDENCE"
    parent_evidence_ids: tuple[str, ...] = ()
    final_holdout_status: str = "INVALID_CONTAMINATED"
    final_holdout_evaluated: bool = False
    evaluation_allowed: bool = True
    shadow_only: bool = True
    automatic_promotion: bool = False
    schema_version: str = SCHEMA_VERSION
    evidence_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.phase.strip() or not self.logical_experiment_id.strip() or not self.dataset_id.strip():
            raise ValueError("phase, experiment id and dataset id are required")
        if not self.code_commit.strip():
            raise ValueError("code commit is required")
        if self.max_data_timestamp >= DEVELOPMENT_CUTOFF:
            raise ValueError("2026 data is INVALID_CONTAMINATED and forbidden")
        if not self.shadow_only:
            raise ValueError("Brian evidence must remain SHADOW_RESEARCH_ONLY")
        if self.automatic_promotion:
            raise ValueError("automatic model promotion is forbidden")
        if self.scope == "final_holdout":
            raise ValueError("no pristine final holdout is currently available")
        if self.final_holdout_evaluated:
            raise ValueError("the contaminated 2026 final holdout must not be evaluated")
        if self.final_holdout_status != "INVALID_CONTAMINATED":
            raise ValueError("current final holdout status must remain INVALID_CONTAMINATED")
        if not self.evaluation_allowed and self.scope in {"validation", "locked_test"}:
            raise ValueError("locked evaluation cannot be recorded when evaluation_allowed is false")
        if self.decision == "SHADOW_CANDIDATE" and not bool(self.gates.get("all_required_gates_passed", False)):
            raise ValueError("SHADOW_CANDIDATE requires all required scientific gates")
        _canonical(self.metrics)
        _canonical(self.gates)
        for parent in self.parent_evidence_ids:
            if not str(parent).strip():
                raise ValueError("parent evidence ids must be non-empty")
        object.__setattr__(self, "evidence_id", content_hash(self.identity_payload()))

    def identity_payload(self) -> dict[str, Any]:
        """Scientific identity only; deliberately excludes wall-clock/runtime metadata."""
        return {
            "schema_version": self.schema_version,
            "phase": self.phase,
            "logical_experiment_id": self.logical_experiment_id,
            "dataset_id": self.dataset_id,
            "code_commit": self.code_commit,
            "scope": self.scope,
            "max_data_timestamp": float(self.max_data_timestamp),
            "metrics": _canonical(self.metrics),
            "gates": _canonical(self.gates),
            "decision": self.decision,
            "parent_evidence_ids": sorted(self.parent_evidence_ids),
            "final_holdout_status": self.final_holdout_status,
            "final_holdout_evaluated": self.final_holdout_evaluated,
            "evaluation_allowed": self.evaluation_allowed,
            "shadow_only": self.shadow_only,
            "automatic_promotion": self.automatic_promotion,
        }

    def manifest(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["evidence_id"] = self.evidence_id
        return payload


@dataclass(frozen=True, slots=True)
class LedgerReceipt:
    evidence_id: str
    logical_experiment_id: str
    duplicate: bool
    ordinal: int


class EvidenceLedger:
    """In-memory append-only evidence registry used to build immutable run artifacts."""

    def __init__(self, records: Sequence[EvidenceRecord] = ()) -> None:
        self._records: list[EvidenceRecord] = []
        self._by_id: dict[str, EvidenceRecord] = {}
        self._by_experiment: dict[str, str] = {}
        for record in records:
            self.append(record)

    def append(self, record: EvidenceRecord) -> LedgerReceipt:
        existing = self._by_id.get(record.evidence_id)
        if existing is not None:
            return LedgerReceipt(record.evidence_id, record.logical_experiment_id, True, self._records.index(existing))

        previous_id = self._by_experiment.get(record.logical_experiment_id)
        if previous_id is not None and previous_id != record.evidence_id:
            raise EvidenceConflictError(
                "logical experiment id already exists with different scientific evidence; "
                "create a new experiment id instead of rewriting history"
            )

        known_ids = set(self._by_id)
        missing_parents = [parent for parent in record.parent_evidence_ids if parent not in known_ids]
        if missing_parents:
            raise EvidenceConflictError(f"unknown parent evidence ids: {missing_parents}")

        ordinal = len(self._records)
        self._records.append(record)
        self._by_id[record.evidence_id] = record
        self._by_experiment[record.logical_experiment_id] = record.evidence_id
        return LedgerReceipt(record.evidence_id, record.logical_experiment_id, False, ordinal)

    def get(self, evidence_id: str) -> EvidenceRecord:
        try:
            return self._by_id[evidence_id]
        except KeyError as exc:
            raise KeyError(f"unknown evidence id: {evidence_id}") from exc

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._records)

    def manifest(self) -> dict[str, Any]:
        rows = [record.manifest() for record in self._records]
        return {
            "schema_version": SCHEMA_VERSION,
            "append_only": True,
            "record_count": len(rows),
            "records": rows,
            "ledger_hash": content_hash(rows),
            "shadow_only": True,
            "automatic_promotion": False,
        }
