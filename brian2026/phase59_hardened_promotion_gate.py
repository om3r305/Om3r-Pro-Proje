from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal
import math
import re

from .phase49_promotion_gate import PromotionReceipt
from .phase50_execution_reconciliation import ReconciliationBatchReport
from .phase58_advanced_overfit_audit import AdvancedOverfitReport

PHASE59_SCHEMA_VERSION = "brian.phase59-hardened-promotion-gate.v1"
HardenedStatus = Literal[
    "PRELIVE_RESEARCH_BLOCKED",
    "EXECUTION_RECONCILIATION_BLOCKED",
    "FINAL_VALIDATION_REQUIRED",
    "HUMAN_REVIEW_READY",
]
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class FinalValidationReceipt:
    dataset_id: str
    dataset_seal_sha256: str
    code_commit: str
    sealed_at: float
    evaluation_started_at: float
    evaluation_finished_at: float
    evaluation_count: int
    used_for_tuning: bool
    contaminated: bool
    passed: bool
    evidence_ref: str
    scope: str = "pristine_final_validation"
    automatic_promotion: bool = False

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.code_commit.strip() or not self.evidence_ref.strip():
            raise ValueError("final validation identity is required")
        if not _HEX64.fullmatch(self.dataset_seal_sha256):
            raise ValueError("dataset_seal_sha256 must be a lowercase 64-char SHA-256 hex")
        if self.scope != "pristine_final_validation":
            raise ValueError("final validation scope must be pristine_final_validation")
        values = (self.sealed_at, self.evaluation_started_at, self.evaluation_finished_at)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("final validation timestamps must be finite")
        if self.sealed_at > self.evaluation_started_at:
            raise ValueError("final dataset must be sealed before evaluation starts")
        if self.evaluation_started_at > self.evaluation_finished_at:
            raise ValueError("final validation finish cannot precede start")
        if self.evaluation_count != 1:
            raise ValueError("final validation is a one-shot evaluation")
        if self.used_for_tuning:
            raise ValueError("final validation dataset cannot be used for tuning")
        if self.contaminated:
            raise ValueError("contaminated data cannot certify final validation")
        if self.automatic_promotion:
            raise ValueError("final validation receipt cannot authorize automatic promotion")


@dataclass(frozen=True, slots=True)
class HardenedPromotionReceipt:
    status: HardenedStatus
    candidate_id: str
    checks: tuple[tuple[str, bool], ...]
    blockers: tuple[str, ...]
    preliminary_status: str
    final_validation_evidence_ref: str | None
    reconciliation_ready: bool
    advanced_robustness_status: str
    human_authorization_required: bool = True
    exchange_adapter_enabled: bool = False
    capital_authorized: bool = False
    automatic_activation: bool = False
    live_execution: bool = False
    schema_version: str = PHASE59_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_hardened_promotion(
    preliminary: PromotionReceipt,
    advanced_robustness: AdvancedOverfitReport,
    reconciliation: ReconciliationBatchReport,
    final_validation: FinalValidationReceipt | None,
) -> HardenedPromotionReceipt:
    """Final pre-live review gate; still does not enable execution.

    Phase 49 is intentionally treated as preliminary. This hardened gate also
    requires multiple-testing/overfit controls, authoritative execution-state
    reconciliation, and a one-shot pristine final validation artifact.
    """
    preliminary_ok = (
        preliminary.status == "MICRO_LIVE_ELIGIBLE"
        and all(value for _, value in preliminary.checks)
        and preliminary.exchange_adapter_enabled is False
        and preliminary.automatic_activation is False
        and preliminary.live_execution is False
    )
    advanced_ok = (
        advanced_robustness.status == "ADVANCED_ROBUSTNESS_CANDIDATE"
        and all(value for _, value in advanced_robustness.checks)
        and advanced_robustness.automatic_promotion is False
        and advanced_robustness.live_execution is False
    )
    reconciliation_ok = (
        reconciliation.ready
        and all(value for _, value in reconciliation.checks)
        and reconciliation.live_execution is False
    )
    final_ok = bool(
        final_validation is not None
        and final_validation.passed
        and final_validation.evaluation_count == 1
        and not final_validation.used_for_tuning
        and not final_validation.contaminated
        and not final_validation.automatic_promotion
    )

    checks = (
        ("phase49_preliminary_passed", preliminary_ok),
        ("phase58_advanced_overfit_passed", advanced_ok),
        ("phase50_execution_state_reconciled", reconciliation_ok),
        ("pristine_final_validation_passed", final_ok),
    )
    blockers = tuple(name for name, passed in checks if not passed)

    if not preliminary_ok or not advanced_ok:
        status: HardenedStatus = "PRELIVE_RESEARCH_BLOCKED"
    elif not reconciliation_ok:
        status = "EXECUTION_RECONCILIATION_BLOCKED"
    elif not final_ok:
        status = "FINAL_VALIDATION_REQUIRED"
    else:
        status = "HUMAN_REVIEW_READY"

    return HardenedPromotionReceipt(
        status=status,
        candidate_id=preliminary.candidate_id,
        checks=checks,
        blockers=blockers,
        preliminary_status=preliminary.status,
        final_validation_evidence_ref=(
            None if final_validation is None else final_validation.evidence_ref
        ),
        reconciliation_ready=reconciliation.ready,
        advanced_robustness_status=advanced_robustness.status,
    )
