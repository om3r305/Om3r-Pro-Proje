from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence, TypeVar
import math

PHASE48_SCHEMA_VERSION = "brian.phase48-causality-auditor.v1"
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TemporalDependency:
    name: str
    decision_timestamp: float
    source_available_timestamp: float
    role: str = "feature"

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.role.strip():
            raise ValueError("temporal dependency identity is required")
        if not math.isfinite(self.decision_timestamp) or not math.isfinite(self.source_available_timestamp):
            raise ValueError("dependency timestamps must be finite")


@dataclass(frozen=True, slots=True)
class TemporalAuditResult:
    passed: bool
    future_dependencies: tuple[str, ...]
    checked: int
    schema_version: str = PHASE48_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_temporal_dependencies(
    dependencies: Sequence[TemporalDependency],
) -> TemporalAuditResult:
    """Fail if a feature/source becomes available after its decision.

    Labels/outcomes should not be passed as feature dependencies. The role field
    is retained in the receipt so callers can audit what was checked.
    """
    rows = tuple(dependencies)
    if not rows:
        raise ValueError("temporal audit requires dependencies")
    future = tuple(sorted(
        row.name
        for row in rows
        if row.source_available_timestamp > row.decision_timestamp
    ))
    return TemporalAuditResult(not future, future, len(rows))


@dataclass(frozen=True, slots=True)
class PartitionContract:
    preprocessing_fit_partition: str
    model_fit_partition: str
    calibration_partition: str
    threshold_selection_partition: str

    def validate(self) -> None:
        if self.preprocessing_fit_partition != "train":
            raise ValueError("preprocessing must fit on train only")
        if self.model_fit_partition != "train":
            raise ValueError("model must fit on train only")
        if self.calibration_partition != "validation":
            raise ValueError("calibration must use validation only")
        if self.threshold_selection_partition != "validation":
            raise ValueError("threshold selection must use validation only")


@dataclass(frozen=True, slots=True)
class LookaheadDifference:
    probe_index: int
    field: str
    baseline_value: object
    sliced_value: object
    category: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LookaheadAuditReport:
    status: str
    probes_requested: int
    probes_completed: int
    minimum_probes: int
    biased_signal_fields: tuple[str, ...]
    biased_indicator_fields: tuple[str, ...]
    differences: tuple[LookaheadDifference, ...]
    schema_version: str = PHASE48_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False
    automatic_promotion: bool = False

    @property
    def has_bias(self) -> bool:
        return bool(self.differences)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "probes_requested": self.probes_requested,
            "probes_completed": self.probes_completed,
            "minimum_probes": self.minimum_probes,
            "biased_signal_fields": self.biased_signal_fields,
            "biased_indicator_fields": self.biased_indicator_fields,
            "differences": [row.to_dict() for row in self.differences],
            "has_bias": self.has_bias,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
            "automatic_promotion": self.automatic_promotion,
        }


def _same_value(left: object, right: object, tolerance: float) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return left == right
    if math.isnan(a) and math.isnan(b):
        return True
    if not math.isfinite(a) or not math.isfinite(b):
        return a == b
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def run_lookahead_probe(
    rows: Sequence[T],
    evaluator: Callable[[Sequence[T]], Sequence[Mapping[str, object]]],
    *,
    probe_indices: Sequence[int],
    signal_fields: Sequence[str],
    indicator_fields: Sequence[str],
    minimum_probes: int = 5,
    tolerance: float = 1e-10,
) -> LookaheadAuditReport:
    """Compare full-history outputs with prefix-only reruns at the same decision.

    This is a clean-room behavioral adaptation of Freqtrade lookahead-analysis:
    establish one full baseline, rerun selected decisions on data sliced at the
    decision boundary, and report any indicator/signal values that move merely
    because future rows were removed.
    """
    data = tuple(rows)
    if not data:
        raise ValueError("lookahead probe requires ordered input rows")
    if minimum_probes < 1:
        raise ValueError("minimum_probes must be positive")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    fields = tuple(dict.fromkeys(tuple(signal_fields) + tuple(indicator_fields)))
    if not fields:
        raise ValueError("at least one signal or indicator field is required")

    baseline = tuple(evaluator(data))
    if len(baseline) != len(data):
        raise ValueError("baseline evaluator output must align one-to-one with input rows")

    differences: list[LookaheadDifference] = []
    completed = 0
    for raw_index in probe_indices:
        index = int(raw_index)
        if index < 0 or index >= len(data):
            raise IndexError(f"probe index out of range: {index}")
        sliced = tuple(evaluator(data[: index + 1]))
        if len(sliced) != index + 1:
            raise ValueError("sliced evaluator output must align one-to-one with prefix")
        completed += 1
        full_row = baseline[index]
        sliced_row = sliced[-1]
        for field in fields:
            left = full_row.get(field)
            right = sliced_row.get(field)
            if _same_value(left, right, tolerance):
                continue
            category = "signal" if field in signal_fields else "indicator"
            differences.append(
                LookaheadDifference(index, field, left, right, category)
            )

    biased_signals = tuple(sorted({
        row.field for row in differences if row.category == "signal"
    }))
    biased_indicators = tuple(sorted({
        row.field for row in differences if row.category == "indicator"
    }))
    if completed < minimum_probes:
        status = "INSUFFICIENT_PROBES"
    elif differences:
        status = "FAIL_LOOKAHEAD"
    else:
        status = "PASS_CAUSALITY"

    return LookaheadAuditReport(
        status=status,
        probes_requested=len(tuple(probe_indices)),
        probes_completed=completed,
        minimum_probes=minimum_probes,
        biased_signal_fields=biased_signals,
        biased_indicator_fields=biased_indicators,
        differences=tuple(differences),
    )


@dataclass(frozen=True, slots=True)
class CausalityPromotionGate:
    temporal: TemporalAuditResult
    lookahead: LookaheadAuditReport
    partition_contract_passed: bool
    eligible: bool
    reasons: tuple[str, ...]
    shadow_only: bool = True
    live_execution: bool = False
    automatic_promotion: bool = False


def causality_gate(
    temporal: TemporalAuditResult,
    lookahead: LookaheadAuditReport,
    partition_contract: PartitionContract,
) -> CausalityPromotionGate:
    reasons: list[str] = []
    try:
        partition_contract.validate()
        partition_ok = True
    except ValueError as exc:
        partition_ok = False
        reasons.append(str(exc))

    if not temporal.passed:
        reasons.append("future source availability detected")
    if lookahead.status == "INSUFFICIENT_PROBES":
        reasons.append("insufficient lookahead probes")
    elif lookahead.has_bias:
        reasons.append("baseline/sliced output divergence detected")

    eligible = (
        temporal.passed
        and lookahead.status == "PASS_CAUSALITY"
        and partition_ok
    )
    return CausalityPromotionGate(
        temporal=temporal,
        lookahead=lookahead,
        partition_contract_passed=partition_ok,
        eligible=eligible,
        reasons=tuple(reasons),
    )
