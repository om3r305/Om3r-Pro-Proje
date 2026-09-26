from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from statistics import fmean
from typing import Literal, Sequence
import json
import math

from .evaluation import LockedFold, evaluate_predictions, fit_fold
from .learning import LogisticRegressionBaseline, metadata_for
from .samples import SupervisedSample
from .splits import WalkForwardFold

PHASE42_SCHEMA_VERSION = "brian.phase42-autonomous-alpha-lab.v1"
FactorOperation = Literal["product", "difference", "sum", "ratio", "abs_difference"]


def _canonical_hash(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return sha256(text.encode("utf-8")).hexdigest()


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0
    mx, my = fmean(xs), fmean(ys)
    dx = [value - mx for value in xs]
    dy = [value - my for value in ys]
    left = math.sqrt(sum(value * value for value in dx))
    right = math.sqrt(sum(value * value for value in dy))
    if left <= 1e-12 or right <= 1e-12:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy)) / (left * right)


@dataclass(frozen=True, slots=True)
class AlphaHypothesis:
    title: str
    rationale: str
    inputs: tuple[str, str]
    operation: FactorOperation = "product"
    expected_metric: str = "balanced_accuracy"

    def __post_init__(self) -> None:
        if not self.title.strip() or not self.rationale.strip():
            raise ValueError("hypothesis title and rationale are required")
        if len(self.inputs) != 2 or not all(str(value).strip() for value in self.inputs):
            raise ValueError("factor hypothesis requires two named inputs")
        if self.inputs[0] == self.inputs[1]:
            raise ValueError("factor inputs must be distinct")
        if self.expected_metric != "balanced_accuracy":
            raise ValueError("Phase 42 currently locks evaluation to balanced_accuracy")

    @property
    def hypothesis_id(self) -> str:
        return _canonical_hash({
            "title": self.title,
            "rationale": self.rationale,
            "inputs": self.inputs,
            "operation": self.operation,
            "expected_metric": self.expected_metric,
        })


@dataclass(frozen=True, slots=True)
class CompiledFactor:
    hypothesis_id: str
    output_feature: str
    operation: FactorOperation
    inputs: tuple[str, str]
    epsilon: float
    source_descriptor: str
    code_hash: str
    candidate_id: str
    training_only: bool = True
    shadow_only: bool = True
    live_execution: bool = False

    def compute(self, values: dict[str, float | None]) -> float | None:
        left = _finite(values.get(self.inputs[0]))
        right = _finite(values.get(self.inputs[1]))
        if left is None or right is None:
            return None
        if self.operation == "product":
            value = left * right
        elif self.operation == "difference":
            value = left - right
        elif self.operation == "sum":
            value = left + right
        elif self.operation == "ratio":
            if abs(right) <= self.epsilon:
                return None
            value = left / right
        elif self.operation == "abs_difference":
            value = abs(left - right)
        else:
            raise ValueError("unsupported factor operation")
        return float(value) if math.isfinite(value) else None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compile_factor(hypothesis: AlphaHypothesis, *, epsilon: float = 1e-12) -> CompiledFactor:
    """Compile a bounded factor description into a deterministic candidate artifact.

    The artifact is data-only and cannot execute orders or mutate production code.
    It mirrors RD-Agent's hypothesis->implementation step while keeping Brian's
    promotion boundary intact.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    output_feature = f"alpha_{hypothesis.operation}_{hypothesis.inputs[0]}__{hypothesis.inputs[1]}"
    source_descriptor = (
        f"{output_feature}={hypothesis.operation}("
        f"{hypothesis.inputs[0]},{hypothesis.inputs[1]})"
    )
    code_hash = _canonical_hash({
        "descriptor": source_descriptor,
        "epsilon": epsilon,
        "schema": PHASE42_SCHEMA_VERSION,
    })
    candidate_id = _canonical_hash({
        "hypothesis_id": hypothesis.hypothesis_id,
        "code_hash": code_hash,
    })
    return CompiledFactor(
        hypothesis_id=hypothesis.hypothesis_id,
        output_feature=output_feature,
        operation=hypothesis.operation,
        inputs=hypothesis.inputs,
        epsilon=epsilon,
        source_descriptor=source_descriptor,
        code_hash=code_hash,
        candidate_id=candidate_id,
    )


def augment_samples(samples: Sequence[SupervisedSample], candidate: CompiledFactor) -> tuple[SupervisedSample, ...]:
    result: list[SupervisedSample] = []
    for row in samples:
        values = row.feature_dict()
        if candidate.output_feature in values:
            raise ValueError(f"candidate feature already exists: {candidate.output_feature}")
        values[candidate.output_feature] = candidate.compute(values)
        result.append(SupervisedSample(
            timestamp=row.timestamp,
            symbol=row.symbol,
            features=tuple(sorted(values.items())),
            label=row.label,
            future_return=row.future_return,
            target_timestamp=row.target_timestamp,
            dataset_id=row.dataset_id,
        ))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class TraceAwareHypothesisGenerator:
    """Deterministic train-only interaction hypothesis generator.

    This is the safe built-in generator. An LLM generator can be plugged in later,
    but it must produce the same AlphaHypothesis contract and may only inspect the
    research trace plus train-partition statistics.
    """
    minimum_observations: int = 20

    def propose(
        self,
        samples: Sequence[SupervisedSample],
        train_indices: Sequence[int],
        *,
        rejected_pairs: Sequence[tuple[str, str]] = (),
    ) -> AlphaHypothesis:
        rows = tuple(samples[int(index)] for index in train_indices)
        if len(rows) < self.minimum_observations:
            raise ValueError("hypothesis generation requires more train observations")
        feature_names = sorted({name for row in rows for name, _ in row.features})
        scores: list[tuple[float, str, float]] = []
        for name in feature_names:
            xs: list[float] = []
            ys: list[float] = []
            for row in rows:
                value = _finite(row.feature_dict().get(name))
                if value is None or not math.isfinite(float(row.future_return)):
                    continue
                xs.append(value)
                ys.append(float(row.future_return))
            if len(xs) < self.minimum_observations:
                continue
            corr = _pearson(xs, ys)
            if math.isfinite(corr):
                scores.append((abs(corr), name, corr))
        if len(scores) < 2:
            raise ValueError("not enough numeric train-only features for hypothesis generation")
        scores.sort(key=lambda item: (-item[0], item[1]))
        rejected = {tuple(sorted(pair)) for pair in rejected_pairs}
        selected: tuple[str, str] | None = None
        for _, left, _ in scores:
            for _, right, _ in scores:
                if left >= right:
                    continue
                if tuple(sorted((left, right))) in rejected:
                    continue
                selected = (left, right)
                break
            if selected is not None:
                break
        if selected is None:
            raise ValueError("no unused feature pair remains for hypothesis generation")
        left, right = selected
        return AlphaHypothesis(
            title=f"Interaction: {left} × {right}",
            rationale=(
                "Train-only univariate screening identified both inputs as individually "
                "informative. Test their interaction without changing labels, timestamps, "
                "or the locked evaluation partition."
            ),
            inputs=(left, right),
            operation="product",
        )


@dataclass(frozen=True, slots=True)
class FoldComparison:
    fold: int
    test_count: int
    baseline_balanced_accuracy: float
    candidate_balanced_accuracy: float
    balanced_accuracy_delta: float
    baseline_brier: float
    candidate_brier: float
    brier_delta: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AlphaExperimentPolicy:
    min_mean_balanced_accuracy_delta: float = 0.005
    max_mean_brier_delta: float = 0.0
    max_fold_balanced_accuracy_regression: float = 0.02
    min_folds: int = 2

    def __post_init__(self) -> None:
        if self.min_folds < 1:
            raise ValueError("min_folds must be positive")
        if self.max_fold_balanced_accuracy_regression < 0:
            raise ValueError("regression tolerance must be non-negative")


@dataclass(frozen=True, slots=True)
class AlphaExperimentOutcome:
    experiment_id: str
    candidate_id: str
    comparisons: tuple[FoldComparison, ...]
    mean_balanced_accuracy_delta: float
    mean_brier_delta: float
    checks: tuple[tuple[str, bool], ...]
    decision: str
    training_only: bool = True
    shadow_only: bool = True
    live_execution: bool = False
    automatic_promotion: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "candidate_id": self.candidate_id,
            "comparisons": [row.to_dict() for row in self.comparisons],
            "mean_balanced_accuracy_delta": self.mean_balanced_accuracy_delta,
            "mean_brier_delta": self.mean_brier_delta,
            "checks": dict(self.checks),
            "decision": self.decision,
            "training_only": self.training_only,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
            "automatic_promotion": self.automatic_promotion,
        }


def _validate_dataset_identity(samples: Sequence[SupervisedSample]) -> str:
    ids = {row.dataset_id for row in samples}
    if len(ids) != 1:
        raise ValueError("Phase 42 requires one immutable dataset identity")
    dataset_id = next(iter(ids))
    if not dataset_id:
        raise ValueError("Phase 42 requires a non-empty dataset identity")
    return str(dataset_id)


def run_factor_experiment(
    hypothesis: AlphaHypothesis,
    candidate: CompiledFactor,
    samples: Sequence[SupervisedSample],
    folds: Sequence[WalkForwardFold],
    *,
    policy: AlphaExperimentPolicy = AlphaExperimentPolicy(),
) -> AlphaExperimentOutcome:
    """Evaluate one compiled factor against the unchanged Brian baseline.

    Baseline and candidate models receive identical train/validation/test indices.
    Model/preprocessing fitting stays train-only, calibration/threshold selection
    stays validation-only, and each locked test view is consumed exactly once per
    model instance.
    """
    if candidate.hypothesis_id != hypothesis.hypothesis_id:
        raise ValueError("candidate does not belong to hypothesis")
    rows = tuple(samples)
    if not rows:
        raise ValueError("experiment requires samples")
    dataset_id = _validate_dataset_identity(rows)
    if len(folds) < policy.min_folds:
        raise ValueError("experiment has insufficient preregistered folds")
    augmented = augment_samples(rows, candidate)
    comparisons: list[FoldComparison] = []

    for fold in folds:
        fold.validate()
        if fold.test.stop > len(rows):
            raise ValueError("fold exceeds sample count")

        baseline_model = LogisticRegressionBaseline(metadata_for(
            "logistic_regression",
            dataset_id,
            PHASE42_SCHEMA_VERSION,
            "baseline",
            fold.fold,
        ))
        candidate_model = LogisticRegressionBaseline(metadata_for(
            "logistic_regression",
            dataset_id,
            PHASE42_SCHEMA_VERSION,
            candidate.code_hash,
            fold.fold,
        ))

        baseline_locked = LockedFold(fold, rows)
        candidate_locked = LockedFold(fold, augmented)
        baseline_thresholds, baseline_probabilities = fit_fold(baseline_model, baseline_locked)
        candidate_thresholds, candidate_probabilities = fit_fold(candidate_model, candidate_locked)
        labels = [rows[index].label for index in fold.test]

        baseline_metrics = evaluate_predictions(baseline_probabilities, labels, baseline_thresholds)
        candidate_metrics = evaluate_predictions(candidate_probabilities, labels, candidate_thresholds)
        comparisons.append(FoldComparison(
            fold=fold.fold,
            test_count=len(labels),
            baseline_balanced_accuracy=baseline_metrics.balanced_accuracy,
            candidate_balanced_accuracy=candidate_metrics.balanced_accuracy,
            balanced_accuracy_delta=candidate_metrics.balanced_accuracy - baseline_metrics.balanced_accuracy,
            baseline_brier=baseline_metrics.brier_score,
            candidate_brier=candidate_metrics.brier_score,
            brier_delta=candidate_metrics.brier_score - baseline_metrics.brier_score,
        ))

    mean_accuracy_delta = fmean(row.balanced_accuracy_delta for row in comparisons)
    mean_brier_delta = fmean(row.brier_delta for row in comparisons)
    checks = (
        ("enough_locked_folds", len(comparisons) >= policy.min_folds),
        ("mean_accuracy_improved", mean_accuracy_delta >= policy.min_mean_balanced_accuracy_delta),
        ("mean_brier_not_worse", mean_brier_delta <= policy.max_mean_brier_delta),
        (
            "no_material_fold_regression",
            all(
                row.balanced_accuracy_delta >= -policy.max_fold_balanced_accuracy_regression
                for row in comparisons
            ),
        ),
    )
    experiment_id = _canonical_hash({
        "schema": PHASE42_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "hypothesis_id": hypothesis.hypothesis_id,
        "candidate_id": candidate.candidate_id,
        "folds": [fold.to_dict() for fold in folds],
        "policy": asdict(policy),
    })
    return AlphaExperimentOutcome(
        experiment_id=experiment_id,
        candidate_id=candidate.candidate_id,
        comparisons=tuple(comparisons),
        mean_balanced_accuracy_delta=mean_accuracy_delta,
        mean_brier_delta=mean_brier_delta,
        checks=checks,
        decision=(
            "RESEARCH_CHALLENGER_CANDIDATE"
            if all(value for _, value in checks)
            else "REJECTED_EXPERIMENT"
        ),
    )


@dataclass(frozen=True, slots=True)
class AlphaFeedback:
    experiment_id: str
    decision: str
    observations: str
    next_action: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_feedback(outcome: AlphaExperimentOutcome) -> AlphaFeedback:
    failed = [name for name, passed in outcome.checks if not passed]
    if not failed:
        return AlphaFeedback(
            experiment_id=outcome.experiment_id,
            decision="KEEP_AS_RESEARCH_CHALLENGER",
            observations=(
                f"mean balanced-accuracy delta={outcome.mean_balanced_accuracy_delta:+.6f}; "
                f"mean Brier delta={outcome.mean_brier_delta:+.6f}; "
                "all preregistered Phase 42 checks passed"
            ),
            next_action=(
                "Send candidate to Phase 41 robustness and trading-PnL evaluation. "
                "Do not promote or execute from this result alone."
            ),
        )
    return AlphaFeedback(
        experiment_id=outcome.experiment_id,
        decision="REJECT_OR_REVISE",
        observations=(
            f"failed checks={','.join(failed)}; "
            f"mean balanced-accuracy delta={outcome.mean_balanced_accuracy_delta:+.6f}; "
            f"mean Brier delta={outcome.mean_brier_delta:+.6f}"
        ),
        next_action="Record failure in trace and generate a different bounded hypothesis.",
    )


@dataclass(frozen=True, slots=True)
class AlphaTraceEntry:
    hypothesis: AlphaHypothesis
    candidate: CompiledFactor
    outcome: AlphaExperimentOutcome
    feedback: AlphaFeedback
    trace_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "hypothesis": {
                **asdict(self.hypothesis),
                "hypothesis_id": self.hypothesis.hypothesis_id,
            },
            "candidate": self.candidate.to_dict(),
            "outcome": self.outcome.to_dict(),
            "feedback": self.feedback.to_dict(),
            "trace_id": self.trace_id,
            "schema_version": PHASE42_SCHEMA_VERSION,
            "training_only": True,
            "shadow_only": True,
            "live_execution": False,
            "automatic_promotion": False,
        }


class AutonomousAlphaLab:
    """RD-Agent-style propose -> compile -> run -> feedback -> record loop.

    The loop deliberately stops at a research challenger. Promotion remains the
    responsibility of Brian's locked evaluation/promotion system.
    """

    def __init__(
        self,
        *,
        generator: TraceAwareHypothesisGenerator | None = None,
        policy: AlphaExperimentPolicy = AlphaExperimentPolicy(),
    ) -> None:
        self.generator = generator or TraceAwareHypothesisGenerator()
        self.policy = policy
        self._trace: list[AlphaTraceEntry] = []

    @property
    def trace(self) -> tuple[AlphaTraceEntry, ...]:
        return tuple(self._trace)

    def run_once(
        self,
        samples: Sequence[SupervisedSample],
        folds: Sequence[WalkForwardFold],
        *,
        hypothesis: AlphaHypothesis | None = None,
    ) -> AlphaTraceEntry:
        if not folds:
            raise ValueError("Alpha Lab requires preregistered folds")
        if hypothesis is None:
            rejected_pairs = tuple(
                entry.hypothesis.inputs
                for entry in self._trace
                if entry.outcome.decision == "REJECTED_EXPERIMENT"
            )
            hypothesis = self.generator.propose(
                samples,
                folds[0].train,
                rejected_pairs=rejected_pairs,
            )
        candidate = compile_factor(hypothesis)
        outcome = run_factor_experiment(
            hypothesis,
            candidate,
            samples,
            folds,
            policy=self.policy,
        )
        feedback = summarize_feedback(outcome)
        trace_id = _canonical_hash({
            "prior_trace_ids": [entry.trace_id for entry in self._trace],
            "hypothesis_id": hypothesis.hypothesis_id,
            "candidate_id": candidate.candidate_id,
            "experiment_id": outcome.experiment_id,
            "feedback": feedback.to_dict(),
        })
        entry = AlphaTraceEntry(hypothesis, candidate, outcome, feedback, trace_id)
        self._trace.append(entry)
        return entry
