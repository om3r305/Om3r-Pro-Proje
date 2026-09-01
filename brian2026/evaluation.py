from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence
import math

from sklearn.metrics import balanced_accuracy_score, brier_score_loss, precision_recall_fscore_support

from .learning import ProbabilityPrediction, SupervisedBaseline
from .metrics import PerformanceMetrics, calculate
from .policy import PolicyThresholds, decide, select_thresholds
from .replay import ReplayPoint, ReplaySettings, replay
from .samples import SupervisedSample
from .splits import WalkForwardFold


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    brier_score: float
    calibration_error: float
    acted_hit_rate: float
    coverage: float
    wait_rate: float


@dataclass(frozen=True, slots=True)
class StrategyEvaluation:
    name: str
    prediction: PredictionMetrics
    trading: PerformanceMetrics
    turnover: float
    cost_burden: float
    validated: bool


@dataclass(slots=True)
class LockedFold:
    fold: WalkForwardFold
    samples: Sequence[SupervisedSample]
    test_accessed: bool = False

    def train(self) -> tuple[SupervisedSample, ...]:
        return tuple(self.samples[i] for i in self.fold.train)

    def validation(self) -> tuple[SupervisedSample, ...]:
        return tuple(self.samples[i] for i in self.fold.validation)

    def test(self, *, purpose: str = "locked_final_evaluation") -> tuple[SupervisedSample, ...]:
        if purpose != "locked_final_evaluation":
            raise ValueError("test fold is locked")
        if self.test_accessed:
            raise RuntimeError("test fold may be evaluated only once")
        self.test_accessed = True
        return tuple(self.samples[i] for i in self.fold.test)


def calibration_error(probabilities: Sequence[ProbabilityPrediction], labels: Sequence[int], bins: int = 10) -> float:
    if not probabilities:
        return 0.0
    confidence = [max(p.down, p.neutral, p.up) for p in probabilities]
    predicted = [(-1, 0, 1)[max(range(3), key=lambda i: (p.down, p.neutral, p.up)[i])] for p in probabilities]
    total = 0.0
    for bucket in range(bins):
        lo, hi = bucket / bins, (bucket + 1) / bins
        indices = [i for i, value in enumerate(confidence) if lo <= value < hi or (bucket == bins - 1 and value == 1)]
        if indices:
            accuracy = sum(predicted[i] == labels[i] for i in indices) / len(indices)
            mean_confidence = sum(confidence[i] for i in indices) / len(indices)
            total += len(indices) / len(labels) * abs(accuracy - mean_confidence)
    return total


def evaluate_predictions(probabilities: Sequence[ProbabilityPrediction], labels: Sequence[int],
                         thresholds: PolicyThresholds) -> PredictionMetrics:
    predicted = [(-1, 0, 1)[max(range(3), key=lambda i: (p.down, p.neutral, p.up)[i])] for p in probabilities]
    precision, recall, _, _ = precision_recall_fscore_support(labels, predicted, labels=(-1, 0, 1), average="macro", zero_division=0)
    balanced = (sum(a == b for a, b in zip(labels, predicted)) / len(labels)
                if len(set(labels)) <= 1 and labels else
                float(balanced_accuracy_score(labels, predicted)) if labels else 0.0)
    encoded = [[1.0 if label == candidate else 0.0 for candidate in (-1, 0, 1)] for label in labels]
    brier = sum(sum((actual - probability) ** 2 for actual, probability in zip(row, (p.down, p.neutral, p.up)))
                for row, p in zip(encoded, probabilities)) / len(labels) if labels else 0.0
    actions = [decide(p, thresholds) for p in probabilities]
    acted = [(a, label) for a, label in zip(actions, labels) if a != "WAIT"]
    hits = sum((a == "BUY" and label == 1) or (a == "SELL" and label == -1) for a, label in acted)
    coverage = len(acted) / len(actions) if actions else 0.0
    return PredictionMetrics(balanced,
                             float(precision), float(recall), float(brier),
                             calibration_error(probabilities, labels), hits / len(acted) if acted else 0.0,
                             coverage, 1.0 - coverage)


def evaluate_with_replay(name: str, probabilities: Sequence[ProbabilityPrediction], labels: Sequence[int],
                         paths: Sequence[Sequence[ReplayPoint]], thresholds: PolicyThresholds,
                         settings: ReplaySettings, *, starting_equity: float = 10_000.0,
                         validated: bool = False) -> StrategyEvaluation:
    actions = [decide(p, thresholds) for p in probabilities]
    results = [replay(path, {"BUY": "LONG", "SELL": "SHORT", "WAIT": "WAIT"}[action], settings)
               for action, path in zip(actions, paths)]
    fills = [result for result in results if result.status == "FILLED"]
    pnls = [result.net_pnl for result in fills]
    fees = sum(result.fees + result.funding for result in fills)
    exposure = sum(result.exposure_seconds for result in fills)
    trading = calculate(pnls, starting_equity=starting_equity, exposure=exposure,
                        decisions=len(actions), waits=actions.count("WAIT"))
    turnover = sum((result.entry_price or 0) * result.filled_quantity * 2 for result in fills)
    return StrategyEvaluation(name, evaluate_predictions(probabilities, labels, thresholds), trading,
                              turnover, fees, validated)


def evaluate_actions(name: str, actions: Sequence[str], labels: Sequence[int],
                     paths: Sequence[Sequence[ReplayPoint]], settings: ReplaySettings,
                     *, starting_equity: float = 10_000.0, validated: bool = False) -> StrategyEvaluation:
    mapping = {"BUY": ProbabilityPrediction(0.0, 0.0, 1.0),
               "SELL": ProbabilityPrediction(1.0, 0.0, 0.0),
               "WAIT": ProbabilityPrediction(0.0, 1.0, 0.0)}
    if any(action not in mapping for action in actions):
        raise ValueError("actions must be BUY, SELL, or WAIT")
    return evaluate_with_replay(name, [mapping[action] for action in actions], labels, paths,
                                PolicyThresholds(0.6, 0.6, 0.1), settings,
                                starting_equity=starting_equity, validated=validated)


REQUIRED_COMPETITORS = ("wait_only", "legacy_rule", "brian_meta", "logistic_regression", "gradient_boosting")


def validate_comparison_suite(results: Mapping[str, StrategyEvaluation]) -> None:
    missing = [name for name in REQUIRED_COMPETITORS if name not in results]
    if missing:
        raise ValueError(f"missing required competitors: {missing}")

def fit_fold(model: SupervisedBaseline, locked: LockedFold) -> tuple[PolicyThresholds, tuple[ProbabilityPrediction, ...]]:
    model.fit(locked.train(), partition="train")
    validation = locked.validation()
    model.calibrate(validation, partition="validation")
    val_probabilities = model.predict_probability(validation)
    thresholds = select_thresholds(val_probabilities, [row.label for row in validation], partition="validation")
    test = locked.test()
    return thresholds, model.predict_probability(test)


@dataclass(frozen=True, slots=True)
class ChampionPolicy:
    min_folds: int = 2
    min_profit_factor: float = 1.0
    min_expectancy: float = 0.0
    max_drawdown: float = math.inf


def champion_candidate(challenger: Sequence[StrategyEvaluation], baselines: Mapping[str, Sequence[StrategyEvaluation]],
                       policy: ChampionPolicy = ChampionPolicy()) -> dict[str, object]:
    reasons: list[str] = []
    if len(challenger) < policy.min_folds:
        reasons.append("insufficient locked OOS folds")
    if any(not result.validated for result in challenger):
        reasons.append("non-locked or unvalidated evidence")
    candidate_expectancy = sum(r.trading.expectancy for r in challenger) / len(challenger) if challenger else -math.inf
    candidate_pf = sum(r.trading.profit_factor for r in challenger) / len(challenger) if challenger else 0.0
    candidate_dd = max((r.trading.max_drawdown for r in challenger), default=math.inf)
    if candidate_pf < policy.min_profit_factor or candidate_expectancy <= policy.min_expectancy or candidate_dd > policy.max_drawdown:
        reasons.append("risk constraints failed")
    for name, results in baselines.items():
        baseline_expectancy = sum(r.trading.expectancy for r in results) / len(results) if results else 0.0
        if candidate_expectancy <= baseline_expectancy:
            reasons.append(f"does not beat {name} on locked OOS expectancy")
    return {"status": "CHAMPION_CANDIDATE" if not reasons else "REJECTED",
            "live_applied": False, "configuration_modified": False, "reasons": reasons}