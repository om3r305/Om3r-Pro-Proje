from __future__ import annotations

from brian2026.phase42_autonomous_alpha_lab import (
    AlphaExperimentPolicy,
    AlphaHypothesis,
    AutonomousAlphaLab,
    augment_samples,
    compile_factor,
)
from brian2026.samples import SupervisedSample
from brian2026.splits import WalkForwardSplitter


def _xor_samples(count: int = 120) -> tuple[SupervisedSample, ...]:
    patterns = (
        (-1.0, -1.0, 1),
        (-1.0, 1.0, -1),
        (1.0, -1.0, -1),
        (1.0, 1.0, 1),
    )
    rows = []
    for index in range(count):
        left, right, label = patterns[index % len(patterns)]
        rows.append(SupervisedSample(
            timestamp=1_700_000_000.0 + index * 300.0,
            symbol="BTCUSDT",
            features=(("x", left), ("y", right)),
            label=label,
            future_return=float(label),
            target_timestamp=1_700_000_000.0 + (index + 1) * 300.0,
            dataset_id="phase42-xor-fixture",
        ))
    return tuple(rows)


def test_compiled_factor_is_deterministic_and_preserves_sample_contract() -> None:
    hypothesis = AlphaHypothesis(
        title="XOR interaction",
        rationale="The interaction should reveal information not linearly present in either parent feature.",
        inputs=("x", "y"),
        operation="product",
    )
    first = compile_factor(hypothesis)
    second = compile_factor(hypothesis)
    assert first == second
    assert first.live_execution is False
    assert first.shadow_only is True

    rows = _xor_samples(4)
    augmented = augment_samples(rows, first)
    assert [row.timestamp for row in augmented] == [row.timestamp for row in rows]
    assert [row.target_timestamp for row in augmented] == [row.target_timestamp for row in rows]
    assert [row.label for row in augmented] == [row.label for row in rows]
    assert [row.dataset_id for row in augmented] == [row.dataset_id for row in rows]
    values = augmented[0].feature_dict()
    assert values[first.output_feature] == 1.0


def test_phase42_runs_real_propose_compile_evaluate_feedback_trace_without_promotion() -> None:
    rows = _xor_samples()
    folds = WalkForwardSplitter(
        train_size=40,
        validation_size=20,
        test_size=20,
        purge=0,
        embargo=0,
        max_folds=2,
    ).split(len(rows))
    hypothesis = AlphaHypothesis(
        title="XOR interaction",
        rationale="A product term should expose a nonlinear interaction that the unchanged linear baseline cannot represent.",
        inputs=("x", "y"),
        operation="product",
    )
    lab = AutonomousAlphaLab(
        policy=AlphaExperimentPolicy(
            min_mean_balanced_accuracy_delta=0.10,
            max_mean_brier_delta=0.0,
            max_fold_balanced_accuracy_regression=0.0,
            min_folds=2,
        )
    )
    entry = lab.run_once(rows, folds, hypothesis=hypothesis)

    assert len(lab.trace) == 1
    assert entry.candidate.hypothesis_id == hypothesis.hypothesis_id
    assert entry.outcome.decision == "RESEARCH_CHALLENGER_CANDIDATE"
    assert entry.outcome.mean_balanced_accuracy_delta > 0.40
    assert entry.outcome.mean_brier_delta < 0.0
    assert entry.feedback.decision == "KEEP_AS_RESEARCH_CHALLENGER"
    assert entry.outcome.live_execution is False
    assert entry.outcome.automatic_promotion is False
    assert entry.to_dict()["shadow_only"] is True


def test_generated_hypothesis_reads_train_partition_only() -> None:
    rows = _xor_samples()
    folds = WalkForwardSplitter(
        train_size=40,
        validation_size=20,
        test_size=20,
        max_folds=2,
    ).split(len(rows))
    lab = AutonomousAlphaLab(
        policy=AlphaExperimentPolicy(
            min_mean_balanced_accuracy_delta=0.0,
            max_mean_brier_delta=1.0,
            max_fold_balanced_accuracy_regression=1.0,
            min_folds=2,
        )
    )
    entry = lab.run_once(rows, folds)
    assert set(entry.hypothesis.inputs) == {"x", "y"}
    assert entry.hypothesis.operation == "product"
    assert entry.candidate.output_feature.startswith("alpha_product_")
    assert entry.trace_id
