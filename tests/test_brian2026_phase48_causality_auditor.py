from __future__ import annotations

import pytest

from brian2026.phase48_causality_auditor import (
    PartitionContract,
    TemporalDependency,
    audit_temporal_dependencies,
    causality_gate,
    run_lookahead_probe,
)


def _causal_evaluator(rows):
    out = []
    running = 0.0
    for index, value in enumerate(rows):
        running += float(value)
        mean = running / (index + 1)
        out.append({
            "indicator": mean,
            "entry_signal": mean > 2.0,
        })
    return out


def _biased_evaluator(rows):
    future_mean = sum(float(value) for value in rows) / len(rows)
    return [
        {
            "indicator": future_mean,
            "entry_signal": future_mean > 2.0,
        }
        for _ in rows
    ]


def test_temporal_dependency_audit_rejects_future_source_availability() -> None:
    good = audit_temporal_dependencies((
        TemporalDependency("closed-5m", 100.0, 100.0),
        TemporalDependency("news-published", 100.0, 99.0),
    ))
    assert good.passed is True

    bad = audit_temporal_dependencies((
        TemporalDependency("macro-revision", 100.0, 101.0),
        TemporalDependency("book", 100.0, 100.0),
    ))
    assert bad.passed is False
    assert bad.future_dependencies == ("macro-revision",)


def test_partition_contract_enforces_train_only_fit_and_validation_only_calibration() -> None:
    PartitionContract("train", "train", "validation", "validation").validate()
    with pytest.raises(ValueError, match="preprocessing"):
        PartitionContract("validation", "train", "validation", "validation").validate()
    with pytest.raises(ValueError, match="calibration"):
        PartitionContract("train", "train", "test", "validation").validate()


def test_prefix_probe_passes_for_causal_strategy() -> None:
    rows = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    report = run_lookahead_probe(
        rows,
        _causal_evaluator,
        probe_indices=(1, 2, 3, 4, 5),
        signal_fields=("entry_signal",),
        indicator_fields=("indicator",),
        minimum_probes=5,
    )
    assert report.status == "PASS_CAUSALITY"
    assert report.has_bias is False
    assert report.differences == ()


def test_prefix_probe_detects_future_dependent_indicator_and_signal() -> None:
    rows = (1.0, 1.0, 1.0, 10.0, 10.0, 10.0)
    report = run_lookahead_probe(
        rows,
        _biased_evaluator,
        probe_indices=(0, 1, 2, 3, 4),
        signal_fields=("entry_signal",),
        indicator_fields=("indicator",),
        minimum_probes=5,
    )
    assert report.status == "FAIL_LOOKAHEAD"
    assert report.has_bias is True
    assert report.biased_indicator_fields == ("indicator",)
    assert report.biased_signal_fields == ("entry_signal",)


def test_too_few_probes_never_claims_no_bias() -> None:
    report = run_lookahead_probe(
        (1.0, 2.0, 3.0),
        _causal_evaluator,
        probe_indices=(1, 2),
        signal_fields=("entry_signal",),
        indicator_fields=("indicator",),
        minimum_probes=3,
    )
    assert report.status == "INSUFFICIENT_PROBES"
    assert report.has_bias is False


def test_causality_gate_requires_temporal_probe_and_partition_contract() -> None:
    temporal = audit_temporal_dependencies((
        TemporalDependency("feature", 100.0, 99.0),
    ))
    lookahead = run_lookahead_probe(
        (1.0, 2.0, 3.0, 4.0, 5.0),
        _causal_evaluator,
        probe_indices=(0, 1, 2, 3, 4),
        signal_fields=("entry_signal",),
        indicator_fields=("indicator",),
        minimum_probes=5,
    )
    gate = causality_gate(
        temporal,
        lookahead,
        PartitionContract("train", "train", "validation", "validation"),
    )
    assert gate.eligible is True
    assert gate.shadow_only is True
    assert gate.live_execution is False
    assert gate.automatic_promotion is False

    bad = causality_gate(
        temporal,
        lookahead,
        PartitionContract("train", "train", "test", "validation"),
    )
    assert bad.eligible is False
    assert any("calibration" in reason for reason in bad.reasons)
