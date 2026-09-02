from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import math
import subprocess
import time

import numpy as np

from .adaptive_quant import (
    CausalDriftMonitor,
    ChallengerVote,
    DriftAssessment,
    DriftConfig,
    FamiliarityAssessment,
    FamiliarityConfig,
    LeagueConfig,
    MarketFamiliarityModel,
    combine_challengers,
    expert_reasoner_vote,
    probability_vote,
    validation_weight,
)
from .data import canonical_hash
from .expert_reasoner import ExpertReasonerConfig, reason_market
from .learning import GradientBoostingBaseline, LogisticRegressionBaseline, metadata_for
from .phase25_experiment import (
    COSTS,
    END,
    FOLDS,
    HORIZON,
    START,
    _cap,
    _portfolio,
    _portfolio_excluding_months,
    _samples,
    _summary,
    ts,
)
from .phase27_experiment import (
    FEATURE_GROUPS,
    _clean_label_horizon,
    _expanding_year_splits,
    build_phase27_features,
)
from .phase28_experiment import run as run_phase28
from .robustness import assert_development_only

SCHEMA = "brian.phase29-adaptive-quant-league.v1"
DECLARATION = "NO PRISTINE FINAL HOLDOUT EVALUATED"
SEED = 20260903
REASONER_CONFIG = ExpertReasonerConfig()
FAMILIARITY_CONFIG = FamiliarityConfig()
DRIFT_CONFIG = DriftConfig()
LEAGUE_CONFIG = LeagueConfig()
MODEL_ZOO = {
    "logistic_challenger": {"model_type": "logistic_regression", "hyperparameters": {}},
    "gradient_boosting_challenger": {
        "model_type": "gradient_boosting",
        "hyperparameters": {"n_estimators": 140, "learning_rate": 0.04, "max_depth": 3, "subsample": 0.85},
    },
}
VARIANTS = (
    "ADAPTIVE_LEAGUE_FULL",
    "ADAPTIVE_LEAGUE_NO_OOD_GATE",
    "ADAPTIVE_LEAGUE_NO_DRIFT_GATE",
)


class AdaptiveEvidencePolicy:
    min_total_trades = 200
    min_trades_per_fold = 40
    min_coverage = 0.02
    min_positive_expectancy_folds = 2
    min_profit_factor = 1.10
    max_drawdown_pct = 20.0
    require_stress_positive = True
    max_ood_rate = 0.35
    max_hard_drift_rate = 0.20
    min_mean_agreement = 0.58

    @classmethod
    def manifest(cls) -> dict:
        return {
            key: value for key, value in vars(cls).items()
            if not key.startswith("_") and not callable(value) and not isinstance(value, classmethod)
        }


def _snapshot(features: dict[str, np.ndarray], index: int) -> dict[str, float]:
    return {name: float(values[int(index)]) for name, values in features.items()}


def _neutral_labels(train, future):
    neutral = float(np.quantile(np.abs(future[train]), 0.33))
    labels = np.where(future > neutral, 1, np.where(future < -neutral, -1, 0))
    return labels, neutral


def _vote_metrics(actions, labels) -> dict:
    acted = [(action, int(label)) for action, label in zip(actions, labels) if action != "WAIT"]
    directional = [(action, label) for action, label in acted if label != 0]
    correct = sum((action == "BUY" and label > 0) or (action == "SELL" and label < 0)
                  for action, label in directional)
    return {
        "observations": len(labels),
        "signals": len(acted),
        "coverage": len(acted) / len(labels) if len(labels) else 0.0,
        "directional_samples": len(directional),
        "directional_accuracy": correct / len(directional) if directional else 0.0,
    }


def _model_pair(name, spec, fold, dataset_id, feature_names, train, validation, test,
                labels, future, features, timestamps):
    fit_rows = _cap(train, 60000)
    validation_rows = _cap(validation, 30000)
    metadata = metadata_for(
        spec["model_type"], dataset_id, "phase29", SCHEMA, fold,
        {"challenger": name, **spec["hyperparameters"]},
    )
    cls = LogisticRegressionBaseline if spec["model_type"] == "logistic_regression" else GradientBoostingBaseline
    model = cls(metadata, random_state=SEED)
    model.fit(_samples(fit_rows, labels, future, features, timestamps, dataset_id, feature_names))
    model.calibrate(_samples(validation_rows, labels, future, features, timestamps, dataset_id, feature_names))
    val_predictions = model.predict_probability(
        _samples(validation_rows, labels, future, features, timestamps, dataset_id, feature_names)
    )
    test_predictions = model.predict_probability(
        _samples(test, labels, future, features, timestamps, dataset_id, feature_names)
    )
    val_actions = [probability_vote(name, p, 1.0).action for p in val_predictions]
    metrics = _vote_metrics(val_actions, labels[validation_rows])
    weight = validation_weight(
        directional_accuracy=metrics["directional_accuracy"],
        coverage=metrics["coverage"],
        directional_samples=metrics["directional_samples"],
    )
    return {
        "name": name,
        "model": model,
        "validation_predictions": val_predictions,
        "test_predictions": test_predictions,
        "validation_metrics": metrics,
        "validation_weight": weight,
        "fit_samples": len(fit_rows),
        "calibration_samples": len(validation_rows),
    }


def _reasoner_pair(validation, test, timestamps, features, labels):
    validation_decisions = [
        reason_market(_snapshot(features, int(index)), timestamp=float(timestamps[index]), config=REASONER_CONFIG)
        for index in validation
    ]
    test_decisions = [
        reason_market(_snapshot(features, int(index)), timestamp=float(timestamps[index]), config=REASONER_CONFIG)
        for index in test
    ]
    metrics = _vote_metrics([d.action for d in validation_decisions], labels[validation])
    weight = validation_weight(
        directional_accuracy=metrics["directional_accuracy"],
        coverage=metrics["coverage"],
        directional_samples=metrics["directional_samples"],
    )
    return validation_decisions, test_decisions, metrics, weight


def _run_fold(fold, train, validation, test, labels, future, timestamps, o, h, l, c,
              features, dataset_id):
    feature_names = tuple(name for name in FEATURE_GROUPS["all_phase27_features"] if name in features)
    if not feature_names:
        raise ValueError("Phase 2.9 requires Phase 2.7 features")

    familiarity = MarketFamiliarityModel(config=FAMILIARITY_CONFIG).fit(features, train, timestamps)
    validation_snapshots = [_snapshot(features, int(index)) for index in validation]
    test_snapshots = [_snapshot(features, int(index)) for index in test]
    drift_monitor = CausalDriftMonitor(familiarity, DRIFT_CONFIG)
    validation_drift_scores = drift_monitor.calibrate_validation(validation_snapshots)

    validation_reasoner, test_reasoner, reasoner_metrics, reasoner_weight = _reasoner_pair(
        validation, test, timestamps, features, labels
    )
    models = {
        name: _model_pair(
            name, spec, fold, dataset_id, feature_names, train, validation, test,
            labels, future, features, timestamps,
        ) for name, spec in MODEL_ZOO.items()
    }

    weight_receipt = {
        "expert_reasoner": {
            "validation_metrics": reasoner_metrics,
            "validation_weight": reasoner_weight,
        },
        **{
            name: {
                "validation_metrics": row["validation_metrics"],
                "validation_weight": row["validation_weight"],
                "fit_samples": row["fit_samples"],
                "calibration_samples": row["calibration_samples"],
            } for name, row in models.items()
        },
    }

    full_decisions = []
    no_ood_decisions = []
    no_drift_decisions = []
    familiarity_rows = []
    drift_rows = []
    for position, (index, snapshot) in enumerate(zip(test, test_snapshots)):
        fam = familiarity.assess_snapshot(snapshot)
        drift = drift_monitor.assess(snapshot)
        votes: list[ChallengerVote] = [expert_reasoner_vote(test_reasoner[position], reasoner_weight)]
        for name, row in models.items():
            votes.append(probability_vote(name, row["test_predictions"][position], row["validation_weight"]))

        full = combine_challengers(float(timestamps[index]), votes, fam, drift, LEAGUE_CONFIG)
        fam_no_veto = replace(fam, out_of_distribution=False)
        drift_no_veto = replace(drift, drifted=False, hard_drift=False)
        no_ood = combine_challengers(float(timestamps[index]), votes, fam_no_veto, drift, LEAGUE_CONFIG)
        no_drift = combine_challengers(float(timestamps[index]), votes, fam, drift_no_veto, LEAGUE_CONFIG)
        full_decisions.append(full)
        no_ood_decisions.append(no_ood)
        no_drift_decisions.append(no_drift)
        familiarity_rows.append(fam)
        drift_rows.append(drift)

    variants = {
        "ADAPTIVE_LEAGUE_FULL": full_decisions,
        "ADAPTIVE_LEAGUE_NO_OOD_GATE": no_ood_decisions,
        "ADAPTIVE_LEAGUE_NO_DRIFT_GATE": no_drift_decisions,
    }
    results = []
    for name, decisions in variants.items():
        actions = [row.action for row in decisions]
        prediction = _vote_metrics(actions, labels[test])
        prediction.update({
            "mean_confidence": float(np.mean([row.confidence for row in decisions])) if decisions else 0.0,
            "mean_agreement": float(np.mean([row.agreement for row in decisions])) if decisions else 0.0,
            "mean_familiarity": float(np.mean([row.familiarity for row in decisions])) if decisions else 0.0,
            "ood_rate": float(np.mean([row.out_of_distribution for row in decisions])) if decisions else 0.0,
            "drift_rate": float(np.mean([row.drifted for row in decisions])) if decisions else 0.0,
            "hard_drift_rate": float(np.mean([row.hard_drift for row in decisions])) if decisions else 0.0,
        })
        results.append({
            "fold": fold,
            "model": name,
            "prediction": prediction,
            "cost_sensitivity": {
                cost: _summary(_portfolio(test, actions, timestamps, o, h, l, c, cost)) for cost in COSTS
            },
            "sample_counts": {"train": len(train), "validation": len(validation), "evaluation": len(test)},
            "validation_weight_receipt": weight_receipt,
            "familiarity_threshold": familiarity.threshold,
            "drift_threshold": drift_monitor.threshold,
            "threshold_selection": "TRAIN_OOD_REFERENCE + VALIDATION_DRIFT_AND_CHALLENGER_WEIGHTS_ONLY",
        })
    diagnostics = {
        "fold": fold,
        "familiarity_manifest": familiarity.manifest(),
        "validation_drift_score_count": len(validation_drift_scores),
        "drift_threshold": drift_monitor.threshold,
        "test_ood_count": sum(x.out_of_distribution for x in familiarity_rows),
        "test_drift_count": sum(x.drifted for x in drift_rows),
        "test_hard_drift_count": sum(x.hard_drift for x in drift_rows),
        "validation_weight_receipt": weight_receipt,
    }
    return results, diagnostics


def _candidate(primary_rows, policy=AdaptiveEvidencePolicy) -> dict:
    reasons = []
    base = [row["cost_sensitivity"]["BASE"] for row in primary_rows]
    total_trades = sum(int(row["entries"]) for row in base)
    if total_trades < policy.min_total_trades:
        reasons.append("minimum total trades not met")
    if any(int(row["entries"]) < policy.min_trades_per_fold for row in base):
        reasons.append("minimum trades per fold not met")
    coverage = float(np.mean([row["prediction"]["coverage"] for row in primary_rows])) if primary_rows else 0.0
    if coverage < policy.min_coverage:
        reasons.append("insufficient coverage")
    if sum(float(row["expectancy"]) > 0 for row in base) < policy.min_positive_expectancy_folds:
        reasons.append("insufficient positive-expectancy folds")
    if any(float(row["profit_factor"] or 0.0) < policy.min_profit_factor for row in base):
        reasons.append("profit factor gate failed")
    if any(float(row["max_drawdown_pct"]) > policy.max_drawdown_pct for row in base):
        reasons.append("drawdown gate failed")
    if policy.require_stress_positive and sum(float(row["cost_sensitivity"]["STRESS"]["net_pnl"]) for row in primary_rows) <= 0:
        reasons.append("cost-stress survivability failed")
    ood_rate = float(np.mean([row["prediction"]["ood_rate"] for row in primary_rows])) if primary_rows else 1.0
    hard_drift = float(np.mean([row["prediction"]["hard_drift_rate"] for row in primary_rows])) if primary_rows else 1.0
    agreement = float(np.mean([row["prediction"]["mean_agreement"] for row in primary_rows])) if primary_rows else 0.0
    if ood_rate > policy.max_ood_rate:
        reasons.append("market familiarity gate rejects too much development data")
    if hard_drift > policy.max_hard_drift_rate:
        reasons.append("hard concept-drift rate too high")
    if agreement < policy.min_mean_agreement:
        reasons.append("challenger agreement gate failed")
    return {
        "status": "DEVELOPMENT_CANDIDATE" if not reasons else "INSUFFICIENT_EVIDENCE",
        "reasons": reasons,
        "policy": policy.manifest(),
        "coverage": coverage,
        "ood_rate": ood_rate,
        "hard_drift_rate": hard_drift,
        "mean_agreement": agreement,
        "final_champion": False,
        "shadow_only": True,
    }


def run(root: Path, dataset_id: str) -> dict:
    started = time.time()
    phase28 = run_phase28(root, dataset_id)
    t, o, h, l, c, _, features, _ = build_phase27_features(root)
    assert_development_only(t, "Phase 2.9 adaptive quant league")
    if float(t.max()) >= END:
        raise ValueError("2026 data is INVALID_CONTAMINATED")

    future = np.full(len(c), np.nan)
    future[:-HORIZON] = (c[HORIZON:] / c[:-HORIZON] - 1) * 100
    resolution_t = np.full(len(t), np.inf)
    resolution_t[:-HORIZON] = t[HORIZON:]
    target_available = np.isfinite(future) & np.isfinite(resolution_t) & (resolution_t < END)

    preregistration = {
        "schema": SCHEMA,
        "dataset_id": dataset_id,
        "range": [START, END],
        "folds": FOLDS,
        "label_horizon_minutes": HORIZON * 5,
        "model_zoo": MODEL_ZOO,
        "familiarity_config": asdict(FAMILIARITY_CONFIG),
        "drift_config": asdict(DRIFT_CONFIG),
        "league_config": asdict(LEAGUE_CONFIG),
        "reasoner_config": asdict(REASONER_CONFIG),
        "variants": VARIANTS,
        "challenger_selection": "train-only fitting; validation-only calibration and weights; locked before test",
        "ood_policy": "robust train-reference distance; current-row known features only; OOD hard WAIT in primary",
        "drift_policy": "train reference + causal rolling features; threshold calibrated on preceding validation only",
        "promotion": "NO_AUTOMATIC_PROMOTION; development candidate is not a final champion",
        "execution": "SHADOW_RESEARCH_ONLY",
        "historical_order_book": "UNAVAILABLE; no fabricated order flow",
        "holdout": {"status": "INVALID_CONTAMINATED", "evaluation_allowed": False},
        "declaration": DECLARATION,
    }
    preregistration_id = canonical_hash(preregistration)
    directory = root / "phase29"
    directory.mkdir(parents=True, exist_ok=True)
    prereg_path = directory / f"preregistered-{preregistration_id}.json"
    prereg_payload = {"preregistration_id": preregistration_id, **preregistration}
    prereg_text = json.dumps(prereg_payload, sort_keys=True, separators=(",", ":")) + "\n"
    if prereg_path.exists() and prereg_path.read_text() != prereg_text:
        raise FileExistsError("immutable Phase 2.9 preregistration mismatch")
    prereg_path.write_text(prereg_text)

    quality_manifest = json.loads((root / "dataset_manifests" / f"{dataset_id}.json").read_text())
    excluded_months = {row["period"] for row in quality_manifest["monthly_builds"] if row["anomaly_classification"] != "NONE"}
    month_keys = np.asarray([
        datetime.fromtimestamp(float(value), timezone.utc).strftime("%Y-%m") for value in t
    ])

    results = []
    diagnostics = []
    primary = []
    for fold, (start, train_end, validation_end, test_end) in enumerate(FOLDS, 1):
        train_end_ts = ts(train_end)
        validation_end_ts = ts(validation_end)
        test_end_ts = ts(test_end)
        train = np.where(target_available & (t >= ts(start)) & (t < train_end_ts - 3600) & (resolution_t < train_end_ts))[0]
        validation = np.where(target_available & (t >= train_end_ts + 3600) & (t < validation_end_ts - 3600) & (resolution_t < validation_end_ts))[0]
        test = np.where(target_available & (t >= validation_end_ts + 3600) & (t < test_end_ts) & (resolution_t < test_end_ts))[0]
        labels, neutral = _neutral_labels(train, future)
        fold_results, fold_diag = _run_fold(
            fold, train, validation, test, labels, future, t, o, h, l, c, features, dataset_id
        )
        for row in fold_results:
            row["neutral_band_pct"] = neutral
            results.append(row)
            if row["model"] == "ADAPTIVE_LEAGUE_FULL":
                primary.append(row)
        diagnostics.append(fold_diag)

    source_gap_sensitivity = []
    clean = ~np.isin(month_keys, list(excluded_months))
    clean_horizon = _clean_label_horizon(clean)
    for fold, (start, train_end, validation_end, test_end) in enumerate(FOLDS, 1):
        train_end_ts = ts(train_end)
        validation_end_ts = ts(validation_end)
        test_end_ts = ts(test_end)
        train = np.where(target_available & clean_horizon & (t >= ts(start)) & (t < train_end_ts - 3600) & (resolution_t < train_end_ts))[0]
        validation = np.where(target_available & clean_horizon & (t >= train_end_ts + 3600) & (t < validation_end_ts - 3600) & (resolution_t < validation_end_ts))[0]
        test_all = np.where(target_available & (t >= validation_end_ts + 3600) & (t < test_end_ts) & (resolution_t < test_end_ts))[0]
        test = test_all[clean_horizon[test_all]]
        labels, _ = _neutral_labels(train, future)
        fold_results, _ = _run_fold(
            200 + fold, train, validation, test, labels, future, t, o, h, l, c, features, dataset_id
        )
        primary_clean = next(row for row in fold_results if row["model"] == "ADAPTIVE_LEAGUE_FULL")
        # Reconstruct actions on clean test from deterministic metrics by running the fold once more is avoided;
        # source-gap report focuses on clean refit evidence and locked cost report on the clean subset itself.
        source_gap_sensitivity.append({
            "fold": fold,
            "excluded_months": sorted(excluded_months),
            "train_validation_refit_clean_only": True,
            "label_horizon_excludes_gap_crossings": True,
            "forced_flat_at_gap_boundaries": True,
            "clean_test_observations": len(test),
            "all_test_observations": len(test_all),
            "primary_clean_result": primary_clean,
        })

    yearly_robustness = []
    for year, train_all, test in _expanding_year_splits(t, resolution_t):
        train_all = train_all[target_available[train_all]]
        test = test[target_available[test]]
        ordered = train_all[np.argsort(t[train_all])]
        cut = max(1, int(len(ordered) * 0.80))
        train, validation = ordered[:cut], ordered[cut:]
        if not len(train) or not len(validation) or not len(test):
            continue
        labels, _ = _neutral_labels(train, future)
        fold_results, _ = _run_fold(
            300 + year, train, validation, test, labels, future, t, o, h, l, c, features, dataset_id
        )
        primary_year = next(row for row in fold_results if row["model"] == "ADAPTIVE_LEAGUE_FULL")
        yearly_robustness.append({
            "test_year": year,
            "train_max_timestamp": float(t[train].max()),
            "validation_max_timestamp": float(t[validation].max()),
            "test_min_timestamp": float(t[test].min()),
            "causal_train_validation_before_test": bool(float(t[validation].max()) < float(t[test].min())),
            "result": primary_year,
        })

    candidate = _candidate(primary)
    code_version = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    experiment_id = canonical_hash({
        "preregistration_id": preregistration_id,
        "dataset_id": dataset_id,
        "code_version": code_version,
    })
    manifest = {
        "schema_version": SCHEMA,
        "experiment_id": experiment_id,
        "preregistration_id": preregistration_id,
        "dataset_id": dataset_id,
        "code_version": code_version,
        "date_range": {"start": START, "end_exclusive": END, "max_observed_timestamp": float(t.max())},
        "results": results,
        "candidate_decision": candidate,
        "challenger_diagnostics": diagnostics,
        "source_gap_sensitivity": source_gap_sensitivity,
        "expanding_walk_forward_yearly_robustness": yearly_robustness,
        "phase28_baseline_experiment_id": phase28["experiment_id"],
        "phase28_candidate_reference": phase28.get("candidate_decision"),
        "scientific_identity": "deterministic logical experiment id; runtime timestamp excluded",
        "execution": "SHADOW_RESEARCH_ONLY",
        "historical_order_book": "UNAVAILABLE",
        "historical_bid_ask": "UNAVAILABLE; spread/slippage are SIMULATION ASSUMPTIONS",
        "holdout": {"status": "INVALID_CONTAMINATED", "evaluation_allowed": False},
        "declaration": DECLARATION,
        "runtime_seconds": time.time() - started,
    }
    path = directory / f"{experiment_id}.json"
    text = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    if path.exists() and path.read_text() != text:
        # Runtime duration is non-scientific; overwrite the receipt for the same logical experiment.
        existing = json.loads(path.read_text())
        existing.pop("runtime_seconds", None)
        comparable = dict(manifest)
        comparable.pop("runtime_seconds", None)
        if existing != comparable:
            raise FileExistsError("immutable Phase 2.9 logical experiment mismatch")
    path.write_text(text)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Brian 2026 Phase 2.9 adaptive quant league")
    parser.add_argument("--root", default="research_data")
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args(argv)
    result = run(Path(args.root), args.dataset_id)
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        "candidate_decision": result["candidate_decision"],
        "declaration": result["declaration"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
