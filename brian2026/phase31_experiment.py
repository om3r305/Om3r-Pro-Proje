from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import json
import subprocess
import time

import numpy as np

from .adaptive_quant import (
    CausalDriftMonitor,
    ChallengerVote,
    DriftConfig,
    FamiliarityConfig,
    MarketFamiliarityModel,
    expert_reasoner_vote,
    probability_vote,
    validation_weight,
)
from .capital_efficiency import (
    OpportunityConfig,
    apply_score_threshold,
    bootstrap_risk_of_ruin,
    choose_capital_policy,
    score_opportunity,
    select_score_threshold,
)
from .data import canonical_hash
from .evidence_ledger import EvidenceRecord
from .expert_reasoner import ExpertReasonerConfig, reason_market
from .learning import GradientBoostingBaseline, LogisticRegressionBaseline, metadata_for
from .phase25_experiment import (
    COSTS, END, FOLDS, HORIZON, START, _bars, _cap, _portfolio, _samples, _summary, ts,
)
from .phase27_experiment import FEATURE_GROUPS, build_phase27_features
from .portfolio import PortfolioConfig, simulate_portfolio
from .robustness import assert_development_only

SCHEMA = "brian.phase31-edge-capital-efficiency.v1"
DECLARATION = "NO PRISTINE FINAL HOLDOUT EVALUATED"
POST_DIAGNOSTIC_DECLARATION = "POST_DIAGNOSTIC_DEVELOPMENT_ONLY; NOT PRISTINE OOS"
SEED = 20260903
REASONER_CONFIG = ExpertReasonerConfig()
FAMILIARITY_CONFIG = FamiliarityConfig()
DRIFT_CONFIG = DriftConfig()
OPPORTUNITY_CONFIG = OpportunityConfig()
MODEL_ZOO = {
    "logistic_challenger": {"model_type": "logistic_regression", "hyperparameters": {}},
    "gradient_boosting_challenger": {
        "model_type": "gradient_boosting",
        "hyperparameters": {"n_estimators": 140, "learning_rate": 0.04, "max_depth": 3, "subsample": 0.85},
    },
}
SCORE_QUANTILES = (0.75, 0.85, 0.90, 0.95, 0.975)
CAPITAL_FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.25)
CALIBRATION_FRACTION = 0.55
VALIDATION_EMBARGO_BARS = 12


class Phase31EvidencePolicy:
    min_total_trades = 150
    min_trades_per_fold = 30
    min_positive_expectancy_folds = 2
    min_profit_factor = 1.05
    require_stress_positive = True
    min_deployable_capital_folds = 2
    max_mean_ruin_probability = 0.05

    @classmethod
    def manifest(cls) -> dict:
        return {
            "min_total_trades": cls.min_total_trades,
            "min_trades_per_fold": cls.min_trades_per_fold,
            "min_positive_expectancy_folds": cls.min_positive_expectancy_folds,
            "min_profit_factor": cls.min_profit_factor,
            "require_stress_positive": cls.require_stress_positive,
            "min_deployable_capital_folds": cls.min_deployable_capital_folds,
            "max_mean_ruin_probability": cls.max_mean_ruin_probability,
        }


def _snapshot(features: dict[str, np.ndarray], index: int) -> dict[str, float]:
    return {name: float(values[int(index)]) for name, values in features.items()}


def _vote_metrics(actions, labels) -> dict:
    acted = [(a, int(y)) for a, y in zip(actions, labels) if a != "WAIT"]
    directional = [(a, y) for a, y in acted if y != 0]
    correct = sum((a == "BUY" and y > 0) or (a == "SELL" and y < 0) for a, y in directional)
    return {
        "observations": len(labels),
        "signals": len(acted),
        "coverage": len(acted) / len(labels) if len(labels) else 0.0,
        "directional_samples": len(directional),
        "directional_accuracy": correct / len(directional) if directional else 0.0,
    }


def _split_validation(validation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(validation) < 4 * VALIDATION_EMBARGO_BARS:
        raise ValueError("Phase 3.1 requires a substantial validation partition")
    pivot = int(len(validation) * CALIBRATION_FRACTION)
    calibration = validation[: max(1, pivot - VALIDATION_EMBARGO_BARS)]
    policy = validation[min(len(validation), pivot + VALIDATION_EMBARGO_BARS):]
    if not len(calibration) or not len(policy):
        raise ValueError("validation split produced an empty partition")
    return calibration, policy


def _fit_model(name, spec, fold, dataset_id, feature_names, train, calibration, policy, test,
               labels, future, features, timestamps):
    fit_rows = _cap(train, 60000)
    calibration_rows = _cap(calibration, 20000)
    metadata = metadata_for(
        spec["model_type"], dataset_id, "phase31", SCHEMA, fold,
        {"challenger": name, "post_diagnostic": True, **spec["hyperparameters"]},
    )
    cls = LogisticRegressionBaseline if spec["model_type"] == "logistic_regression" else GradientBoostingBaseline
    model = cls(metadata, random_state=SEED)
    model.fit(_samples(fit_rows, labels, future, features, timestamps, dataset_id, feature_names))
    model.calibrate(_samples(calibration_rows, labels, future, features, timestamps, dataset_id, feature_names))
    cal_predictions = model.predict_probability(
        _samples(calibration_rows, labels, future, features, timestamps, dataset_id, feature_names)
    )
    cal_actions = [probability_vote(name, p, 1.0).action for p in cal_predictions]
    cal_metrics = _vote_metrics(cal_actions, labels[calibration_rows])
    weight = validation_weight(
        directional_accuracy=cal_metrics["directional_accuracy"],
        coverage=cal_metrics["coverage"],
        directional_samples=cal_metrics["directional_samples"],
    )
    return {
        "name": name,
        "weight": weight,
        "calibration_metrics": cal_metrics,
        "fit_samples": len(fit_rows),
        "calibration_samples": len(calibration_rows),
        "policy_predictions": model.predict_probability(
            _samples(policy, labels, future, features, timestamps, dataset_id, feature_names)
        ),
        "test_predictions": model.predict_probability(
            _samples(test, labels, future, features, timestamps, dataset_id, feature_names)
        ),
    }


def _reasoner_rows(calibration, policy, test, timestamps, features, labels):
    def decisions(indices):
        return [
            reason_market(_snapshot(features, int(i)), timestamp=float(timestamps[i]), config=REASONER_CONFIG)
            for i in indices
        ]
    cal = decisions(calibration)
    metrics = _vote_metrics([x.action for x in cal], labels[calibration])
    weight = validation_weight(
        directional_accuracy=metrics["directional_accuracy"], coverage=metrics["coverage"],
        directional_samples=metrics["directional_samples"],
    )
    return {
        "weight": weight,
        "calibration_metrics": metrics,
        "calibration_decisions": cal,
        "policy_decisions": decisions(policy),
        "test_decisions": decisions(test),
    }


def _assess_partitions(train, calibration, policy, test, timestamps, features):
    familiarity = MarketFamiliarityModel(config=FAMILIARITY_CONFIG).fit(features, train, timestamps)
    monitor = CausalDriftMonitor(familiarity, DRIFT_CONFIG)
    monitor.calibrate_validation([_snapshot(features, int(i)) for i in calibration])

    def assess(indices):
        fam = []
        drift = []
        for i in indices:
            snap = _snapshot(features, int(i))
            fam.append(familiarity.assess_snapshot(snap))
            drift.append(monitor.assess(snap))
        return fam, drift

    policy_fam, policy_drift = assess(policy)
    test_fam, test_drift = assess(test)
    return familiarity, monitor, policy_fam, policy_drift, test_fam, test_drift


def _opportunity_rows(indices, reasoner_decisions, models, prediction_key, fam_rows, drift_rows, timestamps):
    out = []
    for pos, index in enumerate(indices):
        votes: list[ChallengerVote] = [expert_reasoner_vote(reasoner_decisions[pos], models["expert_reasoner_weight"])]
        for name, row in models["challengers"].items():
            votes.append(probability_vote(name, row[prediction_key][pos], row["weight"]))
        out.append(score_opportunity(
            float(timestamps[index]), votes, fam_rows[pos], drift_rows[pos], OPPORTUNITY_CONFIG
        ))
    return out


def _custom_portfolio(indices, actions, t, o, h, l, c, *, starting_equity, fraction, cost="BASE"):
    cfg = PortfolioConfig(
        starting_equity=float(starting_equity), sizing_mode="equity_fraction",
        equity_fraction=float(fraction), max_position_notional=max(float(starting_equity) * 2.0, 1.0),
        max_equity_fraction=float(fraction), stop_loss_pct=1.0, take_profit_pct=2.0,
        max_holding_bars=12, cooldown_bars=1, reversal_enabled=True, **COSTS[cost],
    )
    return simulate_portfolio(_bars(indices, t, o, h, l, c), tuple(actions), cfg)


def _individual_diagnostics(indices, labels, t, o, h, l, c, reasoner_decisions, models, prediction_key):
    rows = []
    actions_by_name = {"expert_reasoner": [x.action for x in reasoner_decisions]}
    for name, row in models.items():
        actions_by_name[name] = [probability_vote(name, p, 1.0).action for p in row[prediction_key]]
    for name, actions in actions_by_name.items():
        rows.append({
            "challenger": name,
            "prediction": _vote_metrics(actions, labels[indices]),
            "base_portfolio": _summary(_portfolio(indices, actions, t, o, h, l, c, "BASE")),
        })
    return rows


def _select_policy(policy_rows, policy, labels, t, o, h, l, c):
    candidates = []
    scores = [x.opportunity_score for x in policy_rows]
    proposed = [x.proposed_action for x in policy_rows]
    for quantile in SCORE_QUANTILES:
        receipt = select_score_threshold(scores, proposed, quantile)
        actions = apply_score_threshold(policy_rows, receipt.threshold)
        result = _portfolio(policy, actions, t, o, h, l, c, "BASE")
        summary = _summary(result)
        coverage = sum(a != "WAIT" for a in actions) / len(actions)
        eligible = summary["entries"] >= 20
        utility = summary["net_pnl"] - summary["max_drawdown"] if eligible else float("-inf")
        candidates.append({
            "quantile": quantile, "threshold": receipt.threshold, "coverage": coverage,
            "eligible": eligible, "utility": utility, "portfolio": summary,
        })
    eligible = [row for row in candidates if row["eligible"]]
    pool = eligible or candidates
    selected = max(pool, key=lambda row: (row["utility"], row["coverage"], -row["quantile"]))
    return selected, candidates


def _capital_candidates(policy, actions, t, o, h, l, c, fold):
    rows = []
    for fraction in CAPITAL_FRACTIONS:
        result = _custom_portfolio(policy, actions, t, o, h, l, c,
                                   starting_equity=500.0, fraction=fraction, cost="BASE")
        summary = _summary(result)
        trade_returns = [
            trade.net_pnl / max(trade.entry_price * trade.quantity, 1e-12)
            for trade in result.trades
        ]
        ruin = bootstrap_risk_of_ruin(
            trade_returns, fraction, seed=SEED + fold * 100 + int(fraction * 1000),
        )
        rows.append({
            "fraction": fraction, "ruin_probability": ruin,
            "net_pnl": summary["net_pnl"], "profit_factor": summary["profit_factor"],
            "entries": summary["entries"], "max_drawdown_pct": summary["max_drawdown_pct"],
            "return_pct": summary["return_pct"],
        })
    return rows


def _flat_500_manifest() -> dict:
    return {
        "starting_equity": 500.0, "ending_equity": 500.0, "net_pnl": 0.0,
        "return_pct": 0.0, "entries": 0, "max_drawdown_pct": 0.0,
        "deployment": "BLOCKED_BY_VALIDATION_CAPITAL_GATE",
    }


def _candidate(folds, policy=Phase31EvidencePolicy) -> dict:
    reasons = []
    base = [row["cost_sensitivity"]["BASE"] for row in folds]
    total = sum(int(x["entries"]) for x in base)
    if total < policy.min_total_trades:
        reasons.append("minimum total trades not met")
    if any(int(x["entries"]) < policy.min_trades_per_fold for x in base):
        reasons.append("minimum trades per fold not met")
    if sum(float(x["expectancy"]) > 0 for x in base) < policy.min_positive_expectancy_folds:
        reasons.append("insufficient positive-expectancy folds")
    if any(float(x["profit_factor"] or 0.0) < policy.min_profit_factor for x in base):
        reasons.append("profit factor gate failed")
    if policy.require_stress_positive and sum(float(row["cost_sensitivity"]["STRESS"]["net_pnl"]) for row in folds) <= 0:
        reasons.append("cost-stress survivability failed")
    deployable = [row["capital_policy"] for row in folds if row["capital_policy"]["deployable"]]
    if len(deployable) < policy.min_deployable_capital_folds:
        reasons.append("capital deployment validation gate failed")
    mean_ruin = float(np.mean([row["ruin_probability"] for row in deployable])) if deployable else 1.0
    if mean_ruin > policy.max_mean_ruin_probability:
        reasons.append("risk-of-ruin gate failed")
    all_passed = not reasons
    return {
        "status": "SHADOW_CANDIDATE" if all_passed else "INSUFFICIENT_EVIDENCE",
        "reasons": reasons, "policy": policy.manifest(), "total_trades": total,
        "deployable_capital_folds": len(deployable), "mean_ruin_probability": mean_ruin,
        "all_required_gates_passed": all_passed, "final_champion": False, "shadow_only": True,
    }


def run(root: Path, dataset_id: str) -> dict:
    started = time.time()
    t, o, h, l, c, _, features, _ = build_phase27_features(root)
    assert_development_only(t, "Phase 3.1 edge attribution and capital efficiency")
    if float(t.max()) >= END:
        raise ValueError("2026 data is INVALID_CONTAMINATED")

    feature_names = tuple(name for name in FEATURE_GROUPS["all_phase27_features"] if name in features)
    if not feature_names:
        raise ValueError("Phase 3.1 requires Phase 2.7 features")

    future = np.full(len(c), np.nan)
    future[:-HORIZON] = (c[HORIZON:] / c[:-HORIZON] - 1) * 100
    resolution_t = np.full(len(t), np.inf)
    resolution_t[:-HORIZON] = t[HORIZON:]
    target_available = np.isfinite(future) & np.isfinite(resolution_t) & (resolution_t < END)

    preregistration = {
        "schema": SCHEMA, "dataset_id": dataset_id, "range": [START, END], "folds": FOLDS,
        "post_diagnostic_declaration": POST_DIAGNOSTIC_DECLARATION,
        "known_development_observation": "Phase 2.9 hard consensus produced zero locked-fold coverage; Phase 3.1 is a new development design, not a rescue claim",
        "validation_split": {"calibration_fraction": CALIBRATION_FRACTION, "embargo_bars": VALIDATION_EMBARGO_BARS},
        "score_quantiles": SCORE_QUANTILES, "capital_fractions": CAPITAL_FRACTIONS,
        "starting_capital_benchmark": 500.0,
        "opportunity_config": asdict(OPPORTUNITY_CONFIG),
        "familiarity_config": asdict(FAMILIARITY_CONFIG), "drift_config": asdict(DRIFT_CONFIG),
        "model_zoo": MODEL_ZOO, "evidence_policy": Phase31EvidencePolicy.manifest(),
        "capital_contract": "position sizing cannot create edge; if validation edge/risk gates fail, capital fraction is zero",
        "selection_contract": "calibration/weights use calibration-validation only; score threshold and capital fraction use policy-validation only; test never selects",
        "execution": "SHADOW_RESEARCH_ONLY",
        "holdout": {"status": "INVALID_CONTAMINATED", "evaluation_allowed": False},
        "declaration": DECLARATION,
    }
    prereg_id = canonical_hash(preregistration)
    directory = root / "phase31"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"preregistered-{prereg_id}.json").write_text(
        json.dumps({"preregistration_id": prereg_id, **preregistration}, sort_keys=True, separators=(",", ":")) + "\n"
    )

    fold_rows = []
    attribution = []
    for fold, (start, train_end, validation_end, test_end) in enumerate(FOLDS, 1):
        train_end_ts, validation_end_ts, test_end_ts = ts(train_end), ts(validation_end), ts(test_end)
        train = np.where(target_available & (t >= ts(start)) & (t < train_end_ts - 3600) & (resolution_t < train_end_ts))[0]
        validation = np.where(target_available & (t >= train_end_ts + 3600) & (t < validation_end_ts - 3600) & (resolution_t < validation_end_ts))[0]
        test = np.where(target_available & (t >= validation_end_ts + 3600) & (t < test_end_ts) & (resolution_t < test_end_ts))[0]
        calibration, policy_validation = _split_validation(validation)
        neutral = float(np.quantile(np.abs(future[train]), 0.33))
        labels = np.where(future > neutral, 1, np.where(future < -neutral, -1, 0))

        reasoner = _reasoner_rows(calibration, policy_validation, test, t, features, labels)
        challengers = {
            name: _fit_model(name, spec, fold, dataset_id, feature_names, train, calibration,
                             policy_validation, test, labels, future, features, t)
            for name, spec in MODEL_ZOO.items()
        }
        models = {"expert_reasoner_weight": reasoner["weight"], "challengers": challengers}
        familiarity, monitor, policy_fam, policy_drift, test_fam, test_drift = _assess_partitions(
            train, calibration, policy_validation, test, t, features
        )
        policy_rows = _opportunity_rows(
            policy_validation, reasoner["policy_decisions"], models, "policy_predictions",
            policy_fam, policy_drift, t,
        )
        test_rows = _opportunity_rows(
            test, reasoner["test_decisions"], models, "test_predictions",
            test_fam, test_drift, t,
        )
        selected, score_curve = _select_policy(policy_rows, policy_validation, labels, t, o, h, l, c)
        policy_actions = apply_score_threshold(policy_rows, selected["threshold"])
        test_actions = apply_score_threshold(test_rows, selected["threshold"])

        capital_candidates = _capital_candidates(policy_validation, policy_actions, t, o, h, l, c, fold)
        capital_policy = choose_capital_policy(capital_candidates)
        if capital_policy.deployable:
            benchmark = _summary(_custom_portfolio(
                test, test_actions, t, o, h, l, c, starting_equity=500.0,
                fraction=capital_policy.equity_fraction, cost="BASE",
            ))
            benchmark["deployment"] = "VALIDATION_GATED_SHADOW_BENCHMARK"
        else:
            benchmark = _flat_500_manifest()

        cost_sensitivity = {
            cost: _summary(_portfolio(test, test_actions, t, o, h, l, c, cost)) for cost in COSTS
        }
        fold_rows.append({
            "fold": fold,
            "sample_counts": {
                "train": len(train), "validation_total": len(validation),
                "calibration_validation": len(calibration), "policy_validation": len(policy_validation),
                "evaluation": len(test),
            },
            "calibration_receipt": {
                "expert_reasoner": {"weight": reasoner["weight"], "metrics": reasoner["calibration_metrics"]},
                **{name: {"weight": row["weight"], "metrics": row["calibration_metrics"],
                          "fit_samples": row["fit_samples"], "calibration_samples": row["calibration_samples"]}
                   for name, row in challengers.items()},
            },
            "familiarity_threshold": familiarity.threshold, "drift_threshold": monitor.threshold,
            "policy_selection": selected, "score_curve": score_curve,
            "prediction": _vote_metrics(test_actions, labels[test]),
            "mean_opportunity_score": float(np.mean([x.opportunity_score for x in test_rows])) if test_rows else 0.0,
            "cost_sensitivity": cost_sensitivity,
            "capital_candidates": capital_candidates,
            "capital_policy": capital_policy.manifest(),
            "capital_500_test_benchmark": benchmark,
        })
        attribution.append({
            "fold": fold,
            "policy_validation": _individual_diagnostics(
                policy_validation, labels, t, o, h, l, c, reasoner["policy_decisions"], challengers, "policy_predictions"
            ),
            "locked_development_test": _individual_diagnostics(
                test, labels, t, o, h, l, c, reasoner["test_decisions"], challengers, "test_predictions"
            ),
        })

    candidate = _candidate(fold_rows)
    code_version = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    experiment_id = canonical_hash({
        "schema": SCHEMA, "preregistration_id": prereg_id, "dataset_id": dataset_id,
        "code_version": code_version,
    })
    evidence = EvidenceRecord(
        phase="phase31-post-diagnostic-development", logical_experiment_id=experiment_id,
        dataset_id=dataset_id, code_commit=code_version, scope="development",
        max_data_timestamp=float(t.max()),
        metrics={
            "total_test_entries": candidate["total_trades"],
            "deployable_capital_folds": candidate["deployable_capital_folds"],
            "mean_ruin_probability": candidate["mean_ruin_probability"],
            "aggregate_500_test_net_pnl": sum(float(row["capital_500_test_benchmark"]["net_pnl"]) for row in fold_rows),
        },
        gates={"all_required_gates_passed": candidate["all_required_gates_passed"], **candidate["policy"]},
        decision="SHADOW_CANDIDATE" if candidate["all_required_gates_passed"] else "INSUFFICIENT_EVIDENCE",
    )
    manifest = {
        "schema_version": SCHEMA, "experiment_id": experiment_id, "preregistration_id": prereg_id,
        "dataset_id": dataset_id, "code_version": code_version,
        "date_range": {"start": START, "end_exclusive": END, "max_observed_timestamp": float(t.max())},
        "post_diagnostic_declaration": POST_DIAGNOSTIC_DECLARATION,
        "results": fold_rows, "challenger_edge_attribution": attribution,
        "candidate_decision": candidate, "evidence_record": evidence.manifest(),
        "execution": "SHADOW_RESEARCH_ONLY",
        "holdout": {"status": "INVALID_CONTAMINATED", "evaluation_allowed": False},
        "declaration": DECLARATION, "runtime_seconds": time.time() - started,
    }
    (directory / f"{experiment_id}.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    print(json.dumps({
        "experiment_id": experiment_id, "candidate_decision": candidate,
        "post_diagnostic_declaration": POST_DIAGNOSTIC_DECLARATION,
        "max_observed_timestamp": float(t.max()), "declaration": DECLARATION,
    }, sort_keys=True), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Brian Phase 3.1 edge attribution and capital efficiency development experiment")
    parser.add_argument("--root", default="research_data")
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args()
    run(Path(args.root), args.dataset_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
