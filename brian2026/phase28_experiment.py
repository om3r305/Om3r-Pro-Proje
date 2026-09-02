from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import subprocess
import time

import numpy as np

from .data import canonical_hash
from .expert_reasoner import ExpertDecision, ExpertReasonerConfig, reason_market
from .phase25_experiment import (
    COSTS, END, FOLDS, HORIZON, START, _portfolio, _portfolio_excluding_months,
    _summary, ts,
)
from .phase27_experiment import (
    _clean_label_horizon, _expanding_year_splits, build_phase27_features,
    run as run_phase27,
)
from .robustness import assert_development_only

SCHEMA = "brian.phase28-expert-reasoner.v1"
DECLARATION = "NO PRISTINE FINAL HOLDOUT EVALUATED"
CONFIG = ExpertReasonerConfig()
VARIANTS = {
    "EXPERT_REASONER_FULL": {},
    "EXPERT_REASONER_NO_RISK_CRITIC": {"use_risk_critic": False},
    "EXPERT_REASONER_SINGLE_TIMEFRAME": {"single_timeframe": True},
    "EXPERT_REASONER_NO_VOLUME": {"use_volume": False},
    "EXPERT_REASONER_NO_DIVERGENCE": {"use_divergence": False},
}


@dataclass(frozen=True, slots=True)
class NarrativeEvidencePolicy:
    min_total_trades: int = 200
    min_trades_per_fold: int = 40
    min_coverage: float = 0.02
    min_positive_expectancy_folds: int = 2
    min_profit_factor: float = 1.10
    max_drawdown_pct: float = 20.0
    require_stress_positive: bool = True
    min_directional_accuracy: float = 0.50
    min_mean_agreement: float = 0.60


def _snapshot(features: dict[str, np.ndarray], index: int) -> dict[str, float]:
    return {name: float(values[index]) for name, values in features.items()}


def _decisions(indices, t, features, **variant) -> list[ExpertDecision]:
    out = []
    for index in indices:
        out.append(reason_market(
            _snapshot(features, int(index)), timestamp=float(t[index]), config=CONFIG, **variant
        ))
    return out


def _actions(decisions) -> list[str]:
    return [item.action for item in decisions]


def _prediction_summary(decisions, labels) -> dict:
    acted = [(decision, int(label)) for decision, label in zip(decisions, labels) if decision.action != "WAIT"]
    correct = 0
    directional = 0
    for decision, label in acted:
        if label == 0:
            continue
        directional += 1
        correct += int((decision.action == "BUY" and label > 0) or (decision.action == "SELL" and label < 0))
    return {
        "observations": len(decisions),
        "signals": len(acted),
        "coverage": len(acted) / len(decisions) if decisions else 0.0,
        "wait_rate": 1.0 - len(acted) / len(decisions) if decisions else 1.0,
        "directional_samples": directional,
        "directional_accuracy": correct / directional if directional else 0.0,
        "mean_confidence": float(np.mean([x.confidence for x in decisions])) if decisions else 0.0,
        "mean_agreement": float(np.mean([x.agreement for x in decisions])) if decisions else 0.0,
        "mean_abs_edge": float(np.mean([abs(x.edge) for x in decisions])) if decisions else 0.0,
        "risk_vetoes": sum(any(expert.name == "risk_critic" and expert.veto for expert in x.experts) for x in decisions),
    }


def _setup_report(decisions, labels, future) -> list[dict]:
    groups: dict[str, list[int]] = {}
    for position, decision in enumerate(decisions):
        groups.setdefault(decision.setup, []).append(position)
    report = []
    for setup, positions in sorted(groups.items()):
        acted = [p for p in positions if decisions[p].action != "WAIT"]
        directional = [p for p in acted if int(labels[p]) != 0]
        correct = sum(
            (decisions[p].action == "BUY" and labels[p] > 0)
            or (decisions[p].action == "SELL" and labels[p] < 0)
            for p in directional
        )
        signed_returns = [
            float(future[p]) * (1.0 if decisions[p].action == "BUY" else -1.0)
            for p in acted
        ]
        report.append({
            "setup": setup,
            "observations": len(positions),
            "signals": len(acted),
            "directional_accuracy": correct / len(directional) if directional else 0.0,
            "mean_signed_forward_return_pct": float(np.mean(signed_returns)) if signed_returns else 0.0,
            "status": "SUFFICIENT" if len(acted) >= 40 else "INSUFFICIENT_SAMPLE",
        })
    return report


def _regime_report(decisions) -> list[dict]:
    groups: dict[str, list[ExpertDecision]] = {}
    for decision in decisions:
        groups.setdefault(decision.regime, []).append(decision)
    return [{
        "regime": regime,
        "observations": len(rows),
        "signals": sum(x.action != "WAIT" for x in rows),
        "wait_rate": sum(x.action == "WAIT" for x in rows) / len(rows),
        "mean_confidence": float(np.mean([x.confidence for x in rows])),
        "mean_agreement": float(np.mean([x.agreement for x in rows])),
    } for regime, rows in sorted(groups.items())]


def _fixed_examples(fold: int, decisions: list[ExpertDecision]) -> list[dict]:
    # Deterministic explanatory examples only. Selection never uses outcomes/PnL.
    wait = [x for x in decisions if x.action == "WAIT"][:5]
    acted = [x for x in decisions if x.action != "WAIT"][:5]
    return [{"fold": fold, **x.manifest()} for x in acted + wait]


def _candidate(primary_rows, policy: NarrativeEvidencePolicy = NarrativeEvidencePolicy()) -> dict:
    reasons: list[str] = []
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
    if sum(float(row["cost_sensitivity"]["STRESS"]["net_pnl"]) for row in primary_rows) <= 0 and policy.require_stress_positive:
        reasons.append("cost-stress survivability failed")
    directional = [row["prediction"] for row in primary_rows if row["prediction"]["directional_samples"]]
    accuracy = float(np.mean([row["directional_accuracy"] for row in directional])) if directional else 0.0
    agreement = float(np.mean([row["prediction"]["mean_agreement"] for row in primary_rows])) if primary_rows else 0.0
    if accuracy < policy.min_directional_accuracy:
        reasons.append("directional accuracy gate failed")
    if agreement < policy.min_mean_agreement:
        reasons.append("expert agreement gate failed")
    return {
        "status": "DEVELOPMENT_CANDIDATE" if not reasons else "INSUFFICIENT_EVIDENCE",
        "reasons": reasons,
        "policy": asdict(policy),
        "coverage": coverage,
        "directional_accuracy": accuracy,
        "mean_agreement": agreement,
        "final_champion": False,
        "shadow_only": True,
    }


def run(root: Path, dataset_id: str) -> dict:
    started = time.time()
    phase27 = run_phase27(root, dataset_id)
    t, o, h, l, c, _, features, structures = build_phase27_features(root)
    assert_development_only(t, "Phase 2.8 expert reasoning")
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
        "purge_seconds": 3600,
        "embargo_seconds": 3600,
        "reasoner_config": asdict(CONFIG),
        "experts": ["structure", "trend_mtf", "momentum", "volume", "mean_reversion", "risk_critic", "head_trader"],
        "scenario_model": ["bull_case", "bear_case", "no_trade_case"],
        "setups": [
            "LIQUIDITY_SWEEP_REVERSAL", "FAILED_BREAK_REVERSAL", "BREAKOUT_RETEST_CONTINUATION",
            "PULLBACK_CONTINUATION", "RANGE_REJECTION", "TREND_EXHAUSTION", "NO_CLEAR_SETUP",
        ],
        "variants": VARIANTS,
        "selection": "NO_TEST_TUNING; fixed thresholds committed before development evidence is reviewed",
        "narratives": "deterministic evidence-linked explanations; scenario strengths are not calibrated probabilities",
        "historical_order_book": "UNAVAILABLE; no fabricated order flow",
        "execution": "SHADOW_RESEARCH_ONLY",
        "holdout": {"status": "INVALID_CONTAMINATED", "evaluation_allowed": False},
        "declaration": DECLARATION,
    }
    preregistration_id = canonical_hash(preregistration)
    directory = root / "phase28"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"preregistered-{preregistration_id}.json").write_text(
        json.dumps({"preregistration_id": preregistration_id, **preregistration}, sort_keys=True, separators=(",", ":")) + "\n"
    )

    quality_manifest = json.loads((root / "dataset_manifests" / f"{dataset_id}.json").read_text())
    excluded_months = {row["period"] for row in quality_manifest["monthly_builds"] if row["anomaly_classification"] != "NONE"}
    month_keys = np.asarray([
        datetime.fromtimestamp(float(value), timezone.utc).strftime("%Y-%m") for value in t
    ])

    results: list[dict] = []
    primary: list[dict] = []
    setup_reports: list[dict] = []
    regime_reports: list[dict] = []
    narrative_examples: list[dict] = []

    for fold, (start, train_end, validation_end, test_end) in enumerate(FOLDS, 1):
        train_end_ts = ts(train_end)
        validation_end_ts = ts(validation_end)
        test_end_ts = ts(test_end)
        train = np.where(target_available & (t >= ts(start)) & (t < train_end_ts - 3600) & (resolution_t < train_end_ts))[0]
        validation = np.where(target_available & (t >= train_end_ts + 3600) & (t < validation_end_ts - 3600) & (resolution_t < validation_end_ts))[0]
        test = np.where(target_available & (t >= validation_end_ts + 3600) & (t < test_end_ts) & (resolution_t < test_end_ts))[0]
        neutral = float(np.quantile(np.abs(future[train]), 0.33))
        labels = np.where(future > neutral, 1, np.where(future < -neutral, -1, 0))

        for name, variant in VARIANTS.items():
            decisions = _decisions(test, t, features, **variant)
            actions = _actions(decisions)
            prediction = _prediction_summary(decisions, labels[test])
            costs = {cost: _summary(_portfolio(test, actions, t, o, h, l, c, cost)) for cost in COSTS}
            row = {
                "fold": fold,
                "model": name,
                "reasoner_config": asdict(CONFIG),
                "prediction": prediction,
                "cost_sensitivity": costs,
                "sample_counts": {"train": len(train), "validation": len(validation), "evaluation": len(test)},
                "threshold_selection": "NONE_FIXED_PREREGISTERED",
            }
            results.append(row)
            if name == "EXPERT_REASONER_FULL":
                primary.append(row)
                setup_reports.extend({"fold": fold, **item} for item in _setup_report(
                    decisions, labels[test], future[test]
                ))
                regime_reports.extend({"fold": fold, **item} for item in _regime_report(decisions))
                narrative_examples.extend(_fixed_examples(fold, decisions))

    source_gap_sensitivity = []
    clean = ~np.isin(month_keys, list(excluded_months))
    clean_horizon = _clean_label_horizon(clean)
    for fold, (start, train_end, validation_end, test_end) in enumerate(FOLDS, 1):
        train_end_ts = ts(train_end)
        validation_end_ts = ts(validation_end)
        test_end_ts = ts(test_end)
        clean_train = np.where(target_available & clean_horizon & (t >= ts(start)) & (t < train_end_ts - 3600) & (resolution_t < train_end_ts))[0]
        test_all = np.where(target_available & (t >= validation_end_ts + 3600) & (t < test_end_ts) & (resolution_t < test_end_ts))[0]
        test = test_all[clean_horizon[test_all]]
        neutral = float(np.quantile(np.abs(future[clean_train]), 0.33))
        labels = np.where(future > neutral, 1, np.where(future < -neutral, -1, 0))
        decisions = _decisions(test, t, features)
        actions_by_index = dict(zip(test.tolist(), _actions(decisions)))
        all_actions = [actions_by_index.get(int(index), "WAIT") for index in test_all]
        source_gap_sensitivity.append({
            "fold": fold,
            "excluded_months": sorted(excluded_months),
            "reasoner_fixed_no_refit": True,
            "target_neutral_band_recomputed_from_clean_train_only": True,
            "forced_flat_at_gap_boundaries": True,
            "label_horizon_excludes_gap_crossings": True,
            "prediction": _prediction_summary(decisions, labels[test]),
            "cost_sensitivity": {cost: _summary(_portfolio_excluding_months(
                test_all, all_actions, t, o, h, l, c, excluded_months, cost
            )) for cost in COSTS},
        })

    yearly_robustness = []
    for year, train, test in _expanding_year_splits(t, resolution_t):
        train = train[target_available[train]]
        test = test[target_available[test]]
        if not len(train) or not len(test):
            continue
        neutral = float(np.quantile(np.abs(future[train]), 0.33))
        labels = np.where(future > neutral, 1, np.where(future < -neutral, -1, 0))
        decisions = _decisions(test, t, features)
        actions = _actions(decisions)
        yearly_robustness.append({
            "test_year": year,
            "train_max_timestamp": float(t[train].max()),
            "test_min_timestamp": float(t[test].min()),
            "causal_train_before_test": bool(float(t[train].max()) < float(t[test].min())),
            "reasoner_fixed_no_refit": True,
            "prediction": _prediction_summary(decisions, labels[test]),
            "cost_sensitivity": {cost: _summary(_portfolio(test, actions, t, o, h, l, c, cost)) for cost in COSTS},
        })

    candidate = _candidate(primary)
    code_version = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    experiment_id = canonical_hash({
        "schema": SCHEMA,
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
        "phase27_reference_experiment_id": phase27["experiment_id"],
        "phase27_reference_candidate_decision": phase27.get("candidate_decision"),
        "results": results,
        "ablations": [row for row in results if row["model"] != "EXPERT_REASONER_FULL"],
        "setup_report": setup_reports,
        "regime_report": regime_reports,
        "source_gap_sensitivity": source_gap_sensitivity,
        "expanding_walk_forward_yearly_robustness": yearly_robustness,
        "narrative_examples": narrative_examples,
        "candidate_decision": candidate,
        "market_structure_summary": {
            "rows": len(structures),
            "confirmed_swings": sum(len(x.confirmed_swings) for x in structures),
        },
        "interpretation_contract": {
            "scenario_strength_is_probability": False,
            "narratives_are_evidence_linked": True,
            "no_test_outcome_used_for_decision": True,
            "no_live_execution": True,
        },
        "execution": "SHADOW_RESEARCH_ONLY",
        "historical_order_book": "UNAVAILABLE",
        "historical_bid_ask": "UNAVAILABLE; spread/slippage are SIMULATION ASSUMPTIONS",
        "holdout": {"status": "INVALID_CONTAMINATED", "evaluation_allowed": False},
        "declaration": DECLARATION,
        "runtime_seconds": time.time() - started,
    }
    path = directory / f"{experiment_id}.json"
    path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({
        "experiment_id": experiment_id,
        "candidate_decision": candidate,
        "max_observed_timestamp": float(t.max()),
        "declaration": DECLARATION,
    }, sort_keys=True), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Brian Phase 2.8 expert chart-reasoning development experiment")
    parser.add_argument("--root", default="research_data")
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args()
    run(Path(args.root), args.dataset_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
