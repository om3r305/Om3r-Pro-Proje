from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import subprocess
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .data import canonical_hash
from .evaluation import evaluate_predictions
from .learning import LogisticRegressionBaseline, metadata_for
from .market_structure import StructureCandle, compute_market_structure
from .phase25_experiment import (
    COSTS, END, FOLDS, HORIZON, START, _bars, _cap, _config, _load, _portfolio, _samples,
    _portfolio_excluding_months, _summary, build_features, run as run_phase25, select_validation_threshold,
    static_brian_actions, threshold_curve, ts,
)
from .policy import PolicyThresholds, decide
from .phase27_specialists import dip_specialist, market_structure_specialist
from .robustness import (EvidencePolicy, assert_development_only, development_candidate,
                         purged_temporal_yearly_splits)
from .structure_exit import StructureExitConfig, apply_structure_exit
from .portfolio import simulate_portfolio

SCHEMA = "brian.phase27-market-structure.v1"
SEED = 20260902
DECLARATION = "NO PRISTINE FINAL HOLDOUT EVALUATED"
FEATURE_GROUPS = {
    "price_returns": ("return_1", "return_5"),
    "classic_technical": ("ema_fast_ratio", "ema_slow_ratio", "ema_slope", "rsi", "atr_pct", "zscore", "bb_position"),
    "candle_price_action": ("body_range_ratio", "upper_wick_ratio", "lower_wick_ratio", "close_location", "range_expansion", "range_contraction", "displacement", "inside_bar", "outside_bar", "consecutive_bullish_closes", "consecutive_bearish_closes", "structural_equilibrium_distance_atr"),
    "market_structure": ("structure_state", "bullish_bos", "bearish_bos", "bullish_choch", "bearish_choch", "latest_high_label", "latest_low_label"),
    "support_resistance": ("support_distance_atr", "resistance_distance_atr", "support_age_bars", "resistance_age_bars", "support_reactions", "resistance_reactions", "inside_support_zone", "inside_resistance_zone"),
    "liquidity_sweep": ("bullish_sweep", "bearish_sweep", "failed_breakdown", "failed_breakout", "bullish_breakout_retest", "bearish_breakout_retest"),
    "divergence": ("bullish_rsi_divergence", "bearish_rsi_divergence"),
    "volume_context": ("volume_zscore", "relative_volume", "volume_expansion", "pullback_volume_contraction", "selling_exhaustion_proxy", "buying_exhaustion_proxy"),
    "dip_rally_quality": ("dip_score", "rally_score", "momentum_deceleration", "momentum_recovery", "acceleration"),
    "structure_15m": ("structure_15m", "bullish_bos_15m", "bearish_bos_15m", "bullish_choch_15m", "bearish_choch_15m"),
    "structure_1h": ("structure_1h", "bullish_bos_1h", "bearish_bos_1h", "bullish_choch_1h", "bearish_choch_1h"),
}
FEATURE_GROUPS["market_structure_combined"] = tuple(dict.fromkeys(
    item for group in ("market_structure", "support_resistance", "liquidity_sweep", "divergence", "structure_15m", "structure_1h")
    for item in FEATURE_GROUPS[group]
))
FEATURE_GROUPS["structure_classic"] = tuple(dict.fromkeys(
    FEATURE_GROUPS["market_structure_combined"] + FEATURE_GROUPS["classic_technical"]
))
FEATURE_GROUPS["structure_technical_mtf"] = tuple(dict.fromkeys(
    FEATURE_GROUPS["structure_classic"] + FEATURE_GROUPS["structure_15m"] +
    FEATURE_GROUPS["structure_1h"] + FEATURE_GROUPS["dip_rally_quality"]
))
FEATURE_GROUPS["all_phase27_features"] = tuple(dict.fromkeys(
    item for values in FEATURE_GROUPS.values() for item in values
))


def _structure_arrays(data):
    candles = tuple(StructureCandle(float(t), float(o), float(h), float(l), float(c), float(v))
                    for t, o, h, l, c, v in zip(*(data[name] for name in
                    ("close_timestamp", "open", "high", "low", "close", "volume"))))
    structures = compute_market_structure(candles)
    numeric = [item.numeric() for item in structures]
    names = set().union(*(row.keys() for row in numeric))
    arrays = {name: np.asarray([row.get(name) if row.get(name) is not None else np.nan
                               for row in numeric], dtype=float) for name in names}
    arrays["latest_high_label"] = np.asarray([{"HH": 1.0, "LH": -1.0}.get(x.latest_high_label, 0.0) for x in structures])
    arrays["latest_low_label"] = np.asarray([{"HL": 1.0, "LL": -1.0}.get(x.latest_low_label, 0.0) for x in structures])
    return structures, arrays


def build_phase27_features(root: Path):
    t, o, h, l, c, v, features = build_features(root)
    structures, arrays = _structure_arrays(dict(zip(
        ("close_timestamp", "open", "high", "low", "close", "volume"), (t, o, h, l, c, v))))
    features.update(arrays)
    for timeframe, suffix in (("15m", "15m"), ("1h", "1h")):
        data = _load(root, timeframe)
        _, higher = _structure_arrays(data)
        positions = np.searchsorted(data["close_timestamp"], t, side="right") - 1
        valid = positions >= 0
        if np.any(data["close_timestamp"][positions[valid]] > t[valid]):
            raise ValueError("higher-timeframe close leakage")
        for source, target in (("structure_state", f"structure_{suffix}"),
                               ("bullish_bos", f"bullish_bos_{suffix}"),
                               ("bearish_bos", f"bearish_bos_{suffix}"),
                               ("bullish_choch", f"bullish_choch_{suffix}"),
                               ("bearish_choch", f"bearish_choch_{suffix}")):
            values = np.full(len(t), np.nan)
            values[valid] = higher[source][positions[valid]]
            features[target] = values
    htf_available = np.isfinite(features["structure_15m"]) & np.isfinite(features["structure_1h"])
    base_dip = features["dip_score"].copy(); base_rally = features["rally_score"].copy()
    features["dip_score"][:] = np.nan; features["rally_score"][:] = np.nan
    usable_dip = htf_available & np.isfinite(base_dip)
    usable_rally = htf_available & np.isfinite(base_rally)
    bullish_context = ((features["structure_15m"] == 1).astype(float) +
                       (features["structure_1h"] == 1).astype(float)) / 2
    bearish_context = ((features["structure_15m"] == -1).astype(float) +
                       (features["structure_1h"] == -1).astype(float)) / 2
    features["dip_score"][usable_dip] = .7 * base_dip[usable_dip] + .3 * bullish_context[usable_dip]
    features["rally_score"][usable_rally] = .7 * base_rally[usable_rally] + .3 * bearish_context[usable_rally]
    assert_development_only(t, "Phase 2.7 feature construction")
    return t, o, h, l, c, v, features, structures


def legacy_simple_dip_actions(indices, close):
    """Explicit causal reconstruction of root DipTracker DNA for comparison."""
    actions = []
    dip = None
    holding = False
    for index in indices:
        price = float(close[index])
        dip = price if dip is None else min(dip, price)
        if not holding and price <= dip * 1.001:
            actions.append("BUY"); holding = True
        elif holding and price >= dip * 1.01:
            actions.append("SELL"); holding = False; dip = price
        else:
            actions.append("WAIT")
    return actions


def _fit_fold(fold, names, train, validation, test, labels, future, features,
              timestamps, dataset_id, o, h, l, c):
    model = LogisticRegressionBaseline(metadata_for(
        "logistic_regression", dataset_id, "phase27", SCHEMA, fold,
        {"feature_group": ",".join(names)}), random_state=SEED)
    model.fit(_samples(_cap(train, 60000), labels, future, features, timestamps, dataset_id, names))
    model.calibrate(_samples(_cap(validation, 30000), labels, future, features, timestamps, dataset_id, names))
    validation_rows = _cap(validation, 30000)
    probabilities = model.predict_probability(_samples(validation_rows, labels, future, features, timestamps, dataset_id, names))
    curve = threshold_curve(probabilities, labels[validation_rows].tolist(), validation_rows, timestamps, o, h, l, c)
    threshold = select_validation_threshold(curve)
    probabilities = model.predict_probability(_samples(test, labels, future, features, timestamps, dataset_id, names))
    policy = PolicyThresholds(threshold, threshold, .1)
    actions = [decide(item, policy) for item in probabilities]
    prediction = asdict(evaluate_predictions(probabilities, labels[test].tolist(), policy))
    costs = {name: _summary(_portfolio(test, actions, timestamps, o, h, l, c, name)) for name in COSTS}
    return threshold, actions, prediction, costs


def _exit_portfolio(indices, actions, t, o, h, l, c, mode, cost):
    if mode == "hybrid":
        return _portfolio(indices, actions, t, o, h, l, c, cost)
    config = replace(_config(cost), take_profit_pct=1_000_000.0)
    return simulate_portfolio(_bars(indices, t, o, h, l, c), actions, config)


def run(root: Path, dataset_id: str):
    started = time.time()
    # This unchanged run supplies the locked Phase 2.5 LR/GB/STATIC baselines and
    # its correctly refitted source-gap sensitivity rather than reimplementing it.
    phase25 = run_phase25(root, dataset_id)
    t, o, h, l, c, _, features, structures = build_phase27_features(root)
    if float(t.max()) >= END:
        raise ValueError("2026 data is INVALID_CONTAMINATED")
    future = np.full(len(c), np.nan)
    future[:-HORIZON] = (c[HORIZON:] / c[:-HORIZON] - 1) * 100
    target_available = np.arange(len(t)) + HORIZON < len(t)
    quality_manifest = json.loads((root / "dataset_manifests" / f"{dataset_id}.json").read_text())
    excluded_months = {row["period"] for row in quality_manifest["monthly_builds"]
                       if row["anomaly_classification"] != "NONE"}
    month_keys = np.asarray([datetime.fromtimestamp(float(value), timezone.utc).strftime("%Y-%m")
                             for value in t])
    preregistration = {
        "schema": SCHEMA, "dataset_id": dataset_id, "range": [START, END],
        "folds": FOLDS, "purge_seconds": 3600, "embargo_seconds": 3600,
        "feature_groups": FEATURE_GROUPS, "costs": COSTS,
        "baselines": ["WAIT", "ALWAYS_LONG", "LEGACY_SIMPLE_DIP", "PHASE25_LOGISTIC_REGRESSION", "PHASE25_GRADIENT_BOOSTING", "STATIC_BRIAN_META", "MARKET_STRUCTURE_ONLY", "DIP_ONLY", "STRUCTURE_CLASSIC", "STRUCTURE_TECHNICAL_MTF"],
        "execution": "SHADOW_RESEARCH_ONLY",
        "holdout": {"status": "INVALID_CONTAMINATED", "evaluation_allowed": False},
        "declaration": DECLARATION,
    }
    preregistration_id = canonical_hash(preregistration)
    directory = root / "phase27"; directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / f"preregistered-{preregistration_id}.json"
    receipt.write_text(json.dumps({"preregistration_id": preregistration_id, **preregistration}, sort_keys=True, separators=(",", ":")) + chr(10))
    results = []; primary = []; curves = {}
    for fold, (start, train_end, validation_end, test_end) in enumerate(FOLDS, 1):
        train = np.where(target_available & (t >= ts(start)) & (t < ts(train_end) - 3600))[0]
        validation = np.where(target_available & (t >= ts(train_end) + 3600) & (t < ts(validation_end) - 3600))[0]
        test = np.where(target_available & (t >= ts(validation_end) + 3600) & (t < ts(test_end)))[0]
        neutral = float(np.quantile(np.abs(future[train]), .33))
        labels = np.where(future > neutral, 1, np.where(future < -neutral, -1, 0))
        for group, names in FEATURE_GROUPS.items():
            available = tuple(name for name in names if name in features)
            finite = np.all(np.column_stack([np.isfinite(features[name]) for name in available]), axis=1)
            tr, va, te = train[finite[train]], validation[finite[validation]], test[finite[test]]
            threshold, actions, prediction, costs = _fit_fold(
                fold, available, tr, va, te, labels, future, features, t, dataset_id, o, h, l, c)
            row = {"fold": fold, "model": group, "features": available,
                   "threshold": threshold, "prediction": prediction, "cost_sensitivity": costs}
            results.append(row)
            if group == "all_phase27_features":
                portfolio = _portfolio(te, actions, t, o, h, l, c)
                primary.append((row, te, actions)); curves[f"fold{fold}"] = portfolio
                for mode in ("structure", "hybrid"):
                    overlaid = apply_structure_exit(actions, [structures[i] for i in te], StructureExitConfig(mode))
                    results.append({"fold": fold, "model": f"all_phase27_features_{mode}_exit",
                                    "cost_sensitivity": {cost: _summary(_exit_portfolio(te, overlaid, t, o, h, l, c, mode, cost)) for cost in COSTS}})
        baselines = {"WAIT": ["WAIT"] * len(test), "ALWAYS_LONG": ["BUY"] * len(test),
                     "LEGACY_SIMPLE_DIP": legacy_simple_dip_actions(test, c)}
        baselines["STATIC_BRIAN_META"], _ = static_brian_actions(test, t, c, features)
        baselines["MARKET_STRUCTURE_SPECIALIST"] = [
            market_structure_specialist(structures[i]).action for i in test]
        baselines["DIP_SPECIALIST"] = [dip_specialist(structures[i]).action for i in test]
        for name, actions in baselines.items():
            results.append({"fold": fold, "model": name,
                            "cost_sensitivity": {cost: _summary(_portfolio(test, actions, t, o, h, l, c, cost)) for cost in COSTS}})
    fold_metrics = []
    for row, _, _ in primary:
        base = row["cost_sensitivity"]["BASE"]
        fold_metrics.append({"trades": base["entries"], "expectancy": base["expectancy"],
                             "profit_factor": base["profit_factor"] or 0.0,
                             "max_drawdown_pct": base["max_drawdown_pct"]})
    candidate = development_candidate(
        fold_metrics, coverage=sum(x[0]["prediction"]["coverage"] for x in primary) / len(primary),
        calibration_samples=sum(len(x[1]) for x in primary),
        stress_net_pnl=sum(x[0]["cost_sensitivity"]["STRESS"]["net_pnl"] for x in primary),
        policy=EvidencePolicy())
    all_names = FEATURE_GROUPS["all_phase27_features"]
    all_finite = np.all(np.column_stack([np.isfinite(features[name]) for name in all_names]), axis=1)
    source_gap_sensitivity = []
    for fold, (start, train_end, validation_end, test_end) in enumerate(FOLDS, 1):
        clean = ~np.isin(month_keys, list(excluded_months))
        train = np.where(target_available & all_finite & clean & (t >= ts(start)) & (t < ts(train_end) - 3600))[0]
        validation = np.where(target_available & all_finite & clean & (t >= ts(train_end) + 3600) & (t < ts(validation_end) - 3600))[0]
        test_all = np.where(target_available & all_finite & (t >= ts(validation_end) + 3600) & (t < ts(test_end)))[0]
        test = test_all[clean[test_all]]
        neutral = float(np.quantile(np.abs(future[train]), .33))
        labels = np.where(future > neutral, 1, np.where(future < -neutral, -1, 0))
        threshold, actions, prediction, costs = _fit_fold(
            fold + 200, all_names, train, validation, test, labels, future, features,
            t, dataset_id, o, h, l, c)
        action_by_index = dict(zip(test.tolist(), actions))
        all_actions = [action_by_index.get(index, "WAIT") for index in test_all]
        sensitivity_costs = {cost: _summary(_portfolio_excluding_months(
            test_all, all_actions, t, o, h, l, c, excluded_months, cost)) for cost in COSTS}
        excluded = _portfolio_excluding_months(test_all, all_actions, t, o, h, l, c, excluded_months)
        source_gap_sensitivity.append({
            "fold": fold, "excluded_months": sorted(excluded_months),
            "refit_preprocessing": True, "refit_model": True, "recalibrated": True,
            "validation_threshold_reselected": True, "forced_flat_at_boundaries": True,
            "selected_threshold": threshold, "prediction": prediction,
            "cost_sensitivity": sensitivity_costs, "base_portfolio": _summary(excluded),
        })
    yearly_robustness = []
    year_groups = [datetime.fromtimestamp(float(value), timezone.utc).year for value in t]
    for split in purged_temporal_yearly_splits(t.tolist(), year_groups,
                                                purge_seconds=3600, embargo_seconds=3600):
        train = np.asarray([i for i in split.train_indices
                            if target_available[i] and all_finite[i]], dtype=int)
        test = np.asarray([i for i in split.test_indices
                           if target_available[i] and all_finite[i]], dtype=int)
        ordered = train[np.argsort(t[train])]
        cut = max(1, int(len(ordered) * .8))
        fit, validation = ordered[:cut], ordered[cut:]
        neutral = float(np.quantile(np.abs(future[fit]), .33))
        labels = np.where(future > neutral, 1, np.where(future < -neutral, -1, 0))
        threshold, actions, prediction, costs = _fit_fold(
            300 + split.test_groups[0], all_names, fit, validation, test, labels,
            future, features, t, dataset_id, o, h, l, c)
        yearly_robustness.append({
            "split_id": split.split_id, "test_groups": split.test_groups,
            "threshold": threshold, "prediction": prediction, "cost_sensitivity": costs,
        })
    regime_report = []
    for (row, indices, actions), (key, portfolio) in zip(primary, curves.items()):
        entry_pnl = {trade.entry_timestamp: trade.net_pnl for trade in portfolio.trades}
        for state in ("UPTREND", "DOWNTREND", "RANGE", "TRANSITION", "UNKNOWN"):
            positions = [j for j, index in enumerate(indices) if structures[index].state == state]
            timestamps = [float(t[indices[j]]) for j in positions]
            pnls = [entry_pnl[value] for value in timestamps if value in entry_pnl]
            regime_report.append({
                "model_fold": key, "regime": state, "observations": len(positions),
                "signals": sum(actions[j] != "WAIT" for j in positions),
                "entries": len(pnls), "expectancy": sum(pnls) / len(pnls) if pnls else 0.0,
                "wait_rate": sum(actions[j] == "WAIT" for j in positions) / len(positions) if positions else 0.0,
                "status": "SUFFICIENT" if len(pnls) >= 40 else "INSUFFICIENT_SAMPLE",
            })
        atr_values = np.asarray([structures[index].atr if structures[index].atr is not None else np.nan
                                 for index in indices])
        median_atr = float(np.nanmedian(atr_values))
        extra_regimes = {
            "HIGH_VOLATILITY": [j for j, value in enumerate(atr_values) if np.isfinite(value) and value > median_atr],
            "LOW_VOLATILITY": [j for j, value in enumerate(atr_values) if np.isfinite(value) and value <= median_atr],
            "POST_SWEEP": [j for j, index in enumerate(indices) if structures[index].bullish_sweep or structures[index].bearish_sweep],
            "POST_BOS": [j for j, index in enumerate(indices) if structures[index].bullish_bos or structures[index].bearish_bos],
            "POST_CHOCH": [j for j, index in enumerate(indices) if structures[index].bullish_choch or structures[index].bearish_choch],
        }
        for regime, positions in extra_regimes.items():
            timestamps = [float(t[indices[j]]) for j in positions]
            pnls = [entry_pnl[value] for value in timestamps if value in entry_pnl]
            regime_report.append({
                "model_fold": key, "regime": regime, "observations": len(positions),
                "signals": sum(actions[j] != "WAIT" for j in positions), "entries": len(pnls),
                "expectancy": sum(pnls) / len(pnls) if pnls else 0.0,
                "wait_rate": sum(actions[j] == "WAIT" for j in positions) / len(positions) if positions else 0.0,
                "status": "SUFFICIENT" if len(pnls) >= 40 else "INSUFFICIENT_SAMPLE",
            })
    curve_dir = directory / "equity_curves"; curve_dir.mkdir(exist_ok=True)
    equity = {}
    for key, portfolio in curves.items():
        path = curve_dir / f"{key}.parquet"
        pq.write_table(pa.Table.from_pylist([asdict(point) for point in portfolio.equity_curve]), path, compression="zstd")
        equity[key] = {"path": str(path), "rows": len(portfolio.equity_curve)}
    manifest = {
        "schema_version": SCHEMA,
        "experiment_id": canonical_hash({"preregistration": preregistration_id, "started": int(started)}),
        "preregistration_id": preregistration_id, "dataset_id": dataset_id,
        "code_version": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "date_range": {"start": START, "end_exclusive": END, "max_observed_timestamp": float(t.max())},
        "results": results, "ablations": [row for row in results if row["model"] in FEATURE_GROUPS],
        "phase25_baseline_experiment_id": phase25["experiment_id"],
        "phase25_baselines": [row for row in phase25["results"] if row["model"] in
                              ("logistic_regression", "gradient_boosting", "STATIC_BRIAN_META")],
        "source_gap_sensitivity": source_gap_sensitivity,
        "phase25_source_gap_reference": phase25["quality"]["sensitivity"],
        "regime_report": regime_report,
        "calibration": [{"fold": row["fold"], **row["prediction"]} for row, _, _ in primary],
        "purged_temporal_yearly_robustness": yearly_robustness,
        "equity_curves": equity, "candidate_decision": candidate,
        "market_structure_summary": {"rows": len(structures), "confirmed_swings": sum(len(x.confirmed_swings) for x in structures)},
        "execution": "SHADOW_RESEARCH_ONLY", "historical_order_book": "UNAVAILABLE",
        "historical_bid_ask": "UNAVAILABLE; spread/slippage are SIMULATION ASSUMPTIONS",
        "holdout": {"status": "INVALID_CONTAMINATED", "evaluation_allowed": False},
        "declaration": DECLARATION, "runtime_seconds": time.time() - started,
    }
    path = directory / f"{manifest['experiment_id']}.json"
    path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + chr(10))
    print(json.dumps({"experiment_id": manifest["experiment_id"], "candidate_decision": candidate,
                      "max_timestamp": float(t.max()), "manifest": str(path)}))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="research_data")
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args(argv)
    run(Path(args.root), args.dataset_id)


if __name__ == "__main__":
    main()
