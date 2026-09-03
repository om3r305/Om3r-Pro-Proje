from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence
import argparse
import json
import math
import os
import tempfile

from .archive import ArchiveManifest, ArchiveSpec, BinanceArchiveAdapter, derive_timeframe, parse_archive
from .counterfactual_learner import CausalCounterfactualLearner
from .curriculum_runner import PolicyObservation
from .data import NormalizedKline, canonical_hash
from .market_gym import GymEpisodeResult, MarketGym, MarketGymConfig
from .phase35_training import PHASE35_GYM_CONFIG, PHASE35_SYMBOLS, PHASE35_TIMEFRAME
from .portfolio import DEVELOPMENT_CUTOFF
from .profit_mode import FrozenNativePolicy, ProfitSeekingShadowPolicy, policy_contract
from .state_fingerprint import portable_state_fingerprint
from .world_model import HistoricalBar, MultiAssetHistory, align_history_intersection

PHASE36_SCHEMA_VERSION = "brian.phase36-frozen-profit-exam.v1"
PHASE36_EXAM_MONTHS: tuple[tuple[int, int], ...] = ((2024, 3), (2024, 4))
PHASE36_SYMBOLS = PHASE35_SYMBOLS
PHASE36_TIMEFRAME = PHASE35_TIMEFRAME
PHASE35_SOURCE_RUN_ID = 33766345728
PHASE35_SOURCE_ARTIFACT = "brian-phase35-training-33766345728"
PHASE35_EXPECTED_PORTABLE_FINGERPRINT = "b534b611543fcf449a371faad208be20ccf7782343996d08b2bd554ed7f720b9"
PHASE35_EXPECTED_RAW_STATE_ID = "de90c35af3525d591f17e2489e64e9c5ebd84f8124e344927d7c829623688d36"
COST_STRESS_MULTIPLIERS: tuple[float, ...] = (1.0, 1.5, 2.0)
MIN_ACTIVE_STEPS = 20


def _write_json(payload: Mapping[str, object], output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".phase36-", suffix=".json", dir=target.parent)
    os.close(descriptor)
    try:
        Path(temporary).write_bytes(content)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def _period(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def _history_from_records(records_by_symbol: Mapping[str, Sequence[NormalizedKline]], provenance: Mapping[str, Sequence[str]], *, label: str) -> MultiAssetHistory:
    if tuple(sorted(records_by_symbol)) != tuple(sorted(PHASE36_SYMBOLS)):
        raise ValueError("Phase 3.6 requires exactly the locked five-asset universe")
    if set(provenance) != set(records_by_symbol):
        raise ValueError("every Phase 3.6 symbol requires verified provenance")
    series: dict[str, tuple[HistoricalBar, ...]] = {}
    identity: list[dict[str, object]] = []
    for symbol in sorted(records_by_symbol):
        rows = tuple(sorted(records_by_symbol[symbol], key=lambda row: row.exchange_open_timestamp))
        if len(rows) < 3:
            raise ValueError(f"insufficient Phase 3.6 rows for {symbol}")
        if any(row.instrument.symbol != symbol or row.timeframe != PHASE36_TIMEFRAME for row in rows):
            raise ValueError("Phase 3.6 rows do not match locked symbol/timeframe")
        if any(row.exchange_close_timestamp >= DEVELOPMENT_CUTOFF for row in rows):
            raise ValueError("2026 INVALID_CONTAMINATED data is forbidden")
        if any(row.observed_timestamp < row.exchange_close_timestamp for row in rows):
            raise ValueError("Phase 3.6 cannot expose a candle before close")
        if any(right.exchange_open_timestamp <= left.exchange_open_timestamp for left, right in zip(rows, rows[1:])):
            raise ValueError("Phase 3.6 rows must be chronological and unique")
        series[symbol] = tuple(HistoricalBar(symbol, float(row.exchange_close_timestamp), float(row.open), float(row.high), float(row.low), float(row.close), float(row.volume)) for row in rows)
        identity.append({"symbol": symbol, "row_count": len(rows), "first_close": rows[0].exchange_close_timestamp, "last_close": rows[-1].exchange_close_timestamp, "raw_ids_hash": canonical_hash([row.raw_id for row in rows]), "provenance_hash": canonical_hash(tuple(provenance[symbol]))})
    history = MultiAssetHistory.from_mapping(canonical_hash({"schema": PHASE36_SCHEMA_VERSION, "label": label, "rows": identity}), series)
    if len(align_history_intersection(history)) < 3:
        raise ValueError("Phase 3.6 has insufficient exact timestamp overlap")
    return history


def fetch_verified_exam_histories(root: str | Path, *, adapter: BinanceArchiveAdapter | None = None) -> tuple[dict[str, MultiAssetHistory], tuple[ArchiveManifest, ...], dict[str, dict[str, int]]]:
    root = Path(root)
    source = adapter or BinanceArchiveAdapter()
    manifests: list[ArchiveManifest] = []
    month_histories: dict[str, MultiAssetHistory] = {}
    month_counts: dict[str, dict[str, int]] = {}
    combined_rows: dict[str, list[NormalizedKline]] = {symbol: [] for symbol in PHASE36_SYMBOLS}
    combined_provenance: dict[str, list[str]] = {symbol: [] for symbol in PHASE36_SYMBOLS}
    for year, month in PHASE36_EXAM_MONTHS:
        if year >= 2026:
            raise ValueError("invalid Phase 3.6 exam declaration")
        label = _period(year, month)
        rows_by_symbol: dict[str, tuple[NormalizedKline, ...]] = {}
        provenance: dict[str, tuple[str, ...]] = {}
        counts: dict[str, int] = {}
        for symbol in PHASE36_SYMBOLS:
            spec = ArchiveSpec(symbol, "1m", year, month)
            manifest = source.fetch(spec, root)
            if not manifest.checksum_verified or manifest.symbol != symbol:
                raise ValueError("Phase 3.6 archive must be checksum verified")
            normalized = parse_archive(spec, manifest.archive_path, manifest)
            if normalized.report.status == "REJECTED" or not normalized.records:
                raise ValueError(f"Phase 3.6 normalization rejected for {symbol} {label}")
            derived = derive_timeframe(normalized.records, PHASE36_TIMEFRAME)
            if not derived:
                raise ValueError(f"no canonical {PHASE36_TIMEFRAME} rows for {symbol} {label}")
            manifests.append(manifest)
            rows_by_symbol[symbol] = derived
            provenance[symbol] = (manifest.provenance_id,)
            counts[symbol] = len(derived)
            combined_rows[symbol].extend(derived)
            combined_provenance[symbol].append(manifest.provenance_id)
        month_histories[label] = _history_from_records(rows_by_symbol, provenance, label=label)
        month_counts[label] = counts
    month_histories["combined"] = _history_from_records({symbol: tuple(rows) for symbol, rows in combined_rows.items()}, {symbol: tuple(rows) for symbol, rows in combined_provenance.items()}, label="2024-03..2024-04")
    return month_histories, tuple(manifests), month_counts


def _metrics(result: GymEpisodeResult) -> dict[str, object]:
    deltas = [float(step.equity_after - step.equity_before) for step in result.trace]
    gains = sum(value for value in deltas if value > 0)
    losses = -sum(value for value in deltas if value < 0)
    pf = gains / losses if losses > 1e-12 else None
    gross = [sum(abs(weight) for _, weight in step.target_weights) for step in result.trace]
    active = sum(value > 1e-12 for value in gross)
    if not all(math.isfinite(value) for value in deltas + gross):
        raise ValueError("Phase 3.6 produced non-finite metrics")
    return {"return_pct": float(result.return_pct), "ending_equity": float(result.ending_equity), "max_drawdown_pct": float(result.max_drawdown_pct), "total_virtual_trading_cost": float(result.total_costs), "total_virtual_turnover": float(result.total_turnover), "rebalance_count": int(result.rebalance_count), "steps": int(result.steps), "active_steps": int(active), "mean_gross_exposure": mean(gross) if gross else 0.0, "net_step_profit_factor": pf, "ruined": bool(result.ruined), "terminal_reason": result.terminal_reason}


def _evaluate(history: MultiAssetHistory, checkpoint: Mapping[str, object], *, mode: str, gym_config: MarketGymConfig) -> dict[str, object]:
    learner = CausalCounterfactualLearner.from_state(dict(checkpoint))
    before_raw = learner.training_state_id
    before_portable = portable_state_fingerprint(learner.state_dict())
    policy = FrozenNativePolicy(learner) if mode == "native" else ProfitSeekingShadowPolicy(learner, gym_config)
    frames = align_history_intersection(history)
    gym = MarketGym(frames, gym_config)
    while not gym.terminated:
        left = max(0, gym.index - learner.config.lookback + 1)
        visible = tuple(frames[left : gym.index + 1])
        obs = PolicyObservation(visible, gym.equity, tuple(sorted(gym.weights.items())), len(visible) - 1, gym_config.starting_equity)
        gym.step(policy.act(obs))
    result = gym.finish()
    if learner.training_state_id != before_raw or portable_state_fingerprint(learner.state_dict()) != before_portable:
        raise ValueError("Phase 3.6 mutated the frozen checkpoint")
    return {"policy": policy_contract(policy), "episode_id": result.episode_id, "metrics": _metrics(result)}


def _stress_config(multiplier: float) -> MarketGymConfig:
    if multiplier <= 0:
        raise ValueError("cost stress multiplier must be positive")
    return replace(PHASE35_GYM_CONFIG, fee_bps=PHASE35_GYM_CONFIG.fee_bps * multiplier, assumed_spread_bps=PHASE35_GYM_CONFIG.assumed_spread_bps * multiplier, slippage_bps=PHASE35_GYM_CONFIG.slippage_bps * multiplier)


def candidate_gate(monthly_profit: Mapping[str, Mapping[str, object]], combined_native: Mapping[str, object], combined_stress: Mapping[str, Mapping[str, object]]) -> tuple[str, dict[str, bool]]:
    base = combined_stress["1.0x"]["metrics"]
    stress15 = combined_stress["1.5x"]["metrics"]
    pf = base["net_step_profit_factor"]
    checks = {
        "both_reserved_months_net_positive": all(float(monthly_profit[label]["metrics"]["return_pct"]) > 0.0 for label in ("2024-03", "2024-04")),
        "combined_net_positive": float(base["return_pct"]) > 0.0,
        "combined_profit_factor_above_one": (pf is None and float(base["return_pct"]) > 0.0) or (pf is not None and float(pf) > 1.0),
        "combined_max_drawdown_at_most_10pct": float(base["max_drawdown_pct"]) <= 10.0,
        "combined_active_steps_at_least_20": int(base["active_steps"]) >= MIN_ACTIVE_STEPS,
        "profit_mode_not_worse_than_native": float(base["return_pct"]) >= float(combined_native["metrics"]["return_pct"]) - 1e-12,
        "one_point_five_x_cost_stress_positive": float(stress15["return_pct"]) > 0.0,
        "no_virtual_ruin": not bool(base["ruined"]),
    }
    return ("DEVELOPMENT_CANDIDATE" if all(checks.values()) else "INSUFFICIENT_EVIDENCE"), checks


def run_exam(checkpoint_path: str | Path, root: str | Path = "research_data", output: str | Path = "cloud_results/phase36_frozen_profit_exam.json", *, adapter: BinanceArchiveAdapter | None = None) -> dict[str, object]:
    checkpoint = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    learner = CausalCounterfactualLearner.from_state(checkpoint)
    raw_id = learner.training_state_id
    portable = portable_state_fingerprint(checkpoint)
    if raw_id != PHASE35_EXPECTED_RAW_STATE_ID:
        raise ValueError("Phase 3.6 checkpoint raw state does not match locked Phase 3.5 artifact")
    if portable != PHASE35_EXPECTED_PORTABLE_FINGERPRINT:
        raise ValueError("Phase 3.6 checkpoint portable fingerprint does not match locked Phase 3.5 artifact")
    histories, manifests, row_counts = fetch_verified_exam_histories(root, adapter=adapter)
    monthly_native: dict[str, dict[str, object]] = {}
    monthly_profit: dict[str, dict[str, object]] = {}
    for label in ("2024-03", "2024-04"):
        monthly_native[label] = _evaluate(histories[label], checkpoint, mode="native", gym_config=PHASE35_GYM_CONFIG)
        monthly_profit[label] = _evaluate(histories[label], checkpoint, mode="profit", gym_config=PHASE35_GYM_CONFIG)
    combined_native = _evaluate(histories["combined"], checkpoint, mode="native", gym_config=PHASE35_GYM_CONFIG)
    combined_stress: dict[str, dict[str, object]] = {}
    for multiplier in COST_STRESS_MULTIPLIERS:
        combined_stress[f"{multiplier:.1f}x"] = _evaluate(histories["combined"], checkpoint, mode="profit", gym_config=_stress_config(multiplier))
    decision, checks = candidate_gate(monthly_profit, combined_native, combined_stress)
    aligned = align_history_intersection(histories["combined"])
    archive_rows = tuple({"symbol": m.symbol, "archive_period": m.archive_period, "checksum_verified": m.checksum_verified, "checksum_value": m.checksum_value, "content_hash": m.content_hash, "provenance_id": m.provenance_id} for m in sorted(manifests, key=lambda row: (row.archive_period, row.symbol)))
    summary = {
        "schema_version": PHASE36_SCHEMA_VERSION,
        "declaration": "FROZEN_PROFIT_SEEKING_SHADOW_DEVELOPMENT_EXAM",
        "scientific_interpretation": "PREREGISTERED_RESERVED_DEVELOPMENT_EXAM_NOT_FINAL_HOLDOUT",
        "source_training_artifact": {"run_id": PHASE35_SOURCE_RUN_ID, "artifact_name": PHASE35_SOURCE_ARTIFACT, "raw_training_state_id": raw_id, "portable_training_state_fingerprint": portable, "episodes_learned": int(learner.training_manifest()["episodes_learned"])},
        "exam_dataset": {"source_months": ("2024-03", "2024-04"), "symbols": PHASE36_SYMBOLS, "timeframe": PHASE36_TIMEFRAME, "monthly_derived_row_counts": row_counts, "combined_dataset_id": histories["combined"].dataset_id, "combined_aligned_frame_count": len(aligned), "max_source_timestamp": max(frame.timestamp for frame in aligned), "archives": archive_rows, "learning_enabled_during_exam": False},
        "base_gym_config": asdict(PHASE35_GYM_CONFIG),
        "monthly": {label: {"native": monthly_native[label], "profit_seeking_shadow": monthly_profit[label]} for label in ("2024-03", "2024-04")},
        "combined": {"frozen_native": combined_native, "profit_seeking_cost_stress": combined_stress},
        "candidate_decision": decision,
        "candidate_gate_checks": checks,
        "contaminated_final_holdout": {"status": "INVALID_CONTAMINATED", "year": 2026, "evaluation_allowed": False, "pristine_final_holdout_evaluated": False},
        "automatic_promotion": False,
        "live_execution": False,
        "shadow_only": True,
    }
    if float(summary["exam_dataset"]["max_source_timestamp"]) >= DEVELOPMENT_CUTOFF:
        raise ValueError("Phase 3.6 contains forbidden 2026 data")
    if portable_state_fingerprint(checkpoint) != portable:
        raise ValueError("Phase 3.6 checkpoint changed during exam")
    _write_json(summary, output)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Brian Phase 3.6 frozen profit-seeking development exam")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default="research_data")
    parser.add_argument("--output", default="cloud_results/phase36_frozen_profit_exam.json")
    args = parser.parse_args(argv)
    run_exam(args.checkpoint, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
