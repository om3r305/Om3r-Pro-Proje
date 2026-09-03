from __future__ import annotations

from dataclasses import asdict
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
from .phase33_learning_smoke import SMOKE_GYM_CONFIG, SMOKE_SYMBOLS, run_smoke
from .portfolio import DEVELOPMENT_CUTOFF
from .profit_mode import FrozenNativePolicy, ProfitSeekingShadowPolicy, policy_contract
from .world_model import HistoricalBar, MultiAssetHistory, align_history_intersection

PHASE34_SCHEMA_VERSION = "brian.phase34-profit-exam.v1"
EXAM_YEAR = 2024
EXAM_MONTH = 2
EXAM_TIMEFRAME = "5m"
EXAM_SYMBOLS: tuple[str, ...] = SMOKE_SYMBOLS
EXAM_GYM_CONFIG: MarketGymConfig = SMOKE_GYM_CONFIG


def _write_json(payload: Mapping[str, object], output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".phase34-profit-", suffix=".json", dir=target.parent)
    os.close(descriptor)
    try:
        Path(temporary).write_bytes(content)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def history_from_verified_month(
    records_by_symbol: Mapping[str, Sequence[NormalizedKline]],
    provenance_ids: Mapping[str, str],
    *,
    year: int,
    month: int,
) -> MultiAssetHistory:
    if tuple(sorted(records_by_symbol)) != tuple(sorted(EXAM_SYMBOLS)):
        raise ValueError("Phase 3.4 exam requires exactly the locked BTC/ETH/SOL universe")
    if set(provenance_ids) != set(records_by_symbol):
        raise ValueError("every exam symbol requires verified provenance")
    if year >= 2026:
        raise ValueError("2026 historical development data is INVALID_CONTAMINATED and forbidden")

    series: dict[str, tuple[HistoricalBar, ...]] = {}
    identity_rows: list[dict[str, object]] = []
    for symbol in sorted(records_by_symbol):
        rows = tuple(sorted(records_by_symbol[symbol], key=lambda row: row.exchange_open_timestamp))
        if len(rows) < 3:
            raise ValueError(f"insufficient verified {EXAM_TIMEFRAME} rows for {symbol}")
        if any(row.instrument.symbol != symbol or row.timeframe != EXAM_TIMEFRAME for row in rows):
            raise ValueError("exam rows must match the locked symbol/timeframe")
        if any(row.exchange_close_timestamp >= DEVELOPMENT_CUTOFF for row in rows):
            raise ValueError("2026 observations are INVALID_CONTAMINATED and forbidden")
        if any(row.observed_timestamp < row.exchange_close_timestamp for row in rows):
            raise ValueError("exam history cannot expose a candle before it closes")
        if any(right.exchange_open_timestamp <= left.exchange_open_timestamp for left, right in zip(rows, rows[1:])):
            raise ValueError("exam rows must be strictly chronological")

        bars = tuple(
            HistoricalBar(
                asset_id=symbol,
                timestamp=float(row.exchange_close_timestamp),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
            for row in rows
        )
        series[symbol] = bars
        identity_rows.append({
            "symbol": symbol,
            "provenance_id": str(provenance_ids[symbol]),
            "row_count": len(rows),
            "first_close_timestamp": rows[0].exchange_close_timestamp,
            "last_close_timestamp": rows[-1].exchange_close_timestamp,
            "derived_raw_ids_hash": canonical_hash([row.raw_id for row in rows]),
        })

    dataset_id = canonical_hash({
        "schema_version": PHASE34_SCHEMA_VERSION,
        "source": "verified-binance-vision-monthly",
        "year": int(year),
        "month": int(month),
        "timeframe": EXAM_TIMEFRAME,
        "rows": identity_rows,
    })
    history = MultiAssetHistory.from_mapping(dataset_id, series)
    aligned = align_history_intersection(history)
    if len(aligned) < 3:
        raise ValueError("verified exam history has insufficient exact timestamp overlap")
    return history


def fetch_verified_exam_history(
    root: str | Path,
    *,
    adapter: BinanceArchiveAdapter | None = None,
) -> tuple[MultiAssetHistory, tuple[ArchiveManifest, ...], dict[str, int]]:
    root = Path(root)
    source = adapter or BinanceArchiveAdapter()
    manifests: list[ArchiveManifest] = []
    derived_by_symbol: dict[str, tuple[NormalizedKline, ...]] = {}
    row_counts: dict[str, int] = {}

    for symbol in EXAM_SYMBOLS:
        spec = ArchiveSpec(symbol, "1m", EXAM_YEAR, EXAM_MONTH)
        manifest = source.fetch(spec, root)
        if not manifest.checksum_verified or manifest.symbol != symbol:
            raise ValueError("exam archive must be checksum-verified and match its symbol")
        normalized = parse_archive(spec, manifest.archive_path, manifest)
        if normalized.report.status == "REJECTED" or not normalized.records:
            raise ValueError(f"verified exam archive normalization rejected for {symbol}")
        derived = derive_timeframe(normalized.records, EXAM_TIMEFRAME)
        if not derived:
            raise ValueError(f"no canonical {EXAM_TIMEFRAME} rows derived for {symbol}")
        manifests.append(manifest)
        derived_by_symbol[symbol] = derived
        row_counts[symbol] = len(derived)

    history = history_from_verified_month(
        derived_by_symbol,
        {manifest.symbol: manifest.provenance_id for manifest in manifests},
        year=EXAM_YEAR,
        month=EXAM_MONTH,
    )
    return history, tuple(manifests), row_counts


def _quarter_returns(result: GymEpisodeResult) -> tuple[float, ...]:
    trace = result.trace
    if not trace:
        return ()
    boundaries = [0, len(trace) // 4, len(trace) // 2, (3 * len(trace)) // 4, len(trace)]
    out: list[float] = []
    for left, right in zip(boundaries, boundaries[1:]):
        if right <= left:
            continue
        start_equity = float(trace[left].equity_before)
        end_equity = float(trace[right - 1].equity_after)
        out.append((end_equity / max(start_equity, 1e-12) - 1.0) * 100.0)
    return tuple(out)


def episode_metrics(result: GymEpisodeResult) -> dict[str, object]:
    deltas = [float(step.equity_after - step.equity_before) for step in result.trace]
    gains = sum(value for value in deltas if value > 0)
    losses = -sum(value for value in deltas if value < 0)
    profit_factor = gains / losses if losses > 1e-12 else None
    quarter_returns = _quarter_returns(result)
    gross_exposures = [sum(abs(weight) for _, weight in step.target_weights) for step in result.trace]
    active_steps = sum(value > 1e-12 for value in gross_exposures)
    if not all(math.isfinite(value) for value in deltas + list(quarter_returns) + gross_exposures):
        raise ValueError("exam metrics contain non-finite values")
    return {
        "return_pct": float(result.return_pct),
        "ending_equity": float(result.ending_equity),
        "max_drawdown_pct": float(result.max_drawdown_pct),
        "total_virtual_trading_cost": float(result.total_costs),
        "total_virtual_turnover": float(result.total_turnover),
        "rebalance_count": int(result.rebalance_count),
        "steps": int(result.steps),
        "active_steps": int(active_steps),
        "mean_gross_exposure": mean(gross_exposures) if gross_exposures else 0.0,
        "net_step_profit_factor": profit_factor,
        "chronological_quarter_returns_pct": quarter_returns,
        "positive_quarters": sum(value > 0 for value in quarter_returns),
        "negative_quarters": sum(value < 0 for value in quarter_returns),
        "ruined": bool(result.ruined),
        "terminal_reason": result.terminal_reason,
        "shadow_only": True,
    }


def evaluate_frozen_policy(
    history: MultiAssetHistory,
    learner: CausalCounterfactualLearner,
    policy: object,
    *,
    gym_config: MarketGymConfig = EXAM_GYM_CONFIG,
) -> tuple[GymEpisodeResult, dict[str, object]]:
    frames = align_history_intersection(history)
    gym = MarketGym(frames, gym_config)
    learner_state_before = learner.training_state_id

    while not gym.terminated:
        left = max(0, gym.index - learner.config.lookback + 1)
        visible = tuple(frames[left : gym.index + 1])
        observation = PolicyObservation(
            visible_frames=visible,
            equity=gym.equity,
            current_weights=tuple(sorted(gym.weights.items())),
            step_index=len(visible) - 1,
            starting_equity=gym_config.starting_equity,
        )
        allocation = policy.act(observation)
        gym.step(allocation)

    result = gym.finish()
    if learner.training_state_id != learner_state_before:
        raise ValueError("Phase 3.4 exam mutated the frozen learner checkpoint")
    return result, episode_metrics(result)


def candidate_gate(native: Mapping[str, object], profit: Mapping[str, object]) -> tuple[str, dict[str, bool]]:
    profit_factor = profit.get("net_step_profit_factor")
    return_pct = float(profit["return_pct"])
    checks = {
        "net_return_positive": return_pct > 0.0,
        "net_profit_factor_above_one": (
            (profit_factor is None and return_pct > 0.0)
            or (profit_factor is not None and float(profit_factor) > 1.0)
        ),
        "max_drawdown_at_most_10pct": float(profit["max_drawdown_pct"]) <= 10.0,
        "at_least_two_positive_chronological_quarters": int(profit["positive_quarters"]) >= 2,
        "not_worse_than_frozen_native_return": return_pct >= float(native["return_pct"]) - 1e-12,
        "no_virtual_ruin": not bool(profit["ruined"]),
    }
    decision = "DEVELOPMENT_CANDIDATE" if all(checks.values()) else "INSUFFICIENT_EVIDENCE"
    return decision, checks


def run_exam(
    root: str | Path = "research_data",
    output: str | Path = "cloud_results/phase34_profit_exam.json",
    training_output: str | Path = "cloud_results/phase34_training_smoke.json",
    checkpoint_output: str | Path = "cloud_results/phase34_training_checkpoint.json",
) -> dict[str, object]:
    training = run_smoke(root, training_output, checkpoint_output)
    checkpoint = json.loads(Path(checkpoint_output).read_text(encoding="utf-8"))
    native_learner = CausalCounterfactualLearner.from_state(checkpoint)
    profit_learner = CausalCounterfactualLearner.from_state(checkpoint)
    checkpoint_id = native_learner.training_state_id
    if checkpoint_id != profit_learner.training_state_id:
        raise ValueError("frozen exam learners did not start from the same checkpoint")
    if checkpoint_id != str(training["learner"]["training_state_id"]):
        raise ValueError("Phase 3.4 training checkpoint differs from Phase 3.3 smoke receipt")

    exam_history, manifests, row_counts = fetch_verified_exam_history(root)
    native_policy = FrozenNativePolicy(native_learner)
    profit_policy = ProfitSeekingShadowPolicy(profit_learner, EXAM_GYM_CONFIG)
    native_result, native_metrics = evaluate_frozen_policy(exam_history, native_learner, native_policy)
    profit_result, profit_metrics = evaluate_frozen_policy(exam_history, profit_learner, profit_policy)
    decision, checks = candidate_gate(native_metrics, profit_metrics)

    aligned = align_history_intersection(exam_history)
    archive_rows = tuple(
        {
            "symbol": manifest.symbol,
            "archive_period": manifest.archive_period,
            "checksum_verified": manifest.checksum_verified,
            "checksum_value": manifest.checksum_value,
            "content_hash": manifest.content_hash,
            "provenance_id": manifest.provenance_id,
        }
        for manifest in sorted(manifests, key=lambda row: row.symbol)
    )
    summary = {
        "schema_version": PHASE34_SCHEMA_VERSION,
        "declaration": "PROFIT_SEEKING_SHADOW_DEVELOPMENT_EXAM",
        "scientific_interpretation": "TRAINING_EXCLUDED_DEVELOPMENT_DIAGNOSTIC_NOT_FINAL_HOLDOUT",
        "training": {
            "source_month": "2024-01",
            "training_state_id": checkpoint_id,
            "episodes_learned": int(training["learner"]["manifest"]["episodes_learned"]),
            "training_only": True,
        },
        "exam_dataset": {
            "dataset_id": exam_history.dataset_id,
            "source_month": f"{EXAM_YEAR:04d}-{EXAM_MONTH:02d}",
            "symbols": EXAM_SYMBOLS,
            "timeframe": EXAM_TIMEFRAME,
            "derived_row_counts": dict(sorted(row_counts.items())),
            "aligned_frame_count": len(aligned),
            "max_source_timestamp": max(frame.timestamp for frame in aligned),
            "archives": archive_rows,
            "learning_enabled_during_exam": False,
        },
        "gym_config": asdict(EXAM_GYM_CONFIG),
        "frozen_native": {
            "policy": policy_contract(native_policy),
            "episode_id": native_result.episode_id,
            "metrics": native_metrics,
        },
        "profit_seeking_shadow": {
            "policy": policy_contract(profit_policy),
            "episode_id": profit_result.episode_id,
            "metrics": profit_metrics,
        },
        "candidate_decision": decision,
        "candidate_gate_checks": checks,
        "holdout": {
            "status": "INVALID_CONTAMINATED",
            "evaluation_allowed": False,
            "pristine_final_holdout_evaluated": False,
        },
        "automatic_promotion": False,
        "live_execution": False,
        "shadow_only": True,
    }
    if float(summary["exam_dataset"]["max_source_timestamp"]) >= DEVELOPMENT_CUTOFF:
        raise ValueError("Phase 3.4 exam contains INVALID_CONTAMINATED 2026 data")
    if native_learner.training_state_id != checkpoint_id or profit_learner.training_state_id != checkpoint_id:
        raise ValueError("exam changed the frozen learner checkpoint")
    _write_json(summary, output)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Brian Phase 3.4 profit-seeking shadow development exam")
    parser.add_argument("--root", default="research_data")
    parser.add_argument("--output", default="cloud_results/phase34_profit_exam.json")
    parser.add_argument("--training-output", default="cloud_results/phase34_training_smoke.json")
    parser.add_argument("--checkpoint-output", default="cloud_results/phase34_training_checkpoint.json")
    args = parser.parse_args(argv)
    run_exam(args.root, args.output, args.training_output, args.checkpoint_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
