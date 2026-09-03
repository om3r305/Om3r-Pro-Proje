from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Mapping, Sequence
import argparse
import json
import math
import os
import tempfile

from .archive import ArchiveManifest, ArchiveSpec, BinanceArchiveAdapter, derive_timeframe, parse_archive
from .counterfactual_learner import CausalCounterfactualLearner
from .curriculum_runner import CurriculumPlan, CurriculumRunner
from .data import NormalizedKline, canonical_hash
from .market_gym import MarketGymConfig
from .portfolio import DEVELOPMENT_CUTOFF
from .state_fingerprint import portable_state_fingerprint
from .world_model import BrianWorldModel, HistoricalBar, MultiAssetHistory, WorldModelConfig, align_history_intersection

PHASE35_SCHEMA_VERSION = "brian.phase35-1000-life-training.v1"
PHASE35_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
PHASE35_TIMEFRAME = "5m"
PHASE35_TRAINING_MONTHS: tuple[tuple[int, int], ...] = tuple(
    (year, month)
    for year, months in ((2023, range(1, 13)), (2024, range(1, 3)))
    for month in months
)
PHASE35_RESERVED_EXAM_MONTHS: tuple[tuple[int, int], ...] = ((2024, 3), (2024, 4))
PHASE35_RESERVED_EXAM_START = datetime(2024, 3, 1, tzinfo=timezone.utc).timestamp()

PHASE35_PLAN = CurriculumPlan(
    real_replay_episodes=250,
    block_bootstrap_episodes=500,
    stress_bootstrap_episodes=250,
    shard_size=100,
)
PHASE35_WORLD_CONFIG = WorldModelConfig(
    horizon_steps=128,
    block_length=32,
    seed=3501,
    stress_return_scale=1.75,
)
PHASE35_GYM_CONFIG = MarketGymConfig(
    starting_equity=500.0,
    fee_bps=10.0,
    assumed_spread_bps=2.0,
    slippage_bps=1.0,
    max_gross_exposure=1.0,
    max_asset_weight=0.35,
    ruin_fraction=0.01,
    allow_short=True,
)


def _write_json(payload: Mapping[str, object], output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".phase35-training-", suffix=".json", dir=target.parent)
    os.close(descriptor)
    try:
        Path(temporary).write_bytes(content)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def _period_label(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def history_from_verified_range(
    records_by_symbol: Mapping[str, Sequence[NormalizedKline]],
    provenance_by_symbol: Mapping[str, Sequence[str]],
) -> MultiAssetHistory:
    if tuple(sorted(records_by_symbol)) != tuple(sorted(PHASE35_SYMBOLS)):
        raise ValueError("Phase 3.5 training requires exactly the locked five-asset universe")
    if set(provenance_by_symbol) != set(records_by_symbol):
        raise ValueError("every Phase 3.5 asset requires verified provenance receipts")

    series: dict[str, tuple[HistoricalBar, ...]] = {}
    identity_rows: list[dict[str, object]] = []
    for symbol in sorted(records_by_symbol):
        rows = tuple(sorted(records_by_symbol[symbol], key=lambda row: row.exchange_open_timestamp))
        if len(rows) < PHASE35_WORLD_CONFIG.horizon_steps + 1:
            raise ValueError(f"insufficient verified {PHASE35_TIMEFRAME} rows for {symbol}")
        if any(row.instrument.symbol != symbol or row.timeframe != PHASE35_TIMEFRAME for row in rows):
            raise ValueError("Phase 3.5 rows must match the locked symbol/timeframe")
        if any(row.exchange_close_timestamp >= DEVELOPMENT_CUTOFF for row in rows):
            raise ValueError("2026 historical observations are INVALID_CONTAMINATED and forbidden")
        if any(row.exchange_close_timestamp >= PHASE35_RESERVED_EXAM_START for row in rows):
            raise ValueError("reserved March/April 2024 development exam data leaked into training")
        if any(row.observed_timestamp < row.exchange_close_timestamp for row in rows):
            raise ValueError("training history cannot expose a candle before close")
        if any(right.exchange_open_timestamp <= left.exchange_open_timestamp for left, right in zip(rows, rows[1:])):
            raise ValueError("Phase 3.5 training rows must be strictly chronological and unique")

        provenance_ids = tuple(str(value) for value in provenance_by_symbol[symbol])
        if len(provenance_ids) != len(PHASE35_TRAINING_MONTHS) or not all(provenance_ids):
            raise ValueError("each training asset requires one verified provenance receipt per locked month")

        series[symbol] = tuple(
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
        identity_rows.append({
            "symbol": symbol,
            "row_count": len(rows),
            "first_close_timestamp": rows[0].exchange_close_timestamp,
            "last_close_timestamp": rows[-1].exchange_close_timestamp,
            "derived_raw_ids_hash": canonical_hash([row.raw_id for row in rows]),
            "provenance_ids_hash": canonical_hash(provenance_ids),
        })

    dataset_id = canonical_hash({
        "schema_version": PHASE35_SCHEMA_VERSION,
        "source": "verified-binance-vision-monthly",
        "months": [_period_label(year, month) for year, month in PHASE35_TRAINING_MONTHS],
        "timeframe": PHASE35_TIMEFRAME,
        "symbols": PHASE35_SYMBOLS,
        "rows": identity_rows,
    })
    history = MultiAssetHistory.from_mapping(dataset_id, series)
    aligned = align_history_intersection(history)
    if len(aligned) < PHASE35_WORLD_CONFIG.horizon_steps + 1:
        raise ValueError("verified Phase 3.5 history has insufficient exact timestamp overlap")
    if max(frame.timestamp for frame in aligned) >= PHASE35_RESERVED_EXAM_START:
        raise ValueError("aligned training history crossed the preregistered exam boundary")
    return history


def fetch_verified_training_history(
    root: str | Path,
    *,
    adapter: BinanceArchiveAdapter | None = None,
) -> tuple[MultiAssetHistory, tuple[ArchiveManifest, ...], dict[str, int]]:
    root = Path(root)
    source = adapter or BinanceArchiveAdapter()
    manifests: list[ArchiveManifest] = []
    derived_by_symbol: dict[str, tuple[NormalizedKline, ...]] = {}
    provenance_by_symbol: dict[str, tuple[str, ...]] = {}
    row_counts: dict[str, int] = {}

    for symbol in PHASE35_SYMBOLS:
        symbol_rows: list[NormalizedKline] = []
        symbol_provenance: list[str] = []
        for year, month in PHASE35_TRAINING_MONTHS:
            if year >= 2026 or (year, month) in PHASE35_RESERVED_EXAM_MONTHS:
                raise ValueError("invalid Phase 3.5 training month declaration")
            spec = ArchiveSpec(symbol, "1m", year, month)
            manifest = source.fetch(spec, root)
            if not manifest.checksum_verified or manifest.symbol != symbol:
                raise ValueError("Phase 3.5 archives must be checksum-verified and match their symbol")
            normalized = parse_archive(spec, manifest.archive_path, manifest)
            if normalized.report.status == "REJECTED" or not normalized.records:
                raise ValueError(f"verified archive normalization rejected for {symbol} {_period_label(year, month)}")
            derived = derive_timeframe(normalized.records, PHASE35_TIMEFRAME)
            if not derived:
                raise ValueError(f"no canonical {PHASE35_TIMEFRAME} rows for {symbol} {_period_label(year, month)}")
            manifests.append(manifest)
            symbol_rows.extend(derived)
            symbol_provenance.append(manifest.provenance_id)
        derived_by_symbol[symbol] = tuple(symbol_rows)
        provenance_by_symbol[symbol] = tuple(symbol_provenance)
        row_counts[symbol] = len(symbol_rows)

    history = history_from_verified_range(derived_by_symbol, provenance_by_symbol)
    return history, tuple(manifests), row_counts


def _episode_diagnostics(shard_results: Sequence[object]) -> dict[str, object]:
    summaries = tuple(summary for result in shard_results for summary in result.memory.summaries)
    if len(summaries) != PHASE35_PLAN.total_episodes:
        raise ValueError("Phase 3.5 curriculum did not produce exactly 1,000 episode summaries")
    returns = [float(row.return_pct) for row in summaries]
    drawdowns = [float(row.max_drawdown_pct) for row in summaries]
    costs = [float(row.total_costs) for row in summaries]
    turnovers = [float(row.total_turnover) for row in summaries]
    if not all(math.isfinite(value) for value in returns + drawdowns + costs + turnovers):
        raise ValueError("Phase 3.5 episode diagnostics contain non-finite values")
    modes: dict[str, int] = {}
    for row in summaries:
        modes[row.world_mode] = modes.get(row.world_mode, 0) + 1
    return {
        "episode_count": len(summaries),
        "positive_return_episodes": sum(value > 0 for value in returns),
        "negative_return_episodes": sum(value < 0 for value in returns),
        "flat_return_episodes": sum(value == 0 for value in returns),
        "ruin_count": sum(bool(row.ruined) for row in summaries),
        "mean_return_pct": mean(returns),
        "median_return_pct": median(returns),
        "mean_max_drawdown_pct": mean(drawdowns),
        "worst_max_drawdown_pct": max(drawdowns),
        "total_virtual_trading_cost": sum(costs),
        "mean_virtual_turnover": mean(turnovers),
        "first_100_mean_return_pct": mean(returns[:100]),
        "last_100_mean_return_pct": mean(returns[-100:]),
        "world_modes": dict(sorted(modes.items())),
        "training_only": True,
    }


def run_training(
    root: str | Path = "research_data",
    output: str | Path = "cloud_results/phase35_training_summary.json",
    checkpoint_output: str | Path = "cloud_results/phase35_training_checkpoint.json",
    progress_output: str | Path = "cloud_results/phase35_training_progress.json",
    *,
    adapter: BinanceArchiveAdapter | None = None,
) -> dict[str, object]:
    history, manifests, row_counts = fetch_verified_training_history(root, adapter=adapter)
    world = BrianWorldModel(history, PHASE35_WORLD_CONFIG)
    learner = CausalCounterfactualLearner()
    runner = CurriculumRunner(world, plan=PHASE35_PLAN, gym_config=PHASE35_GYM_CONFIG)

    shard_results = []
    expected_state: str | None = None
    for shard_index in range(PHASE35_PLAN.shard_count):
        result = runner.run_shard(shard_index, learner, expected_policy_state_in=expected_state)
        if expected_state is not None and result.receipt.policy_state_in != expected_state:
            raise ValueError("Phase 3.5 learner checkpoint chain broke between shards")
        expected_state = result.receipt.policy_state_out
        shard_results.append(result)
        checkpoint = learner.state_dict()
        _write_json(checkpoint, checkpoint_output)
        _write_json({
            "schema_version": PHASE35_SCHEMA_VERSION,
            "completed_shards": shard_index + 1,
            "total_shards": PHASE35_PLAN.shard_count,
            "episodes_learned": learner.training_manifest()["episodes_learned"],
            "raw_training_state_id": learner.training_state_id,
            "portable_training_state_fingerprint": portable_state_fingerprint(checkpoint),
            "training_only": True,
            "shadow_only": True,
        }, progress_output)

    checkpoint = learner.state_dict()
    raw_state_id = learner.training_state_id
    portable_fingerprint = portable_state_fingerprint(checkpoint)
    _write_json(checkpoint, checkpoint_output)

    aligned = world.aligned
    archive_rows = tuple(
        {
            "symbol": manifest.symbol,
            "archive_period": manifest.archive_period,
            "checksum_verified": manifest.checksum_verified,
            "checksum_value": manifest.checksum_value,
            "content_hash": manifest.content_hash,
            "provenance_id": manifest.provenance_id,
        }
        for manifest in sorted(manifests, key=lambda row: (row.symbol, row.archive_period))
    )
    summary = {
        "schema_version": PHASE35_SCHEMA_VERSION,
        "declaration": "PHASE35_1000_LIFE_TRAINING_ONLY_SHADOW",
        "scientific_interpretation": "TRAINING_DIAGNOSTIC_NOT_PROFITABILITY_EVIDENCE",
        "dataset": {
            "dataset_id": history.dataset_id,
            "symbols": PHASE35_SYMBOLS,
            "source_months": tuple(_period_label(year, month) for year, month in PHASE35_TRAINING_MONTHS),
            "timeframe": PHASE35_TIMEFRAME,
            "derived_row_counts": dict(sorted(row_counts.items())),
            "aligned_frame_count": len(aligned),
            "max_source_timestamp": max(frame.timestamp for frame in aligned),
            "archives": archive_rows,
        },
        "curriculum": {
            "plan": asdict(PHASE35_PLAN),
            "plan_id": PHASE35_PLAN.plan_id,
            "world_config": asdict(PHASE35_WORLD_CONFIG),
            "gym_config": asdict(PHASE35_GYM_CONFIG),
            "shard_receipts": [asdict(result.receipt) for result in shard_results],
        },
        "learner": {
            "raw_training_state_id": raw_state_id,
            "portable_training_state_fingerprint": portable_fingerprint,
            "manifest": learner.training_manifest(),
            "checkpoint_path": str(checkpoint_output),
        },
        "episode_diagnostics": _episode_diagnostics(shard_results),
        "preregistered_next_development_exam": {
            "status": "RESERVED_UNSEEN",
            "source_months": tuple(_period_label(year, month) for year, month in PHASE35_RESERVED_EXAM_MONTHS),
            "learning_allowed": False,
            "evaluation_run_now": False,
        },
        "contaminated_final_holdout": {
            "status": "INVALID_CONTAMINATED",
            "year": 2026,
            "evaluation_allowed": False,
            "pristine_final_holdout_evaluated": False,
        },
        "automatic_promotion": False,
        "live_execution": False,
        "shadow_only": True,
    }
    if float(summary["dataset"]["max_source_timestamp"]) >= PHASE35_RESERVED_EXAM_START:
        raise ValueError("Phase 3.5 summary crossed the reserved March 2024 exam boundary")
    if float(summary["dataset"]["max_source_timestamp"]) >= DEVELOPMENT_CUTOFF:
        raise ValueError("Phase 3.5 summary contains INVALID_CONTAMINATED 2026 data")
    if int(summary["learner"]["manifest"]["episodes_learned"]) != PHASE35_PLAN.total_episodes:
        raise ValueError("Phase 3.5 learner did not finish exactly 1,000 lives")
    if shard_results[-1].receipt.policy_state_out != raw_state_id:
        raise ValueError("final Phase 3.5 checkpoint does not match the final shard receipt")
    _write_json(summary, output)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Brian Phase 3.5 locked 1,000-life shadow training")
    parser.add_argument("--root", default="research_data")
    parser.add_argument("--output", default="cloud_results/phase35_training_summary.json")
    parser.add_argument("--checkpoint-output", default="cloud_results/phase35_training_checkpoint.json")
    parser.add_argument("--progress-output", default="cloud_results/phase35_training_progress.json")
    args = parser.parse_args(argv)
    run_training(args.root, args.output, args.checkpoint_output, args.progress_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
