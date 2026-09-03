from __future__ import annotations

from dataclasses import asdict
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
from .world_model import BrianWorldModel, HistoricalBar, MultiAssetHistory, WorldModelConfig, align_history_intersection

SMOKE_SCHEMA_VERSION = "brian.phase33-learning-smoke.v1"
SMOKE_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
SMOKE_YEAR = 2024
SMOKE_MONTH = 1
SMOKE_TIMEFRAME = "5m"
SMOKE_PLAN = CurriculumPlan(
    real_replay_episodes=25,
    block_bootstrap_episodes=50,
    stress_bootstrap_episodes=25,
    shard_size=50,
)
SMOKE_WORLD_CONFIG = WorldModelConfig(
    horizon_steps=128,
    block_length=32,
    seed=3301,
    stress_return_scale=1.75,
)
SMOKE_GYM_CONFIG = MarketGymConfig(
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
    descriptor, temporary = tempfile.mkstemp(prefix=".phase33-smoke-", suffix=".json", dir=target.parent)
    os.close(descriptor)
    try:
        Path(temporary).write_bytes(content)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def history_from_derived(
    records_by_symbol: Mapping[str, Sequence[NormalizedKline]],
    provenance_ids: Mapping[str, str],
) -> MultiAssetHistory:
    if tuple(sorted(records_by_symbol)) != tuple(sorted(SMOKE_SYMBOLS)):
        raise ValueError("Phase 3.3 smoke requires exactly the locked BTC/ETH/SOL universe")
    if set(provenance_ids) != set(records_by_symbol):
        raise ValueError("every smoke symbol requires verified provenance")

    series: dict[str, tuple[HistoricalBar, ...]] = {}
    identity_rows: list[dict[str, object]] = []
    for symbol in sorted(records_by_symbol):
        rows = tuple(sorted(records_by_symbol[symbol], key=lambda row: row.exchange_open_timestamp))
        if len(rows) < SMOKE_WORLD_CONFIG.horizon_steps + 1:
            raise ValueError(f"insufficient verified {SMOKE_TIMEFRAME} rows for {symbol}")
        if any(row.instrument.symbol != symbol or row.timeframe != SMOKE_TIMEFRAME for row in rows):
            raise ValueError("smoke history rows must match the locked symbol/timeframe")
        if any(row.exchange_close_timestamp >= DEVELOPMENT_CUTOFF for row in rows):
            raise ValueError("2026 data is INVALID_CONTAMINATED and forbidden")
        if any(row.observed_timestamp < row.exchange_close_timestamp for row in rows):
            raise ValueError("smoke history cannot contain observations before candle close")
        if any(right.exchange_open_timestamp <= left.exchange_open_timestamp for left, right in zip(rows, rows[1:])):
            raise ValueError("smoke history rows must be strictly chronological")

        bars = tuple(
            HistoricalBar(
                asset_id=symbol,
                # Decisions occur only after the completed candle is available. Using the
                # close timestamp here prevents the Gym from treating the bar as known at open.
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
        "schema_version": SMOKE_SCHEMA_VERSION,
        "source": "verified-binance-vision-monthly",
        "year": SMOKE_YEAR,
        "month": SMOKE_MONTH,
        "timeframe": SMOKE_TIMEFRAME,
        "rows": identity_rows,
    })
    history = MultiAssetHistory.from_mapping(dataset_id, series)
    aligned = align_history_intersection(history)
    if len(aligned) < SMOKE_WORLD_CONFIG.horizon_steps + 1:
        raise ValueError("verified BTC/ETH/SOL history has insufficient exact timestamp overlap")
    return history


def fetch_verified_smoke_history(
    root: str | Path,
    *,
    adapter: BinanceArchiveAdapter | None = None,
) -> tuple[MultiAssetHistory, tuple[ArchiveManifest, ...], dict[str, int]]:
    root = Path(root)
    source = adapter or BinanceArchiveAdapter()
    manifests: list[ArchiveManifest] = []
    derived_by_symbol: dict[str, tuple[NormalizedKline, ...]] = {}
    row_counts: dict[str, int] = {}

    for symbol in SMOKE_SYMBOLS:
        spec = ArchiveSpec(symbol, "1m", SMOKE_YEAR, SMOKE_MONTH)
        manifest = source.fetch(spec, root)
        if not manifest.checksum_verified or manifest.symbol != symbol:
            raise ValueError("smoke archive must be checksum-verified and match its symbol")
        normalized = parse_archive(spec, manifest.archive_path, manifest)
        if normalized.report.status == "REJECTED" or not normalized.records:
            raise ValueError(f"verified archive normalization rejected for {symbol}")
        derived = derive_timeframe(normalized.records, SMOKE_TIMEFRAME)
        if not derived:
            raise ValueError(f"no canonical {SMOKE_TIMEFRAME} rows derived for {symbol}")
        manifests.append(manifest)
        derived_by_symbol[symbol] = derived
        row_counts[symbol] = len(derived)

    history = history_from_derived(
        derived_by_symbol,
        {manifest.symbol: manifest.provenance_id for manifest in manifests},
    )
    return history, tuple(manifests), row_counts


def _aggregate_episode_diagnostics(memories: Sequence[object]) -> dict[str, object]:
    summaries = tuple(summary for memory in memories for summary in memory.summaries)
    if not summaries:
        raise ValueError("smoke curriculum produced no episode summaries")
    returns = [float(row.return_pct) for row in summaries]
    drawdowns = [float(row.max_drawdown_pct) for row in summaries]
    costs = [float(row.total_costs) for row in summaries]
    turnovers = [float(row.total_turnover) for row in summaries]
    if not all(math.isfinite(value) for value in returns + drawdowns + costs + turnovers):
        raise ValueError("smoke diagnostics contain non-finite values")
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
        "first_10_mean_return_pct": mean(returns[:10]),
        "last_10_mean_return_pct": mean(returns[-10:]),
        "world_modes": dict(sorted(modes.items())),
        "training_only": True,
    }


def run_smoke(
    root: str | Path = "research_data",
    output: str | Path = "cloud_results/phase33_learning_smoke.json",
    checkpoint_output: str | Path = "cloud_results/phase33_learner_checkpoint.json",
    *,
    adapter: BinanceArchiveAdapter | None = None,
) -> dict[str, object]:
    history, manifests, row_counts = fetch_verified_smoke_history(root, adapter=adapter)
    world = BrianWorldModel(history, SMOKE_WORLD_CONFIG)
    learner = CausalCounterfactualLearner()
    runner = CurriculumRunner(
        world,
        plan=SMOKE_PLAN,
        gym_config=SMOKE_GYM_CONFIG,
    )

    shard_results = []
    expected_state: str | None = None
    for shard_index in range(SMOKE_PLAN.shard_count):
        result = runner.run_shard(
            shard_index,
            learner,
            expected_policy_state_in=expected_state,
        )
        if expected_state is not None and result.receipt.policy_state_in != expected_state:
            raise ValueError("smoke learner checkpoint chain broke between shards")
        expected_state = result.receipt.policy_state_out
        shard_results.append(result)

    checkpoint = learner.state_dict()
    checkpoint_id = learner.training_state_id
    _write_json(checkpoint, checkpoint_output)

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
        "schema_version": SMOKE_SCHEMA_VERSION,
        "declaration": "TRAINING_ONLY_SHADOW_SMOKE",
        "scientific_interpretation": "INFRASTRUCTURE_AND_LEARNING_DIAGNOSTIC_NOT_PROFITABILITY_EVIDENCE",
        "dataset": {
            "dataset_id": history.dataset_id,
            "symbols": SMOKE_SYMBOLS,
            "source_month": f"{SMOKE_YEAR:04d}-{SMOKE_MONTH:02d}",
            "timeframe": SMOKE_TIMEFRAME,
            "derived_row_counts": dict(sorted(row_counts.items())),
            "aligned_frame_count": len(world.aligned),
            "max_source_timestamp": max(frame.timestamp for frame in world.aligned),
            "archives": archive_rows,
        },
        "curriculum": {
            "plan": asdict(SMOKE_PLAN),
            "plan_id": SMOKE_PLAN.plan_id,
            "world_config": asdict(SMOKE_WORLD_CONFIG),
            "gym_config": asdict(SMOKE_GYM_CONFIG),
            "shard_receipts": [asdict(result.receipt) for result in shard_results],
        },
        "learner": {
            "training_state_id": checkpoint_id,
            "manifest": learner.training_manifest(),
            "checkpoint_path": str(checkpoint_output),
        },
        "episode_diagnostics": _aggregate_episode_diagnostics([result.memory for result in shard_results]),
        "holdout": {
            "status": "INVALID_CONTAMINATED",
            "evaluation_allowed": False,
            "pristine_final_holdout_evaluated": False,
        },
        "shadow_only": True,
        "automatic_promotion": False,
    }
    max_source = float(summary["dataset"]["max_source_timestamp"])
    if max_source >= DEVELOPMENT_CUTOFF:
        raise ValueError("smoke summary contains INVALID_CONTAMINATED 2026 data")
    if summary["episode_diagnostics"]["episode_count"] != SMOKE_PLAN.total_episodes:
        raise ValueError("smoke curriculum episode count mismatch")
    if shard_results[-1].receipt.policy_state_out != checkpoint_id:
        raise ValueError("final learner checkpoint does not match final shard receipt")
    _write_json(summary, output)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Brian Phase 3.3 verified learning smoke")
    parser.add_argument("--root", default="research_data")
    parser.add_argument("--output", default="cloud_results/phase33_learning_smoke.json")
    parser.add_argument("--checkpoint-output", default="cloud_results/phase33_learner_checkpoint.json")
    args = parser.parse_args(argv)
    run_smoke(args.root, args.output, args.checkpoint_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
