from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import argparse
import json
import os
import tempfile

from .phase24 import MultiMonthDatasetManifest, build_btc_history
from .phase25_experiment import run as run_phase25_experiment
from .phase27_experiment import run as run_phase27_experiment
from .portfolio import DEVELOPMENT_CUTOFF

CLOUD_SUMMARY_SCHEMA = "brian.cloud-research-summary.v1"
DEVELOPMENT_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
DEVELOPMENT_END = datetime(2026, 1, 1, tzinfo=timezone.utc)
SMOKE_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
SMOKE_END = datetime(2024, 2, 1, tzinfo=timezone.utc)
SUPPORTED_MODES = ("smoke", "full-development", "phase27-development")
HOLDOUT_STATUS = "INVALID_CONTAMINATED"
EXECUTION_DECLARATION = "SHADOW_RESEARCH_ONLY"
FINAL_HOLDOUT_DECLARATION = "NO PRISTINE FINAL HOLDOUT EVALUATED"

DatasetBuilder = Callable[[str | Path, datetime, datetime], MultiMonthDatasetManifest]
ExperimentRunner = Callable[[Path, str], Mapping[str, Any]]


def mode_range(mode: str) -> tuple[datetime, datetime]:
    if mode == "smoke":
        return SMOKE_START, SMOKE_END
    if mode in ("full-development", "phase27-development"):
        return DEVELOPMENT_START, DEVELOPMENT_END
    raise ValueError(f"unsupported cloud research mode: {mode}")


def _validate_range(start: datetime, end: datetime) -> None:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("cloud research ranges must be timezone-aware")
    if start >= end:
        raise ValueError("cloud research range must be increasing")
    if end > DEVELOPMENT_END:
        raise ValueError("requested end exceeds the 2026 development cutoff")


def _validate_dataset(dataset: MultiMonthDatasetManifest, requested_start: datetime,
                      requested_end: datetime) -> None:
    if dataset.symbol != "BTCUSDT" or dataset.exchange != "binance" or dataset.market_type != "spot":
        raise ValueError("cloud research requires Binance Spot BTCUSDT")
    dataset_start = datetime.fromisoformat(dataset.requested_start)
    dataset_end = datetime.fromisoformat(dataset.requested_end)
    if dataset_end > DEVELOPMENT_END:
        raise ValueError("dataset requested end exceeds the 2026 development cutoff")
    if dataset_start != requested_start or dataset_end != requested_end:
        raise ValueError("dataset requested range does not match the fixed cloud mode range")
    if requested_end > DEVELOPMENT_END or dataset.actual_end >= DEVELOPMENT_CUTOFF:
        raise ValueError("dataset contains INVALID_CONTAMINATED 2026 observations")


def build_cloud_summary(mode: str, dataset: MultiMonthDatasetManifest,
                        experiment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    start, end = mode_range(mode)
    _validate_range(start, end)
    _validate_dataset(dataset, start, end)
    summary: dict[str, Any] = {
        "schema_version": CLOUD_SUMMARY_SCHEMA,
        "run_mode": mode,
        "dataset_id": dataset.dataset_id,
        "dataset_quality": dataset.quality_status,
        "requested_range": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "actual_range": {"start": dataset.actual_start, "end": dataset.actual_end},
        "row_counts": dict(dataset.row_counts),
        "monthly_build_count": len(dataset.monthly_builds),
        "development_cutoff": DEVELOPMENT_END.isoformat(),
        "holdout": {"status": HOLDOUT_STATUS, "evaluation_allowed": False},
        "execution_declaration": EXECUTION_DECLARATION,
    }
    if mode in ("full-development", "phase27-development"):
        if experiment is None:
            raise ValueError(f"{mode} requires a development experiment manifest")
        if experiment.get("declaration") != FINAL_HOLDOUT_DECLARATION:
            raise ValueError("full-development experiment lacks the final holdout declaration")
        max_timestamp = float(experiment["date_range"]["max_observed_timestamp"])
        if max_timestamp >= DEVELOPMENT_CUTOFF:
            raise ValueError("experiment contains INVALID_CONTAMINATED 2026 observations")
        summary.update({
            "phase27_experiment_id" if mode == "phase27-development" else "phase25_experiment_id": experiment["experiment_id"],
            "max_observed_timestamp": max_timestamp,
            "candidate_decisions": experiment.get("candidate_decisions", experiment.get("candidate_decision")),
            "final_holdout_declaration": FINAL_HOLDOUT_DECLARATION,
        })
    return summary


def write_cloud_summary(summary: Mapping[str, Any], output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(dict(summary), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8") + bytes((10,))
    descriptor, temporary = tempfile.mkstemp(prefix=".brian-cloud-", suffix=".json",
                                             dir=target.parent)
    os.close(descriptor)
    try:
        Path(temporary).write_bytes(content)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def run_cloud(mode: str, root: str | Path = "research_data",
              output: str | Path = "cloud_results/brian_cloud_summary.json",
              *, dataset_builder: DatasetBuilder = build_btc_history,
              experiment_runner: ExperimentRunner | None = None) -> dict[str, Any]:
    start, end = mode_range(mode)
    _validate_range(start, end)
    dataset = dataset_builder(root, start, end)
    _validate_dataset(dataset, start, end)
    experiment = None
    if mode in ("full-development", "phase27-development"):
        runner = experiment_runner or (run_phase27_experiment if mode == "phase27-development" else run_phase25_experiment)
        experiment = runner(Path(root), dataset.dataset_id)
    summary = build_cloud_summary(mode, dataset, experiment)
    write_cloud_summary(summary, output)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Brian 2026 public cloud research runner")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, required=True)
    parser.add_argument("--root", default="research_data")
    parser.add_argument("--output", default="cloud_results/brian_cloud_summary.json")
    args = parser.parse_args(argv)
    run_cloud(args.mode, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
