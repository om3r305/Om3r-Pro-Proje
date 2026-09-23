from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .phase46_execution_simulator import LiquidityLevel, OrderBookSnapshot
from .phase56_pretrade_risk_engine import InstrumentRiskLimits
from .phase57_shadow_execution_cycle import ExecutionMarketInput
from .phase89_recovery_startup_gate import RecoveryStartupGateReceipt
from .phase92_recovery_worker_session import run_recovery_startup_once_from_env

PHASE93_SCHEMA_VERSION = "brian.phase93-recovery-worker-entrypoint.v1"

EXIT_READY = 0
EXIT_RECOVERY_BLOCKED = 20
EXIT_MANUAL_REVIEW = 21
EXIT_BUDGET_EXHAUSTED = 22
EXIT_INPUT_ERROR = 30
EXIT_WORKER_ERROR = 40


class RecoveryWorkerEntrypointError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RecoveryWorkerInput:
    markets: Mapping[str, ExecutionMarketInput]
    risk_limits_by_asset: Mapping[str, InstrumentRiskLimits]
    marks: Mapping[str, float]
    schema_version: str = PHASE93_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if any(not asset.strip() for asset in self.markets):
            raise ValueError("market asset ids must be non-empty")
        if any(not asset.strip() for asset in self.risk_limits_by_asset):
            raise ValueError("risk-limit asset ids must be non-empty")
        if any(not asset.strip() for asset in self.marks):
            raise ValueError("mark asset ids must be non-empty")
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in self.marks.values()):
            raise ValueError("marks must be finite and positive")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase93 worker input must remain shadow-only")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RecoveryWorkerEntrypointError(f"{label} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _float(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise RecoveryWorkerEntrypointError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RecoveryWorkerEntrypointError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise RecoveryWorkerEntrypointError(f"{label} must be finite")
    if positive and result <= 0:
        raise RecoveryWorkerEntrypointError(f"{label} must be positive")
    return result


def _positive_int(
    value: object,
    label: str,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise RecoveryWorkerEntrypointError(f"{label} must be integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RecoveryWorkerEntrypointError(f"{label} must be integer") from exc
    if result < minimum:
        raise RecoveryWorkerEntrypointError(
            f"{label} must be at least {minimum}"
        )
    if maximum is not None and result > maximum:
        raise RecoveryWorkerEntrypointError(
            f"{label} must be at most {maximum}"
        )
    return result


def _level(value: object, label: str) -> LiquidityLevel:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RecoveryWorkerEntrypointError(
            f"{label} must be [price, quantity]"
        )
    return LiquidityLevel(
        price=_float(value[0], f"{label}.price", positive=True),
        quantity=_float(value[1], f"{label}.quantity"),
    )


def _snapshot(value: object, label: str) -> OrderBookSnapshot:
    row = _mapping(value, label)
    bids_raw = row.get("bids")
    asks_raw = row.get("asks")
    if not isinstance(bids_raw, (list, tuple)) or not bids_raw:
        raise RecoveryWorkerEntrypointError(f"{label}.bids must be a non-empty array")
    if not isinstance(asks_raw, (list, tuple)) or not asks_raw:
        raise RecoveryWorkerEntrypointError(f"{label}.asks must be a non-empty array")
    return OrderBookSnapshot(
        timestamp=_float(row.get("timestamp"), f"{label}.timestamp"),
        bids=tuple(
            _level(item, f"{label}.bids[{index}]")
            for index, item in enumerate(bids_raw)
        ),
        asks=tuple(
            _level(item, f"{label}.asks[{index}]")
            for index, item in enumerate(asks_raw)
        ),
    )


def _market(asset_id: str, value: object) -> ExecutionMarketInput:
    row = _mapping(value, f"markets.{asset_id}")
    snapshots_raw = row.get("snapshots")
    if not isinstance(snapshots_raw, (list, tuple)) or not snapshots_raw:
        raise RecoveryWorkerEntrypointError(
            f"markets.{asset_id}.snapshots must be a non-empty array"
        )
    return ExecutionMarketInput(
        reference_price=_float(
            row.get("reference_price"),
            f"markets.{asset_id}.reference_price",
            positive=True,
        ),
        tick_size=_float(
            row.get("tick_size"),
            f"markets.{asset_id}.tick_size",
            positive=True,
        ),
        snapshots=tuple(
            _snapshot(item, f"markets.{asset_id}.snapshots[{index}]")
            for index, item in enumerate(snapshots_raw)
        ),
    )


def _limits(asset_id: str, value: object) -> InstrumentRiskLimits:
    row = _mapping(value, f"risk_limits_by_asset.{asset_id}")
    max_notional = row.get("max_notional")
    max_per_order = row.get("max_notional_per_order")
    return InstrumentRiskLimits(
        min_notional=_float(
            row.get("min_notional", 0.0),
            f"risk_limits_by_asset.{asset_id}.min_notional",
        ),
        max_notional=(
            None
            if max_notional is None
            else _float(
                max_notional,
                f"risk_limits_by_asset.{asset_id}.max_notional",
                positive=True,
            )
        ),
        max_notional_per_order=(
            None
            if max_per_order is None
            else _float(
                max_per_order,
                f"risk_limits_by_asset.{asset_id}.max_notional_per_order",
                positive=True,
            )
        ),
    )


def parse_worker_input(payload: Mapping[str, object]) -> RecoveryWorkerInput:
    root = _mapping(payload, "worker input")
    allowed = {"markets", "risk_limits_by_asset", "marks"}
    unknown = sorted(set(root) - allowed)
    if unknown:
        raise RecoveryWorkerEntrypointError(
            "unknown worker input fields: " + ",".join(unknown)
        )

    markets_raw = root.get("markets", {})
    limits_raw = root.get("risk_limits_by_asset", {})
    marks_raw = root.get("marks", {})
    markets_map = _mapping(markets_raw, "markets")
    limits_map = _mapping(limits_raw, "risk_limits_by_asset")
    marks_map = _mapping(marks_raw, "marks")

    return RecoveryWorkerInput(
        markets={
            asset: _market(asset, value)
            for asset, value in markets_map.items()
        },
        risk_limits_by_asset={
            asset: _limits(asset, value)
            for asset, value in limits_map.items()
        },
        marks={
            asset: _float(value, f"marks.{asset}", positive=True)
            for asset, value in marks_map.items()
        },
    )


def _load_json_input(path: str | None, stdin: TextIO) -> Mapping[str, object]:
    if path is not None:
        raw = Path(path).read_text(encoding="utf-8")
    else:
        if hasattr(stdin, "isatty") and stdin.isatty():
            raw = "{}"
        else:
            raw = stdin.read()
            if not raw.strip():
                raw = "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RecoveryWorkerEntrypointError(
            f"worker input is invalid JSON: line {exc.lineno} column {exc.colno}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RecoveryWorkerEntrypointError(
            "worker input root must be a JSON object"
        )
    return payload


def _summary(receipt: RecoveryStartupGateReceipt) -> dict[str, object]:
    return {
        "schema_version": PHASE93_SCHEMA_VERSION,
        "runtime_id": receipt.runtime_id,
        "status": receipt.status,
        "ready_for_normal_work": receipt.ready_for_normal_work,
        "processed_items": receipt.processed_items,
        "max_items": receipt.max_items,
        "admission": {
            "status": receipt.admission.status,
            "blocked": receipt.admission.blocked,
            "original_cycle_id": receipt.admission.original_cycle_id,
            "cancel_risk_receipt_id": receipt.admission.cancel_risk_receipt_id,
            "reason": receipt.admission.reason,
        },
        "step_outcomes": [step.outcome for step in receipt.steps],
        "shadow_only": True,
        "live_execution": False,
    }


def _exit_code(receipt: RecoveryStartupGateReceipt) -> int:
    if receipt.ready_for_normal_work:
        return EXIT_READY
    if receipt.status == "RECOVERY_BUDGET_EXHAUSTED":
        return EXIT_BUDGET_EXHAUSTED
    if any(
        step.outcome == "MANUAL_REVIEW_REQUIRED"
        for step in receipt.steps
    ):
        return EXIT_MANUAL_REVIEW
    return EXIT_RECOVERY_BLOCKED


def _error_payload(kind: str, message: str) -> dict[str, object]:
    return {
        "schema_version": PHASE93_SCHEMA_VERSION,
        "status": kind,
        "ready_for_normal_work": False,
        "error": message[:500],
        "shadow_only": True,
        "live_execution": False,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    worker_runner=run_recovery_startup_once_from_env,
    clock=time.time,
) -> int:
    parser = argparse.ArgumentParser(
        description="Brian shadow recovery one-shot backend worker"
    )
    parser.add_argument(
        "--input",
        help="JSON file with markets/risk_limits_by_asset/marks; defaults to stdin",
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--claim-seconds", type=int, default=None)
    parser.add_argument("--intent-ttl-seconds", type=int, default=None)
    parser.add_argument("--worker-token", default=None)
    parser.add_argument("--source-ref", default="phase93:backend-worker")
    args = parser.parse_args(argv)

    source = os.environ if env is None else env
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    error_stream = sys.stderr if stderr is None else stderr

    try:
        max_items = _positive_int(
            args.max_items
            if args.max_items is not None
            else source.get("BRIAN_RECOVERY_MAX_ITEMS", "4"),
            "max_items",
            minimum=1,
            maximum=32,
        )
        claim_seconds = _positive_int(
            args.claim_seconds
            if args.claim_seconds is not None
            else source.get("BRIAN_RECOVERY_CLAIM_SECONDS", "30"),
            "claim_seconds",
            minimum=10,
            maximum=300,
        )
        intent_ttl = _positive_int(
            args.intent_ttl_seconds
            if args.intent_ttl_seconds is not None
            else source.get("BRIAN_RECOVERY_INTENT_TTL_SECONDS", "60"),
            "intent_ttl_seconds",
            minimum=10,
            maximum=900,
        )
        worker_token = (
            str(args.worker_token).strip()
            if args.worker_token is not None
            else source.get("BRIAN_RECOVERY_WORKER_TOKEN", "").strip()
        )
        if not worker_token:
            worker_token = f"phase93-{uuid.uuid4().hex}"

        worker_input = parse_worker_input(
            _load_json_input(args.input, input_stream)
        )
        receipt = worker_runner(
            max_items=max_items,
            recovery_worker_token=worker_token,
            recovery_claim_seconds=claim_seconds,
            recovery_markets=worker_input.markets,
            recovery_risk_limits_by_asset=worker_input.risk_limits_by_asset,
            recovery_ttl_seconds=intent_ttl,
            marks=worker_input.marks,
            observed_at=float(clock()),
            source_ref=str(args.source_ref),
            env=source,
        )
        payload = _summary(receipt)
        print(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
            file=output_stream,
            flush=True,
        )
        return _exit_code(receipt)
    except (
        RecoveryWorkerEntrypointError,
        ValueError,
        TypeError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        payload = _error_payload("INPUT_ERROR", str(exc))
        print(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
            file=error_stream,
            flush=True,
        )
        return EXIT_INPUT_ERROR
    except Exception as exc:
        # Runtime, lease, transport and durable recovery failures are all
        # machine-visible hard failures. Never serialize repr/traceback because
        # transport exceptions may chain objects that contain secret headers.
        payload = _error_payload("WORKER_ERROR", f"{type(exc).__name__}: {str(exc)}")
        print(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
            file=error_stream,
            flush=True,
        )
        return EXIT_WORKER_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
