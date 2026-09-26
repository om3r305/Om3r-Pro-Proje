from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import TextIO

from .phase92_recovery_worker_session import RecoveryWorkerSession
from .phase94_binance_spot_recovery_evidence import (
    BinanceSpotRecoveryEvidenceProvider,
)
from .phase97_bounded_auto_recovery_drain import (
    BoundedAutoRecoveryDrainReceipt,
    run_bounded_auto_binance_recovery,
)

PHASE98_SCHEMA_VERSION = "brian.phase98-bounded-auto-recovery-entrypoint.v1"

EXIT_READY = 0
EXIT_RECOVERY_BLOCKED = 20
EXIT_MANUAL_REVIEW = 21
EXIT_BUDGET_EXHAUSTED = 22
EXIT_INPUT_ERROR = 30
EXIT_WORKER_ERROR = 40

_ALLOWED_DEPTH_LIMITS = {5, 10, 20, 50, 100, 500, 1000, 5000}


class BoundedAutoRecoveryEntrypointError(RuntimeError):
    pass


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise BoundedAutoRecoveryEntrypointError(
            f"invalid bounded auto-recovery arguments: {message}"
        )


def _positive_int(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        raise BoundedAutoRecoveryEntrypointError(f"{label} must be integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise BoundedAutoRecoveryEntrypointError(
            f"{label} must be integer"
        ) from exc
    if result < minimum or result > maximum:
        raise BoundedAutoRecoveryEntrypointError(
            f"{label} must be in [{minimum},{maximum}]"
        )
    return result


def _positive_float(
    value: object,
    label: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool):
        raise BoundedAutoRecoveryEntrypointError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BoundedAutoRecoveryEntrypointError(
            f"{label} must be numeric"
        ) from exc
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise BoundedAutoRecoveryEntrypointError(
            f"{label} must be finite and in [{minimum},{maximum}]"
        )
    return result


def _attempt_summary(receipt) -> dict[str, object]:
    gate = receipt.gate
    return {
        "status": gate.status,
        "ready_for_normal_work": gate.ready_for_normal_work,
        "processed_items": gate.processed_items,
        "preflight_work_state": receipt.preflight_work_state,
        "original_cycle_id": receipt.original_cycle_id,
        "cancel_risk_receipt_id": receipt.cancel_risk_receipt_id,
        "directive_status": receipt.directive_status,
        "evidence_assets": list(receipt.evidence_assets),
        "evidence_observed_at": {
            asset: observed_at
            for asset, observed_at in receipt.evidence_observed_at
        },
        "decision_at": receipt.decision_at,
        "step_outcomes": [step.outcome for step in gate.steps],
    }


def _summary(receipt: BoundedAutoRecoveryDrainReceipt) -> dict[str, object]:
    return {
        "schema_version": PHASE98_SCHEMA_VERSION,
        "runtime_id": receipt.runtime_id,
        "status": receipt.status,
        "ready_for_normal_work": receipt.ready_for_normal_work,
        "attempt_count": len(receipt.attempts),
        "processed_items": receipt.processed_items,
        "max_items": receipt.max_items,
        "admission": {
            "status": receipt.final_admission.status,
            "blocked": receipt.final_admission.blocked,
            "original_cycle_id": receipt.final_admission.original_cycle_id,
            "cancel_risk_receipt_id": (
                receipt.final_admission.cancel_risk_receipt_id
            ),
            "reason": receipt.final_admission.reason,
        },
        "attempts": [
            _attempt_summary(attempt)
            for attempt in receipt.attempts
        ],
        "shadow_only": True,
        "live_execution": False,
    }


def _exit_code(receipt: BoundedAutoRecoveryDrainReceipt) -> int:
    if receipt.ready_for_normal_work:
        return EXIT_READY
    if receipt.status == "RECOVERY_DRAIN_BUDGET_EXHAUSTED":
        return EXIT_BUDGET_EXHAUSTED
    if any(
        step.outcome == "MANUAL_REVIEW_REQUIRED"
        for attempt in receipt.attempts
        for step in attempt.gate.steps
    ):
        return EXIT_MANUAL_REVIEW
    return EXIT_RECOVERY_BLOCKED


def _safe_worker_error(exc: Exception, env: Mapping[str, str]) -> str:
    message = str(exc)
    secrets: list[str] = []
    for name in ("SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY"):
        value = env.get(name, "").strip()
        if value:
            secrets.append(value)
    raw_keys = env.get("SUPABASE_SECRET_KEYS", "").strip()
    if raw_keys:
        try:
            parsed = json.loads(raw_keys)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            secrets.extend(
                str(value)
                for value in parsed.values()
                if isinstance(value, str) and value
            )
    for secret in sorted(set(secrets), key=len, reverse=True):
        message = message.replace(secret, "<redacted>")
    message = re.sub(
        r"sb_secret_[A-Za-z0-9._-]+",
        "<redacted>",
        message,
    )
    return f"{type(exc).__name__}: {message[:420]}"


def _error_payload(status: str, message: str) -> dict[str, object]:
    return {
        "schema_version": PHASE98_SCHEMA_VERSION,
        "status": status,
        "ready_for_normal_work": False,
        "error": message[:500],
        "shadow_only": True,
        "live_execution": False,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    session_factory=RecoveryWorkerSession.from_env,
    drain_runner=run_bounded_auto_binance_recovery,
    clock=time.time,
) -> int:
    source = os.environ if env is None else env
    output_stream = sys.stdout if stdout is None else stdout
    error_stream = sys.stderr if stderr is None else stderr

    try:
        parser = _JsonArgumentParser(
            description=(
                "Brian bounded shadow recovery drain with causal public "
                "Binance Spot evidence"
            )
        )
        parser.add_argument("--max-items", type=int, default=None)
        parser.add_argument("--claim-seconds", type=int, default=None)
        parser.add_argument("--intent-ttl-seconds", type=int, default=None)
        parser.add_argument("--worker-token", default=None)
        parser.add_argument(
            "--source-ref",
            default="phase98:bounded-auto-binance-recovery",
        )
        parser.add_argument("--depth-limit", type=int, default=None)
        parser.add_argument("--max-spread-bps", type=float, default=None)
        parser.add_argument("--market-timeout-seconds", type=float, default=None)
        parser.add_argument("--max-assets", type=int, default=None)
        args = parser.parse_args(argv)

        max_items = _positive_int(
            args.max_items
            if args.max_items is not None
            else source.get("BRIAN_RECOVERY_MAX_ITEMS", "8"),
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
        depth_limit = _positive_int(
            args.depth_limit
            if args.depth_limit is not None
            else source.get("BRIAN_RECOVERY_BINANCE_DEPTH_LIMIT", "100"),
            "depth_limit",
            minimum=5,
            maximum=5000,
        )
        if depth_limit not in _ALLOWED_DEPTH_LIMITS:
            raise BoundedAutoRecoveryEntrypointError(
                "depth_limit is not a supported Binance depth size"
            )
        max_spread_bps = _positive_float(
            args.max_spread_bps
            if args.max_spread_bps is not None
            else source.get("BRIAN_RECOVERY_BINANCE_MAX_SPREAD_BPS", "30"),
            "max_spread_bps",
            minimum=0.1,
            maximum=500.0,
        )
        market_timeout = _positive_float(
            args.market_timeout_seconds
            if args.market_timeout_seconds is not None
            else source.get("BRIAN_RECOVERY_BINANCE_TIMEOUT_SECONDS", "5.5"),
            "market_timeout_seconds",
            minimum=0.5,
            maximum=20.0,
        )
        max_assets = _positive_int(
            args.max_assets
            if args.max_assets is not None
            else source.get("BRIAN_RECOVERY_BINANCE_MAX_ASSETS", "8"),
            "max_assets",
            minimum=1,
            maximum=32,
        )
        worker_token = (
            str(args.worker_token).strip()
            if args.worker_token is not None
            else source.get("BRIAN_RECOVERY_WORKER_TOKEN", "").strip()
        )
        if not worker_token:
            worker_token = f"phase98-{uuid.uuid4().hex}"
        source_ref = str(args.source_ref).strip()
        if not source_ref:
            raise BoundedAutoRecoveryEntrypointError(
                "source_ref is required"
            )
    except (BoundedAutoRecoveryEntrypointError, ValueError, TypeError) as exc:
        print(
            json.dumps(
                _error_payload("INPUT_ERROR", str(exc)),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            file=error_stream,
            flush=True,
        )
        return EXIT_INPUT_ERROR

    def provider_factory() -> BinanceSpotRecoveryEvidenceProvider:
        return BinanceSpotRecoveryEvidenceProvider(
            timeout_seconds=market_timeout,
            max_assets=max_assets,
            depth_limit=depth_limit,
            max_spread_bps=max_spread_bps,
            clock=clock,
        )

    try:
        with session_factory(env=source) as session:
            receipt = drain_runner(
                session,
                max_items=max_items,
                recovery_worker_token=worker_token,
                recovery_claim_seconds=claim_seconds,
                recovery_ttl_seconds=intent_ttl,
                source_ref=source_ref,
                provider_factory=provider_factory,
                clock=clock,
            )
    except Exception as exc:
        print(
            json.dumps(
                _error_payload(
                    "WORKER_ERROR",
                    _safe_worker_error(exc, source),
                ),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            file=error_stream,
            flush=True,
        )
        return EXIT_WORKER_ERROR

    print(
        json.dumps(
            _summary(receipt),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        file=output_stream,
        flush=True,
    )
    return _exit_code(receipt)


if __name__ == "__main__":
    raise SystemExit(main())
