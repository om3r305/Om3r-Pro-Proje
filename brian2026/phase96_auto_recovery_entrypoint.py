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

from .phase94_binance_spot_recovery_evidence import (
    BinanceSpotRecoveryEvidenceProvider,
)
from .phase95_auto_binance_recovery_worker import (
    AutoRecoveryStartupReceipt,
    run_one_auto_binance_recovery_from_env,
)

PHASE96_SCHEMA_VERSION = "brian.phase96-auto-recovery-entrypoint.v1"

EXIT_READY = 0
EXIT_RECOVERY_BLOCKED = 20
EXIT_MANUAL_REVIEW = 21
EXIT_BUDGET_EXHAUSTED = 22
EXIT_INPUT_ERROR = 30
EXIT_WORKER_ERROR = 40

_ALLOWED_DEPTH_LIMITS = {5, 10, 20, 50, 100, 500, 1000, 5000}


class AutoRecoveryEntrypointError(RuntimeError):
    pass


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise AutoRecoveryEntrypointError(
            f"invalid auto-recovery arguments: {message}"
        )


def _positive_int(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        raise AutoRecoveryEntrypointError(f"{label} must be integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise AutoRecoveryEntrypointError(f"{label} must be integer") from exc
    if result < minimum or result > maximum:
        raise AutoRecoveryEntrypointError(
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
        raise AutoRecoveryEntrypointError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AutoRecoveryEntrypointError(f"{label} must be numeric") from exc
    if (
        not math.isfinite(result)
        or result < minimum
        or result > maximum
    ):
        raise AutoRecoveryEntrypointError(
            f"{label} must be finite and in [{minimum},{maximum}]"
        )
    return result


def _summary(receipt: AutoRecoveryStartupReceipt) -> dict[str, object]:
    gate = receipt.gate
    return {
        "schema_version": PHASE96_SCHEMA_VERSION,
        "runtime_id": gate.runtime_id,
        "status": gate.status,
        "ready_for_normal_work": gate.ready_for_normal_work,
        "processed_items": gate.processed_items,
        "max_items": gate.max_items,
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
        "admission": {
            "status": gate.admission.status,
            "blocked": gate.admission.blocked,
            "original_cycle_id": gate.admission.original_cycle_id,
            "cancel_risk_receipt_id": gate.admission.cancel_risk_receipt_id,
            "reason": gate.admission.reason,
        },
        "step_outcomes": [step.outcome for step in gate.steps],
        "shadow_only": True,
        "live_execution": False,
    }


def _exit_code(receipt: AutoRecoveryStartupReceipt) -> int:
    gate = receipt.gate
    if gate.ready_for_normal_work:
        return EXIT_READY
    if gate.status == "RECOVERY_BUDGET_EXHAUSTED":
        return EXIT_BUDGET_EXHAUSTED
    if any(
        step.outcome == "MANUAL_REVIEW_REQUIRED"
        for step in gate.steps
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
        "schema_version": PHASE96_SCHEMA_VERSION,
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
    worker_runner=run_one_auto_binance_recovery_from_env,
    clock=time.time,
) -> int:
    source = os.environ if env is None else env
    output_stream = sys.stdout if stdout is None else stdout
    error_stream = sys.stderr if stderr is None else stderr

    try:
        parser = _JsonArgumentParser(
            description=(
                "Brian one-shot shadow recovery worker with public "
                "Binance Spot evidence"
            )
        )
        parser.add_argument("--claim-seconds", type=int, default=None)
        parser.add_argument("--intent-ttl-seconds", type=int, default=None)
        parser.add_argument("--worker-token", default=None)
        parser.add_argument(
            "--source-ref",
            default="phase96:auto-binance-recovery",
        )
        parser.add_argument("--depth-limit", type=int, default=None)
        parser.add_argument("--max-spread-bps", type=float, default=None)
        parser.add_argument("--market-timeout-seconds", type=float, default=None)
        parser.add_argument("--max-assets", type=int, default=None)
        args = parser.parse_args(argv)

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
            raise AutoRecoveryEntrypointError(
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
            worker_token = f"phase96-{uuid.uuid4().hex}"
        source_ref = str(args.source_ref).strip()
        if not source_ref:
            raise AutoRecoveryEntrypointError("source_ref is required")
    except (AutoRecoveryEntrypointError, ValueError, TypeError) as exc:
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
        receipt = worker_runner(
            recovery_worker_token=worker_token,
            recovery_claim_seconds=claim_seconds,
            recovery_ttl_seconds=intent_ttl,
            source_ref=source_ref,
            env=source,
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
