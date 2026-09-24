from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from .phase112_edge_bound_crypto_shadow_service import EdgeBoundCryptoShadowService
from .phase114_crypto_shadow_machine_entrypoint import (
    CryptoShadowMachineEntrypointError,
    parse_machine_policy,
)
from .phase115_crypto_shadow_readiness_gate import CryptoShadowReadinessGate

PHASE117_SCHEMA_VERSION = "brian.phase117-readiness-guarded-crypto-shadow.v1"

EXIT_EXECUTED = 0
EXIT_SAFE_FAIL_CLOSED_ONLY = 10
EXIT_NOT_READY = 20
EXIT_INPUT_ERROR = 30
EXIT_WORKER_ERROR = 40


class ReadinessGuardedCryptoShadowError(RuntimeError):
    pass


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ReadinessGuardedCryptoShadowError(
            f"invalid Phase117 arguments: {message}"
        )


def _load_policy(path: str | None, stdin: TextIO) -> Mapping[str, object]:
    if path is None:
        if hasattr(stdin, "isatty") and stdin.isatty():
            raise ReadinessGuardedCryptoShadowError(
                "policy JSON is required on stdin or --policy"
            )
        raw = stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    if not raw.strip():
        raise ReadinessGuardedCryptoShadowError(
            "policy JSON cannot be empty"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReadinessGuardedCryptoShadowError(
            f"policy JSON invalid at line {exc.lineno} column {exc.colno}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ReadinessGuardedCryptoShadowError(
            "policy JSON root must be an object"
        )
    return payload


def _positive_int(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        raise ReadinessGuardedCryptoShadowError(
            f"{label} must be integer"
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ReadinessGuardedCryptoShadowError(
            f"{label} must be integer"
        ) from exc
    if parsed < minimum or parsed > maximum:
        raise ReadinessGuardedCryptoShadowError(
            f"{label} must be in [{minimum},{maximum}]"
        )
    return parsed


def _safe_error(exc: Exception, env: Mapping[str, str]) -> str:
    message = str(exc)
    secret_names = (
        "SUPABASE_SECRET_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "BRIAN_SENSOR_SUPABASE_SECRET_KEY",
        "BRIAN_SENSOR_SUPABASE_SERVICE_ROLE_KEY",
        "BRIAN_EDGE_SUPABASE_SECRET_KEY",
        "BRIAN_EDGE_SUPABASE_SERVICE_ROLE_KEY",
        "BRIAN_COST_SUPABASE_SECRET_KEY",
        "BRIAN_COST_SUPABASE_SERVICE_ROLE_KEY",
        "BRIAN_RUNTIME_SUPABASE_SECRET_KEY",
        "BRIAN_RUNTIME_SUPABASE_SERVICE_ROLE_KEY",
    )
    secrets = {
        str(env.get(name, "")).strip()
        for name in secret_names
        if str(env.get(name, "")).strip()
    }
    for secret in sorted(secrets, key=len, reverse=True):
        message = message.replace(secret, "<redacted>")
    message = re.sub(
        r"sb_secret_[A-Za-z0-9._-]+",
        "<redacted>",
        message,
    )
    return f"{type(exc).__name__}: {message[:420]}"


def _error(status: str, message: str) -> dict[str, object]:
    return {
        "schema_version": PHASE117_SCHEMA_VERSION,
        "status": status,
        "worker_invoked": False,
        "error": message[:500],
        "shadow_only": True,
        "live_execution": False,
    }


def _receipt_summary(receipt) -> dict[str, object]:
    cycle = receipt.cycle
    return {
        "runtime_id": receipt.runtime_id,
        "status": receipt.status,
        "recovery_status": receipt.startup.status,
        "ready_for_normal_shadow": receipt.startup.ready_for_normal_shadow,
        "prefetched": receipt.prefetched,
        "bundle_ref": receipt.bundle_ref,
        "decision_pipeline_id": (
            None
            if cycle is None
            else cycle.decision.pipeline_id
        ),
        "decision_status": (
            None
            if cycle is None
            else cycle.decision.status
        ),
        "execution_status": (
            None
            if cycle is None
            else cycle.execution.status
        ),
        "executed": (
            False
            if cycle is None
            else bool(cycle.execution.executed)
        ),
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    readiness_factory=CryptoShadowReadinessGate.from_env,
    service_factory=EdgeBoundCryptoShadowService.from_env,
    clock=time.time,
) -> int:
    source = dict(os.environ if env is None else env)
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    error_stream = sys.stderr if stderr is None else stderr

    try:
        parser = _JsonArgumentParser(
            description=(
                "Brian readiness-guarded one-shot recovery-first crypto "
                "shadow worker"
            )
        )
        parser.add_argument("--policy", default=None)
        parser.add_argument("--runtime-id", default=None)
        parser.add_argument("--recovery-max-items", type=int, default=None)
        parser.add_argument("--recovery-claim-seconds", type=int, default=None)
        parser.add_argument("--recovery-ttl-seconds", type=int, default=None)
        parser.add_argument("--normal-claim-seconds", type=int, default=None)
        parser.add_argument("--recovery-worker-token", default=None)
        parser.add_argument("--normal-worker-token", default=None)
        parser.add_argument(
            "--recovery-source-ref",
            default="phase117:recovery",
        )
        args = parser.parse_args(argv)

        policy = parse_machine_policy(
            _load_policy(args.policy, input_stream)
        )
        env_runtime = str(source.get("BRIAN_RUNTIME_ID", "")).strip()
        arg_runtime = (
            str(args.runtime_id).strip()
            if args.runtime_id is not None
            else ""
        )
        if arg_runtime and env_runtime and arg_runtime != env_runtime:
            raise ReadinessGuardedCryptoShadowError(
                "--runtime-id conflicts with BRIAN_RUNTIME_ID"
            )
        runtime_id = arg_runtime or env_runtime
        if not runtime_id:
            raise ReadinessGuardedCryptoShadowError(
                "runtime_id is required via --runtime-id or BRIAN_RUNTIME_ID"
            )
        source["BRIAN_RUNTIME_ID"] = runtime_id

        recovery_max_items = _positive_int(
            args.recovery_max_items
            if args.recovery_max_items is not None
            else source.get("BRIAN_RECOVERY_MAX_ITEMS", "8"),
            "recovery_max_items",
            minimum=1,
            maximum=32,
        )
        recovery_claim_seconds = _positive_int(
            args.recovery_claim_seconds
            if args.recovery_claim_seconds is not None
            else source.get("BRIAN_RECOVERY_CLAIM_SECONDS", "30"),
            "recovery_claim_seconds",
            minimum=10,
            maximum=300,
        )
        recovery_ttl_seconds = _positive_int(
            args.recovery_ttl_seconds
            if args.recovery_ttl_seconds is not None
            else source.get("BRIAN_RECOVERY_INTENT_TTL_SECONDS", "60"),
            "recovery_ttl_seconds",
            minimum=10,
            maximum=900,
        )
        normal_claim_seconds = _positive_int(
            args.normal_claim_seconds
            if args.normal_claim_seconds is not None
            else source.get("BRIAN_NORMAL_CLAIM_SECONDS", "45"),
            "normal_claim_seconds",
            minimum=10,
            maximum=300,
        )
        recovery_worker_token = (
            str(args.recovery_worker_token).strip()
            if args.recovery_worker_token is not None
            else str(source.get("BRIAN_RECOVERY_WORKER_TOKEN", "")).strip()
        ) or f"phase117-recovery-{uuid.uuid4().hex}"
        normal_worker_token = (
            str(args.normal_worker_token).strip()
            if args.normal_worker_token is not None
            else str(source.get("BRIAN_NORMAL_WORKER_TOKEN", "")).strip()
        ) or f"phase117-normal-{uuid.uuid4().hex}"
        if recovery_worker_token == normal_worker_token:
            raise ReadinessGuardedCryptoShadowError(
                "recovery and normal worker tokens must be different"
            )
        recovery_source_ref = str(args.recovery_source_ref).strip()
        if not recovery_source_ref:
            raise ReadinessGuardedCryptoShadowError(
                "recovery_source_ref is required"
            )
    except (
        ReadinessGuardedCryptoShadowError,
        CryptoShadowMachineEntrypointError,
        ValueError,
        TypeError,
        OSError,
    ) as exc:
        print(
            json.dumps(
                _error("INPUT_ERROR", str(exc)),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            file=error_stream,
            flush=True,
        )
        return EXIT_INPUT_ERROR

    readiness_timestamp = float(clock())
    try:
        with readiness_factory(env=source, clock=clock) as gate:
            readiness = gate.run(
                runtime_id=runtime_id,
                asset_ids=policy.asset_ids,
                config=policy.config,
                minimum_net_margin_bps=policy.minimum_net_margin_bps,
                decision_timestamp=readiness_timestamp,
            )
    except Exception as exc:
        print(
            json.dumps(
                _error("READINESS_ERROR", _safe_error(exc, source)),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            file=error_stream,
            flush=True,
        )
        return EXIT_WORKER_ERROR

    if readiness.status != "READY_FOR_EDGE_BOUND_SHADOW":
        payload = {
            "schema_version": PHASE117_SCHEMA_VERSION,
            "status": "READINESS_BLOCKED",
            "worker_invoked": False,
            "readiness": readiness.to_dict(),
            "shadow_only": True,
            "live_execution": False,
        }
        print(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            file=output_stream,
            flush=True,
        )
        return (
            EXIT_SAFE_FAIL_CLOSED_ONLY
            if readiness.status == "SAFE_FAIL_CLOSED_ONLY"
            else EXIT_NOT_READY
        )

    try:
        with service_factory(
            asset_ids=policy.asset_ids,
            model_weights=policy.model_weights,
            config=policy.config,
            max_slippage_bps=policy.max_slippage_bps,
            ttl_seconds=policy.ttl_seconds,
            minimum_net_margin_bps=policy.minimum_net_margin_bps,
            env=source,
            clock=clock,
        ) as service:
            receipt = service.run_once(
                recovery_max_items=recovery_max_items,
                recovery_worker_token=recovery_worker_token,
                recovery_claim_seconds=recovery_claim_seconds,
                recovery_ttl_seconds=recovery_ttl_seconds,
                recovery_source_ref=recovery_source_ref,
                normal_worker_token=normal_worker_token,
                normal_claim_seconds=normal_claim_seconds,
                clock=clock,
            )
    except Exception as exc:
        print(
            json.dumps(
                _error("WORKER_ERROR", _safe_error(exc, source)),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            file=error_stream,
            flush=True,
        )
        return EXIT_WORKER_ERROR

    if receipt.runtime_id != runtime_id:
        print(
            json.dumps(
                _error(
                    "WORKER_ERROR",
                    "runtime identity changed between readiness and worker",
                ),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            file=error_stream,
            flush=True,
        )
        return EXIT_WORKER_ERROR

    payload = {
        "schema_version": PHASE117_SCHEMA_VERSION,
        "status": receipt.status,
        "worker_invoked": True,
        "readiness_report_id": readiness.report_id,
        "readiness_observed_at": readiness.observed_at,
        "worker": _receipt_summary(receipt),
        "shadow_only": True,
        "live_execution": False,
    }
    print(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        file=output_stream,
        flush=True,
    )
    return (
        EXIT_EXECUTED
        if receipt.startup.ready_for_normal_shadow
        else EXIT_NOT_READY
    )


if __name__ == "__main__":
    raise SystemExit(main())
