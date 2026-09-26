from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from .phase91_supabase_rpc_transport import (
    SupabaseRecoveryRpcConfigurationError,
)
from .phase114_crypto_shadow_machine_entrypoint import (
    CryptoShadowMachineEntrypointError,
    parse_machine_policy,
)
from .phase115_crypto_shadow_readiness_gate import (
    CryptoShadowReadinessGate,
)
from .phase120_strict_supabase_topology import (
    load_strict_supabase_topology,
)

PHASE116_SCHEMA_VERSION = "brian.phase116-crypto-shadow-readiness-entrypoint.v1"

EXIT_READY = 0
EXIT_SAFE_FAIL_CLOSED_ONLY = 10
EXIT_NOT_READY = 20
EXIT_INPUT_ERROR = 30
EXIT_CHECK_ERROR = 40


class CryptoShadowReadinessEntrypointError(RuntimeError):
    pass


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CryptoShadowReadinessEntrypointError(
            f"invalid Phase116 arguments: {message}"
        )


def _load_policy(path: str | None, stdin: TextIO) -> Mapping[str, object]:
    if path is None:
        if hasattr(stdin, "isatty") and stdin.isatty():
            raise CryptoShadowReadinessEntrypointError(
                "policy JSON is required on stdin or --policy"
            )
        raw = stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    if not raw.strip():
        raise CryptoShadowReadinessEntrypointError(
            "policy JSON cannot be empty"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CryptoShadowReadinessEntrypointError(
            f"policy JSON invalid at line {exc.lineno} column {exc.colno}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CryptoShadowReadinessEntrypointError(
            "policy JSON root must be an object"
        )
    return payload


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
    for scope in (
        "BRIAN_SENSOR",
        "BRIAN_EDGE",
        "BRIAN_COST",
        "BRIAN_RUNTIME",
    ):
        raw = str(
            env.get(f"{scope}_SUPABASE_SECRET_KEYS", "")
        ).strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            secrets.update(
                str(value).strip()
                for value in parsed.values()
                if isinstance(value, str) and value.strip()
            )
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
        "schema_version": PHASE116_SCHEMA_VERSION,
        "status": status,
        "error": message[:500],
        "read_only": True,
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
    gate_factory=CryptoShadowReadinessGate.from_env,
    topology_loader=load_strict_supabase_topology,
    clock=time.time,
) -> int:
    source = os.environ if env is None else env
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    error_stream = sys.stderr if stderr is None else stderr

    try:
        parser = _JsonArgumentParser(
            description=(
                "Brian read-only pre-scheduler readiness probe for the "
                "recovery-first edge-bound crypto shadow machine"
            )
        )
        parser.add_argument("--policy", default=None)
        parser.add_argument("--runtime-id", default=None)
        args = parser.parse_args(argv)

        policy = parse_machine_policy(
            _load_policy(args.policy, input_stream)
        )
        runtime_id = (
            str(args.runtime_id).strip()
            if args.runtime_id is not None
            else str(source.get("BRIAN_RUNTIME_ID", "")).strip()
        )
        if not runtime_id:
            raise CryptoShadowReadinessEntrypointError(
                "runtime_id is required via --runtime-id or BRIAN_RUNTIME_ID"
            )
        topology = topology_loader(source)
    except (
        CryptoShadowReadinessEntrypointError,
        CryptoShadowMachineEntrypointError,
        SupabaseRecoveryRpcConfigurationError,
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

    try:
        with gate_factory(env=source, clock=clock) as gate:
            report = gate.run(
                runtime_id=runtime_id,
                asset_ids=policy.asset_ids,
                config=policy.config,
                minimum_net_margin_bps=policy.minimum_net_margin_bps,
                decision_timestamp=float(clock()),
            )
    except Exception as exc:
        print(
            json.dumps(
                _error("CHECK_ERROR", _safe_error(exc, source)),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            file=error_stream,
            flush=True,
        )
        return EXIT_CHECK_ERROR

    payload = report.to_dict()
    payload["entrypoint_schema_version"] = PHASE116_SCHEMA_VERSION
    payload["topology_id"] = topology.topology_id
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
    if report.status == "READY_FOR_EDGE_BOUND_SHADOW":
        return EXIT_READY
    if report.status == "SAFE_FAIL_CLOSED_ONLY":
        return EXIT_SAFE_FAIL_CLOSED_ONLY
    return EXIT_NOT_READY


if __name__ == "__main__":
    raise SystemExit(main())
