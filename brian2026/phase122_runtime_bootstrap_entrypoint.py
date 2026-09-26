from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from .phase68_operational_risk_governor import OperationalRiskPolicy
from .phase120_strict_supabase_topology import load_strict_supabase_topology
from .phase121_runtime_bootstrap import (
    RuntimeBootstrapSpec,
    RuntimeBootstrapper,
    build_runtime_bootstrap_artifacts,
)

PHASE122_SCHEMA_VERSION = "brian.phase122-runtime-bootstrap-entrypoint.v1"


class RuntimeBootstrapEntrypointError(RuntimeError):
    pass


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise RuntimeBootstrapEntrypointError(
            f"invalid Phase122 arguments: {message}"
        )


def _read_manifest(path: str | None, stdin: TextIO) -> Mapping[str, object]:
    raw = stdin.read() if path is None else Path(path).read_text(encoding="utf-8")
    if not raw.strip():
        raise RuntimeBootstrapEntrypointError("bootstrap manifest cannot be empty")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeBootstrapEntrypointError(
            f"bootstrap manifest invalid at line {exc.lineno} column {exc.colno}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeBootstrapEntrypointError(
            "bootstrap manifest root must be an object"
        )
    return payload


def parse_bootstrap_spec(payload: Mapping[str, object]) -> RuntimeBootstrapSpec:
    raw = dict(payload)
    policy_raw = raw.get("risk_policy")
    if not isinstance(policy_raw, Mapping):
        raise RuntimeBootstrapEntrypointError(
            "risk_policy must be an object"
        )
    try:
        policy = OperationalRiskPolicy(**dict(policy_raw))
        assets_raw = raw["covered_assets"]
        if not isinstance(assets_raw, Sequence) or isinstance(
            assets_raw, (str, bytes)
        ):
            raise TypeError("covered_assets must be an array")
        return RuntimeBootstrapSpec(
            runtime_id=str(raw["runtime_id"]),
            account_id=str(raw["account_id"]),
            covered_assets=tuple(str(value) for value in assets_raw),
            starting_cash_usd=float(raw["starting_cash_usd"]),
            paper_fee_bps=float(raw["paper_fee_bps"]),
            allow_short=raw["allow_short"],  # type: ignore[arg-type]
            observed_at=float(raw["genesis_timestamp"]),
            source_ref=str(raw["source_ref"]),
            risk_policy=policy,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeBootstrapEntrypointError(
            f"invalid bootstrap manifest: {exc}"
        ) from exc


def _manifest_identity(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _emit(payload: Mapping[str, object], stream: TextIO) -> None:
    print(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        file=stream,
        flush=True,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    bootstrapper_factory=RuntimeBootstrapper.from_env,
    topology_loader=load_strict_supabase_topology,
) -> int:
    source = os.environ if env is None else env
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    error_stream = sys.stderr if stderr is None else stderr

    try:
        parser = _JsonArgumentParser(
            description=(
                "Brian one-shot durable shadow runtime bootstrap. "
                "Dry-run is default; --commit is explicit."
            )
        )
        parser.add_argument("--manifest", default=None)
        parser.add_argument("--commit", action="store_true")
        parser.add_argument("--owner-token", default=None)
        args = parser.parse_args(argv)

        raw = _read_manifest(args.manifest, input_stream)
        spec = parse_bootstrap_spec(raw)
        artifacts = build_runtime_bootstrap_artifacts(spec)
        manifest_id = _manifest_identity(raw)

        if not args.commit:
            _emit(
                {
                    "schema_version": PHASE122_SCHEMA_VERSION,
                    "status": "DRY_RUN",
                    "manifest_id": manifest_id,
                    "runtime_id": spec.runtime_id,
                    "genesis_checkpoint": artifacts.runtime.checkpoint().to_dict(),
                    "risk_manifest": artifacts.risk_ledger.manifest(),
                    "shadow_only": True,
                    "live_execution": False,
                },
                output_stream,
            )
            return 0

        topology = topology_loader(source)
        owner_token = (
            str(args.owner_token).strip()
            if args.owner_token is not None
            else str(source.get("BRIAN_BOOTSTRAP_OWNER_TOKEN", "")).strip()
        )
        if not owner_token:
            raise RuntimeBootstrapEntrypointError(
                "commit requires --owner-token or BRIAN_BOOTSTRAP_OWNER_TOKEN"
            )

        with bootstrapper_factory(env=source) as bootstrapper:
            receipt = bootstrapper.bootstrap(
                spec=spec,
                owner_token=owner_token,
            )
        payload = receipt.to_dict()
        payload.update({
            "entrypoint_schema_version": PHASE122_SCHEMA_VERSION,
            "manifest_id": manifest_id,
            "topology_id": topology.topology_id,
        })
        _emit(payload, output_stream)
        return 0
    except Exception as exc:
        _emit(
            {
                "schema_version": PHASE122_SCHEMA_VERSION,
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                "shadow_only": True,
                "live_execution": False,
            },
            error_stream,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
