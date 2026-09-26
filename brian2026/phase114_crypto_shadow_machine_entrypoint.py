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
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .phase44_portfolio_brain import PortfolioRiskLimits
from .phase52_covariance_risk import CovarianceRiskConfig
from .phase53_turnover_rebalance import TurnoverConfig
from .phase54_integrated_shadow_decision import IntegratedShadowConfig
from .phase112_edge_bound_crypto_shadow_service import (
    EdgeBoundCryptoShadowService,
)

PHASE114_SCHEMA_VERSION = "brian.phase114-crypto-shadow-machine-entrypoint.v1"

EXIT_OK = 0
EXIT_RECOVERY_BLOCKED = 20
EXIT_INPUT_ERROR = 30
EXIT_WORKER_ERROR = 40


class CryptoShadowMachineEntrypointError(RuntimeError):
    pass


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CryptoShadowMachineEntrypointError(
            f"invalid Phase114 arguments: {message}"
        )


@dataclass(frozen=True, slots=True)
class CryptoShadowMachinePolicy:
    asset_ids: tuple[str, ...]
    model_weights: Mapping[str, float]
    config: IntegratedShadowConfig
    max_slippage_bps: float
    ttl_seconds: int
    minimum_net_margin_bps: float
    schema_version: str = PHASE114_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.asset_ids:
            raise ValueError("asset_ids are required")
        if not self.model_weights:
            raise ValueError("model_weights are required")
        if (
            not math.isfinite(float(self.max_slippage_bps))
            or self.max_slippage_bps < 0
        ):
            raise ValueError("max_slippage_bps must be finite and non-negative")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if (
            not math.isfinite(float(self.minimum_net_margin_bps))
            or self.minimum_net_margin_bps < 0
        ):
            raise ValueError(
                "minimum_net_margin_bps must be finite and non-negative"
            )
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase114 policy must remain shadow-only")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CryptoShadowMachineEntrypointError(
            f"{label} must be a JSON object"
        )
    return {str(key): item for key, item in value.items()}


def _finite(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise CryptoShadowMachineEntrypointError(
            f"{label} must be numeric"
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CryptoShadowMachineEntrypointError(
            f"{label} must be numeric"
        ) from exc
    if not math.isfinite(result):
        raise CryptoShadowMachineEntrypointError(
            f"{label} must be finite"
        )
    if minimum is not None and result < minimum:
        raise CryptoShadowMachineEntrypointError(
            f"{label} must be >= {minimum}"
        )
    return result


def _integer(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        raise CryptoShadowMachineEntrypointError(
            f"{label} must be integer"
        )
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CryptoShadowMachineEntrypointError(
            f"{label} must be integer"
        ) from exc
    if str(value).strip() not in {str(result), f"{result}.0"} and not isinstance(
        value, int
    ):
        raise CryptoShadowMachineEntrypointError(
            f"{label} must be integer"
        )
    if result < minimum or result > maximum:
        raise CryptoShadowMachineEntrypointError(
            f"{label} must be in [{minimum},{maximum}]"
        )
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise CryptoShadowMachineEntrypointError(
            f"{label} must be boolean"
        )
    return value


def parse_machine_policy(payload: Mapping[str, object]) -> CryptoShadowMachinePolicy:
    root = _mapping(payload, "policy")
    allowed = {
        "asset_ids",
        "model_weights",
        "decision",
        "execution",
    }
    unknown = sorted(set(root) - allowed)
    if unknown:
        raise CryptoShadowMachineEntrypointError(
            "unknown policy fields: " + ",".join(unknown)
        )

    raw_assets = root.get("asset_ids")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise CryptoShadowMachineEntrypointError(
            "asset_ids must be a non-empty array"
        )
    assets = tuple(sorted({
        str(value).strip()
        for value in raw_assets
        if str(value).strip()
    }))
    if len(assets) != len(raw_assets):
        raise CryptoShadowMachineEntrypointError(
            "asset_ids cannot contain blanks or duplicates"
        )
    pattern = re.compile(r"^crypto:[A-Z0-9]{2,20}USDT$")
    if any(pattern.fullmatch(asset) is None for asset in assets):
        raise CryptoShadowMachineEntrypointError(
            "Phase114 supports crypto:*USDT asset ids only"
        )

    raw_weights = _mapping(root.get("model_weights"), "model_weights")
    weights: dict[str, float] = {}
    for raw_name, raw_value in raw_weights.items():
        name = raw_name.strip()
        if not name:
            raise CryptoShadowMachineEntrypointError(
                "model weight name cannot be blank"
            )
        value = _finite(raw_value, f"model_weights.{name}", minimum=0.0)
        weights[name] = value
    if not weights or sum(weights.values()) <= 0:
        raise CryptoShadowMachineEntrypointError(
            "model_weights must contain positive aggregate weight"
        )

    decision = _mapping(root.get("decision"), "decision")
    decision_allowed = {
        "gross_target",
        "position_limits",
        "covariance",
        "turnover",
        "market_neutral",
    }
    unknown_decision = sorted(set(decision) - decision_allowed)
    if unknown_decision:
        raise CryptoShadowMachineEntrypointError(
            "unknown decision fields: " + ",".join(unknown_decision)
        )

    position = _mapping(
        decision.get("position_limits"),
        "decision.position_limits",
    )
    if set(position) != {"max_position_pct", "max_gross_exposure"}:
        raise CryptoShadowMachineEntrypointError(
            "position_limits requires max_position_pct and max_gross_exposure"
        )
    limits = PortfolioRiskLimits(
        max_position_pct=_finite(
            position["max_position_pct"],
            "decision.position_limits.max_position_pct",
            minimum=0.0,
        ),
        max_gross_exposure=_finite(
            position["max_gross_exposure"],
            "decision.position_limits.max_gross_exposure",
            minimum=0.0,
        ),
    )

    covariance_raw = _mapping(
        decision.get("covariance"),
        "decision.covariance",
    )
    covariance_allowed = {
        "alpha",
        "min_observations",
        "max_period_volatility",
        "nan_policy",
    }
    unknown_cov = sorted(set(covariance_raw) - covariance_allowed)
    if unknown_cov:
        raise CryptoShadowMachineEntrypointError(
            "unknown covariance fields: " + ",".join(unknown_cov)
        )
    alpha_value = covariance_raw.get("alpha", "lw")
    if alpha_value != "lw":
        alpha_value = _finite(
            alpha_value,
            "decision.covariance.alpha",
            minimum=0.0,
        )
        if alpha_value > 1:
            raise CryptoShadowMachineEntrypointError(
                "decision.covariance.alpha must be <= 1"
            )
    nan_policy = str(
        covariance_raw.get("nan_policy", "reject")
    ).strip()
    if nan_policy not in {"fill_zero", "reject"}:
        raise CryptoShadowMachineEntrypointError(
            "decision.covariance.nan_policy must be fill_zero or reject"
        )
    covariance = CovarianceRiskConfig(
        alpha=alpha_value,
        min_observations=_integer(
            covariance_raw.get("min_observations", 30),
            "decision.covariance.min_observations",
            minimum=5,
            maximum=1000,
        ),
        max_period_volatility=_finite(
            covariance_raw.get("max_period_volatility"),
            "decision.covariance.max_period_volatility",
            minimum=0.0000001,
        ),
        nan_policy=nan_policy,
    )

    turnover_raw = _mapping(
        decision.get("turnover"),
        "decision.turnover",
    )
    if set(turnover_raw) - {
        "max_l1_turnover",
        "risk_reduction_bypass",
    }:
        raise CryptoShadowMachineEntrypointError(
            "unknown turnover fields"
        )
    turnover = TurnoverConfig(
        max_l1_turnover=_finite(
            turnover_raw.get("max_l1_turnover"),
            "decision.turnover.max_l1_turnover",
            minimum=0.0,
        ),
        risk_reduction_bypass=_boolean(
            turnover_raw.get("risk_reduction_bypass", True),
            "decision.turnover.risk_reduction_bypass",
        ),
    )

    config = IntegratedShadowConfig(
        gross_target=_finite(
            decision.get("gross_target"),
            "decision.gross_target",
            minimum=0.0,
        ),
        position_limits=limits,
        covariance=covariance,
        turnover=turnover,
        market_neutral=_boolean(
            decision.get("market_neutral", False),
            "decision.market_neutral",
        ),
    )

    execution = _mapping(root.get("execution"), "execution")
    execution_allowed = {
        "max_slippage_bps",
        "ttl_seconds",
        "minimum_net_margin_bps",
    }
    unknown_execution = sorted(set(execution) - execution_allowed)
    if unknown_execution:
        raise CryptoShadowMachineEntrypointError(
            "unknown execution fields: " + ",".join(unknown_execution)
        )

    return CryptoShadowMachinePolicy(
        asset_ids=assets,
        model_weights=weights,
        config=config,
        max_slippage_bps=_finite(
            execution.get("max_slippage_bps"),
            "execution.max_slippage_bps",
            minimum=0.0,
        ),
        ttl_seconds=_integer(
            execution.get("ttl_seconds"),
            "execution.ttl_seconds",
            minimum=10,
            maximum=900,
        ),
        minimum_net_margin_bps=_finite(
            execution.get("minimum_net_margin_bps", 2.0),
            "execution.minimum_net_margin_bps",
            minimum=0.0,
        ),
    )


def _load_policy(path: str | None, stdin: TextIO) -> Mapping[str, object]:
    if path is None:
        if hasattr(stdin, "isatty") and stdin.isatty():
            raise CryptoShadowMachineEntrypointError(
                "policy JSON is required on stdin or --policy"
            )
        raw = stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    if not raw.strip():
        raise CryptoShadowMachineEntrypointError(
            "policy JSON cannot be empty"
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CryptoShadowMachineEntrypointError(
            f"policy JSON invalid at line {exc.lineno} column {exc.colno}"
        ) from exc
    if not isinstance(value, Mapping):
        raise CryptoShadowMachineEntrypointError(
            "policy JSON root must be an object"
        )
    return value


def _positive_int_env(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    return _integer(
        value,
        label,
        minimum=minimum,
        maximum=maximum,
    )


def _safe_error(exc: Exception, env: Mapping[str, str]) -> str:
    message = str(exc)
    secrets = [
        env.get("SUPABASE_SECRET_KEY", ""),
        env.get("SUPABASE_SERVICE_ROLE_KEY", ""),
    ]
    for secret in sorted(
        {value for value in secrets if value},
        key=len,
        reverse=True,
    ):
        message = message.replace(secret, "<redacted>")
    message = re.sub(
        r"sb_secret_[A-Za-z0-9._-]+",
        "<redacted>",
        message,
    )
    return f"{type(exc).__name__}: {message[:420]}"


def _summary(receipt) -> dict[str, object]:
    cycle = receipt.cycle
    decision = None if cycle is None else cycle.decision
    execution = None if cycle is None else cycle.execution
    return {
        "schema_version": PHASE114_SCHEMA_VERSION,
        "runtime_id": receipt.runtime_id,
        "status": receipt.status,
        "recovery_status": receipt.startup.status,
        "ready_for_normal_shadow": receipt.startup.ready_for_normal_shadow,
        "prefetched": receipt.prefetched,
        "bundle_ref": receipt.bundle_ref,
        "decision": None if decision is None else {
            "pipeline_id": decision.pipeline_id,
            "status": decision.status,
            "timestamp": decision.timestamp,
            "final_planned_weights": dict(decision.final_planned_weights),
        },
        "execution": None if execution is None else {
            "status": execution.status,
            "executed": execution.executed,
            "risk_version": execution.risk_version,
            "risk_receipt_id": execution.risk_receipt_id,
            "governed_result_id": execution.governed_result_id,
        },
        "shadow_only": True,
        "live_execution": False,
    }


def _error(status: str, message: str) -> dict[str, object]:
    return {
        "schema_version": PHASE114_SCHEMA_VERSION,
        "status": status,
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
    service_factory=EdgeBoundCryptoShadowService.from_env,
    clock=time.time,
) -> int:
    source = os.environ if env is None else env
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    error_stream = sys.stderr if stderr is None else stderr

    try:
        parser = _JsonArgumentParser(
            description=(
                "Brian one-shot recovery-first edge-bound crypto shadow worker"
            )
        )
        parser.add_argument("--policy", default=None)
        parser.add_argument("--recovery-max-items", type=int, default=None)
        parser.add_argument("--recovery-claim-seconds", type=int, default=None)
        parser.add_argument("--recovery-ttl-seconds", type=int, default=None)
        parser.add_argument("--normal-claim-seconds", type=int, default=None)
        parser.add_argument("--recovery-worker-token", default=None)
        parser.add_argument("--normal-worker-token", default=None)
        parser.add_argument(
            "--recovery-source-ref",
            default="phase114:recovery",
        )
        args = parser.parse_args(argv)
        policy = parse_machine_policy(
            _load_policy(args.policy, input_stream)
        )

        recovery_max_items = _positive_int_env(
            args.recovery_max_items
            if args.recovery_max_items is not None
            else source.get("BRIAN_RECOVERY_MAX_ITEMS", "8"),
            "recovery_max_items",
            minimum=1,
            maximum=32,
        )
        recovery_claim_seconds = _positive_int_env(
            args.recovery_claim_seconds
            if args.recovery_claim_seconds is not None
            else source.get("BRIAN_RECOVERY_CLAIM_SECONDS", "30"),
            "recovery_claim_seconds",
            minimum=10,
            maximum=300,
        )
        recovery_ttl_seconds = _positive_int_env(
            args.recovery_ttl_seconds
            if args.recovery_ttl_seconds is not None
            else source.get("BRIAN_RECOVERY_INTENT_TTL_SECONDS", "60"),
            "recovery_ttl_seconds",
            minimum=10,
            maximum=900,
        )
        normal_claim_seconds = _positive_int_env(
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
            else source.get("BRIAN_RECOVERY_WORKER_TOKEN", "").strip()
        ) or f"phase114-recovery-{uuid.uuid4().hex}"
        normal_worker_token = (
            str(args.normal_worker_token).strip()
            if args.normal_worker_token is not None
            else source.get("BRIAN_NORMAL_WORKER_TOKEN", "").strip()
        ) or f"phase114-normal-{uuid.uuid4().hex}"
        if recovery_worker_token == normal_worker_token:
            raise CryptoShadowMachineEntrypointError(
                "recovery and normal worker tokens must be different"
            )
        recovery_source_ref = str(args.recovery_source_ref).strip()
        if not recovery_source_ref:
            raise CryptoShadowMachineEntrypointError(
                "recovery_source_ref is required"
            )
    except (
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
    return (
        EXIT_OK
        if receipt.startup.ready_for_normal_shadow
        else EXIT_RECOVERY_BLOCKED
    )


if __name__ == "__main__":
    raise SystemExit(main())
