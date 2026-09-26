from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from brian2026.phase114_crypto_shadow_machine_entrypoint import (
    EXIT_INPUT_ERROR,
    EXIT_OK,
    EXIT_RECOVERY_BLOCKED,
    EXIT_WORKER_ERROR,
    CryptoShadowMachineEntrypointError,
    main,
    parse_machine_policy,
)


def _policy_payload():
    return {
        "asset_ids": ["crypto:BTCUSDT", "crypto:ETHUSDT"],
        "model_weights": {
            "market_snapshot_analyst": 1.0,
            "derivatives_analyst": 0.5,
        },
        "decision": {
            "gross_target": 0.5,
            "position_limits": {
                "max_position_pct": 0.3,
                "max_gross_exposure": 0.5,
            },
            "covariance": {
                "alpha": "lw",
                "min_observations": 30,
                "max_period_volatility": 0.025,
                "nan_policy": "reject",
            },
            "turnover": {
                "max_l1_turnover": 0.25,
                "risk_reduction_bypass": True,
            },
            "market_neutral": False,
        },
        "execution": {
            "max_slippage_bps": 20.0,
            "ttl_seconds": 60,
            "minimum_net_margin_bps": 2.0,
        },
    }


def _receipt(*, ready=True, status="SHADOW_EXECUTED"):
    decision = SimpleNamespace(
        pipeline_id="p" * 64,
        status="REBALANCE_PLANNED",
        timestamp=123.0,
        final_planned_weights={"crypto:BTCUSDT": 0.2},
    )
    execution = SimpleNamespace(
        status="SHADOW_EXECUTED",
        executed=True,
        risk_version=4,
        risk_receipt_id="risk-1",
        governed_result_id="gov-1",
    )
    cycle = SimpleNamespace(
        decision=decision,
        execution=execution,
    ) if ready else None
    startup = SimpleNamespace(
        status=(
            "READY_FOR_NORMAL_SHADOW"
            if ready
            else "RECOVERY_BLOCKED"
        ),
        ready_for_normal_shadow=ready,
    )
    return SimpleNamespace(
        runtime_id="runtime-114",
        status=status if ready else "RECOVERY_BLOCKED",
        startup=startup,
        prefetched=ready,
        bundle_ref="b" * 64 if ready else None,
        cycle=cycle,
    )


class _Service:
    def __init__(self, receipt):
        self.receipt = receipt
        self.calls = []
        self.closed = False

    def run_once(self, **kwargs):
        self.calls.append(kwargs)
        return self.receipt

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.closed = True


def test_parse_machine_policy_builds_explicit_phase54_configuration() -> None:
    policy = parse_machine_policy(_policy_payload())

    assert policy.asset_ids == (
        "crypto:BTCUSDT",
        "crypto:ETHUSDT",
    )
    assert policy.model_weights == {
        "market_snapshot_analyst": pytest.approx(1.0),
        "derivatives_analyst": pytest.approx(0.5),
    }
    assert policy.config.gross_target == pytest.approx(0.5)
    assert policy.config.position_limits.max_position_pct == pytest.approx(0.3)
    assert policy.config.position_limits.max_gross_exposure == pytest.approx(0.5)
    assert policy.config.covariance.alpha == "lw"
    assert policy.config.covariance.min_observations == 30
    assert policy.config.covariance.max_period_volatility == pytest.approx(0.025)
    assert policy.config.covariance.nan_policy == "reject"
    assert policy.config.turnover.max_l1_turnover == pytest.approx(0.25)
    assert policy.config.turnover.risk_reduction_bypass is True
    assert policy.max_slippage_bps == pytest.approx(20.0)
    assert policy.ttl_seconds == 60
    assert policy.minimum_net_margin_bps == pytest.approx(2.0)
    assert policy.shadow_only is True
    assert policy.live_execution is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda p: p.update({"asset_ids": ["fx:EURUSD"]}),
            "crypto:\\*USDT",
        ),
        (
            lambda p: p.update({"unknown": 1}),
            "unknown policy fields",
        ),
        (
            lambda p: p["model_weights"].clear(),
            "positive aggregate weight",
        ),
        (
            lambda p: p["decision"]["covariance"].update(
                {"alpha": 1.5}
            ),
            "alpha must be <= 1",
        ),
        (
            lambda p: p["execution"].update({"ttl_seconds": 0}),
            "ttl_seconds",
        ),
    ],
)
def test_policy_parser_fails_closed_on_invalid_or_unknown_policy(
    mutator,
    message,
) -> None:
    payload = _policy_payload()
    mutator(payload)
    with pytest.raises(
        (CryptoShadowMachineEntrypointError, ValueError),
        match=message,
    ):
        parse_machine_policy(payload)


def test_main_runs_one_owned_service_cycle_and_emits_bounded_json() -> None:
    service = _Service(_receipt())
    factory_calls = []

    def service_factory(**kwargs):
        factory_calls.append(kwargs)
        return service

    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        [],
        env={
            "SUPABASE_URL": "https://example.supabase.co",
            "BRIAN_RECOVERY_MAX_ITEMS": "4",
        },
        stdin=io.StringIO(json.dumps(_policy_payload())),
        stdout=stdout,
        stderr=stderr,
        service_factory=service_factory,
        clock=lambda: 123.0,
    )

    assert code == EXIT_OK
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["status"] == "SHADOW_EXECUTED"
    assert payload["ready_for_normal_shadow"] is True
    assert payload["prefetched"] is True
    assert payload["bundle_ref"] == "b" * 64
    assert payload["decision"]["pipeline_id"] == "p" * 64
    assert payload["execution"]["executed"] is True
    assert payload["shadow_only"] is True
    assert payload["live_execution"] is False
    assert service.closed is True

    assert len(factory_calls) == 1
    built = factory_calls[0]
    assert built["asset_ids"] == (
        "crypto:BTCUSDT",
        "crypto:ETHUSDT",
    )
    assert built["max_slippage_bps"] == pytest.approx(20.0)
    assert built["ttl_seconds"] == 60
    assert built["minimum_net_margin_bps"] == pytest.approx(2.0)

    assert len(service.calls) == 1
    run = service.calls[0]
    assert run["recovery_max_items"] == 4
    assert run["recovery_worker_token"].startswith("phase114-recovery-")
    assert run["normal_worker_token"].startswith("phase114-normal-")
    assert run["recovery_worker_token"] != run["normal_worker_token"]
    assert run["recovery_claim_seconds"] == 30
    assert run["normal_claim_seconds"] == 45


def test_main_returns_recovery_blocked_without_fabricating_decision() -> None:
    service = _Service(_receipt(ready=False))
    stdout = io.StringIO()
    code = main(
        [],
        env={},
        stdin=io.StringIO(json.dumps(_policy_payload())),
        stdout=stdout,
        stderr=io.StringIO(),
        service_factory=lambda **kwargs: service,
        clock=lambda: 123.0,
    )

    assert code == EXIT_RECOVERY_BLOCKED
    payload = json.loads(stdout.getvalue())
    assert payload["status"] == "RECOVERY_BLOCKED"
    assert payload["prefetched"] is False
    assert payload["bundle_ref"] is None
    assert payload["decision"] is None
    assert payload["execution"] is None


def test_main_rejects_same_recovery_and_normal_worker_token_before_service() -> None:
    calls = 0

    def service_factory(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("must not build service")

    stderr = io.StringIO()
    code = main(
        [
            "--recovery-worker-token",
            "same-token",
            "--normal-worker-token",
            "same-token",
        ],
        env={},
        stdin=io.StringIO(json.dumps(_policy_payload())),
        stdout=io.StringIO(),
        stderr=stderr,
        service_factory=service_factory,
    )

    assert code == EXIT_INPUT_ERROR
    assert calls == 0
    payload = json.loads(stderr.getvalue())
    assert payload["status"] == "INPUT_ERROR"
    assert "must be different" in payload["error"]


def test_main_worker_error_redacts_supabase_secret() -> None:
    secret = "sb_secret_phase114_never_echo"

    def service_factory(**kwargs):
        raise RuntimeError(f"backend failed with {secret}")

    stderr = io.StringIO()
    code = main(
        [],
        env={"SUPABASE_SECRET_KEY": secret},
        stdin=io.StringIO(json.dumps(_policy_payload())),
        stdout=io.StringIO(),
        stderr=stderr,
        service_factory=service_factory,
    )

    assert code == EXIT_WORKER_ERROR
    text = stderr.getvalue()
    assert secret not in text
    payload = json.loads(text)
    assert payload["status"] == "WORKER_ERROR"
    assert "<redacted>" in payload["error"]


def test_main_requires_policy_instead_of_inventing_runtime_defaults() -> None:
    stderr = io.StringIO()
    code = main(
        [],
        env={},
        stdin=io.StringIO(""),
        stdout=io.StringIO(),
        stderr=stderr,
        service_factory=lambda **kwargs: pytest.fail("must not run"),
    )

    assert code == EXIT_INPUT_ERROR
    payload = json.loads(stderr.getvalue())
    assert "policy JSON cannot be empty" in payload["error"]
