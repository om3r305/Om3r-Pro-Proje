from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from brian2026.phase122_runtime_bootstrap_entrypoint import (
    main,
    parse_bootstrap_spec,
)


FIXTURE = {
    "runtime_id": "brian-shadow-main",
    "account_id": "BRIAN-PAPER-RUNTIME",
    "covered_assets": [
        "crypto:BTCUSDT",
        "crypto:ETHUSDT",
    ],
    "starting_cash_usd": 5000.0,
    "paper_fee_bps": 10.0,
    "allow_short": True,
    "genesis_timestamp": 1_790_253_722.853,
    "source_ref": "phase122:test",
    "risk_policy": {
        "drawdown_lookback_seconds": 86400,
        "max_drawdown_fraction": 0.10,
        "daily_loss_lookback_seconds": 86400,
        "max_daily_loss_fraction": 0.07,
        "stoploss_lookback_seconds": 3600,
        "stoploss_limit": 4,
        "stoploss_required_profit": 0.0,
        "stoploss_lock_seconds": 1800,
        "asset_cooldown_seconds": 300,
        "execution_failure_lookback_seconds": 900,
        "max_consecutive_execution_failures": 3,
        "reconciliation_failure_lookback_seconds": 900,
        "max_reconciliation_failures": 2,
        "unknown_outcome_lookback_seconds": 3600,
        "max_unknown_order_outcomes": 1,
        "max_market_data_age_seconds": 30.0,
    },
}


def test_parse_bootstrap_spec_freezes_canonical_assets_and_policy() -> None:
    spec = parse_bootstrap_spec(FIXTURE)

    assert spec.runtime_id == "brian-shadow-main"
    assert spec.account_id == "BRIAN-PAPER-RUNTIME"
    assert spec.covered_assets == (
        "crypto:BTCUSDT",
        "crypto:ETHUSDT",
    )
    assert spec.starting_cash_usd == pytest.approx(5000.0)
    assert spec.risk_policy.max_drawdown_fraction == pytest.approx(0.10)


def test_dry_run_is_default_and_emits_valid_bootstrap_artifacts() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        [],
        env={},
        stdin=io.StringIO(json.dumps(FIXTURE)),
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["status"] == "DRY_RUN"
    assert payload["runtime_id"] == "brian-shadow-main"
    assert len(payload["manifest_id"]) == 64
    assert len(payload["genesis_checkpoint"]["checkpoint_id"]) == 64
    assert len(payload["risk_manifest"]["ledger_hash"]) == 64
    assert payload["risk_manifest"]["current_state"] == "ACTIVE"
    assert payload["risk_manifest"]["entry_count"] == 1
    assert payload["shadow_only"] is True
    assert payload["live_execution"] is False


def test_commit_requires_explicit_topology_before_bootstrap() -> None:
    stderr = io.StringIO()
    code = main(
        ["--commit", "--owner-token", "owner"],
        env={},
        stdin=io.StringIO(json.dumps(FIXTURE)),
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert code == 1
    payload = json.loads(stderr.getvalue())
    assert payload["status"] == "ERROR"
    assert "BRIAN_SENSOR_SUPABASE_URL" in payload["error"]


def test_commit_requires_owner_after_topology_is_validated() -> None:
    calls = []

    def topology_loader(env):
        calls.append(dict(env))
        return SimpleNamespace(topology_id="t" * 64)

    stderr = io.StringIO()
    code = main(
        ["--commit"],
        env={},
        stdin=io.StringIO(json.dumps(FIXTURE)),
        stdout=io.StringIO(),
        stderr=stderr,
        topology_loader=topology_loader,
    )

    assert code == 1
    assert calls == [{}]
    payload = json.loads(stderr.getvalue())
    assert "BRIAN_BOOTSTRAP_OWNER_TOKEN" in payload["error"]


class _Bootstrapper:
    def __init__(self):
        self.calls = []
        self.closed = False

    def bootstrap(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            to_dict=lambda: {
                "runtime_id": "brian-shadow-main",
                "status": "BOOTSTRAPPED",
                "runtime_version": 1,
                "risk_version": 1,
                "checkpoint_id": "c" * 64,
                "risk_ledger_hash": "r" * 64,
                "completed_partial_bootstrap": False,
                "already_bootstrapped": False,
                "bootstrap_id": "b" * 64,
                "shadow_only": True,
                "live_execution": False,
            }
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.closed = True


def test_commit_routes_exact_spec_through_phase121_after_topology_seal() -> None:
    bootstrapper = _Bootstrapper()
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return bootstrapper

    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        ["--commit", "--owner-token", "bootstrap-owner"],
        env={"marker": "env"},
        stdin=io.StringIO(json.dumps(FIXTURE)),
        stdout=stdout,
        stderr=stderr,
        bootstrapper_factory=factory,
        topology_loader=lambda env: SimpleNamespace(topology_id="t" * 64),
    )

    assert code == 0
    assert stderr.getvalue() == ""
    assert factory_calls == [{"env": {"marker": "env"}}]
    assert bootstrapper.closed is True
    assert len(bootstrapper.calls) == 1
    assert bootstrapper.calls[0]["owner_token"] == "bootstrap-owner"
    spec = bootstrapper.calls[0]["spec"]
    assert spec.runtime_id == "brian-shadow-main"
    assert spec.covered_assets == (
        "crypto:BTCUSDT",
        "crypto:ETHUSDT",
    )

    payload = json.loads(stdout.getvalue())
    assert payload["status"] == "BOOTSTRAPPED"
    assert payload["topology_id"] == "t" * 64
    assert len(payload["manifest_id"]) == 64


def test_checked_in_bootstrap_manifest_matches_phase122_contract() -> None:
    payload = json.loads(
        Path("config/brian-shadow-runtime-bootstrap-v1.json").read_text(
            encoding="utf-8"
        )
    )
    spec = parse_bootstrap_spec(payload)

    assert spec.runtime_id == "brian-shadow-main"
    assert spec.account_id == "BRIAN-PAPER-RUNTIME"
    assert spec.covered_assets == (
        "crypto:BTCUSDT",
        "crypto:ETHUSDT",
        "crypto:SOLUSDT",
        "crypto:XRPUSDT",
    )
    assert spec.starting_cash_usd == pytest.approx(5000.0)
    assert spec.observed_at == pytest.approx(1_790_253_722.853)
