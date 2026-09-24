from __future__ import annotations

import json
from pathlib import Path

from brian2026.phase114_crypto_shadow_machine_entrypoint import parse_machine_policy
from brian2026.phase110_supabase_grounded_market_prefetch import _SOURCE_KIND_BY_FAMILY


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "config" / "brian-shadow-machine-policy-v1.json"


def test_shadow_machine_policy_is_valid_and_covers_runtime_assets() -> None:
    payload = json.loads(POLICY.read_text(encoding="utf-8"))
    policy = parse_machine_policy(payload)

    assert policy.asset_ids == (
        "crypto:BTCUSDT",
        "crypto:ETHUSDT",
        "crypto:SOLUSDT",
        "crypto:XRPUSDT",
    )
    assert policy.shadow_only is True
    assert policy.live_execution is False


def test_shadow_machine_policy_weights_every_grounded_runtime_analyst() -> None:
    payload = json.loads(POLICY.read_text(encoding="utf-8"))
    policy = parse_machine_policy(payload)

    expected = {
        f"{source_kind}_analyst"
        for source_kind in _SOURCE_KIND_BY_FAMILY.values()
    }
    assert set(policy.model_weights) == expected
    assert all(weight > 0 for weight in policy.model_weights.values())


def test_shadow_machine_policy_keeps_tested_fail_closed_decision_controls() -> None:
    payload = json.loads(POLICY.read_text(encoding="utf-8"))
    policy = parse_machine_policy(payload)

    assert policy.config.covariance.nan_policy == "reject"
    assert policy.config.covariance.min_observations == 30
    assert policy.config.turnover.risk_reduction_bypass is True
    assert policy.max_slippage_bps == 20.0
    assert policy.ttl_seconds == 60
    assert policy.minimum_net_margin_bps == 2.0
