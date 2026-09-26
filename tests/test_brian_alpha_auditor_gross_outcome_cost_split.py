from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / (
    "supabase/functions/"
    "brian-missed-opportunity-auditor-v3/index.ts"
)
HELPER = ROOT / "supabase/functions/_shared/alpha_audit.ts"


def test_auditor_v3_separates_gross_outcome_from_cost_receipt() -> None:
    source = AUDITOR.read_text(encoding="utf-8")

    assert "resolveAlphaAuditGrossHorizon" in source
    assert "const costAware = resolveAlphaAuditHorizon" in source
    assert "const grossOnly = costAware" in source
    assert 'classification = costAware?.classification ??' in source
    assert '"OUTCOME_RESOLVED_COST_UNAVAILABLE"' in source
    assert "cost_covered: costCovered" in source


def test_unknown_cost_never_becomes_zero_or_missed_receipt() -> None:
    source = AUDITOR.read_text(encoding="utf-8")

    assert (
        '(d.action === "WAIT" || d.action === "VETO") && costAware'
        in source
    )
    assert "skippedCostUncoveredReceipt++" in source
    assert (
        "gross_outcome_resolves_without_cost_but_cost_dependent_receipts_fail_closed"
        in source
    )
    assert "costBps ?? 0" not in source
    assert "estimatedRoundTripCostBps ?? 0" not in source


def test_shared_cost_aware_resolver_remains_fail_closed() -> None:
    source = HELPER.read_text(encoding="utf-8")

    assert "export function resolveAlphaAuditGrossHorizon" in source
    assert "export function resolveAlphaAuditHorizon" in source
    assert "Unknown cost never becomes 0 bps" in source
    assert "return null;" in source
    assert "t <= targetMs" in source


def test_phase124_is_measurement_only_and_adds_no_trade_surface() -> None:
    combined = (
        AUDITOR.read_text(encoding="utf-8")
        + "\n"
        + HELPER.read_text(encoding="utf-8")
    ).lower()

    for forbidden in (
        "place_order",
        "create_order",
        "exchange_api_key",
        "live_execution: true",
        "automatic_promotion",
    ):
        assert forbidden not in combined
