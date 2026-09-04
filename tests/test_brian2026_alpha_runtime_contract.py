"""Static wiring/red-team contract for Brian ALPHA v2 SHADOW runtime.

These tests intentionally complement the pure Deno and real-Postgres behavioral tests. The Edge
Function entrypoint is I/O-heavy and not imported in unit tests, so this file prevents critical
runtime wiring from silently drifting while PR #49 remains SHADOW-only.
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / "supabase/functions/brian-alpha-decision-compiler/index.ts"
AUDIT = ROOT / "supabase/functions/_shared/alpha_audit.ts"
DECISION = ROOT / "supabase/functions/_shared/alpha_decision.ts"
RUNTIME_POLICY = ROOT / "supabase/functions/_shared/alpha_runtime_policy.ts"
CRON_AUTH = ROOT / "supabase/functions/_shared/cron_auth.ts"
AUDITOR = ROOT / "supabase/functions/brian-missed-opportunity-auditor/index.ts"
MACRO = ROOT / "supabase/functions/brian-official-macro-eye/index.ts"
MIGRATIONS = [
    ROOT / "supabase/migrations/202609040015_brian_alpha_decision_compiler.sql",
    ROOT / "supabase/migrations/202609040016_brian_alpha_shadow_state.sql",
    ROOT / "supabase/migrations/202609040017_brian_alpha_shadow_semantics.sql",
]


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_gdelt_is_discovery_only_at_runtime_and_pure_compiler_boundary():
    runtime = _text(COMPILER)
    pure = _text(DECISION)
    assert '.neq("independent_group", "news_gdelt")' in runtime
    assert 'row.independentGroup !== "news_gdelt"' in pure
    assert 'gdelt_role: "discovery_only_no_direction_vote"' in runtime


def test_phase37_evidence_is_pinned_to_frozen_experiment_and_causal_time():
    runtime = _text(COMPILER)
    assert 'FROZEN_PHASE37_EXPERIMENT_ID = "phase37-prospective-live-20260903"' in runtime
    assert '.eq("experiment_id", FROZEN_PHASE37_EXPERIMENT_ID)' in runtime
    assert '.lte("observed_at", new Date(nowMs).toISOString())' in runtime


def test_same_intrabar_tape_cannot_masquerade_as_multiple_independent_domains():
    pure = _text(DECISION)
    for group in ("micro_velocity", "micro_volume", "micro_breakout", "micro_reclaim", "micro_taker_flow"):
        assert f'"{group}"' in pure
    assert 'return INTRABAR_TAPE_GROUPS.has(group) ? "intrabar_tape" : group' in pure


def test_observed_l2_is_wired_and_invalid_observed_book_cannot_degrade():
    runtime = _text(COMPILER)
    policy = _text(RUNTIME_POLICY)
    assert 'compileL2Cost' in runtime
    assert 'parseBinanceDepthSnapshotRaw' in runtime
    assert '/api/v3/depth?symbol=' in runtime
    assert 'ObservedL2InvalidError' in runtime
    assert 'error instanceof ObservedL2InvalidError ? "INVALID" : "UNAVAILABLE"' in runtime
    assert 'selectAlphaRuntimeCost' in runtime
    assert 'params.l2Status !== "UNAVAILABLE"' in policy
    assert 'compileDegradedTopOfBookCost' in policy
    assert 'l2_runtime_status' in runtime
    assert 'l2_observed_costs' in runtime
    assert 'degraded_top_of_book_costs' in runtime


def test_auditor_never_silently_turns_unknown_cost_into_zero_and_keeps_horizon_pure():
    audit = _text(AUDIT)
    assert 'fallbackCost ?? 0' not in audit
    assert 'return null;' in audit
    assert 'const resolvedPoint = resolutionEligible.find' in audit
    assert 'Date.parse(p.observed_at) <= targetMs' in audit
    assert 'const fallbackCost = excursionPoints' in audit
    assert 'decision.action === "OPEN_SHORT" ? -downExcursion : upExcursion' in audit
    auditor = _text(AUDITOR)
    assert 'skipped_unresolved' in auditor
    assert 'fail_closed_no_zero_fallback' in auditor


def test_per_asset_poison_and_context_outage_fail_closed_without_killing_cycle():
    runtime = _text(COMPILER)
    assert 'const poisonByAsset = new Map<string, string>()' in runtime
    assert 'failClosedAssetDecision' in runtime
    assert 'intrabarContextAvailable = false' in runtime
    assert '"CONTEXT_UNAVAILABLE"' in runtime
    assert 'vetoFromPreliminary' in runtime
    assert 'official_macro_context:' in runtime
    assert 'book_ticker:' in runtime
    assert 'l2_${status.toLowerCase()}:${asset}:' in runtime


def test_all_alpha_writers_require_application_level_cron_auth():
    auth = _text(CRON_AUTH)
    assert 'x-brian-cron-key' in auth
    assert 'brian_dashboard_auth' in auth
    assert 'control-v2' in auth
    assert 'constantTimeEqual' in auth
    for path in (COMPILER, AUDITOR, MACRO):
        text = _text(path)
        assert 'import { requireCronAuth } from "../_shared/cron_auth.ts";' in text
        assert 'await requireCronAuth(req, supabase);' in text
        assert 'UNAUTHORIZED' in text


def test_alpha_files_contain_no_authenticated_or_live_execution_surface():
    paths = [COMPILER, AUDIT, DECISION, RUNTIME_POLICY, CRON_AUTH, AUDITOR, MACRO, *MIGRATIONS]
    forbidden = {
        "binance spot order endpoint": re.compile(r"/api/v3/order(?:\b|\?)", re.I),
        "binance futures order endpoint": re.compile(r"/fapi/v\d+/order(?:\b|\?)", re.I),
        "binance sapi execution surface": re.compile(r"/sapi/v\d+/", re.I),
        "binance api-key header": re.compile(r"X-MBX-APIKEY", re.I),
        "binance api secret env": re.compile(r"BINANCE_API_SECRET", re.I),
        "signed query parameter": re.compile(r"signature=", re.I),
        "HMAC constructor": re.compile(r"createHmac\s*\(", re.I),
        "WebCrypto signing": re.compile(r"crypto\.subtle\.sign\s*\(", re.I),
        "runtime live_execution true": re.compile(r"live_execution\s*[:=]\s*true", re.I),
        "runtime shadow_only false": re.compile(r"shadow_only\s*[:=]\s*false", re.I),
        "SQL live_execution default true": re.compile(r"live_execution\s+boolean[^;\n]*default\s+true", re.I),
    }
    failures: list[str] = []
    for path in paths:
        text = _text(path)
        for label, pattern in forbidden.items():
            if pattern.search(text):
                failures.append(f"{path.relative_to(ROOT)}: {label}")
    assert not failures, "forbidden ALPHA execution surface detected:\n" + "\n".join(failures)
