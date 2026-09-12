from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SANDBOX = (ROOT / "supabase/functions/brian-evolution-sandbox/index.ts").read_text(encoding="utf-8")
MIGRATION = (ROOT / "supabase/migrations/202609121720_brian_evolution_runtime_parent_config.sql").read_text(encoding="utf-8")

def test_sandbox_falls_back_to_owner_managed_canonical_parent_config():
    assert 'async function parentCommit(): Promise<string>' in SANDBOX
    assert 'brian_evolution_runtime_config' in SANDBOX
    assert '.eq("config_key", "canonical_parent_commit")' in SANDBOX
    assert 'const canonicalParent = await parentCommit();' in SANDBOX
    assert 'EVOLUTION_PARENT_COMMIT_MISSING' in SANDBOX
    assert 'EVOLUTION_PARENT_COMMIT_INVALID' in SANDBOX

def test_runtime_parent_config_is_read_only_to_service_role():
    lower = MIGRATION.lower()
    assert 'grant select on public.brian_evolution_runtime_config to service_role' in lower
    assert 'revoke insert, update, delete, truncate, references, trigger on public.brian_evolution_runtime_config from service_role' in lower
    assert "config_value ~ '^[0-9a-fa-f]{40}$'" in lower
