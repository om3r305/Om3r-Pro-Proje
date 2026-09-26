$ErrorActionPreference = "Stop"

$requiredSecrets = @(
  "BRIAN_SENSOR_SUPABASE_SECRET_KEY",
  "BRIAN_EDGE_SUPABASE_SECRET_KEY",
  "BRIAN_COST_SUPABASE_SECRET_KEY",
  "BRIAN_RUNTIME_SUPABASE_SECRET_KEY"
)

$missing = @()
foreach ($name in $requiredSecrets) {
  $value = [Environment]::GetEnvironmentVariable($name)
  if ([string]::IsNullOrWhiteSpace($value)) { $missing += $name }
}

if ($missing.Count -gt 0) {
  [Console]::Error.WriteLine("Phase117 blocked: missing scoped backend secret variables: " + ($missing -join ", "))
  exit 30
}

$env:BRIAN_SENSOR_SUPABASE_URL = "https://dliediwlldojkfjzlznm.supabase.co"
$env:BRIAN_EDGE_SUPABASE_URL = "https://qbcjuxhvhwagvqbjyemo.supabase.co"
$env:BRIAN_COST_SUPABASE_URL = "https://dliediwlldojkfjzlznm.supabase.co"
$env:BRIAN_RUNTIME_SUPABASE_URL = "https://dliediwlldojkfjzlznm.supabase.co"
$env:BRIAN_RUNTIME_ID = "brian-shadow-main"

python -m brian2026.phase117_readiness_guarded_crypto_shadow --policy "config/brian-shadow-machine-policy-v1.json" --runtime-id "brian-shadow-main"
exit $LASTEXITCODE
