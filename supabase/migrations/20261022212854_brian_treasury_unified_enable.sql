-- Operator-approved switch-on (2026-10-22, "birleşik kasa sistemini bu tasarımla kur"): the SHADOW
-- treasury now uses the unified evidence lane (gate v7). Instant rollback: set this back to 'false'.
update public.brian_evolution_runtime_config
set config_value = 'true', updated_at = now()
where config_key = 'brian_treasury_unified_enabled';
