-- Brian Engineer continuity metadata v2. DIP remains a protected boundary.
update public.brian_evolution_engineering_control
set metadata=coalesce(metadata,'{}'::jsonb)
  - 'provider_circuit_reason'
  - 'provider_circuit_open_until'
  || jsonb_build_object(
       'provider_continuity_enabled',true,
       'provider_continuity_policy','HOSTED_COPILOT_EXTERNAL_BYOK_LOCAL_OLLAMA_V2',
       'local_fallback_provider','OLLAMA_QWEN3_4B',
       'local_fallback_model','qwen3:4b',
       'local_fallback_offline_capable',true,
       'github_models_retired',true,
       'dip_protected',true,
       'provider_continuity_updated_at',now()
     ),
    updated_at=now()
where control_id='default';
