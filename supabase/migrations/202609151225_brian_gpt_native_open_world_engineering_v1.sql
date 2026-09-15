-- GPT-native engineering policy. DIP remains isolated and live writes remain exact-SHA gated.
update public.brian_evolution_engineering_control
set metadata = coalesce(metadata,'{}'::jsonb) || jsonb_build_object(
  'engineering_cognition_mode','GPT_NATIVE_TOOL_AUGMENTED_V1',
  'engineering_research_mode','OPEN_WORLD_READ_ONLY_TIERED_TRUST_V1',
  'task_evidence_policy','WORLD_REQUIRES_VERIFIED_SOURCE_CORE_ALLOWS_INTERNAL_EVIDENCE',
  'engineering_action_boundary','EXACT_SHA_GPT_GATE',
  'open_world_observation_enabled',true,
  'open_world_observation_policy','PUBLIC_READ_ONLY_UNTRUSTED_DATA',
  'open_world_research_breadth','BROAD_PUBLIC_WEB',
  'gpt_engineering_cockpit',true,
  'approval_display_actor','GPT',
  'world_library_ui_profile','TIERED_LIBRARY_V2',
  'library_direct_link_warning_scope','WORLD_OR_MIXED_ONLY',
  'core_internal_evidence_allowed',true,
  'policy_updated_at',now()
), updated_at=now()
where control_id='default';
