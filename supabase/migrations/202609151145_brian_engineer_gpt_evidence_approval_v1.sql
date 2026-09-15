-- Semantic approval policy. Legacy HUMAN_APPROVAL columns/state remain intact
-- for state-machine compatibility; decision authority is gpt-evidence-gate.
-- DIP remains a protected and isolated scope.
update public.brian_evolution_engineering_control
set metadata = coalesce(metadata,'{}'::jsonb) || jsonb_build_object(
  'approval_mode','GPT_EVIDENCE_GATE_V1',
  'gpt_approval_required',true,
  'gpt_approval_actor','gpt-evidence-gate',
  'human_click_required',false,
  'legacy_human_approval_columns',true,
  'gpt_release_backend_enabled',true,
  'gpt_release_web_enabled',false,
  'gpt_release_web_reason','FAIL_CLOSED_TO_PRESERVE_DIP_ISOLATION_WHILE_PRODUCTION_BASE_DRIFTS',
  'gpt_reject_retry_limit',2,
  'dip_protected',true,
  'approval_policy_updated_at',now()
), updated_at=now()
where control_id='default';
