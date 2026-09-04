-- Brian Control Center auth rotation v2.
-- Plaintext dashboard/cron keys are intentionally NOT stored in git.
-- Existing v1 row remains append-only; runtime selects control-v2.

alter table public.brian_dashboard_auth
  drop constraint if exists brian_dashboard_auth_auth_id_check;

alter table public.brian_dashboard_auth
  add constraint brian_dashboard_auth_auth_id_check
  check (auth_id ~ '^control-v[0-9]+$');

insert into public.brian_dashboard_auth (
  auth_id,
  dashboard_key_sha256,
  cron_key_sha256,
  shadow_only,
  live_execution
) values (
  'control-v2',
  '8476c10a919ba17b4955b1472e56258e02c720a5824e35968641d9178a171ccf',
  'ebe7469778033743d48f7bae68da6b247b329b9b8d16fa8b1dc748bd3485f6d4',
  true,
  false
)
on conflict (auth_id) do nothing;
