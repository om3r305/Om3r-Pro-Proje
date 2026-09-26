-- DIP guardian dispatch is cron/internal only.
-- pg_cron runs this as postgres; service_role keeps explicit maintenance access.
-- Anonymous/signed-in clients must not be able to trigger a SECURITY DEFINER
-- function that reads Vault secrets and dispatches an internal Edge function.

alter function public.brian_dip_position_guardian_dispatch()
  set search_path = pg_catalog, public, net, vault;

revoke execute on function public.brian_dip_position_guardian_dispatch()
  from public, anon, authenticated;

grant execute on function public.brian_dip_position_guardian_dispatch()
  to service_role;
