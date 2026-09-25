-- Engineering preflight is an internal control-plane RPC.
-- The Edge gateway calls it with service_role after GitHub OIDC validation.
-- Remove direct PostgREST execution from anonymous/signed-in clients.
revoke execute on function public.brian_engineering_preflight_v1() from public, anon, authenticated;
grant execute on function public.brian_engineering_preflight_v1() to service_role;
