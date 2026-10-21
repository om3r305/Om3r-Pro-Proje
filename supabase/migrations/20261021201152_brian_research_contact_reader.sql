-- Operator contact for official sources that require a declared User-Agent (SEC fair access:
-- "Company Name contact@domain"). The value lives in Supabase Vault (brian_research_contact_email),
-- never in the repository. Only service_role (edge functions) can read it.
create or replace function public.brian_research_contact_v1()
returns text
language sql
stable
security definer
set search_path = pg_catalog, vault
as $fn$
  select nullif(btrim(decrypted_secret), '')
  from vault.decrypted_secrets
  where name = 'brian_research_contact_email'
  order by created_at desc
  limit 1
$fn$;

revoke all on function public.brian_research_contact_v1() from public, anon, authenticated;
grant execute on function public.brian_research_contact_v1() to service_role;
