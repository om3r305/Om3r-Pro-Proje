-- Research ledger writers are operator-only: remove the EXECUTE that Supabase default privileges
-- grant to service_role on new functions (found on CORE right after 20261020211802 was applied;
-- the local test database has no such default, so tests now simulate it).
revoke execute on function public.brian_record_research_ledger_v1(jsonb, text, text) from service_role;
revoke execute on function public.brian_record_research_result_v1(jsonb) from service_role;
