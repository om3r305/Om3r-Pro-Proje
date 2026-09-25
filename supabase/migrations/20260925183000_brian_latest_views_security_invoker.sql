-- Latest-market/alpha convenience views must honor the caller's RLS context.
-- Both underlying tables are internal and RLS-protected; service_role retains access.
-- This removes SECURITY DEFINER view semantics without changing view definitions.

do $$
begin
  if to_regclass('public.brian_multiasset_market_latest') is not null then
    execute 'alter view public.brian_multiasset_market_latest set (security_invoker = true)';
  end if;

  if to_regclass('public.brian_multiasset_alpha_latest') is not null then
    execute 'alter view public.brian_multiasset_alpha_latest set (security_invoker = true)';
  end if;
end
$$;
