-- Harden the big-move trigger against mutable search_path resolution.
-- Function body only uses NEW row fields and built-ins; behavior is unchanged.
alter function public.brian_guard_big_move_opportunity()
  set search_path = pg_catalog, public;
