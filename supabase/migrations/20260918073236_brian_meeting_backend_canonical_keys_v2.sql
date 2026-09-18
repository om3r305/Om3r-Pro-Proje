
-- Canonicalize backend meeting records by source URL so repeated RSS/article
-- observations do not create duplicate council incidents.
create or replace function public.brian_meeting_backend_canonicalize()
returns trigger
language plpgsql
set search_path to 'pg_catalog','public'
as $$
begin
  if coalesce(new.payload->>'backend_sync','')='brian.meeting-backend-sync.v1' then
    if coalesce(btrim(new.source_uri),'')<>'' then
      new.event_key := 'url:' || btrim(new.source_uri);
    end if;
    if coalesce(new.asset,'')<>'' then
      new.asset := regexp_replace(upper(new.asset),'[^A-Z0-9]','','g');
    end if;
    if coalesce(new.alpha_asset,'')<>'' then
      new.alpha_asset := regexp_replace(upper(new.alpha_asset),'[^A-Z0-9]','','g');
    end if;
  end if;
  return new;
end;
$$;

drop trigger if exists brian_meeting_backend_canonicalize_trg on public.brian_meeting_shadow_ledger;
create trigger brian_meeting_backend_canonicalize_trg
before insert or update on public.brian_meeting_shadow_ledger
for each row execute function public.brian_meeting_backend_canonicalize();

select public.brian_sync_meeting_shadow_ledger();

delete from public.brian_meeting_shadow_ledger old
where old.event_key like 'event:%'
  and old.payload->>'backend_sync'='brian.meeting-backend-sync.v1'
  and coalesce(btrim(old.source_uri),'')<>''
  and exists (
    select 1
    from public.brian_meeting_shadow_ledger canonical
    where canonical.event_key='url:'||btrim(old.source_uri)
  );
