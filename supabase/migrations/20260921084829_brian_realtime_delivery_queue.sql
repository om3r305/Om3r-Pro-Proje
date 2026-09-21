-- Target: brian-realtime only. Original events are retained; acknowledgements are durable.
create table if not exists public.brian_intel_delivery_queue (
  event_id text primary key references public.brian_intel_events(event_id),
  queued_at timestamptz not null default now(),
  lease_token uuid,
  lease_until timestamptz,
  delivered_at timestamptz,
  attempts integer not null default 0
);
alter table public.brian_intel_delivery_queue enable row level security;
revoke all on public.brian_intel_delivery_queue from public, anon, authenticated;
grant all on public.brian_intel_delivery_queue to service_role;
create index if not exists brian_intel_delivery_pending_idx
  on public.brian_intel_delivery_queue(queued_at,event_id) where delivered_at is null;
create schema if not exists brian_private;
create or replace function brian_private.enqueue_intel_delivery()
returns trigger language plpgsql security definer set search_path=pg_catalog,public as $$
begin
  if (new.trust_class='OFFICIAL_PRIMARY' and new.event_kind ~* '^OFFICIAL_')
     or (new.trust_class='INDEPENDENT_PROFESSIONAL' and new.event_kind='DIRECT_WIRE_DISCOVERY'
         and new.source_kind='DIRECT_WIRE_INDEPENDENT'
         and new.metadata @> '{"direct_wire":true,"discovery_only":true}'::jsonb) then
    insert into public.brian_intel_delivery_queue(event_id) values(new.event_id)
    on conflict do nothing;
  end if;
  return new;
end $$;
revoke all on function brian_private.enqueue_intel_delivery() from public,anon,authenticated;
drop trigger if exists brian_intel_delivery_enqueue on public.brian_intel_events;
create trigger brian_intel_delivery_enqueue after insert or update of trust_class,event_kind,source_kind,metadata
on public.brian_intel_events for each row execute function brian_private.enqueue_intel_delivery();

create or replace function public.brian_claim_intel_delivery(p_token uuid)
returns jsonb language sql security invoker set search_path=pg_catalog,public as $$
  with picked as (
    select event_id from public.brian_intel_delivery_queue
    where delivered_at is null and (lease_until is null or lease_until<now())
    order by queued_at,event_id for update skip locked limit 250
  ), claimed as (
    update public.brian_intel_delivery_queue q
    set lease_token=p_token,lease_until=now()+interval '90 seconds',attempts=attempts+1
    from picked p where q.event_id=p.event_id returning q.event_id
  )
  select coalesce(jsonb_agg(to_jsonb(e)), '[]'::jsonb)
  from claimed c join public.brian_intel_events e using(event_id);
$$;
create or replace function public.brian_ack_intel_delivery(p_token uuid,p_ids text[])
returns integer language sql security invoker set search_path=pg_catalog,public as $$
  with acknowledged as (
    update public.brian_intel_delivery_queue set delivered_at=now(),lease_until=null
    where lease_token=p_token and event_id=any(p_ids) and delivered_at is null
    returning event_id
  ) select count(*)::integer from acknowledged;
$$;
revoke all on function public.brian_claim_intel_delivery(uuid),public.brian_ack_intel_delivery(uuid,text[]) from public,anon,authenticated;
grant execute on function public.brian_claim_intel_delivery(uuid),public.brian_ack_intel_delivery(uuid,text[]) to service_role;
-- Recover all retained eligible history, including any previously skipped pages.
insert into public.brian_intel_delivery_queue(event_id)
select event_id from public.brian_intel_events
where (trust_class='OFFICIAL_PRIMARY' and event_kind ~* '^OFFICIAL_')
   or (trust_class='INDEPENDENT_PROFESSIONAL' and event_kind='DIRECT_WIRE_DISCOVERY'
       and source_kind='DIRECT_WIRE_INDEPENDENT'
       and metadata @> '{"direct_wire":true,"discovery_only":true}'::jsonb)
on conflict do nothing;
