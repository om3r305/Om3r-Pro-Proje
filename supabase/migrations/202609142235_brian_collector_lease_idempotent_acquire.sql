-- Make acquire retry-safe when the first successful PostgREST response is lost.
create or replace function public.brian_acquire_collector_lease(
  p_collector_id text,
  p_owner_token text,
  p_lease_seconds integer
)
returns boolean
language plpgsql
security definer
set search_path to 'pg_catalog', 'public'
as $function$
declare
  v_now timestamptz := clock_timestamp();
  v_prior_owner text;
  v_prior_lease_until timestamptz;
  v_had_prior boolean;
  v_rows integer;
  v_acquired boolean;
begin
  if p_collector_id is null or length(trim(p_collector_id))=0 then
    raise exception 'BRIAN_LEASE: p_collector_id is required';
  end if;
  if p_owner_token is null or length(trim(p_owner_token))=0 then
    raise exception 'BRIAN_LEASE: p_owner_token is required';
  end if;
  if p_lease_seconds is null or p_lease_seconds<=0 then
    raise exception 'BRIAN_LEASE: p_lease_seconds must be positive';
  end if;

  select owner_token, lease_until
    into v_prior_owner, v_prior_lease_until
  from public.brian_collector_leases
  where collector_id=p_collector_id
  for update;
  v_had_prior := found;

  if v_had_prior
     and v_prior_owner = p_owner_token
     and v_prior_lease_until > v_now then
    update public.brian_collector_leases
      set lease_until=v_now+make_interval(secs=>p_lease_seconds),
          updated_at=v_now
    where collector_id=p_collector_id and owner_token=p_owner_token;

    insert into public.brian_collector_lease_events(
      collector_id,owner_token,event,observed_at,lease_until,metadata
    ) values (
      p_collector_id,p_owner_token,'RENEWED',v_now,
      v_now+make_interval(secs=>p_lease_seconds),
      jsonb_build_object('lease_seconds',p_lease_seconds,'via','acquire_retry')
    );
    return true;
  end if;

  insert into public.brian_collector_leases(
    collector_id,owner_token,acquired_at,lease_until,updated_at
  ) values (
    p_collector_id,p_owner_token,v_now,
    v_now+make_interval(secs=>p_lease_seconds),v_now
  )
  on conflict(collector_id) do update
    set owner_token=excluded.owner_token,
        acquired_at=excluded.acquired_at,
        lease_until=excluded.lease_until,
        updated_at=excluded.updated_at
  where public.brian_collector_leases.lease_until<=v_now;

  get diagnostics v_rows=row_count;
  v_acquired := v_rows>0;

  insert into public.brian_collector_lease_events(
    collector_id,owner_token,event,observed_at,lease_until,metadata
  ) values (
    p_collector_id,p_owner_token,
    case
      when not v_acquired then 'BLOCKED_ACTIVE'
      when v_had_prior and v_prior_lease_until<=v_now then 'EXPIRED_RECOVERY'
      else 'ACQUIRED'
    end,
    v_now,
    case when v_acquired then v_now+make_interval(secs=>p_lease_seconds) else v_prior_lease_until end,
    jsonb_build_object('lease_seconds',p_lease_seconds,'had_prior_lease',v_had_prior)
  );

  return v_acquired;
end;
$function$;
