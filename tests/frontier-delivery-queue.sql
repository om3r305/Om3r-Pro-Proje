-- Run on Realtime; every fixture and acknowledgement rolls back.
begin;
set local statement_timeout='8s';
do $$
declare template jsonb; a jsonb; b jsonb; c jsonb; n integer;
 t1 uuid:=gen_random_uuid();t2 uuid:=gen_random_uuid();t3 uuid:=gen_random_uuid();
begin
 select to_jsonb(e) into template from public.brian_intel_events e
 where trust_class='OFFICIAL_PRIMARY' and event_kind ~* '^OFFICIAL_' limit 1;
 if template is null then raise exception 'official fixture unavailable'; end if;
 -- Temporarily isolate fixtures from real work. Transaction rollback restores everything.
 update public.brian_intel_delivery_queue set delivered_at=now() where delivered_at is null;
 insert into public.brian_intel_events
 select (jsonb_populate_record(null::public.brian_intel_events,template||jsonb_build_object(
   'event_id','frontier-queue-test-'||i,'first_observed_at','2020-01-01T00:00:00Z'))).*
 from generate_series(1,501) i;
 a:=public.brian_claim_intel_delivery(t1);
 b:=public.brian_claim_intel_delivery(t2);
 c:=public.brian_claim_intel_delivery(t3);
 if jsonb_array_length(a)<>250 or jsonb_array_length(b)<>250 or jsonb_array_length(c)<>1 then raise exception 'pagination lost records'; end if;
 if exists(select 1 from jsonb_array_elements(a) x join jsonb_array_elements(b) y on x->>'event_id'=y->>'event_id') then raise exception 'concurrent batch overlap';end if;
 n:=public.brian_ack_intel_delivery(t2,array(select x->>'event_id' from jsonb_array_elements(a)x));
 if n<>0 then raise exception 'wrong owner can acknowledge';end if;
 n:=public.brian_ack_intel_delivery(t1,array(select x->>'event_id' from jsonb_array_elements(a)x));
 if n<>250 then raise exception 'ack mismatch';end if;
 update public.brian_intel_delivery_queue set lease_until=now()-interval '1 second' where lease_token=t2;
 b:=public.brian_claim_intel_delivery(gen_random_uuid());
 if jsonb_array_length(b)<>250 then raise exception 'expired delivery did not retry';end if;
 if has_function_privilege('anon','public.brian_claim_intel_delivery(uuid)','execute') then raise exception 'anonymous queue access';end if;
end $$;
rollback;
