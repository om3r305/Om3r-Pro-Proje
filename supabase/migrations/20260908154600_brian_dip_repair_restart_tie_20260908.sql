-- One-time production reconciliation for the restart created on 2026-09-08.
-- The migration is intentionally guarded: on databases without this exact session/collision it is a no-op.
update public.brian_dip_session_events s
set requested_at = s.requested_at + interval '1 microsecond'
where s.session_id = 'dip-20260908153736-4280649e'
  and s.event_kind = 'START'
  and exists (
    select 1
    from public.brian_dip_session_events p
    where p.event_kind = 'PAUSE'
      and p.requested_at = s.requested_at
      and p.session_id <> s.session_id
  );
