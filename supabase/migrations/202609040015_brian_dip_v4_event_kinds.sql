-- Dip Expert V4 persistence parity. SHADOW ONLY; no live-execution surface is added.
-- V3 backend already accepts SHORT_OPEN / SHORT_CLOSE, but the database CHECK was still V1-only.

alter table public.brian_dip_events
  drop constraint if exists brian_dip_events_event_kind_check;

alter table public.brian_dip_events
  add constraint brian_dip_events_event_kind_check
  check (event_kind in (
    'ENGINE_START','ENGINE_PAUSE','DIP_ARMED','DIP_NEW_LOW',
    'BUY','SELL','SHORT_OPEN','SHORT_CLOSE',
    'SKIP_CHASE','NO_CASH','INFO'
  ));

comment on constraint brian_dip_events_event_kind_check on public.brian_dip_events is
  'Append-only SHADOW event kinds for Brian Aggressive Dip V4. SHORT_* are simulated USD-M perpetual events; live_execution remains false.';
