begin;

revoke all on schema brian_private from public, anon, authenticated;

revoke all on function brian_private.claim_engineering_task(text,text,text) from public, anon, authenticated;
revoke all on function brian_private.record_engineering_event(uuid,text,text,boolean,text,jsonb) from public, anon, authenticated;
revoke all on function brian_private.measure_engineering_run(uuid,text,jsonb) from public, anon, authenticated;
revoke all on function brian_private.approve_engineering_run(uuid,text,text) from public, anon, authenticated;

commit;
