// One authoritative operator switch for both Realtime data schedulers.
// Unavailable control fails closed; a cached success may never override STOP.
export async function readOperatorState(key: string, fetcher: typeof fetch = fetch): Promise<{enabled:boolean;status:string}> {
  try {
    const response = await fetcher('https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-core-scheduler-bridge', {
      method:'POST', headers:{'content-type':'application/json','x-brian-internal-key':key},
      body:JSON.stringify({action:'operator_status'}), signal:AbortSignal.timeout(6000),
    });
    const body = await response.json();
    if (!response.ok || body.status !== 'SUCCESS' || typeof body.enabled !== 'boolean') {
      return {enabled:false,status:'CONTROL_UNAVAILABLE'};
    }
    return {enabled:body.enabled,status:body.enabled?'RUNNING':'OPERATOR_STOPPED'};
  } catch {
    return {enabled:false,status:'CONTROL_UNAVAILABLE'};
  }
}
