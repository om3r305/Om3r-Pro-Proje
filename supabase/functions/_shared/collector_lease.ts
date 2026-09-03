// Collector-level atomic lease/mutex (Item 1, brian-2026 issue #32). Wraps the
// brian_acquire_collector_lease / brian_renew_collector_lease / brian_release_collector_lease
// Postgres RPCs (see supabase/migrations/202609030009_brian_collector_lease.sql) with the
// acquire -> heartbeat -> work -> release-in-finally pattern every Phase 3.8-4.0 collector needs.
//
// A contended lease returns before any collector work begins -- callers must not fetch market
// data, write sensor observations, or write a second virtual-book chain in that case. A fixed TTL
// alone is only safe for the *normal* runtime case: an invocation that is still genuinely alive
// but slower than its own lease_seconds (a slow upstream API, a slow DB round trip) would
// otherwise have its lease expire out from under it while still executing, letting a second
// invocation take over and run concurrently -- exactly the overlap Item 1 exists to prevent. So
// withCollectorLease renews the lease on a periodic heartbeat for as long as `work` is running;
// only once renewal calls stop (because `work` finished, or the process died) does the lease
// expire on its own TTL and become eligible for EXPIRED_RECOVERY takeover.
//
// See collector_lease.test.ts for the caller-side contract tests (mocked RPC client, no
// network/DB, using Deno's FakeTime to exercise the heartbeat without real delays); the migration
// file plus tests/test_brian2026_collector_lease_postgres.py cover real Postgres atomicity under
// concurrency, including the slow-but-alive-owner scenario this heartbeat closes.
//
// RpcClient is deliberately a minimal structural interface (not the full supabase-js
// SupabaseClient type) so a plain mock object can stand in for tests without any network or
// Supabase dependency, while the real client -- whose `.rpc()` already returns a thenable --
// still satisfies it unchanged.

export interface RpcClient {
  rpc(fn: string, params: Record<string, unknown>): PromiseLike<{ data: unknown; error: unknown }>;
}

export interface LeaseResult<T> {
  contended: boolean;
  ownerToken: string;
  value?: T;
}

export function randomOwnerToken(): string {
  return crypto.randomUUID();
}

export async function acquireCollectorLease(
  client: RpcClient,
  collectorId: string,
  ownerToken: string,
  leaseSeconds: number,
): Promise<boolean> {
  const { data, error } = await client.rpc("brian_acquire_collector_lease", {
    p_collector_id: collectorId,
    p_owner_token: ownerToken,
    p_lease_seconds: leaseSeconds,
  });
  if (error) throw error;
  return data === true;
}

export async function renewCollectorLease(
  client: RpcClient,
  collectorId: string,
  ownerToken: string,
  leaseSeconds: number,
): Promise<boolean> {
  const { data, error } = await client.rpc("brian_renew_collector_lease", {
    p_collector_id: collectorId,
    p_owner_token: ownerToken,
    p_lease_seconds: leaseSeconds,
  });
  if (error) throw error;
  return data === true;
}

export async function releaseCollectorLease(
  client: RpcClient,
  collectorId: string,
  ownerToken: string,
): Promise<boolean> {
  const { data, error } = await client.rpc("brian_release_collector_lease", {
    p_collector_id: collectorId,
    p_owner_token: ownerToken,
  });
  if (error) throw error;
  return data === true;
}

/**
 * Acquires the named collector's mutex, runs `work` if acquired, and releases the lease in a
 * finally block regardless of whether `work` throws or resolves. If the lease is contended,
 * `work` is never invoked and `{ contended: true }` is returned immediately.
 *
 * While `work` is running, the lease is renewed on a heartbeat every `floor(leaseSeconds / 3)`
 * seconds, so a slow-but-alive invocation keeps its lease well ahead of expiry rather than
 * relying on `work` finishing inside the original `leaseSeconds` window. If a renewal call
 * reports that ownership was lost (another invocation already took over, meaning this owner's
 * lease genuinely expired before it renewed in time), the heartbeat stops -- there is no other
 * owner left to protect against by continuing to call renew -- but `work` itself is not forcibly
 * aborted, since no collector here supports cooperative cancellation. Under correct operation
 * (heartbeat interval well below `leaseSeconds`) this should not happen in practice; it is a
 * narrower residual gap than -- and not a reintroduction of -- the no-renewal overlap this
 * function closes, since the database-level mutual exclusion (a second acquire while this owner
 * is still renewing) still holds throughout.
 *
 * If `work` throws, that error still propagates out of this function (after the heartbeat is
 * stopped and the release attempt completes) so the caller's own try/catch and failure-accounting
 * path still run -- a lease acquisition does not swallow collector failures. A failure to
 * release, or to renew, is logged and swallowed so it can never mask the real outcome of `work`.
 */
export async function withCollectorLease<T>(
  client: RpcClient,
  collectorId: string,
  leaseSeconds: number,
  work: (ownerToken: string) => Promise<T>,
): Promise<LeaseResult<T>> {
  const ownerToken = randomOwnerToken();
  const acquired = await acquireCollectorLease(client, collectorId, ownerToken, leaseSeconds);
  if (!acquired) return { contended: true, ownerToken };

  const renewIntervalMs = Math.max(1, Math.floor(leaseSeconds / 3)) * 1000;
  const heartbeat = setInterval(() => {
    renewCollectorLease(client, collectorId, ownerToken, leaseSeconds)
      .then((renewed) => {
        if (!renewed) {
          console.error(
            `collector lease renewal lost ownership for ${collectorId} (owner ${ownerToken}); another invocation has taken over -- stopping further renewal attempts`,
          );
          clearInterval(heartbeat);
        }
      })
      .catch((renewError) => {
        console.error(`collector lease renewal failed for ${collectorId}`, renewError);
      });
  }, renewIntervalMs);

  try {
    const value = await work(ownerToken);
    return { contended: false, ownerToken, value };
  } finally {
    clearInterval(heartbeat);
    try {
      await releaseCollectorLease(client, collectorId, ownerToken);
    } catch (releaseError) {
      console.error(`collector lease release failed for ${collectorId}`, releaseError);
    }
  }
}
