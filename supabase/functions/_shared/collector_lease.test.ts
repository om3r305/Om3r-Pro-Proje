// Caller-side contract tests for
// withCollectorLease/acquireCollectorLease/renewCollectorLease/releaseCollectorLease.
// These use a mocked RpcClient (no network, no Postgres) to prove the calling contract: a
// contended lease returns before any collector work begins and records the expected
// status/metadata, work-thrown errors still propagate after release is attempted, a failed
// release never masks the real outcome, and the heartbeat renews while work is alive and stops
// once ownership is lost (using Deno's FakeTime so this needs no real delays). Real Postgres
// atomicity under concurrency -- including the slow-but-alive-owner scenario the heartbeat
// closes -- is proven separately in tests/test_brian2026_collector_lease_postgres.py against a
// real database.

import { assert, assertEquals, assertRejects } from "jsr:@std/assert@^1.0.0";
import { FakeTime } from "jsr:@std/testing@^1.0.0/time";
import { acquireCollectorLease, releaseCollectorLease, renewCollectorLease, type RpcClient, withCollectorLease } from "./collector_lease.ts";

interface RpcCall {
  fn: string;
  params: Record<string, unknown>;
}

function mockClient(responses: Record<string, { data: unknown; error: unknown }>): { client: RpcClient; calls: RpcCall[] } {
  const calls: RpcCall[] = [];
  const client: RpcClient = {
    rpc(fn, params) {
      calls.push({ fn, params });
      const response = responses[fn] ?? { data: null, error: new Error(`no mock response for ${fn}`) };
      return Promise.resolve(response);
    },
  };
  return { client, calls };
}

Deno.test("acquireCollectorLease: true data means acquired", async () => {
  const { client } = mockClient({ brian_acquire_collector_lease: { data: true, error: null } });
  assertEquals(await acquireCollectorLease(client, "brian-intrabar-eye", "token-1", 55), true);
});

Deno.test("acquireCollectorLease: false data means contended, not an error", async () => {
  const { client } = mockClient({ brian_acquire_collector_lease: { data: false, error: null } });
  assertEquals(await acquireCollectorLease(client, "brian-intrabar-eye", "token-1", 55), false);
});

Deno.test("acquireCollectorLease: an RPC error is rethrown, not swallowed as contention", async () => {
  const { client } = mockClient({ brian_acquire_collector_lease: { data: null, error: new Error("connection reset") } });
  await assertRejects(() => acquireCollectorLease(client, "brian-intrabar-eye", "token-1", 55), Error, "connection reset");
});

Deno.test("releaseCollectorLease: true data means released", async () => {
  const { client } = mockClient({ brian_release_collector_lease: { data: true, error: null } });
  assertEquals(await releaseCollectorLease(client, "brian-intrabar-eye", "token-1"), true);
});

Deno.test("renewCollectorLease: true data means renewed", async () => {
  const { client } = mockClient({ brian_renew_collector_lease: { data: true, error: null } });
  assertEquals(await renewCollectorLease(client, "brian-intrabar-eye", "token-1", 55), true);
});

Deno.test("renewCollectorLease: false data means ownership was lost, not an error", async () => {
  const { client } = mockClient({ brian_renew_collector_lease: { data: false, error: null } });
  assertEquals(await renewCollectorLease(client, "brian-intrabar-eye", "token-1", 55), false);
});

Deno.test("renewCollectorLease: an RPC error is rethrown, not swallowed as lost ownership", async () => {
  const { client } = mockClient({ brian_renew_collector_lease: { data: null, error: new Error("connection reset") } });
  await assertRejects(() => renewCollectorLease(client, "brian-intrabar-eye", "token-1", 55), Error, "connection reset");
});

Deno.test("withCollectorLease: contended acquire returns immediately without running work", async () => {
  const { client, calls } = mockClient({ brian_acquire_collector_lease: { data: false, error: null } });
  let workRan = false;
  const result = await withCollectorLease(client, "brian-sensor-mesh", 240, () => {
    workRan = true;
    return Promise.resolve("should not happen");
  });
  assertEquals(result.contended, true);
  assertEquals(workRan, false, "work must never run when the lease is contended");
  assertEquals(calls.length, 1, "a contended lease must not call release -- there is nothing to release");
  assertEquals(calls[0].fn, "brian_acquire_collector_lease");
  assertEquals(calls[0].params.p_collector_id, "brian-sensor-mesh");
  assertEquals(calls[0].params.p_lease_seconds, 240);
});

Deno.test("withCollectorLease: successful acquire runs work exactly once and releases with the same owner token", async () => {
  const { client, calls } = mockClient({
    brian_acquire_collector_lease: { data: true, error: null },
    brian_release_collector_lease: { data: true, error: null },
  });
  let workCallCount = 0;
  let workSawOwnerToken = "";
  const result = await withCollectorLease(client, "brian-derivatives-eye", 240, (ownerToken) => {
    workCallCount += 1;
    workSawOwnerToken = ownerToken;
    return Promise.resolve(42);
  });
  assertEquals(result.contended, false);
  assertEquals(result.value, 42);
  assertEquals(workCallCount, 1);
  assertEquals(calls.length, 2);
  assertEquals(calls[0].fn, "brian_acquire_collector_lease");
  assertEquals(calls[1].fn, "brian_release_collector_lease");
  assertEquals(calls[0].params.p_owner_token, workSawOwnerToken, "the owner token passed to work must be the same one used to acquire");
  assertEquals(calls[1].params.p_owner_token, workSawOwnerToken, "release must use the same owner token that acquired the lease");
});

Deno.test("withCollectorLease: a throwing work still releases the lease, then the original error propagates", async () => {
  const { client, calls } = mockClient({
    brian_acquire_collector_lease: { data: true, error: null },
    brian_release_collector_lease: { data: true, error: null },
  });
  const workError = new Error("collector work failed");
  await assertRejects(
    () =>
      withCollectorLease(client, "brian-news-eye", 300, () => {
        throw workError;
      }),
    Error,
    "collector work failed",
  );
  assertEquals(calls.map((c) => c.fn), ["brian_acquire_collector_lease", "brian_release_collector_lease"], "release must still be attempted after work throws");
});

Deno.test("withCollectorLease: a failed release does not mask a successful work result", async () => {
  const { client } = mockClient({
    brian_acquire_collector_lease: { data: true, error: null },
    brian_release_collector_lease: { data: null, error: new Error("release RPC unreachable") },
  });
  const result = await withCollectorLease(client, "brian-fx-eye", 300, () => Promise.resolve("ok"));
  assertEquals(result.contended, false);
  assertEquals(result.value, "ok", "a release failure must be logged and swallowed, not surface as the work's outcome");
});

Deno.test("withCollectorLease: distinct invocations use distinct owner tokens", async () => {
  const seen: unknown[] = [];
  const client: RpcClient = {
    rpc(fn, params) {
      if (fn === "brian_acquire_collector_lease") seen.push(params.p_owner_token);
      return Promise.resolve({ data: true, error: null });
    },
  };
  await withCollectorLease(client, "brian-universe-collector", 420, () => Promise.resolve(undefined));
  await withCollectorLease(client, "brian-universe-collector", 420, () => Promise.resolve(undefined));
  assert(seen[0] !== seen[1], "two separate invocations must not reuse the same owner token");
});

Deno.test("withCollectorLease: renews the lease on a heartbeat while work is still running", async () => {
  const time = new FakeTime();
  try {
    const renewOwnerTokens: unknown[] = [];
    let resolveWork: (() => void) | undefined;
    const workGate = new Promise<void>((resolve) => { resolveWork = resolve; });
    const client: RpcClient = {
      rpc(fn, params) {
        if (fn === "brian_acquire_collector_lease") return Promise.resolve({ data: true, error: null });
        if (fn === "brian_renew_collector_lease") {
          renewOwnerTokens.push(params.p_owner_token);
          return Promise.resolve({ data: true, error: null });
        }
        if (fn === "brian_release_collector_lease") return Promise.resolve({ data: true, error: null });
        return Promise.resolve({ data: null, error: new Error(`unexpected rpc ${fn}`) });
      },
    };
    // 60s lease -> heartbeat interval is floor(60/3) = 20s.
    const resultPromise = withCollectorLease(client, "brian-intrabar-eye", 60, async () => {
      await workGate;
      return "done";
    });
    await time.tickAsync(20_000);
    await time.tickAsync(20_000);
    assertEquals(renewOwnerTokens.length, 2, "two 20s ticks on a 60s lease must trigger exactly two renewals");
    assert(renewOwnerTokens[0] === renewOwnerTokens[1], "every renewal for one invocation must use the same owner token");
    resolveWork?.();
    const result = await resultPromise;
    assertEquals(result.contended, false);
    assertEquals(result.value, "done");
  } finally {
    time.restore();
  }
});

Deno.test("withCollectorLease: stops renewing once a renewal reports lost ownership", async () => {
  const time = new FakeTime();
  try {
    let renewCallCount = 0;
    let resolveWork: (() => void) | undefined;
    const workGate = new Promise<void>((resolve) => { resolveWork = resolve; });
    const client: RpcClient = {
      rpc(fn) {
        if (fn === "brian_acquire_collector_lease") return Promise.resolve({ data: true, error: null });
        if (fn === "brian_renew_collector_lease") {
          renewCallCount += 1;
          return Promise.resolve({ data: false, error: null }); // ownership already lost
        }
        if (fn === "brian_release_collector_lease") return Promise.resolve({ data: true, error: null });
        return Promise.resolve({ data: null, error: new Error("unexpected rpc") });
      },
    };
    // 30s lease -> heartbeat interval is floor(30/3) = 10s.
    const resultPromise = withCollectorLease(client, "brian-sensor-mesh", 30, async () => {
      await workGate;
      return "done";
    });
    await time.tickAsync(10_000); // first heartbeat: renewal reports ownership lost, must stop
    await time.tickAsync(10_000); // a second tick must NOT produce a second renewal call
    assertEquals(renewCallCount, 1, "renewal must stop after ownership is lost, not keep retrying forever");
    resolveWork?.();
    await resultPromise;
  } finally {
    time.restore();
  }
});
