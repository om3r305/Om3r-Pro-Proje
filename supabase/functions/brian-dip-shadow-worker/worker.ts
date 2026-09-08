import type { SupabaseClient } from "npm:@supabase/supabase-js@2.116.0";
import {
  calibrate,
  closeLong,
  ENGINE_VERSION,
  evaluatePath,
  hash,
  type J,
  MAX_HOLD_MS,
  METRIC_VERSION,
  n,
  POLICY_VERSION,
  type Resolution,
  type Runtime,
  SYMBOL,
  validateSession,
} from "../_shared/dip_v8.ts";
import { getMarket, pricePath } from "../_shared/dip_v8_market.ts";
import { candidate } from "./decision.ts";
export async function readRuntime(
  db: SupabaseClient,
  sessionId: string,
): Promise<{ runtime: Runtime; state_version: number }> {
  const q = await db.from("brian_dip_v8_runtime").select(
    "runtime,state_version",
  ).eq("session_id", sessionId).maybeSingle();
  if (q.error) throw Error("READ_FAILED:" + q.error.message);
  if (!q.data) throw Error("V8_STATE_MISSING_RECONCILE");
  const rt = q.data.runtime as Runtime;
  if (
    !rt ||
    ![
      rt.start,
      rt.cash,
      rt.realized,
      rt.trades,
      rt.wins,
      rt.losses,
      rt.marketCursor,
      Number(q.data.state_version),
    ].every((x) => typeof x === "number" && Number.isFinite(x)) ||
    rt.start <= 0 || rt.cash < 0 || !Object.hasOwn(rt, "pos")
  ) throw Error("INVALID_RUNTIME_RECONCILE");
  return {
    runtime: structuredClone(rt),
    state_version: Number(q.data.state_version),
  };
}
export async function runWorker(
  db: SupabaseClient,
  owner: string,
  assertOwned: () => void,
  fetcher: typeof fetch = fetch,
): Promise<J> {
  const q = await db.from("brian_dip_session_events").select("*").order(
    "requested_at",
    { ascending: false },
  ).order("event_id", { ascending: false }).limit(1).maybeSingle();
  if (q.error) throw Error("SESSION_READ_FAILED:" + q.error.message);
  const sess = q.data;
  if (!sess) {
    return { status: "NO_ACTIVE_SESSION", worker_version: POLICY_VERSION };
  }
  const cfg = sess.config as J;
  try {
    validateSession(cfg);
  } catch (e) {
    return {
      status: String((e as Error).message),
      worker_version: POLICY_VERSION,
    };
  }
  const sid = String(sess.session_id),
    loaded = await readRuntime(db, sid),
    rt = loaded.runtime;
  if (sess.event_kind !== "START" && !rt.pos) {
    return {
      status: "PAUSED",
      session_id: sid,
      worker_version: POLICY_VERSION,
    };
  }

  const getCal = async (setup: string, direction: string, regime: string) => {
    if (direction === "WAIT") return calibrate([]);
    const c = await db.rpc("brian_dip_v8_calibration", {
      p_setup: setup,
      p_direction: direction,
      p_regime: regime,
    });
    return c.error ? calibrate([], true) : calibrate(c.data || []);
  };
  const events: J[] = [];
  let pathError: string | null = null, resolution: Resolution | null = null;
  if (rt.pos) {
    const p = rt.pos,
      start = p.checked_until || Date.parse(p.opened_at),
      due = Date.parse(p.due_at);
    try {
      const path = await pricePath(
        start,
        due,
        Date.now(),
        p.market_price || p.entry,
        fetcher,
      );
      resolution = evaluatePath({
        direction: "UP",
        target: p.target,
        stop: p.stop,
        start,
        due,
        now: Date.now(),
        end: path.end,
        entry: p.market_price || p.entry,
        segments: path.segments,
      });
      if (resolution.reason === "INDETERMINATE") {
        throw Error("PRICE_PATH_INDETERMINATE");
      }
      p.checked_until = resolution.checkedUntil;
      p.market_price = resolution.price;
      const closed = closeLong(
        rt,
        resolution,
        String(
          (rt.latestThesis?.structure as J | undefined)?.fingerprint || "",
        ),
        Number(
          rt.latestThesis?.signal_at
            ? Date.parse(String(rt.latestThesis.signal_at))
            : 0,
        ),
        Date.now(),
      );
      if (closed) events.push(closed);
    } catch (e) {
      pathError = e instanceof Error ? e.message : String(e);
    }
  }
  // Exit monitoring is independent of multi-timeframe entry data health.
  let market: Awaited<ReturnType<typeof getMarket>> | null = null;
  let marketError: string | null = null;
  try {
    market = await getMarket(fetcher);
  } catch (e) {
    marketError = e instanceof Error ? e.message : String(e);
  }
  assertOwned();
  const c: Awaited<ReturnType<typeof candidate>> = market
    ? await candidate(market, sid, rt, cfg, Date.now(), getCal)
    : {
      thesis: {
        ...(rt.latestThesis || {}),
        thesis_state: "WAIT",
        veto: ["DATA_UNAVAILABLE:" + marketError],
        structure: rt.latestThesis?.structure || { s1: {} },
      },
      decision: null,
      occurrence: "",
      episode: "",
      combinedFp: "",
      last5m: 0,
      direction: "WAIT",
      entry: 0,
      inv: null,
      target: null,
      size: null,
      fee: n(cfg.fee_bps, 10),
      slip: n(cfg.slippage_bps, 1),
      canEnter: false,
    };
  const at = Date.now();
  // Freeze after calibration and every decision input has arrived; never backdate a fill.
  c.thesis.generated_at = c.thesis.decision_time = new Date(at).toISOString();
  if (c.decision) {
    c.decision.decision_at = new Date(at).toISOString();
    c.decision.due_at = new Date(at + MAX_HOLD_MS).toISOString();
  }
  // Only an immutable first decision for a new occurrence can open. No same-run re-entry.
  let decision = c.decision && rt.lastOccurrence !== c.occurrence
    ? c.decision
    : null;
  if (decision) {
    const existing = await db.from("brian_dip_v8_decisions").select(
      "occurrence_id",
    ).eq("occurrence_id", c.occurrence).maybeSingle();
    if (existing.error) {
      throw Error("DECISION_READ_FAILED:" + existing.error.message);
    }
    if (existing.data) decision = null;
  }
  const stale = !market || Date.now() - market.book.receivedAt > 15_000 ||
    Date.now() - at > 20_000;
  if (stale) (c.thesis.veto as string[]).push("STALE_DATA");
  if (pathError) {
    (c.thesis.veto as string[]).push("RECONCILIATION_REQUIRED:" + pathError);
  }
  if (events.length) (c.thesis.veto as string[]).push("CLOSED_THIS_RUN");
  if (sess.event_kind !== "START") {
    (c.thesis.veto as string[]).push("SESSION_PAUSED");
  }
  if (
    market && c.canEnter && decision && !rt.pos && !events.length &&
    !pathError &&
    !stale && sess.event_kind === "START" && c.target && c.inv && c.size
  ) {
    const s = c.size, openedAt = new Date(at).toISOString();
    rt.pos = {
      side: "LONG",
      position_id: "v8-position-" + c.occurrence,
      thesis_id: c.occurrence,
      episode_id: c.episode,
      setup: String(c.thesis.setup),
      regime: String(c.thesis.regime),
      entry: c.entry,
      qty: s.qty,
      notional: s.notional,
      target: c.target,
      stop: c.inv,
      opened_at: openedAt,
      due_at: new Date(at + MAX_HOLD_MS).toISOString(),
      fees_open: s.fees_open,
      fee_bps: c.fee,
      slippage_bps: c.slip,
      spread_bps: market.book.spreadBps,
      venue: "SPOT",
      policy_version: POLICY_VERSION,
      checked_until: at,
      market_price: c.entry,
      actual_fraction: s.actual_fraction,
    };
    rt.cash -= s.notional + s.fees_open;
    events.push({
      event_kind: "BUY",
      position_id: rt.pos.position_id,
      occurrence_id: c.occurrence,
      episode_id: c.episode,
      price: c.entry,
      entry_price: c.entry,
      quantity: s.qty,
      notional: s.notional,
      fees: s.fees_open,
      realized_pnl: 0,
      cash_after: rt.cash,
      equity_after: rt.cash + s.qty * market.book.bid,
      metadata: {
        server_v8: true,
        thesis_id: c.occurrence,
        setup: c.thesis.setup,
        regime: c.thesis.regime,
        venue: "SPOT",
        target: c.target,
        stop: c.inv,
        rr: c.thesis.rr,
        raw_conviction: c.thesis.raw_conviction,
        calibrated_probability: c.thesis.calibrated_probability,
        calibration_samples: c.thesis.calibration_samples,
        actual_fraction: s.actual_fraction,
        policy_version: POLICY_VERSION,
        fee_bps: c.fee,
        slippage_bps: c.slip,
      },
    });
  }
  if (decision) {
    rt.lastOccurrence = c.occurrence;
    decision.evidence = { ...decision.evidence, veto: c.thesis.veto as string[] };
  }
  if ((c.thesis.veto as string[]).some((x) => x !== "CALIBRATING")) {
    c.thesis.thesis_state = "WAIT";
  }
  rt.latestThesis = c.thesis;
  if (market) {
    rt.marketCursor = Math.max(
      rt.marketCursor,
      market.bars["1m"].at(-1)!.ct + 1,
    );
  }
  const recordedAt = new Date().toISOString(), hour = recordedAt.slice(0, 13);
  rt.lastSnapshotHour = hour;
  const mark = (market?.book.bid ?? rt.pos?.market_price ?? 0) *
      (1 - n(cfg.slippage_bps, 1) / 10000),
    liquidationValue = rt.pos
      ? rt.pos.qty * mark * (1 - rt.pos.fee_bps / 10000)
      : 0;
  const equity = rt.cash + liquidationValue,
    unrealized = equity - rt.start - rt.realized;
  const state = {
    start: rt.start,
    cfg: {
      ...cfg,
      symbols: [SYMBOL],
      engine_version: ENGINE_VERSION,
      universe_size: 1,
    },
    symbols: {
      [SYMBOL]: {
        symbol: SYMBOL,
        last: market?.book.mid ?? rt.pos?.market_price ?? null,
        price: market?.book.mid ?? rt.pos?.market_price ?? null,
        pos: rt.pos,
        dip: (c.thesis.structure as { s1: { lastLow?: { p?: number } } }).s1
          .lastLow?.p || null,
        armed: false,
        lastAction: rt.pos ? "LONG" : "WATCH",
        thesis: c.thesis,
        v4: {
          phase: "WATCH",
          lastVeto: (c.thesis.veto as string[]).join(" · "),
        },
      },
    },
    v8: rt,
    serverRuntime: {
      authoritative: true,
      worker_version: POLICY_VERSION,
      policy_version: POLICY_VERSION,
      generated_at: recordedAt,
      market_available_at: market
        ? new Date(market.availableAt).toISOString()
        : null,
      state_version: loaded.state_version + 1,
      browser_executor_disabled: true,
      shadow_only: true,
      live_execution: false,
      universe: [SYMBOL],
      focus: "ETH_ONLY",
      execution_mode: cfg.execution_mode,
      measurement: METRIC_VERSION,
      decision_cadence_seconds: 180,
      status: pathError
        ? "RECONCILIATION_REQUIRED"
        : marketError
        ? "DATA_UNAVAILABLE"
        : "OK",
      market_error: marketError,
      valuation_fresh: !!market,
      path_error: pathError,
    },
  };
  const snapshot = {
    snapshot_id: "v8-hour-" + sid + "-" + hour,
    session_id: sid,
    observed_at: recordedAt,
    cash: rt.cash,
    equity,
    realized_pnl: rt.realized,
    unrealized_pnl: unrealized,
    trade_count: rt.trades,
    win_count: rt.wins,
    loss_count: rt.losses,
    state,
  };
  for (const e of events) {
    e.transition_id = "v8-" + String(e.event_kind).toLowerCase() + "-" +
      e.occurrence_id;
  }
  const commitId = await hash(
    [sid, loaded.state_version, rt.marketCursor, POLICY_VERSION]
      .join("|"),
  );
  assertOwned();
  const committed = await db.rpc("brian_dip_v8_commit", {
    p_session_id: sid,
    p_expected_version: loaded.state_version,
    p_owner_token: owner,
    p_commit_id: commitId,
    p_runtime: rt,
    p_snapshot: snapshot,
    p_decision: decision,
    p_events: events,
  });
  if (committed.error) throw Error("COMMIT_FAILED:" + committed.error.message);
  return {
    status: pathError
      ? "RECONCILIATION_REQUIRED"
      : marketError
      ? "DATA_UNAVAILABLE"
      : "OK",
    market_error: marketError,
    valuation_fresh: !!market,
    session_id: sid,
    worker_version: POLICY_VERSION,
    state_version: loaded.state_version + 1,
    symbol: SYMBOL,
    thesis: c.thesis,
    position: rt.pos,
    equity,
    realized: rt.realized,
    events: events.map((e) => e.event_kind),
    shadow_only: true,
    live_execution: false,
  };
}
