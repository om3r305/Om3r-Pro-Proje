import { assertEquals } from "jsr:@std/assert@1";
import { feedGuardVerdict, matchContract, parseScanner, scannerTickers, type OwnObservation } from "./multiasset_feed_guard.ts";

const NOW = "2026-10-23T08:03:00.000Z";
const at = (min: number) => new Date(Date.parse(NOW) - min * 60_000).toISOString();
// Real 2026-10-23 numbers: Yahoo NG=F 2.718 -> 3.117 at 07:53; TradingView NG1! 2.64 (Nov), NG2! 3.05 (Dec).
const ngQuote = { front: 2.66, next: 3.11, frontDailyChangePct: -2.1 };

Deno.test("contract match: primary price is classified against front/next of the second source", () => {
  assertEquals(matchContract(2.718, ngQuote), "FRONT");
  assertEquals(matchContract(3.117, ngQuote), "NEXT");
  assertEquals(matchContract(5, ngQuote), "UNMATCHED");
  assertEquals(matchContract(3, null), "UNKNOWN");
  assertEquals(matchContract(116.5, { front: 116.72, next: 111.7, frontDailyChangePct: 3 }), "FRONT");
  assertEquals(matchContract(111.7, { front: 116.72, next: 111.7, frontDailyChangePct: 3 }), "NEXT", "real 2026-10-24 Brent: Yahoo BZ=F = Jan contract");
  assertEquals(matchContract(4010, { front: 4007.9, next: 4040, frontDailyChangePct: -1 }), "UNKNOWN", "gold contracts too close to tell apart");
});

Deno.test("2026-10-23 NATGAS roll is vetoed as a contract roll for the next hour", () => {
  const history: OwnObservation[] = [
    { observedAt: at(30), price: 2.717, feedContract: "FRONT" },
    { observedAt: at(20), price: 2.718, feedContract: "FRONT" },
    { observedAt: at(10), price: 3.117, feedContract: "NEXT" },
  ];
  const v = feedGuardVerdict({ assetId: "commodity:NATGAS", nowIso: NOW, price: 3.12, history, quote: ngQuote });
  assertEquals(v.veto, "FEED_CONTRACT_ROLL");
  const later = feedGuardVerdict({ assetId: "commodity:NATGAS", nowIso: new Date(Date.parse(NOW) + 80 * 60_000).toISOString(), price: 3.12,
    history: [...history, { observedAt: at(-60), price: 3.12, feedContract: "NEXT" }], quote: ngQuote });
  assertEquals(later.veto, null, "after the window the (consistent) next-contract series is usable again");
});

Deno.test("a jump without second-source confirmation fails closed; a confirmed spike passes", () => {
  const history: OwnObservation[] = [{ observedAt: at(10), price: 114.26, feedContract: null }];
  const noSource = feedGuardVerdict({ assetId: "commodity:BRENT", nowIso: NOW, price: 109.42, history, quote: null });
  assertEquals(noSource.veto, "PRICE_DISCONTINUITY");
  const flatFront = feedGuardVerdict({ assetId: "commodity:BRENT", nowIso: NOW, price: 109.42, history,
    quote: { front: null, next: null, frontDailyChangePct: 1.4 } });
  assertEquals(flatFront.veto, "PRICE_DISCONTINUITY", "front month up 1.4% does not confirm a -4.3% step");
  const realSpike = feedGuardVerdict({ assetId: "commodity:BRENT", nowIso: NOW, price: 119.3, history,
    quote: { front: null, next: null, frontDailyChangePct: 4.5 } });
  assertEquals(realSpike.veto, null, "a real same-direction move is not blocked");
});

Deno.test("ordinary moves, gaps and non-commodities pass", () => {
  const history: OwnObservation[] = [{ observedAt: at(10), price: 100, feedContract: "FRONT" }];
  assertEquals(feedGuardVerdict({ assetId: "commodity:WTI", nowIso: NOW, price: 101.5, history, quote: { front: 101.4, next: 100, frontDailyChangePct: 2 } }).veto, null);
  assertEquals(feedGuardVerdict({ assetId: "commodity:WTI", nowIso: NOW, price: 110, history: [{ observedAt: at(40), price: 100, feedContract: null }], quote: null }).veto, null,
    "steps across a > 20 min gap (session break) are not single-step jumps");
  assertEquals(feedGuardVerdict({ assetId: "equity:META", nowIso: NOW, price: 120, history, quote: null }).guarded, false);
});

Deno.test("scanner parsing picks front/next by expiration and skips expired or unlisted months", () => {
  const now = "2026-10-24T08:00:00.000Z";
  // Real 2026-10-24 scanner rows (unordered; one expired contract added).
  const q = parseScanner({ data: [
    { s: "NYMEX:NGZ2026", d: [3.05, -3.36, 20261125] }, { s: "NYMEX:NGX2026", d: [2.64, -5.58, 20261028] },
    { s: "NYMEX:NGV2026", d: [2.9, 0, 20261020] },
    { s: "ICEEUR:BRNF2027", d: [111.7, 2.8, 20261130] }, { s: "ICEEUR:BRNZ2026", d: [116.72, 3.14, 20261030] },
    { s: "BAD", d: "x" },
  ] }, now);
  assertEquals(q.get("commodity:NATGAS"), { front: 2.64, next: 3.05, frontDailyChangePct: -5.58 });
  assertEquals(q.get("commodity:BRENT"), { front: 116.72, next: 111.7, frontDailyChangePct: 3.14 });
  assertEquals(q.has("commodity:GOLD"), false);
  const tickers = scannerTickers(now);
  assertEquals(tickers.length, 30);
  assertEquals(tickers.slice(0, 3), ["NYMEX:NGV2026", "NYMEX:NGX2026", "NYMEX:NGZ2026"]);
  assertEquals(scannerTickers("2026-11-15T00:00:00.000Z").includes("NYMEX:NGF2027"), true, "year rolls over");
});
