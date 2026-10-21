"""Out-of-sample test of claim C-01 of arXiv:2602.19520 on public Polymarket data.

Claim (paper repo DATA_SNAPSHOT_NOTE.md): Polymarket political markets are underconfident —
logistic recalibration slope > 1 (mean over reliable time bins = 1.45 on the locked snapshot,
data cut-off 2025-12-31).

This experiment uses ONLY markets whose reference time falls after the paper's cut-off, so the
data was never seen by the paper. It is an ADAPTED replication: one price per market at a fixed
horizon instead of the paper's trade-level, contract-weighted cells (see ledger assumptions).

Public, key-less, read-only HTTP only (gamma-api / clob .polymarket.com). Stdlib only.
Usage: python run.py --out <dir> [--start 2026-01-01] [--end 2026-10-01] [--horizon-hours 30]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify import classify_polymarket_domain  # noqa: E402
from stats import cluster_bootstrap_slope, decide, fit_slope  # noqa: E402

EXPERIMENT_ID = "pm-political-slope-oos-v1"
ALLOWED_HOSTS = {"gamma-api.polymarket.com", "clob.polymarket.com"}
PAPER_CUTOFF = "2025-12-31T23:59:59Z"
STALE_WINDOW_HOURS = 6  # the horizon price must come from (h + 6h, h] before the reference time


def parse_ts(value: str | None) -> float | None:
    if not value:
        return None
    v = value.strip().replace(" ", "T")
    if v.endswith("+00"):
        v += ":00"
    v = v.replace("Z", "+00:00")
    try:
        d = datetime.fromisoformat(v)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


class FetchFailed(RuntimeError):
    """A single request failed after retries (network/5xx); the caller decides what that means."""


MAX_FETCH_FAILURE_RATE = 0.02  # above this share of failed price fetches the sample is REVIEW_REQUIRED


def fetch_failures_too_high(failed: int, used: int) -> bool:
    attempted = failed + used
    return attempted > 0 and failed > MAX_FETCH_FAILURE_RATE * attempted


class Http:
    def __init__(self, min_interval_s: float = 0.12, max_requests: int = 150000):
        self.min_interval_s = min_interval_s
        self.max_requests = max_requests
        self.count = 0
        self._last = 0.0

    def get(self, url: str):
        host = urllib.parse.urlparse(url).hostname
        if host not in ALLOWED_HOSTS:
            raise ValueError(f"host not allowed: {host}")
        if self.count >= self.max_requests:
            # Treated like a failed fetch: the affected listing/price is incomplete -> REVIEW_REQUIRED.
            raise FetchFailed("request budget exhausted")
        for attempt in range(4):
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.count += 1
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "brian-research-lab/1 (+read-only)"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 451):
                    raise RuntimeError(f"access refused ({e.code}); not retried") from e
                if e.code == 429 or e.code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                time.sleep(2 ** attempt)
        raise FetchFailed(f"giving up on {url}")


def binary_resolution(m: dict) -> int | None:
    """1 if YES won, 0 if NO won, None if not a cleanly resolved Yes/No market."""
    try:
        outcomes = json.loads(m.get("outcomes") or "[]")
        prices = json.loads(m.get("outcomePrices") or "[]")
    except (TypeError, ValueError):
        return None
    if outcomes != ["Yes", "No"] or m.get("umaResolutionStatus") != "resolved":
        return None
    if prices == ["1", "0"]:
        return 1
    if prices == ["0", "1"]:
        return 0
    return None


def reference_time(m: dict) -> float | None:
    """Earlier of scheduled end and close: the outcome may become known between the two."""
    times = [t for t in (parse_ts(m.get("endDate")), parse_ts(m.get("closedTime"))) if t is not None]
    return min(times) if times else None


def price_at_horizon(history: list[dict], target_ts: float) -> float | None:
    """Last observed price at or before target_ts, no older than STALE_WINDOW_HOURS. No lookahead."""
    best = None
    for pt in history:
        try:
            t, p = float(pt["t"]), float(pt["p"])
        except (KeyError, TypeError, ValueError):
            continue
        if t <= target_ts and t >= target_ts - STALE_WINDOW_HOURS * 3600 and 0.0 <= p <= 1.0:
            if best is None or t > best[0]:
                best = (t, p)
    return None if best is None else best[1]


def fetch_closed_markets(http: Http, start: str, end: str, max_pages: int, state: dict):
    """Yields markets; state["complete"] is True only if the API reported no further page."""
    cursor = None
    state["complete"] = False
    state["pages"] = 0
    for _ in range(max_pages):
        q = {"closed": "true", "limit": "500", "end_date_min": start, "end_date_max": end}
        if cursor:
            q["after_cursor"] = cursor
        body = http.get("https://gamma-api.polymarket.com/markets/keyset?" + urllib.parse.urlencode(q))
        state["pages"] += 1
        for m in body.get("markets", []):
            yield m
        cursor = body.get("next_cursor")
        if not cursor:
            state["complete"] = True
            return


def month_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Split [start, end) into calendar-month windows so every listing can reach its last page."""
    s, e = datetime.fromtimestamp(parse_ts(start), timezone.utc), datetime.fromtimestamp(parse_ts(end), timezone.utc)
    out, cur = [], s
    while cur < e:
        nxt = datetime(cur.year + (cur.month == 12), cur.month % 12 + 1, 1, tzinfo=timezone.utc)
        hi = min(nxt, e)
        out.append((cur.strftime("%Y-%m-%dT%H:%M:%SZ"), hi.strftime("%Y-%m-%dT%H:%M:%SZ")))
        cur = hi
    return out


def week_chunks(start: str, end: str, days: int = 7) -> list[tuple[str, str]]:
    """Split [start, end) into windows of `days` days so every listing can reach its last page
    (month windows hit the page cap for Jul-Sep 2026 in run-3)."""
    s, e = parse_ts(start), parse_ts(end)
    out, cur = [], s
    while cur < e:
        hi = min(cur + days * 86400, e)
        out.append((datetime.fromtimestamp(cur, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    datetime.fromtimestamp(hi, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
        cur = hi
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2026-01-01T00:00:00Z")
    ap.add_argument("--end", default="2026-10-01T00:00:00Z")
    ap.add_argument("--horizon-hours", type=float, default=30.0)
    ap.add_argument("--max-pages", type=int, default=4000)
    ap.add_argument("--max-markets", type=int, default=30000)
    ap.add_argument("--resamples", type=int, default=2000)
    args = ap.parse_args()
    if parse_ts(args.start) <= parse_ts(PAPER_CUTOFF):
        raise SystemExit("start must be after the paper's data cut-off (out-of-sample only)")
    os.makedirs(args.out, exist_ok=True)

    http = Http()
    started = datetime.now(timezone.utc).isoformat()
    excluded: dict[str, int] = {}
    def skip(reason: str):
        excluded[reason] = excluded.get(reason, 0) + 1

    rows = []
    seen = 0
    chunk_states: list[dict] = []
    def all_markets():
        for lo, hi in week_chunks(args.start, args.end):
            st: dict = {"window": [lo, hi]}
            chunk_states.append(st)
            try:
                yield from fetch_closed_markets(http, lo, hi, args.max_pages, st)
            except FetchFailed:
                st["complete"] = False  # this month is incomplete -> REVIEW_REQUIRED
                st["fetch_failed"] = True
    for m in all_markets():
        seen += 1
        if classify_polymarket_domain(m.get("question") or "") != "Politics":
            skip("NOT_POLITICS"); continue
        y = binary_resolution(m)
        if y is None:
            skip("NOT_CLEAN_BINARY_RESOLUTION"); continue
        ref = reference_time(m)
        if ref is None or ref <= parse_ts(PAPER_CUTOFF) or ref > parse_ts(args.end):
            skip("REFERENCE_TIME_OUT_OF_WINDOW"); continue
        events = m.get("events") or []
        event_id = str(events[0].get("id")) if events and events[0].get("id") is not None else None
        if not event_id:
            skip("NO_EVENT_ID"); continue
        try:
            yes_token = json.loads(m.get("clobTokenIds") or "[]")[0]
        except (ValueError, IndexError, TypeError):
            skip("NO_TOKEN"); continue
        if len(rows) >= args.max_markets:
            skip("MAX_MARKETS_REACHED"); continue
        target = ref - args.horizon_hours * 3600
        try:
            hist = http.get("https://clob.polymarket.com/prices-history?" + urllib.parse.urlencode({
                "market": yes_token, "startTs": int(target - STALE_WINDOW_HOURS * 3600), "endTs": int(target), "fidelity": 60}))
        except FetchFailed:
            skip("HISTORY_FETCH_FAILED"); continue
        p = price_at_horizon(hist.get("history") or [], target)
        if p is None:
            skip("NO_PRICE_AT_HORIZON"); continue
        rows.append({"market_id": str(m.get("id")), "event_id": event_id, "question": m.get("question"),
                     "reference_ts": int(ref), "price_yes": p, "outcome_yes": y})

    rows.sort(key=lambda r: r["market_id"])
    canonical = json.dumps([[r["market_id"], r["event_id"], r["price_yes"], r["outcome_yes"]] for r in rows], separators=(",", ":"))
    fit = fit_slope([r["price_yes"] for r in rows], [r["outcome_yes"] for r in rows])
    boot = cluster_bootstrap_slope(((r["event_id"], r["price_yes"], r["outcome_yes"]) for r in rows), resamples=args.resamples)
    n_events = len({r["event_id"] for r in rows})
    status, reason = decide(len(rows), n_events, boot["lower"], boot["upper"])
    incomplete = [c["window"][0][:10] for c in chunk_states if not c.get("complete")]
    if incomplete:
        # A truncated market list is a biased sample: never report it as PASS/FAIL.
        status, reason = "REVIEW_REQUIRED", f"market listing truncated in {incomplete}; sample may be biased"
    elif fetch_failures_too_high(excluded.get("HISTORY_FETCH_FAILED", 0), len(rows)):
        status, reason = "REVIEW_REQUIRED", f"{excluded['HISTORY_FETCH_FAILED']} price fetches failed (> {MAX_FETCH_FAILURE_RATE:.0%}); missing data may not be random"
    elif excluded.get("MAX_MARKETS_REACHED"):
        status, reason = "REVIEW_REQUIRED", "market cap reached; later markets were dropped and the sample may be biased"

    result = {
        "experiment_id": EXPERIMENT_ID,
        "claim_id": "arxiv:2602.19520#C-01",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": os.environ.get("GITHUB_SHA") or os.environ.get("RESEARCH_CODE_COMMIT"),
        "window": {"start": args.start, "end": args.end, "paper_cutoff": PAPER_CUTOFF},
        "horizon_hours": args.horizon_hours,
        "markets_seen": seen,
        "listing_complete": not incomplete,
        "listing_chunks": [{"window": c["window"], "pages": c.get("pages"), "complete": bool(c.get("complete"))} for c in chunk_states],
        "markets_used": len(rows),
        "events_used": n_events,
        "yes_rate": (sum(r["outcome_yes"] for r in rows) / len(rows)) if rows else None,
        "excluded": dict(sorted(excluded.items())),
        "http_requests": http.count,
        "slope": fit[0] if fit else None,
        "intercept": fit[1] if fit else None,
        "bootstrap": boot,
        "status": status,
        "status_reason": reason,
        "paper_reference": {"value": 1.45, "what": "Politics mean slope over reliable bins, trade-level, locked snapshot", "source": "namanhzz/prediction-market-calibration@143ca87 DATA_SNAPSHOT_NOTE.md", "comparable": False},
        "dataset_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "evidence_class": "OUT_OF_SAMPLE_ADAPTED_REPLICATION",
        "shadow_only": True,
        "trade_signal": False,
    }
    with open(os.path.join(args.out, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ("markets_seen", "markets_used", "events_used", "slope", "bootstrap", "status", "status_reason", "http_requests")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
