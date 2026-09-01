from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from typing import Any
import json
import time
import requests


@dataclass(frozen=True, slots=True)
class FundingObservation:
    exchange: str
    market_type: str
    symbol: str
    effective_timestamp: float
    rate: float
    source: str
    observed_timestamp: float
    ingestion_provenance: str = "offline_public_api"

    def __post_init__(self) -> None:
        if self.market_type != "perpetual" or self.observed_timestamp < self.effective_timestamp:
            raise ValueError("funding must remain explicit perpetual point-in-time data")

    @property
    def observation_id(self) -> str:
        payload = asdict(self).copy(); payload.pop("observed_timestamp")
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class BinancePublicFundingAdapter:
    endpoint = "https://fapi.binance.com/fapi/v1/fundingRate"

    def __init__(self, session: Any | None = None, timeout: float = 20.0) -> None:
        self.session, self.timeout = session or requests.Session(), timeout

    def fetch(self, symbol: str, start: float, end: float, limit: int = 1000) -> tuple[FundingObservation, ...]:
        if start >= end or not symbol.endswith("USDT"):
            raise ValueError("invalid USD-M perpetual funding request")
        cursor, end_ms = int(start * 1000), int(end * 1000)
        observations: list[FundingObservation] = []
        while cursor < end_ms:
            response = self.session.get(self.endpoint, params={"symbol": symbol, "startTime": cursor,
                                        "endTime": end_ms - 1, "limit": limit}, timeout=self.timeout)
            response.raise_for_status(); page = response.json(); retrieved = time.time()
            if not page: break
            for item in page:
                effective = int(item["fundingTime"]) / 1000.0
                observations.append(FundingObservation("binance", "perpetual", symbol, effective,
                                    float(item["fundingRate"]), self.endpoint, max(retrieved, effective)))
            next_cursor = int(page[-1]["fundingTime"]) + 1
            if next_cursor <= cursor: raise ValueError("funding pagination did not advance")
            cursor = next_cursor
            if len(page) < limit: break
        return tuple(sorted(observations, key=lambda row: row.effective_timestamp))