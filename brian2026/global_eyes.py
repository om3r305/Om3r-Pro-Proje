from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

AccessMode = Literal["PUBLIC_NO_KEY", "API_KEY_REQUIRED", "LICENSED_REQUIRED"]
LatencyClass = Literal["FAST_5M", "INTRADAY", "DAILY", "EVENT_DRIVEN"]


@dataclass(frozen=True, slots=True)
class EyeProviderSpec:
    provider_id: str
    domain: str
    access_mode: AccessMode
    latency_class: LatencyClass
    provenance_uri: str
    enabled_live: bool
    notes: str
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.provider_id or not self.domain or not self.provenance_uri:
            raise ValueError("provider identity/provenance must be explicit")
        if self.live_execution:
            raise ValueError("Phase 3.9 providers cannot execute live trades")
        if self.enabled_live and self.access_mode != "PUBLIC_NO_KEY":
            raise ValueError("credentialed/licensed providers cannot be silently enabled")

    def manifest(self) -> dict[str, object]:
        return asdict(self)


GLOBAL_EYE_PROVIDERS: tuple[EyeProviderSpec, ...] = (
    EyeProviderSpec(
        "binance-usdm-public", "derivatives", "PUBLIC_NO_KEY", "FAST_5M",
        "https://fapi.binance.com", True,
        "Public USD-M funding, premium/basis, open-interest and positioning context only.",
    ),
    EyeProviderSpec(
        "gdelt-doc2", "news", "PUBLIC_NO_KEY", "EVENT_DRIVEN",
        "https://api.gdeltproject.org/api/v2/doc/doc", True,
        "Global news discovery. Headline pressure is low-authority evidence, not ground truth.",
    ),
    EyeProviderSpec(
        "ecb-reference-fx", "fx", "PUBLIC_NO_KEY", "DAILY",
        "https://www.ecb.europa.eu/stats/eurofxref/", True,
        "Official ECB EUR reference rates; daily macro context, never fabricated intraday FX.",
    ),
    EyeProviderSpec(
        "x-api", "social_psychology", "API_KEY_REQUIRED", "EVENT_DRIVEN",
        "https://developer.x.com/", False,
        "Credential socket only; unavailable until explicitly provisioned out-of-band.",
    ),
    EyeProviderSpec(
        "reddit-api", "social_psychology", "API_KEY_REQUIRED", "EVENT_DRIVEN",
        "https://www.reddit.com/dev/api/", False,
        "OAuth/provider terms required; no anonymous scraping fallback.",
    ),
    EyeProviderSpec(
        "onchain-provider", "onchain", "API_KEY_REQUIRED", "EVENT_DRIVEN",
        "provider://onchain", False,
        "Provider-neutral wallet/flow socket; economic direction must be evidenced.",
    ),
    EyeProviderSpec(
        "licensed-equities", "equities", "LICENSED_REQUIRED", "INTRADAY",
        "provider://licensed-equities", False,
        "No unofficial scraper is substituted for a licensed equity feed.",
    ),
    EyeProviderSpec(
        "licensed-metals", "metals", "LICENSED_REQUIRED", "INTRADAY",
        "provider://licensed-metals", False,
        "Gold/silver futures or spot feed must be licensed or explicitly proxied.",
    ),
)


def live_provider_ids() -> tuple[str, ...]:
    return tuple(row.provider_id for row in GLOBAL_EYE_PROVIDERS if row.enabled_live)


def unavailable_domains() -> tuple[str, ...]:
    return tuple(sorted({row.domain for row in GLOBAL_EYE_PROVIDERS if not row.enabled_live}))
