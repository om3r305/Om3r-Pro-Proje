from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence


Capability = Literal[
    "market", "announcements", "news", "social", "onchain", "entity_labels",
    "smart_money", "derivatives", "tokenomics", "macro"
]
HistoricalSafety = Literal["point_in_time", "provider_dependent", "prospective_only"]


@dataclass(frozen=True, slots=True)
class ProviderCapability:
    provider: str
    capabilities: tuple[Capability, ...]
    requires_secret: bool
    historical_safety: HistoricalSafety
    authoritative_for: tuple[str, ...] = ()
    default_enabled: bool = False
    notes: str = ""
    schema_version: str = "brian.intel-provider-capability.v1"

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.capabilities:
            raise ValueError("provider and capabilities are required")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("provider capabilities must be unique")
        if self.requires_secret and self.default_enabled:
            raise ValueError("credentialed providers cannot be default-enabled")

    def manifest(self) -> dict:
        return asdict(self)


PROVIDER_REGISTRY: tuple[ProviderCapability, ...] = (
    ProviderCapability(
        "binance_public", ("market", "announcements", "derivatives"), False, "point_in_time",
        authoritative_for=("binance_listing", "binance_delisting", "binance_market_data"),
        default_enabled=True,
        notes="Public exchange endpoints only; no authenticated execution surface.",
    ),
    ProviderCapability(
        "project_official_feeds", ("announcements", "news", "tokenomics"), False, "provider_dependent",
        authoritative_for=("project_announcement",),
        notes="Official project/blog/RSS sources; provenance and publication time required.",
    ),
    ProviderCapability(
        "arkham", ("onchain", "entity_labels", "smart_money"), True, "provider_dependent",
        authoritative_for=("provider_verified_entity_label",),
        notes="Treat predicted/custom labels separately from provider-verified labels.",
    ),
    ProviderCapability(
        "nansen", ("onchain", "entity_labels", "smart_money"), True, "provider_dependent",
        notes="Smart-money/holder/flow intelligence; provider timestamp semantics must be preserved.",
    ),
    ProviderCapability(
        "lunarcrush", ("social",), True, "provider_dependent",
        notes="Cross-platform social activity/sentiment; never treat social popularity as truth by itself.",
    ),
    ProviderCapability(
        "x_api", ("social", "news"), True, "prospective_only",
        notes="Use official API/terms-compliant access; capture raw publication and observation timestamps.",
    ),
    ProviderCapability(
        "reddit_api", ("social", "news"), True, "prospective_only",
        notes="Community signal only; bot/duplicate/source-authenticity checks required.",
    ),
    ProviderCapability(
        "glassnode", ("onchain", "derivatives"), True, "provider_dependent",
        notes="Aggregate on-chain/exchange-flow context; no hindsight use unless historical PIT semantics are verified.",
    ),
    ProviderCapability(
        "macro_official", ("macro", "news"), False, "point_in_time",
        authoritative_for=("scheduled_macro_release", "official_regulatory_release"),
        notes="Official central-bank/regulator/statistics releases where timestamped public archives exist.",
    ),
)


def provider_map(registry: Sequence[ProviderCapability] = PROVIDER_REGISTRY) -> Mapping[str, ProviderCapability]:
    rows = {row.provider: row for row in registry}
    if len(rows) != len(tuple(registry)):
        raise ValueError("duplicate provider id")
    return rows


def enabled_without_secrets(registry: Sequence[ProviderCapability] = PROVIDER_REGISTRY) -> tuple[str, ...]:
    return tuple(sorted(row.provider for row in registry if row.default_enabled and not row.requires_secret))


def historical_replay_allowlist(registry: Sequence[ProviderCapability] = PROVIDER_REGISTRY) -> tuple[str, ...]:
    return tuple(sorted(row.provider for row in registry if row.historical_safety == "point_in_time"))
