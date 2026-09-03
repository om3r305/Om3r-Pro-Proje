from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

AssetClass = Literal[
    "crypto_spot", "fan_token", "equity", "etf", "commodity", "fx", "rates", "macro"
]

CATALOG_SCHEMA_VERSION = "brian.multi-asset-catalog.v1"


@dataclass(frozen=True, slots=True)
class AssetSpec:
    canonical_id: str
    symbol: str
    asset_class: AssetClass
    venue: str
    quote_currency: str
    timezone: str
    trading_calendar: str
    native_frequency: str
    source_name: str
    provenance_uri: str
    is_proxy: bool = False
    proxy_for: str | None = None
    schema_version: str = CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        required = (
            self.canonical_id, self.symbol, self.asset_class, self.venue, self.quote_currency,
            self.timezone, self.trading_calendar, self.native_frequency, self.source_name,
            self.provenance_uri,
        )
        if any(not str(value).strip() for value in required):
            raise ValueError("asset catalog fields must be explicit")
        if self.is_proxy and not (self.proxy_for or "").strip():
            raise ValueError("proxy assets must identify what they proxy")
        if not self.is_proxy and self.proxy_for is not None:
            raise ValueError("non-proxy asset cannot declare proxy_for")

    def manifest(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MultiAssetCatalog:
    assets: tuple[AssetSpec, ...]
    schema_version: str = CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.assets:
            raise ValueError("catalog requires assets")
        ids = [asset.canonical_id for asset in self.assets]
        if len(ids) != len(set(ids)):
            raise ValueError("canonical asset ids must be unique")

    @classmethod
    def from_sequence(cls, assets: Sequence[AssetSpec]) -> "MultiAssetCatalog":
        return cls(tuple(sorted(assets, key=lambda row: row.canonical_id)))

    def get(self, canonical_id: str) -> AssetSpec:
        for asset in self.assets:
            if asset.canonical_id == canonical_id:
                return asset
        raise KeyError(canonical_id)

    def by_class(self, asset_class: AssetClass) -> tuple[AssetSpec, ...]:
        return tuple(asset for asset in self.assets if asset.asset_class == asset_class)

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "assets": [asset.manifest() for asset in self.assets],
            "provider_neutral": True,
            "proxy_disclosure_required": True,
        }
