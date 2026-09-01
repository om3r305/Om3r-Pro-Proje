from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Iterable
import json

from .dataset import MarketDataset
from .features import FeatureSnapshot

CATALOG_SCHEMA_VERSION = "brian.dataset-catalog.v1"


def _hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetCatalogEntry:
    dataset_id: str
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    start_timestamp: float
    end_timestamp: float
    feature_schema_version: str
    observations: int
    source_provenance: tuple[tuple[str, str], ...]
    missing_feature_counts: tuple[tuple[str, int], ...]
    schema_version: str = CATALOG_SCHEMA_VERSION

    @property
    def catalog_id(self) -> str:
        return _hash(asdict(self))


def catalog_dataset(dataset: MarketDataset, snapshots: Iterable[FeatureSnapshot]) -> DatasetCatalogEntry:
    snaps = tuple(snapshots)
    if not dataset.events:
        raise ValueError("cannot catalog an empty market dataset")
    if any(s.dataset_id not in (None, dataset.dataset_id) for s in snaps):
        raise ValueError("snapshot dataset provenance mismatch")
    versions = {s.feature_schema_version for s in snaps}
    if len(versions) > 1:
        raise ValueError("mixed feature schema versions")
    names = sorted({name for s in snaps for name in s.unavailable_features() + list(s.available_features())})
    missing = tuple((name, sum(getattr(s, name, None) is None for s in snaps)) for name in names)
    sources = sorted({pair for event in dataset.events for pair in event.sources})
    return DatasetCatalogEntry(
        dataset.dataset_id,
        tuple(sorted({e.symbol for e in dataset.events})),
        tuple(sorted({e.timeframe for e in dataset.events})),
        min(e.event_timestamp for e in dataset.events),
        max(e.event_timestamp for e in dataset.events),
        next(iter(versions), "unknown"), len(snaps), tuple(sources), missing,
    )