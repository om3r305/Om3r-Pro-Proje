from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence
import math

from .intelligence_fabric import TRUST_WEIGHT, TrustClass, WhaleFlow

EntityRole = Literal[
    "exchange", "fund", "whale", "project", "bridge", "dex", "contract", "deployer", "unknown"
]


def _finite(value: float, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


@dataclass(frozen=True, slots=True)
class EntityLabel:
    address: str
    entity_id: str
    role: EntityRole
    provider: str
    trust_class: TrustClass
    confidence: float
    observed_at: float
    label: str | None = None
    historical_timestamp_verified: bool = False
    schema_version: str = "brian.entity-label.v1"

    def __post_init__(self) -> None:
        if not self.address.strip() or not self.entity_id.strip() or not self.provider.strip():
            raise ValueError("address/entity/provider are required")
        confidence = _finite(self.confidence, "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0,1]")
        _finite(self.observed_at, "observed_at")

    @property
    def authority_score(self) -> float:
        return TRUST_WEIGHT[self.trust_class] * self.confidence


@dataclass(frozen=True, slots=True)
class ResolvedEntity:
    address: str
    entity_id: str | None
    role: EntityRole
    authority_score: float
    provider: str | None
    trust_class: TrustClass | None
    conflicting_labels: bool
    candidates: tuple[EntityLabel, ...]
    schema_version: str = "brian.resolved-entity.v1"


@dataclass(frozen=True, slots=True)
class TransferEdge:
    tx_hash: str
    asset: str
    from_address: str
    to_address: str
    usd_value: float
    observed_at: float
    chain: str
    schema_version: str = "brian.transfer-edge.v1"

    def __post_init__(self) -> None:
        if not all(x.strip() for x in (self.tx_hash, self.asset, self.from_address, self.to_address, self.chain)):
            raise ValueError("transaction fields are required")
        if _finite(self.usd_value, "usd_value") < 0:
            raise ValueError("usd_value cannot be negative")
        _finite(self.observed_at, "observed_at")


@dataclass(frozen=True, slots=True)
class TransferInterpretation:
    flow: WhaleFlow
    confidence: float
    from_entity: ResolvedEntity
    to_entity: ResolvedEntity
    economically_directional: bool
    reasons: tuple[str, ...]
    schema_version: str = "brian.transfer-interpretation.v1"


class EntityGraph:
    """Point-in-time entity-label graph.

    Label resolution is always as-of the query timestamp. A label learned later
    cannot rewrite what Brian believed at an earlier event time.
    """

    def __init__(self, labels: Sequence[EntityLabel] = ()) -> None:
        self._labels: list[EntityLabel] = []
        for row in labels:
            self.add_label(row)

    def add_label(self, label: EntityLabel) -> None:
        self._labels.append(label)

    def labels_for(self, address: str, *, as_of: float) -> tuple[EntityLabel, ...]:
        cutoff = _finite(as_of, "as_of")
        return tuple(sorted(
            (row for row in self._labels if row.address == address and row.observed_at <= cutoff),
            key=lambda row: (-row.authority_score, -row.observed_at, row.provider, row.entity_id),
        ))

    def resolve(self, address: str, *, as_of: float) -> ResolvedEntity:
        candidates = self.labels_for(address, as_of=as_of)
        if not candidates:
            return ResolvedEntity(address, None, "unknown", 0.0, None, None, False, ())
        winner = candidates[0]
        strong = [row for row in candidates if row.authority_score >= max(0.55, winner.authority_score - 0.10)]
        conflicting = len({row.entity_id for row in strong}) > 1
        if conflicting:
            # Conflicting high-authority labels are not silently resolved into false certainty.
            return ResolvedEntity(address, None, "unknown", winner.authority_score,
                                  winner.provider, winner.trust_class, True, candidates)
        return ResolvedEntity(address, winner.entity_id, winner.role, winner.authority_score,
                              winner.provider, winner.trust_class, False, candidates)

    def interpret_transfer(self, edge: TransferEdge) -> TransferInterpretation:
        source = self.resolve(edge.from_address, as_of=edge.observed_at)
        target = self.resolve(edge.to_address, as_of=edge.observed_at)
        reasons: list[str] = []
        confidence = min(source.authority_score if source.entity_id else 0.5,
                         target.authority_score if target.entity_id else 0.5)

        if source.entity_id is not None and source.entity_id == target.entity_id:
            return TransferInterpretation("internal_transfer", confidence, source, target, False,
                                          ("same resolved entity controls both sides",))
        if source.conflicting_labels or target.conflicting_labels:
            return TransferInterpretation("unknown", min(confidence, 0.35), source, target, False,
                                          ("high-authority entity labels conflict",))
        if target.role == "exchange" and source.role != "exchange":
            reasons.append("funds moved from non-exchange entity to exchange")
            return TransferInterpretation("exchange_deposit", confidence, source, target, True, tuple(reasons))
        if source.role == "exchange" and target.role != "exchange":
            reasons.append("funds moved from exchange to non-exchange entity")
            return TransferInterpretation("exchange_withdrawal", confidence, source, target, True, tuple(reasons))
        # A transfer into a DEX contract is not enough to infer buy vs sell without swap leg semantics.
        if source.role == "dex" or target.role == "dex":
            reasons.append("DEX interaction lacks swap-leg semantics; buy/sell not inferred")
        else:
            reasons.append("counterparty roles do not determine economic direction")
        return TransferInterpretation("unknown", confidence, source, target, False, tuple(reasons))

    def assert_historical_labels_safe(self, *, as_of: float) -> None:
        cutoff = _finite(as_of, "as_of")
        unsafe = [
            row for row in self._labels
            if row.observed_at <= cutoff and not row.historical_timestamp_verified
        ]
        if unsafe:
            raise ValueError(
                "historical entity replay contains labels whose point-in-time availability is unverified"
            )
