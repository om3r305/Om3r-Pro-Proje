from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence
import math

from .evidence_ledger import content_hash
from .phase60_shadow_state_ledger import (
    ShadowAccountState,
    ShadowLedgerTransition,
    ShadowStateConflictError,
    ShadowStateLedger,
)
from .phase61_stateful_paper_venue import (
    PaperCycleReceipt,
    PaperFill,
    PaperOrderOutcome,
    PaperPosition,
    PaperVenue,
    PaperVenueConfig,
    PaperVenueConflictError,
)

PHASE63_SCHEMA_VERSION = "brian.phase63-crash-recovery.v1"


class RuntimeRecoveryError(ValueError):
    pass


def _close(a: float, b: float, *, tol: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=tol)


def _paper_receipt_id(receipt: PaperCycleReceipt) -> str:
    return content_hash({
        "schema_version": receipt.schema_version,
        "cycle_id": receipt.cycle_id,
        "cycle_hash": receipt.cycle_hash,
        "outcomes": [row.to_dict() for row in receipt.outcomes],
        "fill_ids": list(receipt.fill_ids),
        "cash_before_usd": receipt.cash_before_usd,
        "cash_after_usd": receipt.cash_after_usd,
        "state_version_before": receipt.state_version_before,
        "state_version_after": receipt.state_version_after,
    })


def _paper_fill_id(fill: PaperFill) -> str:
    return content_hash({
        "schema_version": fill.schema_version,
        "paper_order_id": fill.paper_order_id,
        "asset_id": fill.asset_id,
        "side": fill.side,
        "filled_base": fill.quantity_base,
        "average_fill_price": fill.price,
        "venue_timestamp": fill.timestamp,
    })


@dataclass(frozen=True, slots=True)
class PaperVenueCheckpoint:
    config: PaperVenueConfig
    cash_usd: float
    state_version: int
    fill_sequence: tuple[PaperFill, ...]
    cycle_receipts: tuple[PaperCycleReceipt, ...]
    final_positions: tuple[PaperPosition, ...]
    schema_version: str = PHASE63_SCHEMA_VERSION
    paper_only: bool = True
    live_execution: bool = False
    checkpoint_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.cash_usd):
            raise ValueError("checkpoint cash must be finite")
        if self.state_version < 0:
            raise ValueError("state_version must be non-negative")
        if self.state_version != len(self.cycle_receipts):
            raise ValueError("paper state_version must equal applied cycle count")
        if not self.paper_only or self.live_execution:
            raise ValueError("Phase 63 paper checkpoint cannot contain live execution")

        fill_ids = [fill.fill_id for fill in self.fill_sequence]
        if len(fill_ids) != len(set(fill_ids)):
            raise ValueError("paper checkpoint contains duplicate fill ids")
        cycle_ids = [receipt.cycle_id for receipt in self.cycle_receipts]
        if len(cycle_ids) != len(set(cycle_ids)):
            raise ValueError("paper checkpoint contains duplicate cycle ids")
        assets = [position.asset_id for position in self.final_positions]
        if len(assets) != len(set(assets)):
            raise ValueError("paper checkpoint contains duplicate final positions")

        object.__setattr__(
            self,
            "final_positions",
            tuple(sorted(self.final_positions, key=lambda row: row.asset_id)),
        )
        object.__setattr__(self, "checkpoint_id", content_hash(self.identity_payload()))

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "config": asdict(self.config),
            "cash_usd": float(self.cash_usd),
            "state_version": self.state_version,
            "fill_sequence": [fill.to_dict() for fill in self.fill_sequence],
            "cycle_receipts": [receipt.to_dict() for receipt in self.cycle_receipts],
            "final_positions": [position.to_dict() for position in self.final_positions],
            "paper_only": self.paper_only,
            "live_execution": self.live_execution,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.identity_payload()
        payload["checkpoint_id"] = self.checkpoint_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PaperVenueCheckpoint":
        raw = dict(payload)
        expected_id = str(raw.pop("checkpoint_id"))
        config = PaperVenueConfig(**dict(raw["config"]))  # type: ignore[arg-type]

        fills = tuple(
            PaperFill(**dict(row))  # type: ignore[arg-type]
            for row in raw["fill_sequence"]  # type: ignore[union-attr]
        )
        receipts: list[PaperCycleReceipt] = []
        for receipt_raw in raw["cycle_receipts"]:  # type: ignore[union-attr]
            receipt_data = dict(receipt_raw)
            receipt_data["outcomes"] = tuple(
                PaperOrderOutcome(**dict(row))
                for row in receipt_data["outcomes"]
            )
            receipt_data["fill_ids"] = tuple(receipt_data["fill_ids"])
            receipts.append(PaperCycleReceipt(**receipt_data))
        positions = tuple(
            PaperPosition(
                asset_id=str(dict(row)["asset_id"]),
                quantity=float(dict(row)["quantity"]),
                avg_entry_price=(
                    None
                    if dict(row)["avg_entry_price"] is None
                    else float(dict(row)["avg_entry_price"])
                ),
                realized_pnl_quote=float(dict(row)["realized_pnl_quote"]),
                source_fill_ids=tuple(dict(row)["source_fill_ids"]),
            )
            for row in raw["final_positions"]  # type: ignore[union-attr]
        )
        checkpoint = cls(
            config=config,
            cash_usd=float(raw["cash_usd"]),
            state_version=int(raw["state_version"]),
            fill_sequence=fills,
            cycle_receipts=tuple(receipts),
            final_positions=positions,
            schema_version=str(raw.get("schema_version", PHASE63_SCHEMA_VERSION)),
            paper_only=bool(raw.get("paper_only", True)),
            live_execution=bool(raw.get("live_execution", False)),
        )
        if checkpoint.checkpoint_id != expected_id:
            raise RuntimeRecoveryError("paper checkpoint content hash mismatch")
        return checkpoint


def create_paper_venue_checkpoint(venue: PaperVenue) -> PaperVenueCheckpoint:
    return PaperVenueCheckpoint(
        config=venue.config,
        cash_usd=float(venue.cash_usd),
        state_version=venue.state_version,
        fill_sequence=venue.fills,
        cycle_receipts=venue.cycle_receipts,
        final_positions=tuple(venue.positions.values()),
    )


def restore_paper_venue(checkpoint: PaperVenueCheckpoint) -> PaperVenue:
    """Restore paper venue by replaying persisted fills and validating every cycle."""
    if content_hash(checkpoint.identity_payload()) != checkpoint.checkpoint_id:
        raise RuntimeRecoveryError("paper checkpoint content hash mismatch")

    venue = PaperVenue(checkpoint.config)
    fills_by_id = {fill.fill_id: fill for fill in checkpoint.fill_sequence}
    consumed_fill_ids: list[str] = []

    for expected_version, receipt in enumerate(checkpoint.cycle_receipts):
        if receipt.live_execution or not receipt.paper_only:
            raise RuntimeRecoveryError("live receipt found in paper checkpoint")
        if len(receipt.cycle_hash) != 64:
            raise RuntimeRecoveryError("paper receipt cycle hash is invalid")
        if _paper_receipt_id(receipt) != receipt.receipt_id:
            raise RuntimeRecoveryError(
                f"paper cycle receipt hash mismatch for {receipt.cycle_id}"
            )
        if receipt.state_version_before != expected_version:
            raise RuntimeRecoveryError("paper cycle state_version_before sequence is broken")
        if receipt.state_version_after != expected_version + 1:
            raise RuntimeRecoveryError("paper cycle state_version_after sequence is broken")
        if not _close(receipt.cash_before_usd, venue.cash_usd):
            raise RuntimeRecoveryError("paper cycle cash_before does not match replay state")

        for fill_id in receipt.fill_ids:
            if fill_id in consumed_fill_ids:
                raise RuntimeRecoveryError("paper fill is referenced by multiple cycles")
            try:
                fill = fills_by_id[fill_id]
            except KeyError as exc:
                raise RuntimeRecoveryError(
                    f"paper cycle references missing fill {fill_id}"
                ) from exc
            if fill.cycle_id != receipt.cycle_id:
                raise RuntimeRecoveryError("paper fill cycle_id does not match cycle receipt")
            if _paper_fill_id(fill) != fill.fill_id:
                raise RuntimeRecoveryError("paper fill content hash mismatch")
            try:
                venue._apply_fill(fill)  # package-internal deterministic replay
            except PaperVenueConflictError as exc:
                raise RuntimeRecoveryError(str(exc)) from exc
            consumed_fill_ids.append(fill_id)

        if not _close(receipt.cash_after_usd, venue.cash_usd):
            raise RuntimeRecoveryError("paper cycle cash_after does not match replayed fills")

        venue._state_version = receipt.state_version_after
        venue._cycle_hashes[receipt.cycle_id] = receipt.cycle_hash
        venue._cycle_receipts[receipt.cycle_id] = receipt

    if tuple(consumed_fill_ids) != tuple(fill.fill_id for fill in checkpoint.fill_sequence):
        raise RuntimeRecoveryError("paper checkpoint contains orphan or reordered fills")
    if venue.state_version != checkpoint.state_version:
        raise RuntimeRecoveryError("restored paper state_version mismatch")
    if not _close(venue.cash_usd, checkpoint.cash_usd):
        raise RuntimeRecoveryError("restored paper cash mismatch")

    restored_positions = tuple(sorted(venue.positions.values(), key=lambda row: row.asset_id))
    if restored_positions != checkpoint.final_positions:
        raise RuntimeRecoveryError("restored paper positions do not match checkpoint")

    return venue


def _parse_shadow_state(payload: Mapping[str, object]) -> ShadowAccountState:
    raw = dict(payload)
    expected_id = str(raw.pop("state_id"))
    state = ShadowAccountState(
        account_id=str(raw["account_id"]),
        observed_at=float(raw["observed_at"]),
        equity_usd=float(raw["equity_usd"]),
        available_cash_usd=float(raw["available_cash_usd"]),
        position_weights=tuple(
            (str(asset), float(weight))
            for asset, weight in raw["position_weights"]  # type: ignore[union-attr]
        ),
        covered_assets=tuple(str(asset) for asset in raw["covered_assets"]),  # type: ignore[union-attr]
        source_kind=str(raw["source_kind"]),  # type: ignore[arg-type]
        source_ref=str(raw["source_ref"]),
        reconciliation_hash=(
            None
            if raw.get("reconciliation_hash") is None
            else str(raw["reconciliation_hash"])
        ),
        schema_version=str(raw.get("schema_version", "brian.phase60-shadow-state-ledger.v1")),
        shadow_only=bool(raw.get("shadow_only", True)),
        live_execution=bool(raw.get("live_execution", False)),
    )
    if state.state_id != expected_id:
        raise RuntimeRecoveryError("shadow state content hash mismatch")
    return state


def _parse_shadow_transition(payload: Mapping[str, object]) -> ShadowLedgerTransition:
    raw = dict(payload)
    expected_id = str(raw.pop("transition_id"))
    transition = ShadowLedgerTransition(
        sequence=int(raw["sequence"]),
        kind=str(raw["kind"]),  # type: ignore[arg-type]
        previous_transition_id=(
            None
            if raw.get("previous_transition_id") is None
            else str(raw["previous_transition_id"])
        ),
        cycle_id=None if raw.get("cycle_id") is None else str(raw["cycle_id"]),
        before_state_id=(
            None
            if raw.get("before_state_id") is None
            else str(raw["before_state_id"])
        ),
        after_state_id=str(raw["after_state_id"]),
        payload_hash=str(raw["payload_hash"]),
        reconciliation_hash=(
            None
            if raw.get("reconciliation_hash") is None
            else str(raw["reconciliation_hash"])
        ),
        schema_version=str(raw.get("schema_version", "brian.phase60-shadow-state-ledger.v1")),
    )
    if transition.transition_id != expected_id:
        raise RuntimeRecoveryError("shadow transition content hash mismatch")
    return transition


def restore_shadow_state_ledger(
    manifest: Mapping[str, object],
) -> ShadowStateLedger:
    raw = dict(manifest)
    if raw.get("append_only") is not True:
        raise RuntimeRecoveryError("shadow ledger manifest is not append-only")
    if raw.get("shadow_only") is not True or raw.get("live_execution") is not False:
        raise RuntimeRecoveryError("shadow ledger manifest crossed the live boundary")

    states_raw = raw.get("states")
    transitions_raw = raw.get("transitions")
    if not isinstance(states_raw, Mapping) or not isinstance(transitions_raw, Sequence):
        raise RuntimeRecoveryError("shadow ledger manifest is missing states/transitions")

    states: dict[str, ShadowAccountState] = {}
    for key, value in states_raw.items():
        if not isinstance(value, Mapping):
            raise RuntimeRecoveryError("invalid shadow state payload")
        state = _parse_shadow_state(value)
        if str(key) != state.state_id:
            raise RuntimeRecoveryError("shadow state map key does not match content id")
        states[state.state_id] = state

    transitions = tuple(
        _parse_shadow_transition(row)
        for row in transitions_raw
        if isinstance(row, Mapping)
    )
    if len(transitions) != len(transitions_raw):
        raise RuntimeRecoveryError("invalid shadow transition payload")
    if not transitions or transitions[0].kind != "GENESIS":
        raise RuntimeRecoveryError("shadow ledger manifest has no genesis transition")

    expected_ledger_hash = content_hash([row.to_dict() for row in transitions])
    if raw.get("ledger_hash") != expected_ledger_hash:
        raise RuntimeRecoveryError("shadow ledger manifest hash mismatch")

    genesis_state_id = transitions[0].after_state_id
    try:
        genesis = states[genesis_state_id]
    except KeyError as exc:
        raise RuntimeRecoveryError("genesis state payload is missing") from exc
    if transitions[0].payload_hash != content_hash(genesis.to_dict()):
        raise RuntimeRecoveryError("genesis transition does not hash the genesis state")

    ledger = ShadowStateLedger(genesis)
    ledger._states = states
    ledger._transitions = list(transitions)
    ledger._head_state_id = str(raw["head_state_id"])
    ledger._pending_cycle_id = (
        None
        if raw.get("pending_cycle_id") is None
        else str(raw["pending_cycle_id"])
    )
    ledger._cycle_payload_hash = {}
    ledger._cycle_proposal_transition = {}
    ledger._cycle_terminal_transition = {}
    ledger._cycle_terminal_signature = {}

    for transition in transitions:
        if transition.kind == "CYCLE_PROPOSED":
            assert transition.cycle_id is not None
            ledger._cycle_payload_hash[transition.cycle_id] = transition.payload_hash
            ledger._cycle_proposal_transition[transition.cycle_id] = transition.transition_id
        elif transition.kind == "SIMULATION_CLOSED":
            assert transition.cycle_id is not None
            ledger._cycle_terminal_transition[transition.cycle_id] = transition.transition_id
            ledger._cycle_terminal_signature[transition.cycle_id] = (
                "SIMULATION_CLOSED",
                None,
            )
        elif transition.kind == "RECONCILED_COMMIT":
            assert transition.cycle_id is not None
            ledger._cycle_terminal_transition[transition.cycle_id] = transition.transition_id
            ledger._cycle_terminal_signature[transition.cycle_id] = (
                "RECONCILED_COMMIT",
                transition.reconciliation_hash,
            )

    try:
        ledger.verify_integrity()
    except ShadowStateConflictError as exc:
        raise RuntimeRecoveryError(str(exc)) from exc
    return ledger


@dataclass(frozen=True, slots=True)
class ShadowRuntimeCheckpoint:
    paper: PaperVenueCheckpoint
    shadow_ledger_manifest: Mapping[str, object]
    pending_cycle_id: str | None
    schema_version: str = PHASE63_SCHEMA_VERSION
    live_execution: bool = False
    checkpoint_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.live_execution:
            raise ValueError("runtime checkpoint cannot contain live execution")
        manifest_pending = self.shadow_ledger_manifest.get("pending_cycle_id")
        normalized = None if manifest_pending is None else str(manifest_pending)
        if normalized != self.pending_cycle_id:
            raise ValueError("runtime checkpoint pending_cycle_id mismatch")

        head_id = str(self.shadow_ledger_manifest.get("head_state_id", ""))
        states = self.shadow_ledger_manifest.get("states")
        if not isinstance(states, Mapping) or head_id not in states:
            raise ValueError("runtime checkpoint shadow head state is missing")
        head_payload = states[head_id]
        if not isinstance(head_payload, Mapping):
            raise ValueError("runtime checkpoint shadow head payload is invalid")
        if str(head_payload.get("account_id")) != self.paper.config.account_id:
            raise ValueError("paper venue and shadow ledger account ids differ")

        object.__setattr__(self, "checkpoint_id", content_hash(self.identity_payload()))

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "paper": self.paper.to_dict(),
            "shadow_ledger_manifest": dict(self.shadow_ledger_manifest),
            "pending_cycle_id": self.pending_cycle_id,
            "live_execution": self.live_execution,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.identity_payload()
        payload["checkpoint_id"] = self.checkpoint_id
        return payload


def create_runtime_checkpoint(
    ledger: ShadowStateLedger,
    venue: PaperVenue,
) -> ShadowRuntimeCheckpoint:
    try:
        ledger.verify_integrity()
    except ShadowStateConflictError as exc:
        raise RuntimeRecoveryError(str(exc)) from exc
    if ledger.head_state.account_id != venue.config.account_id:
        raise RuntimeRecoveryError("paper venue and shadow ledger account ids differ")
    return ShadowRuntimeCheckpoint(
        paper=create_paper_venue_checkpoint(venue),
        shadow_ledger_manifest=ledger.manifest(),
        pending_cycle_id=ledger.pending_cycle_id,
    )


def restore_runtime_checkpoint(
    checkpoint: ShadowRuntimeCheckpoint,
) -> tuple[ShadowStateLedger, PaperVenue]:
    if content_hash(checkpoint.identity_payload()) != checkpoint.checkpoint_id:
        raise RuntimeRecoveryError("runtime checkpoint content hash mismatch")
    ledger = restore_shadow_state_ledger(checkpoint.shadow_ledger_manifest)
    venue = restore_paper_venue(checkpoint.paper)
    if ledger.head_state.account_id != venue.config.account_id:
        raise RuntimeRecoveryError("restored account identity mismatch")
    if ledger.pending_cycle_id != checkpoint.pending_cycle_id:
        raise RuntimeRecoveryError("restored pending-cycle identity mismatch")
    return ledger, venue
