from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping
import math

from .evidence_ledger import content_hash
from .phase50_execution_reconciliation import ReconciliationBatchReport
from .phase57_shadow_execution_cycle import ShadowExecutionCycle

PHASE60_SCHEMA_VERSION = "brian.phase60-shadow-state-ledger.v1"
StateSource = Literal["GENESIS", "RECONCILED_PAPER"]
TransitionKind = Literal[
    "GENESIS",
    "CYCLE_PROPOSED",
    "SIMULATION_CLOSED",
    "RECONCILED_COMMIT",
]


class ShadowStateConflictError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ShadowAccountState:
    account_id: str
    observed_at: float
    equity_usd: float
    available_cash_usd: float
    position_weights: tuple[tuple[str, float], ...]
    covered_assets: tuple[str, ...]
    source_kind: StateSource
    source_ref: str
    reconciliation_hash: str | None = None
    schema_version: str = PHASE60_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False
    state_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.account_id.strip() or not self.source_ref.strip():
            raise ValueError("account_id and source_ref are required")
        if not math.isfinite(self.observed_at):
            raise ValueError("observed_at must be finite")
        if not math.isfinite(self.equity_usd) or self.equity_usd <= 0:
            raise ValueError("equity_usd must be positive")
        if not math.isfinite(self.available_cash_usd) or self.available_cash_usd < 0:
            raise ValueError("available_cash_usd must be non-negative")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase 60 state must remain shadow-only")

        assets = [str(asset) for asset, _ in self.position_weights]
        if len(assets) != len(set(assets)):
            raise ValueError("position_weights cannot contain duplicate assets")
        cleaned_weights: list[tuple[str, float]] = []
        for asset, weight in self.position_weights:
            asset_id = str(asset).strip()
            value = float(weight)
            if not asset_id or not math.isfinite(value):
                raise ValueError("position weights require non-empty assets and finite values")
            if abs(value) > 1e-15:
                cleaned_weights.append((asset_id, value))

        covered = tuple(sorted({str(asset).strip() for asset in self.covered_assets if str(asset).strip()}))
        if not covered:
            raise ValueError("covered_assets must not be empty")
        if any(asset not in covered for asset, _ in cleaned_weights):
            raise ValueError("every position-weight asset must be covered by the authoritative state")

        if self.source_kind == "GENESIS":
            if self.reconciliation_hash is not None:
                raise ValueError("genesis state cannot claim reconciliation")
        elif self.source_kind == "RECONCILED_PAPER":
            if not self.reconciliation_hash or len(self.reconciliation_hash) != 64:
                raise ValueError("reconciled paper state requires a reconciliation SHA-256 hash")
        else:
            raise ValueError("unsupported state source kind")

        object.__setattr__(self, "position_weights", tuple(sorted(cleaned_weights)))
        object.__setattr__(self, "covered_assets", covered)
        object.__setattr__(self, "state_id", content_hash(self.identity_payload()))

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "account_id": self.account_id,
            "observed_at": float(self.observed_at),
            "equity_usd": float(self.equity_usd),
            "available_cash_usd": float(self.available_cash_usd),
            "position_weights": list(self.position_weights),
            "covered_assets": list(self.covered_assets),
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "reconciliation_hash": self.reconciliation_hash,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.identity_payload()
        payload["state_id"] = self.state_id
        return payload


@dataclass(frozen=True, slots=True)
class ShadowLedgerTransition:
    sequence: int
    kind: TransitionKind
    previous_transition_id: str | None
    cycle_id: str | None
    before_state_id: str | None
    after_state_id: str
    payload_hash: str
    reconciliation_hash: str | None = None
    schema_version: str = PHASE60_SCHEMA_VERSION
    transition_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if not self.payload_hash or len(self.payload_hash) != 64:
            raise ValueError("payload_hash must be a SHA-256 content hash")
        if not self.after_state_id or len(self.after_state_id) != 64:
            raise ValueError("after_state_id must be a SHA-256 content hash")
        if self.kind == "GENESIS":
            if self.sequence != 0 or self.previous_transition_id is not None:
                raise ValueError("genesis must be sequence zero with no previous transition")
            if self.cycle_id is not None or self.before_state_id is not None:
                raise ValueError("genesis cannot reference a cycle or prior state")
        else:
            if self.sequence == 0 or not self.previous_transition_id:
                raise ValueError("non-genesis transition requires previous transition")
            if not self.cycle_id or not self.before_state_id:
                raise ValueError("non-genesis transition requires cycle and before-state ids")
        if self.kind == "RECONCILED_COMMIT" and not self.reconciliation_hash:
            raise ValueError("reconciled commit requires reconciliation_hash")
        if self.kind != "RECONCILED_COMMIT" and self.reconciliation_hash is not None:
            raise ValueError("only reconciled commits may carry reconciliation_hash")
        object.__setattr__(self, "transition_id", content_hash(self.identity_payload()))

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "kind": self.kind,
            "previous_transition_id": self.previous_transition_id,
            "cycle_id": self.cycle_id,
            "before_state_id": self.before_state_id,
            "after_state_id": self.after_state_id,
            "payload_hash": self.payload_hash,
            "reconciliation_hash": self.reconciliation_hash,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.identity_payload()
        payload["transition_id"] = self.transition_id
        return payload


@dataclass(frozen=True, slots=True)
class ShadowLedgerReceipt:
    transition_id: str
    sequence: int
    cycle_id: str | None
    duplicate: bool
    head_state_id: str
    pending_cycle_id: str | None


class ShadowStateLedger:
    """Append-only continuity ledger for shadow/paper account state.

    Phase 57 simulation receipts are persisted as evidence, but they cannot mutate
    account state. A state mutation is accepted only through Phase 50's ready
    reconciliation report plus an explicit authoritative paper-account snapshot.
    """

    def __init__(self, genesis: ShadowAccountState) -> None:
        if genesis.source_kind != "GENESIS":
            raise ValueError("ledger requires a GENESIS account state")
        self._states: dict[str, ShadowAccountState] = {genesis.state_id: genesis}
        genesis_transition = ShadowLedgerTransition(
            sequence=0,
            kind="GENESIS",
            previous_transition_id=None,
            cycle_id=None,
            before_state_id=None,
            after_state_id=genesis.state_id,
            payload_hash=content_hash(genesis.to_dict()),
        )
        self._transitions: list[ShadowLedgerTransition] = [genesis_transition]
        self._head_state_id = genesis.state_id
        self._pending_cycle_id: str | None = None
        self._cycle_payload_hash: dict[str, str] = {}
        self._cycle_proposal_transition: dict[str, str] = {}
        self._cycle_terminal_transition: dict[str, str] = {}
        self._cycle_terminal_signature: dict[str, tuple[str, str | None]] = {}

    @property
    def head_state(self) -> ShadowAccountState:
        return self._states[self._head_state_id]

    @property
    def transitions(self) -> tuple[ShadowLedgerTransition, ...]:
        return tuple(self._transitions)

    @property
    def pending_cycle_id(self) -> str | None:
        return self._pending_cycle_id

    def _last_transition_id(self) -> str:
        return self._transitions[-1].transition_id

    def _receipt(self, transition: ShadowLedgerTransition, *, duplicate: bool) -> ShadowLedgerReceipt:
        return ShadowLedgerReceipt(
            transition_id=transition.transition_id,
            sequence=transition.sequence,
            cycle_id=transition.cycle_id,
            duplicate=duplicate,
            head_state_id=self._head_state_id,
            pending_cycle_id=self._pending_cycle_id,
        )

    def _transition_by_id(self, transition_id: str) -> ShadowLedgerTransition:
        for transition in self._transitions:
            if transition.transition_id == transition_id:
                return transition
        raise KeyError(transition_id)

    def append_cycle(
        self,
        cycle: ShadowExecutionCycle,
        *,
        expected_state_id: str | None = None,
    ) -> ShadowLedgerReceipt:
        if not cycle.shadow_only or cycle.live_execution or cycle.account_state_mutated:
            raise ShadowStateConflictError(
                "Phase 60 accepts only non-mutating shadow execution cycles"
            )
        if expected_state_id is not None and expected_state_id != self._head_state_id:
            raise ShadowStateConflictError("stale expected_state_id")
        if self._pending_cycle_id is not None and self._pending_cycle_id != cycle.cycle_id:
            raise ShadowStateConflictError(
                f"cycle {self._pending_cycle_id} must be closed before a new cycle starts"
            )

        payload_hash = content_hash(cycle.to_dict())
        previous_hash = self._cycle_payload_hash.get(cycle.cycle_id)
        if previous_hash is not None:
            if previous_hash != payload_hash:
                raise ShadowStateConflictError(
                    "cycle_id already exists with different execution evidence"
                )
            transition = self._transition_by_id(
                self._cycle_proposal_transition[cycle.cycle_id]
            )
            return self._receipt(transition, duplicate=True)

        transition = ShadowLedgerTransition(
            sequence=len(self._transitions),
            kind="CYCLE_PROPOSED",
            previous_transition_id=self._last_transition_id(),
            cycle_id=cycle.cycle_id,
            before_state_id=self._head_state_id,
            after_state_id=self._head_state_id,
            payload_hash=payload_hash,
        )
        self._transitions.append(transition)
        self._cycle_payload_hash[cycle.cycle_id] = payload_hash
        self._cycle_proposal_transition[cycle.cycle_id] = transition.transition_id
        self._pending_cycle_id = cycle.cycle_id
        return self._receipt(transition, duplicate=False)

    def close_simulation_cycle(
        self,
        cycle: ShadowExecutionCycle,
        *,
        expected_state_id: str | None = None,
    ) -> ShadowLedgerReceipt:
        """Close a pure simulator cycle without changing authoritative account state."""
        if self._pending_cycle_id != cycle.cycle_id:
            terminal_id = self._cycle_terminal_transition.get(cycle.cycle_id)
            if terminal_id is not None:
                signature = self._cycle_terminal_signature[cycle.cycle_id]
                if signature == ("SIMULATION_CLOSED", None):
                    return self._receipt(self._transition_by_id(terminal_id), duplicate=True)
            raise ShadowStateConflictError("cycle is not the currently pending cycle")
        if expected_state_id is not None and expected_state_id != self._head_state_id:
            raise ShadowStateConflictError("stale expected_state_id")
        if self._cycle_payload_hash.get(cycle.cycle_id) != content_hash(cycle.to_dict()):
            raise ShadowStateConflictError("cycle evidence changed after proposal")

        transition = ShadowLedgerTransition(
            sequence=len(self._transitions),
            kind="SIMULATION_CLOSED",
            previous_transition_id=self._last_transition_id(),
            cycle_id=cycle.cycle_id,
            before_state_id=self._head_state_id,
            after_state_id=self._head_state_id,
            payload_hash=content_hash({
                "cycle_id": cycle.cycle_id,
                "result": "simulation evidence recorded; authoritative state unchanged",
            }),
        )
        self._transitions.append(transition)
        self._cycle_terminal_transition[cycle.cycle_id] = transition.transition_id
        self._cycle_terminal_signature[cycle.cycle_id] = ("SIMULATION_CLOSED", None)
        self._pending_cycle_id = None
        return self._receipt(transition, duplicate=False)

    def commit_reconciled_state(
        self,
        cycle_id: str,
        reconciliation: ReconciliationBatchReport,
        state: ShadowAccountState,
        *,
        expected_state_id: str | None = None,
    ) -> ShadowLedgerReceipt:
        if self._pending_cycle_id != cycle_id:
            terminal_id = self._cycle_terminal_transition.get(cycle_id)
            if terminal_id is not None:
                reconciliation_hash = content_hash(reconciliation.to_dict())
                signature = self._cycle_terminal_signature[cycle_id]
                if signature == ("RECONCILED_COMMIT", reconciliation_hash):
                    transition = self._transition_by_id(terminal_id)
                    if transition.after_state_id != state.state_id:
                        raise ShadowStateConflictError(
                            "duplicate reconciliation commit changed authoritative state"
                        )
                    return self._receipt(transition, duplicate=True)
            raise ShadowStateConflictError("cycle is not the currently pending cycle")

        proposal = self._transition_by_id(self._cycle_proposal_transition[cycle_id])
        if proposal.before_state_id != self._head_state_id:
            raise ShadowStateConflictError(
                "authoritative head changed after cycle proposal; reconcile stale cycle first"
            )
        if expected_state_id is not None and expected_state_id != self._head_state_id:
            raise ShadowStateConflictError("stale expected_state_id")
        if not reconciliation.ready or not all(value for _, value in reconciliation.checks):
            raise ShadowStateConflictError("Phase 50 reconciliation is not ready")
        if reconciliation.live_execution:
            raise ShadowStateConflictError("live reconciliation is outside the Phase 60 shadow boundary")
        if state.source_kind != "RECONCILED_PAPER":
            raise ShadowStateConflictError("state update must come from RECONCILED_PAPER")
        if state.account_id != self.head_state.account_id:
            raise ShadowStateConflictError("account_id changed across ledger state")
        if state.observed_at < self.head_state.observed_at:
            raise ShadowStateConflictError("authoritative state timestamp moved backwards")

        reconciliation_hash = content_hash(reconciliation.to_dict())
        if state.reconciliation_hash != reconciliation_hash:
            raise ShadowStateConflictError(
                "state reconciliation_hash does not match the Phase 50 report"
            )
        missing_coverage = sorted(set(reconciliation.tracked_assets) - set(state.covered_assets))
        if missing_coverage:
            raise ShadowStateConflictError(
                f"authoritative state does not cover reconciled assets: {missing_coverage}"
            )

        self._states[state.state_id] = state
        transition = ShadowLedgerTransition(
            sequence=len(self._transitions),
            kind="RECONCILED_COMMIT",
            previous_transition_id=self._last_transition_id(),
            cycle_id=cycle_id,
            before_state_id=self._head_state_id,
            after_state_id=state.state_id,
            payload_hash=content_hash({
                "reconciliation": reconciliation.to_dict(),
                "state": state.to_dict(),
            }),
            reconciliation_hash=reconciliation_hash,
        )
        self._transitions.append(transition)
        self._head_state_id = state.state_id
        self._cycle_terminal_transition[cycle_id] = transition.transition_id
        self._cycle_terminal_signature[cycle_id] = (
            "RECONCILED_COMMIT",
            reconciliation_hash,
        )
        self._pending_cycle_id = None
        return self._receipt(transition, duplicate=False)

    def verify_integrity(self) -> bool:
        if not self._transitions:
            raise ShadowStateConflictError("ledger has no genesis transition")
        expected_previous: str | None = None
        current_state_id: str | None = None
        open_cycle: str | None = None

        for expected_sequence, transition in enumerate(self._transitions):
            if transition.sequence != expected_sequence:
                raise ShadowStateConflictError("ledger sequence gap or reorder detected")
            if transition.previous_transition_id != expected_previous:
                raise ShadowStateConflictError("transition hash chain is broken")
            if content_hash(transition.identity_payload()) != transition.transition_id:
                raise ShadowStateConflictError("transition content hash mismatch")

            if transition.kind == "GENESIS":
                if expected_sequence != 0:
                    raise ShadowStateConflictError("genesis is not first")
                current_state_id = transition.after_state_id
            elif transition.kind == "CYCLE_PROPOSED":
                if open_cycle is not None:
                    raise ShadowStateConflictError("overlapping pending cycles detected")
                if transition.before_state_id != current_state_id:
                    raise ShadowStateConflictError("cycle proposed from stale state")
                if transition.after_state_id != current_state_id:
                    raise ShadowStateConflictError("simulation proposal mutated account state")
                open_cycle = transition.cycle_id
            elif transition.kind == "SIMULATION_CLOSED":
                if transition.cycle_id != open_cycle:
                    raise ShadowStateConflictError("simulation close does not match pending cycle")
                if transition.before_state_id != current_state_id or transition.after_state_id != current_state_id:
                    raise ShadowStateConflictError("simulation close mutated account state")
                open_cycle = None
            elif transition.kind == "RECONCILED_COMMIT":
                if transition.cycle_id != open_cycle:
                    raise ShadowStateConflictError("reconciled commit does not match pending cycle")
                if transition.before_state_id != current_state_id:
                    raise ShadowStateConflictError("reconciled commit starts from stale state")
                current_state_id = transition.after_state_id
                if current_state_id not in self._states:
                    raise ShadowStateConflictError("committed state payload is missing")
                open_cycle = None
            else:
                raise ShadowStateConflictError("unknown transition kind")

            expected_previous = transition.transition_id

        if current_state_id != self._head_state_id:
            raise ShadowStateConflictError("ledger head state does not match replayed state")
        if open_cycle != self._pending_cycle_id:
            raise ShadowStateConflictError("pending cycle index does not match replayed ledger")
        if self._head_state_id not in self._states:
            raise ShadowStateConflictError("head state payload is missing")
        if content_hash(self.head_state.identity_payload()) != self.head_state.state_id:
            raise ShadowStateConflictError("head state content hash mismatch")
        return True

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": PHASE60_SCHEMA_VERSION,
            "append_only": True,
            "head_state_id": self._head_state_id,
            "pending_cycle_id": self._pending_cycle_id,
            "states": {
                state_id: state.to_dict()
                for state_id, state in sorted(self._states.items())
            },
            "transitions": [transition.to_dict() for transition in self._transitions],
            "ledger_hash": content_hash([
                transition.to_dict() for transition in self._transitions
            ]),
            "shadow_only": True,
            "live_execution": False,
        }
