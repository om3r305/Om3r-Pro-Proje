from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

from .phase60_shadow_state_ledger import ShadowAccountState, ShadowStateLedger
from .phase61_stateful_paper_venue import PaperVenue, PaperVenueConfig
from .phase64_local_execution_projector import LocalExecutionProjector
from .phase66_runtime_coordinator import ShadowPaperRuntimeCoordinator
from .phase67_durable_runtime_orchestrator import DurableShadowPaperRuntime
from .phase68_operational_risk_governor import (
    EquityPoint,
    OperationalRiskPolicy,
)
from .phase70_durable_runtime_store import DurableRuntimeStore
from .phase71_persisted_runtime_supervisor import (
    PersistedDurableRuntimeSupervisor,
)
from .phase72_operational_risk_ledger import OperationalRiskLedger
from .phase73_operational_risk_store import OperationalRiskStore
from .phase91_supabase_rpc_transport import SupabaseRecoveryRpcTransport

PHASE121_SCHEMA_VERSION = "brian.phase121-runtime-bootstrap.v1"
_CRYPTO_ASSET = re.compile(r"^crypto:[A-Z0-9]{2,20}USDT$")


class RuntimeBootstrapError(RuntimeError):
    pass


class RuntimeBootstrapConflictError(RuntimeBootstrapError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeBootstrapSpec:
    runtime_id: str
    account_id: str
    covered_assets: tuple[str, ...]
    starting_cash_usd: float
    paper_fee_bps: float
    allow_short: bool
    observed_at: float
    source_ref: str
    risk_policy: OperationalRiskPolicy
    schema_version: str = PHASE121_SCHEMA_VERSION

    def __post_init__(self) -> None:
        runtime_id = self.runtime_id.strip()
        account_id = self.account_id.strip()
        source_ref = self.source_ref.strip()
        assets = tuple(sorted({str(asset).strip() for asset in self.covered_assets}))
        if not runtime_id or not account_id or not source_ref:
            raise ValueError("runtime_id, account_id and source_ref are required")
        if not assets or any(not _CRYPTO_ASSET.fullmatch(asset) for asset in assets):
            raise ValueError(
                "covered_assets must contain canonical crypto:*USDT ids"
            )
        if not math.isfinite(self.starting_cash_usd) or self.starting_cash_usd <= 0:
            raise ValueError("starting_cash_usd must be positive")
        if not math.isfinite(self.paper_fee_bps) or self.paper_fee_bps < 0:
            raise ValueError("paper_fee_bps must be finite and non-negative")
        if not math.isfinite(self.observed_at):
            raise ValueError("observed_at must be finite")
        if not isinstance(self.allow_short, bool):
            raise ValueError("allow_short must be boolean")
        object.__setattr__(self, "runtime_id", runtime_id)
        object.__setattr__(self, "account_id", account_id)
        object.__setattr__(self, "source_ref", source_ref)
        object.__setattr__(self, "covered_assets", assets)


@dataclass(frozen=True, slots=True)
class RuntimeBootstrapArtifacts:
    runtime: DurableShadowPaperRuntime
    risk_ledger: OperationalRiskLedger
    genesis_checkpoint_id: str
    risk_ledger_hash: str
    schema_version: str = PHASE121_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if len(self.genesis_checkpoint_id) != 64:
            raise ValueError("genesis_checkpoint_id must be a content hash")
        if len(self.risk_ledger_hash) != 64:
            raise ValueError("risk_ledger_hash must be a content hash")
        if not self.shadow_only or self.live_execution:
            raise ValueError("bootstrap artifacts must remain shadow-only")


@dataclass(frozen=True, slots=True)
class RuntimeBootstrapReceipt:
    runtime_id: str
    status: str
    runtime_version: int
    risk_version: int
    checkpoint_id: str
    risk_ledger_hash: str
    completed_partial_bootstrap: bool
    already_bootstrapped: bool
    bootstrap_id: str = field(init=False)
    schema_version: str = PHASE121_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if self.status not in {
            "BOOTSTRAPPED",
            "COMPLETED_PARTIAL_BOOTSTRAP",
            "ALREADY_BOOTSTRAPPED",
        }:
            raise ValueError("invalid bootstrap status")
        if self.runtime_version <= 0 or self.risk_version <= 0:
            raise ValueError("bootstrap versions must be positive")
        if len(self.checkpoint_id) != 64 or len(self.risk_ledger_hash) != 64:
            raise ValueError("bootstrap hashes must be content hashes")
        if not self.shadow_only or self.live_execution:
            raise ValueError("bootstrap receipt must remain shadow-only")
        payload = {
            "schema_version": self.schema_version,
            "runtime_id": self.runtime_id,
            "status": self.status,
            "runtime_version": self.runtime_version,
            "risk_version": self.risk_version,
            "checkpoint_id": self.checkpoint_id,
            "risk_ledger_hash": self.risk_ledger_hash,
            "completed_partial_bootstrap": self.completed_partial_bootstrap,
            "already_bootstrapped": self.already_bootstrapped,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
        }
        object.__setattr__(
            self,
            "bootstrap_id",
            hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest(),
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return payload


def build_runtime_bootstrap_artifacts(
    spec: RuntimeBootstrapSpec,
) -> RuntimeBootstrapArtifacts:
    genesis = ShadowAccountState(
        account_id=spec.account_id,
        observed_at=float(spec.observed_at),
        equity_usd=float(spec.starting_cash_usd),
        available_cash_usd=float(spec.starting_cash_usd),
        position_weights=(),
        covered_assets=spec.covered_assets,
        source_kind="GENESIS",
        source_ref=spec.source_ref,
    )
    coordinator = ShadowPaperRuntimeCoordinator(
        ShadowStateLedger(genesis),
        PaperVenue(
            PaperVenueConfig(
                account_id=spec.account_id,
                starting_cash_usd=float(spec.starting_cash_usd),
                fee_bps=float(spec.paper_fee_bps),
                allow_short=spec.allow_short,
            )
        ),
        LocalExecutionProjector(spec.account_id),
    )
    runtime = DurableShadowPaperRuntime(coordinator)
    checkpoint = runtime.checkpoint()

    risk_ledger = OperationalRiskLedger(spec.risk_policy)
    governor = risk_ledger.governor()
    receipt = governor.evaluate(
        now=float(spec.observed_at),
        equity_points=(
            EquityPoint(float(spec.observed_at), float(spec.starting_cash_usd)),
        ),
        closed_trades=(),
        health_events=(),
        market_data_timestamp=float(spec.observed_at),
    )
    appended = risk_ledger.append(receipt)
    if appended.trading_state != "ACTIVE" or appended.halt_latched:
        raise RuntimeBootstrapError(
            "bootstrap operational-risk receipt did not initialize ACTIVE"
        )
    manifest = risk_ledger.manifest()
    return RuntimeBootstrapArtifacts(
        runtime=runtime,
        risk_ledger=risk_ledger,
        genesis_checkpoint_id=checkpoint.checkpoint_id,
        risk_ledger_hash=str(manifest["ledger_hash"]),
    )


def _validate_existing_runtime_identity(
    *,
    stored,
    spec: RuntimeBootstrapSpec,
) -> None:
    restored = DurableShadowPaperRuntime.restore(stored.checkpoint)
    head = restored.ledger.head_state
    if head.account_id != spec.account_id:
        raise RuntimeBootstrapConflictError(
            "existing runtime account_id differs from bootstrap spec"
        )
    if tuple(head.covered_assets) != spec.covered_assets:
        raise RuntimeBootstrapConflictError(
            "existing runtime covered_assets differ from bootstrap spec"
        )


def _validate_existing_risk_identity(
    *,
    stored,
    artifacts: RuntimeBootstrapArtifacts,
) -> None:
    if stored.policy_hash != artifacts.risk_ledger.policy_hash:
        raise RuntimeBootstrapConflictError(
            "existing operational-risk policy differs from bootstrap spec"
        )


class RuntimeBootstrapper:
    """Idempotent two-head bootstrap under one Phase70 fencing lease.

    The Phase70 checkpoint is committed first because Phase73 commits require the
    same runtime lease/fence. A crash after the runtime commit but before the risk
    commit leaves a recognizable partial bootstrap. Re-entry may complete that
    partial state only when the existing runtime is still the exact expected
    genesis checkpoint. Any other mixed state fails closed.
    """

    def __init__(
        self,
        *,
        rpc,
        lease_seconds: int = 60,
    ) -> None:
        if not callable(rpc):
            raise TypeError("rpc must be callable")
        if lease_seconds < 10 or lease_seconds > 300:
            raise ValueError("lease_seconds must be in [10,300]")
        self.rpc = rpc
        self.runtime_store = DurableRuntimeStore(rpc)
        self.risk_store = OperationalRiskStore(rpc)
        self.lease_seconds = int(lease_seconds)

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        client=None,
    ) -> "RuntimeBootstrapper":
        rpc = SupabaseRecoveryRpcTransport.from_env(
            env=env,
            client=client,
        )
        source = {} if env is None else env
        raw = source.get("BRIAN_RUNTIME_LEASE_SECONDS", "60")
        try:
            lease_seconds = int(raw)
        except (TypeError, ValueError) as exc:
            rpc.close()
            raise ValueError(
                "BRIAN_RUNTIME_LEASE_SECONDS must be integer"
            ) from exc
        bootstrapper = cls(
            rpc=rpc,
            lease_seconds=lease_seconds,
        )
        bootstrapper._owns_rpc = True
        return bootstrapper

    def bootstrap(
        self,
        *,
        spec: RuntimeBootstrapSpec,
        owner_token: str,
    ) -> RuntimeBootstrapReceipt:
        if not owner_token.strip():
            raise ValueError("owner_token is required")
        artifacts = build_runtime_bootstrap_artifacts(spec)

        existing_runtime = self.runtime_store.load(
            runtime_id=spec.runtime_id
        )
        existing_risk = self.risk_store.load(
            runtime_id=spec.runtime_id
        )

        if existing_runtime is None and existing_risk is not None:
            raise RuntimeBootstrapConflictError(
                "risk head exists without durable runtime head"
            )

        if existing_runtime is not None and existing_risk is not None:
            _validate_existing_runtime_identity(
                stored=existing_runtime,
                spec=spec,
            )
            _validate_existing_risk_identity(
                stored=existing_risk,
                artifacts=artifacts,
            )
            return RuntimeBootstrapReceipt(
                runtime_id=spec.runtime_id,
                status="ALREADY_BOOTSTRAPPED",
                runtime_version=existing_runtime.version,
                risk_version=existing_risk.version,
                checkpoint_id=existing_runtime.checkpoint.checkpoint_id,
                risk_ledger_hash=existing_risk.ledger_hash,
                completed_partial_bootstrap=False,
                already_bootstrapped=True,
            )

        partial = existing_runtime is not None and existing_risk is None
        if partial:
            if (
                existing_runtime.checkpoint.checkpoint_id
                != artifacts.genesis_checkpoint_id
            ):
                raise RuntimeBootstrapConflictError(
                    "runtime advanced beyond expected genesis while risk head is missing"
                )
            _validate_existing_runtime_identity(
                stored=existing_runtime,
                spec=spec,
            )

        supervisor = PersistedDurableRuntimeSupervisor.acquire(
            store=self.runtime_store,
            runtime_id=spec.runtime_id,
            owner_token=owner_token,
            lease_seconds=self.lease_seconds,
            initial_runtime=(
                artifacts.runtime
                if existing_runtime is None
                else None
            ),
        )
        try:
            if partial:
                if (
                    supervisor.runtime.checkpoint().checkpoint_id
                    != artifacts.genesis_checkpoint_id
                ):
                    raise RuntimeBootstrapConflictError(
                        "leased runtime no longer matches partial bootstrap genesis"
                    )
            risk_commit = self.risk_store.commit(
                supervisor.lease,
                expected_version=0,
                ledger=artifacts.risk_ledger,
            )
            if risk_commit.status not in {
                "COMMITTED",
                "DUPLICATE_CURRENT",
            } or not risk_commit.committed:
                raise RuntimeBootstrapError(
                    f"operational-risk bootstrap commit rejected: "
                    f"{risk_commit.status}"
                )

            stored_runtime = self.runtime_store.load(
                runtime_id=spec.runtime_id
            )
            stored_risk = self.risk_store.load(
                runtime_id=spec.runtime_id
            )
            if stored_runtime is None or stored_risk is None:
                raise RuntimeBootstrapError(
                    "bootstrap verification could not reload both durable heads"
                )
            _validate_existing_runtime_identity(
                stored=stored_runtime,
                spec=spec,
            )
            _validate_existing_risk_identity(
                stored=stored_risk,
                artifacts=artifacts,
            )
            return RuntimeBootstrapReceipt(
                runtime_id=spec.runtime_id,
                status=(
                    "COMPLETED_PARTIAL_BOOTSTRAP"
                    if partial
                    else "BOOTSTRAPPED"
                ),
                runtime_version=stored_runtime.version,
                risk_version=stored_risk.version,
                checkpoint_id=stored_runtime.checkpoint.checkpoint_id,
                risk_ledger_hash=stored_risk.ledger_hash,
                completed_partial_bootstrap=partial,
                already_bootstrapped=False,
            )
        finally:
            supervisor.release()

    def close(self) -> bool:
        rpc = getattr(self, "rpc", None)
        if getattr(self, "_owns_rpc", False) and hasattr(rpc, "close"):
            rpc.close()
        return True

    def __enter__(self) -> "RuntimeBootstrapper":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
