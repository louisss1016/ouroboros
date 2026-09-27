"""EventStore ordering: package digest persisted before any actor starts."""

from __future__ import annotations

from pathlib import Path

import pytest

from ouroboros.boundary.events import (
    ACTOR_STARTED,
    ADMISSION_COMPLETED,
    CANDIDATE_VERIFIED,
    CHECK_PACKAGE_ENABLED,
    PACKAGE_FROZEN,
)
from ouroboros.boundary.ledger import (
    BoundaryLeakError,
    BoundaryLedger,
    BoundaryOrderError,
    verify_boundary_order,
)
from ouroboros.boundary.package import seal_package
from ouroboros.persistence.event_store import EventStore

from .conftest import (
    INPUT_DIGEST,
    REPRO_SCRIPT,
    SIGNATURE,
    admission_receipt,
    build_package,
    verification_receipt,
)


@pytest.fixture
async def store():
    event_store = EventStore("sqlite+aiosqlite:///:memory:")
    await event_store.initialize()
    yield event_store
    await event_store.close()


@pytest.fixture
def admission(base_checkout, package):
    return admission_receipt(package, base_checkout)


async def test_package_hash_event_precedes_actor_start(
    store, tmp_path: Path, seed, package, admission
) -> None:
    ledger = BoundaryLedger(store)
    frozen = await ledger.record_package_frozen("task-1/V1", package, seed=seed)
    await ledger.record_admission("task-1/V1", admission)
    workspace = tmp_path / "worker"
    workspace.mkdir()
    (workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    started = await ledger.record_actor_started(
        "actor-1", ["task-1/V1"], workspace=workspace, runtime="codex"
    )

    events = await ledger.events("task-1/V1")
    assert [e.type for e in events] == [PACKAGE_FROZEN, ADMISSION_COMPLETED, ACTOR_STARTED]
    assert frozen.data["package_id"] == package.package_id
    assert frozen.data["seed_digest"] == package.seed_digest
    assert events[0].timestamp < events[-1].timestamp
    assert started[0].data["package_id"] == package.package_id
    assert verify_boundary_order(events) == ()
    # The journal carries ids and digests, never check code or argv.
    journal = repr([e.data for e in events])
    assert SIGNATURE not in journal
    assert REPRO_SCRIPT not in journal


async def test_actor_cannot_start_before_seal_or_admission(store, package) -> None:
    ledger = BoundaryLedger(store)
    with pytest.raises(BoundaryOrderError, match="sealed"):
        await ledger.record_actor_started("actor-1", ["task-1/V1"])
    await ledger.record_package_frozen("task-1/V1", package)
    with pytest.raises(BoundaryOrderError, match="admission"):
        await ledger.record_actor_started("actor-1", ["task-1/V1"])
    assert all(e.type != ACTOR_STARTED for e in await ledger.events("task-1/V1"))


async def test_actor_waits_for_every_bound_boundary(store, package, admission) -> None:
    ledger = BoundaryLedger(store)
    await ledger.record_package_frozen("task-1/V0", package)
    await ledger.record_admission("task-1/V0", admission)
    with pytest.raises(BoundaryOrderError):
        await ledger.record_actor_started("actor-1", ["task-1/V0", "task-1/V1"])
    await ledger.record_construction_failed(
        "task-1/V1", seed_digest=package.seed_digest, input_digest=INPUT_DIGEST, reason="parse"
    )
    started = await ledger.record_actor_started("actor-1", ["task-1/V0", "task-1/V1"])
    assert [e.data["package_id"] for e in started] == [package.package_id, None]


async def test_package_cannot_be_regenerated_after_seal(store, seed, package) -> None:
    ledger = BoundaryLedger(store)
    await ledger.record_package_frozen("task-1/V1", package)
    regenerated = build_package(seed, repro_script=REPRO_SCRIPT + "# retry\n")
    with pytest.raises(BoundaryOrderError, match="regenerated"):
        await ledger.record_package_frozen("task-1/V1", regenerated)


async def test_admission_must_cite_frozen_digest_and_is_single(
    store, seed, package, admission
) -> None:
    ledger = BoundaryLedger(store)
    other = seal_package(build_package(seed, repro_script=REPRO_SCRIPT + "# other\n"))
    await ledger.record_package_frozen("task-1/V1", other)
    with pytest.raises(BoundaryOrderError, match="different package"):
        await ledger.record_admission("task-1/V1", admission)

    await ledger.record_package_frozen("task-2/V1", package)
    await ledger.record_admission("task-2/V1", admission)
    with pytest.raises(BoundaryOrderError, match="already recorded"):
        await ledger.record_admission("task-2/V1", admission)


async def test_workspace_with_generated_check_code_is_refused(
    store, tmp_path: Path, package, admission
) -> None:
    ledger = BoundaryLedger(store)
    await ledger.record_package_frozen("task-1/V1", package)
    await ledger.record_admission("task-1/V1", admission)
    workspace = tmp_path / "worker"
    workspace.mkdir()
    (workspace / "hidden_copy.py").write_text(REPRO_SCRIPT)
    with pytest.raises(BoundaryLeakError):
        await ledger.record_actor_started("actor-1", ["task-1/V1"], workspace=workspace)


async def test_candidate_verification_cites_the_frozen_package(
    store, tmp_path: Path, base_checkout, package, admission
) -> None:
    ledger = BoundaryLedger(store)
    await ledger.record_package_frozen("task-1/V1", package)
    await ledger.record_admission("task-1/V1", admission)
    await ledger.record_actor_started("actor-1", ["task-1/V1"])
    verification = verification_receipt(package, base_checkout)
    event = await ledger.record_candidate_verification("task-1/V1", verification)

    assert event.type == CANDIDATE_VERIFIED
    assert event.data["verdict"] == "fail"
    assert verify_boundary_order(await ledger.events("task-1/V1")) == ()


def test_verify_boundary_order_flags_actor_before_seal(package) -> None:
    from ouroboros.boundary.events import actor_started_event, package_frozen_event

    actor = actor_started_event("b", actor_id="a", package_id=None, runtime=None)
    frozen = package_frozen_event("b", package)
    violations = verify_boundary_order([actor, frozen])
    assert "actor started before the seal" in violations


async def test_the_enabled_record_is_written_once_per_run_and_needs_an_execution_id(
    store: EventStore,
) -> None:
    ledger = BoundaryLedger(store)
    assert not await ledger.check_package_enabled("exec_enabled")
    event = await ledger.record_check_package_enabled("exec_enabled")
    assert event.type == CHECK_PACKAGE_ENABLED and event.aggregate_id == "exec_enabled"
    assert await ledger.check_package_enabled("exec_enabled")
    with pytest.raises(BoundaryOrderError):
        await ledger.record_check_package_enabled("exec_enabled")
    with pytest.raises(BoundaryOrderError):
        await ledger.record_check_package_enabled("")
