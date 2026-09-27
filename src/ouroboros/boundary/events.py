"""Event factories for the check-package boundary.

Event Types (aggregate_type ``boundary``, aggregate_id = boundary id):
    boundary.check_package.enabled - aggregate_id is the run's execution id,
        not a boundary version: the check package was on for this run; the
        first boundary event of the run, written before construction
    boundary.check_package.frozen - package digest and event-safe manifest
    boundary.check_package.construction_failed - no package; verifier indeterminate
    boundary.check_package.admission_completed - base admission receipt
    boundary.actor.started - a worker bound to this boundary started
    boundary.candidate.verified - frozen package run on a candidate
    boundary.check_package.superseded - product regeneration: this boundary
        version was replaced by a later version before any worker bound to it
    boundary.binding.recorded - after the worker stopped: the tier and binding
        of every check (late bindings are data; no code, no cases)
    boundary.acceptance.resumed - a run resumed after its controller died:
        the package decision recomputed on the current workspace (visible
        cases only when the held-out cases were lost with the controller)

Payloads never carry generated file contents, check argv, or check output, so
the shared journal does not expose check code or counterexamples. Every event
names a package by its opaque id (``package.seal_package``), never by an
unkeyed digest. The package record (held-out cases as ids only) and complete
receipts are stored separately (``package.write_package_record``,
``receipts.write_receipt``).
"""

from __future__ import annotations

from typing import Any

from ouroboros.boundary.package import CheckPackage
from ouroboros.boundary.receipts import AdmissionResult, CandidateVerification
from ouroboros.events.base import BaseEvent

BOUNDARY_AGGREGATE_TYPE = "boundary"

PACKAGE_FROZEN = "boundary.check_package.frozen"
CONSTRUCTION_FAILED = "boundary.check_package.construction_failed"
ADMISSION_COMPLETED = "boundary.check_package.admission_completed"
ACTOR_STARTED = "boundary.actor.started"
CANDIDATE_VERIFIED = "boundary.candidate.verified"
SUPERSEDED = "boundary.check_package.superseded"
ACCEPTANCE_RECONCILED = "boundary.acceptance.reconciled"
BINDING_RECORDED = "boundary.binding.recorded"
ACCEPTANCE_RESUMED = "boundary.acceptance.resumed"
REFERENCE_CHECKED = "boundary.oracle.reference_checked"
CHECK_PACKAGE_ENABLED = "boundary.check_package.enabled"


def _event(boundary_id: str, event_type: str, data: dict[str, Any]) -> BaseEvent:
    return BaseEvent(
        type=event_type,
        aggregate_type=BOUNDARY_AGGREGATE_TYPE,
        aggregate_id=boundary_id,
        data=data,
    )


def cite(package_id: str | None, *, prefix: str = "") -> dict[str, str | None]:
    """``{"<prefix>package_id": package_id}``: how every event names a package."""
    return {f"{prefix}package_id": package_id}


def check_package_enabled_event(execution_id: str) -> BaseEvent:
    """The check package is on for the run ``execution_id`` (on the run's own aggregate)."""
    return _event(execution_id, CHECK_PACKAGE_ENABLED, {"execution_id": execution_id})


def package_frozen_event(
    boundary_id: str, package: CheckPackage, *, record_sha256: str | None = None
) -> BaseEvent:
    """The package id and Seed digest (I2) plus the event-safe manifest summary.

    The package must be sealed (``package.seal_package``); its unkeyed digest
    is never recorded. ``record_sha256`` is the SHA-256 of the stored package
    record's bytes (visible cases in full, held-out cases as ids only), so a
    resume in another process can tell whether the record was edited;
    omitted when not given.
    """
    summary = package.manifest_summary()
    extra = {"record_sha256": record_sha256} if record_sha256 is not None else {}
    return _event(
        boundary_id,
        PACKAGE_FROZEN,
        {
            **cite(package.package_id),
            "seed_digest": package.seed_digest,
            **extra,
            "manifest": summary,
        },
    )


def construction_failed_event(
    boundary_id: str, *, seed_digest: str, input_digest: str, reason: str
) -> BaseEvent:
    """Record that no package exists for this boundary; its verifier is indeterminate."""
    return _event(
        boundary_id,
        CONSTRUCTION_FAILED,
        {"seed_digest": seed_digest, "input_digest": input_digest, "reason": reason},
    )


def admission_completed_event(boundary_id: str, result: AdmissionResult) -> BaseEvent:
    """Base admission receipt."""
    return _event(boundary_id, ADMISSION_COMPLETED, result.event_summary())


def actor_started_event(
    boundary_id: str,
    *,
    actor_id: str,
    package_id: str | None,
    runtime: str | None,
) -> BaseEvent:
    """A worker bound to this boundary started after the boundary was sealed."""
    return _event(
        boundary_id,
        ACTOR_STARTED,
        {
            "actor_id": actor_id,
            **cite(package_id),
            "runtime": runtime,
        },
    )


def candidate_verified_event(boundary_id: str, verification: CandidateVerification) -> BaseEvent:
    """Frozen package run on one candidate."""
    return _event(boundary_id, CANDIDATE_VERIFIED, verification.event_summary())


def superseded_event(
    boundary_id: str,
    *,
    superseded_by: str,
    package_id: str | None,
    successor_package_id: str | None,
    reason: str,
) -> BaseEvent:
    """Mark a sealed boundary version as replaced by a later version."""
    return _event(
        boundary_id,
        SUPERSEDED,
        {
            "superseded_by": superseded_by,
            **cite(package_id),
            **cite(successor_package_id, prefix="successor_"),
            "reason": reason,
        },
    )


def acceptance_reconciled_event(
    boundary_id: str,
    *,
    package_id: str | None,
    reconciliation: dict[str, Any],
) -> BaseEvent:
    """Per-criterion acceptance: package verdict, existing verdict (advisory), decision."""
    return _event(
        boundary_id,
        ACCEPTANCE_RECONCILED,
        {**cite(package_id), **reconciliation},
    )


def binding_recorded_event(
    boundary_id: str, *, package_id: str, payload: dict[str, Any]
) -> BaseEvent:
    """Tiers and bindings of every check, recorded after the worker stopped."""
    return _event(boundary_id, BINDING_RECORDED, {**cite(package_id), **payload})


def reference_checked_event(
    boundary_id: str, *, package_id: str, payload: dict[str, Any]
) -> BaseEvent:
    """Cases excluded and criteria uncovered by the reference check (ids, never values)."""
    return _event(boundary_id, REFERENCE_CHECKED, {**cite(package_id), **payload})


def acceptance_resumed_event(
    boundary_id: str, *, package_id: str | None, payload: dict[str, Any]
) -> BaseEvent:
    """The package decision a resumed run recomputed (statuses and reasons, no case values)."""
    return _event(boundary_id, ACCEPTANCE_RESUMED, {**cite(package_id), **payload})
