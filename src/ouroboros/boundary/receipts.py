"""Receipts of a check package run: per-check executions, admission, candidate verification.

These are data only. ``boundary/admission.py`` produces them; the ledger
(``boundary/ledger.py``) records their journal-safe form and ``write_receipt``
stores the complete form outside every checkout.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ouroboros.boundary.oracle import journal_safe_oracle_result, redact_held_out
from ouroboros.boundary.package import (
    CheckRole,
    canonical_json_bytes,
    sha256_bytes,
)


class CheckStatus(StrEnum):
    """Outcome of one check against its role contract."""

    EXPECTED = "expected"
    VIOLATED = "violated"
    INDETERMINATE = "indeterminate"


class PackageVerdict(StrEnum):
    """Admission verdict on the base checkout (after per-check exclusions)."""

    ADMITTED = "admitted"
    REJECTED = "rejected"
    INDETERMINATE = "indeterminate"


class CandidateVerdict(StrEnum):
    """Verdict of the admitted checks on a candidate checkout."""

    PASS = "pass"
    FAIL = "fail"
    INDETERMINATE = "indeterminate"


class CheckExecution(BaseModel, frozen=True):
    """Receipt for one check command on one isolated copy."""

    check_id: str
    role: CheckRole
    argv: tuple[str, ...]
    cwd: str
    status: CheckStatus
    reason: str
    return_code: int | None
    timed_out: bool
    duration_seconds: float
    signature_seen: bool
    stdout_sha256: str
    stderr_sha256: str
    output_tail: str
    protected_digest_before: str
    protected_digest_after: str
    mutated_paths: tuple[str, ...]
    scratch_outputs: tuple[str, ...]
    undeclared_outputs: tuple[str, ...]
    # Oracle split (optional; omitted from receipts and events while unset).
    tier: str | None = None
    binding: dict[str, Any] | None = None
    oracle_result: dict[str, Any] | None = None


class AdmissionResult(BaseModel, frozen=True):
    """Admission receipt on the pinned base checkout."""

    schema_version: Literal["ouroboros.check_admission.v2"] = "ouroboros.check_admission.v2"
    # In-memory identity of the package that ran (``CheckPackage.sha256``);
    # never dumped, so no stored receipt or journal event carries it.
    package_sha256: str = Field(exclude=True)
    # What a stored receipt or journal event cites (``CheckPackage.package_id``);
    # ``None`` for a package that was never sealed.
    package_id: str | None = None
    seed_digest: str
    base_tree_digest: str
    base_tree_digest_after: str
    verdict: PackageVerdict
    reasons: tuple[str, ...]
    protected_bytes_mutated: bool
    timeout_seconds: int
    checks: tuple[CheckExecution, ...]
    started_at: datetime
    completed_at: datetime
    interpreter: str | None = None
    interpreter_source: str | None = None
    # SHA-256 of the pinned interpreter binary (``check_env.CheckInterpreter``):
    # a resumed run checks the interpreter it resolves against it.
    interpreter_sha256: str | None = None
    # SHA-256 of the pinned binary's real path (``CheckInterpreter.realpath_sha256``).
    interpreter_realpath_sha256: str | None = None
    check_tiers: dict[str, str] | None = None
    # Per-check admission (``boundary/per_check.py``): check id to the reason
    # it was excluded while the rest of the package was admitted.
    excluded_checks: dict[str, str] | None = None

    def event_summary(self) -> dict[str, Any]:
        """Return the journal payload: statuses and digests, no argv or output."""
        return _journal_safe(self)


class CandidateVerification(BaseModel, frozen=True):
    """Receipt for running the unchanged frozen package on a candidate."""

    schema_version: Literal["ouroboros.candidate_verification.v2"] = (
        "ouroboros.candidate_verification.v2"
    )
    package_sha256: str = Field(exclude=True)
    package_id: str | None = None
    artifact_tree_digest: str
    artifact_tree_digest_after: str
    verdict: CandidateVerdict
    reasons: tuple[str, ...]
    protected_bytes_mutated: bool
    timeout_seconds: int
    checks: tuple[CheckExecution, ...]
    started_at: datetime
    completed_at: datetime
    interpreter: str | None = None
    interpreter_source: str | None = None
    check_tiers: dict[str, str] | None = None
    bindings: dict[str, dict[str, Any]] | None = None

    def event_summary(self) -> dict[str, Any]:
        """Return the journal payload: statuses and digests, no argv or output."""
        return _journal_safe(self)


_JOURNAL_EXCLUDED_CHECK_FIELDS = frozenset({"argv", "output_tail"})
# Optional receipt fields: omitted when unset (callers that pass no
# interpreter keep byte-identical receipts), and the interpreter's absolute
# path stays in the stored receipt only, never in the journal.
_OPTIONAL_RECEIPT_FIELDS = (
    "interpreter",
    "interpreter_source",
    "interpreter_sha256",
    "interpreter_realpath_sha256",
    "check_tiers",
    "bindings",
    "excluded_checks",
)
_JOURNAL_EXCLUDED_RECEIPT_FIELDS = frozenset({"interpreter"})
# Fields added with the oracle split: omitted while unset, so receipts and
# events of a package without oracles keep their earlier bytes.
_OPTIONAL_CHECK_KEYS = ("tier", "binding", "oracle_result")


def _receipt_dump(receipt: BaseModel) -> dict[str, Any]:
    data = receipt.model_dump(mode="json")
    for key in _OPTIONAL_RECEIPT_FIELDS:
        if data.get(key) is None:
            data.pop(key, None)
    for check in data.get("checks", ()):
        for key in _OPTIONAL_CHECK_KEYS:
            if check.get(key) is None:
                check.pop(key, None)
        if "oracle_result" in check:
            # A held-out case keeps its id and pass/fail only.
            check["oracle_result"] = redact_held_out(check["oracle_result"])
    return data


def _journal_safe(receipt: BaseModel) -> dict[str, Any]:
    """Dump a receipt without check argv or output text.

    The event journal is readable by other tools, so it carries statuses,
    reasons, and digests only. The complete receipt (argv, output tails) is a
    separately stored artifact, see ``write_receipt``.
    """
    data = _receipt_dump(receipt)
    for key in _JOURNAL_EXCLUDED_RECEIPT_FIELDS:
        data.pop(key, None)
    data["checks"] = [
        {key: value for key, value in check.items() if key not in _JOURNAL_EXCLUDED_CHECK_FIELDS}
        for check in data["checks"]
    ]
    for check in data["checks"]:
        if "oracle_result" in check:
            # Case pass/fail and held-out flags only: no inputs, no observations.
            check["oracle_result"] = journal_safe_oracle_result(check["oracle_result"])
    return data


def write_receipt(receipt: AdmissionResult | CandidateVerification, directory: Path) -> Path:
    """Write a complete receipt to ``<directory>/<sha256>.json`` (create-only)."""
    directory.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(_receipt_dump(receipt))
    target = directory / f"{sha256_bytes(data)}.json"
    if not target.exists():
        with open(target, "xb") as handle:
            handle.write(data)
    return target
