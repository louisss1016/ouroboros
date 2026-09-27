"""Build frozen oracle specs and oracle packages from plain data.

Used by the product constructor (``boundary/constructor.py``), which maps its
model reply onto a package with ``package_from_reply``, and available to
callers that assemble packages themselves (the study harness). Every function
here validates; a problem raises ``CheckPackageError`` so the caller records a
construction failure. Nothing here calls a model or runs a check.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any

from pydantic import ValidationError

from ouroboros.boundary.binding import (
    BINDING_GRAMMAR,
    CHECK_DIR,
    BindingError,
    CallKind,
    default_binding_resolves,
    parse_binding,
)
from ouroboros.boundary.oracle import OracleCase, OracleSpec, is_oracle_file
from ouroboros.boundary.package import (
    ORACLE_PACKAGE_SCHEMA,
    AssertionLink,
    CheckPackage,
    CheckPackageError,
    CheckRole,
    CheckSpec,
    PackageFile,
    UncoveredObligation,
    oracle_check,
    oracle_files,
    seed_criterion_keys,
    seed_digest,
)
from ouroboros.core.seed import Seed

CHECK_INTERPRETERS = frozenset({"python3", "python"})
_CHECK_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MIN_SIGNATURE_CHARS = 12


def build_oracle_spec(
    seed: Seed,
    *,
    criterion_index: int,
    check_id: str,
    call_kind: str,
    params: Sequence[str],
    default_binding: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    base_checkout: Path | None = None,
    target_named_in_criterion: bool = False,
) -> OracleSpec:
    """Validate one criterion's oracle data and freeze it.

    Every case must declare ``held_out`` as a JSON boolean: ``false`` for a
    case the specification states, ``true`` for one it withheld. The
    declaration is frozen as given.
    ``default_resolves`` is the tier-A rule: the constructor declared that
    the criterion names the default symbol (``target_named_in_criterion``),
    or the symbol exists in ``base_checkout``.
    """
    keys = seed_criterion_keys(seed)
    if not 0 <= criterion_index < len(keys):
        raise CheckPackageError(f"{check_id}: criterion index out of range")
    key = keys[criterion_index]
    if not isinstance(target_named_in_criterion, bool):
        raise CheckPackageError(f"{check_id}: target_named_in_criterion must be true or false")
    for case in cases:
        if not isinstance(case, Mapping) or not isinstance(case.get("held_out"), bool):
            raise CheckPackageError(f"{check_id}: every case must declare held_out true or false")
    try:
        kind = CallKind(str(call_kind))
        binding = parse_binding(
            {k: v for k, v in default_binding.items() if k not in {"criterion", "criterion_key"}},
            criterion_key=key,
            params=tuple(params),
            call_kind=kind,
        )
        parsed = tuple(OracleCase.model_validate(dict(case)) for case in cases)
    except BindingError as exc:
        raise CheckPackageError(f"{check_id}: default binding: {exc.reason}") from exc
    except (ValidationError, ValueError, TypeError) as exc:
        raise CheckPackageError(f"{check_id}: oracle cases do not match the schema: {exc}") from exc
    resolves = default_binding_resolves(
        binding, base_checkout, named_in_criterion=target_named_in_criterion
    )
    try:
        return OracleSpec(
            criterion_key=key,
            check_id=check_id,
            call_kind=kind,
            params=tuple(params),
            default_binding=binding,
            default_resolves=resolves,
            cases=parsed,
            target_named_in_criterion=target_named_in_criterion,
        )
    except (ValidationError, ValueError) as exc:
        raise CheckPackageError(f"{check_id}: oracle does not match the schema: {exc}") from exc


def assemble_package(
    seed: Seed,
    *,
    input_digest: str,
    generator: str | None,
    oracles: Sequence[tuple[OracleSpec, CheckRole]] = (),
    script_checks: Sequence[CheckSpec] = (),
    script_files: Sequence[PackageFile] = (),
    uncovered: Mapping[str, str] | None = None,
    generated_at: datetime | None = None,
) -> CheckPackage:
    """One package from oracle checks, model-written script checks, and uncovered criteria.

    Criteria neither checked nor listed are added as uncovered with reason
    ``constructor_omitted``. Without oracles the package is a plain v1
    package, byte for byte what the earlier constructor produced.
    """
    keys = seed_criterion_keys(seed)
    checks = [oracle_check(spec, role) for spec, role in oracles] + list(script_checks)
    linked = {link.criterion_key for check in checks for link in check.assertions}
    gaps: dict[str, str] = {
        key: reason for key, reason in (uncovered or {}).items() if key not in linked
    }
    for key in keys:
        if key not in linked and key not in gaps:
            gaps[key] = "constructor_omitted"
    specs = tuple(spec for spec, _role in oracles)
    try:
        return CheckPackage(
            schema_version=ORACLE_PACKAGE_SCHEMA if specs else "ouroboros.check_package.v1",
            seed_digest=seed_digest(seed),
            criterion_keys=keys,
            input_digest=input_digest,
            generated_at=generated_at or datetime.now(UTC),
            generator=generator,
            checks=tuple(checks),
            files=(*oracle_files(specs), *script_files),
            uncovered=tuple(
                UncoveredObligation(criterion_key=key, reason=reason)
                for key, reason in gaps.items()
            ),
            oracles=specs,
            binding_grammar=BINDING_GRAMMAR if specs else None,
        )
    except (ValidationError, ValueError) as exc:
        if isinstance(exc, CheckPackageError):
            raise
        raise CheckPackageError(f"package does not match the schema: {exc}") from exc


def _criterion_key(keys: Sequence[str], raw: object, where: str) -> str:
    if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= len(keys):
        raise CheckPackageError(f"{where}: criterion must be a number from 1 to {len(keys)}")
    return keys[raw - 1]


def _check_file_path(raw: object) -> str:
    if not isinstance(raw, str) or not raw.startswith(f"{CHECK_DIR}/"):
        raise CheckPackageError(f"check files must live under {CHECK_DIR}/: {raw!r}")
    if is_oracle_file(raw):
        raise CheckPackageError(f"the oracle directory is reserved for product files: {raw!r}")
    return raw


def package_from_reply(
    reply: Mapping[str, Any],
    seed: Seed,
    *,
    input_digest: str,
    generator: str,
    generated_at: datetime | None = None,
    base_checkout: Path | None = None,
) -> CheckPackage:
    """Map the constructor's JSON reply onto a ``CheckPackage`` for ``seed``.

    ``oracles`` become frozen oracle checks run by the product harness
    (``boundary/oracle.py``); ``checks`` are model-written scripts. Criteria
    the reply neither links nor lists are added as uncovered with reason
    ``constructor_omitted``. Each oracle's default binding resolves (tier
    ``A``) when the reply declares ``target_named_in_criterion`` or the symbol
    exists in ``base_checkout``. Raises ``CheckPackageError`` on any
    schema problem, so the caller records a construction failure.
    """
    keys = seed_criterion_keys(seed)
    try:
        oracles = []
        for raw in reply.get("oracles") or ():
            number = raw.get("criterion")
            _criterion_key(keys, number, "oracle")
            check_id = str(raw.get("check_id") or f"oracle_{number}")
            if not _CHECK_ID.fullmatch(check_id):
                raise CheckPackageError(f"invalid check_id: {check_id!r}")
            oracles.append(
                (
                    build_oracle_spec(
                        seed,
                        criterion_index=number - 1,
                        check_id=check_id,
                        call_kind=str(raw.get("call_kind") or "function"),
                        params=tuple(str(name) for name in raw.get("params") or ()),
                        default_binding=dict(raw.get("default_binding") or {}),
                        cases=list(raw.get("cases") or ()),
                        base_checkout=base_checkout,
                        target_named_in_criterion=raw.get("target_named_in_criterion", False),
                    ),
                    CheckRole(str(raw.get("role"))),
                )
            )
        files = tuple(
            PackageFile.from_content(_check_file_path(item.get("path")), str(item["content"]))
            for item in reply.get("files") or ()
        )
        file_paths = {item.path for item in files}
        checks: list[CheckSpec] = []
        signatures: set[str] = set()
        for raw in reply.get("checks") or ():
            check_id = str(raw.get("check_id", ""))
            if not _CHECK_ID.fullmatch(check_id):
                raise CheckPackageError(f"invalid check_id: {check_id!r}")
            role = CheckRole(str(raw.get("role")))
            signature = raw.get("failure_signature") or None
            if role is CheckRole.REPRODUCTION:
                if not isinstance(signature, str) or len(signature) < _MIN_SIGNATURE_CHARS:
                    raise CheckPackageError(f"{check_id}: failure_signature is too short")
                if signature in signatures:
                    raise CheckPackageError(f"{check_id}: failure_signature is not unique")
                signatures.add(signature)
            else:
                signature = None
            argv = raw.get("argv")
            if (
                not isinstance(argv, list)
                or len(argv) != 2
                or argv[0] not in CHECK_INTERPRETERS
                or argv[1] not in file_paths
                or str(raw.get("cwd") or ".") != "."
            ):
                # The only accepted shape is the one the prompt prescribes: a
                # packaged script run by the Python interpreter from the root.
                raise CheckPackageError(
                    f"{check_id}: argv must be [python3, <packaged script>] with cwd '.'"
                )
            assertions = tuple(
                AssertionLink(
                    assertion_id=str(link.get("assertion_id") or f"{check_id}.a{number}"),
                    criterion_key=_criterion_key(keys, link.get("criterion"), check_id),
                    file=None,
                    locator=(str(link["locator"]) if link.get("locator") else None),
                )
                for number, link in enumerate(raw.get("assertions") or (), start=1)
            )
            named = raw.get("target_named_in_criterion", False)
            if not isinstance(named, bool):
                raise CheckPackageError(f"{check_id}: target_named_in_criterion must be a boolean")
            checks.append(
                CheckSpec(
                    check_id=check_id,
                    role=role,
                    argv=tuple(str(arg) for arg in argv),
                    cwd=".",
                    assertions=assertions,
                    failure_signature=signature,
                    target_named_in_criterion=named,
                )
            )
        linked = {link.criterion_key for check in checks for link in check.assertions}
        uncovered: dict[str, str] = {}
        for raw in reply.get("uncovered") or ():
            key = _criterion_key(keys, raw.get("criterion"), "uncovered")
            if key not in linked:
                uncovered[key] = str(raw.get("reason") or "unspecified").strip() or "unspecified"
        linked |= {spec.criterion_key for spec, _role in oracles}
        uncovered = {key: reason for key, reason in uncovered.items() if key not in linked}
        return assemble_package(
            seed,
            input_digest=input_digest,
            generator=generator,
            oracles=oracles,
            script_checks=checks,
            script_files=files,
            uncovered=uncovered,
            generated_at=generated_at,
        )
    except (ValidationError, ValueError, KeyError, TypeError, AttributeError) as exc:
        if isinstance(exc, CheckPackageError):
            raise
        raise CheckPackageError(f"constructor reply does not match the schema: {exc}") from exc


__all__ = [
    "CHECK_INTERPRETERS",
    "assemble_package",
    "build_oracle_spec",
    "package_from_reply",
]
