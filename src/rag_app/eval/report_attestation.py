"""HMAC attestation for private RAG evaluation reports."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rag_app.eval.baseline import BaselineReport
from rag_app.eval.gold_set import GoldRecord, gold_record_case_sha256
from rag_app.eval.private_artifacts import (
    PrivateArtifactError,
    read_private_bytes,
    write_private_json_fresh,
)
from rag_app.eval.private_sidecar import PrivateSidecarRecord

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_GIT_SHA_PATTERN = r"^[0-9a-f]{40,64}$"
_MIN_KEY_BYTES = 32
_MAX_KEY_BYTES = 4096
_MAX_ATTESTATION_BYTES = 8 * 1024 * 1024

DEFAULT_ATTESTED_SOURCES = (
    "pyproject.toml",
    "scripts/evaluate_rag_gold_set.py",
    "src/rag_app/config.py",
    "src/rag_app/eval/baseline.py",
    "src/rag_app/eval/gold_set.py",
    "src/rag_app/eval/private_artifacts.py",
    "src/rag_app/eval/private_sidecar.py",
    "src/rag_app/eval/rag_metrics.py",
    "src/rag_app/eval/report_attestation.py",
    "src/rag_app/llm/embeddings.py",
    "src/rag_app/llm/visual.py",
    "src/rag_app/llm/visual_reranker.py",
    "src/rag_app/pipeline/validate.py",
    "src/rag_app/rag/chat.py",
    "src/rag_app/rag/retrieve.py",
    "src/rag_app/storage/s3.py",
    "uv.lock",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AttestedCase(_StrictModel):
    case_id: str = Field(min_length=1, max_length=128)
    gold_case_sha256: str = Field(pattern=_SHA256_PATTERN)
    sidecar_case_sha256: str = Field(pattern=_SHA256_PATTERN)


class AttestedSource(_StrictModel):
    path: str = Field(min_length=1, max_length=512)
    size_bytes: int = Field(ge=1, le=64 * 1024 * 1024)
    sha256: str = Field(pattern=_SHA256_PATTERN)


class ReportAttestation(_StrictModel):
    schema_version: Literal["rag-baseline-attestation-v1"]
    algorithm: Literal["hmac-sha256"]
    key_id: str = Field(pattern=_SHA256_PATTERN)
    created_at: datetime
    report_sha256: str = Field(pattern=_SHA256_PATTERN)
    report_size_bytes: int = Field(ge=1, le=64 * 1024 * 1024)
    gold_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    gold_size_bytes: int = Field(ge=1, le=256 * 1024 * 1024)
    sidecar_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    sidecar_size_bytes: int = Field(ge=1, le=256 * 1024 * 1024)
    case_count: int = Field(ge=1, le=500)
    cases: tuple[AttestedCase, ...] = Field(min_length=1, max_length=500)
    case_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    configuration_sha256: str = Field(pattern=_SHA256_PATTERN)
    corpus_fingerprint_sha256: str = Field(pattern=_SHA256_PATTERN)
    runtime_corpus_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    repository_git_sha: str = Field(pattern=_GIT_SHA_PATTERN)
    repository_git_dirty: Literal[False]
    sources: tuple[AttestedSource, ...] = Field(min_length=1, max_length=32)
    source_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    signature: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        case_ids = [item.case_id for item in self.cases]
        if case_ids != sorted(case_ids) or len(case_ids) != len(set(case_ids)):
            raise ValueError("attested cases must be unique and sorted")
        source_paths = [item.path for item in self.sources]
        if source_paths != sorted(source_paths) or len(source_paths) != len(set(source_paths)):
            raise ValueError("attested sources must be unique and sorted")
        if self.case_count != len(self.cases):
            raise ValueError("case_count does not match cases")
        if self.case_manifest_sha256 != _manifest_sha256(self.cases):
            raise ValueError("case manifest hash mismatch")
        if self.source_manifest_sha256 != _manifest_sha256(self.sources):
            raise ValueError("source manifest hash mismatch")
        return self


class PrivateArtifactAttestation(_StrictModel):
    schema_version: Literal["rag-private-artifact-attestation-v1"]
    algorithm: Literal["hmac-sha256"]
    artifact_type: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,63}$")
    key_id: str = Field(pattern=_SHA256_PATTERN)
    created_at: datetime
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    artifact_size_bytes: int = Field(ge=1, le=256 * 1024 * 1024)
    repository_git_sha: str = Field(pattern=_GIT_SHA_PATTERN)
    repository_git_dirty: Literal[False]
    sources: tuple[AttestedSource, ...] = Field(min_length=1, max_length=32)
    source_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    signature: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        source_paths = [item.path for item in self.sources]
        if source_paths != sorted(source_paths) or len(source_paths) != len(set(source_paths)):
            raise ValueError("attested sources must be unique and sorted")
        if self.source_manifest_sha256 != _manifest_sha256(self.sources):
            raise ValueError("source manifest hash mismatch")
        return self


class ReportAttestationError(ValueError):
    """Fail-closed error that never contains private report or Gold values."""


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _model_payload(value: BaseModel) -> dict[str, Any]:
    return value.model_dump(mode="json")


def _manifest_sha256(values: Sequence[BaseModel]) -> str:
    return _sha256(_canonical_json([_model_payload(item) for item in values]))


def _unsigned_payload(attestation: BaseModel) -> dict[str, Any]:
    return attestation.model_dump(mode="json", exclude={"signature"})


def _signature(attestation: BaseModel, key: bytes) -> str:
    return hmac.new(key, _canonical_json(_unsigned_payload(attestation)), hashlib.sha256).hexdigest()


def _sidecar_case_sha256(sidecar: PrivateSidecarRecord) -> str:
    return _sha256(_canonical_json(sidecar.model_dump(mode="json")))


def build_case_attestations(
    records: Sequence[GoldRecord],
    sidecars: Mapping[str, PrivateSidecarRecord],
) -> tuple[AttestedCase, ...]:
    if len({record.case_id for record in records}) != len(records):
        raise ReportAttestationError("Gold case IDs are not unique")
    if set(sidecars) != {record.case_id for record in records}:
        raise ReportAttestationError("Gold and sidecar case IDs do not match")
    result: list[AttestedCase] = []
    for record in sorted(records, key=lambda item: item.case_id):
        sidecar = sidecars[record.case_id]
        gold_hash = gold_record_case_sha256(record)
        if sidecar.case_id != record.case_id or sidecar.gold_case_sha256 != gold_hash:
            raise ReportAttestationError("Gold and sidecar case binding is invalid")
        result.append(
            AttestedCase(
                case_id=record.case_id,
                gold_case_sha256=gold_hash,
                sidecar_case_sha256=_sidecar_case_sha256(sidecar),
            )
        )
    return tuple(result)


def _read_regular_file_once(
    path: Path,
    *,
    exact_mode: int | None = None,
    expected_location: Path | None = None,
    require_owner: bool = False,
    max_bytes: int,
) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path.expanduser(), flags)
    except OSError:
        raise ReportAttestationError("unable to open attestation input") from None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ReportAttestationError("attestation input must be one regular file")
        if expected_location is not None:
            try:
                actual_location = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
            except OSError:
                raise ReportAttestationError("unable to verify attestation input path") from None
            if actual_location != expected_location:
                raise ReportAttestationError("attestation input path changed while opening")
        if exact_mode is not None and stat.S_IMODE(info.st_mode) != exact_mode:
            raise ReportAttestationError("attestation input has unsafe permissions")
        if require_owner and info.st_uid != os.geteuid():
            raise ReportAttestationError("attestation input has an unexpected owner")
        if info.st_size < 1 or info.st_size > max_bytes:
            raise ReportAttestationError("attestation input size is invalid")
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ReportAttestationError("attestation input changed while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ReportAttestationError("attestation input changed while reading")
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
        ):
            raise ReportAttestationError("attestation input changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_hmac_key(key_path: Path, repository_root: Path) -> bytes:
    if not key_path.expanduser().is_absolute():
        raise ReportAttestationError("attestation key path must be absolute")
    try:
        key_location = key_path.expanduser().resolve(strict=True)
        repository = repository_root.expanduser().resolve(strict=True)
    except OSError:
        raise ReportAttestationError("unable to resolve attestation key") from None
    try:
        key_location.relative_to(repository)
    except ValueError:
        pass
    else:
        raise ReportAttestationError("attestation key must be outside the repository")
    try:
        artifact = read_private_bytes(key_path, max_bytes=_MAX_KEY_BYTES)
    except PrivateArtifactError:
        raise ReportAttestationError("attestation key cannot be read safely") from None
    key = artifact.raw_bytes
    if len(key) < _MIN_KEY_BYTES:
        raise ReportAttestationError("attestation key is too short")
    return key


def _git_output(repository_root: Path, arguments: list[str]) -> str:
    try:
        return subprocess.run(  # noqa: S603
            ["git", *arguments],  # noqa: S607
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        raise ReportAttestationError("unable to verify repository state") from None


def _git_bytes(repository_root: Path, arguments: list[str]) -> bytes:
    try:
        return subprocess.run(  # noqa: S603
            ["git", *arguments],  # noqa: S607
            cwd=repository_root,
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        raise ReportAttestationError("unable to verify repository source") from None


def _clean_git_sha(repository_root: Path) -> str:
    revision = _git_output(repository_root, ["rev-parse", "HEAD"])
    if not re.fullmatch(_GIT_SHA_PATTERN, revision):
        raise ReportAttestationError("repository revision is invalid")
    if _git_output(repository_root, ["status", "--porcelain", "--untracked-files=all"]):
        raise ReportAttestationError("report attestation requires a clean repository")
    return revision


def _source_manifest(
    repository_root: Path,
    source_paths: Sequence[str],
) -> tuple[AttestedSource, ...]:
    repository = repository_root.expanduser().resolve(strict=True)
    if len(set(source_paths)) != len(source_paths) or not source_paths:
        raise ReportAttestationError("attested source paths must be unique")
    result: list[AttestedSource] = []
    for relative_value in sorted(source_paths):
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != relative_value:
            raise ReportAttestationError("attested source path is invalid")
        if not _git_output(repository, ["ls-files", "--error-unmatch", "--", relative_value]):
            raise ReportAttestationError("attested source is not tracked")
        raw = _read_regular_file_once(repository / relative, max_bytes=64 * 1024 * 1024)
        if raw != _git_bytes(repository, ["show", f"HEAD:{relative_value}"]):
            raise ReportAttestationError("attested source does not match the Git revision")
        result.append(AttestedSource(path=relative_value, size_bytes=len(raw), sha256=_sha256(raw)))
    return tuple(result)


def _source_manifest_at_revision(
    repository_root: Path,
    source_paths: Sequence[str],
    revision: str,
) -> tuple[AttestedSource, ...]:
    if not re.fullmatch(_GIT_SHA_PATTERN, revision):
        raise ReportAttestationError("attested repository revision is invalid")
    if len(set(source_paths)) != len(source_paths) or not source_paths:
        raise ReportAttestationError("attested source paths must be unique")
    result: list[AttestedSource] = []
    for relative_value in sorted(source_paths):
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != relative_value:
            raise ReportAttestationError("attested source path is invalid")
        raw = _git_bytes(repository_root, ["show", f"{revision}:{relative_value}"])
        if not raw:
            raise ReportAttestationError("attested source is empty or unavailable")
        result.append(AttestedSource(path=relative_value, size_bytes=len(raw), sha256=_sha256(raw)))
    return tuple(result)


def _parse_report(report_bytes: bytes) -> BaselineReport:
    try:
        json.loads(report_bytes, object_pairs_hook=_reject_duplicate_keys)
        return BaselineReport.model_validate_json(report_bytes, strict=True)
    except (_DuplicateKeyError, json.JSONDecodeError, ValidationError):
        raise ReportAttestationError("baseline report is invalid") from None


def create_report_attestation(
    *,
    report_bytes: bytes,
    gold_bytes: bytes,
    sidecar_bytes: bytes,
    cases: Sequence[AttestedCase],
    key: bytes,
    repository_root: Path,
    source_paths: Sequence[str] = DEFAULT_ATTESTED_SOURCES,
    created_at: datetime | None = None,
) -> ReportAttestation:
    if not _MIN_KEY_BYTES <= len(key) <= _MAX_KEY_BYTES:
        raise ReportAttestationError("attestation key size is invalid")
    if not report_bytes or not gold_bytes or not sidecar_bytes:
        raise ReportAttestationError("attested artifacts must not be empty")
    report = _parse_report(report_bytes)
    ordered_cases = tuple(sorted(cases, key=lambda item: item.case_id))
    if len({item.case_id for item in ordered_cases}) != len(ordered_cases):
        raise ReportAttestationError("attested case IDs are not unique")
    if [item.case_id for item in ordered_cases] != sorted(case.case_id for case in report.cases):
        raise ReportAttestationError("report and attestation case IDs do not match")
    provenance = report.provenance
    if provenance.git_sha is None or provenance.git_dirty is not False:
        raise ReportAttestationError("report does not have clean Git provenance")
    revision = _clean_git_sha(repository_root)
    if provenance.git_sha != revision:
        raise ReportAttestationError("report Git revision does not match the repository")
    if provenance.gold_artifact_sha256 != _sha256(gold_bytes):
        raise ReportAttestationError("report Gold artifact hash does not match")
    if provenance.sidecar_artifact_sha256 != _sha256(sidecar_bytes):
        raise ReportAttestationError("report sidecar artifact hash does not match")
    sources = _source_manifest(repository_root, source_paths)
    timestamp = created_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ReportAttestationError("attestation timestamp must be timezone-aware")
    if timestamp < provenance.evaluated_at:
        raise ReportAttestationError("attestation predates the evaluation report")
    unsigned = ReportAttestation(
        schema_version="rag-baseline-attestation-v1",
        algorithm="hmac-sha256",
        key_id=_sha256(key),
        created_at=timestamp,
        report_sha256=_sha256(report_bytes),
        report_size_bytes=len(report_bytes),
        gold_artifact_sha256=_sha256(gold_bytes),
        gold_size_bytes=len(gold_bytes),
        sidecar_artifact_sha256=_sha256(sidecar_bytes),
        sidecar_size_bytes=len(sidecar_bytes),
        case_count=len(ordered_cases),
        cases=ordered_cases,
        case_manifest_sha256=_manifest_sha256(ordered_cases),
        configuration_sha256=provenance.configuration_sha256,
        corpus_fingerprint_sha256=provenance.corpus_fingerprint_sha256,
        runtime_corpus_snapshot_sha256=provenance.runtime_corpus_snapshot_sha256,
        repository_git_sha=revision,
        repository_git_dirty=False,
        sources=sources,
        source_manifest_sha256=_manifest_sha256(sources),
        signature="0" * 64,
    )
    return unsigned.model_copy(update={"signature": _signature(unsigned, key)})


def attestation_bytes(attestation: ReportAttestation) -> bytes:
    return (
        json.dumps(
            attestation.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def create_private_artifact_attestation(
    *,
    artifact_bytes: bytes,
    artifact_type: str,
    key: bytes,
    repository_root: Path,
    source_paths: Sequence[str],
    created_at: datetime | None = None,
) -> PrivateArtifactAttestation:
    if not _MIN_KEY_BYTES <= len(key) <= _MAX_KEY_BYTES:
        raise ReportAttestationError("attestation key size is invalid")
    if not artifact_bytes or len(artifact_bytes) > 256 * 1024 * 1024:
        raise ReportAttestationError("private artifact size is invalid")
    revision = _clean_git_sha(repository_root)
    sources = _source_manifest(repository_root, source_paths)
    timestamp = created_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ReportAttestationError("attestation timestamp must be timezone-aware")
    try:
        unsigned = PrivateArtifactAttestation(
            schema_version="rag-private-artifact-attestation-v1",
            algorithm="hmac-sha256",
            artifact_type=artifact_type,
            key_id=_sha256(key),
            created_at=timestamp,
            artifact_sha256=_sha256(artifact_bytes),
            artifact_size_bytes=len(artifact_bytes),
            repository_git_sha=revision,
            repository_git_dirty=False,
            sources=sources,
            source_manifest_sha256=_manifest_sha256(sources),
            signature="0" * 64,
        )
    except ValidationError:
        raise ReportAttestationError("private artifact type is invalid") from None
    return unsigned.model_copy(update={"signature": _signature(unsigned, key)})


def private_artifact_attestation_bytes(attestation: PrivateArtifactAttestation) -> bytes:
    return (
        json.dumps(
            attestation.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def load_private_artifact_attestation(raw: bytes) -> PrivateArtifactAttestation:
    if not raw or len(raw) > _MAX_ATTESTATION_BYTES:
        raise ReportAttestationError("attestation size is invalid")
    try:
        json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        return PrivateArtifactAttestation.model_validate_json(raw, strict=True)
    except (_DuplicateKeyError, json.JSONDecodeError, ValidationError):
        raise ReportAttestationError("attestation is invalid") from None


def verify_private_artifact_attestation(
    attestation: PrivateArtifactAttestation,
    *,
    artifact_bytes: bytes,
    expected_artifact_type: str,
    key: bytes,
    repository_root: Path,
) -> None:
    if not _MIN_KEY_BYTES <= len(key) <= _MAX_KEY_BYTES:
        raise ReportAttestationError("attestation key size is invalid")
    if attestation.artifact_type != expected_artifact_type:
        raise ReportAttestationError("private artifact type does not match")
    if attestation.key_id != _sha256(key) or not hmac.compare_digest(
        attestation.signature,
        _signature(attestation, key),
    ):
        raise ReportAttestationError("attestation signature is invalid")
    if attestation.artifact_sha256 != _sha256(artifact_bytes) or attestation.artifact_size_bytes != len(
        artifact_bytes
    ):
        raise ReportAttestationError("private artifact bytes do not match")
    sources = _source_manifest_at_revision(
        repository_root,
        tuple(item.path for item in attestation.sources),
        attestation.repository_git_sha,
    )
    if sources != attestation.sources:
        raise ReportAttestationError("attested source files do not match")


def load_report_attestation(raw: bytes) -> ReportAttestation:
    if not raw or len(raw) > _MAX_ATTESTATION_BYTES:
        raise ReportAttestationError("attestation size is invalid")
    try:
        json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        return ReportAttestation.model_validate_json(raw, strict=True)
    except (_DuplicateKeyError, json.JSONDecodeError, ValidationError):
        raise ReportAttestationError("attestation is invalid") from None


def verify_report_attestation(
    attestation: ReportAttestation,
    *,
    report_bytes: bytes,
    gold_bytes: bytes,
    sidecar_bytes: bytes,
    expected_cases: Sequence[AttestedCase],
    key: bytes,
    repository_root: Path,
) -> None:
    if not _MIN_KEY_BYTES <= len(key) <= _MAX_KEY_BYTES:
        raise ReportAttestationError("attestation key size is invalid")
    if attestation.key_id != _sha256(key) or not hmac.compare_digest(
        attestation.signature,
        _signature(attestation, key),
    ):
        raise ReportAttestationError("attestation signature is invalid")
    report = _parse_report(report_bytes)
    expected = tuple(sorted(expected_cases, key=lambda item: item.case_id))
    checks = (
        attestation.report_sha256 == _sha256(report_bytes),
        attestation.report_size_bytes == len(report_bytes),
        attestation.gold_artifact_sha256 == _sha256(gold_bytes),
        attestation.gold_size_bytes == len(gold_bytes),
        attestation.sidecar_artifact_sha256 == _sha256(sidecar_bytes),
        attestation.sidecar_size_bytes == len(sidecar_bytes),
        attestation.cases == expected,
        attestation.case_manifest_sha256 == _manifest_sha256(expected),
        attestation.configuration_sha256 == report.provenance.configuration_sha256,
        attestation.corpus_fingerprint_sha256 == report.provenance.corpus_fingerprint_sha256,
        attestation.runtime_corpus_snapshot_sha256 == report.provenance.runtime_corpus_snapshot_sha256,
        attestation.repository_git_sha == report.provenance.git_sha,
        report.provenance.git_dirty is False,
        attestation.created_at >= report.provenance.evaluated_at,
    )
    if not all(checks):
        raise ReportAttestationError("attested artifacts or provenance do not match")
    sources = _source_manifest_at_revision(
        repository_root,
        tuple(item.path for item in attestation.sources),
        attestation.repository_git_sha,
    )
    if sources != attestation.sources:
        raise ReportAttestationError("attested source files do not match")


def atomic_write_attestation(path: Path, attestation: ReportAttestation) -> None:
    try:
        write_private_json_fresh(path.expanduser(), attestation_bytes(attestation))
    except (FileExistsError, PrivateArtifactError):
        raise ReportAttestationError("attestation output cannot be published safely") from None


def atomic_write_private_artifact_attestation(
    path: Path,
    attestation: PrivateArtifactAttestation,
) -> None:
    try:
        write_private_json_fresh(
            path.expanduser(),
            private_artifact_attestation_bytes(attestation),
        )
    except (FileExistsError, PrivateArtifactError):
        raise ReportAttestationError("attestation output cannot be published safely") from None


__all__ = [
    "DEFAULT_ATTESTED_SOURCES",
    "AttestedCase",
    "PrivateArtifactAttestation",
    "ReportAttestation",
    "ReportAttestationError",
    "atomic_write_attestation",
    "atomic_write_private_artifact_attestation",
    "attestation_bytes",
    "build_case_attestations",
    "create_private_artifact_attestation",
    "create_report_attestation",
    "load_hmac_key",
    "load_private_artifact_attestation",
    "load_report_attestation",
    "private_artifact_attestation_bytes",
    "verify_private_artifact_attestation",
    "verify_report_attestation",
]
