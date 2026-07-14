"""Strict raw evidence for offline model qualification.

The release gate consumes compact aggregates.  This module owns the private,
auditable observations from which those aggregates are derived and verifies
that no reported counter or latency was supplied independently of the raw run.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import stat
from collections import Counter
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from rag_app.eval.private_artifacts import (
    PrivateArtifactError,
    read_private_json,
    write_private_json_fresh,
)

ModelRole = Literal[
    "llm",
    "embedding",
    "reranker",
    "visual_embedding",
    "visual_reranker",
]
Language = Literal["en", "ru", "zh"]
LongContextOutcome = Literal[
    "completed",
    "overflow_error",
    "oom_error",
    "truncation_error",
    "other_error",
]
LoadOutcome = Literal["completed", "error", "oom_error"]
QualificationCategory = Literal["semantic", "safety", "standards"]
JudgeVerdict = Literal["pass", "fail", "error"]
LoadTarget = Literal["baseline", "candidate"]
RollbackEventKind = Literal[
    "rollback_started",
    "config_restored",
    "code_restored",
    "services_restarted",
    "verification_started",
    "rollback_completed",
]
RollbackProbeKind = Literal[
    "health",
    "root",
    "auth_enabled",
    "anonymous_protected",
    "model_endpoint",
]

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_GIT_SHA_PATTERN = r"^[0-9a-f]{40,64}$"
_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_REASON_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
_MAX_LICENSE_BYTES = 4 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_LOCAL_DIRECTORY_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
_LOCAL_LICENSE_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK

QUALIFICATION_ATTESTED_SOURCES = (
    "deploy/rag-eval/semantic-judge-v1.txt",
    "scripts/run_rag_model_qualification.py",
    "src/rag_app/eval/private_artifacts.py",
    "src/rag_app/eval/qualification_evidence.py",
    "src/rag_app/eval/report_attestation.py",
)


class QualificationEvidenceError(RuntimeError):
    """Sanitized raw-evidence validation or private-file failure."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class QualificationProvenance(_StrictModel):
    generated_at: AwareDatetime
    producer_git_sha: str = Field(pattern=_GIT_SHA_PATTERN)
    git_dirty: Literal[False]
    candidate_role: ModelRole
    candidate_model: str = Field(min_length=1, max_length=256)
    candidate_declared_revision: str = Field(min_length=1, max_length=512)
    candidate_weight_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    baseline_model: str = Field(min_length=1, max_length=256)
    baseline_weight_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    baseline_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    rag_configuration_sha256: str = Field(pattern=_SHA256_PATTERN)
    baseline_report_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_report_sha256: str = Field(pattern=_SHA256_PATTERN)
    gold_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    sidecar_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    corpus_fingerprint_sha256: str = Field(pattern=_SHA256_PATTERN)
    runtime_corpus_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    judge_model: str = Field(min_length=1, max_length=256)
    judge_declared_revision: str = Field(min_length=1, max_length=512)
    judge_weight_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    judge_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    judge_prompt_sha256: str = Field(pattern=_SHA256_PATTERN)
    reference_git_sha: str = Field(pattern=_GIT_SHA_PATTERN)


class LocalLicenseEvidence(_StrictModel):
    role: ModelRole
    model: str = Field(min_length=1, max_length=256)
    weight_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    spdx_license: str = Field(min_length=1, max_length=64)
    source_url: str = Field(min_length=1, max_length=2048)
    local_relative_path: str = Field(min_length=1, max_length=1024)
    license_bytes_base64: str = Field(min_length=1, max_length=6 * 1024 * 1024)
    license_byte_count: int = Field(ge=1, le=_MAX_LICENSE_BYTES)
    license_text_sha256: str = Field(pattern=_SHA256_PATTERN)
    commercial_on_prem_allowed: Literal[True]

    @model_validator(mode="after")
    def validate_exact_bytes(self) -> Self:
        parsed = urlsplit(self.source_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("license source must be a credential-free HTTPS URL")
        relative = PurePosixPath(self.local_relative_path)
        if (
            relative.is_absolute()
            or relative.as_posix() != self.local_relative_path
            or ".." in relative.parts
            or "." in relative.parts
        ):
            raise ValueError("license path must be a normalized relative POSIX path")
        try:
            raw = base64.b64decode(self.license_bytes_base64, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("license bytes are not canonical base64") from None
        if base64.b64encode(raw).decode("ascii") != self.license_bytes_base64:
            raise ValueError("license bytes are not canonical base64")
        if len(raw) != self.license_byte_count:
            raise ValueError("license byte count mismatch")
        if hashlib.sha256(raw).hexdigest() != self.license_text_sha256:
            raise ValueError("license byte hash mismatch")
        return self


class LongContextObservation(_StrictModel):
    case_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    language: Language
    input_tokens: int = Field(ge=1)
    model_context_tokens: int = Field(ge=1)
    outcome: LongContextOutcome
    duration_ms: float = Field(ge=0)
    output_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    error_code: str | None = Field(default=None, pattern=_REASON_PATTERN)

    @model_validator(mode="after")
    def validate_outcome_payload(self) -> Self:
        if self.outcome == "completed":
            if self.output_sha256 is None or self.error_code is not None:
                raise ValueError("completed long-context observation requires only output hash")
        elif self.output_sha256 is not None or self.error_code is None:
            raise ValueError("failed long-context observation requires only error code")
        return self


class LoadAttemptObservation(_StrictModel):
    started_offset_ms: float = Field(ge=0)
    finished_offset_ms: float = Field(gt=0)
    outcome: LoadOutcome
    response_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    error_code: str | None = Field(default=None, pattern=_REASON_PATTERN)

    @model_validator(mode="after")
    def validate_attempt(self) -> Self:
        if self.finished_offset_ms <= self.started_offset_ms:
            raise ValueError("load request must finish after it starts")
        if self.outcome == "completed":
            if self.response_sha256 is None or self.error_code is not None:
                raise ValueError("completed load request requires only response hash")
        elif self.response_sha256 is not None or self.error_code is None:
            raise ValueError("failed load request requires only error code")
        return self

    @property
    def latency_ms(self) -> float:
        return self.finished_offset_ms - self.started_offset_ms


class PairedLoadRequestObservation(_StrictModel):
    request_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    case_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    baseline: LoadAttemptObservation
    candidate: LoadAttemptObservation


class LoadRuntimeEvent(_StrictModel):
    target: LoadTarget
    kind: Literal["restart"]
    offset_ms: float = Field(ge=0)
    evidence_sha256: str = Field(pattern=_SHA256_PATTERN)


class LoadRunObservations(_StrictModel):
    concurrency: int = Field(ge=1)
    baseline_duration_ms: float = Field(gt=0)
    candidate_duration_ms: float = Field(gt=0)
    requests: tuple[PairedLoadRequestObservation, ...] = Field(min_length=1, max_length=100_000)
    runtime_events: tuple[LoadRuntimeEvent, ...] = Field(default=(), max_length=10_000)

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        request_ids = [item.request_id for item in self.requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("load request IDs must be unique")
        for item in self.requests:
            if item.baseline.finished_offset_ms > self.baseline_duration_ms:
                raise ValueError("baseline request exceeds run duration")
            if item.candidate.finished_offset_ms > self.candidate_duration_ms:
                raise ValueError("candidate request exceeds run duration")
        for event in self.runtime_events:
            duration = self.baseline_duration_ms if event.target == "baseline" else self.candidate_duration_ms
            if event.offset_ms > duration:
                raise ValueError("runtime event exceeds run duration")
        peaks = []
        for target in ("baseline", "candidate"):
            timeline = sorted(
                (
                    (getattr(item, target).started_offset_ms, 1),
                    (getattr(item, target).finished_offset_ms, -1),
                )
                for item in self.requests
            )
            events = sorted(
                (event for pair in timeline for event in pair),
                key=lambda item: (item[0], item[1]),
            )
            active = 0
            peak = 0
            for _, delta in events:
                active += delta
                peak = max(peak, active)
            peaks.append(peak)
        if self.concurrency != min(peaks):
            raise ValueError("declared load concurrency does not match request overlap")
        return self


class JudgeCaseObservation(_StrictModel):
    verdict: JudgeVerdict
    response_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    error_code: str | None = Field(default=None, pattern=_REASON_PATTERN)
    reason_codes: tuple[str, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_judgment(self) -> Self:
        if len(self.reason_codes) != len(set(self.reason_codes)) or any(
            not code or len(code) > 64 or not code.replace("_", "a").isalnum() for code in self.reason_codes
        ):
            raise ValueError("judge reason codes must be unique sanitized identifiers")
        if self.verdict == "error":
            if self.error_code is None or self.response_sha256 is not None:
                raise ValueError("judge error requires only error code")
        elif self.error_code is not None or self.response_sha256 is None:
            raise ValueError("judge verdict requires only response hash")
        if self.verdict == "pass" and self.reason_codes:
            raise ValueError("passing judgment cannot contain reason codes")
        return self


class PairedSemanticSafetyObservation(_StrictModel):
    case_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    gold_case_sha256: str = Field(pattern=_SHA256_PATTERN)
    categories: tuple[QualificationCategory, ...] = Field(min_length=1, max_length=3)
    baseline_output_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_output_sha256: str = Field(pattern=_SHA256_PATTERN)
    baseline: JudgeCaseObservation
    candidate: JudgeCaseObservation

    @model_validator(mode="after")
    def validate_categories(self) -> Self:
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("qualification categories must be unique")
        if "semantic" not in self.categories:
            raise ValueError("every paired judgment must belong to semantic qualification")
        return self


class RollbackTraceEvent(_StrictModel):
    sequence: int = Field(ge=0)
    kind: RollbackEventKind
    observed_at: AwareDatetime
    success: bool
    evidence_sha256: str = Field(pattern=_SHA256_PATTERN)


class RollbackProbeObservation(_StrictModel):
    kind: RollbackProbeKind
    target: str = Field(min_length=1, max_length=256)
    passed: bool
    status_code: int | None = Field(default=None, ge=100, le=599)
    response_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        requires_status = self.kind in {
            "health",
            "root",
            "anonymous_protected",
            "model_endpoint",
        }
        if requires_status != (self.status_code is not None):
            raise ValueError("rollback probe status presence is inconsistent with probe kind")
        return self


class RollbackSmokeObservation(_StrictModel):
    case_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    passed: bool
    result_sha256: str = Field(pattern=_SHA256_PATTERN)


class RestoredModelWeightManifest(_StrictModel):
    role: ModelRole
    weight_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)


class RollbackRawEvidence(_StrictModel):
    reference_report_sha256: str = Field(pattern=_SHA256_PATTERN)
    restored_git_sha: str = Field(pattern=_GIT_SHA_PATTERN)
    restored_model_weight_manifests: tuple[RestoredModelWeightManifest, ...] = Field(
        min_length=1, max_length=5
    )
    restored_configuration_sha256: str = Field(pattern=_SHA256_PATTERN)
    restored_rag_configuration_sha256: str = Field(pattern=_SHA256_PATTERN)
    restored_runtime_corpus_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    trace: tuple[RollbackTraceEvent, ...] = Field(min_length=2, max_length=10_000)
    probes: tuple[RollbackProbeObservation, ...] = Field(min_length=5, max_length=10_000)
    smoke: tuple[RollbackSmokeObservation, ...] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_trace_and_checks(self) -> Self:
        roles = [item.role for item in self.restored_model_weight_manifests]
        if len(roles) != len(set(roles)):
            raise ValueError("restored model weight roles must be unique")
        expected_trace = [
            "rollback_started",
            "config_restored",
            "code_restored",
            "services_restarted",
            "verification_started",
            "rollback_completed",
        ]
        if [item.kind for item in self.trace] != expected_trace:
            raise ValueError("rollback trace must contain the complete ordered lifecycle")
        if [item.sequence for item in self.trace] != list(range(len(self.trace))):
            raise ValueError("rollback trace sequence must be contiguous")
        if any(
            right.observed_at < left.observed_at
            for left, right in zip(self.trace, self.trace[1:], strict=False)
        ):
            raise ValueError("rollback trace timestamps must be monotonic")
        singleton_kinds: tuple[RollbackProbeKind, ...] = (
            "health",
            "root",
            "auth_enabled",
            "anonymous_protected",
        )
        counts = Counter(item.kind for item in self.probes)
        if any(counts[kind] != 1 for kind in singleton_kinds) or counts["model_endpoint"] < 1:
            raise ValueError("rollback probes require singleton core checks and model endpoints")
        probe_keys = [(item.kind, item.target) for item in self.probes]
        if len(probe_keys) != len(set(probe_keys)):
            raise ValueError("rollback probe identities must be unique")
        smoke_ids = [item.case_id for item in self.smoke]
        if len(smoke_ids) != len(set(smoke_ids)):
            raise ValueError("rollback smoke case IDs must be unique")
        return self


class LongContextAggregate(_StrictModel):
    case_count: int = Field(ge=1)
    completed_count: int = Field(ge=0)
    language_counts: dict[Language, int]
    minimum_input_tokens: int = Field(ge=1)
    model_context_tokens: int = Field(ge=1)
    overflow_errors: int = Field(ge=0)
    oom_errors: int = Field(ge=0)
    truncation_errors: int = Field(ge=0)
    other_errors: int = Field(ge=0)


class LoadAggregate(_StrictModel):
    concurrency: int = Field(ge=1)
    request_count: int = Field(ge=1)
    completed_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    restart_count: int = Field(ge=0)
    oom_count: int = Field(ge=0)
    baseline_p95_ms: float = Field(gt=0)
    candidate_p95_ms: float = Field(gt=0)
    baseline_throughput_rps: float = Field(gt=0)
    candidate_throughput_rps: float = Field(gt=0)


class SemanticSafetyAggregate(_StrictModel):
    judge_model: str = Field(min_length=1, max_length=256)
    judge_weight_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    judge_prompt_sha256: str = Field(pattern=_SHA256_PATTERN)
    case_count: int = Field(ge=1)
    baseline_semantic_passed: int = Field(ge=0)
    candidate_semantic_passed: int = Field(ge=0)
    safety_case_count: int = Field(ge=1)
    baseline_safety_passed: int = Field(ge=0)
    candidate_safety_passed: int = Field(ge=0)
    standards_case_count: int = Field(ge=1)
    baseline_standards_passed: int = Field(ge=0)
    candidate_standards_passed: int = Field(ge=0)
    judge_error_count: int = Field(ge=0)


class RollbackAggregate(_StrictModel):
    reference_report_sha256: str = Field(pattern=_SHA256_PATTERN)
    restored_git_sha: str = Field(pattern=_GIT_SHA_PATTERN)
    restored_model_weight_manifests: tuple[RestoredModelWeightManifest, ...] = Field(
        min_length=1, max_length=5
    )
    restored_configuration_sha256: str = Field(pattern=_SHA256_PATTERN)
    restored_rag_configuration_sha256: str = Field(pattern=_SHA256_PATTERN)
    restored_runtime_corpus_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    duration_seconds: float = Field(ge=0)
    smoke_case_count: int = Field(ge=1)
    smoke_passed_count: int = Field(ge=0)
    health_ok: bool
    root_ok: bool
    auth_enabled: bool
    anonymous_protected_status: int = Field(ge=100, le=599)
    model_endpoints_ok: bool


class QualificationAggregates(_StrictModel):
    long_context: LongContextAggregate
    load: LoadAggregate
    semantic_safety: SemanticSafetyAggregate
    rollback: RollbackAggregate


class RawQualificationEvidence(_StrictModel):
    schema_version: Literal["rag-model-qualification-raw-v1"] = "rag-model-qualification-raw-v1"
    provenance: QualificationProvenance
    license: LocalLicenseEvidence
    long_context_observations: tuple[LongContextObservation, ...] = Field(min_length=1, max_length=10_000)
    load_observations: LoadRunObservations
    semantic_safety_observations: tuple[PairedSemanticSafetyObservation, ...] = Field(
        min_length=1, max_length=10_000
    )
    rollback_trace: RollbackRawEvidence
    aggregates: QualificationAggregates

    @model_validator(mode="after")
    def validate_bindings(self) -> Self:
        provenance = self.provenance
        license_evidence = self.license
        if (
            license_evidence.role != provenance.candidate_role
            or license_evidence.model != provenance.candidate_model
            or license_evidence.weight_manifest_sha256 != provenance.candidate_weight_manifest_sha256
        ):
            raise ValueError("local license is not bound to the candidate model")
        if self.rollback_trace.reference_report_sha256 != provenance.baseline_report_sha256:
            raise ValueError("rollback trace is not bound to the baseline report")
        if self.rollback_trace.restored_git_sha != provenance.reference_git_sha:
            raise ValueError("rollback trace is not bound to reference git")
        restored_weights = {
            item.role: item.weight_manifest_sha256
            for item in self.rollback_trace.restored_model_weight_manifests
        }
        if restored_weights.get(provenance.candidate_role) != (provenance.baseline_weight_manifest_sha256):
            raise ValueError("rollback trace is not bound to the baseline model weight")
        if self.rollback_trace.restored_configuration_sha256 != (provenance.baseline_config_sha256):
            raise ValueError("rollback trace is not bound to the baseline configuration")
        if self.rollback_trace.restored_rag_configuration_sha256 != (provenance.rag_configuration_sha256):
            raise ValueError("rollback trace is not bound to the RAG configuration")
        if self.rollback_trace.restored_runtime_corpus_snapshot_sha256 != (
            provenance.runtime_corpus_snapshot_sha256
        ):
            raise ValueError("rollback trace is not bound to the runtime corpus snapshot")
        long_ids = [item.case_id for item in self.long_context_observations]
        paired_ids = [item.case_id for item in self.semantic_safety_observations]
        if len(long_ids) != len(set(long_ids)) or len(paired_ids) != len(set(paired_ids)):
            raise ValueError("qualification observation case IDs must be unique per suite")
        return self


def _license_relative_path(path: Path, model_root: Path) -> PurePosixPath:
    source = path.expanduser().absolute()
    root = model_root.expanduser().absolute()
    try:
        relative = source.relative_to(root)
    except ValueError:
        raise QualificationEvidenceError("local license must be below the model root") from None
    normalized = PurePosixPath(relative.as_posix())
    if (
        not normalized.parts
        or normalized.is_absolute()
        or any(part in {"", ".", ".."} for part in normalized.parts)
    ):
        raise QualificationEvidenceError("local license path is not normalized")
    return normalized


def _read_local_license_bytes(
    path: Path,
    *,
    model_root: Path,
    expected_uid: int,
) -> tuple[PurePosixPath, bytes]:
    """Read a model LICENSE through a pinned no-follow descriptor chain."""

    relative = _license_relative_path(path, model_root)
    try:
        directory_fd = os.open(model_root.expanduser().absolute(), _LOCAL_DIRECTORY_FLAGS)
    except OSError as error:
        raise QualificationEvidenceError(f"local model root is unsafe ({type(error).__name__})") from None
    descriptor = -1
    try:
        root_metadata = os.fstat(directory_fd)
        if root_metadata.st_uid != expected_uid:
            raise QualificationEvidenceError("local model root has an unexpected owner")
        for component in relative.parts[:-1]:
            next_fd = os.open(component, _LOCAL_DIRECTORY_FLAGS, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
            metadata = os.fstat(directory_fd)
            if metadata.st_uid != expected_uid:
                raise QualificationEvidenceError("local license parent has an unexpected owner")
        descriptor = os.open(
            relative.parts[-1],
            _LOCAL_LICENSE_FLAGS,
            dir_fd=directory_fd,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != expected_uid:
            raise QualificationEvidenceError("local license must be a regular file owned by the model owner")
        if before.st_size < 1 or before.st_size > _MAX_LICENSE_BYTES:
            raise QualificationEvidenceError("local license size is outside the supported range")
        content = bytearray()
        while True:
            block = os.read(
                descriptor,
                min(1024 * 1024, _MAX_LICENSE_BYTES + 1 - len(content)),
            )
            if not block:
                break
            content.extend(block)
            if len(content) > _MAX_LICENSE_BYTES:
                raise QualificationEvidenceError("local license size is outside the supported range")
        after = os.fstat(descriptor)
        fingerprint_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        fingerprint_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if len(content) != before.st_size or fingerprint_before != fingerprint_after:
            raise QualificationEvidenceError("local license changed while being read")
        return relative, bytes(content)
    except QualificationEvidenceError:
        raise
    except OSError as error:
        raise QualificationEvidenceError(
            f"local license cannot be opened safely ({type(error).__name__})"
        ) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_fd)


def capture_local_license(
    path: Path,
    *,
    model_root: Path,
    role: ModelRole,
    model: str,
    weight_manifest_sha256: str,
    spdx_license: str,
    source_url: str,
    commercial_on_prem_allowed: Literal[True],
) -> LocalLicenseEvidence:
    """Capture exact bytes from a non-symlink LICENSE below the local model root."""

    relative, raw = _read_local_license_bytes(
        path,
        model_root=model_root,
        expected_uid=os.geteuid(),
    )
    return LocalLicenseEvidence(
        role=role,
        model=model,
        weight_manifest_sha256=weight_manifest_sha256,
        spdx_license=spdx_license,
        source_url=source_url,
        local_relative_path=relative.as_posix(),
        license_bytes_base64=base64.b64encode(raw).decode("ascii"),
        license_byte_count=len(raw),
        license_text_sha256=hashlib.sha256(raw).hexdigest(),
        commercial_on_prem_allowed=commercial_on_prem_allowed,
    )


def _nearest_rank_p95(values: Sequence[float]) -> float:
    if not values:
        raise QualificationEvidenceError("load run has no completed latency observations")
    ordered = sorted(values)
    return ordered[max(math.ceil(0.95 * len(ordered)) - 1, 0)]


def aggregate_long_context(
    observations: Sequence[LongContextObservation],
) -> LongContextAggregate:
    if not observations:
        raise QualificationEvidenceError("long-context observations are empty")
    context_windows = {item.model_context_tokens for item in observations}
    if len(context_windows) != 1:
        raise QualificationEvidenceError("long-context observations mix context windows")
    outcomes = Counter(item.outcome for item in observations)
    languages = Counter(item.language for item in observations)
    return LongContextAggregate(
        case_count=len(observations),
        completed_count=outcomes["completed"],
        language_counts={language: languages[language] for language in ("en", "ru", "zh")},
        minimum_input_tokens=min(item.input_tokens for item in observations),
        model_context_tokens=next(iter(context_windows)),
        overflow_errors=outcomes["overflow_error"],
        oom_errors=outcomes["oom_error"],
        truncation_errors=outcomes["truncation_error"],
        other_errors=outcomes["other_error"],
    )


def aggregate_load(observations: LoadRunObservations) -> LoadAggregate:
    baseline_completed = [
        item.baseline for item in observations.requests if item.baseline.outcome == "completed"
    ]
    candidate_completed = [
        item.candidate for item in observations.requests if item.candidate.outcome == "completed"
    ]
    baseline_p95 = _nearest_rank_p95([item.latency_ms for item in baseline_completed])
    candidate_p95 = _nearest_rank_p95([item.latency_ms for item in candidate_completed])
    candidate_errors = len(observations.requests) - len(candidate_completed)
    return LoadAggregate(
        concurrency=observations.concurrency,
        request_count=len(observations.requests),
        completed_count=len(candidate_completed),
        error_count=candidate_errors,
        restart_count=sum(
            item.target == "candidate" and item.kind == "restart" for item in observations.runtime_events
        ),
        oom_count=sum(item.candidate.outcome == "oom_error" for item in observations.requests),
        baseline_p95_ms=baseline_p95,
        candidate_p95_ms=candidate_p95,
        baseline_throughput_rps=len(baseline_completed) / (observations.baseline_duration_ms / 1000),
        candidate_throughput_rps=len(candidate_completed) / (observations.candidate_duration_ms / 1000),
    )


def aggregate_semantic_safety(
    observations: Sequence[PairedSemanticSafetyObservation],
    *,
    provenance: QualificationProvenance,
) -> SemanticSafetyAggregate:
    if not observations:
        raise QualificationEvidenceError("semantic/safety observations are empty")
    safety = [item for item in observations if "safety" in item.categories]
    standards = [item for item in observations if "standards" in item.categories]
    if not safety or not standards:
        raise QualificationEvidenceError("semantic/safety observations lack required slices")

    def passed(items: Sequence[PairedSemanticSafetyObservation], side: str) -> int:
        return sum(getattr(item, side).verdict == "pass" for item in items)

    return SemanticSafetyAggregate(
        judge_model=provenance.judge_model,
        judge_weight_manifest_sha256=provenance.judge_weight_manifest_sha256,
        judge_prompt_sha256=provenance.judge_prompt_sha256,
        case_count=len(observations),
        baseline_semantic_passed=passed(observations, "baseline"),
        candidate_semantic_passed=passed(observations, "candidate"),
        safety_case_count=len(safety),
        baseline_safety_passed=passed(safety, "baseline"),
        candidate_safety_passed=passed(safety, "candidate"),
        standards_case_count=len(standards),
        baseline_standards_passed=passed(standards, "baseline"),
        candidate_standards_passed=passed(standards, "candidate"),
        judge_error_count=sum(
            judgment.verdict == "error"
            for item in observations
            for judgment in (item.baseline, item.candidate)
        ),
    )


def aggregate_rollback(raw: RollbackRawEvidence) -> RollbackAggregate:
    probes_by_kind: dict[RollbackProbeKind, list[RollbackProbeObservation]] = {}
    for probe in raw.probes:
        probes_by_kind.setdefault(probe.kind, []).append(probe)
    started = raw.trace[0].observed_at
    completed = raw.trace[-1].observed_at
    duration = (completed - started).total_seconds()
    if duration < 0:
        raise QualificationEvidenceError("rollback duration is negative")
    anonymous = probes_by_kind["anonymous_protected"][0]
    if anonymous.status_code is None:
        raise QualificationEvidenceError("anonymous rollback probe lacks status")
    trace_ok = all(item.success for item in raw.trace)
    return RollbackAggregate(
        reference_report_sha256=raw.reference_report_sha256,
        restored_git_sha=raw.restored_git_sha,
        restored_model_weight_manifests=raw.restored_model_weight_manifests,
        restored_configuration_sha256=raw.restored_configuration_sha256,
        restored_rag_configuration_sha256=raw.restored_rag_configuration_sha256,
        restored_runtime_corpus_snapshot_sha256=(raw.restored_runtime_corpus_snapshot_sha256),
        duration_seconds=duration,
        smoke_case_count=len(raw.smoke),
        smoke_passed_count=sum(item.passed for item in raw.smoke),
        health_ok=trace_ok and probes_by_kind["health"][0].passed,
        root_ok=probes_by_kind["root"][0].passed,
        auth_enabled=probes_by_kind["auth_enabled"][0].passed,
        anonymous_protected_status=anonymous.status_code,
        model_endpoints_ok=all(item.passed for item in probes_by_kind["model_endpoint"]),
    )


def recompute_qualification_aggregates(
    *,
    provenance: QualificationProvenance,
    long_context_observations: Sequence[LongContextObservation],
    load_observations: LoadRunObservations,
    semantic_safety_observations: Sequence[PairedSemanticSafetyObservation],
    rollback_trace: RollbackRawEvidence,
) -> QualificationAggregates:
    return QualificationAggregates(
        long_context=aggregate_long_context(long_context_observations),
        load=aggregate_load(load_observations),
        semantic_safety=aggregate_semantic_safety(semantic_safety_observations, provenance=provenance),
        rollback=aggregate_rollback(rollback_trace),
    )


def build_raw_qualification_evidence(
    *,
    provenance: QualificationProvenance,
    license: LocalLicenseEvidence,
    long_context_observations: Sequence[LongContextObservation],
    load_observations: LoadRunObservations,
    semantic_safety_observations: Sequence[PairedSemanticSafetyObservation],
    rollback_trace: RollbackRawEvidence,
) -> RawQualificationEvidence:
    """Build an evidence artifact whose aggregates are derived only from raw rows."""

    aggregates = recompute_qualification_aggregates(
        provenance=provenance,
        long_context_observations=long_context_observations,
        load_observations=load_observations,
        semantic_safety_observations=semantic_safety_observations,
        rollback_trace=rollback_trace,
    )
    return RawQualificationEvidence(
        provenance=provenance,
        license=license,
        long_context_observations=tuple(long_context_observations),
        load_observations=load_observations,
        semantic_safety_observations=tuple(semantic_safety_observations),
        rollback_trace=rollback_trace,
        aggregates=aggregates,
    )


def verify_raw_qualification_evidence(
    evidence: RawQualificationEvidence,
) -> QualificationAggregates:
    """Recompute every aggregate and fail closed on any stored mismatch."""

    expected = recompute_qualification_aggregates(
        provenance=evidence.provenance,
        long_context_observations=evidence.long_context_observations,
        load_observations=evidence.load_observations,
        semantic_safety_observations=evidence.semantic_safety_observations,
        rollback_trace=evidence.rollback_trace,
    )
    if expected != evidence.aggregates:
        raise QualificationEvidenceError("stored qualification aggregates do not match raw evidence")
    return expected


def qualification_evidence_json_schema() -> dict[str, Any]:
    return RawQualificationEvidence.model_json_schema()


def canonical_evidence_bytes(evidence: RawQualificationEvidence) -> bytes:
    verify_raw_qualification_evidence(evidence)
    return (
        json.dumps(
            evidence.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def qualification_evidence_sha256(evidence: RawQualificationEvidence) -> str:
    return hashlib.sha256(canonical_evidence_bytes(evidence)).hexdigest()


def _private_path(path: Path, repository_root: Path) -> Path:
    source = path.expanduser()
    root = repository_root.expanduser()
    if not source.is_absolute():
        source = Path.cwd() / source
    if not root.is_absolute():
        root = Path.cwd() / root
    try:
        relative = source.relative_to(root)
    except ValueError:
        return source
    if not relative.parts or relative.parts[0] != ".private":
        raise QualificationEvidenceError("private artifact path violates repository policy")
    return source


def write_private_qualification_evidence(
    path: Path,
    evidence: RawQualificationEvidence,
    *,
    repository_root: Path,
) -> str:
    """Atomically publish one fresh owner-only JSON artifact and return its SHA-256."""

    destination = _private_path(path, repository_root)
    content = canonical_evidence_bytes(evidence)
    try:
        written = write_private_json_fresh(
            destination,
            content,
            max_bytes=_MAX_ARTIFACT_BYTES,
        )
    except FileExistsError:
        raise QualificationEvidenceError("private qualification output already exists") from None
    except PrivateArtifactError as error:
        raise QualificationEvidenceError(
            f"private qualification output is unsafe ({type(error).__name__})"
        ) from None
    return written.sha256


def load_private_qualification_evidence(
    path: Path,
    *,
    repository_root: Path,
) -> RawQualificationEvidence:
    """Load a private artifact, strictly parse it and recompute all aggregates."""

    source = _private_path(path, repository_root)
    try:
        artifact = read_private_json(
            source,
            parser=lambda raw: RawQualificationEvidence.model_validate_json(raw, strict=True),
            max_bytes=_MAX_ARTIFACT_BYTES,
        )
        evidence = artifact.value
        verify_raw_qualification_evidence(evidence)
    except QualificationEvidenceError:
        raise
    except PrivateArtifactError as error:
        raise QualificationEvidenceError(
            f"private qualification input is invalid ({type(error).__name__})"
        ) from None
    return evidence


__all__ = [
    "JudgeCaseObservation",
    "LoadAggregate",
    "LoadAttemptObservation",
    "LoadRunObservations",
    "LoadRuntimeEvent",
    "LocalLicenseEvidence",
    "LongContextAggregate",
    "LongContextObservation",
    "PairedLoadRequestObservation",
    "PairedSemanticSafetyObservation",
    "QUALIFICATION_ATTESTED_SOURCES",
    "QualificationAggregates",
    "QualificationEvidenceError",
    "QualificationProvenance",
    "RawQualificationEvidence",
    "RestoredModelWeightManifest",
    "RollbackAggregate",
    "RollbackProbeObservation",
    "RollbackRawEvidence",
    "RollbackSmokeObservation",
    "RollbackTraceEvent",
    "SemanticSafetyAggregate",
    "aggregate_load",
    "aggregate_long_context",
    "aggregate_rollback",
    "aggregate_semantic_safety",
    "build_raw_qualification_evidence",
    "canonical_evidence_bytes",
    "capture_local_license",
    "load_private_qualification_evidence",
    "qualification_evidence_json_schema",
    "qualification_evidence_sha256",
    "recompute_qualification_aggregates",
    "verify_raw_qualification_evidence",
    "write_private_qualification_evidence",
]
