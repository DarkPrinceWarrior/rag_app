"""Strict private sidecar contract shared by generation and offline evaluation."""

from __future__ import annotations

import json
import stat
import uuid
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rag_app.eval.gold_set import GoldRecord, gold_record_case_sha256, text_sha256
from rag_app.eval.rag_metrics import quantity_unit_metrics

MAX_SIDECAR_RECORDS = 500
MAX_SIDECAR_LINE_BYTES = 256 * 1024

_CASE_ID_PATTERN = r"^ragq-[a-z0-9][a-z0-9._-]{7,63}$"
_SCOPE_ID_PATTERN = r"^scope-sha256:[0-9a-f]{64}$"
_DOCUMENT_REF_PATTERN = r"^doc-sha256:[0-9a-f]{64}$"
_EVIDENCE_ID_PATTERN = (
    r"^ev-sha256:[0-9a-f]{64}:p[1-9][0-9]*:"
    r"(?:text|table|formula|figure|scan):[0-9a-f]{64}$"
)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )


class SidecarDocument(_StrictModel):
    document_id: uuid.UUID
    document_ref: str = Field(pattern=_DOCUMENT_REF_PATTERN)
    source_lang: Literal["ru", "en", "zh"]


class SidecarEvidence(_StrictModel):
    evidence_id: str = Field(pattern=_EVIDENCE_ID_PATTERN)
    document_id: uuid.UUID
    document_ref: str = Field(pattern=_DOCUMENT_REF_PATTERN)
    chunk_id: uuid.UUID
    chunk_index: int = Field(strict=True, ge=0)
    kind: str = Field(min_length=1, max_length=32)
    heading_path: str = Field(max_length=2_000)
    page: int = Field(strict=True, ge=1, le=100_000)
    page_start: int | None = Field(ge=0)
    page_end: int | None = Field(ge=0)
    text_sha256: str = Field(pattern=_SHA256_PATTERN)
    content_sha256: str = Field(pattern=_SHA256_PATTERN)
    exact_quote: str = Field(min_length=1, max_length=4_000)
    retrieval_score: float | None = None

    @model_validator(mode="after")
    def validate_hashes_and_pages(self) -> Self:
        if text_sha256(self.exact_quote) != self.content_sha256:
            raise ValueError("content hash does not match exact quote")
        if self.page_start is not None and self.page_end is not None:
            if self.page_end < self.page_start:
                raise ValueError("page_end must not precede page_start")
        return self


class RetrievalProbe(_StrictModel):
    document_id: uuid.UUID
    document_ref: str = Field(pattern=_DOCUMENT_REF_PATTERN)
    chunk_id: uuid.UUID
    page: int = Field(strict=True, ge=1, le=100_000)
    page_start: int | None = Field(ge=0)
    page_end: int | None = Field(ge=0)
    content_sha256: str = Field(pattern=_SHA256_PATTERN)
    retrieval_score: float


class QuantitySpec(_StrictModel):
    value: str = Field(min_length=1, max_length=64)
    unit: str = Field(max_length=32)


class SidecarQuantities(_StrictModel):
    expected: tuple[QuantitySpec, ...] = Field(max_length=128)
    supported: tuple[QuantitySpec, ...] = Field(max_length=256)


class SidecarClassification(_StrictModel):
    content_types: tuple[Literal["text", "table", "formula", "figure", "scan"], ...]
    challenge_tags: tuple[Literal["numbers", "units", "standards", "prompt_injection", "leakage"], ...]
    has_numbers: bool
    has_units: bool
    has_standards: bool


class SidecarGeneration(_StrictModel):
    model: str = Field(min_length=1, max_length=256)
    seed: int = Field(strict=True)


class PrivateSidecarRecord(_StrictModel):
    schema_version: Literal["private-rag-generator-v1"]
    case_id: str = Field(pattern=_CASE_ID_PATTERN)
    gold_case_sha256: str = Field(pattern=_SHA256_PATTERN)
    scope_id: str = Field(pattern=_SCOPE_ID_PATTERN)
    stratum: Literal["single_hop", "multi_hop", "cross_document", "no_answer"]
    language: Literal["ru", "en", "zh"]
    source_documents: tuple[SidecarDocument, ...] = Field(min_length=1, max_length=64)
    classification: SidecarClassification
    generation: SidecarGeneration
    exact_evidence: tuple[SidecarEvidence, ...] = Field(max_length=16)
    retrieval_probe: tuple[RetrievalProbe, ...] = Field(default=(), max_length=32)
    quantities: SidecarQuantities
    validation: dict[str, bool]

    @model_validator(mode="after")
    def validate_unique_private_refs(self) -> Self:
        for values, label in (
            ([item.document_id for item in self.source_documents], "document IDs"),
            ([item.document_ref for item in self.source_documents], "document refs"),
            ([item.evidence_id for item in self.exact_evidence], "evidence IDs"),
            ([item.chunk_id for item in self.exact_evidence], "evidence chunk IDs"),
            ([item.chunk_id for item in self.retrieval_probe], "retrieval probe chunk IDs"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"sidecar {label} must be unique")
        allowed_validations = (
            {"answer_supported", "question_unambiguous", "uses_all_evidence"},
            {"answerable_from_top8"},
        )
        if set(self.validation) not in allowed_validations:
            raise ValueError("validation keys do not match a supported generator result")
        return self


class PrivateSidecarError(ValueError):
    """Failure message intentionally contains no private record values."""

    def __init__(self, messages: list[str]) -> None:
        self.messages = tuple(messages)
        super().__init__("; ".join(messages))


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _sanitized_errors(line_number: int, error: ValidationError) -> list[str]:
    return [
        f"line {line_number}: invalid sidecar value ({item['type']})"
        for item in error.errors(include_url=False, include_context=False, include_input=False)
    ]


def load_private_sidecar(
    path: Path,
    *,
    repository_root: Path | None = None,
) -> list[PrivateSidecarRecord]:
    """Read a root/user-only JSONL without ever including its values in errors."""

    source = path.expanduser()
    if source.is_symlink():
        raise PrivateSidecarError(["sidecar must not be a symlink"])
    resolved_source = source.resolve()
    resolved_root = (repository_root or Path.cwd()).expanduser().resolve()
    try:
        relative = resolved_source.relative_to(resolved_root)
    except ValueError:
        pass
    else:
        if not relative.parts or relative.parts[0] != ".private":
            raise PrivateSidecarError(["private sidecar inside the repository must be under .private/"])
    try:
        info = resolved_source.stat()
    except OSError as error:
        raise PrivateSidecarError([f"unable to stat sidecar ({type(error).__name__})"]) from None
    if not stat.S_ISREG(info.st_mode):
        raise PrivateSidecarError(["sidecar must be a regular file"])
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise PrivateSidecarError(["sidecar permissions must be exactly 0600"])
    try:
        text = resolved_source.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise PrivateSidecarError([f"unable to read sidecar ({type(error).__name__})"]) from None
    if text.startswith("\ufeff"):
        raise PrivateSidecarError(["sidecar UTF-8 BOM is not allowed"])
    lines = text.splitlines()
    if not lines:
        raise PrivateSidecarError(["sidecar is empty"])
    if len(lines) > MAX_SIDECAR_RECORDS:
        raise PrivateSidecarError(["sidecar has too many records"])

    records: list[PrivateSidecarRecord] = []
    errors: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"line {line_number}: blank lines are not allowed")
            continue
        if len(line.encode("utf-8")) > MAX_SIDECAR_LINE_BYTES:
            errors.append(f"line {line_number}: sidecar record exceeds size limit")
            continue
        try:
            json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            records.append(PrivateSidecarRecord.model_validate_json(line, strict=True))
        except _DuplicateKeyError:
            errors.append(f"line {line_number}: duplicate JSON key")
        except json.JSONDecodeError:
            errors.append(f"line {line_number}: invalid JSON")
        except ValidationError as error:
            errors.extend(_sanitized_errors(line_number, error))
    if errors:
        raise PrivateSidecarError(errors[:50])
    return records


def _quantity_payload(values: tuple[QuantitySpec, ...]) -> list[dict[str, str]]:
    return [{"value": item.value, "unit": item.unit} for item in values]


def _bind_record(record: GoldRecord, sidecar: PrivateSidecarRecord) -> None:
    if sidecar.gold_case_sha256 != gold_record_case_sha256(record):
        raise PrivateSidecarError(["gold case hash mismatch"])
    if sidecar.scope_id != record.scope_id:
        raise PrivateSidecarError(["gold scope mismatch"])
    if sidecar.language != record.language:
        raise PrivateSidecarError(["gold language mismatch"])
    expected_stratum = (
        "no_answer"
        if not record.answerable
        else {
            "single": "single_hop",
            "multi": "multi_hop",
            "cross_document": "cross_document",
        }[record.hop_type]
    )
    if sidecar.stratum != expected_stratum:
        raise PrivateSidecarError(["gold reasoning stratum mismatch"])

    gold_snapshots = {item.document_ref: item for item in record.document_scope}
    gold_documents = set(gold_snapshots)
    sidecar_documents = {item.document_ref for item in sidecar.source_documents}
    if not sidecar_documents.issubset(gold_documents):
        raise PrivateSidecarError(["sidecar document is outside gold scope"])
    document_ref_by_id = {item.document_id: item.document_ref for item in sidecar.source_documents}
    gold_evidence = {item.evidence_id: item for item in record.evidence}
    private_evidence = {item.evidence_id: item for item in sidecar.exact_evidence}
    if set(private_evidence) != set(gold_evidence):
        raise PrivateSidecarError(["sidecar evidence set does not match gold evidence"])
    for evidence_id, item in private_evidence.items():
        expected = gold_evidence[evidence_id]
        if (
            item.document_ref != expected.document_ref
            or item.page != expected.page
            or item.content_sha256 != expected.content_sha256
            or document_ref_by_id.get(item.document_id) != item.document_ref
        ):
            raise PrivateSidecarError(["sidecar evidence locator mismatch"])
    for probe in sidecar.retrieval_probe:
        snapshot = gold_snapshots.get(probe.document_ref)
        if snapshot is None or probe.page > snapshot.page_count:
            raise PrivateSidecarError(["retrieval probe locator is outside gold scope"])

    if tuple(sidecar.classification.content_types) != tuple(record.content_types):
        raise PrivateSidecarError(["sidecar content classification mismatch"])
    if tuple(sidecar.classification.challenge_tags) != tuple(record.challenge_tags):
        raise PrivateSidecarError(["sidecar challenge classification mismatch"])
    expected_flags = {
        "numbers": sidecar.classification.has_numbers,
        "units": sidecar.classification.has_units,
        "standards": sidecar.classification.has_standards,
    }
    if any(value != (name in record.challenge_tags) for name, value in expected_flags.items()):
        raise PrivateSidecarError(["sidecar challenge flags mismatch"])

    expected_quantities = _quantity_payload(sidecar.quantities.expected)
    supported_quantities = _quantity_payload(sidecar.quantities.supported)
    if not record.answerable and (expected_quantities or supported_quantities):
        raise PrivateSidecarError(["no-answer sidecar quantities must be empty"])
    try:
        quantity_unit_metrics(
            record.reference_answer or "",
            expected_quantities,
            supported_quantities=supported_quantities,
            answerable=record.answerable,
        )
    except ValueError:
        raise PrivateSidecarError(["sidecar quantity annotations are inconsistent"]) from None


def bind_gold_sidecar(
    records: list[GoldRecord],
    sidecars: list[PrivateSidecarRecord],
) -> dict[str, PrivateSidecarRecord]:
    """Bind sidecar records 1:1 to gold records and reject every mismatch."""

    if len({item.case_id for item in sidecars}) != len(sidecars):
        raise PrivateSidecarError(["sidecar case IDs must be unique"])
    gold_ids = {record.case_id for record in records}
    private_by_id = {item.case_id: item for item in sidecars}
    if set(private_by_id) != gold_ids:
        raise PrivateSidecarError(["sidecar cases do not match the gold set"])
    for record in records:
        _bind_record(record, private_by_id[record.case_id])
    return private_by_id


__all__ = [
    "PrivateSidecarError",
    "PrivateSidecarRecord",
    "QuantitySpec",
    "RetrievalProbe",
    "SidecarClassification",
    "SidecarDocument",
    "SidecarEvidence",
    "SidecarGeneration",
    "SidecarQuantities",
    "bind_gold_sidecar",
    "load_private_sidecar",
]
