"""Strict, privacy-preserving contract for the private RAG gold set."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, ValidationError, model_validator

SCHEMA_VERSION: Literal["rag-gold-v1"] = "rag-gold-v1"
MIN_RECORDS = 200
MAX_RECORDS = 500
MIN_NO_ANSWER_SHARE = 0.20
RELEASE_MIN_CLASS_COUNT = 5
MAX_JSONL_LINE_BYTES = 256 * 1024

GoldMode = Literal["candidate", "release"]
GoldStatus = Literal["candidate", "reviewed"]
Language = Literal["ru", "en", "zh"]
HopType = Literal["single", "multi", "cross_document"]
ContentType = Literal["text", "table", "formula", "figure", "scan"]
ChallengeTag = Literal[
    "numbers",
    "units",
    "standards",
    "prompt_injection",
    "leakage",
]

REQUIRED_LANGUAGES: tuple[Language, ...] = ("ru", "en", "zh")
REQUIRED_HOP_TYPES: tuple[HopType, ...] = ("single", "multi", "cross_document")
REQUIRED_CONTENT_TYPES: tuple[ContentType, ...] = (
    "text",
    "table",
    "formula",
    "figure",
    "scan",
)
REQUIRED_CHALLENGE_TAGS: tuple[ChallengeTag, ...] = (
    "numbers",
    "units",
    "standards",
    "prompt_injection",
    "leakage",
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_CASE_ID_PATTERN = r"^ragq-[a-z0-9][a-z0-9._-]{7,63}$"
_REVIEWER_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{2,63}$"
_SCOPE_ID_PATTERN = r"^scope-sha256:[0-9a-f]{64}$"
_DOCUMENT_REF_PATTERN = r"^doc-sha256:[0-9a-f]{64}$"
_EVIDENCE_ID_PATTERN = (
    r"^ev-sha256:[0-9a-f]{64}:p[1-9][0-9]*:"
    r"(?:text|table|formula|figure|scan):[0-9a-f]{64}$"
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )


def normalize_text_content(value: str) -> str:
    """Canonicalize human-authored text before hashing, without collapsing whitespace."""

    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n")).strip()


def text_sha256(value: str) -> str:
    return hashlib.sha256(normalize_text_content(value).encode("utf-8")).hexdigest()


def bytes_sha256(value: bytes) -> str:
    """Hash source bytes exactly as read from object storage."""

    return hashlib.sha256(value).hexdigest()


def make_scope_id(owner_sub: str) -> str:
    """Create a non-reversible owner scope identifier without persisting ``owner_sub``."""

    if not owner_sub:
        raise ValueError("owner_sub must not be empty")
    return f"scope-sha256:{hashlib.sha256(owner_sub.encode('utf-8')).hexdigest()}"


def make_document_ref(source_sha256: str) -> str:
    return f"doc-sha256:{source_sha256}"


def make_evidence_id(
    document_sha256: str,
    page: int,
    content_type: ContentType,
    content_sha256: str,
) -> str:
    return f"ev-sha256:{document_sha256}:p{page}:{content_type}:{content_sha256}"


def parsed_chunks_sha256(chunks: Sequence[Mapping[str, Any]]) -> str:
    """Hash the full ordered parser output without database UUIDs or filenames."""

    canonical: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for chunk in chunks:
        idx = chunk.get("idx")
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")
        kind = chunk.get("kind")
        heading_path = chunk.get("heading_path")
        text = chunk.get("text")
        if type(idx) is not int or idx < 0 or idx in seen_indices:
            raise ValueError("chunk idx values must be unique non-negative integers")
        if page_start is not None and (type(page_start) is not int or page_start < 0):
            raise ValueError("page_start must be a non-negative integer or null")
        if page_end is not None and (type(page_end) is not int or page_end < 0):
            raise ValueError("page_end must be a non-negative integer or null")
        if page_start is not None and page_end is not None and page_end < page_start:
            raise ValueError("page_end must not precede page_start")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("chunk kind must be a non-empty string")
        if not isinstance(heading_path, str) or not isinstance(text, str):
            raise ValueError("chunk heading_path and text must be strings")
        seen_indices.add(idx)
        canonical.append(
            {
                "heading_path": normalize_text_content(heading_path),
                "idx": idx,
                "kind": kind.strip(),
                "page_end": page_end,
                "page_start": page_start,
                "text": normalize_text_content(text),
            }
        )
    canonical.sort(key=lambda item: item["idx"])
    return hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()


class DocumentSnapshot(_StrictModel):
    """Immutable corpus document version; titles and file paths are deliberately omitted."""

    document_ref: str = Field(pattern=_DOCUMENT_REF_PATTERN)
    source_sha256: str = Field(pattern=_SHA256_PATTERN)
    parsed_content_sha256: str = Field(pattern=_SHA256_PATTERN)
    page_count: int = Field(strict=True, ge=1, le=100_000)

    @model_validator(mode="after")
    def validate_document_ref(self) -> Self:
        if self.document_ref != make_document_ref(self.source_sha256):
            raise ValueError("document_ref does not match source_sha256")
        return self


class EvidenceRef(_StrictModel):
    """Stable evidence locator without a private excerpt."""

    evidence_id: str = Field(pattern=_EVIDENCE_ID_PATTERN)
    document_ref: str = Field(pattern=_DOCUMENT_REF_PATTERN)
    page: int = Field(strict=True, ge=1, le=100_000)
    content_type: ContentType
    content_sha256: str = Field(pattern=_SHA256_PATTERN)
    relevance_grade: int = Field(strict=True, ge=1, le=3)
    bbox: tuple[float, float, float, float] | None

    @model_validator(mode="after")
    def validate_locator(self) -> Self:
        document_sha256 = self.document_ref.removeprefix("doc-sha256:")
        expected = make_evidence_id(
            document_sha256,
            self.page,
            self.content_type,
            self.content_sha256,
        )
        if self.evidence_id != expected:
            raise ValueError("evidence_id does not match its hash locator fields")
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in self.bbox):
                raise ValueError("bbox values must be finite normalized coordinates")
            if x1 >= x2 or y1 >= y2:
                raise ValueError("bbox must have positive area")
        return self


class ReviewMetadata(_StrictModel):
    reviewer_id: str = Field(pattern=_REVIEWER_ID_PATTERN)
    reviewed_at: AwareDatetime
    case_sha256: str = Field(pattern=_SHA256_PATTERN)


class GoldRecord(_StrictModel):
    """One JSONL record. Cross-record release constraints are validated separately."""

    schema_version: Literal["rag-gold-v1"]
    case_id: str = Field(pattern=_CASE_ID_PATTERN)
    status: GoldStatus
    scope_id: str = Field(pattern=_SCOPE_ID_PATTERN)
    language: Language
    question: str = Field(min_length=8, max_length=4_000)
    question_sha256: str = Field(pattern=_SHA256_PATTERN)
    answerable: StrictBool
    reference_answer: str | None = Field(max_length=16_000)
    reference_answer_sha256: str | None = Field(pattern=_SHA256_PATTERN)
    hop_type: HopType
    content_types: tuple[ContentType, ...] = Field(min_length=1, max_length=5)
    challenge_tags: tuple[ChallengeTag, ...] = Field(max_length=5)
    document_scope: tuple[DocumentSnapshot, ...] = Field(min_length=1, max_length=64)
    evidence: tuple[EvidenceRef, ...] = Field(max_length=16)
    review: ReviewMetadata | None

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        if self.question_sha256 != text_sha256(self.question):
            raise ValueError("question_sha256 does not match question")
        if len(self.content_types) != len(set(self.content_types)):
            raise ValueError("content_types must be unique")
        if len(self.challenge_tags) != len(set(self.challenge_tags)):
            raise ValueError("challenge_tags must be unique")

        documents = {item.document_ref: item for item in self.document_scope}
        if len(documents) != len(self.document_scope):
            raise ValueError("document_scope references must be unique")
        evidence_ids = {item.evidence_id for item in self.evidence}
        if len(evidence_ids) != len(self.evidence):
            raise ValueError("evidence_id values must be unique within a case")
        for item in self.evidence:
            document = documents.get(item.document_ref)
            if document is None:
                raise ValueError("evidence document is outside document_scope")
            if item.page > document.page_count:
                raise ValueError("evidence page exceeds document page_count")
            if item.content_type not in self.content_types:
                raise ValueError("evidence content_type is missing from content_types")

        if self.answerable:
            if self.reference_answer is None or self.reference_answer_sha256 is None:
                raise ValueError("answerable cases require reference_answer and its hash")
            if self.reference_answer_sha256 != text_sha256(self.reference_answer):
                raise ValueError("reference_answer_sha256 does not match reference_answer")
            if not self.evidence:
                raise ValueError("answerable cases require evidence")
        elif self.reference_answer is not None or self.reference_answer_sha256 is not None:
            raise ValueError("non-answerable cases must not contain a reference answer")
        elif self.evidence:
            raise ValueError("non-answerable cases must not declare relevant evidence")

        if self.hop_type == "multi" and self.answerable and len(self.evidence) < 2:
            raise ValueError("multi-hop cases require at least two evidence references")
        if self.hop_type == "cross_document" and self.answerable:
            evidence_documents = {item.document_ref for item in self.evidence}
            if len(evidence_documents) < 2:
                raise ValueError("cross-document cases require evidence from at least two documents")
        if "leakage" in self.challenge_tags and self.answerable:
            raise ValueError("leakage probes must expect refusal/abstention")

        if self.status == "candidate" and self.review is not None:
            raise ValueError("candidate cases must not contain review metadata")
        if self.status == "reviewed":
            if self.review is None:
                raise ValueError("reviewed cases require review metadata")
            if self.review.case_sha256 != gold_record_case_sha256(self):
                raise ValueError("review metadata does not match the reviewed case content")
        return self


class GoldSetReport(_StrictModel):
    mode: GoldMode
    record_count: int
    no_answer_count: int
    no_answer_share: float
    status_counts: dict[str, int]
    language_counts: dict[str, int]
    hop_type_counts: dict[str, int]
    content_type_counts: dict[str, int]
    challenge_tag_counts: dict[str, int]


class GoldSetValidationError(ValueError):
    """Validation failure whose message never contains private record values."""

    def __init__(self, messages: list[str]) -> None:
        self.messages = tuple(messages)
        super().__init__("; ".join(messages))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def gold_record_case_sha256(record: GoldRecord) -> str:
    payload = record.model_dump(mode="json", exclude={"status", "review"})
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_coverage(
    counts: Mapping[Any, int],
    required: tuple[str, ...],
    *,
    dimension: str,
    minimum: int,
) -> None:
    missing = sum(counts[value] < minimum for value in required)
    if missing:
        raise GoldSetValidationError([f"{dimension}: {missing} required classes are below minimum {minimum}"])


def validate_gold_set(
    records: list[GoldRecord],
    *,
    mode: GoldMode = "candidate",
) -> GoldSetReport:
    if not MIN_RECORDS <= len(records) <= MAX_RECORDS:
        raise GoldSetValidationError(
            [f"record_count must be in [{MIN_RECORDS}, {MAX_RECORDS}], got {len(records)}"]
        )
    if len({record.case_id for record in records}) != len(records):
        raise GoldSetValidationError(["case_id values must be unique"])
    if len({record.question_sha256 for record in records}) != len(records):
        raise GoldSetValidationError(["question content hashes must be unique"])

    # Leakage probes are policy-refusal cases and cannot satisfy the missing-evidence quota.
    no_answer_count = sum(
        not record.answerable and "leakage" not in record.challenge_tags for record in records
    )
    if no_answer_count / len(records) < MIN_NO_ANSWER_SHARE:
        raise GoldSetValidationError(
            [
                "no-answer share must be at least "
                f"{MIN_NO_ANSWER_SHARE:.0%}, got {no_answer_count / len(records):.1%}"
            ]
        )

    status_counts = Counter(record.status for record in records)
    language_counts = Counter(record.language for record in records)
    hop_counts = Counter(record.hop_type for record in records if record.answerable)
    content_counts = Counter(value for record in records for value in record.content_types)
    challenge_counts = Counter(value for record in records for value in record.challenge_tags)

    class_minimum = RELEASE_MIN_CLASS_COUNT if mode == "release" else 1
    _require_coverage(
        language_counts,
        REQUIRED_LANGUAGES,
        dimension="languages",
        minimum=class_minimum,
    )
    _require_coverage(
        hop_counts,
        REQUIRED_HOP_TYPES,
        dimension="hop_types",
        minimum=class_minimum,
    )
    _require_coverage(
        content_counts,
        REQUIRED_CONTENT_TYPES,
        dimension="content_types",
        minimum=class_minimum,
    )
    _require_coverage(
        challenge_counts,
        REQUIRED_CHALLENGE_TAGS,
        dimension="challenge_tags",
        minimum=class_minimum,
    )
    if mode == "release" and status_counts["reviewed"] != len(records):
        raise GoldSetValidationError(["release mode requires every case to be reviewed"])

    return GoldSetReport(
        mode=mode,
        record_count=len(records),
        no_answer_count=no_answer_count,
        no_answer_share=round(no_answer_count / len(records), 6),
        status_counts=dict(sorted(status_counts.items())),
        language_counts=dict(sorted(language_counts.items())),
        hop_type_counts=dict(sorted(hop_counts.items())),
        content_type_counts=dict(sorted(content_counts.items())),
        challenge_tag_counts=dict(sorted(challenge_counts.items())),
    )


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _safe_location(error: Mapping[str, Any]) -> str:
    known_fields = {
        "schema_version",
        "case_id",
        "status",
        "scope_id",
        "language",
        "question",
        "question_sha256",
        "answerable",
        "reference_answer",
        "reference_answer_sha256",
        "hop_type",
        "content_types",
        "challenge_tags",
        "document_scope",
        "evidence",
        "review",
        "document_ref",
        "source_sha256",
        "parsed_content_sha256",
        "page_count",
        "evidence_id",
        "page",
        "content_type",
        "content_sha256",
        "relevance_grade",
        "bbox",
        "reviewer_id",
        "reviewed_at",
        "case_sha256",
    }
    parts = [
        str(part) if isinstance(part, int) or part in known_fields else "<field>" for part in error["loc"]
    ]
    return ".".join(parts) or "record"


def _sanitized_pydantic_errors(line_number: int, error: ValidationError) -> list[str]:
    return [
        f"line {line_number}, {_safe_location(item)}: invalid value ({item['type']})"
        for item in error.errors(include_url=False, include_context=False, include_input=False)
    ]


def ensure_private_gold_path(path: Path, repository_root: Path) -> Path:
    """Only allow in-repository private data below the ignored ``.private`` tree."""

    resolved_path = path.expanduser().resolve()
    resolved_root = repository_root.expanduser().resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError:
        return resolved_path
    if not relative.parts or relative.parts[0] != ".private":
        raise GoldSetValidationError(["private gold set inside the repository must be under .private/"])
    return resolved_path


def load_gold_set(
    path: Path,
    *,
    mode: GoldMode = "candidate",
    repository_root: Path | None = None,
) -> tuple[list[GoldRecord], GoldSetReport]:
    source = ensure_private_gold_path(path, repository_root or Path.cwd())
    try:
        raw = source.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise GoldSetValidationError([f"unable to read private JSONL ({type(error).__name__})"]) from None
    if text.startswith("\ufeff"):
        raise GoldSetValidationError(["UTF-8 BOM is not allowed"])

    lines = text.splitlines()
    if not lines:
        raise GoldSetValidationError(["JSONL is empty"])
    if len(lines) > MAX_RECORDS:
        raise GoldSetValidationError([f"JSONL contains more than {MAX_RECORDS} lines"])

    records: list[GoldRecord] = []
    errors: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"line {line_number}: blank lines are not allowed")
            continue
        if len(line.encode("utf-8")) > MAX_JSONL_LINE_BYTES:
            errors.append(f"line {line_number}: record exceeds size limit")
            continue
        try:
            json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            records.append(GoldRecord.model_validate_json(line, strict=True))
        except _DuplicateKeyError:
            errors.append(f"line {line_number}: duplicate JSON key")
        except json.JSONDecodeError:
            errors.append(f"line {line_number}: invalid JSON")
        except ValidationError as error:
            errors.extend(_sanitized_pydantic_errors(line_number, error))
    if errors:
        raise GoldSetValidationError(errors[:50])
    return records, validate_gold_set(records, mode=mode)


def gold_record_json_schema() -> dict[str, Any]:
    return GoldRecord.model_json_schema()


__all__ = [
    "MAX_RECORDS",
    "MIN_NO_ANSWER_SHARE",
    "MIN_RECORDS",
    "RELEASE_MIN_CLASS_COUNT",
    "SCHEMA_VERSION",
    "DocumentSnapshot",
    "EvidenceRef",
    "GoldRecord",
    "GoldSetReport",
    "GoldSetValidationError",
    "ReviewMetadata",
    "bytes_sha256",
    "ensure_private_gold_path",
    "gold_record_case_sha256",
    "gold_record_json_schema",
    "load_gold_set",
    "make_document_ref",
    "make_evidence_id",
    "make_scope_id",
    "normalize_text_content",
    "parsed_chunks_sha256",
    "text_sha256",
    "validate_gold_set",
]
