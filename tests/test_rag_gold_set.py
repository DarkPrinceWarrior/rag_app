from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_app.eval.gold_set import (
    GoldRecord,
    GoldSetValidationError,
    ReviewMetadata,
    bytes_sha256,
    ensure_private_gold_path,
    gold_record_case_sha256,
    gold_record_json_schema,
    load_gold_set,
    make_document_ref,
    make_evidence_id,
    make_scope_id,
    parsed_chunks_sha256,
    text_sha256,
    validate_gold_set,
)

CONTENT_TYPES = ("text", "table", "formula", "figure", "scan")
HOP_TYPES = ("single", "multi", "cross_document")
LANGUAGES = ("ru", "en", "zh")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _raw_record(
    index: int,
    *,
    answerable: bool = True,
    hop_type: str = "single",
    content_type: str = "text",
    challenge_tags: list[str] | None = None,
) -> dict:
    document_count = 2 if hop_type == "cross_document" and answerable else 1
    documents: list[dict] = []
    for document_index in range(document_count):
        source_hash = _sha(f"document-{index}-{document_index}")
        documents.append(
            {
                "document_ref": make_document_ref(source_hash),
                "source_sha256": source_hash,
                "parsed_content_sha256": _sha(f"parsed-{index}-{document_index}"),
                "page_count": 10,
            }
        )

    evidence: list[dict] = []
    if answerable:
        evidence_count = 2 if hop_type in {"multi", "cross_document"} else 1
        for evidence_index in range(evidence_count):
            document = documents[evidence_index % len(documents)]
            content_hash = _sha(f"evidence-{index}-{evidence_index}")
            evidence.append(
                {
                    "evidence_id": make_evidence_id(
                        document["source_sha256"], evidence_index + 1, content_type, content_hash
                    ),
                    "document_ref": document["document_ref"],
                    "page": evidence_index + 1,
                    "content_type": content_type,
                    "content_sha256": content_hash,
                    "relevance_grade": 3,
                    "bbox": [0.1, 0.2, 0.8, 0.9],
                }
            )

    question = f"Synthetic technical question number {index}?"
    answer = f"Synthetic reference answer {index}." if answerable else None
    return {
        "schema_version": "rag-gold-v1",
        "case_id": f"ragq-synthetic-{index:04d}",
        "status": "candidate",
        "scope_id": make_scope_id("synthetic-owner"),
        "language": LANGUAGES[index % len(LANGUAGES)],
        "question": question,
        "question_sha256": text_sha256(question),
        "answerable": answerable,
        "reference_answer": answer,
        "reference_answer_sha256": text_sha256(answer) if answer else None,
        "hop_type": hop_type,
        "content_types": [content_type],
        "challenge_tags": challenge_tags or [],
        "document_scope": documents,
        "evidence": evidence,
        "review": None,
    }


def _candidate_records() -> list[GoldRecord]:
    records: list[GoldRecord] = []
    for index in range(200):
        if index < 40:
            raw = _raw_record(index, answerable=False, content_type=CONTENT_TYPES[index % 5])
        else:
            answerable = not 40 <= index < 45
            hop_type = HOP_TYPES[(index - 40) % len(HOP_TYPES)] if answerable else "single"
            tags: list[str] = []
            if 40 <= index < 45:
                tags = ["leakage"]
            elif 45 <= index < 50:
                tags = ["prompt_injection"]
            elif 50 <= index < 55:
                tags = ["numbers"]
            elif 55 <= index < 60:
                tags = ["units"]
            elif 60 <= index < 65:
                tags = ["standards"]
            raw = _raw_record(
                index,
                answerable=answerable,
                hop_type=hop_type,
                content_type=CONTENT_TYPES[index % 5],
                challenge_tags=tags,
            )
        records.append(GoldRecord.model_validate_json(json.dumps(raw), strict=True))
    return records


def _reviewed(record: GoldRecord) -> GoldRecord:
    raw = record.model_dump(mode="json")
    raw["status"] = "reviewed"
    raw["review"] = ReviewMetadata(
        reviewer_id="reviewer-01",
        reviewed_at=datetime(2026, 7, 13, tzinfo=UTC),
        case_sha256=gold_record_case_sha256(record),
    ).model_dump(mode="json")
    return GoldRecord.model_validate_json(json.dumps(raw), strict=True)


def _write_jsonl(path: Path, records: list[GoldRecord]) -> None:
    path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )


def test_canonical_hashes_and_cross_document_invariants() -> None:
    raw = _raw_record(1, hop_type="cross_document", content_type="table")
    record = GoldRecord.model_validate_json(json.dumps(raw), strict=True)

    assert len(record.evidence) == 2
    assert len({item.document_ref for item in record.evidence}) == 2
    assert record.evidence[0].evidence_id.startswith("ev-sha256:")

    raw["question_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="question_sha256"):
        GoldRecord.model_validate_json(json.dumps(raw), strict=True)


def test_scope_and_document_content_hashes_are_canonical() -> None:
    assert make_scope_id("owner-sub-1") == f"scope-sha256:{_sha('owner-sub-1')}"
    assert bytes_sha256(b"original-minio-bytes") == hashlib.sha256(b"original-minio-bytes").hexdigest()
    chunks = [
        {
            "idx": 1,
            "kind": "table",
            "heading_path": "Section 2",
            "page_start": 1,
            "page_end": 1,
            "text": "A | B\r\n1 | 2",
            "database_uuid": "ignored",
        },
        {
            "idx": 0,
            "kind": "text",
            "heading_path": "Section 1",
            "page_start": 0,
            "page_end": 0,
            "text": "Pressure 42 bar",
        },
    ]
    assert parsed_chunks_sha256(chunks) == parsed_chunks_sha256(list(reversed(chunks)))
    changed = [*chunks[:-1], {**chunks[-1], "text": "Pressure 43 bar"}]
    assert parsed_chunks_sha256(chunks) != parsed_chunks_sha256(changed)

    raw = _raw_record(2)
    raw["scope_id"] = "owner-sub-1"
    with pytest.raises(ValidationError, match="scope_id"):
        GoldRecord.model_validate_json(json.dumps(raw), strict=True)


def test_candidate_and_release_modes_enforce_coverage_and_review() -> None:
    candidates = _candidate_records()
    report = validate_gold_set(candidates, mode="candidate")

    assert report.record_count == 200
    assert report.no_answer_count == 40
    assert report.no_answer_share == 0.2
    assert set(report.language_counts) == set(LANGUAGES)
    assert set(report.content_type_counts) == set(CONTENT_TYPES)
    with pytest.raises(GoldSetValidationError, match="every case"):
        validate_gold_set(candidates, mode="release")

    release = [_reviewed(record) for record in candidates]
    release_report = validate_gold_set(release, mode="release")
    assert release_report.status_counts == {"reviewed": 200}


def test_no_answer_quota_excludes_leakage_refusal_probes() -> None:
    records = _candidate_records()
    for index in range(40):
        raw = records[index].model_dump(mode="json")
        raw["challenge_tags"] = ["leakage"]
        records[index] = GoldRecord.model_validate_json(json.dumps(raw), strict=True)

    with pytest.raises(GoldSetValidationError, match="no-answer share"):
        validate_gold_set(records)


def test_loader_is_strict_and_does_not_echo_private_content(tmp_path: Path) -> None:
    records = _candidate_records()
    path = tmp_path / "gold.jsonl"
    _write_jsonl(path, records)
    loaded, report = load_gold_set(path, repository_root=Path.cwd())
    assert len(loaded) == report.record_count == 200

    private_marker = "PRIVATE-CUSTOMER-SECRET-4711"
    first = records[0].model_dump(mode="json")
    first["question"] = private_marker
    lines = [json.dumps(first), *[record.model_dump_json() for record in records[1:]]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(GoldSetValidationError) as caught:
        load_gold_set(path, repository_root=Path.cwd())
    assert private_marker not in str(caught.value)


def test_loader_rejects_duplicate_json_keys_without_echoing_values(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text('{"question":"private-a","question":"private-b"}\n', encoding="utf-8")

    with pytest.raises(GoldSetValidationError, match="duplicate JSON key") as caught:
        load_gold_set(path, repository_root=Path.cwd())
    assert "private-a" not in str(caught.value)
    assert "private-b" not in str(caught.value)


def test_in_repository_gold_path_must_be_private(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    private = repository / ".private" / "rag_gold" / "gold.jsonl"
    public = repository / "docs" / "gold.jsonl"

    assert ensure_private_gold_path(private, repository) == private.resolve()
    with pytest.raises(GoldSetValidationError, match="under .private"):
        ensure_private_gold_path(public, repository)


def test_checked_in_json_schema_matches_model() -> None:
    schema_path = Path("docs/schemas/rag_gold_record.schema.json")
    assert json.loads(schema_path.read_text(encoding="utf-8")) == gold_record_json_schema()
