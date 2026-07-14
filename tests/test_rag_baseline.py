from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rag_app.eval.baseline import (
    BaselineEvaluationError,
    BaselineModelRevisions,
    BaselineObservation,
    RetrievedUnit,
    RuntimeModelRevision,
    aggregate_metrics,
    require_loopback_database_url,
    require_loopback_endpoint,
    require_loopback_url,
    score_observation,
)
from rag_app.eval.gold_set import (
    DocumentSnapshot,
    EvidenceRef,
    GoldRecord,
    gold_record_case_sha256,
    make_document_ref,
    make_evidence_id,
    make_scope_id,
    text_sha256,
)
from rag_app.eval.private_sidecar import (
    PrivateSidecarError,
    PrivateSidecarRecord,
    RetrievalProbe,
    bind_gold_sidecar,
    load_private_sidecar,
)


def _script_module():
    path = Path(__file__).parents[1] / "scripts" / "evaluate_rag_gold_set.py"
    spec = importlib.util.spec_from_file_location("evaluate_rag_gold_set", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SCRIPT = _script_module()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _model_revisions() -> BaselineModelRevisions:
    revision = RuntimeModelRevision(
        endpoint_metadata_sha256="f" * 64,
        runtime_version_sha256="d" * 64,
        runtime_process_sha256="c" * 64,
        local_config_manifest_sha256="e" * 64,
        weight_manifest_sha256="b" * 64,
        weight_file_count=1,
        weight_bytes=1024,
        declared_revision="synthetic-revision",
    )
    return BaselineModelRevisions(
        llm=revision,
        embedding=revision,
        reranker=revision,
    )


def _case(*, answerable: bool = True) -> tuple[GoldRecord, PrivateSidecarRecord, uuid.UUID]:
    source_hash = _sha("synthetic-source")
    document_ref = make_document_ref(source_hash)
    quote = "Required operating pressure is 42 bar."
    content_hash = text_sha256(quote)
    evidence_id = make_evidence_id(source_hash, 2, "text", content_hash)
    chunk_id = uuid.UUID("10000000-0000-0000-0000-000000000001")
    document_id = uuid.UUID("20000000-0000-0000-0000-000000000001")
    question = "What is the required operating pressure?"
    answer = "The required operating pressure is 42 bar." if answerable else None
    record = GoldRecord(
        schema_version="rag-gold-v1",
        case_id="ragq-synthetic-baseline-0001",
        status="candidate",
        scope_id=make_scope_id("synthetic-owner"),
        language="en",
        question=question,
        question_sha256=text_sha256(question),
        answerable=answerable,
        reference_answer=answer,
        reference_answer_sha256=text_sha256(answer) if answer else None,
        hop_type="single",
        content_types=("text",),
        challenge_tags=("numbers", "units") if answerable else (),
        document_scope=(
            DocumentSnapshot(
                document_ref=document_ref,
                source_sha256=source_hash,
                parsed_content_sha256=_sha("parsed-content"),
                page_count=3,
            ),
        ),
        evidence=(
            (
                EvidenceRef(
                    evidence_id=evidence_id,
                    document_ref=document_ref,
                    page=2,
                    content_type="text",
                    content_sha256=content_hash,
                    relevance_grade=3,
                    bbox=None,
                ),
            )
            if answerable
            else ()
        ),
        review=None,
    )
    payload = {
        "schema_version": "private-rag-generator-v1",
        "case_id": record.case_id,
        "gold_case_sha256": gold_record_case_sha256(record),
        "scope_id": record.scope_id,
        "stratum": "single_hop" if answerable else "no_answer",
        "language": "en",
        "source_documents": [
            {
                "document_id": str(document_id),
                "document_ref": document_ref,
                "source_lang": "en",
            }
        ],
        "classification": {
            "content_types": ["text"],
            "challenge_tags": ["numbers", "units"] if answerable else [],
            "has_numbers": answerable,
            "has_units": answerable,
            "has_standards": False,
        },
        "generation": {"model": "synthetic-local-model", "seed": 3086},
        "exact_evidence": (
            [
                {
                    "evidence_id": evidence_id,
                    "document_id": str(document_id),
                    "document_ref": document_ref,
                    "chunk_id": str(chunk_id),
                    "chunk_index": 0,
                    "kind": "section",
                    "heading_path": "Synthetic",
                    "page": 2,
                    "page_start": 1,
                    "page_end": 1,
                    "text_sha256": _sha(quote),
                    "content_sha256": content_hash,
                    "exact_quote": quote,
                    "retrieval_score": None,
                }
            ]
            if answerable
            else []
        ),
        "retrieval_probe": [],
        "quantities": {
            "expected": [{"value": "42", "unit": "bar"}] if answerable else [],
            "supported": [{"value": "42", "unit": "bar"}] if answerable else [],
        },
        "validation": (
            {
                "answer_supported": True,
                "question_unambiguous": True,
                "uses_all_evidence": True,
            }
            if answerable
            else {"answerable_from_top8": False}
        ),
    }
    return record, PrivateSidecarRecord.model_validate_json(json.dumps(payload)), chunk_id


def test_private_sidecar_loader_requires_0600_and_binds_gold(tmp_path: Path) -> None:
    record, sidecar, _ = _case()
    path = tmp_path / "sidecar.jsonl"
    path.write_text(sidecar.model_dump_json() + "\n", encoding="utf-8")
    os.chmod(path, 0o600)

    loaded = load_private_sidecar(path)
    assert bind_gold_sidecar([record], loaded) == {record.case_id: sidecar}

    os.chmod(path, 0o640)
    with pytest.raises(PrivateSidecarError, match="0600"):
        load_private_sidecar(path)


def test_production_endpoints_are_loopback_only() -> None:
    assert require_loopback_url("http://127.0.0.1:8006/v1", name="test")
    assert require_loopback_endpoint("localhost:9000", name="test")
    assert require_loopback_database_url("postgresql+asyncpg://rag:secret@localhost:5433/rag_app")
    with pytest.raises(BaselineEvaluationError, match="loopback"):
        require_loopback_url("http://model.internal:8006/v1", name="test")
    with pytest.raises(BaselineEvaluationError, match="loopback"):
        require_loopback_database_url("postgresql+asyncpg://rag:secret@db.internal:5432/rag_app")
    with pytest.raises(BaselineEvaluationError, match="credential-free"):
        require_loopback_url("http://token@localhost:8006/v1", name="test")
    with pytest.raises(BaselineEvaluationError, match="credential-free"):
        require_loopback_endpoint("http://localhost:9000?redirect=external", name="test")


def test_release_cli_is_default_and_requires_enough_results_for_at_10() -> None:
    parser = _SCRIPT.build_parser()
    args = parser.parse_args(["gold.jsonl", "sidecar.jsonl"])
    assert args.mode == "release"
    assert args.top_k == 10
    with pytest.raises(SystemExit):
        parser.parse_args(["gold.jsonl", "sidecar.jsonl", "--top-k", "9"])


def test_baseline_report_is_written_atomically_with_0600(tmp_path: Path) -> None:
    report = tmp_path / "private" / "baseline.json"
    report.parent.mkdir(mode=0o700)
    _SCRIPT._atomic_write_report(report, {"case_count": 200})
    assert json.loads(report.read_text(encoding="utf-8")) == {"case_count": 200}
    assert os.stat(report).st_mode & 0o777 == 0o600

    os.chmod(report, 0o644)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        _SCRIPT._atomic_write_report(report, {"case_count": 201})
    assert json.loads(report.read_text(encoding="utf-8")) == {"case_count": 200}


def test_cached_scope_revalidates_each_sidecar_document_mapping() -> None:
    record, sidecar, _ = _case()
    fingerprint = hashlib.sha256(
        json.dumps(
            [item.model_dump(mode="json") for item in record.document_scope],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    verified_document = sidecar.source_documents[0]
    runner = object.__new__(_SCRIPT.ProductionBaselineRunner)
    runner._scope_cache = {
        record.scope_id: (
            "synthetic-owner",
            fingerprint,
            {verified_document.document_id: verified_document.document_ref},
            "f" * 64,
        )
    }
    corrupted_document = verified_document.model_copy(update={"document_id": uuid.uuid4()})
    corrupted_sidecar = sidecar.model_copy(update={"source_documents": (corrupted_document,)})

    with pytest.raises(BaselineEvaluationError, match="document mapping"):
        asyncio.run(runner._resolve_and_verify_scope(record, corrupted_sidecar))


def test_sidecar_errors_never_echo_private_values(tmp_path: Path) -> None:
    marker = "PRIVATE-CUSTOMER-CONTENT-9182"
    path = tmp_path / "sidecar.jsonl"
    path.write_text(json.dumps({"unexpected": marker}) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)

    with pytest.raises(PrivateSidecarError) as caught:
        load_private_sidecar(path)
    assert marker not in str(caught.value)


def test_sidecar_inside_repository_must_be_under_private(tmp_path: Path) -> None:
    record, sidecar, _ = _case()
    repository = tmp_path / "repo"
    public = repository / "docs" / "sidecar.jsonl"
    private = repository / ".private" / "sidecar.jsonl"
    public.parent.mkdir(parents=True)
    private.parent.mkdir(parents=True)
    for path in (public, private):
        path.write_text(sidecar.model_dump_json() + "\n", encoding="utf-8")
        os.chmod(path, 0o600)

    with pytest.raises(PrivateSidecarError, match="under .private"):
        load_private_sidecar(public, repository_root=repository)
    loaded = load_private_sidecar(private, repository_root=repository)
    assert bind_gold_sidecar([record], loaded)[record.case_id] == sidecar


def test_sidecar_retrieval_probe_rejects_duplicate_chunks() -> None:
    _, sidecar, chunk_id = _case(answerable=False)
    document = sidecar.source_documents[0]
    probe = {
        "document_id": str(document.document_id),
        "document_ref": document.document_ref,
        "chunk_id": str(chunk_id),
        "page": 1,
        "page_start": 0,
        "page_end": 0,
        "content_sha256": _sha("probe"),
        "retrieval_score": 0.7,
    }
    raw = sidecar.model_dump(mode="json")
    raw["retrieval_probe"] = [probe, probe]
    with pytest.raises(ValidationError, match="retrieval probe chunk IDs"):
        PrivateSidecarRecord.model_validate_json(json.dumps(raw))


def test_sidecar_retrieval_probe_must_stay_inside_gold_snapshot() -> None:
    record, sidecar, chunk_id = _case(answerable=False)
    document = sidecar.source_documents[0]
    raw = sidecar.model_dump(mode="json")
    raw["retrieval_probe"] = [
        {
            "document_id": str(document.document_id),
            "document_ref": document.document_ref,
            "chunk_id": str(chunk_id),
            "page": 99,
            "page_start": 98,
            "page_end": 98,
            "content_sha256": _sha("probe"),
            "retrieval_score": 0.7,
        }
    ]
    invalid = PrivateSidecarRecord.model_validate_json(json.dumps(raw))
    with pytest.raises(PrivateSidecarError, match="retrieval probe locator"):
        bind_gold_sidecar([record], [invalid])


def test_runtime_retrieval_probe_requires_exact_page_locator() -> None:
    record, sidecar, chunk_id = _case(answerable=False)
    document = sidecar.source_documents[0]
    body = "Synthetic retrieval probe body."
    probe = RetrievalProbe(
        document_id=document.document_id,
        document_ref=document.document_ref,
        chunk_id=chunk_id,
        page=2,
        page_start=1,
        page_end=1,
        content_sha256=text_sha256(body),
        retrieval_score=0.7,
    )
    row = SimpleNamespace(
        document_id=document.document_id,
        page_start=1,
        page_end=1,
        body=body,
    )
    document_refs = {document.document_id: document.document_ref}

    assert _SCRIPT._retrieval_probe_matches(
        probe,
        row,
        document_refs,
        record.document_scope[0],
    )
    assert not _SCRIPT._retrieval_probe_matches(
        probe.model_copy(update={"page": 3}),
        row,
        document_refs,
        record.document_scope[0],
    )
    assert not _SCRIPT._retrieval_probe_matches(
        probe.model_copy(update={"page_start": 0}),
        row,
        document_refs,
        record.document_scope[0],
    )


def test_production_baseline_disables_reranker_fallback() -> None:
    record, sidecar, _ = _case(answerable=False)
    calls: list[dict] = []

    class SessionContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *args):
            return None

    class StrictRetriever:
        async def retrieve(self, session, question, **kwargs):
            calls.append(kwargs)
            return []

    async def resolve_scope(_record, _sidecar):
        document = sidecar.source_documents[0]
        return "synthetic-owner", {document.document_id: document.document_ref}

    async def verify_evidence(_record, _sidecar, _document_refs):
        return None

    runner = object.__new__(_SCRIPT.ProductionBaselineRunner)
    runner.sessionmaker = SessionContext
    runner.retriever = StrictRetriever()
    runner.top_k = 10
    runner._resolve_and_verify_scope = resolve_scope
    runner._verify_case_evidence = verify_evidence

    observation = asyncio.run(runner.run_case(record, sidecar))

    assert observation.answer == _SCRIPT._NO_RESULTS_ANSWER
    assert calls == [
        {
            "top_k": 10,
            "owner_sub": "synthetic-owner",
            "allow_rerank_fallback": False,
        }
    ]


def test_baseline_scores_retrieval_citations_quantities_and_latency() -> None:
    record, sidecar, chunk_id = _case()
    observation = BaselineObservation(
        case_id=record.case_id,
        gold_case_sha256=gold_record_case_sha256(record),
        scope_id=record.scope_id,
        answer="The required operating pressure is 42 bar [1].",
        retrieved=(
            RetrievedUnit(
                chunk_id=chunk_id,
                document_id=sidecar.source_documents[0].document_id,
            ),
        ),
        retrieval_ms=20.0,
        generation_ms=30.0,
        total_ms=50.0,
    )

    metrics = score_observation(record, sidecar, observation)
    assert metrics.answerability_correct
    assert metrics.ranked["1"].recall["value"] == 1.0
    assert metrics.ranked["1"].mrr["value"] == 1.0
    assert metrics.citation["citation_precision"] == 1.0
    assert metrics.citation["citation_recall"] == 1.0
    assert metrics.quantities["quantity_unit_accuracy"] == 1.0

    provenance = _SCRIPT._build_provenance(
        [record],
        mode="release",
        top_k=10,
        gold_artifact_sha256="a" * 64,
        sidecar_artifact_sha256="b" * 64,
        evaluated_at=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
        git_sha="c" * 40,
        git_dirty=False,
        runtime_corpus_snapshot_sha256="f" * 64,
        model_revisions=_model_revisions(),
    )
    report = aggregate_metrics([metrics], provenance=provenance)
    assert report.answerability_accuracy == 1.0
    assert report.latency_ms["total_p95"] == 50.0
    assert "answer" not in report.model_dump(mode="json")["cases"][0]
    assert report.provenance.runner == "retrieval_direct_answer"
    assert report.provenance.gold_artifact_sha256 == "a" * 64


def test_provenance_contains_only_reproducible_non_private_identifiers() -> None:
    record, _, _ = _case()
    provenance = _SCRIPT._build_provenance(
        [record],
        mode="release",
        top_k=10,
        gold_artifact_sha256="d" * 64,
        sidecar_artifact_sha256="e" * 64,
        evaluated_at=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
        git_sha=None,
        git_dirty=None,
        runtime_corpus_snapshot_sha256="f" * 64,
        model_revisions=_model_revisions(),
    )
    payload = provenance.model_dump(mode="json")
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["evaluation_mode"] == "release"
    assert payload["scope_count"] == 1
    assert payload["document_snapshot_count"] == 1
    assert payload["models"]["llm"]
    assert payload["configuration"]["answer_route"] == "doc_only"
    assert len(payload["configuration_sha256"]) == 64
    assert record.question not in serialized
    assert record.reference_answer not in serialized
    assert record.document_scope[0].document_ref not in serialized
    assert payload["runtime_corpus_snapshot_sha256"] == "f" * 64
    assert payload["model_revisions"]["llm"]["declared_revision"] == "synthetic-revision"


def test_runtime_model_revision_hashes_local_config_without_exposing_path(tmp_path: Path) -> None:
    model_root = tmp_path / "private-model-name"
    model_root.mkdir()
    config = model_root / "config.json"
    config.write_text(
        json.dumps({"architectures": ["SyntheticModel"], "_commit_hash": "a" * 40}),
        encoding="utf-8",
    )
    weight = model_root / "model.safetensors"
    weight.write_bytes(b"synthetic-weights-v1")
    metadata = {
        "id": "synthetic-model",
        "object": "model",
        "owned_by": "vllm",
        "root": str(model_root),
        "max_model_len": 16384,
    }

    first = _SCRIPT._runtime_model_revision(
        "synthetic-model",
        metadata,
        base_url="http://127.0.0.1:8006/v1",
        runtime_version_sha256="d" * 64,
    )
    serialized = first.model_dump_json()
    assert first.declared_revision == "a" * 40
    assert first.local_config_manifest_sha256 is not None
    assert first.weight_manifest_sha256 is not None
    assert first.weight_file_count == 1
    assert first.weight_bytes == len(b"synthetic-weights-v1")
    assert str(model_root) not in serialized

    config.write_text(
        json.dumps({"architectures": ["SyntheticModelV2"], "_commit_hash": "b" * 40}),
        encoding="utf-8",
    )
    second = _SCRIPT._runtime_model_revision(
        "synthetic-model",
        metadata,
        base_url="http://127.0.0.1:8006/v1",
        runtime_version_sha256="d" * 64,
    )
    assert second.local_config_manifest_sha256 != first.local_config_manifest_sha256
    assert second.weight_manifest_sha256 == first.weight_manifest_sha256
    assert second.declared_revision == "b" * 40

    weight.write_bytes(b"synthetic-weights-v2")
    third = _SCRIPT._runtime_model_revision(
        "synthetic-model",
        metadata,
        base_url="http://127.0.0.1:8006/v1",
        runtime_version_sha256="d" * 64,
    )
    assert third.weight_manifest_sha256 != second.weight_manifest_sha256


def test_baseline_case_seed_is_stable_and_case_specific() -> None:
    first = _SCRIPT._case_seed("ragq-case-one")
    assert first == _SCRIPT._case_seed("ragq-case-one")
    assert first != _SCRIPT._case_seed("ragq-case-two")
    assert 2026071300 <= first < 2027071300


def test_runtime_scope_digest_detects_reindex_state_changes() -> None:
    document = SimpleNamespace(
        id=uuid.UUID(int=1),
        s3_key_original="private/source.pdf",
        page_count=2,
        chunk_count=1,
        indexed_at=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
        updated_at=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
    )
    chunk = SimpleNamespace(
        id=uuid.UUID(int=2),
        document_id=document.id,
        idx=0,
        kind="section",
        heading_path="Private heading",
        page_start=0,
        page_end=0,
        text_en="Source text",
        body="Translated text",
        emb_en_text="[0.1,0.2]",
        emb_ru_text="[0.3,0.4]",
        meta={"bbox": [0, 0, 1, 1]},
    )

    first = _SCRIPT._runtime_scope_digest([document], [chunk], [])
    changed = SimpleNamespace(**{**vars(chunk), "emb_ru_text": "[0.5,0.6]"})
    second = _SCRIPT._runtime_scope_digest([document], [changed], [])
    assert first != second


def test_no_answer_requires_abstention_and_scope_binding() -> None:
    record, sidecar, _ = _case(answerable=False)
    observation = BaselineObservation(
        case_id=record.case_id,
        gold_case_sha256=gold_record_case_sha256(record),
        scope_id=record.scope_id,
        answer="Insufficient evidence in the selected documents.",
        retrieved=(),
        retrieval_ms=5.0,
        generation_ms=1.0,
        total_ms=6.0,
    )
    assert score_observation(record, sidecar, observation).answerability_correct

    mismatched = observation.model_copy(update={"scope_id": make_scope_id("foreign-owner")})
    with pytest.raises(ValueError, match="scope"):
        score_observation(record, sidecar, mismatched)
