from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import stat
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from rag_app.config import settings
from rag_app.db.models import SegmentKind
from rag_app.eval import parser_shadow_retrieval as shadow_retrieval
from rag_app.eval.gold_set import parsed_chunks_sha256
from rag_app.eval.parser_shadow_retrieval import (
    ControlCorpus,
    EvidenceLocator,
    RetrievalCase,
    ShadowChunk,
    ShadowRetrievalError,
    build_retrieval_cases,
    decide_candidate,
    evaluate_pair,
    load_control_corpus,
    load_parser_corpus,
    retrieval_metrics,
    source_evidence_manifest_sha256,
    stable_text_chunks,
    validate_local_retrieval_endpoints,
    validate_pair_linkage,
    write_report,
)
from rag_app.pipeline.segments import SegmentDraft, content_list_to_segments

_SOURCE_SHA = "a" * 64
_DOCUMENT_REF = f"doc-sha256:{_SOURCE_SHA}"


class _FakeEmbedder:
    def __init__(self) -> None:
        self.document_batches: list[list[str]] = []
        self.queries: list[str] = []

    async def embed(self, texts: list[str], batch: int | None = None) -> list[list[float]]:
        del batch
        self.document_batches.append(texts)
        return [[1.0, 0.0] if "needle" in text else [0.0, 1.0] for text in texts]

    async def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [1.0, 0.0]


class _FakeReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, texts))
        return [1.0 if "needle" in text else 0.0 for text in texts]


def _chunk(index: int, *, page: int, relevant: bool = False, variant: str = "b") -> ShadowChunk:
    return ShadowChunk(
        key=f"{variant}{index:063d}",
        document_ref=_DOCUMENT_REF,
        source_sha256=_SOURCE_SHA,
        index=index,
        kind="section",
        text=f"{'needle source' if relevant else 'noise'} {variant} {index}",
        page_start=page,
        page_end=page,
    )


def _case() -> RetrievalCase:
    return RetrievalCase(
        case_sha256="c" * 64,
        query="find the needle",
        answerable=True,
        language="en",
        hop_type="single",
        content_types=("text",),
        document_refs=frozenset({_DOCUMENT_REF}),
        locators=(
            EvidenceLocator(
                key="d" * 64,
                document_ref=_DOCUMENT_REF,
                page_start=2,
                page_end=2,
                grade=3,
                source_text="needle source",
                source_text_sha256=hashlib.sha256(b"needle source").hexdigest(),
                source_anchors=("needle source",),
            ),
        ),
    )


def test_stable_text_chunks_are_hard_bounded_and_preserve_pages() -> None:
    drafts = [
        SegmentDraft(0, SegmentKind.heading, "Section", 0, heading_level=1),
        SegmentDraft(1, SegmentKind.paragraph, "word " * 50, 1),
        SegmentDraft(2, SegmentKind.table, "cell " * 30, 2),
    ]

    chunks = stable_text_chunks(
        drafts,
        source_sha256=_SOURCE_SHA,
        page_count=3,
        max_chars=64,
    )

    assert chunks
    assert all(len(chunk.text) <= 64 for chunk in chunks)
    assert chunks[0].page_start == 0
    assert chunks[-1].kind == "table"
    assert chunks[-1].page_start == chunks[-1].page_end == 2
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_retrieval_metrics_use_document_and_page_intersection() -> None:
    locator = EvidenceLocator(
        key="e" * 64,
        document_ref=_DOCUMENT_REF,
        page_start=3,
        page_end=4,
        grade=3,
        source_text="needle source",
        source_text_sha256=hashlib.sha256(b"needle source").hexdigest(),
        source_anchors=("needle source",),
    )
    wrong_document = ShadowChunk(
        key="f" * 64,
        document_ref=f"doc-sha256:{'b' * 64}",
        source_sha256="b" * 64,
        index=0,
        kind="section",
        text="same page, wrong document",
        page_start=3,
        page_end=3,
    )
    matching = _chunk(1, page=4, relevant=True)

    metrics = retrieval_metrics(
        [wrong_document, matching],
        [wrong_document, matching],
        [locator],
        latency_ms=12.5,
    )

    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr_at_10"] == 0.5
    assert metrics["ndcg_at_10"] == pytest.approx(1 / 1.584962500721156)
    assert metrics["evidence_coverage"] == 1.0
    assert metrics["page_coverage"] == 0.5


def test_overlapping_source_anchors_survive_chunk_boundaries() -> None:
    source_text = " ".join(f"token-{index:02d}" for index in range(40))
    anchors = shadow_retrieval._source_anchors(source_text)
    locator = EvidenceLocator(
        key="a" * 64,
        document_ref=_DOCUMENT_REF,
        page_start=1,
        page_end=1,
        grade=3,
        source_text=source_text,
        source_text_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
        source_anchors=anchors,
    )
    boundary_piece = replace(_chunk(0, page=1), text=f"prefix {anchors[len(anchors) // 2]} suffix")

    metrics = retrieval_metrics([boundary_piece], [boundary_piece], [locator], latency_ms=0.0)

    assert len(anchors) > 1
    assert metrics["recall_at_1"] == 1.0


def test_source_evidence_manifest_changes_with_database_text() -> None:
    chunk_id = uuid.uuid4()

    first = source_evidence_manifest_sha256({chunk_id: "source evidence"})
    second = source_evidence_manifest_sha256({chunk_id: "mutated source evidence"})

    assert len(first) == 64
    assert first != second
    assert str(chunk_id) not in first


def test_public_report_allows_text_content_type_but_not_text_field() -> None:
    shadow_retrieval._assert_public_report(
        {"slices": {"content_type": {"text": {"count": 1}}}}
    )

    with pytest.raises(ShadowRetrievalError, match=r"forbidden private field \(text\)"):
        shadow_retrieval._assert_public_report({"result": {"text": "private"}})


def test_evaluate_pair_uses_one_embedder_and_reranker_for_both_variants() -> None:
    baseline = [
        _chunk(index, page=2 if index == 0 else index + 10, relevant=index == 0) for index in range(10)
    ]
    candidate = [
        _chunk(index, page=2 if index == 0 else index + 10, relevant=index == 0, variant="c")
        for index in range(10)
    ]
    embedder = _FakeEmbedder()
    reranker = _FakeReranker()

    result = asyncio.run(
        evaluate_pair(
            [_case()],
            baseline,
            candidate,
            embedder,
            reranker,
            dense_top_k=10,
            rerank_top_k=10,
        )
    )

    assert len(embedder.document_batches) == 2
    assert embedder.queries == ["find the needle"]
    assert len(reranker.calls) == 2
    assert result.ranked_cases[0].baseline["recall_at_1"] == 1.0
    assert result.ranked_cases[0].candidate["recall_at_1"] == 1.0


def test_decision_rejects_more_than_one_point_rare_slice_regression() -> None:
    base_metrics = {
        "recall_at_1": 0.8,
        "recall_at_5": 0.9,
        "recall_at_10": 0.95,
        "mrr_at_10": 0.85,
        "ndcg_at_10": 0.9,
        "evidence_coverage": 1.0,
        "page_coverage": 1.0,
        "page_coverage_at_10": 0.95,
        "latency_mean_ms": 10.0,
    }
    overall_delta = {key: 0.0 for key in base_metrics}
    rare_delta = {**overall_delta, "recall_at_5": -0.02}
    aggregates = {
        "answerable": {
            "count": 200,
            "baseline": base_metrics,
            "candidate": base_metrics,
            "delta": overall_delta,
        },
        "no_answer_probe": {
            "count": 67,
            "baseline": base_metrics,
            "candidate": base_metrics,
            "delta": overall_delta,
        },
        "slices": {
            "language": {
                "zh": {
                    "count": 5,
                    "baseline": base_metrics,
                    "candidate": base_metrics,
                    "delta": rare_delta,
                }
            }
        },
    }

    decision = decide_candidate(aggregates, max_regression=0.01)

    assert not decision["accepted"]
    assert decision["rare_slices"] == 1
    assert "slice_regression:language:zh:recall_at_5" in decision["failure_codes"]


def test_decision_rejects_no_answer_probe_regression() -> None:
    equal_delta = {
        "recall_at_5": 0.0,
        "ndcg_at_10": 0.0,
        "evidence_coverage": 0.0,
        "page_coverage": 0.0,
    }
    no_answer_delta = {**equal_delta, "recall_at_5": -0.02}
    viable_baseline = {"recall_at_10": 1.0, "evidence_coverage": 1.0}
    aggregates = {
        "answerable": {"count": 169, "baseline": viable_baseline, "delta": equal_delta},
        "no_answer_probe": {
            "count": 67,
            "baseline": viable_baseline,
            "delta": no_answer_delta,
        },
        "slices": {},
    }

    decision = decide_candidate(aggregates, max_regression=0.01)

    assert not decision["accepted"]
    assert decision["failure_codes"] == ["no_answer_probe_regression:recall_at_5"]


def test_decision_rejects_equal_zero_corpora() -> None:
    zero_metrics = {
        "recall_at_5": 0.0,
        "recall_at_10": 0.0,
        "ndcg_at_10": 0.0,
        "evidence_coverage": 0.0,
        "page_coverage": 0.0,
    }
    zero_delta = {key: 0.0 for key in zero_metrics}
    aggregates = {
        "answerable": {"count": 100, "baseline": zero_metrics, "delta": zero_delta},
        "no_answer_probe": {"count": 20, "baseline": zero_metrics, "delta": zero_delta},
        "slices": {},
    }

    decision = decide_candidate(aggregates)

    assert not decision["accepted"]
    assert decision["failure_codes"] == [
        "baseline_below_floor:answerable:evidence_coverage",
        "baseline_below_floor:answerable:recall_at_10",
        "baseline_below_floor:no_answer_probe:evidence_coverage",
        "baseline_below_floor:no_answer_probe:recall_at_10",
    ]


def test_write_report_is_atomic_mode_0600_and_never_replaces(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    output = private / "report.json"
    report = {
        "schema_version": "parser-shadow-retrieval-v1",
        "case_hashes": ["a" * 64],
        "aggregates": {"recall_at_5": 1.0},
    }

    digest = write_report(output, report)

    assert len(digest) == 64
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(output.read_text(encoding="utf-8"))["case_hashes"] == ["a" * 64]
    with pytest.raises(ShadowRetrievalError, match="unable to publish"):
        write_report(output, report)


def test_load_parser_corpus_requires_exactly_one_content_list(tmp_path: Path) -> None:
    source_bytes = b"private source PDF"
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    document_dir = tmp_path / "output" / "mineru" / source_sha256
    document_dir.mkdir(parents=True)
    (document_dir / "a_content_list.json").write_text("[]", encoding="utf-8")
    (document_dir / "b_content_list.json").write_text("[]", encoding="utf-8")
    pdf_root = tmp_path / "private"
    pdf_root.mkdir()
    (pdf_root / f"{source_sha256}.pdf").write_bytes(source_bytes)
    summary = _summary("3.3.1", {source_sha256: 1})

    with pytest.raises(ShadowRetrievalError, match="exactly one"):
        load_parser_corpus(
            tmp_path / "output",
            {source_sha256: 1},
            summary=summary,
            pdf_root=pdf_root,
            max_chars=64,
        )


def _summary(
    version: str,
    documents: dict[str, int],
    *,
    source_revision: str = "private-gold-snapshot-v1",
) -> dict[str, Any]:
    return {
        "benchmark_schema_version": 2,
        "source_revision": source_revision,
        "backends": ["mineru"],
        "run_label": f"mineru-{version}",
        "runtime_provenance": {
            "client": {"version": version},
            "server": {
                "mineru_version": version,
                "vllm_version": "0.21.0",
            },
            "model": {
                "snapshot_sha": "1" * 64,
                "manifest_sha256": "2" * 64,
            },
            "controlled": {
                "parser_backend": "vlm-http-client",
                "table_enable": False,
                "server_inference_args": ["--gpu-memory-utilization", "0.30"],
                "repetition_penalty": 1.1,
                "sampling_patch_sha256": "5" * 64,
            },
        },
        "results": {
            f"{source_sha256}.pdf": {
                "source_sha256": source_sha256,
                "mineru": {
                    "status": "ok",
                    "n_pages": page_count,
                    "raw_stats": {"text_sha256": "3" * 64},
                    "text_sha256": "4" * 64,
                },
            }
            for source_sha256, page_count in documents.items()
        },
    }


def test_benchmark_text_hash_includes_native_and_point_bboxes() -> None:
    draft = SegmentDraft(
        0,
        SegmentKind.paragraph,
        "bbox text",
        0,
        meta={"bbox": [1, 2, 30, 40], "bbox_pt": [0.1, 0.2, 3.0, 4.0]},
    )
    manifest = [
        {
            "bbox": [1, 2, 30, 40],
            "bbox_pt": [0.1, 0.2, 3.0, 4.0],
            "heading_level": None,
            "idx": 0,
            "kind": "paragraph",
            "page_idx": 0,
            "source_text": "bbox text",
            "table_cells": None,
        }
    ]
    expected = hashlib.sha256(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert shadow_retrieval._benchmark_text_sha256([draft]) == expected


def test_runtime_provenance_accepts_huggingface_40_char_revision() -> None:
    summary = _summary("3.4.4", {_SOURCE_SHA: 1})
    summary["runtime_provenance"]["model"]["snapshot_sha"] = "a" * 40

    provenance = shadow_retrieval._runtime_provenance(summary)

    assert provenance["model_snapshot_sha"] == "a" * 40


def _write_bound_output(
    output_root: Path,
    pdf_root: Path,
    text_value: str,
) -> tuple[str, dict[str, Any]]:
    source_bytes = b"private source PDF for binding"
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    pdf_root.mkdir(parents=True, exist_ok=True)
    (pdf_root / f"{source_sha256}.pdf").write_bytes(source_bytes)
    content_dir = output_root / "mineru" / source_sha256
    content_dir.mkdir(parents=True)
    content = [{"type": "text", "text": text_value, "page_idx": 0}]
    (content_dir / "document_content_list.json").write_text(
        json.dumps(content),
        encoding="utf-8",
    )
    text_digest = shadow_retrieval._benchmark_text_sha256(content_list_to_segments(content))
    summary = _summary("3.3.1", {source_sha256: 1})
    result = summary["results"][f"{source_sha256}.pdf"]["mineru"]
    result["raw_stats"]["text_sha256"] = text_digest
    result["text_sha256"] = text_digest
    return source_sha256, summary


def test_parser_corpus_rejects_swapped_or_modified_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_root = tmp_path / "private"
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    source_sha256, baseline_summary = _write_bound_output(baseline_root, pdf_root, "alpha")
    _, candidate_summary = _write_bound_output(candidate_root, pdf_root, "beta")
    monkeypatch.setattr(shadow_retrieval, "pdf_info", lambda _path: (1, False))

    corpus = load_parser_corpus(
        baseline_root,
        {source_sha256: 1},
        summary=baseline_summary,
        pdf_root=pdf_root,
        max_chars=64,
    )
    assert corpus.document_count == 1

    with pytest.raises(ShadowRetrievalError, match="raw text hash"):
        load_parser_corpus(
            candidate_root,
            {source_sha256: 1},
            summary=baseline_summary,
            pdf_root=pdf_root,
            max_chars=64,
        )

    content_path = next((baseline_root / "mineru" / source_sha256).glob("*_content_list.json"))
    content_path.write_text(
        json.dumps([{"type": "text", "text": "modified", "page_idx": 0}]),
        encoding="utf-8",
    )
    with pytest.raises(ShadowRetrievalError, match="raw text hash"):
        load_parser_corpus(
            baseline_root,
            {source_sha256: 1},
            summary=baseline_summary,
            pdf_root=pdf_root,
            max_chars=64,
        )

    assert candidate_summary != baseline_summary


def test_build_cases_uses_db_source_text_instead_of_translated_quote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_id = uuid.uuid4()
    source_text = "Source language evidence"
    translated_quote = "Переведенная цитата"
    evidence = SimpleNamespace(
        evidence_id="ev-1",
        chunk_id=chunk_id,
        document_ref=_DOCUMENT_REF,
        page=3,
        page_start=2,
        page_end=2,
        exact_quote=translated_quote,
    )
    record = SimpleNamespace(
        case_id="ragq-test0001",
        answerable=True,
        question="question",
        language="en",
        hop_type="single",
        content_types=("text",),
        document_scope=(SimpleNamespace(document_ref=_DOCUMENT_REF, page_count=3),),
        evidence=(SimpleNamespace(evidence_id="ev-1", relevance_grade=3),),
    )
    sidecar = SimpleNamespace(exact_evidence=(evidence,), retrieval_probe=())
    monkeypatch.setattr(shadow_retrieval, "gold_record_case_sha256", lambda _record: "c" * 64)

    cases = build_retrieval_cases(
        cast(Any, [record]),
        cast(Any, {record.case_id: sidecar}),
        {chunk_id: source_text},
    )

    locator = cases[0].locators[0]
    assert locator.source_text == source_text
    assert translated_quote not in locator.source_text
    wrong = replace(_chunk(0, page=2, relevant=False), text=translated_quote)
    right = replace(_chunk(1, page=2, relevant=False), text=f"prefix {source_text} suffix")
    wrong_metrics = retrieval_metrics([wrong], [wrong], [locator], latency_ms=0.0)
    right_metrics = retrieval_metrics([right], [right], [locator], latency_ms=0.0)
    assert wrong_metrics["recall_at_1"] == 0.0
    assert right_metrics["recall_at_1"] == 1.0


def test_no_answer_probe_requires_its_source_chunk_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_id = uuid.uuid4()
    probe = SimpleNamespace(
        chunk_id=chunk_id,
        document_ref=_DOCUMENT_REF,
        page=2,
        page_start=1,
        page_end=1,
    )
    record = SimpleNamespace(
        case_id="ragq-test0002",
        answerable=False,
        question="unanswerable question",
        language="en",
        hop_type="single",
        content_types=("text",),
        document_scope=(SimpleNamespace(document_ref=_DOCUMENT_REF, page_count=3),),
        evidence=(),
    )
    sidecar = SimpleNamespace(exact_evidence=(), retrieval_probe=(probe,))
    monkeypatch.setattr(shadow_retrieval, "gold_record_case_sha256", lambda _record: "d" * 64)

    case = build_retrieval_cases(
        cast(Any, [record]),
        cast(Any, {record.case_id: sidecar}),
        {chunk_id: "probe source text"},
    )[0]
    wrong = replace(_chunk(0, page=1), text="other text")
    right = replace(_chunk(1, page=1), text="probe source text")

    assert retrieval_metrics([wrong], [wrong], case.locators, latency_ms=0.0)["recall_at_1"] == 0.0
    assert retrieval_metrics([right], [right], case.locators, latency_ms=0.0)["recall_at_1"] == 1.0


def test_no_answer_without_retrieval_probe_is_not_scored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = SimpleNamespace(
        case_id="ragq-test0003",
        answerable=False,
        question="unbound negative question",
        language="en",
        hop_type="single",
        content_types=("text",),
        document_scope=(SimpleNamespace(document_ref=_DOCUMENT_REF, page_count=3),),
        evidence=(),
    )
    sidecar = SimpleNamespace(exact_evidence=(), retrieval_probe=())
    monkeypatch.setattr(shadow_retrieval, "gold_record_case_sha256", lambda _record: "e" * 64)

    assert build_retrieval_cases(
        cast(Any, [record]),
        cast(Any, {record.case_id: sidecar}),
        {},
    ) == []


def test_validate_pair_linkage_rejects_non_gold_parser_documents() -> None:
    extra_sha = "b" * 64
    baseline = _summary("3.3.1", {_SOURCE_SHA: 3, extra_sha: 1})
    candidate = _summary("3.4.4", {_SOURCE_SHA: 3, extra_sha: 1})
    fake_record = SimpleNamespace(document_scope=(SimpleNamespace(source_sha256=_SOURCE_SHA, page_count=3),))

    controls = ControlCorpus(
        chunks=(),
        documents={"c" * 64: 1},
        pdf_documents={_SOURCE_SHA: 3, extra_sha: 1},
        artifact_sha256="d" * 64,
        manifest_sha256="e" * 64,
        chunks_manifest_sha256="f" * 64,
        source_revision="private-gold-snapshot-v1",
    )

    with pytest.raises(ShadowRetrievalError, match="outside Gold"):
        validate_pair_linkage(
            baseline,
            candidate,
            cast(Any, [fake_record]),
            controls,
        )


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _control_fixture(
    tmp_path: Path,
    *,
    wrong_artifact_sha: bool = False,
    wrong_parsed_sha: bool = False,
) -> tuple[Path, list[Any], dict[str, int], dict[str, int]]:
    private = tmp_path / "corpus"
    private.mkdir(mode=0o700)
    pdf_pages = (9, 9, 9, 9, 9, 9, 7)
    control_pages = (28, 28, 27)
    pdf_documents = {f"{index + 1:064x}": pages for index, pages in enumerate(pdf_pages)}
    control_documents = {f"{index + 100:064x}": pages for index, pages in enumerate(control_pages)}
    controls: list[dict[str, Any]] = []
    snapshots: list[Any] = [
        SimpleNamespace(
            source_sha256=source_sha256,
            page_count=page_count,
            parsed_content_sha256="9" * 64,
        )
        for source_sha256, page_count in pdf_documents.items()
    ]
    for index, (source_sha256, page_count) in enumerate(control_documents.items()):
        chunks = [
            {
                "idx": 0,
                "kind": "section",
                "heading_path": "",
                "page_start": 0,
                "page_end": page_count - 1,
                "text": f"private control {index}",
            }
        ]
        parsed_sha256 = parsed_chunks_sha256(chunks)
        if wrong_parsed_sha and index == 0:
            parsed_sha256 = "0" * 64
        controls.append(
            {
                "source_sha256": source_sha256,
                "page_count": page_count,
                "parsed_content_sha256": parsed_sha256,
                "chunks": chunks,
            }
        )
        snapshots.append(
            SimpleNamespace(
                source_sha256=source_sha256,
                page_count=page_count,
                parsed_content_sha256=parsed_sha256,
            )
        )
    controls_path = private / "controls.json"
    controls_bytes = _canonical_json(
        {
            "schema_version": 1,
            "source": "private-rag-gold-ooxml-controls",
            "controls": controls,
        }
    )
    controls_path.write_bytes(controls_bytes)
    controls_path.chmod(0o600)
    controls_sha256 = hashlib.sha256(controls_bytes).hexdigest()
    manifest_sha256 = "f" * 64 if wrong_artifact_sha else controls_sha256
    pages: list[dict[str, Any]] = [
        {
            "file": f"{source_sha256}.pdf",
            "sha256": source_sha256,
            "category": "layout",
            "selection": {
                "document_ref": f"doc-sha256:{source_sha256}",
                "page_count": page_count,
            },
        }
        for source_sha256, page_count in pdf_documents.items()
    ]
    revision_payload = {
        "pdfs": [
            {
                "document_ref": page["selection"]["document_ref"],
                "page_count": page["selection"]["page_count"],
                "sha256": page["sha256"],
            }
            for page in pages
        ],
        "controls": {"sha256": controls_sha256, "count": 3},
    }
    source_revision = hashlib.sha256(_canonical_json(revision_payload)).hexdigest()
    manifest = {
        "manifest_version": 1,
        "source": "private-rag-gold-release",
        "source_revision": source_revision,
        "pages": pages,
        "controls": {
            "file": "controls.json",
            "sha256": manifest_sha256,
            "count": 3,
        },
    }
    manifest_path = private / "manifest.json"
    manifest_path.write_bytes(_canonical_json(manifest))
    manifest_path.chmod(0o600)
    return controls_path, [SimpleNamespace(document_scope=tuple(snapshots))], pdf_documents, control_documents


def test_control_artifact_validates_mode_sha_and_parsed_snapshot(tmp_path: Path) -> None:
    controls_path, records, pdf_documents, control_documents = _control_fixture(tmp_path)

    controls = load_control_corpus(controls_path, cast(Any, records))

    assert controls.pdf_documents == pdf_documents
    assert controls.documents == control_documents
    assert len(controls.chunks) == 3
    assert stat.S_IMODE(controls_path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("wrong_artifact_sha", "wrong_parsed_sha", "message"),
    [
        (True, False, "artifact SHA"),
        (False, True, "parsed_chunks_sha256"),
    ],
)
def test_control_artifact_rejects_hash_mismatch(
    tmp_path: Path,
    wrong_artifact_sha: bool,
    wrong_parsed_sha: bool,
    message: str,
) -> None:
    controls_path, records, _pdfs, _controls = _control_fixture(
        tmp_path,
        wrong_artifact_sha=wrong_artifact_sha,
        wrong_parsed_sha=wrong_parsed_sha,
    )

    with pytest.raises(ShadowRetrievalError, match=message):
        load_control_corpus(controls_path, cast(Any, records))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing a Gold document"),
        ("extra", "outside Gold"),
        ("moved", "moved between corpus partitions"),
        ("controlled", "controlled runtime field"),
    ],
)
def test_pair_linkage_rejects_missing_extra_or_moved_control(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    controls_path, records, pdf_documents, _control_documents = _control_fixture(tmp_path)
    controls = load_control_corpus(controls_path, cast(Any, records))
    if mutation == "missing":
        documents = dict(controls.documents)
        documents.pop(next(iter(documents)))
        controls = replace(controls, documents=documents)
    elif mutation == "extra":
        controls = replace(controls, documents={**controls.documents, "e" * 64: 1})
    elif mutation == "moved":
        moved = dict(controls.pdf_documents)
        moved.pop(next(iter(moved)))
        moved[next(iter(controls.documents))] = 9
        controls = replace(controls, pdf_documents=moved)
    baseline = _summary("3.3.1", pdf_documents, source_revision=controls.source_revision)
    candidate = _summary("3.4.4", pdf_documents, source_revision=controls.source_revision)
    if mutation == "controlled":
        candidate["runtime_provenance"]["controlled"]["table_enable"] = True

    with pytest.raises(ShadowRetrievalError, match=message):
        validate_pair_linkage(baseline, candidate, cast(Any, records), controls)


def test_identical_controls_compete_in_both_retrieval_candidates() -> None:
    parser_chunks = [
        _chunk(index, page=2 if index == 0 else index + 10, relevant=index == 0) for index in range(10)
    ]
    control_sha = "b" * 64
    control = ShadowChunk(
        key="0" * 64,
        document_ref=f"doc-sha256:{control_sha}",
        source_sha256=control_sha,
        index=0,
        kind="section",
        text="needle control",
        page_start=0,
        page_end=0,
    )
    base_case = _case()
    case = replace(
        base_case,
        document_refs=frozenset({*base_case.document_refs, control.document_ref}),
    )

    without_control = asyncio.run(
        evaluate_pair(
            [case],
            parser_chunks,
            parser_chunks,
            _FakeEmbedder(),
            _FakeReranker(),
            dense_top_k=10,
            rerank_top_k=10,
        )
    )
    with_control = asyncio.run(
        evaluate_pair(
            [case],
            [*parser_chunks, control],
            [*parser_chunks, control],
            _FakeEmbedder(),
            _FakeReranker(),
            dense_top_k=10,
            rerank_top_k=10,
        )
    )

    assert without_control.ranked_cases[0].baseline["recall_at_1"] == 1.0
    assert with_control.ranked_cases[0].baseline["recall_at_1"] == 0.0
    assert (
        with_control.ranked_cases[0].baseline["recall_at_1"]
        == with_control.ranked_cases[0].candidate["recall_at_1"]
    )


def test_write_report_rejects_uuid_or_s3_leakage(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    report = {
        "schema_version": "parser-shadow-retrieval-v1",
        "case_hashes": ["a" * 64],
        "aggregates": {"note": "s3://private-bucket/object"},
    }

    with pytest.raises(ShadowRetrievalError, match="private identifier"):
        write_report(private / "report.json", report)


def test_private_queries_require_loopback_model_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embed_base_url", "https://external.example/v1")

    with pytest.raises(ShadowRetrievalError, match="loopback-only"):
        validate_local_retrieval_endpoints()


def test_source_evidence_resolver_is_exact_read_only_and_disposes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = Path(__file__).parents[1] / "scripts" / "evaluate_parser_shadow_retrieval.py"
    spec = importlib.util.spec_from_file_location("evaluate_parser_shadow_retrieval_source", script)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    chunk_id = uuid.uuid4()
    document_id = uuid.uuid4()
    statements: list[str] = []

    class _Result:
        def all(self) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    id=chunk_id,
                    document_id=document_id,
                    page_start=2,
                    page_end=2,
                    text_en="source evidence",
                )
            ]

    class _Context:
        def __init__(self, value: object) -> None:
            self.value = value

        async def __aenter__(self) -> object:
            return self.value

        async def __aexit__(self, *args: object) -> None:
            del args

    class _Connection:
        def begin(self) -> _Context:
            return _Context(self)

        async def execute(self, statement: object) -> _Result:
            statements.append(str(statement))
            return _Result()

    class _Engine:
        def __init__(self) -> None:
            self.disposed = False

        def connect(self) -> _Context:
            return _Context(_Connection())

        async def dispose(self) -> None:
            self.disposed = True

    engine = _Engine()
    monkeypatch.setattr(runner, "create_engine", lambda: engine)
    locator = SimpleNamespace(
        chunk_id=chunk_id,
        document_id=document_id,
        page_start=2,
        page_end=2,
    )
    sidecar = SimpleNamespace(stratum="single_hop", exact_evidence=(locator,), retrieval_probe=())

    result = asyncio.run(runner._load_source_text_by_chunk_id({"case": sidecar}))

    assert result == {chunk_id: "source evidence"}
    assert "REPEATABLE READ, READ ONLY" in statements[0]
    assert engine.disposed


def test_cli_closes_embedder_client_when_evaluation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = Path(__file__).parents[1] / "scripts" / "evaluate_parser_shadow_retrieval.py"
    spec = importlib.util.spec_from_file_location("evaluate_parser_shadow_retrieval", script)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    closed = False

    class _Client:
        async def close(self) -> None:
            nonlocal closed
            closed = True

    class _Embedder:
        def __init__(self) -> None:
            self.client = _Client()

    async def fail_evaluation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("expected failure")

    async def load_source_text(*args: object, **kwargs: object) -> dict[uuid.UUID, str]:
        del args, kwargs
        return {uuid.uuid4(): "source evidence"}

    monkeypatch.setattr(runner, "validate_local_retrieval_endpoints", lambda: None)
    monkeypatch.setattr(
        runner,
        "load_benchmark_summary",
        lambda path: ({}, "a" * 64 if path.name == "baseline.json" else "b" * 64),
    )
    monkeypatch.setattr(runner, "_sha256_file", lambda path: "c" * 64)
    monkeypatch.setattr(runner, "load_gold_set", lambda *args, **kwargs: ([object()] * 236, None))
    monkeypatch.setattr(runner, "load_private_sidecar", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner, "bind_gold_sidecar", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_load_source_text_by_chunk_id", load_source_text)
    monkeypatch.setattr(
        runner,
        "load_control_corpus",
        lambda *args, **kwargs: SimpleNamespace(chunks=()),
    )
    monkeypatch.setattr(runner, "validate_pair_linkage", lambda *args, **kwargs: ({}, {}, {}, {}))
    monkeypatch.setattr(
        runner,
        "load_parser_corpus",
        lambda *args, **kwargs: SimpleNamespace(chunks=()),
    )
    monkeypatch.setattr(runner, "build_retrieval_cases", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner, "evaluate_pair", fail_evaluation)
    monkeypatch.setattr(runner, "Embedder", _Embedder)
    monkeypatch.setattr(runner, "Reranker", object)
    args = SimpleNamespace(
        baseline_report=tmp_path / "baseline.json",
        candidate_report=tmp_path / "candidate.json",
        baseline_output=tmp_path / "baseline",
        candidate_output=tmp_path / "candidate",
        gold=tmp_path / "gold.jsonl",
        sidecar=tmp_path / "sidecar.jsonl",
        controls=tmp_path / "controls.json",
        output=tmp_path / "report.json",
        gold_mode="release",
        dense_top_k=20,
        rerank_top_k=10,
    )

    with pytest.raises(RuntimeError, match="expected failure"):
        asyncio.run(runner._run(args))

    assert closed
