from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from rag_app.db.rls import current_principal
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
from rag_app.eval.private_sidecar import PrivateSidecarRecord
from rag_app.rag.retrieve import RetrievalTrace, RetrievedChunk


def _script_module():
    path = Path(__file__).parents[1] / "scripts" / "evaluate_retrieval_bm25.py"
    spec = importlib.util.spec_from_file_location("evaluate_retrieval_bm25", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _script_module()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _case(
    index: int,
    *,
    owner: str,
    answerable: bool = True,
    include_probe: bool = False,
) -> tuple[GoldRecord, PrivateSidecarRecord, uuid.UUID, uuid.UUID]:
    source_sha = _sha(f"source:{owner}")
    document_ref = make_document_ref(source_sha)
    document_id = uuid.uuid5(uuid.NAMESPACE_URL, f"document:{owner}")
    chunk_id = uuid.uuid5(uuid.NAMESPACE_URL, f"chunk:{owner}:{index}")
    probe_id = uuid.uuid5(uuid.NAMESPACE_URL, f"probe:{owner}:{index}")
    quote = f"Private exact evidence for case {index}."
    content_sha = text_sha256(quote)
    evidence_id = make_evidence_id(source_sha, 1, "text", content_sha)
    question = f"What exact value is specified for test case number {index}?"
    answer = f"The exact test value is {index}." if answerable else None
    evidence = (
        (
            EvidenceRef(
                evidence_id=evidence_id,
                document_ref=document_ref,
                page=1,
                content_type="text",
                content_sha256=content_sha,
                relevance_grade=3,
                bbox=None,
            ),
        )
        if answerable
        else ()
    )
    record = GoldRecord(
        schema_version="rag-gold-v1",
        case_id=f"ragq-test-bm25-{index:04d}",
        status="candidate",
        scope_id=make_scope_id(owner),
        language="en",
        question=question,
        question_sha256=text_sha256(question),
        answerable=answerable,
        reference_answer=answer,
        reference_answer_sha256=text_sha256(answer) if answer else None,
        hop_type="single",
        content_types=("text",),
        challenge_tags=("numbers",) if answerable else (),
        document_scope=(
            DocumentSnapshot(
                document_ref=document_ref,
                source_sha256=source_sha,
                parsed_content_sha256=_sha(f"parsed:{owner}"),
                page_count=1,
            ),
        ),
        evidence=evidence,
        review=None,
    )
    sidecar = PrivateSidecarRecord.model_validate(
        {
            "schema_version": "private-rag-generator-v1",
            "case_id": record.case_id,
            "gold_case_sha256": gold_record_case_sha256(record),
            "scope_id": record.scope_id,
            "stratum": "single_hop" if answerable else "no_answer",
            "language": "en",
            "source_documents": (
                {
                    "document_id": document_id,
                    "document_ref": document_ref,
                    "source_lang": "en",
                },
            ),
            "classification": {
                "content_types": ("text",),
                "challenge_tags": ("numbers",) if answerable else (),
                "has_numbers": answerable,
                "has_units": False,
                "has_standards": False,
            },
            "generation": {"model": "synthetic", "seed": index},
            "exact_evidence": (
                (
                    {
                        "evidence_id": evidence_id,
                        "document_id": document_id,
                        "document_ref": document_ref,
                        "chunk_id": chunk_id,
                        "chunk_index": index,
                        "kind": "section",
                        "heading_path": "Synthetic",
                        "page": 1,
                        "page_start": 0,
                        "page_end": 0,
                        "text_sha256": text_sha256(quote),
                        "content_sha256": content_sha,
                        "exact_quote": quote,
                        "retrieval_score": None,
                    },
                )
                if answerable
                else ()
            ),
            "retrieval_probe": (
                (
                    {
                        "document_id": document_id,
                        "document_ref": document_ref,
                        "chunk_id": probe_id,
                        "page": 1,
                        "page_start": 0,
                        "page_end": 0,
                        "content_sha256": _sha("probe body"),
                        "retrieval_score": 1.0,
                    },
                )
                if include_probe
                else ()
            ),
            "quantities": {"expected": (), "supported": ()},
            "validation": {
                "answer_supported": answerable,
                "question_unambiguous": True,
                "uses_all_evidence": answerable,
            },
        },
        strict=True,
    )
    return record, sidecar, chunk_id, probe_id


def _config(*, threshold: float = 0.1) -> Any:
    return runner.RetrievalConfig(
        dense_top_k=10,
        sparse_top_k=10,
        rrf_k=60,
        rerank_top_k=10,
        rerank_min_score=threshold,
        final_top_k=10,
    )


def test_exact_evidence_is_the_only_relevance_source() -> None:
    record, sidecar, exact_id, probe_id = _case(
        1,
        owner="owner-a",
        include_probe=True,
    )
    binding = runner.build_case_bindings([record], {record.case_id: sidecar})[record.case_id]

    assert binding.relevance == {exact_id: 3}
    assert probe_id not in binding.relevance
    metrics = runner.score_ranking([probe_id], binding.relevance, answerable=True)
    assert metrics.recall["5"] == 0.0


def test_retrieval_no_answer_count_includes_leakage_probes() -> None:
    regular = _case(1, owner="owner-a", answerable=False)[0]
    leakage = _case(2, owner="owner-a", answerable=False)[0].model_copy(
        update={"challenge_tags": ("leakage",)}
    )

    assert runner._retrieval_no_answer_count((regular, leakage)) == 2


def test_split_is_deterministic_and_keeps_document_clusters_disjoint() -> None:
    records = [
        _case(index, owner=f"document-{index}")[0].model_copy(
            update={"scope_id": make_scope_id(f"owner-{index % 2}")}
        )
        for index in range(1, 5)
    ]

    first = runner.stratified_cluster_split(records, seed=17, locked_fraction=0.5)
    second = runner.stratified_cluster_split(records, seed=17, locked_fraction=0.5)

    assert first == second
    assert set(first.tuning_cluster_ids).isdisjoint(first.locked_cluster_ids)
    by_id = {record.case_id: record for record in records}
    tuning_documents = {
        snapshot.document_ref
        for case_id in first.tuning_case_ids
        for snapshot in by_id[case_id].document_scope
    }
    locked_documents = {
        snapshot.document_ref
        for case_id in first.locked_case_ids
        for snapshot in by_id[case_id].document_scope
    }
    tuning_scopes = {by_id[case_id].scope_id for case_id in first.tuning_case_ids}
    locked_scopes = {by_id[case_id].scope_id for case_id in first.locked_case_ids}
    assert tuning_documents.isdisjoint(locked_documents)
    assert tuning_scopes == locked_scopes == {make_scope_id("owner-0"), make_scope_id("owner-1")}


def test_split_fails_when_only_one_independent_cluster_exists() -> None:
    records = [_case(1, owner="owner-a")[0], _case(2, owner="owner-a")[0]]
    with pytest.raises(runner.RetrievalEvaluationError, match="two independent"):
        runner.stratified_cluster_split(records, seed=17, locked_fraction=0.5)


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def begin(self):
        return self


class _FakeRetriever:
    def __init__(
        self,
        chunk: RetrievedChunk,
        *,
        flip: bool = False,
        reorder_chunk: RetrievedChunk | None = None,
        swap_boundary: bool = False,
        change_candidates: bool = False,
        duplicate_candidate: bool = False,
        irrelevant_jitter: bool = False,
        never_final_chunk: RetrievedChunk | None = None,
        minor_reorder_chunks: tuple[RetrievedChunk, ...] = (),
        divergent_reorder: bool = False,
    ) -> None:
        self.chunk = chunk
        self.flip = flip
        self.reorder_chunk = reorder_chunk
        self.swap_boundary = swap_boundary
        self.change_candidates = change_candidates
        self.duplicate_candidate = duplicate_candidate
        self.irrelevant_jitter = irrelevant_jitter
        self.never_final_chunk = never_final_chunk
        self.minor_reorder_chunks = minor_reorder_chunks
        self.divergent_reorder = divergent_reorder
        self.calls = 0

    async def retrieve_with_trace(self, session, query, **kwargs):
        del session, query
        assert current_principal().user_sub == "owner-a"
        self.calls += 1
        rows = (self.chunk,) if self.reorder_chunk is None else (self.chunk, self.reorder_chunk)
        rows = (*rows, *self.minor_reorder_chunks)
        if self.never_final_chunk is not None:
            rows = (*rows, self.never_final_chunk)
        final = () if self.flip and self.calls % 2 == 0 else rows
        reranked = rows
        if self.reorder_chunk is not None and self.calls % 2 == 0:
            reranked = tuple(reversed(rows))
            final = reranked
        if self.minor_reorder_chunks and self.calls % 2 == 0:
            reranked = (
                (*rows[1:9], rows[0], rows[9]) if self.divergent_reorder else (*rows[:-2], rows[-1], rows[-2])
            )
            final = reranked
        if self.swap_boundary:
            first = self.calls % 2
            rows[first].score = 0.5001
            rows[(first + 1) % 2].score = 0.5
            reranked = (rows[first], rows[(first + 1) % 2])
            final = reranked[:1]
        if self.change_candidates:
            final = (rows[0],)
            reranked = rows if self.calls % 2 else (rows[0],)
        if self.duplicate_candidate:
            final = (rows[0],)
            reranked = (rows[0], rows[0])
        if self.irrelevant_jitter:
            rows[0].score = 1.0
            rows[1].score = 0.9 if self.calls % 2 else 0.1
            reranked = rows
            final = (rows[0],)
        if self.never_final_chunk is not None:
            rows[0].score = 0.505 if self.calls % 2 else 0.495
            rows[1].score = 0.495 if self.calls % 2 else 0.505
            rows[2].score = 0.5
            reranked = (rows[0], rows[2], rows[1]) if self.calls % 2 else (rows[1], rows[2], rows[0])
            final = reranked[:1]
        return RetrievalTrace(
            requested_sparse_backend=kwargs["sparse_backend"],
            sparse_engine=(
                "postgres_fts" if kwargs["sparse_backend"] == "postgres_fts" else "pg_textsearch_en"
            ),
            dense=rows,
            sparse=rows,
            hybrid_pre_rerank=rows,
            reranked=reranked,
            final=final,
            stage_latency_ms={
                "embedding": 1.0,
                "dense_sql": 2.0,
                "sparse_sql": 3.0,
                "fusion": 0.5,
                "rerank": 4.0,
                "visual": 0.0,
                "total": 10.5,
            },
            reranker_fallback=False,
        )


class _LoadRetriever(_FakeRetriever):
    def __init__(
        self,
        chunk: RetrievedChunk,
        *,
        reranker_tracker: Any | None = None,
        fail_call: int | None = None,
    ) -> None:
        super().__init__(chunk)
        self.active = 0
        self.max_active = 0
        self.reranker_tracker = reranker_tracker
        self.fail_call = fail_call

    async def retrieve_with_trace(self, session, query, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.005)
            if self.fail_call is not None and self.calls + 1 == self.fail_call:
                self.calls += 1
                raise RuntimeError("synthetic load failure")
            trace = await super().retrieve_with_trace(session, query, **kwargs)
            if self.reranker_tracker is not None:
                await self.reranker_tracker.rerank(
                    query,
                    [item.text_ru or item.text_en for item in trace.hybrid_pre_rerank],
                )
            return trace
        finally:
            self.active -= 1


def _execute_artifact(
    *,
    index: int = 1,
    answerable: bool = True,
    flip: bool = False,
    reorder: bool = False,
    swap_boundary: bool = False,
    change_candidates: bool = False,
    duplicate_candidate: bool = False,
    irrelevant_jitter: bool = False,
    never_final_tie: bool = False,
    minor_reorder: bool = False,
    divergent_reorder: bool = False,
):
    record, sidecar, chunk_id, _ = _case(index, owner="owner-a", answerable=answerable)
    binding = runner.build_case_bindings([record], {record.case_id: sidecar})[record.case_id]
    document_id = sidecar.source_documents[0].document_id
    chunk = RetrievedChunk(
        id=chunk_id,
        document_id=document_id,
        filename="private.pdf",
        heading_path="Synthetic",
        kind="section",
        page_start=0,
        page_end=0,
        text_en="private body",
        text_ru="",
        meta={},
    )
    scope = runner._VerifiedScope(
        owner_sub="owner-a",
        document_refs={document_id: sidecar.source_documents[0].document_ref},
        document_ids=frozenset({document_id}),
        evidence=runner.ScopeEvidence(
            scope_id=record.scope_id,
            case_count=1,
            document_count=1,
            chunk_count=1,
            corpus_sha256="a" * 64,
        ),
    )
    reorder_chunk = (
        RetrievedChunk(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"chunk:owner-a:{index}:rerank-tie"),
            document_id=document_id,
            filename="private.pdf",
            heading_path="Synthetic",
            kind="section",
            page_start=0,
            page_end=0,
            text_en="private alternate body",
            text_ru="",
            meta={},
        )
        if reorder
        or swap_boundary
        or change_candidates
        or duplicate_candidate
        or irrelevant_jitter
        or never_final_tie
        else None
    )
    never_final_chunk = (
        RetrievedChunk(
            id=uuid.UUID(int=1),
            document_id=document_id,
            filename="private.pdf",
            heading_path="Synthetic",
            kind="section",
            page_start=0,
            page_end=0,
            text_en="private never-final body",
            text_ru="",
            meta={},
        )
        if never_final_tie
        else None
    )
    minor_reorder_chunks = (
        tuple(
            RetrievedChunk(
                id=uuid.uuid5(uuid.NAMESPACE_URL, f"chunk:owner-a:{index}:minor-reorder:{position}"),
                document_id=document_id,
                filename="private.pdf",
                heading_path="Synthetic",
                kind="section",
                page_start=0,
                page_end=0,
                text_en=f"private alternate body {position}",
                text_ru="",
                meta={},
            )
            for position in range(9)
        )
        if minor_reorder or divergent_reorder
        else ()
    )
    return (
        asyncio.run(
            runner._execute_case(
                retriever=_FakeRetriever(
                    chunk,
                    flip=flip,
                    reorder_chunk=reorder_chunk,
                    swap_boundary=swap_boundary,
                    change_candidates=change_candidates,
                    duplicate_candidate=duplicate_candidate,
                    irrelevant_jitter=irrelevant_jitter,
                    never_final_chunk=never_final_chunk,
                    minor_reorder_chunks=minor_reorder_chunks,
                    divergent_reorder=divergent_reorder,
                ),
                sessionmaker=lambda: _Session(),
                binding=binding,
                scope=scope,
                config=_config(),
                backend="pg_textsearch",
                variant="candidate",
                split="tuning",
                run_id="b" * 64,
                repeat_count=2,
                query_embedding_sha256="e" * 64,
            )
        ),
        record,
        sidecar,
    )


def test_case_execution_sets_principal_and_serializes_no_content() -> None:
    artifact, record, sidecar = _execute_artifact()
    raw = artifact.model_dump_json()

    assert artifact.observation.deterministic is True
    assert artifact.observation.returned_count == 1
    assert artifact.observation.abstained is False
    assert record.question not in raw
    assert sidecar.exact_evidence[0].exact_quote not in raw
    assert "private body" not in raw


def test_case_execution_fails_on_nondeterministic_order() -> None:
    with pytest.raises(runner.RetrievalEvaluationError, match="not deterministic"):
        _execute_artifact(flip=True)


def test_case_execution_rejects_low_agreement_reranker_consensus() -> None:
    with pytest.raises(runner.RetrievalEvaluationError, match="min_pairwise_rank_agreement=0.000000"):
        _execute_artifact(reorder=True)


def test_case_execution_applies_high_agreement_borda_consensus() -> None:
    artifact, _, _ = _execute_artifact(minor_reorder=True)

    assert artifact.observation.deterministic is True
    assert artifact.observation.reranker_consensus_applied is True
    assert artifact.observation.reranker_consensus_method == "borda-rank-v1"
    assert artifact.observation.reranker_min_pairwise_rank_agreement == pytest.approx(44 / 45)
    assert artifact.observation.reranker_max_score_delta == 0.0
    assert len(set(artifact.observation.repeat_order_sha256)) == 2


def test_case_execution_rejects_midpoint_borda_false_pass() -> None:
    with pytest.raises(runner.RetrievalEvaluationError, match="min_pairwise_rank_agreement=0.822222"):
        _execute_artifact(divergent_reorder=True)


def test_case_execution_applies_consensus_at_top_k_boundary() -> None:
    artifact, _, _ = _execute_artifact(swap_boundary=True)

    assert artifact.observation.deterministic is True
    assert artifact.observation.reranker_consensus_applied is True
    assert artifact.observation.reranker_consensus_method == "mean-score-v1"
    assert artifact.observation.reranker_min_pairwise_rank_agreement == 0.0
    assert artifact.observation.returned_count == 1
    assert len(set(artifact.observation.repeat_order_sha256)) == 2


def test_case_execution_consensus_cannot_introduce_never_final_candidate() -> None:
    artifact, _, _ = _execute_artifact(never_final_tie=True)

    final_ids = artifact.observation.pools["final"].ranked_chunk_ids
    assert artifact.observation.reranker_consensus_applied is True
    assert artifact.observation.reranker_consensus_method == "mean-score-v1"
    assert uuid.UUID(int=1) not in final_ids


def test_case_execution_rejects_changed_reranker_candidates() -> None:
    with pytest.raises(runner.RetrievalEvaluationError, match="candidate universe changed"):
        _execute_artifact(change_candidates=True)


def test_case_execution_rejects_duplicate_reranker_candidates() -> None:
    with pytest.raises(runner.RetrievalEvaluationError, match="duplicate chunk IDs"):
        _execute_artifact(duplicate_candidate=True)


def test_case_execution_excludes_irrelevant_score_jitter_from_output_delta() -> None:
    artifact, _, _ = _execute_artifact(irrelevant_jitter=True)

    assert artifact.observation.deterministic is True
    assert artifact.observation.reranker_max_score_delta == 0.0
    assert artifact.observation.reranker_all_max_score_delta == pytest.approx(0.8)


class _CountingEmbedder(runner.Embedder):
    def __init__(self) -> None:
        self.query_calls = 0

    async def embed_query(self, query: str) -> list[float]:
        del query
        self.query_calls += 1
        return [0.25] * runner.settings.embed_dim


class _StableReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, tuple(texts)))
        return [0.9 if text[:4000].endswith("a") else 0.8 for text in texts]


class _AlternatingReranker:
    def __init__(self) -> None:
        self.calls = 0

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        del query
        self.calls += 1
        scores = [0.9, 0.1]
        return scores if self.calls % 2 else list(reversed(scores))


class _InvalidReranker:
    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        del query
        return [float("nan")] * len(texts)


def _reranker_revision() -> Any:
    return runner.ModelEndpointRevision(
        model=runner.settings.rerank_model,
        declared_revision="test-revision",
        endpoint_metadata_sha256="a" * 64,
        runtime_version_sha256="b" * 64,
        weight_manifest_sha256="c" * 64,
        model_config_sha256="d" * 64,
    )


def _collected_rerank_set(*, query: str = "private query") -> Any:
    return runner._CollectedRerankSet(
        case_id="ragq-test-bm25-0001",
        split="tuning",
        query=query,
        config_sha256="e" * 64,
        backend="pg_textsearch",
        rerank_min_score=0.1,
        final_top_k=10,
        candidates=((uuid.UUID(int=1), "private a"), (uuid.UUID(int=2), "private b")),
    )


def _collected_rerank_sets(*, query: str = "private query") -> tuple[Any, Any]:
    candidate = _collected_rerank_set(query=query)
    baseline = runner._CollectedRerankSet(
        case_id=candidate.case_id,
        split=candidate.split,
        query=candidate.query,
        config_sha256=candidate.config_sha256,
        backend="postgres_fts",
        rerank_min_score=candidate.rerank_min_score,
        final_top_k=candidate.final_top_k,
        candidates=candidate.candidates,
    )
    return baseline, candidate


def test_paired_reranker_canonical_prewarms_then_freezes() -> None:
    delegate = _StableReranker()
    reranker = runner._PairedReranker(delegate)
    evidence = asyncio.run(
        reranker.preload(
            _collected_rerank_sets(),
            questions=["private query"],
            revision=_reranker_revision(),
            replay_count=3,
        )
    )
    calls_after_prewarm = len(delegate.calls)

    first = asyncio.run(reranker.rerank("private query", ["private a", "private b"]))
    second = asyncio.run(reranker.rerank("private query", ["private b", "private a"]))

    assert calls_after_prewarm == 5
    assert len(delegate.calls) == calls_after_prewarm
    assert first == list(reversed(second))
    assert evidence.unique_pair_count == 2
    assert evidence.live_pair_score_count == 6
    assert evidence.min_split_half_rank_agreement == 1.0
    assert evidence.min_same_set_single_replay_rank_agreement == 1.0
    assert evidence.min_single_replay_common_rank_agreement == 1.0
    assert evidence.min_single_replay_set_jaccard == 1.0
    assert evidence.nonempty_candidate_set_count == 2
    assert evidence.single_replay_set_comparison_count == 6
    assert evidence.single_replay_set_mismatch_count == 0
    assert len(evidence.stability_slices) == 2
    assert evidence.batch_shape_sample_count == 2
    assert evidence.min_batch_shape_rank_agreement == 1.0
    assert "private" not in evidence.model_dump_json()
    assert reranker.cache_stats() == (4, 0)


def test_paired_reranker_frozen_miss_fails_without_live_call() -> None:
    delegate = _StableReranker()
    reranker = runner._PairedReranker(delegate)
    asyncio.run(
        reranker.preload(
            _collected_rerank_sets(),
            questions=["private query"],
            revision=_reranker_revision(),
            replay_count=3,
        )
    )
    calls_after_prewarm = len(delegate.calls)

    with pytest.raises(runner.RetrievalEvaluationError, match="missed a measured pair"):
        asyncio.run(reranker.rerank("private query", ["private unseen"]))

    assert len(delegate.calls) == calls_after_prewarm
    assert reranker.cache_stats() == (0, 1)


def test_paired_reranker_rejects_unstable_split_half_ranking() -> None:
    reranker = runner._PairedReranker(_AlternatingReranker())

    with pytest.raises(runner.RetrievalEvaluationError, match="rank agreement"):
        asyncio.run(
            reranker.preload(
                _collected_rerank_sets(),
                questions=["private query"],
                revision=_reranker_revision(),
                replay_count=3,
            )
        )


def test_reranker_evidence_caps_rare_single_replay_set_mismatches() -> None:
    delegate = _StableReranker()
    reranker = runner._PairedReranker(delegate)
    evidence = asyncio.run(
        reranker.preload(
            _collected_rerank_sets(),
            questions=["private query"],
            revision=_reranker_revision(),
            replay_count=3,
        )
    )
    payload = evidence.model_dump(mode="python")
    expanded_slice = {
        **payload["stability_slices"][0],
        "nonempty_candidate_set_count": 100,
        "single_replay_set_comparison_count": 300,
        "single_replay_set_mismatch_count": 3,
        "min_single_replay_set_jaccard": 0.8,
    }
    payload["stability_slices"] = (expanded_slice, payload["stability_slices"][1])
    payload.update(
        candidate_set_count=101,
        nonempty_candidate_set_count=101,
        min_single_replay_set_jaccard=0.8,
        single_replay_set_comparison_count=303,
        single_replay_set_mismatch_count=3,
    )

    accepted = runner.RerankerScoreEvidence.model_validate(payload, strict=True)
    assert accepted.single_replay_set_mismatch_count == 3

    with pytest.raises(ValueError, match="mismatch ratio"):
        excessive_slice = {
            **expanded_slice,
            "single_replay_set_mismatch_count": 4,
        }
        runner.RerankerScoreEvidence.model_validate(
            {
                **payload,
                "single_replay_set_mismatch_count": 4,
                "stability_slices": (excessive_slice, payload["stability_slices"][1]),
            },
            strict=True,
        )
    with pytest.raises(ValueError, match="coverage"):
        runner.RerankerScoreEvidence.model_validate(
            {**payload, "single_replay_set_comparison_count": 304},
            strict=True,
        )
    with pytest.raises(ValueError, match="overlap"):
        low_overlap_slice = {
            **expanded_slice,
            "min_single_replay_set_jaccard": 0.79,
        }
        runner.RerankerScoreEvidence.model_validate(
            {
                **payload,
                "min_single_replay_set_jaccard": 0.79,
                "stability_slices": (low_overlap_slice, payload["stability_slices"][1]),
            },
            strict=True,
        )

    unstable_slice = {
        **evidence.model_dump(mode="python")["stability_slices"][0],
        "single_replay_set_mismatch_count": 1,
        "min_single_replay_set_jaccard": 0.8,
    }
    dilution_slice = {
        **evidence.model_dump(mode="python")["stability_slices"][1],
        "nonempty_candidate_set_count": 100,
        "single_replay_set_comparison_count": 300,
    }
    diluted_payload = {
        **evidence.model_dump(mode="python"),
        "candidate_set_count": 101,
        "nonempty_candidate_set_count": 101,
        "single_replay_set_comparison_count": 303,
        "single_replay_set_mismatch_count": 1,
        "min_single_replay_set_jaccard": 0.8,
        "stability_slices": (unstable_slice, dilution_slice),
    }
    with pytest.raises(ValueError, match="stability-slice mismatch ratio"):
        runner.RerankerScoreEvidence.model_validate(diluted_payload, strict=True)

    low_rank_slice = {
        **expanded_slice,
        "min_split_half_rank_agreement": 0.89,
    }
    with pytest.raises(ValueError, match="rank agreement"):
        runner.RerankerScoreEvidence.model_validate(
            {
                **payload,
                "min_split_half_rank_agreement": 0.89,
                "stability_slices": (low_rank_slice, payload["stability_slices"][1]),
            },
            strict=True,
        )


def test_reranker_evidence_requires_all_release_config_slices() -> None:
    reranker = runner._PairedReranker(_StableReranker())
    evidence = asyncio.run(
        reranker.preload(
            _collected_rerank_sets(),
            questions=["private query"],
            revision=_reranker_revision(),
            replay_count=3,
        )
    )

    with pytest.raises(ValueError, match="release config"):
        evidence.require_release_config("e" * 64)

    payload = evidence.model_dump(mode="python")
    locked_slices = tuple({**item, "split": "locked"} for item in payload["stability_slices"])
    payload.update(
        candidate_set_count=4,
        nonempty_candidate_set_count=4,
        single_replay_set_comparison_count=12,
        batch_shape_sample_count=payload["batch_shape_sample_count"] * 2,
        batch_shape_live_pair_score_count=payload["batch_shape_live_pair_score_count"] * 2,
        stability_slices=(*payload["stability_slices"], *locked_slices),
    )
    complete = runner.RerankerScoreEvidence.model_validate(payload, strict=True)

    complete.require_release_config("e" * 64)


def test_rank_overlap_agreement_rejects_reversed_common_items() -> None:
    full = tuple(uuid.UUID(int=value) for value in range(1, 11))
    changed = (*reversed(full[1:]), uuid.UUID(int=11))

    assert len(set(full) & set(changed)) == 9
    assert runner._rank_overlap_agreement(changed, full) == 0.0


def test_paired_reranker_rejects_invalid_live_scores() -> None:
    reranker = runner._PairedReranker(_InvalidReranker())

    with pytest.raises(runner.RetrievalEvaluationError, match="scores are invalid"):
        asyncio.run(
            reranker.preload(
                _collected_rerank_sets(),
                questions=["private query"],
                revision=_reranker_revision(),
                replay_count=3,
            )
        )


def test_paired_reranker_writes_attempt_before_live_failure(tmp_path: Path) -> None:
    reranker = runner._PairedReranker(_InvalidReranker())
    attempt = tmp_path / "reranker-attempt.json"

    with pytest.raises(runner.RetrievalEvaluationError, match="scores are invalid"):
        asyncio.run(
            reranker.preload(
                _collected_rerank_sets(),
                questions=["private query"],
                revision=_reranker_revision(),
                replay_count=3,
                attempt_output=attempt,
            )
        )

    payload = json.loads(attempt.read_text())
    assert payload["attempt_id"]
    assert "private" not in attempt.read_text()
    assert attempt.stat().st_mode & 0o777 == 0o600


def test_paired_reranker_coalesces_identical_truncated_inputs() -> None:
    prefix = "x" * 4000
    candidate_set = runner._CollectedRerankSet(
        case_id="ragq-test-bm25-0001",
        split="tuning",
        query="private query",
        config_sha256="e" * 64,
        backend="pg_textsearch",
        rerank_min_score=0.1,
        final_top_k=10,
        candidates=(
            (uuid.UUID(int=1), prefix + "a"),
            (uuid.UUID(int=2), prefix + "b"),
        ),
    )
    baseline_set = runner._CollectedRerankSet(
        case_id=candidate_set.case_id,
        split=candidate_set.split,
        query=candidate_set.query,
        config_sha256=candidate_set.config_sha256,
        backend="postgres_fts",
        rerank_min_score=candidate_set.rerank_min_score,
        final_top_k=candidate_set.final_top_k,
        candidates=candidate_set.candidates,
    )
    delegate = _StableReranker()
    reranker = runner._PairedReranker(delegate)

    evidence = asyncio.run(
        reranker.preload(
            [baseline_set, candidate_set],
            questions=["private query"],
            revision=_reranker_revision(),
            replay_count=3,
        )
    )
    scores = asyncio.run(reranker.rerank("private query", [prefix + "a", prefix + "b"]))

    assert evidence.unique_pair_count == 1
    assert [len(texts) for _, texts in delegate.calls].count(1) == 3
    assert [len(texts) for _, texts in delegate.calls].count(2) == 2
    assert scores == [0.8, 0.8]
    assert reranker.cache_stats() == (2, 0)


def test_live_counting_reranker_records_calls_and_pairs() -> None:
    reranker = runner._LiveCountingReranker(_StableReranker())

    scores = asyncio.run(reranker.rerank("private query", ["private a", "private b"]))

    assert scores == [0.9, 0.8]
    assert reranker.live_call_count == 1
    assert reranker.live_pair_score_count == 2


def test_live_counting_reranker_rejects_invalid_scores_without_counting() -> None:
    reranker = runner._LiveCountingReranker(_InvalidReranker())

    with pytest.raises(runner.RetrievalEvaluationError, match="invalid load scores"):
        asyncio.run(reranker.rerank("private query", ["private a"]))

    assert reranker.live_call_count == 0
    assert reranker.live_pair_score_count == 0


def test_paired_query_embedder_pins_vector_and_returns_a_copy() -> None:
    delegate = _CountingEmbedder()
    embedder = runner._PairedQueryEmbedder(delegate)
    asyncio.run(embedder.preload(["private query", "private query"]))

    first = asyncio.run(embedder.embed_query("private query"))
    first[0] = 1.0
    second = asyncio.run(embedder.embed_query("private query"))

    assert delegate.query_calls == 1
    assert second[0] == 0.25
    with pytest.raises(runner.RetrievalEvaluationError, match="unknown question"):
        asyncio.run(embedder.embed_query("another private query"))


def test_query_embedding_evidence_rejects_live_call_mismatch() -> None:
    with pytest.raises(ValueError, match="live-call count"):
        runner.QueryEmbeddingEvidence(
            protocol="single-live-vector-per-question-v1",
            cache_scope="run",
            reuse_scope="tuning+locked+variants+repeats",
            preloaded=True,
            unique_question_count=2,
            live_call_count=1,
            vector_manifest_sha256="a" * 64,
            config_sha256="b" * 64,
        )


def test_no_answer_aggregation_reports_returned_and_abstained_counts() -> None:
    artifact, _, _ = _execute_artifact(answerable=False)
    aggregate = runner.aggregate_pool([artifact])

    assert aggregate.no_answer_cases == 1
    assert aggregate.returned_count == 1
    assert aggregate.no_answer_returned_count == 1
    assert aggregate.no_answer_abstained_count == 0
    assert aggregate.no_answer_false_positive_rate == 1.0


def test_no_answer_statistical_clusters_use_generation_lineage() -> None:
    first_record, first_sidecar, _, _ = _case(
        1,
        owner="owner-a",
        answerable=False,
    )
    second_record, second_sidecar, _, _ = _case(
        2,
        owner="owner-a",
        answerable=False,
    )
    shared_generation = second_sidecar.generation.model_copy(update={"seed": first_sidecar.generation.seed})
    same_lineage = second_sidecar.model_copy(update={"generation": shared_generation})
    same_bindings = runner.build_case_bindings(
        [first_record, second_record],
        {
            first_record.case_id: first_sidecar,
            second_record.case_id: same_lineage,
        },
    )
    distinct_bindings = runner.build_case_bindings(
        [first_record, second_record],
        {
            first_record.case_id: first_sidecar,
            second_record.case_id: second_sidecar,
        },
    )

    same_clusters = runner._statistical_cluster_ids(same_bindings)
    distinct_clusters = runner._statistical_cluster_ids(distinct_bindings)

    assert len(set(same_clusters.values())) == 1
    assert len(set(distinct_clusters.values())) == 2


def test_sweep_selection_uses_quality_then_fingerprint() -> None:
    base = dict(
        answerable_cases=1,
        no_answer_cases=0,
        recall={"1": 0.0, "5": 0.8, "10": 1.0},
        mrr={"1": 0.0, "5": 0.5, "10": 0.5},
        no_answer_false_positive_rate=None,
        returned_count=10,
        abstained_count=0,
        no_answer_returned_count=0,
        no_answer_abstained_count=0,
    )
    weaker = runner.AggregateMetrics(
        **base,
        ndcg={"1": 0.0, "5": 0.7, "10": 0.7},
        latency_ms={"mean": 1.0, "p50": 1.0, "p95": 1.0},
    )
    stronger = runner.AggregateMetrics(
        **base,
        ndcg={"1": 0.0, "5": 0.8, "10": 0.8},
        latency_ms={"mean": 5.0, "p50": 5.0, "p95": 5.0},
    )
    assert runner.select_tuning_config({"a" * 64: weaker, "b" * 64: stronger}) == "b" * 64
    assert runner.select_tuning_config({"a" * 64: stronger, "b" * 64: stronger}) == "a" * 64


def test_split_ignores_single_cluster_labels_without_document_leakage() -> None:
    records = []
    for index in range(12):
        record = _case(index, owner=f"document-{index}")[0]
        records.append(
            record.model_copy(
                update={
                    "scope_id": make_scope_id(f"scope-{index % 2}"),
                    "content_types": ("scan",) if index == 0 else ("text",),
                }
            )
        )

    split = runner.stratified_cluster_split(records, seed=2026071409, locked_fraction=0.75)
    by_id = {record.case_id: record for record in records}
    locked_documents = {
        snapshot.document_ref
        for case_id in split.locked_case_ids
        for snapshot in by_id[case_id].document_scope
    }
    tuning_documents = {
        snapshot.document_ref
        for case_id in split.tuning_case_ids
        for snapshot in by_id[case_id].document_scope
    }

    assert len(split.locked_case_ids) == 9
    assert len(split.tuning_case_ids) == 3
    assert locked_documents.isdisjoint(tuning_documents)
    assert set(split.locked_case_ids) | set(split.tuning_case_ids) == set(by_id)
    assert (
        sum(split.distribution[partition].get("content:scan", 0) for partition in ("locked", "tuning")) == 1
    )


def test_external_evidence_is_corpus_bound_and_private(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "retrieval-rls-evidence-v1",
                "kind": "rls",
                "passed": True,
                "corpus_snapshot_sha256": "a" * 64,
                "principals": [
                    {
                        "principal_ref": f"scope-sha256:{'c' * 64}",
                        "probe_count": 1,
                        "leak_count": 0,
                        "evidence_sha256": "d" * 64,
                    }
                ],
            }
        )
    )
    path.chmod(0o600)
    evidence = runner._parse_external_evidence(
        path,
        expected_kind="rls",
        corpus_snapshot_sha256="a" * 64,
    )
    assert evidence.passed is True
    with pytest.raises(runner.RetrievalEvaluationError, match="binding mismatch"):
        runner._parse_external_evidence(
            path,
            expected_kind="rls",
            corpus_snapshot_sha256="b" * 64,
        )


def test_resume_rejects_case_from_another_run(tmp_path: Path) -> None:
    artifact, record, sidecar = _execute_artifact()
    path = tmp_path / "case.json"
    path.write_bytes(runner._canonical_bytes(artifact.model_dump(mode="json")))
    path.chmod(0o600)
    binding = runner.build_case_bindings([record], {record.case_id: sidecar})[record.case_id]

    with pytest.raises(runner.RetrievalEvaluationError, match="identity"):
        runner.load_resumed_case(
            path,
            run_id="c" * 64,
            split="tuning",
            variant="candidate",
            config_sha256=_config().fingerprint,
            query_embedding_sha256=artifact.query_embedding_sha256,
            binding=binding,
        )


def test_variant_abstention_must_match_returned_count() -> None:
    artifact, _, _ = _execute_artifact()
    payload = artifact.observation.model_dump(mode="python")
    payload["abstained"] = True
    with pytest.raises(ValueError, match="abstention must match"):
        runner.VariantObservation.model_validate(payload, strict=True)


def test_locked_decision_requires_target_gains() -> None:
    artifacts = [
        _execute_artifact(index=1)[0],
        _execute_artifact(index=2)[0],
        _execute_artifact(index=3, answerable=False)[0],
        _execute_artifact(index=4, answerable=False)[0],
    ]
    baseline = [
        item.model_copy(update={"observation": item.observation.model_copy(update={"variant": "baseline"})})
        for item in artifacts
    ]
    candidate = [
        item.model_copy(update={"observation": item.observation.model_copy(update={"variant": "candidate"})})
        for item in artifacts
    ]
    clusters = {
        item.case_id: f"cluster-sha256:{hashlib.sha256(item.case_id.encode()).hexdigest()}"
        for item in baseline
    }
    policy = runner.RetrievalGatePolicy.model_validate_json(
        (Path(__file__).parents[1] / "deploy/rag-eval/retrieval-policy-v2.json").read_bytes(),
        strict=True,
    ).model_copy(update={"bootstrap_samples": 1_000})

    decision = runner.evaluate_locked_decision(
        baseline,
        candidate,
        tuning_case_count=2,
        cluster_ids=clusters,
        policy=policy,
    )

    assert decision.accepted is False
    assert "locked_target_gain:lexical_recall_at_5" in decision.failure_codes
    assert "locked_target_gain:ndcg_at_10" in decision.failure_codes


def test_locked_decision_rejects_mismatched_query_embedding() -> None:
    artifact = _execute_artifact(index=1)[0]
    baseline = artifact.model_copy(
        update={"observation": artifact.observation.model_copy(update={"variant": "baseline"})}
    )
    candidate = artifact.model_copy(update={"query_embedding_sha256": "f" * 64})

    with pytest.raises(runner.RetrievalEvaluationError, match="embedding evidence differs"):
        runner._assert_query_embedding_pairing([baseline], [candidate])


def _load_inputs():
    records = [_case(1, owner="owner-a"), _case(2, owner="owner-a")]
    bindings = runner.build_case_bindings(
        [item[0] for item in records],
        {item[0].case_id: item[1] for item in records},
    )
    first_sidecar = records[0][1]
    document_id = first_sidecar.source_documents[0].document_id
    chunk = RetrievedChunk(
        id=records[0][2],
        document_id=document_id,
        filename="private.pdf",
        heading_path="Synthetic",
        kind="section",
        page_start=0,
        page_end=0,
        text_en="private body",
        text_ru="",
        meta={},
    )
    scope = runner._VerifiedScope(
        owner_sub="owner-a",
        document_refs={document_id: first_sidecar.source_documents[0].document_ref},
        document_ids=frozenset({document_id}),
        evidence=runner.ScopeEvidence(
            scope_id=records[0][0].scope_id,
            case_count=2,
            document_count=1,
            chunk_count=1,
            corpus_sha256="a" * 64,
        ),
    )
    return records, bindings, {records[0][0].scope_id: scope}, chunk


def _load_reranker_tracker() -> Any:
    return runner._LiveCountingReranker(_StableReranker())


def test_generate_load_evidence_bounds_concurrency_and_aggregates(tmp_path: Path) -> None:
    records, bindings, scopes, chunk = _load_inputs()
    tracker = _load_reranker_tracker()
    retriever = _LoadRetriever(chunk, reranker_tracker=tracker)
    output = tmp_path / "load.json"

    envelope = asyncio.run(
        runner.generate_load_evidence(
            output=output,
            retriever=retriever,
            reranker_tracker=tracker,
            sessionmaker=lambda: _Session(),
            bindings=bindings,
            scopes=scopes,
            locked_case_ids=[item[0].case_id for item in records],
            config=_config(),
            corpus_snapshot_sha256="a" * 64,
            runtime_binding_sha256=_sha("runtime-binding"),
            concurrency=2,
            requests_per_backend=4,
        )
    )

    assert retriever.max_active == 2
    assert envelope.baseline.request_count == 4
    assert envelope.candidate.request_count == 4
    assert envelope.baseline.error_count == 0
    assert output.stat().st_mode & 0o777 == 0o600
    raw = output.with_name("load.raw.json").read_text()
    assert "private body" not in raw
    observations = runner.RawLoadEvidence.model_validate_json(raw).observations
    assert all(item.order_in_pair in (0, 1) for item in observations)


def test_generate_load_evidence_fails_closed_on_request_error(tmp_path: Path) -> None:
    records, bindings, scopes, chunk = _load_inputs()
    tracker = _load_reranker_tracker()
    with pytest.raises(runner.RetrievalEvaluationError, match="failed or missing"):
        asyncio.run(
            runner.generate_load_evidence(
                output=tmp_path / "load.json",
                retriever=_LoadRetriever(
                    chunk,
                    reranker_tracker=tracker,
                    fail_call=2,
                ),
                reranker_tracker=tracker,
                sessionmaker=lambda: _Session(),
                bindings=bindings,
                scopes=scopes,
                locked_case_ids=[item[0].case_id for item in records],
                config=_config(),
                corpus_snapshot_sha256="a" * 64,
                runtime_binding_sha256=_sha("runtime-binding"),
                concurrency=2,
                requests_per_backend=2,
            )
        )


def test_generate_load_evidence_rejects_zero_live_reranker_calls(tmp_path: Path) -> None:
    records, bindings, scopes, chunk = _load_inputs()
    tracker = _load_reranker_tracker()

    with pytest.raises(runner.RetrievalEvaluationError, match="did not execute"):
        asyncio.run(
            runner.generate_load_evidence(
                output=tmp_path / "load.json",
                retriever=_LoadRetriever(chunk),
                reranker_tracker=tracker,
                sessionmaker=lambda: _Session(),
                bindings=bindings,
                scopes=scopes,
                locked_case_ids=[item[0].case_id for item in records],
                config=_config(),
                corpus_snapshot_sha256="a" * 64,
                runtime_binding_sha256=_sha("runtime-binding"),
                concurrency=2,
                requests_per_backend=2,
            )
        )


def test_generate_load_evidence_rejects_invalid_live_reranker_scores(tmp_path: Path) -> None:
    records, bindings, scopes, chunk = _load_inputs()
    tracker = runner._LiveCountingReranker(_InvalidReranker())

    with pytest.raises(runner.RetrievalEvaluationError, match="did not execute"):
        asyncio.run(
            runner.generate_load_evidence(
                output=tmp_path / "load.json",
                retriever=_LoadRetriever(chunk, reranker_tracker=tracker),
                reranker_tracker=tracker,
                sessionmaker=lambda: _Session(),
                bindings=bindings,
                scopes=scopes,
                locked_case_ids=[item[0].case_id for item in records],
                config=_config(),
                corpus_snapshot_sha256="a" * 64,
                runtime_binding_sha256=_sha("runtime-binding"),
                concurrency=2,
                requests_per_backend=2,
            )
        )


def test_candidate_provenance_binds_legacy_fts_manifest() -> None:
    policy = runner.RetrievalGatePolicy.model_validate_json(
        (Path(__file__).parents[1] / "deploy/rag-eval/retrieval-policy-v2.json").read_bytes(),
        strict=True,
    )
    indexes = (*policy.required_baseline_indexes, *policy.required_candidate_indexes)
    definitions = {item.name: item.canonical_definition for item in indexes}
    database = runner.DatabaseEvidence(
        image_ref="rag-postgres-bm25:test",
        image_digest=policy.required_candidate_image_digest,
        server_version_num=170_000,
        extensions={"pg_textsearch": policy.required_pg_textsearch_version},
        index_definitions=definitions,
        index_definitions_sha256={
            name: hashlib.sha256(definition.encode()).hexdigest() for name, definition in definitions.items()
        },
        extension_binary_sha256=policy.required_extension_binary_sha256,
        extension_binary_bytes=1_746_152,
        extension_version=policy.required_pg_textsearch_version,
        extension_commit=policy.required_pg_textsearch_commit,
        package_sha256=policy.required_pg_textsearch_package_sha256,
        base_image_digest=policy.required_base_postgres_image_digest,
        build_recipe_sha256=policy.required_build_recipe_sha256,
        prepare_sql_sha256=policy.required_prepare_sql_sha256,
        baseline_index_manifest_sha256=policy.required_baseline_index_manifest_sha256,
        candidate_index_manifest_sha256=policy.required_candidate_index_manifest_sha256,
    )
    engine = runner._gate_sparse_engine(
        "pg_textsearch",
        policy=policy,
        database=database,
    )
    assert engine.pg_textsearch is not None
    assert (
        engine.pg_textsearch.legacy_fts_index_manifest_sha256
        == policy.required_baseline_index_manifest_sha256
    )


def test_corpus_verifier_awaits_storage_read(monkeypatch: pytest.MonkeyPatch) -> None:
    record, sidecar, chunk_id, _ = _case(1, owner="owner-a")
    document_id = sidecar.source_documents[0].document_id
    snapshot = record.document_scope[0]
    document = SimpleNamespace(
        id=document_id,
        s3_key_original="private/source.pdf",
        page_count=1,
        chunk_count=1,
        indexed_at=None,
        updated_at=None,
    )
    chunk = SimpleNamespace(
        id=chunk_id,
        document_id=document_id,
        idx=1,
        kind="section",
        heading_path="Synthetic",
        page_start=0,
        page_end=0,
        text_en="Private exact evidence for case 1.",
        body="Private exact evidence for case 1.",
        emb_en_text="[0.1]",
        emb_ru_text="[0.1]",
        meta={},
    )
    storage = SimpleNamespace(get_bytes=AsyncMock(return_value=b"source-bytes"))
    verifier = runner._CorpusVerifier(lambda: _Session(), storage)
    verifier._load_scope_rows = AsyncMock(return_value=("owner-a", [document], [chunk], []))
    verifier._verify_case_evidence = AsyncMock(return_value=None)
    monkeypatch.setattr(runner, "bytes_sha256", lambda _: snapshot.source_sha256)
    monkeypatch.setattr(
        runner,
        "parsed_chunks_sha256",
        lambda _: snapshot.parsed_content_sha256,
    )

    asyncio.run(verifier.verify([record], {record.case_id: sidecar}))

    storage.get_bytes.assert_awaited_once_with("originals", "private/source.pdf")


def test_rejected_qualification_uses_nonzero_exit_code() -> None:
    report = SimpleNamespace(release_accepted=False)
    assert runner._result_exit_code(report, "qualification") == 2
    assert runner._result_exit_code(report, "dev") == 0


def test_qualification_resume_rejects_unsigned_and_tampered_case(tmp_path: Path) -> None:
    artifact, record, sidecar = _execute_artifact()
    binding = runner.build_case_bindings([record], {record.case_id: sidecar})[record.case_id]
    path = tmp_path / "case.json"
    key = b"case-attestation-key-material-32-bytes!"
    runner.load_or_write_case(path, artifact)
    with pytest.raises(runner.RetrievalEvaluationError, match="missing its case HMAC"):
        runner.load_resumed_case(
            path,
            run_id=artifact.run_id,
            split=artifact.split,
            variant=artifact.observation.variant,
            config_sha256=artifact.observation.config_sha256,
            query_embedding_sha256=artifact.query_embedding_sha256,
            binding=binding,
            hmac_key=key,
        )

    path.unlink()
    runner.load_or_write_case(path, artifact, hmac_key=key)
    payload = json.loads(path.read_text())
    payload["observation"]["pools"]["final"]["latency_ms"] += 1.0
    path.write_text(json.dumps(payload, sort_keys=True))
    path.chmod(0o600)
    with pytest.raises(runner.RetrievalEvaluationError, match="HMAC verification failed"):
        runner.load_resumed_case(
            path,
            run_id=artifact.run_id,
            split=artifact.split,
            variant=artifact.observation.variant,
            config_sha256=artifact.observation.config_sha256,
            query_embedding_sha256=artifact.query_embedding_sha256,
            binding=binding,
            hmac_key=key,
        )


def test_load_envelope_is_recomputed_from_raw_observations(tmp_path: Path) -> None:
    records, bindings, scopes, chunk = _load_inputs()
    locked_ids = [item[0].case_id for item in records]
    output = tmp_path / "load.json"
    tracker = _load_reranker_tracker()
    asyncio.run(
        runner.generate_load_evidence(
            output=output,
            retriever=_LoadRetriever(chunk, reranker_tracker=tracker),
            reranker_tracker=tracker,
            sessionmaker=lambda: _Session(),
            bindings=bindings,
            scopes=scopes,
            locked_case_ids=locked_ids,
            config=_config(),
            corpus_snapshot_sha256="a" * 64,
            runtime_binding_sha256=_sha("runtime-binding"),
            concurrency=2,
            requests_per_backend=4,
        )
    )
    payload = json.loads(output.read_text())
    payload["candidate"]["p95_latency_ms"] += 1000.0
    output.write_text(json.dumps(payload, sort_keys=True))
    output.chmod(0o600)

    with pytest.raises(runner.RetrievalEvaluationError, match="does not bind"):
        runner._validate_load_evidence_binding(
            output,
            config=_config(),
            locked_case_ids=locked_ids,
            bindings=bindings,
            corpus_snapshot_sha256="a" * 64,
            runtime_binding_sha256=_sha("runtime-binding"),
            concurrency=2,
            requests_per_backend=4,
        )


def test_load_raw_rejects_declared_concurrency_above_observed_peak(tmp_path: Path) -> None:
    records, bindings, scopes, chunk = _load_inputs()
    locked_ids = [item[0].case_id for item in records]
    output = tmp_path / "load.json"
    tracker = _load_reranker_tracker()
    asyncio.run(
        runner.generate_load_evidence(
            output=output,
            retriever=_LoadRetriever(chunk, reranker_tracker=tracker),
            reranker_tracker=tracker,
            sessionmaker=lambda: _Session(),
            bindings=bindings,
            scopes=scopes,
            locked_case_ids=locked_ids,
            config=_config(),
            corpus_snapshot_sha256="a" * 64,
            runtime_binding_sha256=_sha("runtime-binding"),
            concurrency=2,
            requests_per_backend=4,
        )
    )
    raw_path = output.with_name("load.raw.json")
    raw_payload = json.loads(raw_path.read_text())
    legacy_payload = dict(raw_payload)
    legacy_payload.pop("embedding_protocol")
    with pytest.raises(ValueError):
        runner.RawLoadEvidence.model_validate(legacy_payload, strict=True)
    with pytest.raises(ValueError):
        runner.RawLoadEvidence.model_validate(
            {**raw_payload, "embedding_protocol": "cached"},
            strict=True,
        )
    for required_field in (
        "reranker_protocol",
        "reranker_live_call_count",
        "reranker_live_pair_score_count",
        "runtime_binding_sha256",
    ):
        incomplete_payload = dict(raw_payload)
        incomplete_payload.pop(required_field)
        with pytest.raises(ValueError):
            runner.RawLoadEvidence.model_validate(incomplete_payload, strict=True)
    with pytest.raises(ValueError):
        runner.RawLoadEvidence.model_validate(
            {**raw_payload, "reranker_protocol": "cached"},
            strict=True,
        )
    with pytest.raises(ValueError):
        runner.RawLoadEvidence.model_validate(
            {**raw_payload, "reranker_live_call_count": 7},
            strict=True,
        )
    raw_payload["concurrency"] = 3
    raw_path.write_text(json.dumps(raw_payload, sort_keys=True))
    raw_path.chmod(0o600)

    with pytest.raises(runner.RetrievalEvaluationError, match="declared concurrency"):
        runner._validate_load_evidence_binding(
            output,
            config=_config(),
            locked_case_ids=locked_ids,
            bindings=bindings,
            corpus_snapshot_sha256="a" * 64,
            runtime_binding_sha256=_sha("runtime-binding"),
        )


def test_runtime_database_provenance_rejects_unmeasured_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = runner.RetrievalGatePolicy.model_validate_json(
        (Path(__file__).parents[1] / "deploy/rag-eval/retrieval-policy-v2.json").read_bytes(),
        strict=True,
    )
    binary = tmp_path / "pg_textsearch.so"
    binary.write_bytes(b"not-the-qualified-extension")
    monkeypatch.setattr(
        runner,
        "_inspect_database_container",
        lambda _: ("rag-postgres-bm25:test", policy.required_candidate_image_digest),
    )

    with pytest.raises(runner.RetrievalEvaluationError, match="provenance"):
        runner._runtime_database_evidence(
            container="rag-postgres-bm25",
            extension_binary_path=binary,
            policy=policy,
        )


def test_database_container_name_rejects_command_injection() -> None:
    with pytest.raises(runner.RetrievalEvaluationError, match="container name"):
        runner._inspect_database_container("postgres;touch /tmp/unsafe")


def test_database_evidence_preserves_index_definition_literals() -> None:
    policy = runner.RetrievalGatePolicy.model_validate_json(
        (Path(__file__).parents[1] / "deploy/rag-eval/retrieval-policy-v2.json").read_bytes(),
        strict=True,
    )
    expected = {
        item.name: item.canonical_definition
        for item in (*policy.required_baseline_indexes, *policy.required_candidate_indexes)
    }

    class Result:
        def __init__(self, *, scalar: int | None = None, rows: list[Any] | None = None) -> None:
            self.scalar = scalar
            self.rows = rows or []

        def scalar_one(self) -> int:
            assert self.scalar is not None
            return self.scalar

        def all(self) -> list[Any]:
            return self.rows

    class Connection:
        async def __aenter__(self) -> Connection:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def execute(self, statement: object, _: object = None) -> Result:
            query = str(statement)
            if query == "SHOW server_version_num":
                return Result(scalar=170010)
            if "pg_extension" in query:
                return Result(
                    rows=[
                        SimpleNamespace(
                            extname="pg_textsearch",
                            extversion=policy.required_pg_textsearch_version,
                        )
                    ]
                )
            return Result(
                rows=[
                    SimpleNamespace(indexname=name, indexdef=definition)
                    for name, definition in sorted(expected.items())
                ]
            )

    class Engine:
        def connect(self) -> Connection:
            return Connection()

    runtime = runner.DatabaseEvidence(
        image_ref="rag-postgres-bm25:test",
        image_digest=policy.required_candidate_image_digest,
        server_version_num=170010,
        extensions={},
        index_definitions={},
        index_definitions_sha256={},
        extension_binary_sha256=policy.required_extension_binary_sha256,
        extension_binary_bytes=1,
        extension_version=policy.required_pg_textsearch_version,
        extension_commit=policy.required_pg_textsearch_commit,
        package_sha256=policy.required_pg_textsearch_package_sha256,
        base_image_digest=policy.required_base_postgres_image_digest,
        build_recipe_sha256=policy.required_build_recipe_sha256,
        prepare_sql_sha256=policy.required_prepare_sql_sha256,
        baseline_index_manifest_sha256=policy.required_baseline_index_manifest_sha256,
        candidate_index_manifest_sha256=policy.required_candidate_index_manifest_sha256,
    )

    evidence = asyncio.run(runner._database_evidence(Engine(), policy=policy, runtime=runtime))

    assert evidence.index_definitions == expected
    assert "\n" in expected["ix_chunks_bm25_en_v1"]


def test_hybrid_metric_uses_rrf_pool_not_dense_prefix() -> None:
    artifact, _, _ = _execute_artifact()
    relevant = artifact.relevant_chunk_ids[0]
    irrelevant = tuple(uuid.uuid4() for _ in range(20))

    def pool(ids: tuple[uuid.UUID, ...]) -> Any:
        return runner.PoolObservation(
            ranked_chunk_ids=ids,
            order_sha256=runner._sha256_order(ids),
            latency_ms=1.0,
            metrics=runner.score_ranking(
                ids,
                {relevant: 3},
                answerable=True,
            ),
        )

    pools = dict(artifact.observation.pools)
    pools["dense"] = pool(irrelevant)
    pools["sparse"] = pool((relevant,))
    pools["hybrid"] = pool((relevant,))
    changed = artifact.model_copy(
        update={"observation": artifact.observation.model_copy(update={"pools": pools})}
    )

    result = runner._gate_case_result(
        changed,
        cluster_id=f"cluster-sha256:{'1' * 64}",
    )
    assert result.metrics is not None
    assert result.metrics.hybrid_union_recall_at_20 == 1.0


def test_sparse_engine_contract_is_language_and_script_bound() -> None:
    en_record = _case(1, owner="owner-a")[0]
    zh_question = "该文件规定的精确数值是多少？"
    zh_record = en_record.model_copy(
        update={
            "language": "zh",
            "question": zh_question,
            "question_sha256": text_sha256(zh_question),
        }
    )
    assert runner._expected_sparse_engine(en_record, "pg_textsearch") == "pg_textsearch_en"
    assert runner._expected_sparse_engine(zh_record, "pg_textsearch") == "postgres_fts"
    invalid_zh = zh_record.model_copy(
        update={
            "question": en_record.question,
            "question_sha256": en_record.question_sha256,
        }
    )
    with pytest.raises(runner.RetrievalEvaluationError, match="script routing"):
        runner._expected_sparse_engine(invalid_zh, "pg_textsearch")


def test_qualification_policy_must_be_git_tracked(tmp_path: Path) -> None:
    with pytest.raises(runner.RetrievalEvaluationError, match="tracked repository"):
        runner._tracked_repository_source(tmp_path / "policy.json")
    assert runner._tracked_repository_source(Path(__file__).parents[1] / "pyproject.toml") == (
        "pyproject.toml"
    )


def test_qualification_rejects_split_operational_evidence() -> None:
    with pytest.raises(runner.RetrievalEvaluationError, match="consolidated operational"):
        runner._evidence_paths_from_args(
            [
                "rls=/tmp/rls.json",
                "load=/tmp/load.json",
                "update=/tmp/update.json",
                "delete=/tmp/delete.json",
                "restart=/tmp/restart.json",
            ],
            mode="qualification",
            operational_evidence=None,
            load_evidence=None,
            generated_load_evidence=None,
        )


def test_operational_summary_requires_bound_raw_artifact(tmp_path: Path) -> None:
    policy = runner.RetrievalGatePolicy.model_validate_json(
        (Path(__file__).parents[1] / "deploy/rag-eval/retrieval-policy-v2.json").read_bytes(),
        strict=True,
    )
    principals = tuple(
        runner.RlsPrincipalEvidence(
            principal_ref=f"principal-sha256:{index:064x}",
            probe_count=1,
            leak_count=0,
            evidence_sha256=f"{index + 100:064x}",
        )
        for index in range(10)
    )
    evidence = runner.OperationalEvidence(
        schema_version="rag-operational-evidence-v3",
        corpus_snapshot_sha256=policy.corpus_snapshot_sha256,
        candidate_image_digest=policy.required_candidate_image_digest,
        candidate_index_manifest_sha256=policy.required_candidate_index_manifest_sha256,
        rls_principals=principals,
        update_visible=True,
        update_visibility_seconds=1.0,
        delete_hidden=True,
        delete_visibility_seconds=1.0,
        restart_recovered=True,
        restart_recovery_seconds=1.0,
        determinism_replays=3,
        determinism_mismatches=0,
        determinism_seconds=1.0,
        rollback_succeeded=True,
        rollback_seconds=1.0,
        rollback_backend="postgres_fts",
        rollback_index_manifest_sha256=policy.required_baseline_index_manifest_sha256,
        raw_evidence_sha256="f" * 64,
    )
    path = tmp_path / "operational_evidence.json"
    path.write_bytes(runner._canonical_bytes(evidence.model_dump(mode="json")))
    path.chmod(0o600)

    with pytest.raises(runner.RetrievalEvaluationError, match="raw evidence is invalid"):
        runner._parse_operational_evidence(
            path,
            corpus_snapshot_sha256=policy.corpus_snapshot_sha256,
            policy=policy,
        )
