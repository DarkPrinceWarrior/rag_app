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
    def __init__(self, chunk: RetrievedChunk, *, flip: bool = False) -> None:
        self.chunk = chunk
        self.flip = flip
        self.calls = 0

    async def retrieve_with_trace(self, session, query, **kwargs):
        del session, query
        assert current_principal().user_sub == "owner-a"
        self.calls += 1
        rows = (self.chunk,)
        final = () if self.flip and self.calls % 2 == 0 else rows
        return RetrievalTrace(
            requested_sparse_backend=kwargs["sparse_backend"],
            sparse_engine=(
                "postgres_fts" if kwargs["sparse_backend"] == "postgres_fts" else "pg_textsearch_en"
            ),
            dense=rows,
            sparse=rows,
            hybrid_pre_rerank=rows,
            reranked=rows,
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
    def __init__(self, chunk: RetrievedChunk, *, fail_call: int | None = None) -> None:
        super().__init__(chunk)
        self.active = 0
        self.max_active = 0
        self.fail_call = fail_call

    async def retrieve_with_trace(self, session, query, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.005)
            if self.fail_call is not None and self.calls + 1 == self.fail_call:
                self.calls += 1
                raise RuntimeError("synthetic load failure")
            return await super().retrieve_with_trace(session, query, **kwargs)
        finally:
            self.active -= 1


def _execute_artifact(*, index: int = 1, answerable: bool = True, flip: bool = False):
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
    return (
        asyncio.run(
            runner._execute_case(
                retriever=_FakeRetriever(chunk, flip=flip),
                sessionmaker=lambda: _Session(),
                binding=binding,
                scope=scope,
                config=_config(),
                backend="pg_textsearch",
                variant="candidate",
                split="tuning",
                run_id="b" * 64,
                repeat_count=2,
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


def test_sweep_selection_uses_quality_then_latency() -> None:
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
            binding=binding,
        )


def test_variant_abstention_must_match_returned_count() -> None:
    artifact, _, _ = _execute_artifact()
    payload = artifact.observation.model_dump(mode="python")
    payload["abstained"] = True
    with pytest.raises(ValueError, match="abstention must match"):
        runner.VariantObservation.model_validate(payload, strict=True)


def test_locked_decision_requires_target_gains() -> None:
    baseline = [
        _execute_artifact(index=1)[0],
        _execute_artifact(index=2)[0],
        _execute_artifact(index=3, answerable=False)[0],
        _execute_artifact(index=4, answerable=False)[0],
    ]
    candidate = list(baseline)
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


def test_generate_load_evidence_bounds_concurrency_and_aggregates(tmp_path: Path) -> None:
    records, bindings, scopes, chunk = _load_inputs()
    retriever = _LoadRetriever(chunk)
    output = tmp_path / "load.json"

    envelope = asyncio.run(
        runner.generate_load_evidence(
            output=output,
            retriever=retriever,
            sessionmaker=lambda: _Session(),
            bindings=bindings,
            scopes=scopes,
            locked_case_ids=[item[0].case_id for item in records],
            config=_config(),
            corpus_snapshot_sha256="a" * 64,
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
    with pytest.raises(runner.RetrievalEvaluationError, match="failed or missing"):
        asyncio.run(
            runner.generate_load_evidence(
                output=tmp_path / "load.json",
                retriever=_LoadRetriever(chunk, fail_call=2),
                sessionmaker=lambda: _Session(),
                bindings=bindings,
                scopes=scopes,
                locked_case_ids=[item[0].case_id for item in records],
                config=_config(),
                corpus_snapshot_sha256="a" * 64,
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
            binding=binding,
            hmac_key=key,
        )


def test_load_envelope_is_recomputed_from_raw_observations(tmp_path: Path) -> None:
    records, bindings, scopes, chunk = _load_inputs()
    locked_ids = [item[0].case_id for item in records]
    output = tmp_path / "load.json"
    asyncio.run(
        runner.generate_load_evidence(
            output=output,
            retriever=_LoadRetriever(chunk),
            sessionmaker=lambda: _Session(),
            bindings=bindings,
            scopes=scopes,
            locked_case_ids=locked_ids,
            config=_config(),
            corpus_snapshot_sha256="a" * 64,
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
            corpus_snapshot_sha256="a" * 64,
            concurrency=2,
            requests_per_backend=4,
        )


def test_load_raw_rejects_declared_concurrency_above_observed_peak(tmp_path: Path) -> None:
    records, bindings, scopes, chunk = _load_inputs()
    locked_ids = [item[0].case_id for item in records]
    output = tmp_path / "load.json"
    asyncio.run(
        runner.generate_load_evidence(
            output=output,
            retriever=_LoadRetriever(chunk),
            sessionmaker=lambda: _Session(),
            bindings=bindings,
            scopes=scopes,
            locked_case_ids=locked_ids,
            config=_config(),
            corpus_snapshot_sha256="a" * 64,
            concurrency=2,
            requests_per_backend=4,
        )
    )
    raw_path = output.with_name("load.raw.json")
    raw_payload = json.loads(raw_path.read_text())
    raw_payload["concurrency"] = 3
    raw_path.write_text(json.dumps(raw_payload, sort_keys=True))
    raw_path.chmod(0o600)

    with pytest.raises(runner.RetrievalEvaluationError, match="declared concurrency"):
        runner._validate_load_evidence_binding(
            output,
            config=_config(),
            locked_case_ids=locked_ids,
            corpus_snapshot_sha256="a" * 64,
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
