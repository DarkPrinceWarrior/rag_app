from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from rag_app.eval.retrieval_gate import (
    ContentType,
    Language,
    LoadEvidence,
    OperationalEvidence,
    OwnerScopeProvenance,
    PgTextsearchProvenance,
    RetrievalCaseMetrics,
    RetrievalCaseResult,
    RetrievalConfiguration,
    RetrievalGateError,
    RetrievalGatePolicy,
    RetrievalModelRevisions,
    RetrievalProvenance,
    RetrievalReport,
    RlsPrincipalEvidence,
    RuntimeModelRevision,
    SparseBackend,
    SparseEngineProvenance,
    SparseIndexDefinition,
    canonical_sha256,
    evaluate_retrieval_gate,
)

_POLICY_PATH = Path("deploy/rag-eval/retrieval-policy-v2.json")
_NOW = datetime(2026, 7, 14, 9, tzinfo=UTC)
_GOLD_SHA256 = "c5ac4752dba3f7303832847c6caa5e4aa21f262ab5be1fe2768ac17be8ef1578"
_SIDECAR_SHA256 = "1635d3a59465fd6b4f9b9ca56c97bbc091a1f1c7c4e7c46c707a1272924ec717"
_CORPUS_SHA256 = "83e2c7098f58fd405955553131f3b2072cbe4d95dd53d54fea88e6eccaeaa5d1"
_PACKAGE_SHA256 = "dc9a823e16f59b24b8c7f17c07a497b728be6c043643b4d56a65c912fa254349"
_BINARY_SHA256 = "c3d0d5e0a9ff0be5fd0b9aa0710f9640ce6b7b7a0ece4075a7ba1dd9c27f3c5e"
_COMMIT = "578ff529894992fb9e67cae4c69424e65c84868e"
_BASE_IMAGE = "sha256:feb68f4f15446397d8cac7f4fe48fe4586de83160d1fc48b46283312d1a33966"
_CANDIDATE_IMAGE = "sha256:f8dc1dbfd85dcde9cdece2b5265f8653bae78b44b5e0ece865b3869f3b58aff2"
_DOCKERFILE_SHA256 = "7781086b50b6112edc9499ae7e2aa21704d0dbd9c079ae428e105658f4e2d1c1"
_PREPARE_SQL_SHA256 = "e68797a79c3d336f787d93d9acef1cca2d7f5bb3f20e007762174ca15e3df02f"
_BASELINE_INDEX_MANIFEST_SHA256 = "0a67ee68830af6cf0e818ece54d074eff86bfacd1eb27bbf26182b13c8aa6aeb"
_CONTENT_TYPES: tuple[ContentType, ...] = ("text", "table", "formula", "figure", "scan")
_LANGUAGES: tuple[Language, ...] = ("ru", "en", "zh")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _scope(index: int) -> str:
    return f"scope-sha256:{_sha(f'scope:{index}')}"


def _principal(index: int) -> str:
    return f"principal-sha256:{_sha(f'principal:{index}')}"


def _cluster(index: int) -> str:
    return f"cluster-sha256:{_sha(f'cluster:{index}')}"


def _index(
    name: str,
    method: Literal["gin", "bm25"],
    definition: str,
    *,
    text_config: Literal["russian", "english"] | None = None,
    k1: float | None = None,
    b: float | None = None,
) -> SparseIndexDefinition:
    return SparseIndexDefinition(
        name=name,
        access_method=method,
        text_config=text_config,
        k1=k1,
        b=b,
        canonical_definition=definition,
        definition_sha256=_sha(definition),
    )


def _engine(backend: SparseBackend) -> SparseEngineProvenance:
    if backend == "postgres_fts":
        indexes: tuple[SparseIndexDefinition, ...] = (
            _index(
                "ix_chunks_tsv",
                "gin",
                "CREATE INDEX ix_chunks_tsv ON public.chunks USING gin (tsv)",
            ),
        )
        extension = None
    else:
        indexes = (
            _index(
                "ix_chunks_bm25_en_v1",
                "bm25",
                "CREATE INDEX ix_chunks_bm25_en_v1 ON public.chunks USING bm25 "
                "((((COALESCE(text_en, ''::text) || '\n'::text) || "
                "COALESCE(text_ru, ''::text)))) WITH "
                "(text_config=english, k1='1.2', b='0.75')",
                text_config="english",
                k1=1.2,
                b=0.75,
            ),
            _index(
                "ix_chunks_bm25_ru_v1",
                "bm25",
                "CREATE INDEX ix_chunks_bm25_ru_v1 ON public.chunks USING bm25 "
                "((((COALESCE(text_ru, ''::text) || '\n'::text) || "
                "COALESCE(text_en, ''::text)))) WITH "
                "(text_config=russian, k1='1.2', b='0.75')",
                text_config="russian",
                k1=1.2,
                b=0.75,
            ),
        )
        extension = PgTextsearchProvenance(
            version="1.3.1",
            extension_commit=_COMMIT,
            package_sha256=_PACKAGE_SHA256,
            extension_binary_sha256=_BINARY_SHA256,
            extension_binary_path="/usr/lib/postgresql/17/lib/pg_textsearch.so",
            extension_binary_bytes=1_746_152,
            container_image_digest=_CANDIDATE_IMAGE,
            base_postgres_image_digest=_BASE_IMAGE,
            build_recipe_sha256=_DOCKERFILE_SHA256,
            prepare_sql_sha256=_PREPARE_SQL_SHA256,
            legacy_fts_index_manifest_sha256=_BASELINE_INDEX_MANIFEST_SHA256,
            spdx_license="PostgreSQL",
        )
    return SparseEngineProvenance(
        backend=backend,
        indexes=indexes,
        index_manifest_sha256=canonical_sha256(indexes),
        pg_textsearch=extension,
    )


def _metrics(*, candidate: bool, degraded: bool = False) -> RetrievalCaseMetrics:
    if degraded:
        return RetrievalCaseMetrics(
            recall_at_5=0.48,
            recall_at_10=0.48,
            mrr_at_10=0.48,
            ndcg_at_10=0.48,
            lexical_recall_at_5=0.48,
            lexical_recall_at_50=0.48,
            hybrid_union_recall_at_20=0.48,
        )
    gain = 0.01 if candidate else 0.0
    return RetrievalCaseMetrics(
        recall_at_5=0.5 + gain,
        recall_at_10=0.5 + gain,
        mrr_at_10=0.5 + gain,
        ndcg_at_10=0.53 if candidate else 0.5,
        lexical_recall_at_5=0.55 if candidate else 0.5,
        lexical_recall_at_50=0.5 + gain,
        hybrid_union_recall_at_20=0.5 + gain,
    )


def _cases(
    *,
    candidate: bool,
    degraded_language: str | None = None,
    empty_formula_slice: bool = False,
    degraded_no_answer: bool = False,
) -> tuple[RetrievalCaseResult, ...]:
    result: list[RetrievalCaseResult] = []
    for index in range(236):
        language = _LANGUAGES[index % 3]
        content_type = _CONTENT_TYPES[index % len(_CONTENT_TYPES)]
        answerable = index < 169
        if answerable and empty_formula_slice and content_type == "formula":
            content_type = "text"
        abstained = not answerable and index % 3 != 0
        if candidate and degraded_no_answer and not answerable:
            abstained = False
        result.append(
            RetrievalCaseResult(
                case_id=f"ragq-case-{index:04d}",
                gold_case_sha256=_sha(f"gold:{index}"),
                reviewed=True,
                scope_id=_scope(index % 2),
                cluster_id=_cluster(index % 12),
                language=language,
                content_types=(content_type,),
                answerable=answerable,
                sparse_engine=(
                    "postgres_fts"
                    if not candidate or language == "zh"
                    else "pg_textsearch_ru"
                    if language == "ru"
                    else "pg_textsearch_en"
                ),
                metrics=(
                    _metrics(
                        candidate=candidate,
                        degraded=candidate and language == degraded_language,
                    )
                    if answerable
                    else None
                ),
                returned_count=0 if abstained else 5,
                abstained=abstained,
                retrieval_ms=100 + index,
            )
        )
    return tuple(result)


def _models() -> RetrievalModelRevisions:
    return RetrievalModelRevisions(
        embedding=RuntimeModelRevision(
            model="qwen3-embedding-8b",
            declared_revision="weights-v1",
            endpoint_metadata_sha256=_sha("embedding-metadata"),
            runtime_version_sha256=_sha("tei-runtime"),
            weight_manifest_sha256=_sha("embedding-weights"),
            model_config_sha256=_sha("embedding-config"),
        ),
        reranker=RuntimeModelRevision(
            model="qwen3-reranker-4b",
            declared_revision="weights-v1",
            endpoint_metadata_sha256=_sha("reranker-metadata"),
            runtime_version_sha256=_sha("tei-runtime"),
            weight_manifest_sha256=_sha("reranker-weights"),
            model_config_sha256=_sha("reranker-config"),
        ),
    )


def test_case_abstention_must_match_returned_count() -> None:
    payload = _cases(candidate=False)[0].model_dump(mode="python")
    payload["abstained"] = True

    with pytest.raises(ValidationError, match="abstention must match"):
        RetrievalCaseResult.model_validate(payload, strict=True)


def _configuration() -> RetrievalConfiguration:
    return RetrievalConfiguration(
        dense_top_k=50,
        sparse_top_k=50,
        rrf_k=60,
        rerank_top_k=20,
        final_top_k=5,
        rerank_min_score=0.0,
        embedding_dim=1024,
        visual_enabled=False,
    )


def _provenance(
    engine: SparseEngineProvenance,
    cases: tuple[RetrievalCaseResult, ...],
) -> RetrievalProvenance:
    scopes = tuple(
        OwnerScopeProvenance(
            scope_id=scope_id,
            case_count=sum(item.scope_id == scope_id for item in cases),
            document_count=5,
            chunk_count=146,
            corpus_sha256=_sha(f"corpus:{scope_id}"),
        )
        for scope_id in sorted({_scope(0), _scope(1)})
    )
    configuration = _configuration()
    return RetrievalProvenance(
        repo_sha="a" * 40,
        git_dirty=False,
        gold_artifact_sha256=_GOLD_SHA256,
        sidecar_artifact_sha256=_SIDECAR_SHA256,
        corpus_snapshot_sha256=_CORPUS_SHA256,
        postgres_server_version_num=170010,
        reviewed_case_count=len(cases),
        owner_scopes=scopes,
        owner_scope_manifest_sha256=canonical_sha256(scopes),
        models=_models(),
        configuration=configuration,
        configuration_sha256=canonical_sha256(configuration),
        sparse_engine=engine,
    )


def _load(*, p95_ms: float = 1_050.0, duration_seconds: float = 20.0) -> LoadEvidence:
    completed = 200
    return LoadEvidence(
        concurrency=10,
        request_count=completed,
        completed_count=completed,
        error_count=0,
        duration_seconds=duration_seconds,
        p95_latency_ms=p95_ms,
        throughput_rps=completed / duration_seconds,
        raw_observations_sha256=_sha(f"load:{p95_ms}:{duration_seconds}"),
    )


def _operations(baseline_engine: SparseEngineProvenance) -> OperationalEvidence:
    principal_refs = sorted((_scope(0), _scope(1), *(_principal(index) for index in range(8))))
    rls = tuple(
        RlsPrincipalEvidence(
            principal_ref=principal_ref,
            probe_count=20,
            leak_count=0,
            evidence_sha256=_sha(f"rls:{principal_ref}"),
        )
        for principal_ref in principal_refs
    )
    return OperationalEvidence(
        schema_version="rag-operational-evidence-v3",
        corpus_snapshot_sha256=_CORPUS_SHA256,
        candidate_image_digest=_CANDIDATE_IMAGE,
        candidate_index_manifest_sha256=_engine("pg_textsearch").index_manifest_sha256,
        rls_principals=rls,
        update_visible=True,
        update_visibility_seconds=2.0,
        delete_hidden=True,
        delete_visibility_seconds=2.0,
        restart_recovered=True,
        restart_recovery_seconds=30.0,
        determinism_replays=3,
        determinism_mismatches=0,
        determinism_seconds=45.0,
        rollback_succeeded=True,
        rollback_seconds=120.0,
        rollback_backend=baseline_engine.backend,
        rollback_index_manifest_sha256=baseline_engine.index_manifest_sha256,
        raw_evidence_sha256=_sha("operations"),
    )


def _report(
    *,
    candidate: bool,
    degraded_language: str | None = None,
    empty_formula_slice: bool = False,
    degraded_no_answer: bool = False,
) -> RetrievalReport:
    cases = _cases(
        candidate=candidate,
        degraded_language=degraded_language,
        empty_formula_slice=empty_formula_slice,
        degraded_no_answer=degraded_no_answer,
    )
    engine = _engine("pg_textsearch" if candidate else "postgres_fts")
    manifest = [
        item.model_dump(
            mode="json",
            exclude={
                "sparse_engine",
                "metrics",
                "returned_count",
                "abstained",
                "retrieval_ms",
            },
        )
        for item in cases
    ]
    return RetrievalReport(
        evaluated_at=_NOW,
        provenance=_provenance(engine, cases),
        case_count=len(cases),
        case_manifest_sha256=canonical_sha256(manifest),
        cases=cases,
        load=_load(p95_ms=1_050.0 if candidate else 1_000.0),
        operations=_operations(_engine("postgres_fts")),
    )


def _policy() -> RetrievalGatePolicy:
    production = RetrievalGatePolicy.model_validate_json(_POLICY_PATH.read_text())
    return production.model_copy(update={"bootstrap_samples": 1_000})


def _replace_report(
    report: RetrievalReport,
    *,
    cases: tuple[RetrievalCaseResult, ...] | None = None,
    provenance: RetrievalProvenance | None = None,
    load: LoadEvidence | None = None,
    operations: OperationalEvidence | None = None,
) -> RetrievalReport:
    updated_cases = cases or report.cases
    manifest = [
        item.model_dump(
            mode="json",
            exclude={
                "sparse_engine",
                "metrics",
                "returned_count",
                "abstained",
                "retrieval_ms",
            },
        )
        for item in updated_cases
    ]
    return RetrievalReport(
        evaluated_at=report.evaluated_at,
        provenance=provenance or report.provenance,
        case_count=len(updated_cases),
        case_manifest_sha256=canonical_sha256(manifest),
        cases=updated_cases,
        load=load or report.load,
        operations=operations or report.operations,
    )


def test_production_policy_is_strict_and_pinned() -> None:
    policy = RetrievalGatePolicy.model_validate_json(_POLICY_PATH.read_text())

    assert policy.expected_case_count == 236
    assert policy.expected_no_answer_case_count == 67
    assert policy.sidecar_artifact_sha256 == _SIDECAR_SHA256
    assert policy.min_rls_principal_count == 10
    assert policy.bootstrap_samples == 20_000
    assert policy.required_pg_textsearch_version == "1.3.1"
    assert policy.required_pg_textsearch_commit == _COMMIT
    assert policy.required_pg_textsearch_package_sha256 == _PACKAGE_SHA256
    assert policy.required_extension_binary_sha256 == _BINARY_SHA256
    assert policy.required_base_postgres_image_digest == _BASE_IMAGE
    assert policy.required_candidate_image_digest == _CANDIDATE_IMAGE
    assert policy.required_candidate_index_manifest_sha256 == _engine("pg_textsearch").index_manifest_sha256
    assert policy.required_build_recipe_sha256 == _DOCKERFILE_SHA256
    assert policy.required_prepare_sql_sha256 == _PREPARE_SQL_SHA256
    assert policy.required_baseline_indexes == _engine("postgres_fts").indexes
    assert policy.required_baseline_index_manifest_sha256 == _engine("postgres_fts").index_manifest_sha256
    assert policy.target_lexical_recall_at_5_gain == 0.03
    assert policy.target_ndcg_at_10_gain == 0.02


def test_gate_accepts_only_sparse_engine_change_and_bootstrap_is_deterministic() -> None:
    baseline = _report(candidate=False)
    candidate = _report(candidate=True)
    policy = _policy()

    first = evaluate_retrieval_gate(baseline, candidate, policy, evaluated_at=_NOW)
    second = evaluate_retrieval_gate(baseline, candidate, policy, evaluated_at=_NOW)

    assert first.accepted
    assert first.failure_codes == ()
    assert first.metrics == second.metrics
    assert first.slices == second.slices
    assert first.no_answer == second.no_answer
    assert first.no_answer.eligible_case_count == 67
    assert first.no_answer.passed
    assert len(first.metrics) == 7
    assert len(first.slices) == (3 + 5 + 2) * 7


@pytest.mark.parametrize("change", ["configuration", "models", "repo"])
def test_gate_rejects_changes_outside_sparse_allowlist(change: str) -> None:
    baseline = _report(candidate=False)
    candidate = _report(candidate=True)
    provenance = candidate.provenance
    if change == "configuration":
        configuration = provenance.configuration.model_copy(update={"rrf_k": 61})
        provenance = provenance.model_copy(
            update={
                "configuration": configuration,
                "configuration_sha256": canonical_sha256(configuration),
            }
        )
    elif change == "models":
        embedding = provenance.models.embedding.model_copy(update={"model": "other-embedding"})
        provenance = provenance.model_copy(
            update={"models": provenance.models.model_copy(update={"embedding": embedding})}
        )
    else:
        provenance = provenance.model_copy(update={"repo_sha": "b" * 40})

    with pytest.raises(RetrievalGateError, match="outside the sparse engine allowlist"):
        evaluate_retrieval_gate(
            baseline,
            _replace_report(candidate, provenance=provenance),
            _policy(),
        )


def test_gate_rejects_unpinned_extension_and_index_contract() -> None:
    baseline = _report(candidate=False)
    candidate = _report(candidate=True)
    engine = candidate.provenance.sparse_engine
    assert engine.pg_textsearch is not None
    wrong_extension = engine.pg_textsearch.model_copy(update={"version": "1.3.0"})
    wrong_engine = engine.model_copy(update={"pg_textsearch": wrong_extension})
    provenance = candidate.provenance.model_copy(update={"sparse_engine": wrong_engine})

    with pytest.raises(RetrievalGateError, match="pg_textsearch provenance"):
        evaluate_retrieval_gate(
            baseline,
            _replace_report(candidate, provenance=provenance),
            _policy(),
        )


def test_gate_rejects_sidecar_binary_and_baseline_index_mismatches() -> None:
    baseline = _report(candidate=False)
    candidate = _report(candidate=True)
    policy = _policy()

    wrong_sidecar = candidate.provenance.model_copy(update={"sidecar_artifact_sha256": "0" * 64})
    with pytest.raises(RetrievalGateError, match="Sidecar artifact"):
        evaluate_retrieval_gate(
            baseline,
            _replace_report(candidate, provenance=wrong_sidecar),
            policy,
        )

    candidate_engine = candidate.provenance.sparse_engine
    assert candidate_engine.pg_textsearch is not None
    wrong_binary = candidate_engine.pg_textsearch.model_copy(update={"extension_binary_sha256": "0" * 64})
    wrong_candidate_engine = candidate_engine.model_copy(update={"pg_textsearch": wrong_binary})
    wrong_candidate_provenance = candidate.provenance.model_copy(
        update={"sparse_engine": wrong_candidate_engine}
    )
    with pytest.raises(RetrievalGateError, match="pg_textsearch provenance"):
        evaluate_retrieval_gate(
            baseline,
            _replace_report(candidate, provenance=wrong_candidate_provenance),
            policy,
        )

    wrong_fallback = candidate_engine.pg_textsearch.model_copy(
        update={"legacy_fts_index_manifest_sha256": "0" * 64}
    )
    wrong_fallback_engine = candidate_engine.model_copy(update={"pg_textsearch": wrong_fallback})
    wrong_fallback_provenance = candidate.provenance.model_copy(
        update={"sparse_engine": wrong_fallback_engine}
    )
    with pytest.raises(RetrievalGateError, match="pg_textsearch provenance"):
        evaluate_retrieval_gate(
            baseline,
            _replace_report(candidate, provenance=wrong_fallback_provenance),
            policy,
        )

    alternate_index = _index(
        "ix_chunks_tsv",
        "gin",
        "CREATE INDEX ix_chunks_tsv ON public.chunks USING gin (to_tsvector(text_ru))",
    )
    alternate_engine = SparseEngineProvenance(
        backend="postgres_fts",
        indexes=(alternate_index,),
        index_manifest_sha256=canonical_sha256((alternate_index,)),
        pg_textsearch=None,
    )
    wrong_baseline_provenance = baseline.provenance.model_copy(update={"sparse_engine": alternate_engine})
    with pytest.raises(RetrievalGateError, match="baseline GIN index contract"):
        evaluate_retrieval_gate(
            _replace_report(baseline, provenance=wrong_baseline_provenance),
            candidate,
            policy,
        )


def test_gate_rejects_case_metadata_mismatch() -> None:
    baseline = _report(candidate=False)
    candidate = _report(candidate=True)
    changed = list(candidate.cases)
    changed[0] = changed[0].model_copy(update={"language": "en", "sparse_engine": "pg_textsearch_en"})

    with pytest.raises(RetrievalGateError, match="case metadata differs"):
        evaluate_retrieval_gate(
            baseline,
            _replace_report(candidate, cases=tuple(changed)),
            _policy(),
        )


def test_operational_evidence_requires_v3_bindings() -> None:
    payload = _operations(_engine("postgres_fts")).model_dump(mode="python")
    for field in (
        "schema_version",
        "corpus_snapshot_sha256",
        "candidate_image_digest",
        "candidate_index_manifest_sha256",
    ):
        incomplete = dict(payload)
        incomplete.pop(field)
        with pytest.raises(ValidationError, match=field):
            OperationalEvidence.model_validate(incomplete, strict=True)


def test_report_rejects_wrong_operational_corpus_and_sparse_route() -> None:
    candidate = _report(candidate=True)
    wrong_corpus = candidate.operations.model_copy(update={"corpus_snapshot_sha256": "0" * 64})
    with pytest.raises(ValidationError, match="corpus binding"):
        _replace_report(candidate, operations=wrong_corpus)

    changed = list(candidate.cases)
    ru_index = next(index for index, item in enumerate(changed) if item.language == "ru")
    changed[ru_index] = changed[ru_index].model_copy(update={"sparse_engine": "postgres_fts"})
    with pytest.raises(ValidationError, match="backend/language routing contract"):
        _replace_report(candidate, cases=tuple(changed))


def test_gate_revalidates_direct_model_copy_updates() -> None:
    baseline = _report(candidate=False)
    candidate = _report(candidate=True)
    wrong_operations = baseline.operations.model_copy(update={"corpus_snapshot_sha256": "0" * 64})
    bypassed_baseline = baseline.model_copy(update={"operations": wrong_operations})
    bypassed_candidate = candidate.model_copy(update={"operations": wrong_operations})

    with pytest.raises(RetrievalGateError, match="strict revalidation"):
        evaluate_retrieval_gate(bypassed_baseline, bypassed_candidate, _policy())

    changed = list(candidate.cases)
    ru_index = next(index for index, item in enumerate(changed) if item.language == "ru")
    changed[ru_index] = changed[ru_index].model_copy(update={"sparse_engine": "postgres_fts"})
    bypassed_candidate = candidate.model_copy(update={"cases": tuple(changed)})
    with pytest.raises(RetrievalGateError, match="strict revalidation"):
        evaluate_retrieval_gate(baseline, bypassed_candidate, _policy())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_image_digest", "sha256:" + "0" * 64),
        ("candidate_index_manifest_sha256", "0" * 64),
    ],
)
def test_gate_rejects_wrong_operational_candidate_binding(field: str, value: str) -> None:
    baseline = _report(candidate=False)
    candidate = _report(candidate=True)
    operations = baseline.operations.model_copy(update={field: value})

    with pytest.raises(RetrievalGateError, match="operational evidence candidate binding"):
        evaluate_retrieval_gate(
            _replace_report(baseline, operations=operations),
            _replace_report(candidate, operations=operations),
            _policy(),
        )


def test_gate_rejects_different_operational_evidence() -> None:
    baseline = _report(candidate=False)
    candidate = _report(candidate=True)
    different = candidate.operations.model_copy(update={"determinism_seconds": 46.0})

    with pytest.raises(RetrievalGateError, match="identical operational evidence"):
        evaluate_retrieval_gate(
            baseline,
            _replace_report(candidate, operations=different),
            _policy(),
        )


def test_gate_enforces_relative_and_absolute_latency_caps() -> None:
    baseline = _report(candidate=False)
    candidate = _report(candidate=True)
    candidate_load = _load(p95_ms=1_200.0)

    decision = evaluate_retrieval_gate(
        baseline,
        _replace_report(candidate, load=candidate_load),
        _policy(),
        evaluated_at=_NOW,
    )

    assert not decision.performance.latency_passed
    assert decision.performance.maximum_candidate_p95_ms == 1_100.0
    assert "latency_p95" in decision.failure_codes


def test_gate_fails_target_and_language_slice_regression() -> None:
    baseline = _report(candidate=False)
    candidate = _report(candidate=True, degraded_language="ru")

    decision = evaluate_retrieval_gate(baseline, candidate, _policy(), evaluated_at=_NOW)

    assert not decision.accepted
    assert any(code.startswith("slice_regression:language:ru:") for code in decision.failure_codes)
    assert "target_gain:lexical_recall_at_5" in decision.failure_codes


def test_no_answer_abstention_has_separate_paired_gate() -> None:
    baseline = _report(candidate=False)
    candidate = _report(candidate=True, degraded_no_answer=True)

    decision = evaluate_retrieval_gate(baseline, candidate, _policy(), evaluated_at=_NOW)

    assert not decision.no_answer.passed
    assert decision.no_answer.eligible_case_count == 67
    assert decision.no_answer.candidate_abstention_rate == 0.0
    assert "no_answer_abstention_regression" in decision.failure_codes


def test_no_answer_outcomes_are_not_case_metadata() -> None:
    baseline = _report(candidate=False)
    candidate = _report(candidate=True)
    changed = list(candidate.cases)
    no_answer_index = next(
        index for index, item in enumerate(changed) if not item.answerable and not item.abstained
    )
    changed[no_answer_index] = changed[no_answer_index].model_copy(update={"returned_count": 17})

    decision = evaluate_retrieval_gate(
        baseline,
        _replace_report(candidate, cases=tuple(changed)),
        _policy(),
        evaluated_at=_NOW,
    )

    assert decision.accepted


def test_gate_fails_closed_on_empty_required_content_slice() -> None:
    baseline = _report(candidate=False, empty_formula_slice=True)
    candidate = _report(candidate=True, empty_formula_slice=True)

    with pytest.raises(RetrievalGateError, match="required metric slice is empty"):
        evaluate_retrieval_gate(baseline, candidate, _policy())


def test_gate_requires_67_no_answer_cases_and_10_hashed_rls_principals() -> None:
    baseline = _report(candidate=False)
    candidate = _report(candidate=True)
    assert len(candidate.operations.rls_principals) == 10

    fewer = tuple(item for item in candidate.operations.rls_principals if item.principal_ref != _principal(0))
    reduced_operations = candidate.operations.model_copy(update={"rls_principals": fewer})
    with pytest.raises(RetrievalGateError, match="RLS principal coverage"):
        evaluate_retrieval_gate(
            baseline,
            _replace_report(candidate, operations=reduced_operations),
            _policy(),
        )

    with pytest.raises(ValidationError):
        RlsPrincipalEvidence(
            principal_ref="user01",
            probe_count=1,
            leak_count=0,
            evidence_sha256="0" * 64,
        )

    baseline_cases = list(baseline.cases)
    candidate_cases = list(candidate.cases)
    for cases in (baseline_cases, candidate_cases):
        cases[0] = cases[0].model_copy(
            update={
                "answerable": False,
                "metrics": None,
                "returned_count": 0,
                "abstained": True,
            }
        )
    with pytest.raises(RetrievalGateError, match="exactly 67 no-answer"):
        evaluate_retrieval_gate(
            _replace_report(baseline, cases=tuple(baseline_cases)),
            _replace_report(candidate, cases=tuple(candidate_cases)),
            _policy(),
        )


def test_gate_enforces_load_rls_lifecycle_determinism_and_rollback() -> None:
    baseline = _report(candidate=False)
    candidate = _report(candidate=True)
    bad_load = _load(p95_ms=2_000.0, duration_seconds=25.0)
    first_rls = candidate.operations.rls_principals[0].model_copy(update={"leak_count": 1})
    bad_operations = candidate.operations.model_copy(
        update={
            "rls_principals": (first_rls, *candidate.operations.rls_principals[1:]),
            "update_visible": False,
            "update_visibility_seconds": 601.0,
            "delete_hidden": False,
            "restart_recovered": False,
            "determinism_mismatches": 1,
            "rollback_succeeded": False,
            "rollback_backend": "pg_textsearch",
        }
    )
    baseline = _replace_report(baseline, operations=bad_operations)
    candidate = _replace_report(candidate, load=bad_load, operations=bad_operations)

    decision = evaluate_retrieval_gate(baseline, candidate, _policy(), evaluated_at=_NOW)

    assert not decision.accepted
    expected = {
        "latency_p95",
        "throughput",
        "rls_leak:candidate",
        "update_visibility",
        "delete_visibility",
        "restart_recovery",
        "determinism",
        "rollback",
        "rollback_binding",
    }
    assert expected <= set(decision.failure_codes)


def test_strict_models_reject_nonfinite_metric_and_tampered_index_hash() -> None:
    with pytest.raises(ValidationError):
        RetrievalCaseMetrics(
            recall_at_5=float("nan"),
            recall_at_10=0.5,
            mrr_at_10=0.5,
            ndcg_at_10=0.5,
            lexical_recall_at_5=0.5,
            lexical_recall_at_50=0.5,
            hybrid_union_recall_at_20=0.5,
        )
    with pytest.raises(ValidationError, match="index definition hash mismatch"):
        SparseIndexDefinition(
            name="ix_chunks_tsv",
            access_method="gin",
            canonical_definition="CREATE INDEX ix_chunks_tsv ON chunks USING gin (tsv)",
            definition_sha256="0" * 64,
        )
