"""Fail-closed release gate for sparse-retrieval backend changes.

The model-release gate intentionally freezes the complete retrieval
configuration.  This module is a separate protocol for comparing two runs
where only the sparse engine and its extension/index provenance may differ.
It contains no database or production mutation code.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Language = Literal["ru", "en", "zh"]
ContentType = Literal["text", "table", "formula", "figure", "scan"]
SparseBackend = Literal["postgres_fts", "pg_textsearch"]
SparseCaseEngine = Literal["postgres_fts", "pg_textsearch_ru", "pg_textsearch_en"]
MetricName = Literal[
    "recall_at_5",
    "recall_at_10",
    "mrr_at_10",
    "ndcg_at_10",
    "lexical_recall_at_5",
    "lexical_recall_at_50",
    "hybrid_union_recall_at_20",
]
SliceKind = Literal["language", "content", "scope"]

_METRICS: tuple[MetricName, ...] = (
    "recall_at_5",
    "recall_at_10",
    "mrr_at_10",
    "ndcg_at_10",
    "lexical_recall_at_5",
    "lexical_recall_at_50",
    "hybrid_union_recall_at_20",
)
_LANGUAGES: tuple[Language, ...] = ("ru", "en", "zh")
_CONTENT_TYPES: tuple[ContentType, ...] = ("text", "table", "formula", "figure", "scan")
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_IMAGE_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_GIT_SHA_PATTERN = r"^[0-9a-f]{40,64}$"
_SCOPE_PATTERN = r"^scope-sha256:[0-9a-f]{64}$"
_PRINCIPAL_PATTERN = r"^(?:scope|principal)-sha256:[0-9a-f]{64}$"
_CLUSTER_PATTERN = r"^cluster-sha256:[0-9a-f]{64}$"
_CASE_PATTERN = r"^ragq-[a-z0-9][a-z0-9._-]{7,63}$"


class RetrievalGateError(RuntimeError):
    """Sanitized invalid-input or comparability failure."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


def canonical_sha256(value: BaseModel | Mapping[str, Any] | Sequence[Any]) -> str:
    """Hash a model or JSON-compatible value with one canonical encoding."""

    payload = _canonical_payload(value)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_payload(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _canonical_payload(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_payload(item) for item in value]
    return value


class RuntimeModelRevision(_StrictModel):
    model: str = Field(min_length=1, max_length=256)
    declared_revision: str = Field(min_length=1, max_length=256)
    endpoint_metadata_sha256: str = Field(pattern=_SHA256_PATTERN)
    runtime_version_sha256: str = Field(pattern=_SHA256_PATTERN)
    weight_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_config_sha256: str = Field(pattern=_SHA256_PATTERN)


class RetrievalModelRevisions(_StrictModel):
    embedding: RuntimeModelRevision
    reranker: RuntimeModelRevision


class RetrievalConfiguration(_StrictModel):
    dense_top_k: int = Field(ge=1, le=1_000)
    sparse_top_k: int = Field(ge=1, le=1_000)
    rrf_k: int = Field(ge=1, le=10_000)
    rerank_top_k: int = Field(ge=1, le=1_000)
    final_top_k: int = Field(ge=1, le=100)
    rerank_min_score: float = Field(ge=-100, le=100)
    embedding_dim: int = Field(ge=1, le=65_536)
    visual_enabled: bool

    @model_validator(mode="after")
    def validate_candidate_depths(self) -> Self:
        if self.rerank_top_k > self.dense_top_k + self.sparse_top_k:
            raise ValueError("rerank_top_k exceeds the hybrid candidate pool")
        if self.final_top_k > self.rerank_top_k:
            raise ValueError("final_top_k exceeds rerank_top_k")
        return self


class OwnerScopeProvenance(_StrictModel):
    scope_id: str = Field(pattern=_SCOPE_PATTERN)
    case_count: int = Field(ge=1, le=500)
    document_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
    corpus_sha256: str = Field(pattern=_SHA256_PATTERN)


class SparseIndexDefinition(_StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,62}$")
    access_method: Literal["gin", "bm25"]
    text_config: Literal["russian", "english"] | None = None
    k1: float | None = Field(default=None, gt=0, le=10)
    b: float | None = Field(default=None, ge=0, le=1)
    canonical_definition: str = Field(min_length=1, max_length=16_384)
    definition_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        if self.canonical_definition != self.canonical_definition.strip():
            raise ValueError("canonical index definition contains surrounding whitespace")
        actual = hashlib.sha256(self.canonical_definition.encode()).hexdigest()
        if self.definition_sha256 != actual:
            raise ValueError("index definition hash mismatch")
        bm25_options = (self.text_config, self.k1, self.b)
        if self.access_method == "bm25" and any(value is None for value in bm25_options):
            raise ValueError("BM25 index requires text_config, k1 and b")
        if self.access_method == "gin" and any(value is not None for value in bm25_options):
            raise ValueError("GIN index cannot declare BM25 options")
        return self


class PgTextsearchProvenance(_StrictModel):
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    extension_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    package_sha256: str = Field(pattern=_SHA256_PATTERN)
    extension_binary_sha256: str = Field(pattern=_SHA256_PATTERN)
    extension_binary_path: Literal["/usr/lib/postgresql/17/lib/pg_textsearch.so"]
    extension_binary_bytes: int = Field(ge=1)
    container_image_digest: str = Field(pattern=_IMAGE_DIGEST_PATTERN)
    base_postgres_image_digest: str = Field(pattern=_IMAGE_DIGEST_PATTERN)
    build_recipe_sha256: str = Field(pattern=_SHA256_PATTERN)
    prepare_sql_sha256: str = Field(pattern=_SHA256_PATTERN)
    legacy_fts_index_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    spdx_license: Literal["PostgreSQL"]


class SparseEngineProvenance(_StrictModel):
    backend: SparseBackend
    indexes: tuple[SparseIndexDefinition, ...] = Field(min_length=1, max_length=16)
    index_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    pg_textsearch: PgTextsearchProvenance | None = None

    @model_validator(mode="after")
    def validate_engine(self) -> Self:
        names = [item.name for item in self.indexes]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("sparse indexes must be unique and sorted by name")
        if self.index_manifest_sha256 != canonical_sha256(self.indexes):
            raise ValueError("index manifest hash mismatch")
        if self.backend == "pg_textsearch":
            if self.pg_textsearch is None or any(item.access_method != "bm25" for item in self.indexes):
                raise ValueError("pg_textsearch requires BM25 indexes and extension provenance")
        elif self.pg_textsearch is not None or any(item.access_method != "gin" for item in self.indexes):
            raise ValueError("postgres_fts requires GIN indexes without extension provenance")
        return self


class RetrievalProvenance(_StrictModel):
    repo_sha: str = Field(pattern=_GIT_SHA_PATTERN)
    git_dirty: Literal[False]
    gold_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    sidecar_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    corpus_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    postgres_server_version_num: int = Field(ge=170_000, lt=180_000)
    reviewed_case_count: int = Field(ge=1, le=500)
    owner_scopes: tuple[OwnerScopeProvenance, ...] = Field(min_length=1, max_length=100)
    owner_scope_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    models: RetrievalModelRevisions
    configuration: RetrievalConfiguration
    configuration_sha256: str = Field(pattern=_SHA256_PATTERN)
    sparse_engine: SparseEngineProvenance

    @model_validator(mode="after")
    def validate_manifests(self) -> Self:
        scope_ids = [item.scope_id for item in self.owner_scopes]
        if scope_ids != sorted(scope_ids) or len(scope_ids) != len(set(scope_ids)):
            raise ValueError("owner scopes must be unique and sorted by scope_id")
        if self.owner_scope_manifest_sha256 != canonical_sha256(self.owner_scopes):
            raise ValueError("owner scope manifest hash mismatch")
        if self.configuration_sha256 != canonical_sha256(self.configuration):
            raise ValueError("retrieval configuration hash mismatch")
        return self


class RetrievalCaseMetrics(_StrictModel):
    recall_at_5: float = Field(ge=0, le=1)
    recall_at_10: float = Field(ge=0, le=1)
    mrr_at_10: float = Field(ge=0, le=1)
    ndcg_at_10: float = Field(ge=0, le=1)
    lexical_recall_at_5: float = Field(ge=0, le=1)
    lexical_recall_at_50: float = Field(ge=0, le=1)
    hybrid_union_recall_at_20: float = Field(ge=0, le=1)


class RetrievalCaseResult(_StrictModel):
    case_id: str = Field(pattern=_CASE_PATTERN)
    gold_case_sha256: str = Field(pattern=_SHA256_PATTERN)
    reviewed: Literal[True]
    scope_id: str = Field(pattern=_SCOPE_PATTERN)
    cluster_id: str = Field(pattern=_CLUSTER_PATTERN)
    language: Language
    content_types: tuple[ContentType, ...] = Field(min_length=1, max_length=5)
    answerable: bool
    sparse_engine: SparseCaseEngine
    metrics: RetrievalCaseMetrics | None
    returned_count: int = Field(ge=0, le=1_000)
    abstained: bool
    retrieval_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if len(self.content_types) != len(set(self.content_types)):
            raise ValueError("case content types must be unique")
        if self.answerable != (self.metrics is not None):
            raise ValueError("only answerable cases may contain retrieval metrics")
        if self.abstained != (self.returned_count == 0):
            raise ValueError("abstention must match the returned retrieval count")
        return self


class LoadEvidence(_StrictModel):
    concurrency: int = Field(ge=1)
    request_count: int = Field(ge=1)
    completed_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    duration_seconds: float = Field(gt=0)
    p95_latency_ms: float = Field(gt=0)
    throughput_rps: float = Field(gt=0)
    raw_observations_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_aggregate(self) -> Self:
        if self.completed_count + self.error_count != self.request_count:
            raise ValueError("load counts do not add up")
        expected = self.completed_count / self.duration_seconds
        if not math.isclose(self.throughput_rps, expected, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("load throughput does not match raw counts and duration")
        return self


class RlsPrincipalEvidence(_StrictModel):
    principal_ref: str = Field(pattern=_PRINCIPAL_PATTERN)
    probe_count: int = Field(ge=1)
    leak_count: int = Field(ge=0)
    evidence_sha256: str = Field(pattern=_SHA256_PATTERN)


class OperationalEvidence(_StrictModel):
    schema_version: Literal["rag-operational-evidence-v3"]
    corpus_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_image_digest: str = Field(pattern=_IMAGE_DIGEST_PATTERN)
    candidate_index_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    rls_principals: tuple[RlsPrincipalEvidence, ...] = Field(min_length=1, max_length=100)
    update_visible: bool
    update_visibility_seconds: float = Field(ge=0)
    delete_hidden: bool
    delete_visibility_seconds: float = Field(ge=0)
    restart_recovered: bool
    restart_recovery_seconds: float = Field(ge=0)
    determinism_replays: int = Field(ge=2)
    determinism_mismatches: int = Field(ge=0)
    determinism_seconds: float = Field(ge=0)
    rollback_succeeded: bool
    rollback_seconds: float = Field(ge=0)
    rollback_backend: SparseBackend
    rollback_index_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    raw_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_scopes(self) -> Self:
        principal_refs = [item.principal_ref for item in self.rls_principals]
        if principal_refs != sorted(principal_refs) or len(principal_refs) != len(set(principal_refs)):
            raise ValueError("RLS evidence principals must be unique and sorted")
        return self


class RetrievalReport(_StrictModel):
    schema_version: Literal["rag-retrieval-report-v2"] = "rag-retrieval-report-v2"
    evaluated_at: datetime
    provenance: RetrievalProvenance
    case_count: int = Field(ge=1, le=500)
    case_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    cases: tuple[RetrievalCaseResult, ...] = Field(min_length=1, max_length=500)
    load: LoadEvidence
    operations: OperationalEvidence

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.evaluated_at.tzinfo is None or self.evaluated_at.utcoffset() is None:
            raise ValueError("evaluated_at must be timezone-aware")
        case_ids = [item.case_id for item in self.cases]
        if case_ids != sorted(case_ids) or len(case_ids) != len(set(case_ids)):
            raise ValueError("retrieval cases must be unique and sorted by case_id")
        if self.case_count != len(self.cases):
            raise ValueError("case_count does not match cases")
        if self.provenance.reviewed_case_count != self.case_count:
            raise ValueError("reviewed case count does not match report")
        if self.operations.corpus_snapshot_sha256 != self.provenance.corpus_snapshot_sha256:
            raise ValueError("operational evidence corpus binding mismatch")
        expected_engines: dict[str, SparseCaseEngine] = {
            item.case_id: (
                "postgres_fts"
                if self.provenance.sparse_engine.backend == "postgres_fts" or item.language == "zh"
                else "pg_textsearch_ru"
                if item.language == "ru"
                else "pg_textsearch_en"
            )
            for item in self.cases
        }
        if any(item.sparse_engine != expected_engines[item.case_id] for item in self.cases):
            raise ValueError("case sparse engine does not match backend/language routing contract")
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
            for item in self.cases
        ]
        if self.case_manifest_sha256 != canonical_sha256(manifest):
            raise ValueError("case manifest hash mismatch")

        declared_scopes = {item.scope_id: item for item in self.provenance.owner_scopes}
        observed_counts: dict[str, int] = defaultdict(int)
        for item in self.cases:
            observed_counts[item.scope_id] += 1
        if set(observed_counts) != set(declared_scopes) or any(
            observed_counts[scope_id] != declared.case_count for scope_id, declared in declared_scopes.items()
        ):
            raise ValueError("case owner scopes do not match provenance")
        principal_refs = {item.principal_ref for item in self.operations.rls_principals}
        if not set(declared_scopes).issubset(principal_refs):
            raise ValueError("RLS evidence does not cover every declared Gold owner scope")
        if any(
            not item.startswith("principal-sha256:") for item in principal_refs.difference(declared_scopes)
        ):
            raise ValueError("extra RLS evidence must use hashed principal references")
        return self


class RequiredBm25Index(_StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,62}$")
    text_config: Literal["russian", "english"]
    k1: float = Field(gt=0, le=10)
    b: float = Field(ge=0, le=1)
    canonical_definition: str = Field(min_length=1, max_length=16_384)
    definition_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        if self.canonical_definition != self.canonical_definition.strip():
            raise ValueError("canonical required index definition has surrounding whitespace")
        if hashlib.sha256(self.canonical_definition.encode()).hexdigest() != self.definition_sha256:
            raise ValueError("required BM25 index definition hash mismatch")
        return self


class RetrievalGatePolicy(_StrictModel):
    schema_version: Literal["rag-retrieval-policy-v2"] = "rag-retrieval-policy-v2"
    policy_id: str = Field(min_length=1, max_length=128)
    gold_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    sidecar_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    corpus_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_case_count: Literal[236] = 236
    expected_no_answer_case_count: Literal[67] = 67
    min_owner_scope_count: int = Field(default=2, ge=2, le=100)
    min_rls_principal_count: int = Field(default=10, ge=10, le=100)
    baseline_backend: Literal["postgres_fts"] = "postgres_fts"
    candidate_backend: Literal["pg_textsearch"] = "pg_textsearch"
    required_baseline_indexes: tuple[SparseIndexDefinition, ...] = Field(min_length=1, max_length=8)
    required_baseline_index_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    required_pg_textsearch_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    required_pg_textsearch_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    required_pg_textsearch_package_sha256: str = Field(pattern=_SHA256_PATTERN)
    required_extension_binary_sha256: str = Field(pattern=_SHA256_PATTERN)
    required_base_postgres_image_digest: str = Field(pattern=_IMAGE_DIGEST_PATTERN)
    required_candidate_image_digest: str = Field(pattern=_IMAGE_DIGEST_PATTERN)
    required_candidate_index_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    required_build_recipe_sha256: str = Field(pattern=_SHA256_PATTERN)
    required_prepare_sql_sha256: str = Field(pattern=_SHA256_PATTERN)
    required_candidate_indexes: tuple[RequiredBm25Index, ...] = Field(min_length=1, max_length=8)
    bootstrap_samples: int = Field(default=20_000, ge=1_000, le=100_000)
    bootstrap_seed: int = Field(default=2026071409, ge=0)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1)
    global_max_regression: float = Field(default=0.01, ge=0, le=0.2)
    slice_max_regression: float = Field(default=0.01, ge=0, le=0.2)
    target_lexical_recall_at_5_gain: float = Field(default=0.03, gt=0, le=1)
    target_ndcg_at_10_gain: float = Field(default=0.02, gt=0, le=1)
    max_p95_ratio: float = Field(default=1.10, ge=1, le=10)
    max_p95_absolute_increase_ms: float = Field(default=250, ge=0)
    min_throughput_ratio: float = Field(default=0.90, gt=0, le=1)
    min_load_concurrency: int = Field(default=10, ge=1)
    min_load_requests: int = Field(default=200, ge=1)
    min_determinism_replays: int = Field(default=3, ge=2)
    max_operational_seconds: float = Field(default=600, gt=0)
    required_languages: tuple[Language, ...] = _LANGUAGES
    required_content_types: tuple[ContentType, ...] = _CONTENT_TYPES

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.required_languages != _LANGUAGES:
            raise ValueError("policy must require RU, EN and ZH slices in canonical order")
        if self.required_content_types != _CONTENT_TYPES:
            raise ValueError("policy must require every content slice in canonical order")
        if any(item.access_method != "gin" for item in self.required_baseline_indexes):
            raise ValueError("baseline policy requires only GIN indexes")
        if self.required_baseline_index_manifest_sha256 != canonical_sha256(self.required_baseline_indexes):
            raise ValueError("baseline policy index manifest hash mismatch")
        names = [item.name for item in self.required_candidate_indexes]
        configs = [item.text_config for item in self.required_candidate_indexes]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("required candidate indexes must be unique and sorted")
        if set(configs) != {"russian", "english"} or len(configs) != 2:
            raise ValueError("candidate policy requires exactly one Russian and one English BM25 index")
        candidate_indexes = tuple(
            SparseIndexDefinition(
                name=item.name,
                access_method="bm25",
                text_config=item.text_config,
                k1=item.k1,
                b=item.b,
                canonical_definition=item.canonical_definition,
                definition_sha256=item.definition_sha256,
            )
            for item in self.required_candidate_indexes
        )
        if self.required_candidate_index_manifest_sha256 != canonical_sha256(candidate_indexes):
            raise ValueError("candidate policy index manifest hash mismatch")
        return self


class MetricDecision(_StrictModel):
    metric: MetricName
    eligible_case_count: int = Field(ge=1)
    cluster_count: int = Field(ge=2)
    baseline: float = Field(ge=0, le=1)
    candidate: float = Field(ge=0, le=1)
    improvement: float = Field(ge=-1, le=1)
    ci_low: float = Field(ge=-1, le=1)
    ci_high: float = Field(ge=-1, le=1)
    noninferiority_passed: bool
    target_gain: float | None = Field(default=None, ge=0, le=1)
    target_passed: bool
    passed: bool


class SliceDecision(_StrictModel):
    kind: SliceKind
    slice_id: str = Field(min_length=2, max_length=128)
    metric: MetricName
    eligible_case_count: int = Field(ge=1)
    cluster_count: int = Field(ge=2)
    baseline: float = Field(ge=0, le=1)
    candidate: float = Field(ge=0, le=1)
    improvement: float = Field(ge=-1, le=1)
    ci_low: float = Field(ge=-1, le=1)
    ci_high: float = Field(ge=-1, le=1)
    passed: bool


class NoAnswerDecision(_StrictModel):
    eligible_case_count: Literal[67]
    cluster_count: int = Field(ge=2)
    baseline_abstention_rate: float = Field(ge=0, le=1)
    candidate_abstention_rate: float = Field(ge=0, le=1)
    improvement: float = Field(ge=-1, le=1)
    ci_low: float = Field(ge=-1, le=1)
    ci_high: float = Field(ge=-1, le=1)
    noninferiority_margin: float = Field(ge=0, le=1)
    passed: bool


class PerformanceDecision(_StrictModel):
    baseline_p95_ms: float = Field(gt=0)
    candidate_p95_ms: float = Field(gt=0)
    maximum_candidate_p95_ms: float = Field(gt=0)
    latency_passed: bool
    baseline_throughput_rps: float = Field(gt=0)
    candidate_throughput_rps: float = Field(gt=0)
    throughput_ratio: float = Field(gt=0)
    throughput_passed: bool
    load_integrity_passed: bool


class RetrievalGateDecision(_StrictModel):
    schema_version: Literal["rag-retrieval-decision-v2"] = "rag-retrieval-decision-v2"
    evaluated_at: datetime
    policy_id: str
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    baseline_report_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_report_sha256: str = Field(pattern=_SHA256_PATTERN)
    accepted: bool
    failure_codes: tuple[str, ...]
    metrics: tuple[MetricDecision, ...] = Field(min_length=len(_METRICS))
    slices: tuple[SliceDecision, ...] = Field(min_length=1)
    no_answer: NoAnswerDecision
    performance: PerformanceDecision

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.evaluated_at.tzinfo is None or self.evaluated_at.utcoffset() is None:
            raise ValueError("decision evaluated_at must be timezone-aware")
        if self.accepted == bool(self.failure_codes):
            raise ValueError("accepted decision and failure codes are inconsistent")
        if len(self.failure_codes) != len(set(self.failure_codes)):
            raise ValueError("failure codes must be unique")
        if tuple(item.metric for item in self.metrics) != _METRICS:
            raise ValueError("global metric decisions are incomplete or unordered")
        return self


def _comparable_provenance(provenance: RetrievalProvenance) -> dict[str, Any]:
    payload = provenance.model_dump(mode="json")
    payload.pop("sparse_engine")
    return payload


def _index_contract(
    index: SparseIndexDefinition,
) -> tuple[str, str | None, float | None, float | None, str]:
    return index.name, index.text_config, index.k1, index.b, index.definition_sha256


def _required_index_contract(index: RequiredBm25Index) -> tuple[str, str, float, float, str]:
    return index.name, index.text_config, index.k1, index.b, index.definition_sha256


def _validate_comparability(
    baseline: RetrievalReport,
    candidate: RetrievalReport,
    policy: RetrievalGatePolicy,
) -> None:
    if (
        baseline.case_count != policy.expected_case_count
        or candidate.case_count != policy.expected_case_count
    ):
        raise RetrievalGateError("retrieval reports must contain exactly 236 reviewed cases")
    for report in (baseline, candidate):
        if report.provenance.gold_artifact_sha256 != policy.gold_artifact_sha256:
            raise RetrievalGateError("Gold artifact does not match retrieval policy")
        if report.provenance.sidecar_artifact_sha256 != policy.sidecar_artifact_sha256:
            raise RetrievalGateError("private Sidecar artifact does not match retrieval policy")
        if report.provenance.corpus_snapshot_sha256 != policy.corpus_snapshot_sha256:
            raise RetrievalGateError("corpus snapshot does not match retrieval policy")
        if len(report.provenance.owner_scopes) < policy.min_owner_scope_count:
            raise RetrievalGateError("owner-scope coverage is below policy minimum")
        if len(report.operations.rls_principals) < policy.min_rls_principal_count:
            raise RetrievalGateError("RLS principal coverage is below policy minimum")
        if sum(not item.answerable for item in report.cases) != policy.expected_no_answer_case_count:
            raise RetrievalGateError("retrieval report must contain exactly 67 no-answer cases")
    if _comparable_provenance(baseline.provenance) != _comparable_provenance(candidate.provenance):
        raise RetrievalGateError("provenance differs outside the sparse engine allowlist")
    baseline_engine = baseline.provenance.sparse_engine
    if baseline_engine.backend != policy.baseline_backend:
        raise RetrievalGateError("baseline sparse backend does not match policy")
    if (
        baseline_engine.indexes != policy.required_baseline_indexes
        or baseline_engine.index_manifest_sha256 != policy.required_baseline_index_manifest_sha256
    ):
        raise RetrievalGateError("baseline GIN index contract does not match policy")
    candidate_engine = candidate.provenance.sparse_engine
    if candidate_engine.backend != policy.candidate_backend or candidate_engine.pg_textsearch is None:
        raise RetrievalGateError("candidate sparse backend does not match policy")
    extension = candidate_engine.pg_textsearch
    if (
        extension.version != policy.required_pg_textsearch_version
        or extension.extension_commit != policy.required_pg_textsearch_commit
        or extension.package_sha256 != policy.required_pg_textsearch_package_sha256
        or extension.extension_binary_sha256 != policy.required_extension_binary_sha256
        or extension.base_postgres_image_digest != policy.required_base_postgres_image_digest
        or extension.container_image_digest != policy.required_candidate_image_digest
        or extension.build_recipe_sha256 != policy.required_build_recipe_sha256
        or extension.prepare_sql_sha256 != policy.required_prepare_sql_sha256
        or extension.legacy_fts_index_manifest_sha256 != policy.required_baseline_index_manifest_sha256
        or extension.spdx_license != "PostgreSQL"
    ):
        raise RetrievalGateError("pg_textsearch provenance does not match policy")
    actual_indexes = tuple(_index_contract(item) for item in candidate_engine.indexes)
    required_indexes = tuple(_required_index_contract(item) for item in policy.required_candidate_indexes)
    if actual_indexes != required_indexes:
        raise RetrievalGateError("candidate BM25 index contract does not match policy")
    if candidate_engine.index_manifest_sha256 != policy.required_candidate_index_manifest_sha256:
        raise RetrievalGateError("candidate BM25 index manifest does not match policy")
    if baseline.operations != candidate.operations:
        raise RetrievalGateError("paired reports must use identical operational evidence")
    if (
        baseline.operations.candidate_image_digest != extension.container_image_digest
        or baseline.operations.candidate_index_manifest_sha256 != candidate_engine.index_manifest_sha256
    ):
        raise RetrievalGateError("operational evidence candidate binding mismatch")

    baseline_cases = {item.case_id: item for item in baseline.cases}
    candidate_cases = {item.case_id: item for item in candidate.cases}
    if baseline_cases.keys() != candidate_cases.keys():
        raise RetrievalGateError("paired report case IDs differ")
    outcome_fields = {
        "sparse_engine",
        "metrics",
        "returned_count",
        "abstained",
        "retrieval_ms",
    }
    for case_id, left in baseline_cases.items():
        right = candidate_cases[case_id]
        left_metadata = left.model_dump(mode="json", exclude=outcome_fields)
        right_metadata = right.model_dump(mode="json", exclude=outcome_fields)
        if left_metadata != right_metadata:
            raise RetrievalGateError("paired report case metadata differs")


def _metric_value(case: RetrievalCaseResult, metric: MetricName) -> float | None:
    return None if case.metrics is None else float(getattr(case.metrics, metric))


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise RetrievalGateError("bootstrap produced no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _cluster_bootstrap(
    rows: Sequence[tuple[str, float, float]],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float, float, float, float, int]:
    grouped: dict[str, list[float]] = defaultdict(list)
    baseline_values: list[float] = []
    candidate_values: list[float] = []
    for cluster_id, baseline, candidate in rows:
        if not all(math.isfinite(value) for value in (baseline, candidate)):
            raise RetrievalGateError("retrieval metric contains a non-finite value")
        grouped[cluster_id].append(candidate - baseline)
        baseline_values.append(baseline)
        candidate_values.append(candidate)
    if not rows:
        raise RetrievalGateError("required metric slice is empty")
    if len(grouped) < 2:
        raise RetrievalGateError("required metric slice has fewer than two bootstrap clusters")

    clusters = tuple((sum(values), len(values)) for _, values in sorted(grouped.items()))
    rng = random.Random(seed)
    bootstrapped: list[float] = []
    for _ in range(samples):
        total = 0.0
        count = 0
        for _ in range(len(clusters)):
            cluster_sum, cluster_count = clusters[rng.randrange(len(clusters))]
            total += cluster_sum
            count += cluster_count
        bootstrapped.append(total / count)
    tail = (1 - confidence_level) / 2
    baseline_mean = math.fsum(baseline_values) / len(baseline_values)
    candidate_mean = math.fsum(candidate_values) / len(candidate_values)
    improvement = candidate_mean - baseline_mean
    return (
        baseline_mean,
        candidate_mean,
        improvement,
        _quantile(bootstrapped, tail),
        _quantile(bootstrapped, 1 - tail),
        len(clusters),
    )


def _rows_for_metric(
    baseline: Mapping[str, RetrievalCaseResult],
    candidate: Mapping[str, RetrievalCaseResult],
    metric: MetricName,
    *,
    predicate: Callable[[RetrievalCaseResult], bool],
) -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    for case_id in sorted(baseline):
        left = baseline[case_id]
        if not predicate(left):
            continue
        right = candidate[case_id]
        left_value = _metric_value(left, metric)
        right_value = _metric_value(right, metric)
        if (left_value is None) != (right_value is None):
            raise RetrievalGateError("paired metric eligibility differs")
        if left_value is not None and right_value is not None:
            rows.append((left.cluster_id, left_value, right_value))
    return rows


def _bootstrap_seed(policy: RetrievalGatePolicy, *parts: str) -> int:
    digest = hashlib.sha256(":".join(parts).encode()).digest()
    return policy.bootstrap_seed ^ int.from_bytes(digest[:8], "big")


def _target_gain(policy: RetrievalGatePolicy, metric: MetricName) -> float | None:
    if metric == "lexical_recall_at_5":
        return policy.target_lexical_recall_at_5_gain
    if metric == "ndcg_at_10":
        return policy.target_ndcg_at_10_gain
    return None


def _p95(values: Sequence[float]) -> float:
    if not values:
        raise RetrievalGateError("latency sample is empty")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _language_predicate(language: Language) -> Callable[[RetrievalCaseResult], bool]:
    def matches(case: RetrievalCaseResult) -> bool:
        return case.language == language

    return matches


def _content_predicate(content: ContentType) -> Callable[[RetrievalCaseResult], bool]:
    def matches(case: RetrievalCaseResult) -> bool:
        return content in case.content_types

    return matches


def _scope_predicate(scope_id: str) -> Callable[[RetrievalCaseResult], bool]:
    def matches(case: RetrievalCaseResult) -> bool:
        return case.scope_id == scope_id

    return matches


def _answerable_predicate(
    predicate: Callable[[RetrievalCaseResult], bool],
) -> Callable[[RetrievalCaseResult], bool]:
    def matches(case: RetrievalCaseResult) -> bool:
        return case.answerable and predicate(case)

    return matches


def _is_answerable(case: RetrievalCaseResult) -> bool:
    return case.answerable


def evaluate_retrieval_gate(
    baseline: RetrievalReport,
    candidate: RetrievalReport,
    policy: RetrievalGatePolicy,
    *,
    evaluated_at: datetime | None = None,
) -> RetrievalGateDecision:
    """Evaluate a paired sparse-backend experiment without mutating production."""

    try:
        baseline = RetrievalReport.model_validate(baseline.model_dump(mode="python"), strict=True)
        candidate = RetrievalReport.model_validate(candidate.model_dump(mode="python"), strict=True)
        policy = RetrievalGatePolicy.model_validate(policy.model_dump(mode="python"), strict=True)
    except ValidationError:
        raise RetrievalGateError("retrieval gate input failed strict revalidation") from None
    _validate_comparability(baseline, candidate, policy)
    baseline_by_id = {item.case_id: item for item in baseline.cases}
    candidate_by_id = {item.case_id: item for item in candidate.cases}
    failure_codes: list[str] = []
    metric_decisions: list[MetricDecision] = []

    for metric in _METRICS:
        rows = _rows_for_metric(
            baseline_by_id,
            candidate_by_id,
            metric,
            predicate=_is_answerable,
        )
        baseline_mean, candidate_mean, improvement, ci_low, ci_high, clusters = _cluster_bootstrap(
            rows,
            samples=policy.bootstrap_samples,
            seed=_bootstrap_seed(policy, "global", metric),
            confidence_level=policy.confidence_level,
        )
        noninferiority_passed = ci_low >= -policy.global_max_regression
        target_gain = _target_gain(policy, metric)
        target_passed = target_gain is None or ci_low >= target_gain
        passed = noninferiority_passed and target_passed
        if not noninferiority_passed:
            failure_codes.append(f"global_regression:{metric}")
        if not target_passed:
            failure_codes.append(f"target_gain:{metric}")
        metric_decisions.append(
            MetricDecision(
                metric=metric,
                eligible_case_count=len(rows),
                cluster_count=clusters,
                baseline=baseline_mean,
                candidate=candidate_mean,
                improvement=improvement,
                ci_low=ci_low,
                ci_high=ci_high,
                noninferiority_passed=noninferiority_passed,
                target_gain=target_gain,
                target_passed=target_passed,
                passed=passed,
            )
        )

    slice_specs: list[tuple[SliceKind, str, Callable[[RetrievalCaseResult], bool]]] = []
    slice_specs.extend(
        ("language", language, _language_predicate(language)) for language in policy.required_languages
    )
    slice_specs.extend(
        ("content", content, _content_predicate(content)) for content in policy.required_content_types
    )
    slice_specs.extend(
        ("scope", scope.scope_id, _scope_predicate(scope.scope_id))
        for scope in baseline.provenance.owner_scopes
    )

    slice_decisions: list[SliceDecision] = []
    for kind, slice_id, predicate in slice_specs:
        for metric in _METRICS:
            rows = _rows_for_metric(
                baseline_by_id,
                candidate_by_id,
                metric,
                predicate=_answerable_predicate(predicate),
            )
            baseline_mean, candidate_mean, improvement, ci_low, ci_high, clusters = _cluster_bootstrap(
                rows,
                samples=policy.bootstrap_samples,
                seed=_bootstrap_seed(policy, kind, slice_id, metric),
                confidence_level=policy.confidence_level,
            )
            passed = ci_low >= -policy.slice_max_regression
            if not passed:
                failure_codes.append(f"slice_regression:{kind}:{slice_id}:{metric}")
            slice_decisions.append(
                SliceDecision(
                    kind=kind,
                    slice_id=slice_id,
                    metric=metric,
                    eligible_case_count=len(rows),
                    cluster_count=clusters,
                    baseline=baseline_mean,
                    candidate=candidate_mean,
                    improvement=improvement,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    passed=passed,
                )
            )

    no_answer_rows = [
        (
            left.cluster_id,
            float(left.abstained),
            float(candidate_by_id[case_id].abstained),
        )
        for case_id, left in sorted(baseline_by_id.items())
        if not left.answerable
    ]
    (
        baseline_abstention,
        candidate_abstention,
        no_answer_improvement,
        no_answer_ci_low,
        no_answer_ci_high,
        no_answer_clusters,
    ) = _cluster_bootstrap(
        no_answer_rows,
        samples=policy.bootstrap_samples,
        seed=_bootstrap_seed(policy, "no_answer", "abstention_rate"),
        confidence_level=policy.confidence_level,
    )
    no_answer_passed = no_answer_ci_low >= -policy.global_max_regression
    if not no_answer_passed:
        failure_codes.append("no_answer_abstention_regression")
    no_answer = NoAnswerDecision(
        eligible_case_count=policy.expected_no_answer_case_count,
        cluster_count=no_answer_clusters,
        baseline_abstention_rate=baseline_abstention,
        candidate_abstention_rate=candidate_abstention,
        improvement=no_answer_improvement,
        ci_low=no_answer_ci_low,
        ci_high=no_answer_ci_high,
        noninferiority_margin=policy.global_max_regression,
        passed=no_answer_passed,
    )

    left_load = baseline.load
    right_load = candidate.load
    load_integrity_passed = all(
        load.concurrency >= policy.min_load_concurrency
        and load.request_count >= policy.min_load_requests
        and load.error_count == 0
        for load in (left_load, right_load)
    )
    if not load_integrity_passed:
        failure_codes.append("load_integrity")
    max_candidate_p95 = min(
        left_load.p95_latency_ms * policy.max_p95_ratio,
        left_load.p95_latency_ms + policy.max_p95_absolute_increase_ms,
    )
    latency_passed = right_load.p95_latency_ms <= max_candidate_p95
    if not latency_passed:
        failure_codes.append("latency_p95")
    throughput_ratio = right_load.throughput_rps / left_load.throughput_rps
    throughput_passed = throughput_ratio >= policy.min_throughput_ratio
    if not throughput_passed:
        failure_codes.append("throughput")

    for label, report in (("baseline", baseline), ("candidate", candidate)):
        if any(item.leak_count != 0 for item in report.operations.rls_principals):
            failure_codes.append(f"rls_leak:{label}")
    operations = candidate.operations
    timed_checks = (
        ("update_visibility", operations.update_visible, operations.update_visibility_seconds),
        ("delete_visibility", operations.delete_hidden, operations.delete_visibility_seconds),
        ("restart_recovery", operations.restart_recovered, operations.restart_recovery_seconds),
        ("rollback", operations.rollback_succeeded, operations.rollback_seconds),
    )
    for name, succeeded, duration in timed_checks:
        if not succeeded or duration > policy.max_operational_seconds:
            failure_codes.append(name)
    if (
        operations.determinism_replays < policy.min_determinism_replays
        or operations.determinism_mismatches != 0
        or operations.determinism_seconds > policy.max_operational_seconds
    ):
        failure_codes.append("determinism")
    baseline_engine = baseline.provenance.sparse_engine
    if (
        operations.rollback_backend != baseline_engine.backend
        or operations.rollback_index_manifest_sha256 != baseline_engine.index_manifest_sha256
    ):
        failure_codes.append("rollback_binding")

    performance = PerformanceDecision(
        baseline_p95_ms=left_load.p95_latency_ms,
        candidate_p95_ms=right_load.p95_latency_ms,
        maximum_candidate_p95_ms=max_candidate_p95,
        latency_passed=latency_passed,
        baseline_throughput_rps=left_load.throughput_rps,
        candidate_throughput_rps=right_load.throughput_rps,
        throughput_ratio=throughput_ratio,
        throughput_passed=throughput_passed,
        load_integrity_passed=load_integrity_passed,
    )
    unique_failures = tuple(dict.fromkeys(failure_codes))
    return RetrievalGateDecision(
        evaluated_at=evaluated_at or datetime.now(UTC),
        policy_id=policy.policy_id,
        policy_sha256=canonical_sha256(policy),
        baseline_report_sha256=canonical_sha256(baseline),
        candidate_report_sha256=canonical_sha256(candidate),
        accepted=not unique_failures,
        failure_codes=unique_failures,
        metrics=tuple(metric_decisions),
        slices=tuple(slice_decisions),
        no_answer=no_answer,
        performance=performance,
    )


__all__ = [
    "LoadEvidence",
    "MetricDecision",
    "NoAnswerDecision",
    "OperationalEvidence",
    "OwnerScopeProvenance",
    "PgTextsearchProvenance",
    "RequiredBm25Index",
    "RetrievalCaseMetrics",
    "RetrievalCaseResult",
    "RetrievalConfiguration",
    "RetrievalGateDecision",
    "RetrievalGateError",
    "RetrievalGatePolicy",
    "RetrievalModelRevisions",
    "RetrievalProvenance",
    "RetrievalReport",
    "RlsPrincipalEvidence",
    "RuntimeModelRevision",
    "SparseCaseEngine",
    "SparseEngineProvenance",
    "SparseIndexDefinition",
    "canonical_sha256",
    "evaluate_retrieval_gate",
]
