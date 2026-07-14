"""Run a private RAG baseline against loopback production services without DB writes."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import subprocess
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from sqlalchemy import text as sql
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from rag_app.config import settings
from rag_app.eval.baseline import (
    BaselineConfiguration,
    BaselineEvaluationError,
    BaselineModelIdentifiers,
    BaselineModelRevisions,
    BaselineObservation,
    BaselineProvenance,
    RetrievedUnit,
    RuntimeModelRevision,
    evaluate_baseline,
    require_loopback_database_url,
    require_loopback_endpoint,
    require_loopback_url,
)
from rag_app.eval.gold_set import (
    DocumentSnapshot,
    GoldRecord,
    bytes_sha256,
    ensure_private_gold_path,
    make_document_ref,
    make_scope_id,
    parse_gold_set_bytes,
    parsed_chunks_sha256,
    text_sha256,
)
from rag_app.eval.private_artifacts import read_private_bytes, write_private_json_fresh
from rag_app.eval.private_sidecar import (
    PrivateSidecarRecord,
    RetrievalProbe,
    bind_gold_sidecar,
    parse_private_sidecar_bytes,
)
from rag_app.eval.report_attestation import (
    atomic_write_attestation,
    build_case_attestations,
    create_report_attestation,
    load_hmac_key,
    verify_report_attestation,
)
from rag_app.llm.embeddings import Embedder, Reranker
from rag_app.llm.visual import VisualEmbedder
from rag_app.llm.visual_reranker import VisualReranker
from rag_app.rag.chat import CHAT_SYSTEM_PROMPT, ChatEngine
from rag_app.rag.retrieve import Retriever
from rag_app.storage.s3 import Storage

_NO_RESULTS_ANSWER = "В библиотеке не нашлось проиндексированных фрагментов по этому запросу."
_MODEL_CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
)
_MODEL_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")
_BASELINE_TEMPERATURE = 0.2
_BASELINE_TOP_P = 0.8
_BASELINE_OUTPUT_TOKENS = 2048
_BASELINE_SEED_NAMESPACE = 2026071300
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _artifact_sha256(path: Path) -> str:
    return hashlib.sha256(path.expanduser().resolve().read_bytes()).hexdigest()


def _git_state(repository_root: Path) -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(  # noqa: S603
                ["git", "status", "--porcelain"],  # noqa: S607
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        return None, dirty
    return revision, dirty


def _corpus_fingerprint(records: list[GoldRecord]) -> tuple[str, int, int]:
    scopes: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        documents = scopes.setdefault(record.scope_id, {})
        for snapshot in record.document_scope:
            payload = snapshot.model_dump(mode="json")
            previous = documents.setdefault(snapshot.document_ref, payload)
            if previous != payload:
                raise BaselineEvaluationError("gold corpus snapshot conflict")
    canonical = [
        {
            "scope_id": scope_id,
            "documents": [documents[key] for key in sorted(documents)],
        }
        for scope_id, documents in sorted(scopes.items())
    ]
    fingerprint = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return fingerprint, len(scopes), sum(len(documents) for documents in scopes.values())


def _safe_declared_revision(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or len(normalized) > 256 or not re.fullmatch(r"[A-Za-z0-9._/@:+-]+", normalized):
        return None
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_model_artifact_evidence(
    root_value: Any,
) -> tuple[str | None, str | None, int, int, str | None]:
    if not isinstance(root_value, str):
        return None, None, 0, 0, None
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        return None, None, 0, 0, None
    try:
        root = root.resolve(strict=True)
    except OSError:
        return None, None, 0, 0, None
    if not root.is_dir():
        return None, None, 0, 0, None

    config_manifest: list[dict[str, Any]] = []
    declared_revision: str | None = None
    for name in _MODEL_CONFIG_FILES:
        path = root / name
        try:
            info = path.stat()
        except OSError:
            continue
        if not path.is_file() or info.st_size > 16 * 1024 * 1024:
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        config_manifest.append(
            {
                "name": name,
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        if name == "config.json":
            try:
                config = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(config, dict):
                for key in ("_commit_hash", "revision", "model_revision"):
                    declared_revision = _safe_declared_revision(config.get(key))
                    if declared_revision is not None:
                        break
    weight_manifest: list[dict[str, Any]] = []
    weight_bytes = 0
    for path in sorted(root.rglob("*")):
        try:
            info = path.stat()
        except OSError:
            continue
        if not path.is_file() or path.suffix.lower() not in _MODEL_WEIGHT_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        weight_manifest.append(
            {
                "name": relative,
                "size": info.st_size,
                "sha256": _sha256_file(path),
            }
        )
        weight_bytes += info.st_size
    config_digest = (
        hashlib.sha256(
            json.dumps(config_manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if config_manifest
        else None
    )
    weight_digest = (
        hashlib.sha256(
            json.dumps(weight_manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if weight_manifest
        else None
    )
    return config_digest, weight_digest, len(weight_manifest), weight_bytes, declared_revision


def _runtime_process_sha256(root_value: Any, base_url: str) -> str | None:
    if not isinstance(root_value, str):
        return None
    port = urlsplit(base_url).port
    if port is None:
        return None
    matches: list[str] = []
    try:
        processes = sorted(Path("/proc").iterdir(), key=lambda path: path.name)
    except OSError:
        return None
    root_bytes = root_value.encode()
    port_bytes = str(port).encode()
    for process in processes:
        if not process.name.isdigit():
            continue
        try:
            raw = (process / "cmdline").read_bytes()
        except OSError:
            continue
        arguments = tuple(value for value in raw.split(b"\0") if value)
        has_port = any(
            value == b"--port=" + port_bytes
            or (value == port_bytes and index > 0 and arguments[index - 1] == b"--port")
            for index, value in enumerate(arguments)
        )
        if root_bytes in raw and has_port:
            matches.append(hashlib.sha256(raw).hexdigest())
    if not matches:
        return None
    return hashlib.sha256(_canonical_json(matches)).hexdigest()


def _runtime_model_revision(
    model: str,
    metadata: dict[str, Any],
    *,
    base_url: str,
    runtime_version_sha256: str,
) -> RuntimeModelRevision:
    if metadata.get("id") != model:
        raise BaselineEvaluationError("runtime model identity mismatch")
    root_value = metadata.get("root")
    config_digest, weight_digest, weight_count, weight_bytes, local_revision = _local_model_artifact_evidence(
        root_value
    )
    declared_revision = _safe_declared_revision(metadata.get("revision")) or local_revision
    stable_metadata = {
        "id": metadata.get("id"),
        "object": metadata.get("object"),
        "owned_by": metadata.get("owned_by"),
        "parent": metadata.get("parent"),
        "max_model_len": metadata.get("max_model_len"),
        "root_name": Path(root_value).name if isinstance(root_value, str) else None,
    }
    return RuntimeModelRevision(
        endpoint_metadata_sha256=hashlib.sha256(
            json.dumps(stable_metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        runtime_version_sha256=runtime_version_sha256,
        runtime_process_sha256=_runtime_process_sha256(root_value, base_url),
        local_config_manifest_sha256=config_digest,
        weight_manifest_sha256=weight_digest,
        weight_file_count=weight_count,
        weight_bytes=weight_bytes,
        declared_revision=declared_revision,
    )


def _models_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return f"{normalized}/models" if normalized.endswith("/v1") else f"{normalized}/v1/models"


def _version_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return f"{normalized}/version"


async def _fetch_runtime_model_revision(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    endpoint_name: str,
) -> RuntimeModelRevision:
    require_loopback_url(base_url, name=endpoint_name)
    try:
        response = await client.get(_models_url(base_url))
        response.raise_for_status()
        payload = response.json()
        version_response = await client.get(_version_url(base_url))
        version_response.raise_for_status()
        version_payload = version_response.json()
    except (httpx.HTTPError, ValueError):
        raise BaselineEvaluationError("runtime model provenance request failed") from None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise BaselineEvaluationError("runtime model provenance response is invalid")
    matches = [item for item in data if isinstance(item, dict) and item.get("id") == model]
    if len(matches) != 1:
        raise BaselineEvaluationError("runtime model provenance is ambiguous")
    version = version_payload.get("version") if isinstance(version_payload, dict) else None
    if not isinstance(version, str) or not version or len(version) > 128:
        raise BaselineEvaluationError("runtime version provenance is invalid")
    return _runtime_model_revision(
        model,
        matches[0],
        base_url=base_url,
        runtime_version_sha256=hashlib.sha256(_canonical_json({"version": version})).hexdigest(),
    )


def _require_complete_model_provenance(revisions: BaselineModelRevisions) -> None:
    required = [revisions.llm, revisions.embedding, revisions.reranker]
    required.extend(
        revision
        for revision in (revisions.visual_embedding, revisions.visual_reranker)
        if revision is not None
    )
    if any(
        revision.local_config_manifest_sha256 is None
        or revision.weight_manifest_sha256 is None
        or revision.weight_file_count < 1
        or revision.weight_bytes < 1
        or revision.runtime_process_sha256 is None
        for revision in required
    ):
        raise BaselineEvaluationError("runtime model provenance is incomplete")


def _case_seed(case_id: str) -> int:
    offset = int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16) % 1_000_000
    return _BASELINE_SEED_NAMESPACE + offset


async def _collect_model_revisions() -> BaselineModelRevisions:
    requests = [
        (settings.llm_base_url, settings.llm_model, "LLM endpoint"),
        (settings.embed_base_url, settings.embed_model, "embedding endpoint"),
        (settings.rerank_base_url, settings.rerank_model, "reranker endpoint"),
    ]
    if settings.visual_enabled:
        requests.extend(
            [
                (
                    settings.visual_embed_base_url,
                    settings.visual_embed_model,
                    "visual embedding endpoint",
                ),
                (
                    settings.visual_rerank_base_url,
                    settings.visual_rerank_model,
                    "visual reranker endpoint",
                ),
            ]
        )
    async with httpx.AsyncClient(timeout=30.0) as client:
        revisions = await asyncio.gather(
            *(
                _fetch_runtime_model_revision(
                    client,
                    base_url=base_url,
                    model=model,
                    endpoint_name=name,
                )
                for base_url, model, name in requests
            )
        )
    return BaselineModelRevisions(
        llm=revisions[0],
        embedding=revisions[1],
        reranker=revisions[2],
        visual_embedding=revisions[3] if settings.visual_enabled else None,
        visual_reranker=revisions[4] if settings.visual_enabled else None,
    )


def _build_provenance(
    records: list[GoldRecord],
    *,
    mode: Literal["candidate", "release"],
    top_k: int,
    gold_artifact_sha256: str,
    sidecar_artifact_sha256: str,
    evaluated_at: datetime,
    git_sha: str | None,
    git_dirty: bool | None,
    runtime_corpus_snapshot_sha256: str,
    model_revisions: BaselineModelRevisions,
) -> BaselineProvenance:
    corpus_fingerprint, scope_count, document_count = _corpus_fingerprint(records)
    models = BaselineModelIdentifiers(
        llm=settings.llm_model,
        embedding=settings.embed_model,
        reranker=settings.rerank_model,
        visual_embedding=settings.visual_embed_model if settings.visual_enabled else None,
        visual_reranker=settings.visual_rerank_model if settings.visual_enabled else None,
    )
    configuration = BaselineConfiguration(
        top_k=top_k,
        dense_top_k=settings.rag_dense_top_k,
        sparse_top_k=settings.rag_sparse_top_k,
        rerank_top_k=settings.rag_rerank_top_k,
        rerank_min_score=settings.rag_rerank_min_score,
        embedding_dim=settings.embed_dim,
        visual_enabled=settings.visual_enabled,
        context_max_chars=settings.rag_context_max_chars,
        context_window_tokens=settings.chat_context_window,
        output_tokens=_BASELINE_OUTPUT_TOKENS,
        answer_route="doc_only",
        prompt_sha256=hashlib.sha256(CHAT_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        temperature=_BASELINE_TEMPERATURE,
        top_p=_BASELINE_TOP_P,
        seed_namespace=_BASELINE_SEED_NAMESPACE,
        seed_strategy="case-id-sha256-v1",
        enable_thinking=False,
    )
    configuration_sha256 = hashlib.sha256(
        json.dumps(
            configuration.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return BaselineProvenance(
        runner="retrieval_direct_answer",
        evaluation_mode=mode,
        evaluated_at=evaluated_at,
        git_sha=git_sha,
        git_dirty=git_dirty,
        gold_artifact_sha256=gold_artifact_sha256,
        sidecar_artifact_sha256=sidecar_artifact_sha256,
        corpus_fingerprint_sha256=corpus_fingerprint,
        runtime_corpus_snapshot_sha256=runtime_corpus_snapshot_sha256,
        scope_count=scope_count,
        document_snapshot_count=document_count,
        models=models,
        model_revisions=model_revisions,
        configuration=configuration,
        configuration_sha256=configuration_sha256,
    )


def create_readonly_sessionmaker() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(
        require_loopback_database_url(settings.database_url),
        pool_pre_ping=True,
        connect_args={
            "server_settings": {
                "application_name": "rag_gold_baseline_readonly",
                "default_transaction_read_only": "on",
            }
        },
    )
    return engine, async_sessionmaker(engine, expire_on_commit=False)


def _retrieval_probe_matches(
    probe: RetrievalProbe,
    row: Any,
    document_refs: dict[uuid.UUID, str],
    snapshot: DocumentSnapshot,
) -> bool:
    body = (row.body or "").strip()
    expected_page = min(max(int(row.page_start or 0) + 1, 1), snapshot.page_count)
    return (
        row.document_id == probe.document_id
        and document_refs.get(row.document_id) == probe.document_ref
        and row.page_start == probe.page_start
        and row.page_end == probe.page_end
        and probe.page == expected_page
        and text_sha256(body) == probe.content_sha256
    )


def _value_sha256(value: Any) -> str:
    if isinstance(value, (dict, list)):
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    else:
        encoded = str(value if value is not None else "").encode()
    return hashlib.sha256(encoded).hexdigest()


def _runtime_scope_digest(
    document_rows: list[Any],
    chunk_rows: list[Any],
    page_rows: list[Any],
) -> str:
    payload = {
        "documents": [
            {
                "id": str(row.id),
                "s3_key_sha256": _value_sha256(row.s3_key_original),
                "page_count": row.page_count,
                "chunk_count": row.chunk_count,
                "indexed_at": row.indexed_at.isoformat() if row.indexed_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
            for row in document_rows
        ],
        "chunks": [
            {
                "id": str(row.id),
                "document_id": str(row.document_id),
                "idx": row.idx,
                "kind": row.kind,
                "heading_path_sha256": _value_sha256(row.heading_path or ""),
                "page_start": row.page_start,
                "page_end": row.page_end,
                "text_en_sha256": _value_sha256(row.text_en or ""),
                "body_sha256": _value_sha256(row.body or ""),
                "emb_en_sha256": _value_sha256(row.emb_en_text),
                "emb_ru_sha256": _value_sha256(row.emb_ru_text),
                "meta_sha256": _value_sha256(row.meta or {}),
            }
            for row in chunk_rows
        ],
        "page_embeddings": [
            {
                "id": str(row.id),
                "document_id": str(row.document_id),
                "page_idx": row.page_idx,
                "embedding_sha256": _value_sha256(row.emb_text),
                "meta_sha256": _value_sha256(row.meta or {}),
            }
            for row in page_rows
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ProductionBaselineRunner:
    def __init__(
        self,
        engine: AsyncEngine,
        sessionmaker: async_sessionmaker[AsyncSession],
        *,
        top_k: int,
    ) -> None:
        require_loopback_url(settings.embed_base_url, name="embedding endpoint")
        require_loopback_url(settings.rerank_base_url, name="reranker endpoint")
        require_loopback_url(settings.llm_base_url, name="LLM endpoint")
        require_loopback_endpoint(settings.s3_endpoint, name="MinIO endpoint")
        if settings.visual_enabled:
            require_loopback_url(settings.visual_embed_base_url, name="visual embedding endpoint")
            require_loopback_url(settings.visual_rerank_base_url, name="visual reranker endpoint")
        self.engine = engine
        self.sessionmaker = sessionmaker
        self.top_k = top_k
        self.storage = Storage()
        self.embedder = Embedder()
        visual_embedder = VisualEmbedder() if settings.visual_enabled else None
        visual_reranker = VisualReranker() if settings.visual_enabled else None
        self.retriever = Retriever(
            self.embedder,
            Reranker(),
            visual_embedder,
            visual_reranker,
            self.storage,
        )
        self.chat = ChatEngine()
        self._scope_cache: dict[str, tuple[str, str, dict[uuid.UUID, str], str]] = {}

    async def close(self) -> None:
        await self.embedder.client.close()
        await self.chat.client.close()
        await self.engine.dispose()

    async def _load_scope_rows(
        self,
        sidecar: PrivateSidecarRecord,
    ) -> tuple[str, list[Any], list[Any], list[Any]]:
        source_ids = [item.document_id for item in sidecar.source_documents]
        async with self.sessionmaker() as session:
            owner_rows = (
                await session.execute(
                    sql(
                        "SELECT id, owner_sub FROM documents "
                        "WHERE id = ANY(CAST(:ids AS uuid[])) AND status = 'done'"
                    ),
                    {"ids": source_ids},
                )
            ).all()
            if len(owner_rows) != len(source_ids):
                raise BaselineEvaluationError("source document resolution mismatch")
            owners = {row.owner_sub for row in owner_rows}
            if len(owners) != 1:
                raise BaselineEvaluationError("source documents do not resolve to one owner")
            owner_sub = str(next(iter(owners)))
            if make_scope_id(owner_sub) != sidecar.scope_id:
                raise BaselineEvaluationError("resolved owner scope hash mismatch")
            document_rows = (
                await session.execute(
                    sql(
                        "SELECT id, s3_key_original, page_count, chunk_count, indexed_at, updated_at "
                        "FROM documents "
                        "WHERE owner_sub = :owner AND status = 'done' ORDER BY id"
                    ),
                    {"owner": owner_sub},
                )
            ).all()
            document_ids = [row.id for row in document_rows]
            chunk_rows = (
                await session.execute(
                    sql(
                        "SELECT id, document_id, idx, kind, heading_path, page_start, page_end, "
                        "text_en, COALESCE(NULLIF(text_ru, ''), text_en) AS body, meta, "
                        "emb_en::text AS emb_en_text, emb_ru::text AS emb_ru_text "
                        "FROM chunks WHERE document_id = ANY(CAST(:ids AS uuid[])) "
                        "ORDER BY document_id, idx, id"
                    ),
                    {"ids": document_ids},
                )
            ).all()
            page_rows = (
                await session.execute(
                    sql(
                        "SELECT id, document_id, page_idx, emb::text AS emb_text, meta "
                        "FROM page_embeddings WHERE document_id = ANY(CAST(:ids AS uuid[])) "
                        "ORDER BY document_id, page_idx, id"
                    ),
                    {"ids": document_ids},
                )
            ).all()
        return owner_sub, list(document_rows), list(chunk_rows), list(page_rows)

    async def _resolve_and_verify_scope(
        self,
        record: GoldRecord,
        sidecar: PrivateSidecarRecord,
    ) -> tuple[str, dict[uuid.UUID, str]]:
        expected_scope_fingerprint = hashlib.sha256(
            json.dumps(
                [item.model_dump(mode="json") for item in record.document_scope],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        cached = self._scope_cache.get(record.scope_id)
        if cached is not None:
            owner_sub, cached_fingerprint, cached_document_refs, _ = cached
            if cached_fingerprint != expected_scope_fingerprint:
                raise BaselineEvaluationError("gold scope snapshot changed between cases")
            for item in sidecar.source_documents:
                if cached_document_refs.get(item.document_id) != item.document_ref:
                    raise BaselineEvaluationError("sidecar document mapping mismatch")
            return owner_sub, cached_document_refs

        owner_sub, document_rows, chunk_rows, page_rows = await self._load_scope_rows(sidecar)
        chunks_by_document: dict[uuid.UUID, list[Any]] = defaultdict(list)
        for row in chunk_rows:
            chunks_by_document[row.document_id].append(row)
        actual_snapshots: dict[str, tuple[str, int]] = {}
        document_refs: dict[uuid.UUID, str] = {}
        for document in document_rows:
            rows = chunks_by_document.get(document.id, [])
            if not rows:
                raise BaselineEvaluationError("scope document has no indexed chunks")
            source_bytes = await self.storage.get_bytes(settings.bucket_originals, document.s3_key_original)
            source_sha256 = bytes_sha256(source_bytes)
            document_ref = make_document_ref(source_sha256)
            page_count = max(
                int(document.page_count or 0),
                max(int(row.page_end or 0) + 1 for row in rows),
                1,
            )
            parsed_sha256 = parsed_chunks_sha256(
                [
                    {
                        "idx": row.idx,
                        "kind": row.kind,
                        "heading_path": row.heading_path or "",
                        "page_start": row.page_start,
                        "page_end": row.page_end,
                        "text": (row.text_en or "").strip(),
                    }
                    for row in rows
                ]
            )
            actual_snapshots[document_ref] = (parsed_sha256, page_count)
            document_refs[document.id] = document_ref

        expected = {
            item.document_ref: (item.parsed_content_sha256, item.page_count) for item in record.document_scope
        }
        if actual_snapshots != expected:
            raise BaselineEvaluationError("production corpus hash snapshot mismatch")
        for item in sidecar.source_documents:
            if document_refs.get(item.document_id) != item.document_ref:
                raise BaselineEvaluationError("sidecar document mapping mismatch")
        self._scope_cache[record.scope_id] = (
            owner_sub,
            expected_scope_fingerprint,
            document_refs,
            _runtime_scope_digest(list(document_rows), list(chunk_rows), list(page_rows)),
        )
        return owner_sub, document_refs

    async def _verify_case_evidence(
        self,
        record: GoldRecord,
        sidecar: PrivateSidecarRecord,
        document_refs: dict[uuid.UUID, str],
    ) -> None:
        chunk_ids = [item.chunk_id for item in sidecar.exact_evidence]
        chunk_ids.extend(item.chunk_id for item in sidecar.retrieval_probe)
        if not chunk_ids:
            return
        async with self.sessionmaker() as session:
            rows = (
                await session.execute(
                    sql(
                        "SELECT id, document_id, idx, kind, heading_path, page_start, page_end, "
                        "COALESCE(NULLIF(text_ru, ''), text_en) AS body "
                        "FROM chunks WHERE id = ANY(CAST(:ids AS uuid[]))"
                    ),
                    {"ids": list(dict.fromkeys(chunk_ids))},
                )
            ).all()
        by_id = {row.id: row for row in rows}
        if len(by_id) != len(set(chunk_ids)):
            raise BaselineEvaluationError("sidecar chunk resolution mismatch")
        for item in sidecar.exact_evidence:
            row = by_id[item.chunk_id]
            body = (row.body or "").strip()
            if (
                row.document_id != item.document_id
                or document_refs.get(row.document_id) != item.document_ref
                or row.idx != item.chunk_index
                or row.kind != item.kind
                or (row.heading_path or "") != item.heading_path
                or row.page_start != item.page_start
                or row.page_end != item.page_end
                or hashlib.sha256(body.encode()).hexdigest() != item.text_sha256
                or item.exact_quote not in body
                or text_sha256(item.exact_quote) != item.content_sha256
            ):
                raise BaselineEvaluationError("production evidence hash/locator mismatch")
        snapshots = {item.document_ref: item for item in record.document_scope}
        for probe in sidecar.retrieval_probe:
            row = by_id[probe.chunk_id]
            snapshot = snapshots.get(probe.document_ref)
            if snapshot is None or not _retrieval_probe_matches(
                probe,
                row,
                document_refs,
                snapshot,
            ):
                raise BaselineEvaluationError("production retrieval probe mismatch")

    async def verify_corpus_snapshot(
        self,
        records: list[GoldRecord],
        sidecars: Mapping[str, PrivateSidecarRecord],
    ) -> str:
        """Rebuild and verify every live scope/evidence locator, returning an opaque digest."""

        self._scope_cache.clear()
        case_bindings: list[dict[str, Any]] = []
        for record in sorted(records, key=lambda item: item.case_id):
            sidecar = sidecars[record.case_id]
            _, document_refs = await self._resolve_and_verify_scope(record, sidecar)
            await self._verify_case_evidence(record, sidecar, document_refs)
            case_bindings.append(
                {
                    "gold_case_sha256": sidecar.gold_case_sha256,
                    "evidence": sorted(
                        (
                            str(item.chunk_id),
                            item.text_sha256,
                            item.content_sha256,
                            item.page,
                            item.page_start,
                            item.page_end,
                        )
                        for item in sidecar.exact_evidence
                    ),
                    "retrieval_probe": sorted(
                        (
                            str(item.chunk_id),
                            item.content_sha256,
                            item.page,
                            item.page_start,
                            item.page_end,
                        )
                        for item in sidecar.retrieval_probe
                    ),
                }
            )
        payload = {
            "scopes": [
                {"scope_id": scope_id, "runtime_sha256": cached[3]}
                for scope_id, cached in sorted(self._scope_cache.items())
            ],
            "case_bindings": case_bindings,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    async def run_case(
        self,
        record: GoldRecord,
        sidecar: PrivateSidecarRecord,
    ) -> BaselineObservation:
        owner_sub, document_refs = await self._resolve_and_verify_scope(record, sidecar)
        await self._verify_case_evidence(record, sidecar, document_refs)
        total_start = time.monotonic()
        retrieval_start = time.monotonic()
        async with self.sessionmaker() as session:
            chunks = await self.retriever.retrieve(
                session,
                record.question,
                top_k=self.top_k,
                owner_sub=owner_sub,
                allow_rerank_fallback=False,
            )
        retrieval_ms = (time.monotonic() - retrieval_start) * 1000
        if any(document_refs.get(chunk.document_id) is None for chunk in chunks):
            raise BaselineEvaluationError("retriever escaped the verified owner scope")

        generation_start = time.monotonic()
        if chunks:
            parts: list[str] = []
            async for delta in self.chat.stream_answer(
                record.question,
                chunks,
                [],
                route="doc_only",
                temperature=_BASELINE_TEMPERATURE,
                top_p=_BASELINE_TOP_P,
                max_tokens=_BASELINE_OUTPUT_TOKENS,
                seed=_case_seed(record.case_id),
            ):
                parts.append(delta)
            answer = "".join(parts).strip()
        else:
            answer = _NO_RESULTS_ANSWER
        generation_ms = (time.monotonic() - generation_start) * 1000
        return BaselineObservation(
            case_id=record.case_id,
            gold_case_sha256=sidecar.gold_case_sha256,
            scope_id=record.scope_id,
            answer=answer,
            retrieved=tuple(
                RetrievedUnit(chunk_id=chunk.id, document_id=chunk.document_id) for chunk in chunks
            ),
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=(time.monotonic() - total_start) * 1000,
        )


def _atomic_write_report(path: Path, payload: dict[str, Any]) -> bytes:
    content = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if not path.parent.exists():
        raise BaselineEvaluationError("baseline report parent must already exist")
    write_private_json_fresh(path, content)
    return content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--mode", choices=("candidate", "release"), default="release")
    parser.add_argument("--top-k", type=int, default=10, choices=range(10, 65))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--attestation-key", type=Path)
    return parser


async def async_main() -> int:
    args = build_parser().parse_args()
    try:
        if (args.attestation is None) != (args.attestation_key is None):
            raise BaselineEvaluationError("--attestation and --attestation-key must be provided together")
        if args.attestation is not None and args.report is None:
            raise BaselineEvaluationError("--attestation requires --report")
        if args.mode == "release" and args.attestation is None:
            raise BaselineEvaluationError("release mode requires signed report attestation")
        ensure_private_gold_path(args.gold, REPOSITORY_ROOT)
        ensure_private_gold_path(args.sidecar, REPOSITORY_ROOT)
        report_path = args.report.expanduser() if args.report is not None else None
        if report_path is not None:
            ensure_private_gold_path(report_path, REPOSITORY_ROOT)
        attestation_path = args.attestation.expanduser() if args.attestation is not None else None
        if attestation_path is not None:
            ensure_private_gold_path(attestation_path, REPOSITORY_ROOT)
        gold_artifact = read_private_bytes(args.gold, max_bytes=256 * 1024 * 1024)
        sidecar_artifact = read_private_bytes(args.sidecar, max_bytes=256 * 1024 * 1024)
        records, _ = parse_gold_set_bytes(gold_artifact.raw_bytes, mode=args.mode)
        sidecars = parse_private_sidecar_bytes(sidecar_artifact.raw_bytes)
        bound = bind_gold_sidecar(records, sidecars)
        gold_artifact_sha256 = gold_artifact.sha256
        sidecar_artifact_sha256 = sidecar_artifact.sha256
        git_sha, git_dirty = _git_state(REPOSITORY_ROOT)
        if args.mode == "release" and (git_sha is None or git_dirty is not False):
            raise BaselineEvaluationError("release baseline requires a clean Git revision")
        engine, sessionmaker = create_readonly_sessionmaker()
        runner = ProductionBaselineRunner(engine, sessionmaker, top_k=args.top_k)
        try:
            runtime_snapshot = await runner.verify_corpus_snapshot(records, bound)
            model_revisions = await _collect_model_revisions()
            if args.mode == "release":
                _require_complete_model_provenance(model_revisions)
            provenance = _build_provenance(
                records,
                mode=args.mode,
                top_k=args.top_k,
                gold_artifact_sha256=gold_artifact_sha256,
                sidecar_artifact_sha256=sidecar_artifact_sha256,
                evaluated_at=datetime.now(UTC),
                git_sha=git_sha,
                git_dirty=git_dirty,
                runtime_corpus_snapshot_sha256=runtime_snapshot,
                model_revisions=model_revisions,
            )
            report = await evaluate_baseline(
                records,
                bound,
                runner,
                provenance=provenance,
            )
            if await runner.verify_corpus_snapshot(records, bound) != runtime_snapshot:
                raise BaselineEvaluationError("production corpus changed during evaluation")
            if await _collect_model_revisions() != model_revisions:
                raise BaselineEvaluationError("runtime model changed during evaluation")
        finally:
            await runner.close()
        payload = report.model_dump(mode="json")
        if report_path is not None:
            report_bytes = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            )
            if attestation_path is not None:
                key = load_hmac_key(args.attestation_key, REPOSITORY_ROOT)
                cases = build_case_attestations(records, bound)
                attestation = create_report_attestation(
                    report_bytes=report_bytes,
                    gold_bytes=gold_artifact.raw_bytes,
                    sidecar_bytes=sidecar_artifact.raw_bytes,
                    cases=cases,
                    key=key,
                    repository_root=REPOSITORY_ROOT,
                )
                verify_report_attestation(
                    attestation,
                    report_bytes=report_bytes,
                    gold_bytes=gold_artifact.raw_bytes,
                    sidecar_bytes=sidecar_artifact.raw_bytes,
                    expected_cases=cases,
                    key=key,
                    repository_root=REPOSITORY_ROOT,
                )
            _atomic_write_report(report_path, payload)
            if attestation_path is not None:
                atomic_write_attestation(attestation_path, attestation)
    except Exception as error:  # noqa: BLE001 - fail closed without leaking DB/private values
        print(f"baseline rejected: {type(error).__name__}")
        return 2
    summary = {key: value for key, value in payload.items() if key != "cases"}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
