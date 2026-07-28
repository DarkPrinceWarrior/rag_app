"""Retrieval-бенч эмбеддеров на реальной библиотеке (§ 12.1 шаг 3).

Фаза 1 (один раз): из случайных чанков прод-LLM генерирует по вопросу,
ответ на который содержится именно в этом чанке (chunk_id = эталон).
Кэш в --qa-file — ОДИН набор вопросов для всех сравниваемых эмбеддеров.

Фаза 2: эмбеддим все чанки (EN и RU) и вопросы заданным эмбеддером,
косинус в памяти (max по двум языкам — как LEAST в проде), метрики
recall@1 / recall@5 / MRR@10 / nDCG@10 и задержка эмбеддинга.

Префиксы запроса и фрагмента задаются отдельно и добавляются ПОСЛЕ одинакового
символьного усечения исходного текста. Это позволяет честно сравнивать
Qwen3-Embedding и Nemotron на одном QA и одном снимке корпуса.

--truncate-dim N — MRL-усечение (отрезать и L2-нормировать): сравнение
качества при dim, влезающем в HNSW-лимит pgvector (2000).

Запуск (на сервере):
  uv run python scripts/eval_retrieval.py --make-qa 60
  uv run python scripts/eval_retrieval.py --url http://127.0.0.1:8002/v1 --model qwen3-embedding-8b
  uv run python scripts/eval_retrieval.py --url ... --model nemotron-3-embed-8b \
    --query-prefix "query: " --passage-prefix "passage: " --truncate-dim 1024 \
    --output /root/model_trials/nemotron-retrieval.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import secrets
import stat
import statistics
import time
from pathlib import Path
from typing import Any, TypedDict, cast
from urllib.parse import urlsplit

import httpx
from openai import AsyncOpenAI
from sqlalchemy import select

from rag_app.config import settings
from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.db.models import Chunk, Document, DocumentStatus
from rag_app.eval.private_artifacts import write_private_json_fresh

QA_PROMPT = """\
Вот фрагмент технической документации:

---
{text}
---

Сформулируй ОДИН конкретный вопрос на русском языке, ответ на который содержится
именно в этом фрагменте (про числа, требования или условия из него).
Выведи только сам вопрос, без пояснений."""

_BATCH_SIZE = 32
_INPUT_TRUNCATION_CHARS = 8000
_MAX_PREFIX_CHARS = 4000
_MAX_QA_BYTES = 4 * 1024 * 1024
_MAX_QA_RECORDS = 10_000
_MAX_QUESTION_CHARS = 8000
_MAX_MAKE_QA = 1000
_MAX_TRUNCATE_DIM = 65_536
_MAX_ENDPOINT_CHARS = 2048
_MAX_MODEL_CHARS = 512
_MAX_REPORT_BYTES = 2 * 1024 * 1024
_REPORT_SCHEMA = "retrieval-embedding-eval-v2"
_REDACTED = "<redacted>"
_QA_GENERATION_SEED = 3086
_QA_OVERSAMPLE_FACTOR = 3
_LATENCY_QUERY_REPEATS = 3
_HTTP_TIMEOUT_S = 300.0


class QARecord(TypedDict):
    chunk_id: str
    question: str


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_prefix(prefix: str, *, name: str) -> str:
    if not isinstance(prefix, str):
        raise TypeError(f"{name} must be a string")
    if len(prefix) > _MAX_PREFIX_CHARS:
        raise ValueError(f"{name} exceeds {_MAX_PREFIX_CHARS} characters")
    return prefix


def _validate_truncate_dim(dim: int | None) -> None:
    if dim is not None and not 1 <= dim <= _MAX_TRUNCATE_DIM:
        raise ValueError(f"truncate_dim must be in [1, {_MAX_TRUNCATE_DIM}]")


def _validate_endpoint(url: str, model: str) -> tuple[str, str]:
    normalized_url = url.strip()
    normalized_model = model.strip()
    if not normalized_url or len(normalized_url) > _MAX_ENDPOINT_CHARS:
        raise ValueError("embedding endpoint URL is empty or too long")
    if not normalized_model or len(normalized_model) > _MAX_MODEL_CHARS:
        raise ValueError("embedding model is empty or too long")
    parsed = urlsplit(normalized_url)
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("embedding endpoint URL has an invalid port") from None
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1"
        or port is None
        or not 1 <= port <= 65_535
    ):
        raise ValueError(
            "model endpoint must be a credential-free loopback /v1 URL with an explicit port"
        )
    return normalized_url.rstrip("/"), normalized_model


def _prepare_text(prefix: str, text: str) -> str:
    """Truncate the shared raw body, then apply the model-specific prefix."""

    body = text.strip()[:_INPUT_TRUNCATION_CHARS]
    prepared = f"{prefix}{body}"
    return prepared or "."


def _safe_http_client(timeout_s: float = _HTTP_TIMEOUT_S) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s),
        trust_env=False,
        follow_redirects=False,
    )


async def _probe_server_version(client: httpx.AsyncClient, base_url: str) -> str:
    response = await client.get(f"{base_url.removesuffix('/v1')}/version")
    if response.is_redirect:
        raise ValueError("model server version endpoint redirected")
    response.raise_for_status()
    if len(response.content) > 4096:
        raise ValueError("model server version response is too large")
    try:
        value = response.json()
    except ValueError:
        raise ValueError("model server version response is not JSON") from None
    version = value.get("version") if isinstance(value, dict) else None
    if not isinstance(version, str) or not 1 <= len(version.strip()) <= 128:
        raise ValueError("model server version is invalid")
    return version.strip()


async def _validate_served_model(client: AsyncOpenAI, model: str) -> tuple[str, str]:
    page = await client.models.list()
    identifiers = sorted({item.id for item in page.data if isinstance(item.id, str)})
    if model not in identifiers:
        raise ValueError("requested model is absent from the endpoint model registry")
    matching = next(item for item in page.data if item.id == model)
    if hasattr(matching, "model_dump"):
        model_card = matching.model_dump(mode="json")
    else:
        model_card = {"id": matching.id}
    return (
        _sha256_bytes(_canonical_json_bytes(identifiers)),
        _sha256_bytes(_canonical_json_bytes(model_card)),
    )


def _runner_sha256() -> str:
    return _sha256_bytes(Path(__file__).read_bytes())


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _write_private_jsonl_fresh(path: Path, raw: bytes) -> tuple[str, int]:
    """Atomically publish new mode-0600 JSONL without following the final path."""

    if not raw or len(raw) > _MAX_QA_BYTES:
        raise ValueError("QA output is empty or exceeds the size limit")
    target = path.expanduser()
    if target.name in {"", ".", ".."}:
        raise ValueError("QA output filename is invalid")
    directory_fd = os.open(
        target.parent,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    temporary_name = f".retrieval-qa-{os.getpid()}-{secrets.token_hex(12)}.tmp"
    descriptor = -1
    temporary_exists = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        temporary_exists = True
        os.fchmod(descriptor, 0o600)
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written < 1:
                raise OSError("short QA output write")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(
            temporary_name,
            target.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.unlink(temporary_name, dir_fd=directory_fd)
        temporary_exists = False
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)
    return _sha256_bytes(raw), len(raw)


def _load_qa(qa_file: Path) -> tuple[list[QARecord], str, int]:
    try:
        descriptor = os.open(
            qa_file.expanduser(),
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
    except OSError as error:
        raise ValueError(f"unable to open QA file ({type(error).__name__})") from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("QA file must be a regular non-symlink file")
        if not 1 <= metadata.st_size <= _MAX_QA_BYTES:
            raise ValueError("QA file is empty or exceeds the size limit")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                raise ValueError("QA file ended before its declared size")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ValueError("QA file grew while it was read")
        raw = b"".join(chunks)
    except OSError as error:
        raise ValueError(f"unable to read QA file ({type(error).__name__})") from None
    finally:
        os.close(descriptor)

    records: list[QARecord] = []
    seen_questions: set[str] = set()
    try:
        text = raw.decode("utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise ValueError(f"QA line {line_number} is empty")
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_constant,
            )
            if not isinstance(value, dict) or set(value) != {"chunk_id", "question"}:
                raise ValueError(f"QA line {line_number} has an invalid schema")
            chunk_id = value["chunk_id"]
            question = value["question"]
            if not isinstance(chunk_id, str) or not 1 <= len(chunk_id.strip()) <= 128:
                raise ValueError(f"QA line {line_number} has an invalid chunk_id")
            if (
                not isinstance(question, str)
                or not 10 < len(question.strip()) <= _MAX_QUESTION_CHARS
            ):
                raise ValueError(f"QA line {line_number} has an invalid question")
            normalized_question = question.strip()
            if normalized_question in seen_questions:
                raise ValueError("QA questions must be unique")
            seen_questions.add(normalized_question)
            records.append({"chunk_id": chunk_id.strip(), "question": normalized_question})
            if len(records) > _MAX_QA_RECORDS:
                raise ValueError("QA file contains too many records")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"invalid QA file ({type(error).__name__})") from None
    if not records:
        raise ValueError("QA file contains no records")
    return records, _sha256_bytes(raw), len(raw)


def _validate_corpus(chunks: list[Chunk], qa: list[QARecord]) -> None:
    if not chunks:
        raise ValueError("retrieval corpus is empty")
    chunk_ids = [str(chunk.id) for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("retrieval corpus contains duplicate chunk IDs")
    if any(not ((chunk.text_en or "").strip() or (chunk.text_ru or "").strip()) for chunk in chunks):
        raise ValueError("retrieval corpus contains an empty bilingual chunk")
    missing = {record["chunk_id"] for record in qa}.difference(chunk_ids)
    if missing:
        raise ValueError(f"{len(missing)} QA gold chunks are absent from the retrieval corpus")


def _corpus_manifest_sha256(chunks: list[Chunk]) -> str:
    rows = sorted(
        [
        {
            "chunk_id": str(chunk.id),
            "text_en_sha256": _sha256_bytes((chunk.text_en or "").encode("utf-8")),
            "text_ru_sha256": _sha256_bytes((chunk.text_ru or "").encode("utf-8")),
        }
        for chunk in chunks
        ],
        key=lambda row: row["chunk_id"],
    )
    return _sha256_bytes(_canonical_json_bytes(rows))


def _require_fresh_output(path: Path) -> None:
    try:
        path.expanduser().lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise ValueError(f"unable to inspect report output ({type(error).__name__})") from None
    raise FileExistsError("refusing to replace an existing private artifact")


async def load_chunks() -> list[Chunk]:
    engine = create_engine()
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            statement = (
                select(Chunk)
                .join(Document, Document.id == Chunk.document_id)
                .where(Document.status == DocumentStatus.done)
                .order_by(Chunk.document_id, Chunk.idx, Chunk.id)
            )
            return list((await session.execute(statement)).scalars().all())
    finally:
        await engine.dispose()


async def make_qa(n: int, qa_file: Path) -> None:
    if not 1 <= n <= _MAX_MAKE_QA:
        raise ValueError(f"make_qa must be in [1, {_MAX_MAKE_QA}]")
    _require_fresh_output(qa_file)
    chunks = await load_chunks()
    if len(chunks) < n:
        raise ValueError("retrieval corpus has fewer chunks than the requested QA count")
    base_url, model = _validate_endpoint(settings.llm_base_url, settings.llm_model)
    rng = random.Random(_QA_GENERATION_SEED)
    candidates = list(chunks)
    rng.shuffle(candidates)
    sample = candidates[: min(len(candidates), n * _QA_OVERSAMPLE_FACTOR)]
    sem = asyncio.Semaphore(8)

    async def gen(client: AsyncOpenAI, position: int, chunk: Chunk) -> dict[str, str]:
        async with sem:
            prompt = QA_PROMPT.format(text=(chunk.text_ru or chunk.text_en)[:2500])
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=120,
                seed=_QA_GENERATION_SEED + position,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            if resp.model != model:
                raise ValueError("QA generator returned a different model identifier")
            return {
                "chunk_id": str(chunk.id),
                "question": " ".join((resp.choices[0].message.content or "").split()),
            }

    async with _safe_http_client(120.0) as http_client:
        server_version = await _probe_server_version(http_client, base_url)
        async with AsyncOpenAI(
            base_url=base_url,
            api_key=settings.llm_api_key,
            timeout=120.0,
            max_retries=0,
            http_client=http_client,
        ) as client:
            await _validate_served_model(client, model)
            generated = await asyncio.gather(
                *(gen(client, position, chunk) for position, chunk in enumerate(sample))
            )

    qa: list[QARecord] = []
    seen_questions: set[str] = set()
    for row in generated:
        question = row["question"]
        question_key = question.casefold()
        if not 10 < len(question) <= _MAX_QUESTION_CHARS or question_key in seen_questions:
            continue
        seen_questions.add(question_key)
        qa.append({"chunk_id": row["chunk_id"], "question": question})
        if len(qa) == n:
            break
    if len(qa) != n:
        raise ValueError(
            f"QA generation produced {len(qa)} unique valid questions; expected exactly {n}"
        )
    raw = (
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for row in qa
        )
        + "\n"
    ).encode("utf-8")
    qa_sha256, qa_bytes = _write_private_jsonl_fresh(qa_file, raw)
    print(
        f"вопросов: {len(qa)} → {qa_file} sha256={qa_sha256} "
        f"bytes={qa_bytes} server={server_version}"
    )


def _norm(vec: list[float], dim: int | None) -> list[float]:
    _validate_truncate_dim(dim)
    if not vec or not all(math.isfinite(value) for value in vec):
        raise ValueError("embedding vector is empty or non-finite")
    if dim is not None:
        if dim > len(vec):
            raise ValueError("truncate_dim exceeds the native embedding dimension")
        vec = vec[:dim]
    squared_norm = math.fsum(value * value for value in vec)
    if not math.isfinite(squared_norm) or squared_norm <= 0.0:
        raise ValueError("embedding vector has no finite positive norm")
    norm = math.sqrt(squared_norm)
    return [value / norm for value in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _ranked_indices(chunks: list[Chunk], similarities: list[float]) -> list[int]:
    if len(chunks) != len(similarities):
        raise ValueError("ranking inputs have inconsistent lengths")
    return sorted(
        range(len(chunks)),
        key=lambda index: (-similarities[index], str(chunks[index].id)),
    )


def _metrics_from_ranks(ranks: list[int]) -> dict[str, float]:
    if not ranks or any(rank < 1 for rank in ranks):
        raise ValueError("retrieval ranks must be positive and non-empty")
    count = len(ranks)
    return {
        "recall@1": sum(rank == 1 for rank in ranks) / count,
        "recall@5": sum(rank <= 5 for rank in ranks) / count,
        "mrr@10": math.fsum(1 / rank for rank in ranks if rank <= 10) / count,
        "ndcg@10": (
            math.fsum(1 / math.log2(rank + 1) for rank in ranks if rank <= 10) / count
        ),
    }


async def evaluate(
    url: str,
    model: str,
    qa_file: Path,
    truncate_dim: int | None,
    *,
    query_prefix: str | None = None,
    passage_prefix: str = "",
    output: Path | None = None,
) -> dict[str, object]:
    runner_sha256 = _runner_sha256()
    _validate_truncate_dim(truncate_dim)
    url, model = _validate_endpoint(url, model)
    if query_prefix is None:
        query_prefix = f"Instruct: {settings.embed_query_instruction}\nQuery: "
    query_prefix = _validate_prefix(query_prefix, name="query_prefix")
    passage_prefix = _validate_prefix(passage_prefix, name="passage_prefix")
    if output is not None:
        _require_fresh_output(output)
    qa, qa_sha256, qa_size_bytes = _load_qa(qa_file)
    chunks = await load_chunks()
    _validate_corpus(chunks, qa)
    observed_dim: int | None = None
    observed_native_dim: int | None = None

    async def embed(
        client: AsyncOpenAI,
        texts: list[str],
        prefix: str,
    ) -> list[list[float]]:
        nonlocal observed_dim, observed_native_dim
        if not texts:
            raise ValueError("embedding input batch is empty")
        output_vectors: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = [_prepare_text(prefix, text) for text in texts[i : i + _BATCH_SIZE]]
            response = await client.embeddings.create(model=model, input=batch)
            if response.model != model:
                raise ValueError("embedding endpoint returned a different model identifier")
            ordered = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(batch))):
                raise ValueError("embedding response indices do not match the request batch")
            for item in ordered:
                native_dim = len(item.embedding)
                if observed_native_dim is None:
                    observed_native_dim = native_dim
                elif native_dim != observed_native_dim:
                    raise ValueError("embedding endpoint returned inconsistent native dimensions")
            vectors = [_norm(item.embedding, truncate_dim) for item in ordered]
            for vector in vectors:
                if observed_dim is None:
                    observed_dim = len(vector)
                elif len(vector) != observed_dim:
                    raise ValueError("embedding endpoint returned inconsistent dimensions")
            output_vectors.extend(vectors)
        if len(output_vectors) != len(texts):
            raise ValueError("embedding endpoint returned an incomplete response")
        return output_vectors

    english_texts = [cast(str, chunk.text_en or "") for chunk in chunks]
    russian_texts = [cast(str, chunk.text_ru or "") for chunk in chunks]
    question_texts = [record["question"] for record in qa]
    query_latency_samples_ms: list[float] = []

    async with _safe_http_client() as http_client:
        server_version = await _probe_server_version(http_client, url)
        async with AsyncOpenAI(
            base_url=url,
            api_key="local",
            timeout=_HTTP_TIMEOUT_S,
            max_retries=0,
            http_client=http_client,
        ) as client:
            served_models_sha256, served_model_card_sha256 = await _validate_served_model(
                client, model
            )

            warmup_started = time.perf_counter()
            await embed(client, english_texts[: min(_BATCH_SIZE, len(english_texts))], passage_prefix)
            warmup_ms = (time.perf_counter() - warmup_started) * 1000

            run_started = time.perf_counter()
            corpus_started = time.perf_counter()
            emb_en = await embed(client, english_texts, passage_prefix)
            emb_ru = await embed(client, russian_texts, passage_prefix)
            corpus_ms = (time.perf_counter() - corpus_started) * 1000

            q_emb: list[list[float]] | None = None
            for _ in range(_LATENCY_QUERY_REPEATS):
                query_started = time.perf_counter()
                current = await embed(client, question_texts, query_prefix)
                query_latency_samples_ms.append((time.perf_counter() - query_started) * 1000)
                if q_emb is None:
                    q_emb = current
            if q_emb is None:
                raise ValueError("query embedding repeat loop produced no vectors")
            total_ms = (time.perf_counter() - run_started) * 1000

    ranks: list[int] = []
    per_case: list[dict[str, object]] = []
    for case_index, (record, query_vector) in enumerate(zip(qa, q_emb, strict=True)):
        similarities = [
            max(_dot(query_vector, english), _dot(query_vector, russian))
            for english, russian in zip(emb_en, emb_ru, strict=True)
        ]
        order = _ranked_indices(chunks, similarities)
        rank = next(
            (
                position + 1
                for position, index in enumerate(order)
                if str(chunks[index].id) == record["chunk_id"]
            ),
            None,
        )
        if rank is None:
            raise ValueError("QA gold chunk disappeared during ranking")
        ranks.append(rank)
        per_case.append(
            {
                "case_index": case_index,
                "question_sha256": _sha256_bytes(record["question"].encode("utf-8")),
                "gold_chunk_id": record["chunk_id"],
                "rank": rank,
                "hit_at_1": rank == 1,
                "hit_at_5": rank <= 5,
                "reciprocal_rank_at_10": 1 / rank if rank <= 10 else 0.0,
                "discounted_gain_at_10": (
                    1 / math.log2(rank + 1) if rank <= 10 else 0.0
                ),
            }
        )

    count = len(ranks)
    metrics = _metrics_from_ranks(ranks)
    dimension = observed_dim
    if dimension is None:
        raise ValueError("embedding endpoint returned no vectors")
    native_dimension = observed_native_dim
    if native_dimension is None:
        raise ValueError("embedding endpoint returned no native vectors")
    sorted_query_latencies = sorted(query_latency_samples_ms)
    query_p50_ms = statistics.median(sorted_query_latencies)
    query_p95_index = max(0, math.ceil(0.95 * len(sorted_query_latencies)) - 1)
    query_p95_ms = sorted_query_latencies[query_p95_index]
    if _runner_sha256() != runner_sha256:
        raise ValueError("evaluation runner changed while qualification was in progress")
    report: dict[str, object] = {
        "schema_version": _REPORT_SCHEMA,
        "endpoint": {
            "model": _REDACTED,
            "model_sha256": _sha256_bytes(model.encode("utf-8")),
            "url": _REDACTED,
            "url_sha256": _sha256_bytes(url.encode("utf-8")),
        },
        "qa": {
            "sha256": qa_sha256,
            "size_bytes": qa_size_bytes,
            "count": count,
        },
        "corpus": {
            "count": len(chunks),
            "manifest_sha256": _corpus_manifest_sha256(chunks),
            "selection": "done-documents",
            "order": "document_id,idx,id",
            "ranking_tie_break": "chunk_id",
        },
        "configuration": {
            "query_prefix": query_prefix,
            "passage_prefix": passage_prefix,
            "input_truncation_chars": _INPUT_TRUNCATION_CHARS,
            "input_truncation_policy": "raw-body-before-prefix",
            "truncate_dim": truncate_dim,
            "batch_size": _BATCH_SIZE,
            "query_latency_repeats": _LATENCY_QUERY_REPEATS,
        },
        "dim": dimension,
        "native_dim": native_dimension,
        "metrics": metrics,
        "cases": per_case,
        "latency_ms": {
            "diagnostic_only": True,
            "warmup": round(warmup_ms, 3),
            "corpus_embedding": round(corpus_ms, 3),
            "query_embedding_samples": [
                round(value, 3) for value in query_latency_samples_ms
            ],
            "query_embedding_p50": round(query_p50_ms, 3),
            "query_embedding_p95": round(query_p95_ms, 3),
            "total": round(total_ms, 3),
        },
        "provenance": {
            "runner_sha256": runner_sha256,
            "python_version": platform.python_version(),
            "openai_version": _package_version("openai"),
            "httpx_version": _package_version("httpx"),
            "server_version": server_version,
            "served_models_manifest_sha256": served_models_sha256,
            "served_model_card_sha256": served_model_card_sha256,
        },
    }
    if output is not None:
        artifact = write_private_json_fresh(
            output.expanduser(),
            _canonical_json_bytes(report),
            max_bytes=_MAX_REPORT_BYTES,
        )
        print(f"отчёт: {output} sha256={artifact.sha256}")

    print(
        f"{model}{f' (trunc {truncate_dim})' if truncate_dim else ''} | dim={dimension} | "
        f"вопросов={count}, чанков={len(chunks)}\n"
        f"  recall@1={metrics['recall@1']:.3f}  recall@5={metrics['recall@5']:.3f}  "
        f"MRR@10={metrics['mrr@10']:.3f}  nDCG@10={metrics['ndcg@10']:.3f}  "
        f"индексация корпуса: {corpus_ms / 1000:.1f} c"
    )
    return report


async def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--make-qa", type=int, default=0)
    parser.add_argument("--qa-file", default="/tmp/retrieval_qa.jsonl")
    parser.add_argument("--url")
    parser.add_argument("--model")
    parser.add_argument("--truncate-dim", type=int, default=None)
    parser.add_argument(
        "--query-prefix",
        default=None,
        help="точный префикс запроса; по умолчанию сохраняется текущая Qwen-инструкция",
    )
    parser.add_argument("--passage-prefix", default="", help="точный префикс каждого чанка")
    parser.add_argument("--output", type=Path, default=None, help="новый JSON-отчёт (перезапись запрещена)")
    args = parser.parse_args()

    qa_file = Path(args.qa_file)
    if args.make_qa:
        if args.url or args.model or args.output is not None:
            raise SystemExit("--make-qa нельзя смешивать с параметрами оценки")
        await make_qa(args.make_qa, qa_file)
        return
    if not (args.url and args.model):
        raise SystemExit("нужно --url и --model (или --make-qa N)")
    await evaluate(
        args.url,
        args.model,
        qa_file,
        args.truncate_dim,
        query_prefix=args.query_prefix,
        passage_prefix=args.passage_prefix,
        output=args.output,
    )


if __name__ == "__main__":
    asyncio.run(main())
