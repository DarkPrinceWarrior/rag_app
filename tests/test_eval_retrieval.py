from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import math
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts/eval_retrieval.py"
_SPEC = importlib.util.spec_from_file_location("eval_retrieval_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
target = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = target
_SPEC.loader.exec_module(target)


class _FakeEmbeddings:
    response_model = "candidate/model"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        assert model == "candidate/model"
        self.calls.append(input)
        data = []
        for index, text in enumerate(input):
            if "target-a" in text:
                vector = [1.0, 0.0]
            elif "target-b" in text:
                vector = [0.0, 1.0]
            else:
                vector = [1.0, 1.0]
            data.append(SimpleNamespace(index=index, embedding=vector))
        return SimpleNamespace(data=data, model=self.response_model)


class _FakeModels:
    identifiers = ["candidate/model"]

    async def list(self) -> SimpleNamespace:
        return SimpleNamespace(data=[SimpleNamespace(id=value) for value in self.identifiers])


class _FakeChatCompletions:
    duplicate_questions = False

    async def create(self, *, model: str, messages: list[dict[str, str]], **kwargs: Any) -> SimpleNamespace:
        assert model == "candidate/model"
        assert kwargs["temperature"] == 0.0
        assert isinstance(kwargs["seed"], int)
        prompt = messages[0]["content"]
        marker = "одинаковый" if self.duplicate_questions else (
            "target-a" if "target-a" in prompt else "target-b"
        )
        return SimpleNamespace(
            model=model,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=f"Какое требование относится к {marker}?")
                )
            ],
        )


class _FakeOpenAI:
    instances: list[_FakeOpenAI] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.embeddings = _FakeEmbeddings()
        self.models = _FakeModels()
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())
        self.closed = False
        self.instances.append(self)

    async def __aenter__(self) -> _FakeOpenAI:
        return self

    async def __aexit__(self, *_: Any) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _stub_server_version(monkeypatch: pytest.MonkeyPatch) -> None:
    async def probe_server_version(_: Any, base_url: str) -> str:
        assert base_url.endswith("/v1")
        return "test-vllm"

    monkeypatch.setattr(target, "_probe_server_version", probe_server_version)


def _qa_file(path: Path, rows: list[dict[str, str]]) -> tuple[Path, bytes]:
    raw = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
    path.write_bytes(raw)
    return path, raw


def _chunks() -> list[Any]:
    return [
        SimpleNamespace(id="chunk-a", text_en="target-a english", text_ru="target-a русский"),
        SimpleNamespace(id="chunk-b", text_en="target-b english", text_ru="target-b русский"),
    ]


def test_prepare_text_truncates_body_before_applying_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(target, "_INPUT_TRUNCATION_CHARS", 8)

    assert target._prepare_text("prefix-", "body") == "prefix-body"
    assert target._prepare_text("prefix-", "123456789") == "prefix-12345678"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:18018/not-v1",
        "http://127.0.0.1:18018/v1/embeddings",
        "http://example.com:18018/v1",
    ],
)
def test_validate_endpoint_requires_exact_loopback_v1(url: str) -> None:
    with pytest.raises(ValueError, match="loopback /v1"):
        target._validate_endpoint(url, "candidate/model")


def test_corpus_manifest_is_order_independent() -> None:
    chunks = _chunks()

    assert target._corpus_manifest_sha256(chunks) == target._corpus_manifest_sha256(
        list(reversed(chunks))
    )


def test_rank_ties_are_broken_by_chunk_id_not_input_order() -> None:
    chunks = list(reversed(_chunks()))

    order = target._ranked_indices(chunks, [0.5, 0.5])

    assert [chunks[index].id for index in order] == ["chunk-a", "chunk-b"]


def test_metrics_cover_nonperfect_ranks_and_cutoff() -> None:
    metrics = target._metrics_from_ranks([1, 2, 11])

    assert metrics["recall@1"] == pytest.approx(1 / 3)
    assert metrics["recall@5"] == pytest.approx(2 / 3)
    assert metrics["mrr@10"] == pytest.approx(0.5)
    assert metrics["ndcg@10"] == pytest.approx(
        (1.0 + 1.0 / math.log2(3)) / 3
    )


@pytest.mark.parametrize("value", [0, -1, target._MAX_TRUNCATE_DIM + 1])
def test_truncate_dim_bounds_fail_closed(value: int) -> None:
    with pytest.raises(ValueError, match="truncate_dim"):
        target._validate_truncate_dim(value)


def test_load_qa_rejects_empty_and_duplicate_questions(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        target._load_qa(empty)

    duplicate = tmp_path / "duplicate.jsonl"
    _qa_file(
        duplicate,
        [
            {"chunk_id": "chunk-a", "question": "Где находится target-a?"},
            {"chunk_id": "chunk-b", "question": "Где находится target-a?"},
        ],
    )
    with pytest.raises(ValueError, match="unique"):
        target._load_qa(duplicate)


def test_validate_corpus_rejects_missing_gold_chunk() -> None:
    with pytest.raises(ValueError, match="absent"):
        target._validate_corpus(
            _chunks(),
            [{"chunk_id": "missing", "question": "Какой здесь отсутствует фрагмент?"}],
        )


@pytest.mark.parametrize(
    "vector,dim",
    [
        ([], None),
        ([0.0, 0.0], None),
        ([float("nan"), 1.0], None),
        ([1.0, 0.0], 3),
    ],
)
def test_norm_rejects_invalid_vectors(vector: list[float], dim: int | None) -> None:
    with pytest.raises(ValueError):
        target._norm(vector, dim)


def test_evaluate_writes_redacted_create_only_report_with_exact_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeOpenAI.instances.clear()
    monkeypatch.setattr(target, "AsyncOpenAI", _FakeOpenAI)

    async def load_chunks() -> list[Any]:
        return _chunks()

    monkeypatch.setattr(target, "load_chunks", load_chunks)
    qa_path, qa_raw = _qa_file(
        tmp_path / "qa.jsonl",
        [
            {"chunk_id": "chunk-a", "question": "Где находится target-a?"},
            {"chunk_id": "chunk-b", "question": "Где находится target-b?"},
        ],
    )
    output = tmp_path / "report.json"
    report = asyncio.run(
        target.evaluate(
            "http://127.0.0.1:18018/v1/",
            "candidate/model",
            qa_path,
            2,
            query_prefix="query: ",
            passage_prefix="passage: ",
            output=output,
        )
    )

    persisted = json.loads(output.read_text("utf-8"))
    assert persisted == report
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert persisted["schema_version"] == "retrieval-embedding-eval-v2"
    assert persisted["endpoint"]["model"] == "<redacted>"
    assert persisted["endpoint"]["url"] == "<redacted>"
    assert persisted["endpoint"]["model_sha256"] == hashlib.sha256(b"candidate/model").hexdigest()
    assert persisted["endpoint"]["url_sha256"] == hashlib.sha256(
        b"http://127.0.0.1:18018/v1"
    ).hexdigest()
    assert persisted["qa"] == {
        "sha256": hashlib.sha256(qa_raw).hexdigest(),
        "size_bytes": len(qa_raw),
        "count": 2,
    }
    assert persisted["corpus"]["count"] == 2
    assert persisted["dim"] == 2
    assert persisted["native_dim"] == 2
    assert persisted["metrics"] == {
        "recall@1": 1.0,
        "recall@5": 1.0,
        "mrr@10": 1.0,
        "ndcg@10": 1.0,
    }
    assert [row["rank"] for row in persisted["cases"]] == [1, 1]
    assert all(len(row["question_sha256"]) == 64 for row in persisted["cases"])
    assert all(row["discounted_gain_at_10"] == 1.0 for row in persisted["cases"])
    assert persisted["configuration"]["query_prefix"] == "query: "
    assert persisted["configuration"]["passage_prefix"] == "passage: "
    assert persisted["configuration"]["input_truncation_policy"] == "raw-body-before-prefix"
    assert len(persisted["latency_ms"]["query_embedding_samples"]) == 3
    assert persisted["latency_ms"]["diagnostic_only"] is True
    assert persisted["provenance"]["server_version"] == "test-vllm"
    calls = _FakeOpenAI.instances[0].embeddings.calls
    assert calls[0] == ["passage: target-a english", "passage: target-b english"]
    assert calls[1] == ["passage: target-a english", "passage: target-b english"]
    assert calls[2] == ["passage: target-a русский", "passage: target-b русский"]
    assert calls[3:] == [
        ["query: Где находится target-a?", "query: Где находится target-b?"],
        ["query: Где находится target-a?", "query: Где находится target-b?"],
        ["query: Где находится target-a?", "query: Где находится target-b?"],
    ]
    client = _FakeOpenAI.instances[0]
    assert client.kwargs["max_retries"] == 0
    assert client.kwargs["http_client"].follow_redirects is False
    assert client.kwargs["http_client"].trust_env is False
    assert client.closed is True
    raw_report = output.read_text("utf-8")
    assert "candidate/model" not in raw_report
    assert "127.0.0.1" not in raw_report

    with pytest.raises(FileExistsError, match="refusing to replace"):
        asyncio.run(
            target.evaluate(
                "http://127.0.0.1:18018/v1",
                "candidate/model",
                qa_path,
                2,
                query_prefix="query: ",
                passage_prefix="passage: ",
                output=output,
            )
        )


def test_evaluate_preserves_legacy_default_query_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeOpenAI.instances.clear()
    monkeypatch.setattr(target, "AsyncOpenAI", _FakeOpenAI)
    monkeypatch.setattr(target.settings, "embed_query_instruction", "legacy instruction")

    async def load_chunks() -> list[Any]:
        return _chunks()

    monkeypatch.setattr(target, "load_chunks", load_chunks)
    qa_path, _ = _qa_file(
        tmp_path / "qa.jsonl",
        [{"chunk_id": "chunk-a", "question": "Где находится target-a?"}],
    )

    report = asyncio.run(
        target.evaluate(
            "http://localhost:18018/v1",
            "candidate/model",
            qa_path,
            None,
        )
    )

    assert (
        report["configuration"]["query_prefix"]
        == "Instruct: legacy instruction\nQuery: "
    )
    assert _FakeOpenAI.instances[0].embeddings.calls[3] == [
        "Instruct: legacy instruction\nQuery: Где находится target-a?"
    ]


def test_evaluate_rejects_wrong_response_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeOpenAI.instances.clear()
    monkeypatch.setattr(target, "AsyncOpenAI", _FakeOpenAI)
    monkeypatch.setattr(_FakeEmbeddings, "response_model", "wrong/model")

    async def load_chunks() -> list[Any]:
        return _chunks()

    monkeypatch.setattr(target, "load_chunks", load_chunks)
    qa_path, _ = _qa_file(
        tmp_path / "qa.jsonl",
        [{"chunk_id": "chunk-a", "question": "Где находится target-a?"}],
    )

    with pytest.raises(ValueError, match="different model"):
        asyncio.run(
            target.evaluate(
                "http://127.0.0.1:18018/v1",
                "candidate/model",
                qa_path,
                None,
            )
        )

    assert _FakeOpenAI.instances[0].closed is True


def test_evaluate_requires_requested_model_in_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeOpenAI.instances.clear()
    monkeypatch.setattr(target, "AsyncOpenAI", _FakeOpenAI)
    monkeypatch.setattr(_FakeModels, "identifiers", ["other/model"])

    async def load_chunks() -> list[Any]:
        return _chunks()

    monkeypatch.setattr(target, "load_chunks", load_chunks)
    qa_path, _ = _qa_file(
        tmp_path / "qa.jsonl",
        [{"chunk_id": "chunk-a", "question": "Где находится target-a?"}],
    )

    with pytest.raises(ValueError, match="absent"):
        asyncio.run(
            target.evaluate(
                "http://127.0.0.1:18018/v1",
                "candidate/model",
                qa_path,
                None,
            )
        )

    assert _FakeOpenAI.instances[0].embeddings.calls == []
    assert _FakeOpenAI.instances[0].closed is True


def test_make_qa_writes_exact_create_only_private_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeOpenAI.instances.clear()
    monkeypatch.setattr(target, "AsyncOpenAI", _FakeOpenAI)
    monkeypatch.setattr(target.settings, "llm_base_url", "http://127.0.0.1:18018/v1")
    monkeypatch.setattr(target.settings, "llm_model", "candidate/model")
    monkeypatch.setattr(target.settings, "llm_api_key", "secret")

    async def load_chunks() -> list[Any]:
        return _chunks()

    monkeypatch.setattr(target, "load_chunks", load_chunks)
    output = tmp_path / "qa.jsonl"

    asyncio.run(target.make_qa(2, output))

    rows, _, _ = target._load_qa(output)
    assert len(rows) == 2
    assert len({row["question"].casefold() for row in rows}) == 2
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert _FakeOpenAI.instances[0].closed is True

    with pytest.raises(FileExistsError, match="refusing to replace"):
        asyncio.run(target.make_qa(2, output))


def test_make_qa_fails_without_exact_unique_count_and_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeOpenAI.instances.clear()
    monkeypatch.setattr(target, "AsyncOpenAI", _FakeOpenAI)
    monkeypatch.setattr(_FakeChatCompletions, "duplicate_questions", True)
    monkeypatch.setattr(target.settings, "llm_base_url", "http://127.0.0.1:18018/v1")
    monkeypatch.setattr(target.settings, "llm_model", "candidate/model")
    monkeypatch.setattr(target.settings, "llm_api_key", "secret")

    async def load_chunks() -> list[Any]:
        return _chunks()

    monkeypatch.setattr(target, "load_chunks", load_chunks)
    output = tmp_path / "qa.jsonl"

    with pytest.raises(ValueError, match="expected exactly 2"):
        asyncio.run(target.make_qa(2, output))

    assert not output.exists()
