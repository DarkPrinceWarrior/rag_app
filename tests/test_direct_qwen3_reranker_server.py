from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def _load_server(monkeypatch: pytest.MonkeyPatch) -> Any:
    torch = ModuleType("torch")
    torch.are_deterministic_algorithms_enabled = lambda: True  # type: ignore[attr-defined]
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = object  # type: ignore[attr-defined]
    transformers.AutoTokenizer = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    path = Path(__file__).parents[1] / "scripts" / "direct_qwen3_reranker_server.py"
    spec = importlib.util.spec_from_file_location("direct_qwen3_reranker_server_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_rerank_contract_preserves_indexes_and_sorts_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _load_server(monkeypatch)
    server._runtime = SimpleNamespace(score=lambda query, documents: [0.1, 0.9])

    response = server.rerank(
        server.RerankRequest(
            model="qwen3-reranker-4b",
            query="wrapped query",
            documents=["wrapped document a", "wrapped document b"],
        )
    )

    assert [(item.index, item.relevance_score) for item in response.results] == [
        (1, 0.9),
        (0, 0.1),
    ]


def test_models_and_health_expose_deterministic_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _load_server(monkeypatch)
    server._runtime = object()

    assert server.models().data[0].id == "qwen3-reranker-4b"
    health = server.healthz()
    assert health.status == "ok"
    assert health.model_loaded is True
    assert health.deterministic_algorithms is True
    assert health.dtype == "bfloat16"


def test_health_exposes_requested_float32_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIRECT_RERANK_DTYPE", "float32")
    server = _load_server(monkeypatch)

    assert server.healthz().dtype == "float32"


def test_rerank_rejects_unknown_model(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _load_server(monkeypatch)
    server._runtime = SimpleNamespace(score=lambda query, documents: [0.5])

    with pytest.raises(server.HTTPException) as exc_info:
        server.rerank(
            server.RerankRequest(
                model="another-model",
                query="wrapped query",
                documents=["wrapped document"],
            )
        )

    assert exc_info.value.status_code == 404


def test_token_packing_preserves_suffix_when_context_overflows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _load_server(monkeypatch)
    suffix = (90, 91, 92)

    packed = server._pack_token_ids(
        [*range(20), *suffix],
        suffix,
        max_length=10,
    )

    assert packed == [*range(7), *suffix]
    assert tuple(packed[-len(suffix) :]) == suffix


def test_token_packing_rejects_missing_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _load_server(monkeypatch)

    with pytest.raises(RuntimeError, match="missing the official suffix"):
        server._pack_token_ids([1, 2, 3], (8, 9), max_length=10)
