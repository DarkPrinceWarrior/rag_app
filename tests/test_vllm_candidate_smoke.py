from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_smoke() -> Any:
    path = Path(__file__).parents[1] / "deploy" / "vllm" / "smoke_candidate.py"
    spec = importlib.util.spec_from_file_location("vllm_candidate_smoke_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reranker_probes_cover_languages_numbers_and_both_positions() -> None:
    smoke = _load_smoke()

    assert {probe.name for probe in smoke.RERANK_PROBES} == {"en", "ru", "zh", "numeric"}
    assert {probe.relevant_index for probe in smoke.RERANK_PROBES} == {0, 1}


def test_reranker_payload_uses_application_manual_template() -> None:
    smoke = _load_smoke()

    payload = smoke._rerank_payload("qwen3-reranker-4b", smoke.RERANK_PROBES[0])

    assert "<Instruct>:" in payload["query"]
    assert "<Query>:" in payload["query"]
    assert all(document.startswith("<Document>:") for document in payload["documents"])
    assert all(document.endswith("<think>\n\n</think>\n\n") for document in payload["documents"])


def test_index_aware_validation_rejects_sorted_but_semantically_wrong_results() -> None:
    smoke = _load_smoke()
    probe = smoke.RERANK_PROBES[0]
    assert probe.relevant_index == 1
    response = {
        "results": [
            {"index": 0, "relevance_score": 0.99},
            {"index": 1, "relevance_score": 0.01},
        ]
    }

    with pytest.raises(AssertionError, match="semantic ranking failed"):
        smoke._validate_rerank_response(response, probe)


def test_index_aware_validation_uses_indexes_not_response_order() -> None:
    smoke = _load_smoke()
    probe = smoke.RERANK_PROBES[0]
    response = {
        "results": [
            {"index": 0, "relevance_score": 0.05},
            {"index": 1, "relevance_score": 0.95},
        ]
    }

    assert smoke._validate_rerank_response(response, probe) == pytest.approx(0.9)


@pytest.mark.parametrize(
    "results",
    [
        [{"index": 1, "relevance_score": 0.9}],
        [
            {"index": 1, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.1},
        ],
        [
            {"index": 1, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.1},
        ],
        [
            {"index": 1, "relevance_score": float("nan")},
            {"index": 0, "relevance_score": 0.1},
        ],
    ],
)
def test_index_aware_validation_fails_closed_on_invalid_contract(results: list[dict[str, Any]]) -> None:
    smoke = _load_smoke()

    with pytest.raises(AssertionError):
        smoke._validate_rerank_response({"results": results}, smoke.RERANK_PROBES[0])
