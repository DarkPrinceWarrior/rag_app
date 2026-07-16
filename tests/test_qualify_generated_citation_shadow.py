from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


qualification = _load_script("qualify_selective_citation_models")
shadow = _load_script("qualify_generated_citation_shadow")


def _case(case_id: str = "ragq-case-0001", *, language: str = "ru") -> object:
    return qualification.QualificationCase(
        case_id=case_id,
        answerable=True,
        language=language,
        question="Какое давление?",
        positive_context=("Рабочее давление равно 5 МПа.",),
    )


def test_runtime_snapshot_is_strict_and_hashes_model_root() -> None:
    snapshot = shadow.parse_runtime_snapshot(
        {"version": "0.24.0"},
        {
            "data": [
                {
                    "id": "qwen3.5-35b-a3b",
                    "created": 999999999,
                    "max_model_len": 16384,
                    "root": "/private/model/path",
                }
            ]
        },
        "process_start_time_seconds 123.5\n",
        expected_model="qwen3.5-35b-a3b",
    )

    assert snapshot.version == "0.24.0"
    assert snapshot.process_start_time_seconds == 123.5
    assert snapshot.model_root_sha256 != "/private/model/path"
    assert len(snapshot.digest) == 64


@pytest.mark.parametrize(
    "url",
    ["https://127.0.0.1:8006", "http://192.168.1.2:8006", "http://user@localhost:8006"],
)
def test_qwen_endpoint_must_be_plain_loopback(url: str) -> None:
    with pytest.raises(qualification.QualificationError, match="loopback|credentials"):
        shadow._require_loopback_url(url)


def test_teacher_labels_require_exact_unique_index_coverage() -> None:
    labels = shadow.parse_teacher_labels(
        {
            "claims": [
                {"index": 1, "verdict": "unsupported"},
                {"index": 0, "verdict": "supported"},
            ]
        },
        claim_count=2,
    )

    assert labels == (True, False)
    with pytest.raises(qualification.QualificationError, match="coverage|every"):
        shadow.parse_teacher_labels(
            {"claims": [{"index": 0, "verdict": "supported"}]},
            claim_count=2,
        )
    assert shadow.parse_teacher_labels(
        {"claims": [{"index": 0, "verdict": "supported"}]},
        claim_count=2,
        missing_as_unsupported=True,
    ) == (True, False)


def test_generated_observation_contains_no_private_text() -> None:
    private_answer = "Секретное давление равно 5 МПа."
    claim = qualification.AtomicClaim(private_answer, "ru")
    generated = shadow.GeneratedCase(
        qualification=qualification.QualificationCase(
            case_id="ragq-case-0001",
            answerable=True,
            language="ru",
            question="Секретный вопрос",
            answer=private_answer,
            claims=(claim,),
            positive_context=("Секретный источник",),
        ),
        teacher_labels=(True,),
    )

    payload = shadow.build_observation_payload(
        (generated,),
        {"ragq-case-0001": (0.75,)},
    )
    serialized = json.dumps(payload, ensure_ascii=False)

    assert set(payload["cases"][0]) == {"case_id", "answerable", "language", "claims"}
    assert payload["cases"][0]["claims"] == [{"score": 0.75, "supported": True}]
    assert "Секрет" not in serialized
    assert private_answer not in serialized


def test_source_question_is_restored_for_no_answer_case() -> None:
    no_answer = qualification.QualificationCase(
        case_id="ragq-case-0001",
        answerable=False,
        language="zh",
    )
    source = type("Source", (), {"case_id": "ragq-case-0001", "question": "问题是什么？"})()

    restored = shadow.attach_source_questions((no_answer,), (source,))

    assert restored[0].question == "问题是什么？"


def test_language_threshold_and_router_choose_best_eligible_coverage() -> None:
    curve = {
        "answerability_target": 0.85,
        "semantic_precision_target": 0.90,
        "language_curves": {
            language: [
                {
                    "threshold": 0.5,
                    "coverage": 0.4,
                    "semantic_precision": 0.95,
                    "answerability_accuracy": 0.90,
                },
                {
                    "threshold": 0.6,
                    "coverage": 0.3,
                    "semantic_precision": 1.0,
                    "answerability_accuracy": 0.95,
                },
            ]
            for language in ("ru", "en", "zh")
        },
    }
    thresholds = shadow.select_language_thresholds(curve)
    backends = {
        "hhem": {"language_thresholds": thresholds},
        "lettuce_router": {
            "language_thresholds": {
                language: {**thresholds[language], "coverage": 0.5}
                for language in ("ru", "en", "zh")
            }
        },
    }

    router = shadow.choose_router(backends)

    assert thresholds["ru"]["threshold"] == 0.5
    assert router["gate"] == "GO"
    assert all(route["backend"] == "lettuce_router" for route in router["routes"].values())


def test_router_fails_when_one_language_has_no_eligible_backend() -> None:
    point = {
        "threshold": 0.5,
        "coverage": 0.4,
        "semantic_precision": 0.95,
        "answerability_accuracy": 0.90,
    }
    router = shadow.choose_router(
        {
            "hhem": {
                "language_thresholds": {"ru": point, "en": point, "zh": None},
            }
        }
    )

    assert router["gate"] == "NO-GO"
    assert router["routes"]["zh"] is None
