from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "qualify_selective_citation_models.py"
    spec = importlib.util.spec_from_file_location("qualify_selective_citation_models", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


qualification = _load_script()


def _case(
    case_id: str,
    *,
    language: str = "ru",
    answerable: bool = True,
    answer: str = "Давление равно 5 МПа",
    context: tuple[str, ...] = ("Рабочее давление равно 5 МПа",),
) -> object:
    return qualification.QualificationCase(
        case_id=case_id,
        answerable=answerable,
        language=language,
        question="Какое давление?" if answerable else None,
        answer=answer if answerable else None,
        claims=(qualification.AtomicClaim(answer, language),) if answerable else (),
        positive_context=context if answerable else (),
        negative_context=("Температура равна 20 °C",) if answerable else (),
        negative_case_id="other" if answerable else None,
        negative_overlap=0.0 if answerable else None,
    )


def test_token_predictions_convert_to_conservative_support_score() -> None:
    score = qualification.support_score_from_token_predictions(
        [{"prob": 0.1}, {"prob": 0.8}, {"prob": 0.3}]
    )

    assert score == pytest.approx(0.2)


@pytest.mark.parametrize("value", [True, math.nan, -0.1, 1.1, "0.5"])
def test_token_predictions_fail_closed_on_invalid_probability(value: object) -> None:
    with pytest.raises(qualification.QualificationError):
        qualification.support_score_from_token_predictions([{"prob": value}])


def test_observation_payload_contains_no_private_text() -> None:
    cases = (
        _case("ragq-case-0001"),
        _case("ragq-case-0002", language="en", answer="Pressure is 5 MPa"),
        _case("ragq-case-0003", language="zh", answerable=False),
    )
    scores = {
        "ragq-case-0001": (qualification.PairScores(0.9, 0.1),),
        "ragq-case-0002": (qualification.PairScores(0.8, 0.2),),
    }

    payload = qualification.build_observation_payload(cases, scores)
    serialized = __import__("json").dumps(payload, ensure_ascii=False)

    assert payload["case_count"] == 3
    assert payload["cases"][2]["claims"] == []
    assert "Давление" not in serialized
    assert "Pressure" not in serialized
    assert set(payload["cases"][0]) == {"case_id", "answerable", "language", "claims"}


def test_observation_payload_requires_exact_answerable_coverage() -> None:
    with pytest.raises(qualification.QualificationError, match="cover"):
        qualification.build_observation_payload((_case("ragq-case-0001"),), {})


def test_calibration_accepts_json_array_claims_and_no_answer_case() -> None:
    cases = (
        _case("ragq-case-0001"),
        _case("ragq-case-0002", answerable=False),
    )
    payload = qualification.build_observation_payload(
        cases,
        {"ragq-case-0001": (qualification.PairScores(0.9, 0.1),)},
    )

    result = qualification._calibrate(
        payload,
        answerability_target=0.5,
        semantic_precision_target=0.5,
    )

    assert result["case_count"] == 2
    assert result["qualified"] is True


def test_roc_auc_is_tie_aware() -> None:
    assert qualification.roc_auc([True, False], [1.0, 0.0]) == 1.0
    assert qualification.roc_auc([True, False], [0.5, 0.5]) == 0.5
    assert qualification.roc_auc([True, False], [0.0, 1.0]) == 0.0


def test_hhem_batch_plan_is_deterministic_and_isolates_long_inputs() -> None:
    lengths = [100, 700, 3000, 1300, 100, 700, 700, 700, 700]

    batches = qualification.hhem_batch_plan(lengths, max_batch_size=8)

    assert batches[0] == (2,)
    assert batches[1] == (3, 1)
    assert batches[2] == (5, 6, 7, 8)
    assert batches[3] == (0, 4)
    assert sorted(index for batch in batches for index in batch) == list(range(len(lengths)))


def test_score_summary_is_language_partitioned_without_text() -> None:
    cases = tuple(
        _case(
            f"ragq-case-{index:04d}",
            language=language,
            answer="私密文本" if language == "zh" else "private text",
        )
        for index, language in enumerate(("ru", "ru", "en", "en", "zh", "zh"), 1)
    )
    scores = {
        item.case_id: (qualification.PairScores(0.9, 0.1),)
        for item in cases
    }

    summary = qualification.summarize_scores(cases, scores)
    serialized = __import__("json").dumps(summary, ensure_ascii=False)

    assert set(summary) == {"ru", "en", "zh"}
    assert all(summary[language]["roc_auc"] == 1.0 for language in summary)
    assert "private text" not in serialized
    assert "私密文本" not in serialized


def test_snapshot_requires_pinned_revision_and_expected_license(tmp_path: Path) -> None:
    revision = "a" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot / "README.md").write_text("---\nlicense: mit\n---\n", encoding="utf-8")

    assert qualification._assert_snapshot(snapshot, revision, "mit") == snapshot.resolve()
    with pytest.raises(qualification.QualificationError, match="license"):
        qualification._assert_snapshot(snapshot, revision, "apache-2.0")
