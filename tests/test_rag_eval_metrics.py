from __future__ import annotations

import json
import math

import pytest

from rag_app.eval.rag_metrics import (
    citation_metrics,
    extract_quantity_mentions,
    mrr_at_k,
    ndcg_at_k,
    normalize_number,
    normalize_unit,
    quantity_unit_metrics,
    recall_at_k,
)


def test_ranked_metrics_use_stable_graded_relevance() -> None:
    ranked = ["sha-a:p3", "sha-a:p2", "sha-a:p1", "sha-a:p4"]
    relevance = {"sha-a:p1": 3, "sha-a:p2": 1, "sha-a:p9": 2}

    recall = recall_at_k(ranked, relevance, 3)
    mrr = mrr_at_k(ranked, relevance, 10)
    ndcg = ndcg_at_k(ranked, relevance, 3)

    assert recall["value"] == 2 / 3
    assert recall["matched_count"] == 2
    assert mrr["value"] == 0.5
    observed = (2**1 - 1) / math.log2(3) + (2**3 - 1) / math.log2(4)
    ideal = (2**3 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3) + (2**1 - 1) / math.log2(4)
    assert ndcg["value"] == pytest.approx(observed / ideal)


def test_ranked_metrics_reject_duplicate_refs_and_invalid_grades() -> None:
    with pytest.raises(ValueError, match="unique"):
        recall_at_k(["doc:p1", "doc:p1"], {"doc:p1": 1}, 5)
    with pytest.raises(ValueError, match="finite"):
        ndcg_at_k(["doc:p1"], {"doc:p1": float("nan")}, 10)
    with pytest.raises(ValueError, match="positive integer"):
        mrr_at_k(["doc:p1"], {"doc:p1": 1}, 0)
    with pytest.raises(ValueError, match="sequence"):
        recall_at_k("doc:p1", {"doc:p1": 1}, 5)
    with pytest.raises(ValueError, match="finite"):
        ndcg_at_k(["doc:p1"], {"doc:p1": 1_000}, 10)


def test_no_answer_ranked_metrics_are_explicit_and_json_friendly() -> None:
    empty = recall_at_k([], {}, 5, answerable=False)
    noisy = ndcg_at_k(["doc:p1"], {}, 10, answerable=False)

    assert empty["value"] is None
    assert not empty["eligible"]
    assert empty["correct_abstention"] is True
    assert noisy["correct_abstention"] is False
    json.dumps({"empty": empty, "noisy": noisy}, sort_keys=True)
    with pytest.raises(ValueError, match="must not declare"):
        mrr_at_k([], {"doc:p1": 1}, 10, answerable=False)


def test_citation_metrics_separate_validity_relevance_and_coverage() -> None:
    result = citation_metrics(
        [2, 2, 4, 0],
        [["doc:p1"], ["doc:p2", "doc:p3"], ["doc:p8"]],
        {"doc:p1", "doc:p2", "doc:p3"},
    )

    assert result["citation_validity"] == 0.5
    assert result["citation_precision"] == 1.0
    assert result["citation_recall"] == 2 / 3
    assert result["citation_f1"] == pytest.approx(0.8)
    assert result["invalid_ranks"] == [4, 0]
    assert result["unique_valid_rank_count"] == 1


def test_irrelevant_valid_citation_reduces_precision() -> None:
    result = citation_metrics(
        [1, 2],
        [["doc:p1"], ["doc:p9"]],
        {"doc:p1", "doc:p2"},
    )

    assert result["citation_validity"] == 1.0
    assert result["citation_precision"] == 0.5
    assert result["citation_recall"] == 0.5
    assert result["citation_f1"] == 0.5


def test_no_answer_citations_are_explicit() -> None:
    quiet = citation_metrics([], [], set(), answerable=False)
    cited = citation_metrics([1], [["doc:p1"]], set(), answerable=False)

    assert quiet["citation_validity"] == 1.0
    assert quiet["citation_precision"] is None
    assert quiet["correct_abstention"] is True
    assert cited["correct_abstention"] is False


def test_number_and_unit_normalization_preserve_sign_and_decimal() -> None:
    assert normalize_number("−００１６,５０００") == "-16.5"
    assert normalize_number("-0.50") == "-0.5"
    assert normalize_number("+000") == "0"
    assert normalize_unit("МПа") == "MPa"
    assert normalize_unit("м³/ч") == "m3/h"
    assert normalize_unit("°С") == "degC"
    with pytest.raises(ValueError, match="unsupported quantity value"):
        normalize_number("1e3")
    with pytest.raises(ValueError, match="unsupported quantity unit"):
        normalize_unit("furlong")
    with pytest.raises(ValueError, match="unsupported quantity unit"):
        normalize_unit("mPa")


def test_ambiguous_comma_requires_an_explicit_policy() -> None:
    assert normalize_number("0,125") == "0.125"
    with pytest.raises(ValueError, match="ambiguous decimal/thousands comma"):
        normalize_number("1,234")
    assert normalize_number("1,234", comma_policy="decimal") == "1.234"
    assert normalize_number("1,234", comma_policy="thousands") == "1234"


def test_decimal_comma_cannot_match_a_thousandfold_value() -> None:
    correct = quantity_unit_metrics(
        "Давление 0,125 MPa.",
        [{"value": "0.125", "unit": "MPa"}],
    )
    wrong = quantity_unit_metrics(
        "Давление 0,125 MPa.",
        [{"value": "125", "unit": "MPa"}],
    )

    assert correct["quantity_unit_accuracy"] == 1.0
    assert wrong["quantity_unit_accuracy"] == 0.0
    assert wrong["quantity_unit_recall"] == 0.0


@pytest.mark.parametrize("value", ["1,234,56", "1,234,5678", "0,125,5"])
def test_repeated_mixed_commas_fail_closed(value: str) -> None:
    with pytest.raises(ValueError, match="ambiguous repeated commas"):
        normalize_number(value)
    with pytest.raises(ValueError, match="ambiguous repeated commas"):
        quantity_unit_metrics(
            f"Давление {value} MPa.",
            [{"value": "123456", "unit": "MPa"}],
        )


def test_quantity_metrics_match_aliases_and_strip_citation_numbers() -> None:
    result = quantity_unit_metrics(
        "Минимальное давление составляет −１６,５ МПа [1].",
        [{"value": "-16.5", "unit": "MPa"}],
    )

    assert result["quantity_unit_accuracy"] == 1.0
    assert result["quantity_unit_recall"] == 1.0
    assert result["mentioned_number_count"] == 1
    assert result["unsupported_number_rate"] == 0.0


def test_wrong_unit_fails_closed_without_losing_numeric_support() -> None:
    mentions = extract_quantity_mentions("Давление 16,5 furlong.")
    result = quantity_unit_metrics(
        "Давление 16,5 furlong.",
        [{"value": "16.5", "unit": "MPa"}],
    )

    assert mentions == [
        {"value": "16.5", "unit": None, "raw_unit": "furlong", "unit_valid": False}
    ]
    assert result["quantity_unit_recall"] == 0.0
    assert result["quantity_unit_accuracy"] == 0.0
    assert result["unsupported_number_rate"] == 0.0
    assert result["invalid_unit_count"] == 1


def test_quantity_metrics_distinguish_supported_context_from_unsupported_numbers() -> None:
    answer = "Рабочее давление 16,5 MPa, испытательное давление 20 bar [2]."
    expected = [{"value": "16.5", "unit": "MPa"}]

    strict = quantity_unit_metrics(answer, expected)
    contextual = quantity_unit_metrics(
        answer,
        expected,
        supported_quantities=[
            *expected,
            {"value": "20", "unit": "bar"},
        ],
    )

    assert strict["unsupported_number_count"] == 1
    assert strict["unsupported_number_rate"] == 0.5
    assert strict["quantity_unit_accuracy"] == 0.0
    assert contextual["unsupported_number_rate"] == 0.0
    assert contextual["quantity_unit_accuracy"] == 1.0


def test_no_answer_quantity_metrics_are_explicit() -> None:
    quiet = quantity_unit_metrics("Ответа в документах нет.", [], answerable=False)
    noisy = quantity_unit_metrics("Возможно, 42 psi.", [], answerable=False)

    assert quiet["quantity_unit_accuracy"] is None
    assert quiet["correct_abstention"] is True
    assert noisy["correct_abstention"] is False
    assert noisy["unsupported_number_rate"] == 1.0
    json.dumps({"quiet": quiet, "noisy": noisy}, sort_keys=True)


def test_expected_quantities_must_be_supported() -> None:
    with pytest.raises(ValueError, match="must include"):
        quantity_unit_metrics(
            "16.5 MPa",
            [{"value": "16.5", "unit": "MPa"}],
            supported_quantities=[{"value": "20", "unit": "bar"}],
        )
