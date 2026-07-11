"""Deterministic downstream RAG metrics over stable, external evidence references.

The functions in this module intentionally know nothing about database rows, chunk
UUIDs, model clients, or application state. Callers map retrieved units to stable
corpus references (for example ``source-sha256:page:12``) before scoring.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Literal, TypedDict

from rag_app.pipeline.validate import extract_numbers

EvidenceRef = str
CommaPolicy = Literal["reject_ambiguous", "decimal", "thousands"]


class RankedMetricResult(TypedDict):
    metric: str
    k: int
    answerable: bool
    eligible: bool
    value: float | None
    retrieved_count: int
    relevant_count: int
    matched_count: int
    correct_abstention: bool | None


class CitationMetricResult(TypedDict):
    answerable: bool
    eligible: bool
    citation_validity: float
    citation_precision: float | None
    citation_recall: float | None
    citation_f1: float | None
    citation_count: int
    valid_citation_count: int
    unique_valid_rank_count: int
    relevant_cited_rank_count: int
    relevant_evidence_count: int
    covered_evidence_count: int
    invalid_ranks: list[int]
    correct_abstention: bool | None


class QuantityMention(TypedDict):
    value: str
    unit: str | None
    raw_unit: str
    unit_valid: bool


class QuantityMetricResult(TypedDict):
    answerable: bool
    eligible: bool
    quantity_unit_accuracy: float | None
    quantity_unit_recall: float | None
    unsupported_number_rate: float
    expected_quantity_count: int
    matched_quantity_count: int
    mentioned_number_count: int
    unsupported_number_count: int
    invalid_unit_count: int
    correct_abstention: bool | None


_CITATION_RE = re.compile(r"\[\d{1,4}\]")
_NUMBER_LITERAL_RE = re.compile(
    r"[+\-\N{MINUS SIGN}\N{EN DASH}\N{EM DASH}]?\s*"
    r"[0-9\N{FULLWIDTH DIGIT ZERO}-\N{FULLWIDTH DIGIT NINE}]+"
    r"(?:[ \N{NO-BREAK SPACE},][0-9\N{FULLWIDTH DIGIT ZERO}-\N{FULLWIDTH DIGIT NINE}]{3})*"
    r"(?:[.,][0-9\N{FULLWIDTH DIGIT ZERO}-\N{FULLWIDTH DIGIT NINE}]+)?"
)
_NUMBER_MENTION_RE = re.compile(
    rf"(?<![\w\d])(?P<number>{_NUMBER_LITERAL_RE.pattern})(?![\w\d])"
    r"(?:\s*(?P<unit>(?:[%\N{DEGREE SIGN}\N{DEGREE CELSIUS}\N{MICRO SIGN}\N{GREEK SMALL LETTER MU}]"
    r"|[A-Za-zА-Яа-яЁё])"
    r"[%\N{DEGREE SIGN}\N{DEGREE CELSIUS}\N{MICRO SIGN}\N{GREEK SMALL LETTER MU}"
    r"A-Za-zА-Яа-яЁё0-9\N{SUPERSCRIPT TWO}\N{SUPERSCRIPT THREE}/^\N{MIDDLE DOT}*_-]{0,23}))?"
)
_FULLWIDTH_DIGITS = str.maketrans({chr(0xFF10 + index): str(index) for index in range(10)})
_MINUS_CHARS = {"-", "\N{MINUS SIGN}", "\N{EN DASH}", "\N{EM DASH}"}


def _unit_key(value: str) -> str:
    return (
        value.strip()
        .replace(" ", "")
        .replace("\N{SUPERSCRIPT TWO}", "2")
        .replace("\N{SUPERSCRIPT THREE}", "3")
        .replace("^", "")
        .replace("\N{MIDDLE DOT}", "*")
        .replace("\N{GREEK SMALL LETTER MU}", "\N{MICRO SIGN}")
    )


_UNIT_GROUPS: dict[str, tuple[str, ...]] = {
    "%": ("%",),
    "degC": (
        "\N{DEGREE SIGN}C",
        "\N{DEGREE SIGN}c",
        "\N{DEGREE SIGN}С",
        "\N{DEGREE SIGN}с",
        "\N{DEGREE CELSIUS}",
    ),
    "degF": ("\N{DEGREE SIGN}F",),
    "K": ("K", "К"),
    "Pa": ("Pa", "Па"),
    "kPa": ("kPa", "кПа"),
    "MPa": ("MPa", "МПа"),
    "GPa": ("GPa", "ГПа"),
    "bar": ("bar", "Bar", "бар"),
    "mbar": ("mbar", "мбар"),
    "psi": ("psi",),
    "mm": ("mm", "мм"),
    "cm": ("cm", "см"),
    "m": ("m", "м"),
    "km": ("km", "км"),
    "mm2": ("mm2", "mm^2", "mm\N{SUPERSCRIPT TWO}", "мм2", "мм\N{SUPERSCRIPT TWO}"),
    "cm2": ("cm2", "cm^2", "cm\N{SUPERSCRIPT TWO}", "см2", "см\N{SUPERSCRIPT TWO}"),
    "m2": ("m2", "m^2", "m\N{SUPERSCRIPT TWO}", "м2", "м\N{SUPERSCRIPT TWO}"),
    "mm3": ("mm3", "mm^3", "mm\N{SUPERSCRIPT THREE}", "мм3", "мм\N{SUPERSCRIPT THREE}"),
    "cm3": ("cm3", "cm^3", "cm\N{SUPERSCRIPT THREE}", "см3", "см\N{SUPERSCRIPT THREE}"),
    "m3": ("m3", "m^3", "m\N{SUPERSCRIPT THREE}", "м3", "м\N{SUPERSCRIPT THREE}"),
    "g": ("g", "г"),
    "kg": ("kg", "кг"),
    "t": ("t", "т"),
    "s": ("s", "sec", "с", "сек"),
    "min": ("min", "мин"),
    "h": ("h", "hr", "ч", "час"),
    "d": ("d", "day", "д", "дн"),
    "mL": ("mL", "ml", "мл"),
    "L": ("L", "l", "л"),
    "Hz": ("Hz", "Гц"),
    "kHz": ("kHz", "кГц"),
    "W": ("W", "Вт"),
    "kW": ("kW", "кВт"),
    "MW": ("MW", "МВт"),
    "V": ("V", "В"),
    "kV": ("kV", "кВ"),
    "A": ("A", "А"),
    "mA": ("mA", "мА"),
    "N": ("N", "Н"),
    "kN": ("kN", "кН"),
    "N*m": ("N*m", "N\N{MIDDLE DOT}m", "Н*м", "Н\N{MIDDLE DOT}м"),
    "kN*m": ("kN*m", "kN\N{MIDDLE DOT}m", "кН*м", "кН\N{MIDDLE DOT}м"),
    "m/s": ("m/s", "м/с"),
    "m/s2": ("m/s2", "m/s^2", "m/s\N{SUPERSCRIPT TWO}", "м/с2", "м/с\N{SUPERSCRIPT TWO}"),
    "kg/m3": ("kg/m3", "kg/m^3", "kg/m\N{SUPERSCRIPT THREE}", "кг/м3", "кг/м\N{SUPERSCRIPT THREE}"),
    "m3/h": ("m3/h", "m^3/h", "m\N{SUPERSCRIPT THREE}/h", "м3/ч", "м\N{SUPERSCRIPT THREE}/ч"),
    "L/min": ("L/min", "л/мин"),
    "rpm": ("rpm", "об/мин"),
}


def _build_unit_aliases() -> dict[str, str]:
    output: dict[str, str] = {}
    for canonical, aliases in _UNIT_GROUPS.items():
        for alias in aliases:
            key = _unit_key(alias)
            previous = output.get(key)
            if previous is not None and previous != canonical:
                raise RuntimeError(f"conflicting unit alias {alias!r}: {previous} vs {canonical}")
            output[key] = canonical
    return output


_UNIT_ALIASES = _build_unit_aliases()


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def _validated_ranked_refs(ranked_refs: Sequence[EvidenceRef]) -> list[EvidenceRef]:
    if isinstance(ranked_refs, (str, bytes)):
        raise ValueError("ranked evidence references must be a sequence of strings")
    ranked = list(ranked_refs)
    if any(not isinstance(ref, str) or not ref.strip() for ref in ranked):
        raise ValueError("ranked evidence references must be non-empty strings")
    if len(ranked) != len(set(ranked)):
        raise ValueError("ranked evidence references must be unique")
    return ranked


def _validated_relevance(
    relevance: Mapping[EvidenceRef, float],
    *,
    answerable: bool,
) -> dict[EvidenceRef, float]:
    validated: dict[EvidenceRef, float] = {}
    for ref, raw_grade in relevance.items():
        if not isinstance(ref, str) or not ref.strip():
            raise ValueError("relevance references must be non-empty strings")
        if isinstance(raw_grade, bool) or not isinstance(raw_grade, (int, float)):
            raise ValueError("relevance grades must be finite non-negative numbers")
        grade = float(raw_grade)
        if not math.isfinite(grade) or not 0 <= grade <= 100:
            raise ValueError("relevance grades must be finite non-negative numbers")
        validated[ref] = grade
    relevant = {ref for ref, grade in validated.items() if grade > 0}
    if answerable and not relevant:
        raise ValueError("answerable questions require at least one relevant evidence reference")
    if not answerable and relevant:
        raise ValueError("no-answer questions must not declare relevant evidence")
    return validated


def _ineligible_ranked_result(
    metric: str,
    k: int,
    ranked: Sequence[EvidenceRef],
) -> RankedMetricResult:
    return {
        "metric": metric,
        "k": k,
        "answerable": False,
        "eligible": False,
        "value": None,
        "retrieved_count": min(k, len(ranked)),
        "relevant_count": 0,
        "matched_count": 0,
        "correct_abstention": len(ranked) == 0,
    }


def recall_at_k(
    ranked_refs: Sequence[EvidenceRef],
    relevance: Mapping[EvidenceRef, float],
    k: int,
    *,
    answerable: bool = True,
) -> RankedMetricResult:
    """Evidence recall in the first ``k`` unique retrieved units."""

    _validate_k(k)
    ranked = _validated_ranked_refs(ranked_refs)
    grades = _validated_relevance(relevance, answerable=answerable)
    if not answerable:
        return _ineligible_ranked_result(f"recall@{k}", k, ranked)
    relevant = {ref for ref, grade in grades.items() if grade > 0}
    matched = relevant.intersection(ranked[:k])
    return {
        "metric": f"recall@{k}",
        "k": k,
        "answerable": True,
        "eligible": True,
        "value": len(matched) / len(relevant),
        "retrieved_count": min(k, len(ranked)),
        "relevant_count": len(relevant),
        "matched_count": len(matched),
        "correct_abstention": None,
    }


def mrr_at_k(
    ranked_refs: Sequence[EvidenceRef],
    relevance: Mapping[EvidenceRef, float],
    k: int,
    *,
    answerable: bool = True,
) -> RankedMetricResult:
    """Reciprocal rank of the first positively graded evidence unit."""

    _validate_k(k)
    ranked = _validated_ranked_refs(ranked_refs)
    grades = _validated_relevance(relevance, answerable=answerable)
    if not answerable:
        return _ineligible_ranked_result(f"mrr@{k}", k, ranked)
    first_rank = next(
        (rank for rank, ref in enumerate(ranked[:k], start=1) if grades.get(ref, 0.0) > 0),
        None,
    )
    relevant_count = sum(grade > 0 for grade in grades.values())
    return {
        "metric": f"mrr@{k}",
        "k": k,
        "answerable": True,
        "eligible": True,
        "value": 0.0 if first_rank is None else 1.0 / first_rank,
        "retrieved_count": min(k, len(ranked)),
        "relevant_count": relevant_count,
        "matched_count": int(first_rank is not None),
        "correct_abstention": None,
    }


def ndcg_at_k(
    ranked_refs: Sequence[EvidenceRef],
    relevance: Mapping[EvidenceRef, float],
    k: int,
    *,
    answerable: bool = True,
) -> RankedMetricResult:
    """Graded nDCG using ``(2**grade - 1) / log2(rank + 1)`` gains."""

    _validate_k(k)
    ranked = _validated_ranked_refs(ranked_refs)
    grades = _validated_relevance(relevance, answerable=answerable)
    if not answerable:
        return _ineligible_ranked_result(f"ndcg@{k}", k, ranked)

    def discounted_gain(values: Sequence[float]) -> float:
        return sum((2.0**grade - 1.0) / math.log2(rank + 1) for rank, grade in enumerate(values, 1))

    observed = [grades.get(ref, 0.0) for ref in ranked[:k]]
    relevant_grades = [grade for grade in grades.values() if grade > 0]
    ideal = sorted(relevant_grades, reverse=True)[:k]
    ideal_gain = discounted_gain(ideal)
    value = discounted_gain(observed) / ideal_gain
    return {
        "metric": f"ndcg@{k}",
        "k": k,
        "answerable": True,
        "eligible": True,
        "value": value,
        "retrieved_count": min(k, len(ranked)),
        "relevant_count": len(relevant_grades),
        "matched_count": sum(grade > 0 for grade in observed),
        "correct_abstention": None,
    }


def citation_metrics(
    citation_ranks: Sequence[int],
    ranked_evidence_refs: Sequence[Collection[EvidenceRef]],
    relevant_evidence_refs: Collection[EvidenceRef],
    *,
    answerable: bool = True,
) -> CitationMetricResult:
    """Score raw ``[n]`` citations against stable evidence covered by each rank.

    Validity is mention-level and therefore retains duplicate citations. Precision
    is rank-level over unique valid cited ranks; recall is evidence-level over the
    union of stable references covered by those ranks.
    """

    ranks = list(citation_ranks)
    if any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks):
        raise ValueError("citation ranks must be integers")
    ranked: list[set[EvidenceRef]] = []
    for refs in ranked_evidence_refs:
        if isinstance(refs, (str, bytes)):
            raise ValueError("each rank must contain a collection of evidence references")
        group = set(refs)
        if any(not isinstance(ref, str) or not ref.strip() for ref in group):
            raise ValueError("ranked evidence references must be non-empty strings")
        ranked.append(group)
    if isinstance(relevant_evidence_refs, (str, bytes)):
        raise ValueError("relevant evidence references must be a collection of strings")
    relevant = set(relevant_evidence_refs)
    if any(not isinstance(ref, str) or not ref.strip() for ref in relevant):
        raise ValueError("relevant evidence references must be non-empty strings")
    if answerable and not relevant:
        raise ValueError("answerable questions require relevant citation evidence")
    if not answerable and relevant:
        raise ValueError("no-answer questions must not declare relevant citation evidence")

    valid_mentions = [rank for rank in ranks if 1 <= rank <= len(ranked)]
    invalid_ranks = [rank for rank in ranks if not 1 <= rank <= len(ranked)]
    validity = len(valid_mentions) / len(ranks) if ranks else 1.0
    unique_valid_ranks = list(dict.fromkeys(valid_mentions))
    cited_groups = [ranked[rank - 1] for rank in unique_valid_ranks]
    relevant_cited_ranks = sum(bool(group & relevant) for group in cited_groups)
    covered = set().union(*cited_groups).intersection(relevant) if cited_groups else set()

    if not answerable:
        precision: float | None = None
        recall: float | None = None
        f1: float | None = None
        eligible = False
        correct_abstention: bool | None = not ranks
    else:
        precision = relevant_cited_ranks / len(cited_groups) if cited_groups else 0.0
        recall = len(covered) / len(relevant)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        eligible = True
        correct_abstention = None
    return {
        "answerable": answerable,
        "eligible": eligible,
        "citation_validity": validity,
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": f1,
        "citation_count": len(ranks),
        "valid_citation_count": len(valid_mentions),
        "unique_valid_rank_count": len(unique_valid_ranks),
        "relevant_cited_rank_count": relevant_cited_ranks,
        "relevant_evidence_count": len(relevant),
        "covered_evidence_count": len(covered),
        "invalid_ranks": invalid_ranks,
        "correct_abstention": correct_abstention,
    }


def normalize_number(
    value: str,
    *,
    comma_policy: CommaPolicy = "reject_ambiguous",
) -> str:
    """Normalize one signed decimal without silently accepting unsupported syntax."""

    if not isinstance(value, str):
        raise ValueError("quantity values must be strings")
    if comma_policy not in {"reject_ambiguous", "decimal", "thousands"}:
        raise ValueError(f"unsupported comma policy: {comma_policy!r}")
    normalized_input = value.translate(_FULLWIDTH_DIGITS).strip()
    if _NUMBER_LITERAL_RE.fullmatch(normalized_input) is None:
        raise ValueError(f"unsupported quantity value: {value!r}")
    negative = normalized_input[0] in _MINUS_CHARS
    unsigned = (
        normalized_input[1:].strip()
        if normalized_input[0] in _MINUS_CHARS | {"+"}
        else normalized_input
    )
    compact = unsigned.replace(" ", "").replace("\N{NO-BREAK SPACE}", "")
    if "." in compact:
        decimal_text = compact.replace(",", "")
    elif compact.count(",") > 1:
        groups = compact.split(",")
        if any(len(group) != 3 for group in groups[1:]):
            raise ValueError(f"ambiguous repeated commas in quantity value: {value!r}")
        decimal_text = "".join(groups)
    elif "," in compact:
        integer, fraction = compact.split(",", maxsplit=1)
        ambiguous = len(fraction) == 3 and bool(integer.lstrip("0"))
        if ambiguous and comma_policy == "reject_ambiguous":
            raise ValueError(
                f"ambiguous decimal/thousands comma in quantity value: {value!r}"
            )
        thousands = ambiguous and comma_policy == "thousands"
        decimal_text = integer + fraction if thousands else f"{integer}.{fraction}"
    else:
        decimal_text = compact
    try:
        decimal_value = Decimal(decimal_text)
    except InvalidOperation as exc:
        raise ValueError(f"unsupported quantity value: {value!r}") from exc
    if not decimal_value.is_finite():
        raise ValueError(f"unsupported quantity value: {value!r}")
    if decimal_value == 0:
        return "0"
    parsed_magnitudes = extract_numbers(format(decimal_value, "f"))
    if sum(parsed_magnitudes.values()) != 1:
        raise ValueError(f"unsupported quantity value: {value!r}")
    magnitude = next(iter(parsed_magnitudes))
    if magnitude.startswith("."):
        magnitude = f"0{magnitude}"
    return f"-{magnitude}" if negative else magnitude


def normalize_unit(value: str) -> str:
    """Map a deliberately bounded set of technical unit aliases to canonical names."""

    if not isinstance(value, str):
        raise ValueError("quantity units must be strings")
    if not value.strip():
        return ""
    normalized = _UNIT_ALIASES.get(_unit_key(value))
    if normalized is None:
        raise ValueError(f"unsupported quantity unit: {value!r}")
    return normalized


def extract_quantity_mentions(
    text: str,
    *,
    comma_policy: CommaPolicy = "reject_ambiguous",
) -> list[QuantityMention]:
    """Extract signed numeric mentions and an immediately following bounded unit."""

    stripped = _CITATION_RE.sub("", text)
    mentions: list[QuantityMention] = []
    for match in _NUMBER_MENTION_RE.finditer(stripped):
        raw_unit = match.group("unit") or ""
        try:
            unit = normalize_unit(raw_unit)
        except ValueError:
            unit = None
        mentions.append(
            {
                "value": normalize_number(match.group("number"), comma_policy=comma_policy),
                "unit": unit,
                "raw_unit": raw_unit,
                "unit_valid": unit is not None,
            }
        )
    return mentions


def _normalize_quantities(
    values: Sequence[Mapping[str, str]],
    *,
    comma_policy: CommaPolicy,
) -> Counter[tuple[str, str]]:
    normalized: Counter[tuple[str, str]] = Counter()
    for item in values:
        if set(item) != {"value", "unit"}:
            raise ValueError("each quantity must contain exactly value and unit")
        normalized[
            (
                normalize_number(item["value"], comma_policy=comma_policy),
                normalize_unit(item["unit"]),
            )
        ] += 1
    return normalized


def quantity_unit_metrics(
    answer: str,
    expected_quantities: Sequence[Mapping[str, str]],
    *,
    supported_quantities: Sequence[Mapping[str, str]] | None = None,
    answerable: bool = True,
    comma_policy: CommaPolicy = "reject_ambiguous",
) -> QuantityMetricResult:
    """Score quantity-unit pairs and numeric claims in an answer.

    ``quantity_unit_accuracy`` is an exact per-question score: every expected pair
    must be present and every mentioned pair must be supported. The less strict
    ``quantity_unit_recall`` reports expected multiset coverage. Unsupported-number
    rate ignores units and measures numeric values absent from the supplied evidence.
    Citation labels are removed before extraction.
    """

    expected = _normalize_quantities(expected_quantities, comma_policy=comma_policy)
    supported = _normalize_quantities(
        expected_quantities if supported_quantities is None else supported_quantities,
        comma_policy=comma_policy,
    )
    if not answerable and expected:
        raise ValueError("no-answer questions must not declare expected quantities")
    if any((expected - supported).values()):
        raise ValueError("supported quantities must include all expected quantities")

    mentions = extract_quantity_mentions(answer, comma_policy=comma_policy)
    predicted_pairs: Counter[tuple[str, str]] = Counter(
        (mention["value"], mention["unit"])
        for mention in mentions
        if mention["unit"] is not None
    )
    matched = expected & predicted_pairs
    supported_pairs = set(supported)
    mentioned_values = [mention["value"] for mention in mentions]
    supported_values = {value for value, _ in supported}
    unsupported_number_count = sum(value not in supported_values for value in mentioned_values)
    unsupported_pair_count = sum(
        mention["unit"] is None or (mention["value"], mention["unit"]) not in supported_pairs
        for mention in mentions
    )
    unsupported_number_rate = (
        unsupported_number_count / len(mentioned_values) if mentioned_values else 0.0
    )
    invalid_unit_count = sum(not mention["unit_valid"] for mention in mentions)

    if not answerable:
        eligible = False
        accuracy: float | None = None
        recall: float | None = None
        correct_abstention: bool | None = not mentions
    elif not expected:
        eligible = False
        accuracy = None
        recall = None
        correct_abstention = None
    else:
        eligible = True
        recall = sum(matched.values()) / sum(expected.values())
        accuracy = float(recall == 1.0 and unsupported_pair_count == 0)
        correct_abstention = None
    return {
        "answerable": answerable,
        "eligible": eligible,
        "quantity_unit_accuracy": accuracy,
        "quantity_unit_recall": recall,
        "unsupported_number_rate": unsupported_number_rate,
        "expected_quantity_count": sum(expected.values()),
        "matched_quantity_count": sum(matched.values()),
        "mentioned_number_count": len(mentions),
        "unsupported_number_count": unsupported_number_count,
        "invalid_unit_count": invalid_unit_count,
        "correct_abstention": correct_abstention,
    }


__all__ = [
    "CitationMetricResult",
    "EvidenceRef",
    "QuantityMention",
    "QuantityMetricResult",
    "RankedMetricResult",
    "citation_metrics",
    "extract_quantity_mentions",
    "mrr_at_k",
    "ndcg_at_k",
    "normalize_number",
    "normalize_unit",
    "quantity_unit_metrics",
    "recall_at_k",
]
