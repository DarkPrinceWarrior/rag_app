"""Калибровка selective citation threshold по risk–coverage curve."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CalibrationClaim(_StrictModel):
    score: float = Field(ge=0, le=1)
    supported: bool


class CalibrationCase(_StrictModel):
    case_id: str = Field(min_length=1)
    answerable: bool
    language: str = Field(pattern=r"^(ru|en|zh)$")
    claims: tuple[CalibrationClaim, ...] = ()


class RiskCoveragePoint(_StrictModel):
    threshold: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    risk: float = Field(ge=0, le=1)
    semantic_precision: float = Field(ge=0, le=1)
    answerability_accuracy: float = Field(ge=0, le=1)
    retained_claims: int = Field(ge=0)


class CalibrationResult(_StrictModel):
    case_count: int = Field(ge=1)
    answerability_target: float = Field(ge=0, le=1)
    semantic_precision_target: float = Field(ge=0, le=1)
    selected_threshold: float | None = Field(default=None, ge=0, le=1)
    qualified: bool
    curve: tuple[RiskCoveragePoint, ...]
    language_curves: dict[str, tuple[RiskCoveragePoint, ...]]


def risk_coverage_curve(
    cases: Sequence[CalibrationCase],
    thresholds: Sequence[float],
) -> tuple[RiskCoveragePoint, ...]:
    if not cases:
        raise ValueError("citation calibration requires at least one case")
    if not thresholds:
        raise ValueError("citation calibration requires thresholds")
    total_claims = sum(len(case.claims) for case in cases)
    points: list[RiskCoveragePoint] = []
    for threshold in sorted(set(thresholds)):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("citation threshold must be in [0, 1]")
        retained = [
            claim
            for case in cases
            for claim in case.claims
            if claim.score >= threshold
        ]
        correct = sum(claim.supported for claim in retained)
        precision = correct / len(retained) if retained else 1.0
        answerability_correct = 0
        for case in cases:
            answered = any(claim.score >= threshold for claim in case.claims)
            answerability_correct += answered if case.answerable else not answered
        points.append(
            RiskCoveragePoint(
                threshold=threshold,
                coverage=len(retained) / total_claims if total_claims else 0.0,
                risk=1.0 - precision,
                semantic_precision=precision,
                answerability_accuracy=answerability_correct / len(cases),
                retained_claims=len(retained),
            )
        )
    return tuple(points)


def calibrate_threshold(
    cases: Sequence[CalibrationCase],
    thresholds: Sequence[float],
    *,
    answerability_target: float = 0.85,
    semantic_precision_target: float = 0.90,
) -> CalibrationResult:
    curve = risk_coverage_curve(cases, thresholds)
    language_curves = {
        language: risk_coverage_curve(
            [case for case in cases if case.language == language],
            thresholds,
        )
        for language in sorted({case.language for case in cases})
    }
    eligible = [
        point
        for index, point in enumerate(curve)
        if point.answerability_accuracy >= answerability_target
        and point.semantic_precision >= semantic_precision_target
        and all(
            language_curve[index].answerability_accuracy >= answerability_target
            and language_curve[index].semantic_precision >= semantic_precision_target
            for language_curve in language_curves.values()
        )
    ]
    selected = max(
        eligible,
        key=lambda point: (point.coverage, point.semantic_precision, -point.threshold),
        default=None,
    )
    return CalibrationResult(
        case_count=len(cases),
        answerability_target=answerability_target,
        semantic_precision_target=semantic_precision_target,
        selected_threshold=selected.threshold if selected is not None else None,
        qualified=selected is not None,
        curve=curve,
        language_curves=language_curves,
    )
