#!/usr/bin/env python3
"""Офлайн-квалификация HHEM/Lettuce на закрытом RAG Gold без текстов в отчёте.

Скрипт читает три приватных JSONL-снимка: reviewed release, исходный Gold с
reference answer и связанный generator sidecar с exact evidence. Для каждого
answerable case он строит одну положительную пару и одну детерминированную
hard-negative пару с максимально непохожим evidence другого case того же языка.
В выходные артефакты попадают только case id, язык, метка и числовой score.

Модельные библиотеки импортируются только внутри runner'ов: они не являются
runtime-зависимостью приложения. Все пути к весам обязаны быть локальными.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from rag_app.eval.citation_calibration import (  # type: ignore[import-untyped]
    CalibrationCase,
    calibrate_threshold,
)
from rag_app.eval.gold_set import (  # type: ignore[import-untyped]
    GoldRecord,
    ensure_private_gold_path,
    parse_gold_set_bytes,
)
from rag_app.eval.private_artifacts import (  # type: ignore[import-untyped]
    read_private_bytes,
    write_private_json_fresh,
)
from rag_app.eval.private_sidecar import (  # type: ignore[import-untyped]
    PrivateSidecarRecord,
    parse_private_sidecar_bytes,
)
from rag_app.rag.selective_citations import extract_claim_spans  # type: ignore[import-untyped]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
Language = Literal["ru", "en", "zh"]
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_LICENSE_LINE = re.compile(r"^license:\s*['\"]?([A-Za-z0-9_.+-]+)", re.MULTILINE)
_MAX_PRIVATE_BYTES = 256 * 1024 * 1024


class QualificationError(RuntimeError):
    """Qualification input or model failed closed."""


@dataclass(frozen=True, slots=True)
class AtomicClaim:
    text: str
    language: Language


@dataclass(frozen=True, slots=True)
class QualificationCase:
    case_id: str
    answerable: bool
    language: Language
    question: str | None = None
    answer: str | None = None
    claims: tuple[AtomicClaim, ...] = ()
    positive_context: tuple[str, ...] = ()
    negative_context: tuple[str, ...] = ()
    negative_case_id: str | None = None
    negative_overlap: float | None = None


@dataclass(frozen=True, slots=True)
class PairScores:
    positive: float
    negative: float


class CaseScorer(Protocol):
    def score_cases(
        self,
        cases: Sequence[QualificationCase],
    ) -> dict[str, tuple[PairScores, ...]]: ...


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _assert_snapshot(path: Path, revision: str, expected_license: str) -> Path:
    if not _REVISION.fullmatch(revision):
        raise QualificationError("model revision must be a full lowercase commit SHA")
    resolved = path.expanduser().resolve()
    if not resolved.is_dir() or resolved.name != revision:
        raise QualificationError("model snapshot path does not match the pinned revision")
    for filename in ("config.json", "model.safetensors", "README.md"):
        if not (resolved / filename).is_file():
            raise QualificationError("model snapshot is incomplete")
    card = (resolved / "README.md").read_text(encoding="utf-8")[:64_000]
    match = _LICENSE_LINE.search(card)
    if match is None or match.group(1).casefold() != expected_license.casefold():
        raise QualificationError("model card license does not match qualification policy")
    return resolved


def _assert_auxiliary_snapshot(path: Path, revision: str) -> Path:
    if not _REVISION.fullmatch(revision):
        raise QualificationError("auxiliary revision must be a full lowercase commit SHA")
    resolved = path.expanduser().resolve()
    if not resolved.is_dir() or resolved.name != revision or not (resolved / "config.json").is_file():
        raise QualificationError("auxiliary snapshot path does not match the pinned revision")
    return resolved


def _char_ngrams(text: str, size: int = 3) -> frozenset[str]:
    normalized = "".join(character.casefold() for character in text if character.isalnum())
    if not normalized:
        return frozenset()
    if len(normalized) < size:
        return frozenset({normalized})
    return frozenset(normalized[index : index + size] for index in range(len(normalized) - size + 1))


def _jaccard(left: str, right: str) -> float:
    left_ngrams = _char_ngrams(left)
    right_ngrams = _char_ngrams(right)
    union = left_ngrams | right_ngrams
    return len(left_ngrams & right_ngrams) / len(union) if union else 0.0


def _unique_by_case_id(values: Sequence[Any], *, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        case_id = cast(str, value.case_id)
        if case_id in result:
            raise QualificationError(f"duplicate case id in {label}")
        result[case_id] = value
    return result


def build_qualification_cases(
    release_records: Sequence[GoldRecord],
    source_records: Sequence[GoldRecord],
    sidecar_records: Sequence[PrivateSidecarRecord],
) -> tuple[QualificationCase, ...]:
    """Join private snapshots and create deterministic same-language negatives."""

    release = _unique_by_case_id(release_records, label="release")
    source = _unique_by_case_id(source_records, label="source gold")
    sidecars = _unique_by_case_id(sidecar_records, label="source sidecar")
    if not release:
        raise QualificationError("release is empty")
    if not set(release) <= set(source) or not set(release) <= set(sidecars):
        raise QualificationError("release is not fully covered by source gold and sidecar")

    positive: dict[str, QualificationCase] = {}
    for case_id in sorted(release):
        published = release[case_id]
        gold = source[case_id]
        sidecar = sidecars[case_id]
        if (
            published.language != gold.language
            or published.language != sidecar.language
            or published.answerable != gold.answerable
            or published.scope_id != gold.scope_id
            or published.scope_id != sidecar.scope_id
        ):
            raise QualificationError("private Gold linkage mismatch")
        language = cast(Language, published.language)
        if not published.answerable:
            if gold.reference_answer is not None or sidecar.exact_evidence:
                raise QualificationError("no-answer case unexpectedly contains private answer/evidence")
            positive[case_id] = QualificationCase(
                case_id=case_id,
                answerable=False,
                language=language,
            )
            continue
        if not gold.reference_answer or not sidecar.exact_evidence:
            raise QualificationError("answerable case is missing private answer/evidence")
        positive[case_id] = QualificationCase(
            case_id=case_id,
            answerable=True,
            language=language,
            question=gold.question,
            answer=gold.reference_answer,
            claims=tuple(
                AtomicClaim(span.claim, cast(Language, span.language))
                for span in extract_claim_spans(gold.reference_answer)
                if not span.non_factual
            ),
            positive_context=tuple(item.exact_quote for item in sidecar.exact_evidence),
        )
        if not positive[case_id].claims:
            raise QualificationError("answerable case did not yield factual claim spans")

    answerable_by_language: dict[Language, list[QualificationCase]] = defaultdict(list)
    for item in positive.values():
        if item.answerable:
            answerable_by_language[item.language].append(item)
    if any(len(answerable_by_language[language]) < 2 for language in ("ru", "en", "zh")):
        raise QualificationError("each language needs at least two answerable cases for hard negatives")

    output: list[QualificationCase] = []
    for case_id in sorted(positive):
        item = positive[case_id]
        if not item.answerable:
            output.append(item)
            continue
        assert item.answer is not None
        candidates = [
            candidate
            for candidate in answerable_by_language[item.language]
            if candidate.case_id != case_id
        ]
        negative = min(
            candidates,
            key=lambda candidate: (
                _jaccard(item.answer or "", "\n".join(candidate.positive_context)),
                candidate.case_id,
            ),
        )
        overlap = _jaccard(item.answer, "\n".join(negative.positive_context))
        output.append(
            QualificationCase(
                case_id=item.case_id,
                answerable=True,
                language=item.language,
                question=item.question,
                answer=item.answer,
                claims=item.claims,
                positive_context=item.positive_context,
                negative_context=negative.positive_context,
                negative_case_id=negative.case_id,
                negative_overlap=overlap,
            )
        )
    return tuple(output)


def support_score_from_token_predictions(predictions: Sequence[dict[str, Any]]) -> float:
    """Conservative claim support = minimum token support = 1 - max P(hallucination)."""

    if not predictions:
        raise QualificationError("Lettuce returned an empty token sequence")
    hallucination: list[float] = []
    for item in predictions:
        raw = item.get("prob")
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise QualificationError("Lettuce token probability is not numeric")
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise QualificationError("Lettuce token probability is outside [0, 1]")
        hallucination.append(value)
    return 1.0 - max(hallucination)


def hhem_batch_plan(lengths: Sequence[int], max_batch_size: int) -> tuple[tuple[int, ...], ...]:
    """Length-bucketed HHEM batches; long quadratic-attention inputs run alone."""

    if not lengths or max_batch_size < 1 or any(length < 1 for length in lengths):
        raise QualificationError("invalid HHEM batch plan inputs")
    ordered = sorted(range(len(lengths)), key=lambda index: (-lengths[index], index))
    batches: list[tuple[int, ...]] = []
    cursor = 0
    while cursor < len(ordered):
        longest = lengths[ordered[cursor]]
        adaptive_limit = 1 if longest > 2048 else 2 if longest > 1024 else 4 if longest > 512 else 8
        size = min(max_batch_size, adaptive_limit, len(ordered) - cursor)
        batches.append(tuple(ordered[cursor : cursor + size]))
        cursor += size
    return tuple(batches)


class HhemScorer:
    def __init__(
        self,
        model_path: Path,
        tokenizer_path: Path,
        *,
        batch_size: int,
        device: str,
    ) -> None:
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoConfig,
                AutoModelForSequenceClassification,
            )
        except ImportError as error:
            raise QualificationError("HHEM qualification dependencies are unavailable") from error

        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        config.foundation = str(tokenizer_path)
        self._model: Any = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
        ).to(device)
        self._model.eval()
        self._torch = torch
        self._batch_size = batch_size

    def _score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        tokenizer = self._model.tokenzier
        prompt = self._model.prompt
        lengths = [
            len(
                tokenizer(
                    prompt.format(text1=premise, text2=hypothesis),
                    add_special_tokens=True,
                )["input_ids"]
            )
            for premise, hypothesis in pairs
        ]
        batches = hhem_batch_plan(lengths, self._batch_size)
        scores: list[float | None] = [None] * len(pairs)
        with self._torch.inference_mode():
            for batch_number, indexes in enumerate(batches, 1):
                batch = [pairs[index] for index in indexes]
                raw = self._model.predict(batch).detach().cpu().tolist()
                if len(raw) != len(indexes):
                    raise QualificationError("HHEM returned an invalid batch size")
                for index, value in zip(indexes, raw, strict=True):
                    scores[index] = float(value)
                print(
                    json.dumps(
                        {
                            "stage": "hhem",
                            "batch": batch_number,
                            "batches": len(batches),
                            "max_tokens": max(lengths[index] for index in indexes),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if any(
            value is None
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for value in scores
        ):
            raise QualificationError("HHEM returned invalid scores")
        return [cast(float, value) for value in scores]

    def score_cases(
        self,
        cases: Sequence[QualificationCase],
    ) -> dict[str, tuple[PairScores, ...]]:
        answerable = [item for item in cases if item.answerable]
        pairs: list[tuple[str, str]] = []
        for item in answerable:
            for claim in item.claims:
                pairs.append(("\n".join(item.positive_context), claim.text))
                pairs.append(("\n".join(item.negative_context), claim.text))
        values = self._score(pairs)
        result: dict[str, tuple[PairScores, ...]] = {}
        cursor = 0
        for item in answerable:
            case_scores: list[PairScores] = []
            for _claim in item.claims:
                case_scores.append(PairScores(values[cursor], values[cursor + 1]))
                cursor += 2
            result[item.case_id] = tuple(case_scores)
        if cursor != len(values):
            raise QualificationError("HHEM score cursor did not consume every pair")
        return result


class LettuceRouterScorer:
    def __init__(
        self,
        en_model_path: Path,
        zh_model_path: Path,
        *,
        eurobert_code_revision: str,
        device: str,
        max_length: int,
    ) -> None:
        try:
            from lettucedetect.models.inference import (  # type: ignore[import-not-found]
                HallucinationDetector,
            )
        except ImportError as error:
            raise QualificationError("Lettuce qualification dependency is unavailable") from error

        self._en = HallucinationDetector(
            method="transformer",
            model_path=str(en_model_path),
            lang="en",
            device=device,
            max_length=max_length,
            local_files_only=True,
        )
        self._zh = HallucinationDetector(
            method="transformer",
            model_path=str(zh_model_path),
            lang="cn",
            device=device,
            max_length=max_length,
            trust_remote_code=True,
            code_revision=eurobert_code_revision,
            local_files_only=True,
        )

    def _score(
        self,
        claim: AtomicClaim,
        context: tuple[str, ...],
    ) -> float:
        detector = self._en if claim.language == "en" else self._zh
        predictions = detector.predict(
            context=list(context),
            question=None,
            answer=claim.text,
            output_format="tokens",
        )
        return support_score_from_token_predictions(predictions)

    def score_cases(
        self,
        cases: Sequence[QualificationCase],
    ) -> dict[str, tuple[PairScores, ...]]:
        return {
            item.case_id: tuple(
                PairScores(
                    self._score(claim, item.positive_context),
                    self._score(claim, item.negative_context),
                )
                for claim in item.claims
            )
            for item in cases
            if item.answerable
        }


def build_observation_payload(
    cases: Sequence[QualificationCase],
    scores: dict[str, tuple[PairScores, ...]],
) -> dict[str, Any]:
    expected = {item.case_id for item in cases if item.answerable}
    if set(scores) != expected:
        raise QualificationError("scorer did not cover every answerable case exactly once")
    observations: list[dict[str, Any]] = []
    for item in cases:
        claims: list[dict[str, Any]] = []
        if item.answerable:
            pairs = scores[item.case_id]
            if len(pairs) != len(item.claims):
                raise QualificationError("scorer claim count differs from extracted claim count")
            claims = [
                claim
                for pair in pairs
                for claim in (
                    {"score": pair.positive, "supported": True},
                    {"score": pair.negative, "supported": False},
                )
            ]
        observations.append(
            {
                "case_id": item.case_id,
                "answerable": item.answerable,
                "language": item.language,
                "claims": claims,
            }
        )
    return {
        "schema_version": "citation-calibration-observations-v1",
        "case_count": len(observations),
        "cases": observations,
    }


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise QualificationError("cannot summarize an empty score set")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Tie-aware Mann-Whitney ROC AUC without a qualification-only dependency."""

    if len(labels) != len(scores) or not labels:
        raise QualificationError("ROC AUC labels/scores are inconsistent")
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise QualificationError("ROC AUC requires both classes")
    ranked = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ranked[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def summarize_scores(
    cases: Sequence[QualificationCase],
    scores: dict[str, tuple[PairScores, ...]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for language in ("ru", "en", "zh"):
        selected_cases = [
            item for item in cases if item.answerable and item.language == language
        ]
        pairs = [pair for item in selected_cases for pair in scores[item.case_id]]
        positive = [item.positive for item in pairs]
        negative = [item.negative for item in pairs]
        labels = [True] * len(positive) + [False] * len(negative)
        values = positive + negative
        overlaps = [
            cast(float, item.negative_overlap)
            for item in cases
            if item.answerable and item.language == language
        ]
        result[language] = {
            "answerable_cases": len(selected_cases),
            "factual_claims": len(pairs),
            "pair_count": len(values),
            "roc_auc": roc_auc(labels, values),
            "positive": {
                "mean": statistics.fmean(positive),
                "p05": _quantile(positive, 0.05),
                "p50": _quantile(positive, 0.50),
                "p95": _quantile(positive, 0.95),
            },
            "negative": {
                "mean": statistics.fmean(negative),
                "p05": _quantile(negative, 0.05),
                "p50": _quantile(negative, 0.50),
                "p95": _quantile(negative, 0.95),
            },
            "negative_ngram_overlap": {
                "mean": statistics.fmean(overlaps),
                "max": max(overlaps),
            },
        }
    return result


def _calibrate(
    payload: dict[str, Any],
    *,
    answerability_target: float,
    semantic_precision_target: float,
) -> dict[str, Any]:
    cases = [CalibrationCase.model_validate(item) for item in payload["cases"]]
    result = calibrate_threshold(
        cases,
        [index / 100 for index in range(101)],
        answerability_target=answerability_target,
        semantic_precision_target=semantic_precision_target,
    )
    return result.model_dump(mode="json")


def _runtime_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("torch", "transformers", "lettucedetect", "huggingface-hub"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "missing"
    return result


def _git_sha() -> str | None:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return value if _REVISION.fullmatch(value) else None


def _score_backend(
    name: str,
    scorer_factory: Callable[[], CaseScorer],
    cases: Sequence[QualificationCase],
    output_dir: Path,
    *,
    answerability_target: float,
    semantic_precision_target: float,
) -> dict[str, Any]:
    started = time.monotonic()
    scorer = scorer_factory()
    load_s = time.monotonic() - started
    inference_started = time.monotonic()
    scores = scorer.score_cases(cases)
    inference_s = time.monotonic() - inference_started
    payload = build_observation_payload(cases, scores)
    observation_path = output_dir / f"{name}-observations.json"
    write_private_json_fresh(
        observation_path,
        (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode(),
    )
    return {
        "observation_file": observation_path.name,
        "observation_sha256": _file_sha256(observation_path),
        "load_s": load_s,
        "inference_s": inference_s,
        "pairs_per_s": (2 * sum(len(values) for values in scores.values())) / inference_s
        if inference_s
        else None,
        "scores": summarize_scores(cases, scores),
        "calibration": _calibrate(
            payload,
            answerability_target=answerability_target,
            semantic_precision_target=semantic_precision_target,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_gold", type=Path)
    parser.add_argument("source_gold", type=Path)
    parser.add_argument("source_sidecar", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-cases", type=int, default=236)
    parser.add_argument("--answerability-target", type=float, default=0.85)
    parser.add_argument("--semantic-precision-target", type=float, default=0.90)
    parser.add_argument("--hhem-model", type=Path, required=True)
    parser.add_argument("--hhem-revision", required=True)
    parser.add_argument("--hhem-tokenizer", type=Path, required=True)
    parser.add_argument("--hhem-tokenizer-revision", required=True)
    parser.add_argument("--hhem-batch-size", type=int, default=8)
    parser.add_argument("--hhem-device", default="cpu")
    parser.add_argument("--lettuce-en-model", type=Path, required=True)
    parser.add_argument("--lettuce-en-revision", required=True)
    parser.add_argument("--lettuce-zh-model", type=Path, required=True)
    parser.add_argument("--lettuce-zh-revision", required=True)
    parser.add_argument("--eurobert-code", type=Path, required=True)
    parser.add_argument("--eurobert-code-revision", required=True)
    parser.add_argument("--lettuce-device", default="cpu")
    parser.add_argument("--lettuce-max-length", type=int, default=4096)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.expected_cases < 1:
        raise SystemExit("--expected-cases must be positive")
    if not 1 <= args.hhem_batch_size <= 64:
        raise SystemExit("--hhem-batch-size must be in [1, 64]")
    if not 128 <= args.lettuce_max_length <= 8192:
        raise SystemExit("--lettuce-max-length must be in [128, 8192]")
    if not 1 <= args.torch_threads <= 64:
        raise SystemExit("--torch-threads must be in [1, 64]")

    release_path = ensure_private_gold_path(args.release_gold, REPOSITORY_ROOT)
    source_path = ensure_private_gold_path(args.source_gold, REPOSITORY_ROOT)
    sidecar_path = ensure_private_gold_path(args.source_sidecar, REPOSITORY_ROOT)
    output_dir = ensure_private_gold_path(args.output_dir, REPOSITORY_ROOT)
    if not output_dir.is_dir():
        raise QualificationError("private output directory must already exist")

    os.environ["HF_HOME"] = str(args.hf_home.expanduser().resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    hhem_model = _assert_snapshot(args.hhem_model, args.hhem_revision, "apache-2.0")
    hhem_tokenizer = _assert_auxiliary_snapshot(
        args.hhem_tokenizer,
        args.hhem_tokenizer_revision,
    )
    lettuce_en = _assert_snapshot(args.lettuce_en_model, args.lettuce_en_revision, "mit")
    lettuce_zh = _assert_snapshot(args.lettuce_zh_model, args.lettuce_zh_revision, "mit")
    _assert_auxiliary_snapshot(args.eurobert_code, args.eurobert_code_revision)

    try:
        import torch  # type: ignore[import-not-found]

        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)
    except ImportError as error:
        raise QualificationError("PyTorch is unavailable") from error

    release_artifact = read_private_bytes(release_path, max_bytes=_MAX_PRIVATE_BYTES)
    source_artifact = read_private_bytes(source_path, max_bytes=_MAX_PRIVATE_BYTES)
    sidecar_artifact = read_private_bytes(sidecar_path, max_bytes=_MAX_PRIVATE_BYTES)
    release_records, _ = parse_gold_set_bytes(release_artifact.raw_bytes, mode="release")
    source_records, _ = parse_gold_set_bytes(source_artifact.raw_bytes, mode="candidate")
    sidecar_records = parse_private_sidecar_bytes(sidecar_artifact.raw_bytes)
    cases = build_qualification_cases(release_records, source_records, sidecar_records)
    if len(cases) != args.expected_cases:
        raise QualificationError("reviewed release case count differs from expected")

    summary: dict[str, Any] = {
        "schema_version": "citation-model-qualification-v1",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "case_count": len(cases),
        "answerable_count": sum(item.answerable for item in cases),
        "factual_claim_count": sum(len(item.claims) for item in cases),
        "languages": {
            language: sum(item.language == language for item in cases)
            for language in ("ru", "en", "zh")
        },
        "synthetic_negative_method": "minimum_same_language_character_trigram_jaccard_v1",
        "artifacts": {
            "release_sha256": release_artifact.sha256,
            "source_gold_sha256": source_artifact.sha256,
            "source_sidecar_sha256": sidecar_artifact.sha256,
        },
        "runtime": _runtime_versions(),
        "models": {
            "hhem": {
                "revision": args.hhem_revision,
                "license": "apache-2.0",
                "model_sha256": _file_sha256(hhem_model / "model.safetensors"),
                "tokenizer_revision": args.hhem_tokenizer_revision,
                "declared_languages": ["en"],
            },
            "lettuce_en": {
                "revision": args.lettuce_en_revision,
                "license": "mit",
                "model_sha256": _file_sha256(lettuce_en / "model.safetensors"),
                "declared_languages": ["en"],
            },
            "lettuce_zh": {
                "revision": args.lettuce_zh_revision,
                "license": "mit",
                "model_sha256": _file_sha256(lettuce_zh / "model.safetensors"),
                "code_revision": args.eurobert_code_revision,
                "declared_languages": ["zh"],
                "ru_mode": "exploratory_zero_shot_not_declared_by_model_card",
            },
        },
        "backends": {},
    }
    summary["backends"]["hhem"] = _score_backend(
        "hhem",
        lambda: HhemScorer(
            hhem_model,
            hhem_tokenizer,
            batch_size=args.hhem_batch_size,
            device=args.hhem_device,
        ),
        cases,
        output_dir,
        answerability_target=args.answerability_target,
        semantic_precision_target=args.semantic_precision_target,
    )
    summary["backends"]["lettuce_router"] = _score_backend(
        "lettuce-router",
        lambda: LettuceRouterScorer(
            lettuce_en,
            lettuce_zh,
            eurobert_code_revision=args.eurobert_code_revision,
            device=args.lettuce_device,
            max_length=args.lettuce_max_length,
        ),
        cases,
        output_dir,
        answerability_target=args.answerability_target,
        semantic_precision_target=args.semantic_precision_target,
    )
    summary["decision"] = {
        "hhem_local_gate": "GO" if summary["backends"]["hhem"]["calibration"]["qualified"] else "NO-GO",
        "lettuce_router_local_gate": (
            "GO"
            if summary["backends"]["lettuce_router"]["calibration"]["qualified"]
            else "NO-GO"
        ),
        "production_gate": "NO-GO",
        "production_blockers": [
            "offline corpus uses reviewed Gold positives plus synthetic mismatched-evidence negatives",
            "no released RU verifier model with a declared RU capability was qualified",
            "full generated-answer shadow baseline remains required before production",
        ],
    }
    summary_path = output_dir / "citation-model-qualification.json"
    write_private_json_fresh(
        summary_path,
        (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "hhem": summary["decision"]["hhem_local_gate"],
                "lettuce_router": summary["decision"]["lettuce_router_local_gate"],
                "production": "NO-GO",
                "report_sha256": _file_sha256(summary_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
