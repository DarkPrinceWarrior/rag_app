"""Compare two paired parser benchmark summaries and emit a fail-closed decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import statistics
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

_SHA256_LENGTH = 64
_MIN_PAIRED_PAGES = 20
_QUALITY_REGRESSION_TOLERANCE = 0.01
_STRUCTURE_REGRESSION_TOLERANCE = 0.01
_BACKFILL_REGRESSION_TOLERANCE = 0.02
_LATENCY_REGRESSION_RATIO = 1.10
_PRACTICAL_QUALITY_GAIN = 0.01
_PRACTICAL_STRUCTURE_GAIN = 0.02
_PRACTICAL_SPEEDUP = 1.05
_SIGN_TEST_ALPHA = 0.05
_DUPLICATE_CHARACTER_REGRESSION_TOLERANCE = 0.001
_PRACTICAL_DUPLICATE_CHARACTER_GAIN = 0.002
_BENCHMARK_SCHEMA_VERSION = 2
_KNOWN_CATEGORIES = {"table", "chart", "layout", "text_formatting"}

DecisionStatus = Literal["accepted", "rejected", "inconclusive"]


class ParserComparisonError(ValueError):
    """The supplied benchmark artifacts are invalid or incomparable."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_runtime_evidence(runtime_pair: Mapping[str, Mapping[str, Any]]) -> None:
    for artifact, runtime in runtime_pair.items():
        evidence = runtime["evidence"]
        assert isinstance(evidence, Mapping)
        path = Path(evidence["path"])
        if not path.is_absolute():
            raise ParserComparisonError(f"{artifact}: runtime evidence path must be absolute")
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise ParserComparisonError(f"{artifact}: runtime evidence file is unavailable") from exc
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ParserComparisonError(f"{artifact}: runtime evidence must be a regular non-symlink file")
        try:
            actual_sha256 = _file_sha256(path)
        except OSError as exc:
            raise ParserComparisonError(f"{artifact}: runtime evidence file cannot be read") from exc
        if actual_sha256 != evidence["sha256"]:
            raise ParserComparisonError(f"{artifact}: runtime evidence SHA256 mismatch")


def _number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ParserComparisonError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ParserComparisonError(f"{field} must be a finite number")
    return result


def _optional_number(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    return _number(value, field=field)


def _unit_interval(value: float | None, *, field: str) -> float | None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ParserComparisonError(f"{field} must be between 0 and 1")
    return value


def _nested(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _first_number(value: Mapping[str, Any], paths: Sequence[tuple[str, ...]], *, field: str) -> float | None:
    for path in paths:
        candidate = _nested(value, *path)
        if candidate is not None:
            return _optional_number(candidate, field=field)
    return None


def _first_string(value: Mapping[str, Any], paths: Sequence[tuple[str, ...]]) -> str | None:
    for path in paths:
        candidate = _nested(value, *path)
        if isinstance(candidate, str):
            return candidate
    return None


def _required_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParserComparisonError(f"{field} must be a non-empty string")
    return value.strip()


def _sha256_string(value: Any, *, field: str) -> str:
    result = _required_string(value, field=field)
    if len(result) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in result):
        raise ParserComparisonError(f"{field} must be a lowercase SHA256 digest")
    return result


def _model_snapshot_sha(value: Any, *, field: str) -> str:
    result = _required_string(value, field=field)
    if len(result) not in {40, _SHA256_LENGTH} or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ParserComparisonError(f"{field} must be a lowercase 40- or 64-character SHA")
    return result


def _runtime_provenance(summary: Mapping[str, Any], *, artifact: str) -> dict[str, Any]:
    if summary.get("benchmark_schema_version") != _BENCHMARK_SCHEMA_VERSION:
        raise ParserComparisonError(
            f"{artifact}: benchmark_schema_version must be {_BENCHMARK_SCHEMA_VERSION}"
        )
    run_label = _required_string(summary.get("run_label"), field=f"{artifact}.run_label")
    provenance = summary.get("runtime_provenance")
    if not isinstance(provenance, Mapping):
        raise ParserComparisonError(f"{artifact}.runtime_provenance must be an object")

    client_version = _required_string(
        _nested(provenance, "client", "version"),
        field=f"{artifact}.runtime_provenance.client.version",
    )
    started_at = _required_string(
        _nested(provenance, "server", "started_at"),
        field=f"{artifact}.runtime_provenance.server.started_at",
    )
    try:
        parsed_started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ParserComparisonError(
            f"{artifact}.runtime_provenance.server.started_at must be ISO 8601"
        ) from exc
    if parsed_started_at.tzinfo is None:
        raise ParserComparisonError(
            f"{artifact}.runtime_provenance.server.started_at must include a timezone"
        )

    server = {
        "started_at": started_at,
        "endpoint": _required_string(
            _nested(provenance, "server", "endpoint"),
            field=f"{artifact}.runtime_provenance.server.endpoint",
        ),
        "vllm_version": _required_string(
            _nested(provenance, "server", "vllm_version"),
            field=f"{artifact}.runtime_provenance.server.vllm_version",
        ),
        "mineru_version": _required_string(
            _nested(provenance, "server", "mineru_version"),
            field=f"{artifact}.runtime_provenance.server.mineru_version",
        ),
    }
    model = {
        "snapshot_sha": _model_snapshot_sha(
            _nested(provenance, "model", "snapshot_sha"),
            field=f"{artifact}.runtime_provenance.model.snapshot_sha",
        ),
        "manifest_sha256": _sha256_string(
            _nested(provenance, "model", "manifest_sha256"),
            field=f"{artifact}.runtime_provenance.model.manifest_sha256",
        ),
    }
    evidence = {
        "path": _required_string(
            _nested(provenance, "evidence", "path"),
            field=f"{artifact}.runtime_provenance.evidence.path",
        ),
        "sha256": _sha256_string(
            _nested(provenance, "evidence", "sha256"),
            field=f"{artifact}.runtime_provenance.evidence.sha256",
        ),
    }
    controlled_value = provenance.get("controlled")
    if not isinstance(controlled_value, Mapping):
        raise ParserComparisonError(f"{artifact}.runtime_provenance.controlled must be an object")
    table_enable = controlled_value.get("table_enable")
    if not isinstance(table_enable, bool):
        raise ParserComparisonError(
            f"{artifact}.runtime_provenance.controlled.table_enable must be boolean"
        )
    server_inference_args = controlled_value.get("server_inference_args")
    if not isinstance(server_inference_args, list) or not server_inference_args or not all(
        isinstance(item, str) and item for item in server_inference_args
    ):
        raise ParserComparisonError(
            f"{artifact}.runtime_provenance.controlled.server_inference_args must be a non-empty string list"
        )
    repetition_penalty = _number(
        controlled_value.get("repetition_penalty"),
        field=f"{artifact}.runtime_provenance.controlled.repetition_penalty",
    )
    if repetition_penalty <= 0:
        raise ParserComparisonError(
            f"{artifact}.runtime_provenance.controlled.repetition_penalty must be positive"
        )
    controlled = {
        "parser_backend": _required_string(
            controlled_value.get("parser_backend"),
            field=f"{artifact}.runtime_provenance.controlled.parser_backend",
        ),
        "table_enable": table_enable,
        "server_inference_args": list(server_inference_args),
        "repetition_penalty": repetition_penalty,
        "sampling_patch_sha256": _sha256_string(
            controlled_value.get("sampling_patch_sha256"),
            field=f"{artifact}.runtime_provenance.controlled.sampling_patch_sha256",
        ),
    }
    return {
        "run_label": run_label,
        "client": {"version": client_version},
        "server": server,
        "model": model,
        "evidence": evidence,
        "controlled": controlled,
    }


def _validate_runtime_pair(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    left = _runtime_provenance(baseline, artifact="baseline")
    right = _runtime_provenance(candidate, artifact="candidate")
    if left["run_label"] == right["run_label"]:
        raise ParserComparisonError("baseline and candidate run labels must differ")
    if left["client"]["version"] == right["client"]["version"]:
        raise ParserComparisonError("baseline and candidate client versions must differ")
    if left["server"]["mineru_version"] == right["server"]["mineru_version"]:
        raise ParserComparisonError("baseline and candidate server MinerU versions must differ")
    if (
        left["server"]["started_at"] == right["server"]["started_at"]
        and left["server"]["endpoint"] == right["server"]["endpoint"]
    ):
        raise ParserComparisonError("baseline and candidate server runtime identity must differ")
    if left["evidence"]["sha256"] == right["evidence"]["sha256"]:
        raise ParserComparisonError("baseline and candidate runtime evidence SHA256 must differ")
    for artifact, runtime in (("baseline", left), ("candidate", right)):
        if runtime["client"]["version"] != runtime["server"]["mineru_version"]:
            raise ParserComparisonError(f"{artifact}: client and server MinerU versions must match")
    for path, label in (
        (("server", "vllm_version"), "vLLM version"),
        (("model", "snapshot_sha"), "model snapshot SHA"),
        (("model", "manifest_sha256"), "model manifest SHA256"),
    ):
        if _nested(left, *path) != _nested(right, *path):
            raise ParserComparisonError(f"baseline and candidate {label} differ")
    if left["controlled"] != right["controlled"]:
        raise ParserComparisonError("baseline and candidate controlled runtime settings differ")
    if _canonical_sha256(left) == _canonical_sha256(right):
        raise ParserComparisonError("baseline and candidate runtime provenance is identical")
    return {"baseline": left, "candidate": right}


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _distribution(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": round(statistics.mean(values), 6),
        "median": round(statistics.median(values), 6),
        "p95": round(_nearest_rank(values, 0.95), 6),
    }


def _paired_metric(baseline: Sequence[float], candidate: Sequence[float]) -> dict[str, Any]:
    if len(baseline) != len(candidate):
        raise AssertionError("paired metric lengths differ")
    if not baseline:
        return {"paired_count": 0, "baseline_mean": None, "candidate_mean": None, "delta": None}
    baseline_mean = statistics.mean(baseline)
    candidate_mean = statistics.mean(candidate)
    return {
        "paired_count": len(baseline),
        "baseline_mean": round(baseline_mean, 6),
        "candidate_mean": round(candidate_mean, 6),
        "delta": round(candidate_mean - baseline_mean, 6),
    }


def two_sided_sign_test(baseline: Sequence[float], candidate: Sequence[float]) -> dict[str, Any]:
    """Exact two-sided paired sign test; ties are excluded."""

    if len(baseline) != len(candidate):
        raise ParserComparisonError("sign test inputs must be paired")
    faster = sum(right < left for left, right in zip(baseline, candidate, strict=True))
    slower = sum(right > left for left, right in zip(baseline, candidate, strict=True))
    ties = len(baseline) - faster - slower
    observations = faster + slower
    if observations == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(observations, index) for index in range(min(faster, slower) + 1))
        p_value = min(1.0, 2.0 * tail / (2**observations))
    return {
        "candidate_faster": faster,
        "candidate_slower": slower,
        "ties": ties,
        "non_tied_count": observations,
        "p_value": round(p_value, 12),
    }


def _validated_pages(
    summary: Mapping[str, Any],
    *,
    backend: str,
    artifact: str,
) -> dict[str, Mapping[str, Any]]:
    backends = summary.get("backends")
    if not isinstance(backends, list) or backend not in backends:
        raise ParserComparisonError(f"{artifact}: backend {backend!r} is not declared")
    results = summary.get("results")
    if not isinstance(results, dict) or not results:
        raise ParserComparisonError(f"{artifact}: results must be a non-empty object")
    pages: dict[str, Mapping[str, Any]] = {}
    for filename, page in results.items():
        if not isinstance(filename, str) or not filename or not isinstance(page, Mapping):
            raise ParserComparisonError(f"{artifact}: invalid result entry")
        source_sha256 = page.get("source_sha256")
        if (
            not isinstance(source_sha256, str)
            or len(source_sha256) != _SHA256_LENGTH
            or any(char not in "0123456789abcdef" for char in source_sha256)
        ):
            raise ParserComparisonError(f"{artifact}: invalid source SHA for {filename}")
        category = page.get("category")
        if category not in _KNOWN_CATEGORIES:
            raise ParserComparisonError(f"{artifact}: invalid category for {filename}")
        if not isinstance(page.get("selection"), Mapping):
            raise ParserComparisonError(f"{artifact}: selection must be an object for {filename}")
        result = page.get(backend)
        if not isinstance(result, Mapping) or result.get("status") not in {"ok", "error"}:
            raise ParserComparisonError(f"{artifact}: invalid {backend} result for {filename}")
        pages[filename] = page
    return pages


def _validate_corpus(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline_pages: Mapping[str, Mapping[str, Any]],
    candidate_pages: Mapping[str, Mapping[str, Any]],
) -> None:
    if baseline.get("source") != candidate.get("source"):
        raise ParserComparisonError("benchmark source differs")
    if baseline.get("source_revision") != candidate.get("source_revision"):
        raise ParserComparisonError("benchmark source revision differs")
    if set(baseline_pages) != set(candidate_pages):
        raise ParserComparisonError("benchmark page sets differ")
    for filename in sorted(baseline_pages):
        left = baseline_pages[filename]
        right = candidate_pages[filename]
        if left.get("source_sha256") != right.get("source_sha256"):
            raise ParserComparisonError(f"source SHA differs for {filename}")
        for field in ("category", "selection"):
            if _canonical_sha256(left.get(field)) != _canonical_sha256(right.get(field)):
                raise ParserComparisonError(f"corpus {field} differs for {filename}")


def _has_positive_annotation(page: Mapping[str, Any], field: str, *, filename: str) -> bool:
    selection = page["selection"]
    assert isinstance(selection, Mapping)
    value = selection.get(field)
    if value is None:
        return False
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ParserComparisonError(f"selection.{field} must be a positive integer for {filename}")
    return True


def _quality_values(result: Mapping[str, Any]) -> tuple[float | None, float | None, float | None]:
    final = _unit_interval(
        _first_number(result, (("quality", "score"),), field="quality.score"),
        field="quality.score",
    )
    raw = _unit_interval(
        _first_number(
            result,
            (("quality", "raw_parser", "score"),),
            field="raw quality score",
        ),
        field="raw quality score",
    )
    backfill = _unit_interval(
        _first_number(
            result,
            (("quality", "backfilled_page_ratio"),),
            field="backfilled_page_ratio",
        ),
        field="backfilled_page_ratio",
    )
    return final, raw, backfill


def _structure_value(result: Mapping[str, Any], name: str, *, raw: bool = False) -> float | None:
    final_paths = {
        "bbox_valid_ratio": (
            ("benchmark", "bbox_valid_ratio"),
            ("structure", "bbox_valid_ratio"),
            ("bbox_valid_ratio",),
        ),
        "reading_order_score": (
            ("benchmark", "reading_order_score"),
            ("structure", "reading_order_score"),
            ("reading_order_score",),
        ),
    }
    paths = (("raw_stats", name),) if raw else final_paths[name]
    return _unit_interval(
        _first_number(result, paths, field=f"{'raw_' if raw else ''}{name}"),
        field=f"{'raw_' if raw else ''}{name}",
    )


def _structure_has_evidence(
    result: Mapping[str, Any], name: str, *, raw: bool = False
) -> bool:
    if name == "bbox_valid_ratio":
        bbox_paths = (("raw_stats", "bbox_segments"),) if raw else (("bbox_segments",),)
        count = _first_number(
            result, bbox_paths, field=f"{'raw_' if raw else ''}bbox_segments"
        )
        return count is not None and count > 0
    if raw:
        return _nested(result, "raw_stats", "reading_order_evidence") is True
    for path in (
        ("benchmark", "reading_order_evidence"),
        ("structure", "reading_order_evidence"),
        ("reading_order_evidence",),
    ):
        value = _nested(result, *path)
        if value is not None:
            return value is True
    return False


def _table_values(result: Mapping[str, Any]) -> tuple[float | None, float | None, float | None]:
    detected = _nested(result, "benchmark", "table_detected")
    detected_value = float(detected) if isinstance(detected, bool) else None
    cell_ratio = _first_number(
        result,
        (("benchmark", "table_cell_count_ratio"),),
        field="table_cell_count_ratio",
    )
    row_ratio = _first_number(
        result,
        (("benchmark", "table_row_count_ratio"),),
        field="table_row_count_ratio",
    )
    if cell_ratio is not None and cell_ratio < 0:
        raise ParserComparisonError("table_cell_count_ratio must be non-negative")
    if row_ratio is not None and row_ratio < 0:
        raise ParserComparisonError("table_row_count_ratio must be non-negative")

    def fidelity(ratio: float | None) -> float | None:
        if ratio is None or ratio == 0:
            return ratio
        return min(ratio, 1.0 / ratio)

    return (
        detected_value,
        fidelity(cell_ratio),
        fidelity(row_ratio),
    )


def _chart_value(result: Mapping[str, Any]) -> float | None:
    preserved = _nested(result, "benchmark", "visual_region_preserved")
    return float(preserved) if isinstance(preserved, bool) else None


def _stats_value(result: Mapping[str, Any], name: str, *, raw: bool = False) -> float | None:
    paths = (("raw_stats", name),) if raw else ((name,),)
    return _unit_interval(
        _first_number(result, paths, field=f"{'raw_' if raw else ''}{name}"),
        field=f"{'raw_' if raw else ''}{name}",
    )


def _stats_nonnegative_value(
    result: Mapping[str, Any], name: str, *, raw: bool = False
) -> float | None:
    paths = (("raw_stats", name),) if raw else ((name,),)
    value = _first_number(result, paths, field=f"{'raw_' if raw else ''}{name}")
    if value is not None and value < 0:
        raise ParserComparisonError(f"{'raw_' if raw else ''}{name} must be non-negative")
    return value


def _text_sha256(result: Mapping[str, Any], *, raw: bool = False) -> str | None:
    paths = (
        (("raw_stats", "text_sha256"),)
        if raw
        else (("text_sha256",), ("text", "sha256"), ("provenance", "text_sha256"))
    )
    value = _first_string(result, paths)
    if value is None:
        return None
    if len(value) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in value):
        prefix = "raw_stats." if raw else ""
        raise ParserComparisonError(f"{prefix}text_sha256 must be a lowercase SHA256 digest")
    return value


def compare_summaries(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    backend: str = "mineru",
    verify_runtime_evidence: bool = False,
) -> dict[str, Any]:
    runtime_provenance = _validate_runtime_pair(baseline, candidate)
    if verify_runtime_evidence:
        _verify_runtime_evidence(runtime_provenance)
    baseline_pages = _validated_pages(baseline, backend=backend, artifact="baseline")
    candidate_pages = _validated_pages(candidate, backend=backend, artifact="candidate")
    _validate_corpus(baseline, candidate, baseline_pages, candidate_pages)

    error_pairs = {
        "both_ok": 0,
        "baseline_only_error": 0,
        "candidate_only_error": 0,
        "both_error": 0,
    }
    baseline_latency: list[float] = []
    candidate_latency: list[float] = []
    baseline_final: list[float] = []
    candidate_final: list[float] = []
    baseline_raw: list[float] = []
    candidate_raw: list[float] = []
    baseline_backfill: list[float] = []
    candidate_backfill: list[float] = []
    structure_pairs: dict[str, tuple[list[float], list[float]]] = {
        name: ([], [])
        for name in (
            "table_detection_rate",
            "table_cell_coverage",
            "table_row_coverage",
            "chart_visual_preservation_rate",
            "bbox_valid_ratio",
            "reading_order_score",
            "raw_bbox_valid_ratio",
            "raw_reading_order_score",
        )
    }
    structure_eligible_counts = {
        name: 0
        for name in (
            "bbox_valid_ratio",
            "reading_order_score",
            "raw_bbox_valid_ratio",
            "raw_reading_order_score",
        )
    }
    duplicate_pairs: dict[str, tuple[list[float], list[float]]] = {
        "final": ([], []),
        "raw_parser": ([], []),
    }
    duplicate_eligible_counts = {"final": 0, "raw_parser": 0}
    duplicate_source_chars_missing = {"final": False, "raw_parser": False}
    text_hash_counts = {
        "final": {"paired": 0, "identical": 0},
        "raw_parser": {"paired": 0, "identical": 0},
    }

    for filename in sorted(baseline_pages):
        baseline_page = baseline_pages[filename]
        candidate_page = candidate_pages[filename]
        baseline_result = baseline_page[backend]
        candidate_result = candidate_page[backend]
        assert isinstance(baseline_result, Mapping) and isinstance(candidate_result, Mapping)
        baseline_ok = baseline_result["status"] == "ok"
        candidate_ok = candidate_result["status"] == "ok"
        if baseline_ok and candidate_ok:
            error_pairs["both_ok"] += 1
        elif baseline_ok:
            error_pairs["candidate_only_error"] += 1
            continue
        elif candidate_ok:
            error_pairs["baseline_only_error"] += 1
            continue
        else:
            error_pairs["both_error"] += 1
            continue

        left_latency = _number(baseline_result.get("latency_s"), field="latency_s")
        right_latency = _number(candidate_result.get("latency_s"), field="latency_s")
        if left_latency < 0 or right_latency < 0:
            raise ParserComparisonError("latency_s must be non-negative")
        baseline_latency.append(left_latency)
        candidate_latency.append(right_latency)

        left_final, left_raw, left_backfill = _quality_values(baseline_result)
        right_final, right_raw, right_backfill = _quality_values(candidate_result)
        if left_final is not None and right_final is not None:
            baseline_final.append(left_final)
            candidate_final.append(right_final)
        if left_raw is not None and right_raw is not None:
            baseline_raw.append(left_raw)
            candidate_raw.append(right_raw)
        if left_backfill is not None and right_backfill is not None:
            baseline_backfill.append(left_backfill)
            candidate_backfill.append(right_backfill)

        category = baseline_page.get("category")
        if category == "table":
            left_table = _table_values(baseline_result)
            right_table = _table_values(candidate_result)
            for name, left_value, right_value in zip(
                ("table_detection_rate", "table_cell_coverage", "table_row_coverage"),
                left_table,
                right_table,
                strict=True,
            ):
                if left_value is not None and right_value is not None:
                    structure_pairs[name][0].append(left_value)
                    structure_pairs[name][1].append(right_value)
        if category == "chart":
            left_chart = _chart_value(baseline_result)
            right_chart = _chart_value(candidate_result)
            if left_chart is not None and right_chart is not None:
                structure_pairs["chart_visual_preservation_rate"][0].append(left_chart)
                structure_pairs["chart_visual_preservation_rate"][1].append(right_chart)

        for name in ("bbox_valid_ratio", "reading_order_score"):
            left_final_evidence = _structure_has_evidence(baseline_result, name)
            right_final_evidence = _structure_has_evidence(candidate_result, name)
            final_evidence = left_final_evidence or right_final_evidence
            if final_evidence:
                structure_eligible_counts[name] += 1
            left_value = _structure_value(baseline_result, name) if left_final_evidence else 0.0
            right_value = _structure_value(candidate_result, name) if right_final_evidence else 0.0
            if final_evidence and left_value is not None and right_value is not None:
                structure_pairs[name][0].append(left_value)
                structure_pairs[name][1].append(right_value)
            raw_name = f"raw_{name}"
            left_raw_evidence = _structure_has_evidence(baseline_result, name, raw=True)
            right_raw_evidence = _structure_has_evidence(candidate_result, name, raw=True)
            raw_evidence = left_raw_evidence or right_raw_evidence
            if raw_evidence:
                structure_eligible_counts[raw_name] += 1
            left_raw_value = (
                _structure_value(baseline_result, name, raw=True) if left_raw_evidence else 0.0
            )
            right_raw_value = (
                _structure_value(candidate_result, name, raw=True) if right_raw_evidence else 0.0
            )
            if raw_evidence and left_raw_value is not None and right_raw_value is not None:
                structure_pairs[raw_name][0].append(left_raw_value)
                structure_pairs[raw_name][1].append(right_raw_value)

        for raw, label in ((False, "final"), (True, "raw_parser")):
            left_source_chars = _stats_nonnegative_value(
                baseline_result, "source_chars", raw=raw
            )
            right_source_chars = _stats_nonnegative_value(
                candidate_result, "source_chars", raw=raw
            )
            left_duplicate_ratio = _stats_value(
                baseline_result, "adjacent_duplicate_character_ratio", raw=raw
            )
            right_duplicate_ratio = _stats_value(
                candidate_result, "adjacent_duplicate_character_ratio", raw=raw
            )
            if left_source_chars is None or right_source_chars is None:
                duplicate_source_chars_missing[label] = True
            elif left_source_chars > 0 and right_source_chars > 0:
                duplicate_eligible_counts[label] += 1
            if (
                left_source_chars is not None
                and right_source_chars is not None
                and left_source_chars > 0
                and right_source_chars > 0
                and left_duplicate_ratio is not None
                and right_duplicate_ratio is not None
            ):
                duplicate_pairs[label][0].append(left_duplicate_ratio)
                duplicate_pairs[label][1].append(right_duplicate_ratio)

            left_hash = _text_sha256(baseline_result, raw=raw)
            right_hash = _text_sha256(candidate_result, raw=raw)
            if left_hash is not None and right_hash is not None:
                text_hash_counts[label]["paired"] += 1
                text_hash_counts[label]["identical"] += left_hash == right_hash

    latency_baseline_stats = _distribution(baseline_latency)
    latency_candidate_stats = _distribution(candidate_latency)
    latency_speedup: dict[str, float | None] = {"mean": None, "median": None, "p95": None}
    if latency_baseline_stats is not None and latency_candidate_stats is not None:
        latency_speedup = {
            name: None
            if latency_candidate_stats[name] <= 0
            else round(latency_baseline_stats[name] / latency_candidate_stats[name], 6)
            for name in latency_speedup
        }
    sign_test = two_sided_sign_test(baseline_latency, candidate_latency)
    final_quality = _paired_metric(baseline_final, candidate_final)
    raw_quality = _paired_metric(baseline_raw, candidate_raw)
    backfill = _paired_metric(baseline_backfill, candidate_backfill)
    structure = {name: _paired_metric(left, right) for name, (left, right) in structure_pairs.items()}
    duplicate_characters = {
        name: _paired_metric(left, right) for name, (left, right) in duplicate_pairs.items()
    }
    for name, metric in duplicate_characters.items():
        metric["eligible_paired_count"] = duplicate_eligible_counts[name]
        metric["excluded_empty_text_pairs"] = (
            error_pairs["both_ok"] - duplicate_eligible_counts[name]
        )
    text_identity = {
        name: {
            "paired_count": counts["paired"],
            "identical_count": counts["identical"],
            "changed_count": counts["paired"] - counts["identical"],
            "identity_rate": None
            if counts["paired"] == 0
            else round(counts["identical"] / counts["paired"], 6),
        }
        for name, counts in text_hash_counts.items()
    }

    regressions: list[str] = []
    missing_evidence: list[str] = []
    improvements: list[str] = []
    paired_ok = error_pairs["both_ok"]
    if error_pairs["candidate_only_error"]:
        regressions.append("new_candidate_errors")
    if error_pairs["both_error"]:
        missing_evidence.append("unresolved_paired_errors")
    if paired_ok < _MIN_PAIRED_PAGES:
        missing_evidence.append("insufficient_paired_pages")

    for name, metric in (("final_quality", final_quality), ("raw_quality", raw_quality)):
        if metric["paired_count"] != paired_ok:
            missing_evidence.append(f"missing_{name}")
        elif metric["delta"] < -_QUALITY_REGRESSION_TOLERANCE:
            regressions.append(f"{name}_regression")
        elif metric["delta"] >= _PRACTICAL_QUALITY_GAIN:
            improvements.append(f"{name}_improvement")
    if backfill["paired_count"] != paired_ok:
        missing_evidence.append("missing_backfill_ratio")
    elif backfill["delta"] > _BACKFILL_REGRESSION_TOLERANCE:
        regressions.append("backfill_regression")

    category_counts = {
        category: sum(page["category"] == category for page in baseline_pages.values())
        for category in ("table", "chart")
    }
    annotated_table_counts = {
        field: sum(
            page["category"] == "table" and _has_positive_annotation(page, field, filename=filename)
            for filename, page in baseline_pages.items()
        )
        for field in ("cells", "rows")
    }
    required_structure = {
        "bbox_valid_ratio": structure_eligible_counts["bbox_valid_ratio"],
        "reading_order_score": structure_eligible_counts["reading_order_score"],
        "raw_bbox_valid_ratio": structure_eligible_counts["raw_bbox_valid_ratio"],
        "raw_reading_order_score": structure_eligible_counts["raw_reading_order_score"],
        "table_detection_rate": category_counts["table"],
        "table_cell_coverage": annotated_table_counts["cells"],
        "table_row_coverage": annotated_table_counts["rows"],
        "chart_visual_preservation_rate": category_counts["chart"],
    }
    evidence_structure = set(structure_eligible_counts)
    for name, required_count in required_structure.items():
        if name in evidence_structure and required_count < _MIN_PAIRED_PAGES:
            missing_evidence.append(f"insufficient_{name}_evidence")
            continue
        if required_count == 0:
            continue
        metric = structure[name]
        if metric["paired_count"] != required_count:
            missing_evidence.append(f"missing_{name}")
        elif metric["delta"] < -_STRUCTURE_REGRESSION_TOLERANCE:
            regressions.append(f"{name}_regression")
        elif metric["delta"] >= _PRACTICAL_STRUCTURE_GAIN:
            improvements.append(f"{name}_improvement")

    for name, metric in duplicate_characters.items():
        eligible_count = duplicate_eligible_counts[name]
        if duplicate_source_chars_missing[name]:
            missing_evidence.append(f"missing_{name}_source_chars")
        elif eligible_count < _MIN_PAIRED_PAGES:
            missing_evidence.append(f"insufficient_{name}_nonempty_text_pairs")
        elif metric["paired_count"] != eligible_count:
            missing_evidence.append(f"missing_{name}_adjacent_duplicate_character_ratio")
        elif metric["delta"] > _DUPLICATE_CHARACTER_REGRESSION_TOLERANCE:
            regressions.append(f"{name}_adjacent_duplicate_character_regression")
        elif metric["delta"] <= -_PRACTICAL_DUPLICATE_CHARACTER_GAIN:
            improvements.append(f"{name}_adjacent_duplicate_character_improvement")

    for name, identity in text_identity.items():
        if identity["paired_count"] != paired_ok:
            missing_evidence.append(f"missing_{name}_text_hashes")
    if latency_baseline_stats is None or latency_candidate_stats is None:
        missing_evidence.append("missing_latency")
    else:
        median_speedup = latency_speedup["median"]
        p95_speedup = latency_speedup["p95"]
        regression_floor = 1 / _LATENCY_REGRESSION_RATIO
        median_regressed = median_speedup is not None and median_speedup < regression_floor
        p95_regressed = p95_speedup is not None and p95_speedup < regression_floor
        if median_regressed or p95_regressed:
            regressions.append("latency_regression")
        if (
            median_speedup is not None
            and median_speedup >= _PRACTICAL_SPEEDUP
            and sign_test["candidate_faster"] > sign_test["candidate_slower"]
            and sign_test["p_value"] <= _SIGN_TEST_ALPHA
        ):
            improvements.append("latency_improvement")

    regressions = sorted(set(regressions))
    missing_evidence = sorted(set(missing_evidence))
    improvements = sorted(set(improvements))
    if regressions:
        status: DecisionStatus = "rejected"
    elif missing_evidence or not improvements:
        status = "inconclusive"
    else:
        status = "accepted"

    return {
        "schema_version": "parser-benchmark-comparison-v2",
        "backend": backend,
        "status": status,
        "runtime_provenance": runtime_provenance,
        "corpus": {
            "source": baseline.get("source"),
            "source_revision": baseline.get("source_revision"),
            "page_count": len(baseline_pages),
            "manifest_sha256": _canonical_sha256(
                {
                    filename: {
                        "source_sha256": page["source_sha256"],
                        "category": page.get("category"),
                        "selection": page.get("selection"),
                    }
                    for filename, page in sorted(baseline_pages.items())
                }
            ),
        },
        "paired_errors": error_pairs,
        "latency_s": {
            "paired_count": len(baseline_latency),
            "baseline": latency_baseline_stats,
            "candidate": latency_candidate_stats,
            "speedup_baseline_over_candidate": latency_speedup,
            "two_sided_sign_test": sign_test,
        },
        "quality": {"final": final_quality, "raw_parser": raw_quality},
        "backfill_ratio": backfill,
        "structure": structure,
        "adjacent_duplicate_characters": duplicate_characters,
        "text_hash_identity": text_identity,
        "gate": {
            "regressions": regressions,
            "missing_evidence": missing_evidence,
            "improvements": improvements,
            "policy": {
                "min_paired_pages": _MIN_PAIRED_PAGES,
                "quality_regression_tolerance": _QUALITY_REGRESSION_TOLERANCE,
                "structure_regression_tolerance": _STRUCTURE_REGRESSION_TOLERANCE,
                "backfill_regression_tolerance": _BACKFILL_REGRESSION_TOLERANCE,
                "latency_regression_ratio": _LATENCY_REGRESSION_RATIO,
                "practical_quality_gain": _PRACTICAL_QUALITY_GAIN,
                "practical_structure_gain": _PRACTICAL_STRUCTURE_GAIN,
                "practical_speedup": _PRACTICAL_SPEEDUP,
                "sign_test_alpha": _SIGN_TEST_ALPHA,
                "duplicate_character_regression_tolerance": (_DUPLICATE_CHARACTER_REGRESSION_TOLERANCE),
                "practical_duplicate_character_gain": _PRACTICAL_DUPLICATE_CHARACTER_GAIN,
            },
        },
    }


def _load_summary(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ParserComparisonError(f"cannot read benchmark summary: {path}") from exc
    if not isinstance(value, Mapping):
        raise ParserComparisonError(f"benchmark summary must be an object: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--backend", default="mineru")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        baseline = _load_summary(args.baseline)
        candidate = _load_summary(args.candidate)
        report = compare_summaries(
            baseline,
            candidate,
            backend=args.backend,
            verify_runtime_evidence=True,
        )
        report["artifacts"] = {
            "baseline_sha256": _file_sha256(args.baseline),
            "candidate_sha256": _file_sha256(args.candidate),
        }
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.write_text(payload, encoding="utf-8")
        sys.stdout.write(payload)
    except ParserComparisonError as exc:
        print(f"parser benchmark comparison invalid: {exc}", file=sys.stderr)
        raise SystemExit(4) from None
    exit_codes: dict[DecisionStatus, int] = {
        "accepted": 0,
        "rejected": 2,
        "inconclusive": 3,
    }
    raise SystemExit(exit_codes[report["status"]])


if __name__ == "__main__":
    main()
