"""Build a deterministic manifest for the license-clear OCR corpus.

The builder is deliberately offline: it reads a lightweight JSONL candidate
catalog and never downloads document images, PDFs, or dataset archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

_DEFAULT_SEED = "docragenslate-ocr-v1"
_HASH_ALGORITHM = "sha256-nul-v1"
_CANONICAL_JSON_ALGORITHM = "sha256-json-sort-keys-ascii-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CONTENT_ADDRESSED_SCHEMES = {"artifact+sha256", "sha256"}


@dataclass(frozen=True)
class DatasetSpec:
    source: str
    source_url: str
    revision: str
    license: str
    quotas: Mapping[str, int]
    expected_inputs: Mapping[str, int]
    license_components: Mapping[str, str] | None = None
    source_components: Mapping[str, Mapping[str, str]] | None = None


DATASETS: dict[str, DatasetSpec] = {
    "pubtables_v2": DatasetSpec(
        source="kensho/PubTables-v2",
        source_url="https://huggingface.co/datasets/kensho/PubTables-v2",
        revision="aa575e798cb00a296925e2086addb3e3fd9a1903",
        license="CDLA-Permissive-2.0",
        quotas={"pages_2": 8, "pages_3": 4, "pages_4": 2},
        expected_inputs={"pages_2": 2, "pages_3": 3, "pages_4": 4},
    ),
    "varex": DatasetSpec(
        source="ibm-research/VAREX",
        source_url="https://huggingface.co/datasets/ibm-research/VAREX",
        revision="2dfc3386a4567c7d56bf1abf4d12ff42afed27b6",
        license="CDLA-Permissive-2.0",
        quotas={"Flat": 10, "Nested": 10, "Table": 10},
        expected_inputs={"Flat": 2, "Nested": 2, "Table": 2},
    ),
    "ai2d_rst": DatasetSpec(
        source="thiippal/AI2D-RST",
        source_url="https://github.com/thiippal/AI2D-RST",
        revision="76cf0f8fadd1f431545fa540c58ef8a24ea31335",
        license="CC-BY-4.0 AND CC-BY-SA-4.0",
        quotas={"diagram": 12},
        expected_inputs={"diagram": 1},
        license_components={
            "annotations": "CC-BY-4.0",
            "source_images": "CC-BY-SA-4.0",
        },
        source_components={
            "annotations": {
                "source": "thiippal/AI2D-RST",
                "source_url": "https://github.com/thiippal/AI2D-RST",
                "revision": "76cf0f8fadd1f431545fa540c58ef8a24ea31335",
            },
            "source_images": {
                "source": "AllenAI/AI2D",
                "source_url": "https://registry.opendata.aws/allenai-diagrams/",
                "revision": "content-addressed-by-input-sha256",
            },
        },
    ),
    "mws_vision_bench": DatasetSpec(
        source="MTSAIR/MWS-Vision-Bench",
        source_url="https://huggingface.co/datasets/MTSAIR/MWS-Vision-Bench",
        revision="b8d473734b79343cac2b74f692a29ab191c7d11d",
        license="MIT",
        quotas={
            "document_parsing_ru": 6,
            "full_page_ocr_ru": 6,
            "key_information_extraction_ru": 6,
            "reasoning_vqa_ru": 6,
            "text_grounding_ru": 6,
        },
        expected_inputs={
            "document_parsing_ru": 1,
            "full_page_ocr_ru": 1,
            "key_information_extraction_ru": 1,
            "reasoning_vqa_ru": 1,
            "text_grounding_ru": 1,
        },
    ),
}


def _selection_hash(seed: str, dataset: str, stratum: str, canonical_id: str) -> str:
    value = "\0".join((seed, dataset, stratum, canonical_id)).encode()
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _catalog_hash(candidates: Sequence[dict[str, Any]]) -> str:
    rows = sorted(_canonical_json(candidate) for candidate in candidates)
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row)
        digest.update(b"\n")
    return digest.hexdigest()


def _read_candidates(path: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"candidate on line {line_number} must be a JSON object")
        candidates.append(value)
    return candidates


def _require_string(metadata: Mapping[str, Any], key: str, context: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: metadata.{key} must be a non-empty string")
    return value


def _safe_pinned_path(uri: str, expected_prefix: str, context: str) -> None:
    parsed = urlparse(uri)
    if parsed.query or parsed.fragment or parsed.params:
        raise ValueError(f"{context}.uri must not contain query, fragment or params")
    raw_path_lower = parsed.path.lower()
    if "%2f" in raw_path_lower or "%2e" in raw_path_lower:
        raise ValueError(f"{context}.uri must not contain encoded path separators")
    decoded_path = unquote(parsed.path)
    if not decoded_path.startswith(expected_prefix) or decoded_path == expected_prefix:
        raise ValueError(f"{context}.uri must use the exact pinned source revision path")
    if any(segment in {"", ".", ".."} for segment in decoded_path.split("/")[1:]):
        raise ValueError(f"{context}.uri contains an unsafe path segment")


def _validate_uri_policy(
    dataset: str,
    component: str,
    uri: str,
    sha256: str,
    *,
    spec: DatasetSpec,
    context: str,
) -> None:
    parsed = urlparse(uri)
    if dataset in {"pubtables_v2", "varex", "mws_vision_bench"}:
        if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
            raise ValueError(f"{context}.uri must use trusted HTTPS Hugging Face storage")
        prefix = f"/datasets/{spec.source}/resolve/{spec.revision}/"
        _safe_pinned_path(uri, prefix, context)
        return
    if dataset != "ai2d_rst":
        raise ValueError(f"no URI policy for dataset: {dataset}")
    if component == "annotation":
        if parsed.scheme != "https" or parsed.netloc != "raw.githubusercontent.com":
            raise ValueError(f"{context}.uri must use trusted raw.githubusercontent.com storage")
        prefix = f"/thiippal/AI2D-RST/{spec.revision}/"
        _safe_pinned_path(uri, prefix, context)
        return
    if component != "source_image":
        raise ValueError(f"unsupported AI2D component: {component}")
    if parsed.scheme not in _CONTENT_ADDRESSED_SCHEMES or parsed.netloc != sha256:
        raise ValueError(
            f"{context}.uri must be content-addressed with sha256 as the exact authority component"
        )
    if parsed.query or parsed.fragment or parsed.params:
        raise ValueError(f"{context}.uri must not contain query, fragment or params")
    if parsed.scheme == "sha256" and parsed.path not in {"", "/"}:
        raise ValueError(f"{context}.uri sha256 scheme must not add a mutable path")
    if parsed.scheme == "artifact+sha256":
        decoded_path = unquote(parsed.path)
        if any(
            segment.lower() in {".", "..", "latest", "main", "master"}
            for segment in decoded_path.split("/")
        ):
            raise ValueError(f"{context}.uri contains a mutable or unsafe artifact path")


def _validate_reference(
    value: Any,
    *,
    dataset: str,
    component: str,
    spec: DatasetSpec,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    uri = value.get("uri")
    if not isinstance(uri, str) or not uri:
        raise ValueError(f"{context}.uri must be an absolute URI")
    sha256 = value.get("sha256")
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
        raise ValueError(f"{context}.sha256 must be 64 lowercase hexadecimal characters")
    _validate_uri_policy(dataset, component, uri, sha256, spec=spec, context=context)
    return {"uri": uri, "sha256": sha256}


def _expected_roles(dataset: str, stratum: str, expected_inputs: int) -> set[str]:
    if dataset == "pubtables_v2":
        return {f"page_{index}" for index in range(1, expected_inputs + 1)}
    if dataset == "varex":
        return {"image_200dpi", "image_50dpi"}
    if dataset in {"ai2d_rst", "mws_vision_bench"}:
        return {"image"}
    raise ValueError(f"no input-role schema for dataset: {dataset}")


def _validate_metadata(
    dataset: str,
    stratum: str,
    metadata: Any,
    *,
    canonical_id: str,
    group_id: str,
    spec: DatasetSpec,
    expected_inputs: int,
    context: str,
) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError(f"{context}: metadata must be an object")
    if dataset == "pubtables_v2":
        document_id = _require_string(metadata, "document_id", context)
        _require_string(metadata, "table_id", context)
        page_indices = metadata.get("page_indices")
        if (
            not isinstance(page_indices, list)
            or len(page_indices) != expected_inputs
            or any(
                not isinstance(index, int) or isinstance(index, bool) or index < 0
                for index in page_indices
            )
            or page_indices != sorted(set(page_indices))
        ):
            raise ValueError(
                f"{context}: metadata.page_indices must contain {expected_inputs} unique sorted integers"
            )
        _validate_reference(
            metadata.get("annotation"),
            dataset=dataset,
            component="annotation",
            spec=spec,
            context=f"{context}.metadata.annotation",
        )
        expected_group_id = document_id
    elif dataset == "varex":
        expected_group_id = _require_string(metadata, "doc_id", context)
        _validate_reference(
            metadata.get("source_record"),
            dataset=dataset,
            component="source_record",
            spec=spec,
            context=f"{context}.metadata.source_record",
        )
        schema = metadata.get("schema")
        ground_truth = metadata.get("ground_truth")
        if not isinstance(schema, dict) or not schema:
            raise ValueError(f"{context}: metadata.schema must be a non-empty object")
        if not isinstance(ground_truth, dict) or not ground_truth:
            raise ValueError(f"{context}: metadata.ground_truth must be a non-empty object")
        properties = schema.get("properties")
        if schema.get("type") != "object" or not isinstance(properties, dict) or not properties:
            raise ValueError(f"{context}: metadata.schema requires object type and non-empty properties")
        if stratum == "Nested" and not any(
            isinstance(value, dict)
            and (value.get("type") == "object" or isinstance(value.get("properties"), dict))
            for value in properties.values()
        ):
            raise ValueError(f"{context}: Nested schema requires a nested object property")
        if stratum == "Nested" and not any(isinstance(value, dict) for value in ground_truth.values()):
            raise ValueError(f"{context}: Nested ground_truth requires a nested object value")
        if stratum == "Table" and not any(
            isinstance(value, dict) and value.get("type") == "array"
            for value in properties.values()
        ):
            raise ValueError(f"{context}: Table schema requires an array property")
        if stratum == "Table" and not any(isinstance(value, list) for value in ground_truth.values()):
            raise ValueError(f"{context}: Table ground_truth requires an array value")
    elif dataset == "ai2d_rst":
        expected_group_id = _require_string(metadata, "diagram_id", context)
        _validate_reference(
            metadata.get("annotation"),
            dataset=dataset,
            component="annotation",
            spec=spec,
            context=f"{context}.metadata.annotation",
        )
    elif dataset == "mws_vision_bench":
        expected_group_id = _require_string(metadata, "image_path", context)
        question = _require_string(metadata, "question", context)
        if metadata.get("task_type") != stratum:
            raise ValueError(f"{context}: metadata.task_type must match stratum {stratum!r}")
        answers = metadata.get("answers")
        if (
            not isinstance(answers, list)
            or not answers
            or any(not isinstance(answer, str) or not answer for answer in answers)
        ):
            raise ValueError(f"{context}: metadata.answers must be a non-empty list of strings")
        _validate_reference(
            metadata.get("source_record"),
            dataset=dataset,
            component="source_record",
            spec=spec,
            context=f"{context}.metadata.source_record",
        )
    else:
        raise ValueError(f"no metadata schema for dataset: {dataset}")
    if group_id != expected_group_id:
        raise ValueError(
            f"{context}: group_id must equal the physical content id {expected_group_id!r}"
        )
    normalized = dict(metadata)
    if dataset == "varex":
        schema_hash = _canonical_json_hash(normalized["schema"])
        ground_truth_hash = _canonical_json_hash(normalized["ground_truth"])
        for key, value in (
            ("schema_canonical_sha256", schema_hash),
            ("ground_truth_canonical_sha256", ground_truth_hash),
        ):
            if key in normalized and normalized[key] != value:
                raise ValueError(f"{context}: metadata.{key} does not match canonical content")
            normalized[key] = value
    if dataset == "mws_vision_bench":
        record_hash = _canonical_json_hash(
            {
                "id": canonical_id,
                "image_path": expected_group_id,
                "task_type": stratum,
                "question": question,
                "answers": answers,
            }
        )
        key = "source_record_canonical_sha256"
        if key in normalized and normalized[key] != record_hash:
            raise ValueError(f"{context}: metadata.{key} does not match the bound record")
        normalized[key] = record_hash
    return normalized


def _validate_candidate(
    candidate: dict[str, Any],
    *,
    specs: Mapping[str, DatasetSpec],
) -> dict[str, Any]:
    allowed_fields = {"dataset", "stratum", "canonical_id", "group_id", "inputs", "metadata"}
    unknown_fields = set(candidate) - allowed_fields
    if unknown_fields:
        raise ValueError(f"candidate contains unsupported fields: {sorted(unknown_fields)}")
    dataset = candidate.get("dataset")
    if not isinstance(dataset, str) or dataset not in specs:
        raise ValueError(f"unsupported dataset: {dataset!r}")
    stratum = candidate.get("stratum")
    if not isinstance(stratum, str) or stratum not in specs[dataset].quotas:
        raise ValueError(f"unsupported stratum for {dataset}: {stratum!r}")
    canonical_id = candidate.get("canonical_id")
    if not isinstance(canonical_id, str) or not canonical_id.strip():
        raise ValueError("canonical_id must be a non-empty string")
    group_id = candidate.get("group_id")
    if not isinstance(group_id, str) or not group_id.strip():
        raise ValueError(f"{dataset}/{canonical_id}: group_id must be a non-empty string")
    context = f"{dataset}/{canonical_id}"
    inputs = candidate.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError(f"{context}: inputs must be a non-empty list")
    expected = specs[dataset].expected_inputs[stratum]
    if len(inputs) != expected:
        raise ValueError(
            f"{context}: stratum {stratum!r} requires {expected} inputs, "
            f"got {len(inputs)}"
        )
    normalized_inputs: list[dict[str, Any]] = []
    for index, item in enumerate(inputs):
        if not isinstance(item, dict):
            raise ValueError(f"{context}: input {index} must be an object")
        unknown_input_fields = set(item) - {"role", "uri", "sha256"}
        if unknown_input_fields:
            raise ValueError(
                f"{context}: input {index} contains unsupported fields: {sorted(unknown_input_fields)}"
            )
        role = item.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"{context}: input {index} requires a non-empty role")
        reference = _validate_reference(
            item,
            dataset=dataset,
            component="source_image" if dataset == "ai2d_rst" else "input",
            spec=specs[dataset],
            context=f"{context}.inputs[{index}]",
        )
        normalized_inputs.append({"role": role, **reference})
    roles = [item["role"] for item in normalized_inputs]
    expected_roles = _expected_roles(dataset, stratum, expected)
    if set(roles) != expected_roles or len(roles) != len(set(roles)):
        raise ValueError(f"{context}: input roles must be exactly {sorted(expected_roles)}")
    normalized_inputs.sort(key=lambda item: item["role"])
    uris = [item["uri"] for item in normalized_inputs]
    if len(uris) != len(set(uris)):
        raise ValueError(f"{context}: input uris must be unique")
    input_hashes = [item["sha256"] for item in normalized_inputs]
    if len(input_hashes) != len(set(input_hashes)):
        raise ValueError(f"{context}: input sha256 values must be unique")
    metadata = _validate_metadata(
        dataset,
        stratum,
        candidate.get("metadata"),
        canonical_id=canonical_id,
        group_id=group_id,
        spec=specs[dataset],
        expected_inputs=expected,
        context=context,
    )
    return {
        "dataset": dataset,
        "stratum": stratum,
        "canonical_id": canonical_id,
        "group_id": group_id,
        "inputs": normalized_inputs,
        "metadata": metadata,
    }


def _select_dataset(
    dataset: str,
    candidates: Sequence[dict[str, Any]],
    *,
    seed: str,
    spec: DatasetSpec,
) -> list[dict[str, Any]]:
    ranked: dict[str, list[dict[str, Any]]] = {stratum: [] for stratum in spec.quotas}
    for candidate in candidates:
        selection_hash = _selection_hash(
            seed,
            dataset,
            candidate["stratum"],
            candidate["canonical_id"],
        )
        ranked[candidate["stratum"]].append({**candidate, "selection_hash": selection_hash})
    for stratum, values in ranked.items():
        values.sort(key=lambda item: (item["selection_hash"], item["canonical_id"]))
        best_by_group: dict[str, dict[str, Any]] = {}
        for candidate in values:
            best_by_group.setdefault(candidate["group_id"], candidate)
        ranked[stratum] = list(best_by_group.values())

    # Bipartite b-matching: quota slots on the left, physical groups on the right.
    # Augmenting paths preserve deterministic hash order but can move an earlier
    # group to another stratum, avoiding the false shortages of greedy selection.
    strata = sorted(
        spec.quotas,
        key=lambda stratum: (len(ranked[stratum]) / spec.quotas[stratum], stratum),
    )
    slots = [
        (stratum, slot_index)
        for stratum in strata
        for slot_index in range(spec.quotas[stratum])
    ]
    group_to_slot: dict[str, tuple[str, int]] = {}
    slot_to_candidate: dict[tuple[str, int], dict[str, Any]] = {}

    def augment(slot: tuple[str, int], seen_groups: set[str]) -> bool:
        stratum, _ = slot
        for candidate in ranked[stratum]:
            group_id = candidate["group_id"]
            if group_id in seen_groups:
                continue
            seen_groups.add(group_id)
            previous_slot = group_to_slot.get(group_id)
            if previous_slot is None or augment(previous_slot, seen_groups):
                group_to_slot[group_id] = slot
                slot_to_candidate[slot] = candidate
                return True
        return False

    unmatched: list[tuple[str, int]] = []
    for slot in slots:
        if not augment(slot, set()):
            unmatched.append(slot)
    if unmatched:
        matched_counts = {
            stratum: sum(slot[0] == stratum for slot in slot_to_candidate)
            for stratum in spec.quotas
        }
        shortages = {
            stratum: spec.quotas[stratum] - matched_counts[stratum]
            for stratum in sorted(spec.quotas)
            if matched_counts[stratum] < spec.quotas[stratum]
        }
        raise ValueError(f"not enough unique groups for {dataset}: {shortages}")

    selected = list(slot_to_candidate.values())
    return sorted(selected, key=lambda item: (item["stratum"], item["selection_hash"]))


def _validate_selected_assets(selected: Sequence[dict[str, Any]]) -> None:
    seen_uris: dict[str, tuple[str, str]] = {}
    seen_hashes: dict[str, tuple[str, str]] = {}
    for candidate in selected:
        owner = (candidate["dataset"], candidate["group_id"])
        for item in candidate["inputs"]:
            for key, seen in (("uri", seen_uris), ("sha256", seen_hashes)):
                previous = seen.get(item[key])
                if previous is not None and previous != owner:
                    raise ValueError(
                        f"duplicate selected input {key} across groups: {previous} and {owner}"
                    )
                seen[item[key]] = owner


def _validate_specs(specs: Mapping[str, DatasetSpec]) -> None:
    if not specs:
        raise ValueError("at least one dataset spec is required")
    for dataset, spec in specs.items():
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset spec: {dataset}")
        pinned = DATASETS[dataset]
        pinned_metadata = (
            "source",
            "source_url",
            "revision",
            "license",
            "license_components",
            "source_components",
        )
        if any(getattr(spec, field) != getattr(pinned, field) for field in pinned_metadata):
            raise ValueError(f"{dataset}: source and license metadata must match the pinned spec")
        if not spec.source or not spec.source_url or not spec.revision or not spec.license:
            raise ValueError(f"{dataset}: source, source_url, revision and license are required")
        if set(spec.quotas) != set(spec.expected_inputs):
            raise ValueError(f"{dataset}: quotas and expected_inputs strata must match")
        if not set(spec.quotas) <= set(pinned.quotas):
            raise ValueError(f"{dataset}: unsupported quota stratum")
        if any(spec.expected_inputs[stratum] != pinned.expected_inputs[stratum] for stratum in spec.quotas):
            raise ValueError(f"{dataset}: expected_inputs must match the pinned spec")
        if any(value < 1 for value in (*spec.quotas.values(), *spec.expected_inputs.values())):
            raise ValueError(f"{dataset}: quotas and expected_inputs must be positive")


def build_manifest(
    candidates: Iterable[dict[str, Any]],
    *,
    seed: str = _DEFAULT_SEED,
    specs: Mapping[str, DatasetSpec] = DATASETS,
) -> dict[str, Any]:
    if not seed:
        raise ValueError("seed must be non-empty")
    _validate_specs(specs)
    validated = [_validate_candidate(candidate, specs=specs) for candidate in candidates]
    identities: set[tuple[str, str]] = set()
    for candidate in validated:
        identity = (candidate["dataset"], candidate["canonical_id"])
        if identity in identities:
            raise ValueError(f"duplicate candidate: {'/'.join(identity)}")
        identities.add(identity)

    by_dataset: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in specs}
    for candidate in validated:
        by_dataset[candidate["dataset"]].append(candidate)

    selected: list[dict[str, Any]] = []
    for dataset in sorted(specs):
        selected.extend(
            _select_dataset(dataset, by_dataset[dataset], seed=seed, spec=specs[dataset])
        )
    _validate_selected_assets(selected)

    dataset_metadata: dict[str, Any] = {}
    summary_by_dataset: dict[str, Any] = {}
    for dataset in sorted(specs):
        spec = specs[dataset]
        metadata: dict[str, Any] = {
            "source": spec.source,
            "source_url": spec.source_url,
            "revision": spec.revision,
            "license": spec.license,
            "quotas": dict(spec.quotas),
        }
        if spec.license_components is not None:
            metadata["license_components"] = dict(spec.license_components)
        if spec.source_components is not None:
            metadata["source_components"] = {
                name: dict(component) for name, component in spec.source_components.items()
            }
        dataset_metadata[dataset] = metadata
        items = [item for item in selected if item["dataset"] == dataset]
        summary_by_dataset[dataset] = {
            "selected_units": len(items),
            "input_count": sum(len(item["inputs"]) for item in items),
            "by_stratum": {
                stratum: sum(item["stratum"] == stratum for item in items)
                for stratum in sorted(spec.quotas)
            },
        }

    return {
        "manifest_version": 1,
        "selection": {
            "algorithm": _HASH_ALGORITHM,
            "seed": seed,
            "catalog_sha256": _catalog_hash(validated),
            "canonical_json_algorithm": _CANONICAL_JSON_ALGORITHM,
            "group_policy": "claimed_identity_only; one selected unit per dataset/group_id",
        },
        "verification_state": "metadata_only_unverified",
        "materialization_requirements": {
            "verify_every_referenced_object_bytes_against_sha256": True,
            "verify_group_id_against_trusted_physical_identity": True,
            "reject_manifest_on_any_mismatch": True,
        },
        "datasets": dataset_metadata,
        "summary": {
            "candidate_count": len(validated),
            "selected_units": len(selected),
            "input_count": sum(len(item["inputs"]) for item in selected),
            "by_dataset": summary_by_dataset,
        },
        "selected": selected,
    }


def _spec_document(specs: Mapping[str, DatasetSpec] = DATASETS) -> dict[str, Any]:
    return {
        dataset: {
            "source": spec.source,
            "source_url": spec.source_url,
            "revision": spec.revision,
            "license": spec.license,
            "quotas": dict(spec.quotas),
            "expected_inputs": dict(spec.expected_inputs),
            **(
                {"license_components": dict(spec.license_components)}
                if spec.license_components is not None
                else {}
            ),
            **(
                {
                    "source_components": {
                        name: dict(component) for name, component in spec.source_components.items()
                    }
                }
                if spec.source_components is not None
                else {}
            ),
        }
        for dataset, spec in sorted(specs.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an offline, hash-ranked OCR acceptance manifest from JSONL metadata."
    )
    parser.add_argument("candidates", type=Path, nargs="?", help="lightweight candidate catalog (JSONL)")
    parser.add_argument("output", type=Path, nargs="?", help="output manifest JSON")
    parser.add_argument("--seed", default=_DEFAULT_SEED)
    parser.add_argument(
        "--print-specs",
        action="store_true",
        help="print pinned dataset metadata and catalog quotas without reading candidates",
    )
    args = parser.parse_args()
    if args.print_specs:
        print(json.dumps(_spec_document(), ensure_ascii=False, indent=2))
        return
    if args.candidates is None or args.output is None:
        parser.error("candidates and output are required unless --print-specs is used")

    manifest = build_manifest(_read_candidates(args.candidates), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"selected {manifest['summary']['selected_units']} units / "
        f"{manifest['summary']['input_count']} inputs -> {args.output}"
    )


if __name__ == "__main__":
    main()
