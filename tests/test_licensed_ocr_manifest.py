from __future__ import annotations

import copy
import hashlib
import json
import runpy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "build_licensed_ocr_manifest.py")
)
DATASETS = _SCRIPT["DATASETS"]
_selection_hash = _SCRIPT["_selection_hash"]
_canonical_json_hash = _SCRIPT["_canonical_json_hash"]
build_manifest = _SCRIPT["build_manifest"]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _reference(dataset: str, value: str, *, component: str = "input") -> dict[str, str]:
    sha256 = _sha(f"{dataset}:{value}")
    spec = DATASETS[dataset]
    if dataset == "ai2d_rst" and component == "source_image":
        uri = f"sha256://{sha256}"
    elif dataset == "ai2d_rst" and component == "annotation":
        uri = f"https://raw.githubusercontent.com/thiippal/AI2D-RST/{spec.revision}/{value}"
    else:
        uri = f"https://huggingface.co/datasets/{spec.source}/resolve/{spec.revision}/{value}"
    return {
        "uri": uri,
        "sha256": sha256,
    }


def _roles(dataset: str, stratum: str) -> list[str]:
    count = DATASETS[dataset].expected_inputs[stratum]
    if dataset == "pubtables_v2":
        return [f"page_{index}" for index in range(1, count + 1)]
    if dataset == "varex":
        return ["image_200dpi", "image_50dpi"]
    return ["image"]


def _candidate(
    dataset: str,
    stratum: str,
    canonical_id: str,
    *,
    group_id: str | None = None,
) -> dict[str, Any]:
    physical_id = group_id or f"{dataset}-group-{canonical_id}"
    inputs = [
        {
            "role": role,
            **_reference(
                dataset,
                f"{physical_id}/{role}",
                component="source_image" if dataset == "ai2d_rst" else "input",
            ),
        }
        for role in _roles(dataset, stratum)
    ]
    if dataset == "pubtables_v2":
        metadata: dict[str, Any] = {
            "document_id": physical_id,
            "table_id": canonical_id,
            "page_indices": list(range(len(inputs))),
            "annotation": _reference(dataset, f"{canonical_id}/annotation.json", component="annotation"),
        }
    elif dataset == "varex":
        if stratum == "Nested":
            schema = {
                "type": "object",
                "properties": {
                    "person": {"type": "object", "properties": {"name": {"type": "string"}}}
                },
            }
            ground_truth: dict[str, Any] = {"person": {"name": canonical_id}}
        elif stratum == "Table":
            schema = {
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"value": {"type": "string"}}},
                    }
                },
            }
            ground_truth = {"rows": [{"value": canonical_id}]}
        else:
            schema = {"type": "object", "properties": {"field": {"type": "string"}}}
            ground_truth = {"field": canonical_id}
        metadata = {
            "doc_id": physical_id,
            "schema": schema,
            "ground_truth": ground_truth,
            "source_record": _reference(dataset, f"records/{canonical_id}.json"),
        }
    elif dataset == "ai2d_rst":
        metadata = {
            "diagram_id": physical_id,
            "annotation": _reference(dataset, f"{canonical_id}/annotation.json", component="annotation"),
        }
    elif dataset == "mws_vision_bench":
        metadata = {
            "image_path": physical_id,
            "task_type": stratum,
            "question": f"Question {canonical_id}",
            "answers": [f"Answer {canonical_id}"],
            "source_record": _reference(dataset, f"records/{canonical_id}.json"),
        }
    else:
        raise AssertionError(dataset)
    return {
        "dataset": dataset,
        "stratum": stratum,
        "canonical_id": canonical_id,
        "group_id": physical_id,
        "inputs": inputs,
        "metadata": metadata,
    }


def _specs_for(dataset: str, quotas: dict[str, int]) -> dict[str, Any]:
    spec = DATASETS[dataset]
    return {
        dataset: replace(
            spec,
            quotas=quotas,
            expected_inputs={stratum: spec.expected_inputs[stratum] for stratum in quotas},
        )
    }


def _complete_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for dataset, spec in DATASETS.items():
        for stratum, quota in spec.quotas.items():
            for index in range(quota):
                candidates.append(_candidate(dataset, stratum, f"{stratum}-{index:02d}"))
    return candidates


def test_selection_hash_uses_pinned_nul_separated_rule() -> None:
    assert _selection_hash("docragenslate-ocr-v1", "varex", "Flat", "1044") == (
        "38e2351bf9eb0bd6d0a776789e181737961b927fe9b76848c06266bd7333fa32"
    )


def test_matching_finds_feasible_selection_across_adversarial_group_conflicts() -> None:
    dataset = "mws_vision_bench"
    strata = ["document_parsing_ru", "full_page_ocr_ru", "key_information_extraction_ru"]
    specs = _specs_for(dataset, dict.fromkeys(strata, 1))

    def ids_in_hash_order(stratum: str, prefix: str) -> list[str]:
        values = [f"{prefix}-0", f"{prefix}-1"]
        return sorted(
            values,
            key=lambda value: _selection_hash("docragenslate-ocr-v1", dataset, stratum, value),
        )

    first_ids = ids_in_hash_order(strata[0], "a")
    second_ids = ids_in_hash_order(strata[1], "b")
    third_ids = ids_in_hash_order(strata[2], "c")
    candidates = [
        _candidate(dataset, strata[0], first_ids[0], group_id="group-1"),
        _candidate(dataset, strata[0], first_ids[1], group_id="group-2"),
        _candidate(dataset, strata[1], second_ids[0], group_id="group-2"),
        _candidate(dataset, strata[1], second_ids[1], group_id="group-3"),
        _candidate(dataset, strata[2], third_ids[0], group_id="group-1"),
        _candidate(dataset, strata[2], third_ids[1], group_id="group-2"),
    ]

    first = build_manifest(candidates, specs=specs)
    second = build_manifest(reversed(candidates), specs=specs)

    assert first == second
    assert first["summary"]["selected_units"] == 3
    assert len({item["group_id"] for item in first["selected"]}) == 3
    assert first["summary"]["by_dataset"][dataset]["by_stratum"] == dict.fromkeys(sorted(strata), 1)


def test_matching_reports_a_real_unique_group_shortage() -> None:
    dataset = "mws_vision_bench"
    strata = ["document_parsing_ru", "full_page_ocr_ru"]
    specs = _specs_for(dataset, dict.fromkeys(strata, 1))
    candidates = [
        _candidate(dataset, strata[0], "a", group_id="same-image"),
        _candidate(dataset, strata[1], "b", group_id="same-image"),
    ]

    with pytest.raises(ValueError, match="not enough unique groups"):
        build_manifest(candidates, specs=specs)


@pytest.mark.parametrize("field", ["uri", "sha256"])
def test_manifest_rejects_duplicate_selected_assets_across_groups(field: str) -> None:
    dataset = "varex"
    specs = _specs_for(dataset, {"Flat": 2})
    first = _candidate(dataset, "Flat", "form-a")
    second = _candidate(dataset, "Flat", "form-b")
    second["inputs"][0][field] = first["inputs"][0][field]

    with pytest.raises(ValueError, match=f"duplicate selected input {field}"):
        build_manifest([first, second], specs=specs)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda item: item.pop("role"), "requires a non-empty role"),
        (lambda item: item.__setitem__("sha256", item["sha256"].upper()), "64 lowercase"),
    ],
)
def test_manifest_rejects_unpinned_or_incomplete_input_references(mutation: Any, message: str) -> None:
    candidate = _candidate("ai2d_rst", "diagram", "diagram-a")
    mutation(candidate["inputs"][0])

    with pytest.raises(ValueError, match=message):
        build_manifest([candidate], specs=_specs_for("ai2d_rst", {"diagram": 1}))


@pytest.mark.parametrize("revision", ["main", "master", "latest"])
def test_hf_policy_rejects_mutable_revisions(revision: str) -> None:
    dataset = "varex"
    candidate = _candidate(dataset, "Flat", "form-a")
    candidate["inputs"][0]["uri"] = (
        f"https://huggingface.co/datasets/{DATASETS[dataset].source}/resolve/{revision}/image.jpg"
    )

    with pytest.raises(ValueError, match="exact pinned source revision path"):
        build_manifest([candidate], specs=_specs_for(dataset, {"Flat": 1}))


@pytest.mark.parametrize(
    ("uri", "message"),
    [
        (
            "https://evil.example/datasets/ibm-research/VAREX/resolve/"
            "2dfc3386a4567c7d56bf1abf4d12ff42afed27b6/image.jpg",
            "trusted HTTPS Hugging Face",
        ),
        ("file:///tmp/varex/image.jpg", "trusted HTTPS Hugging Face"),
        (
            "https://huggingface.co/datasets/ibm-research/VAREX/resolve/"
            "2dfc3386a4567c7d56bf1abf4d12ff42afed27b6/image.jpg?source=evil",
            "must not contain query",
        ),
    ],
)
def test_hf_policy_rejects_evil_file_and_query_tricks(uri: str, message: str) -> None:
    candidate = _candidate("varex", "Flat", "form-a")
    candidate["inputs"][0]["uri"] = uri

    with pytest.raises(ValueError, match=message):
        build_manifest([candidate], specs=_specs_for("varex", {"Flat": 1}))


def test_ai2d_content_addressed_input_uri_is_allowed() -> None:
    candidate = _candidate("ai2d_rst", "diagram", "diagram-a")
    item = candidate["inputs"][0]
    item["uri"] = f"artifact+sha256://{item['sha256']}/stored/diagram-a.png"

    manifest = build_manifest([candidate], specs=_specs_for("ai2d_rst", {"diagram": 1}))

    assert manifest["selected"][0]["inputs"][0]["uri"] == item["uri"]


def test_ai2d_component_provenance_is_not_interchangeable() -> None:
    image_from_raw = _candidate("ai2d_rst", "diagram", "diagram-a")
    image_from_raw["inputs"][0]["uri"] = image_from_raw["metadata"]["annotation"]["uri"]
    annotation_from_artifact = _candidate("ai2d_rst", "diagram", "diagram-b")
    annotation = annotation_from_artifact["metadata"]["annotation"]
    annotation["uri"] = f"sha256://{annotation['sha256']}"
    annotation_from_master = _candidate("ai2d_rst", "diagram", "diagram-c")
    annotation_from_master["metadata"]["annotation"]["uri"] = (
        "https://raw.githubusercontent.com/thiippal/AI2D-RST/master/annotation.json"
    )

    specs = _specs_for("ai2d_rst", {"diagram": 1})
    with pytest.raises(ValueError, match="exact authority component"):
        build_manifest([image_from_raw], specs=specs)
    with pytest.raises(ValueError, match="trusted raw.githubusercontent.com"):
        build_manifest([annotation_from_artifact], specs=specs)
    with pytest.raises(ValueError, match="exact pinned source revision path"):
        build_manifest([annotation_from_master], specs=specs)


def test_ai2d_content_address_requires_digest_as_exact_component() -> None:
    candidate = _candidate("ai2d_rst", "diagram", "diagram-a")
    item = candidate["inputs"][0]
    item["uri"] = f"artifact+sha256://evil.example/objects/{item['sha256']}"

    with pytest.raises(ValueError, match="exact authority component"):
        build_manifest([candidate], specs=_specs_for("ai2d_rst", {"diagram": 1}))


def test_manifest_rejects_wrong_source_specific_roles() -> None:
    candidate = _candidate("varex", "Flat", "form-a")
    candidate["inputs"][1]["role"] = "another_200dpi_image"

    with pytest.raises(ValueError, match="input roles must be exactly"):
        build_manifest([candidate], specs=_specs_for("varex", {"Flat": 1}))


def test_manifest_requires_group_id_and_binds_it_to_physical_content() -> None:
    candidate = _candidate("mws_vision_bench", "document_parsing_ru", "qa-1")
    missing = copy.deepcopy(candidate)
    missing.pop("group_id")
    mismatched = copy.deepcopy(candidate)
    mismatched["group_id"] = "another-image"

    specs = _specs_for("mws_vision_bench", {"document_parsing_ru": 1})
    with pytest.raises(ValueError, match="group_id must be a non-empty string"):
        build_manifest([missing], specs=specs)
    with pytest.raises(ValueError, match="group_id must equal the physical content id"):
        build_manifest([mismatched], specs=specs)


@pytest.mark.parametrize(
    ("dataset", "stratum", "metadata_key"),
    [
        ("pubtables_v2", "pages_2", "annotation"),
        ("varex", "Flat", "ground_truth"),
        ("varex", "Flat", "source_record"),
        ("ai2d_rst", "diagram", "annotation"),
        ("mws_vision_bench", "document_parsing_ru", "answers"),
        ("mws_vision_bench", "document_parsing_ru", "source_record"),
    ],
)
def test_manifest_requires_source_specific_metadata(
    dataset: str, stratum: str, metadata_key: str
) -> None:
    candidate = _candidate(dataset, stratum, "sample")
    candidate["metadata"].pop(metadata_key)

    with pytest.raises(ValueError, match=f"metadata.{metadata_key}"):
        build_manifest([candidate], specs=_specs_for(dataset, {stratum: 1}))


@pytest.mark.parametrize("stratum", ["Flat", "Nested", "Table"])
def test_varex_requires_non_empty_schema_and_ground_truth(stratum: str) -> None:
    specs = _specs_for("varex", {stratum: 1})
    empty_schema = _candidate("varex", stratum, "empty-schema")
    empty_schema["metadata"]["schema"] = {}
    empty_ground_truth = _candidate("varex", stratum, "empty-gt")
    empty_ground_truth["metadata"]["ground_truth"] = {}

    with pytest.raises(ValueError, match="schema must be a non-empty object"):
        build_manifest([empty_schema], specs=specs)
    with pytest.raises(ValueError, match="ground_truth must be a non-empty object"):
        build_manifest([empty_ground_truth], specs=specs)


@pytest.mark.parametrize(
    ("stratum", "field", "message"),
    [
        ("Nested", "schema", "Nested schema requires"),
        ("Nested", "ground_truth", "Nested ground_truth requires"),
        ("Table", "schema", "Table schema requires"),
        ("Table", "ground_truth", "Table ground_truth requires"),
    ],
)
def test_varex_checks_top_level_shape_by_stratum(stratum: str, field: str, message: str) -> None:
    candidate = _candidate("varex", stratum, "wrong-shape")
    if field == "schema":
        candidate["metadata"][field] = {
            "type": "object",
            "properties": {"field": {"type": "string"}},
        }
    else:
        candidate["metadata"][field] = {"field": "not structured"}

    with pytest.raises(ValueError, match=message):
        build_manifest([candidate], specs=_specs_for("varex", {stratum: 1}))


def test_varex_manifest_fixes_canonical_schema_and_ground_truth_hashes() -> None:
    candidate = _candidate("varex", "Table", "form-a")
    manifest = build_manifest([candidate], specs=_specs_for("varex", {"Table": 1}))
    metadata = manifest["selected"][0]["metadata"]

    assert metadata["schema_canonical_sha256"] == _canonical_json_hash(candidate["metadata"]["schema"])
    assert metadata["ground_truth_canonical_sha256"] == _canonical_json_hash(
        candidate["metadata"]["ground_truth"]
    )


def test_mws_canonical_source_record_hash_binds_question_answers_and_task() -> None:
    dataset = "mws_vision_bench"
    stratum = "document_parsing_ru"
    candidate = _candidate(dataset, stratum, "qa-1")
    changed = copy.deepcopy(candidate)
    changed["metadata"]["question"] = "Changed question"

    first = build_manifest([candidate], specs=_specs_for(dataset, {stratum: 1}))
    second = build_manifest([changed], specs=_specs_for(dataset, {stratum: 1}))

    first_hash = first["selected"][0]["metadata"]["source_record_canonical_sha256"]
    second_hash = second["selected"][0]["metadata"]["source_record_canonical_sha256"]
    assert first_hash != second_hash


def test_full_pinned_corpus_builds_exact_86_units_and_138_inputs() -> None:
    manifest = build_manifest(_complete_candidates())

    assert manifest["summary"]["selected_units"] == 86
    assert manifest["summary"]["input_count"] == 138
    assert manifest["summary"]["by_dataset"] == {
        "ai2d_rst": {
            "selected_units": 12,
            "input_count": 12,
            "by_stratum": {"diagram": 12},
        },
        "mws_vision_bench": {
            "selected_units": 30,
            "input_count": 30,
            "by_stratum": {
                "document_parsing_ru": 6,
                "full_page_ocr_ru": 6,
                "key_information_extraction_ru": 6,
                "reasoning_vqa_ru": 6,
                "text_grounding_ru": 6,
            },
        },
        "pubtables_v2": {
            "selected_units": 14,
            "input_count": 36,
            "by_stratum": {"pages_2": 8, "pages_3": 4, "pages_4": 2},
        },
        "varex": {
            "selected_units": 30,
            "input_count": 60,
            "by_stratum": {"Flat": 10, "Nested": 10, "Table": 10},
        },
    }
    assert len({(item["dataset"], item["group_id"]) for item in manifest["selected"]}) == 86
    assert json.loads(json.dumps(manifest))["selection"]["algorithm"] == "sha256-nul-v1"
    assert manifest["verification_state"] == "metadata_only_unverified"
    assert manifest["selection"]["group_policy"].startswith("claimed_identity_only")
    assert manifest["materialization_requirements"] == {
        "verify_every_referenced_object_bytes_against_sha256": True,
        "verify_group_id_against_trusted_physical_identity": True,
        "reject_manifest_on_any_mismatch": True,
    }


def test_dataset_licenses_revisions_and_quotas_are_exactly_pinned() -> None:
    assert {
        dataset: {
            "source": spec.source,
            "revision": spec.revision,
            "license": spec.license,
            "quotas": dict(spec.quotas),
        }
        for dataset, spec in DATASETS.items()
    } == {
        "pubtables_v2": {
            "source": "kensho/PubTables-v2",
            "revision": "aa575e798cb00a296925e2086addb3e3fd9a1903",
            "license": "CDLA-Permissive-2.0",
            "quotas": {"pages_2": 8, "pages_3": 4, "pages_4": 2},
        },
        "varex": {
            "source": "ibm-research/VAREX",
            "revision": "2dfc3386a4567c7d56bf1abf4d12ff42afed27b6",
            "license": "CDLA-Permissive-2.0",
            "quotas": {"Flat": 10, "Nested": 10, "Table": 10},
        },
        "ai2d_rst": {
            "source": "thiippal/AI2D-RST",
            "revision": "76cf0f8fadd1f431545fa540c58ef8a24ea31335",
            "license": "CC-BY-4.0 AND CC-BY-SA-4.0",
            "quotas": {"diagram": 12},
        },
        "mws_vision_bench": {
            "source": "MTSAIR/MWS-Vision-Bench",
            "revision": "b8d473734b79343cac2b74f692a29ab191c7d11d",
            "license": "MIT",
            "quotas": {
                "document_parsing_ru": 6,
                "full_page_ocr_ru": 6,
                "key_information_extraction_ru": 6,
                "reasoning_vqa_ru": 6,
                "text_grounding_ru": 6,
            },
        },
    }
    assert DATASETS["ai2d_rst"].license_components == {
        "annotations": "CC-BY-4.0",
        "source_images": "CC-BY-SA-4.0",
    }


def test_manifest_rejects_overridden_source_or_license_metadata() -> None:
    dataset = "varex"
    candidate = _candidate(dataset, "Flat", "form-a")
    spec = replace(DATASETS[dataset], revision="main", license="Unknown")

    with pytest.raises(ValueError, match="must match the pinned spec"):
        build_manifest([candidate], specs={dataset: spec})
