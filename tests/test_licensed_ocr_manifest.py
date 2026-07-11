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

PUB_IMAGES = {
    "uri": "https://huggingface.co/datasets/kensho/PubTables-v2/resolve/aa575e798cb00a296925e2086addb3e3fd9a1903/PubTables-v2_Full-Documents_test_images.tar.gz",
    "sha256": "0d42821fb1dce5713a86c327bec5fabbe214bb5ebbd0cfc75cd2ef89b7c7230e",
    "format": "tar.gz",
}
PUB_TABLES = {
    "uri": "https://huggingface.co/datasets/kensho/PubTables-v2/resolve/aa575e798cb00a296925e2086addb3e3fd9a1903/PubTables-v2_Full-Documents_test_tables.tar.gz",
    "sha256": "dfd10e0dc4cb3e92d0f521e8a135e6e96094d8e90e130fb3fe25c9fa31b3a3de",
    "format": "tar.gz",
}
VAREX_SHARD = {
    "uri": "https://huggingface.co/datasets/ibm-research/VAREX/resolve/2dfc3386a4567c7d56bf1abf4d12ff42afed27b6/data/benchmark-00000-of-00004.parquet",
    "sha256": "f0328edd6242318f97eb85fdd63466ec6a9db1482b7edd3ddb92de2bc535e147",
    "format": "parquet",
}
AI2D_IMAGES = {
    "uri": "https://ai2-public-datasets.s3.us-west-2.amazonaws.com/diagrams/ai2d-all.zip",
    "sha256": "1a6b77eebb8b7dbdf76a0ba6ca76c2f97ce8f81d8ee33b06593aa722e54c4786",
    "format": "zip",
}
AI2D_ANNOTATIONS = {
    "uri": "https://www.kielipankki.fi/download/AI2D-RST/v1.1/ai2d-rst-v1-1.zip",
    "sha256": "eb11d67507e08eb9bfd0f5944da7ca32cfcffa13e119b04ac5054effa65a759a",
    "format": "zip",
}
MWS_RECORDS = {
    "uri": "https://huggingface.co/datasets/MTSAIR/MWS-Vision-Bench/resolve/e204166bde25f7dcaaffb9313b855de67b516e5d/metadata.jsonl",
    "sha256": "c234a569583858bfab13399169ec9951da12edf6d88a5cd4c4efae8a1fd4197d",
    "format": "jsonl",
}
MWS_TYPES = {
    "document_parsing_ru": "document parsing ru",
    "full_page_ocr_ru": "full-page OCR ru",
    "key_information_extraction_ru": "key information extraction ru",
    "reasoning_vqa_ru": "reasoning VQA ru",
    "text_grounding_ru": "text grounding ru",
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _archive(container: dict[str, str], member: str) -> dict[str, Any]:
    return {
        "kind": "archive_member",
        "container": copy.deepcopy(container),
        "member": member,
        "sha256": _sha(f"{container['uri']}#{member}"),
    }


def _parquet_field(row_index: int, field: str) -> dict[str, Any]:
    return {
        "kind": "parquet_field",
        "container": copy.deepcopy(VAREX_SHARD),
        "row_index": row_index,
        "field": field,
        "sha256": _sha(f"varex:{row_index}:{field}"),
    }


def _candidate(
    dataset: str,
    stratum: str,
    canonical_id: str,
    *,
    group_id: str | None = None,
) -> dict[str, Any]:
    physical_id = group_id or f"{dataset}-group-{canonical_id}"
    row_index = int(_sha(canonical_id)[:8], 16)
    if dataset == "pubtables_v2":
        page_indices = list(range(DATASETS[dataset].expected_inputs[stratum]))
        inputs = [
            {
                "role": f"page_{ordinal}",
                **_archive(
                    PUB_IMAGES,
                    f"Full Documents/test/images/{physical_id}_page_{page_index}.jpg",
                ),
            }
            for ordinal, page_index in enumerate(page_indices, start=1)
        ]
        metadata: dict[str, Any] = {
            "document_id": physical_id,
            "table_id": canonical_id,
            "page_indices": page_indices,
            "annotation": _archive(
                PUB_TABLES, f"Full Documents/test/tables/{physical_id}_tables.json"
            ),
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
        inputs = [
            {"role": "image_200dpi", **_parquet_field(row_index, "image")},
            {"role": "image_50dpi", **_parquet_field(row_index, "image_50dpi")},
        ]
        record = {
            "doc_id": physical_id,
            "ground_truth": ground_truth,
            "schema": schema,
            "split": stratum,
        }
        metadata = {
            "doc_id": physical_id,
            "schema": schema,
            "ground_truth": ground_truth,
            "source_record": {
                "kind": "parquet_row",
                "container": copy.deepcopy(VAREX_SHARD),
                "row_index": row_index,
                "fields": ["doc_id", "ground_truth", "schema", "split"],
                "sha256": _canonical_json_hash(record),
            },
        }
    elif dataset == "ai2d_rst":
        inputs = [
            {"role": "image", **_archive(AI2D_IMAGES, f"ai2d/images/{physical_id}.png")}
        ]
        metadata = {
            "diagram_id": physical_id,
            "annotation": _archive(
                AI2D_ANNOTATIONS,
                f"ai2d-rst-v1-1/json/ai2d-rst/{physical_id}.png.json",
            ),
        }
    elif dataset == "mws_vision_bench":
        image_path = physical_id if physical_id.startswith("images/") else f"images/{physical_id}.jpg"
        physical_id = image_path
        question = f"Question {canonical_id}"
        answers = [f"Answer {canonical_id}"]
        dataset_name = "MWS synthetic test fixture"
        record = {
            "answers": answers,
            "dataset_name": dataset_name,
            "file_name": image_path,
            "id": canonical_id,
            "question": question,
            "type": MWS_TYPES[stratum],
        }
        inputs = [
            {
                "role": "image",
                "kind": "direct",
                "uri": (
                    "https://huggingface.co/datasets/MTSAIR/MWS-Vision-Bench/resolve/"
                    f"{DATASETS[dataset].revision}/{image_path}"
                ),
                "sha256": _sha(f"mws:{image_path}"),
            }
        ]
        metadata = {
            "image_path": image_path,
            "task_type": stratum,
            "dataset_name": dataset_name,
            "question": question,
            "answers": answers,
            "source_record": {
                "kind": "jsonl_row",
                "container": copy.deepcopy(MWS_RECORDS),
                "row_index": row_index,
                "fields": ["answers", "dataset_name", "file_name", "id", "question", "type"],
                "sha256": _canonical_json_hash(record),
            },
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
    return [
        _candidate(dataset, stratum, f"{stratum}-{index:02d}")
        for dataset, spec in DATASETS.items()
        for stratum, quota in spec.quotas.items()
        for index in range(quota)
    ]


def test_selection_hash_uses_pinned_nul_separated_rule() -> None:
    assert _selection_hash("docragenslate-ocr-v1", "varex", "Flat", "1044") == (
        "38e2351bf9eb0bd6d0a776789e181737961b927fe9b76848c06266bd7333fa32"
    )


def test_matching_finds_feasible_selection_across_adversarial_group_conflicts() -> None:
    dataset = "mws_vision_bench"
    strata = ["document_parsing_ru", "full_page_ocr_ru", "key_information_extraction_ru"]
    specs = _specs_for(dataset, dict.fromkeys(strata, 1))
    candidates = [
        _candidate(dataset, strata[0], "a-0", group_id="group-1"),
        _candidate(dataset, strata[0], "a-1", group_id="group-2"),
        _candidate(dataset, strata[1], "b-0", group_id="group-2"),
        _candidate(dataset, strata[1], "b-1", group_id="group-3"),
        _candidate(dataset, strata[2], "c-0", group_id="group-1"),
        _candidate(dataset, strata[2], "c-1", group_id="group-2"),
    ]

    first = build_manifest(candidates, specs=specs)
    second = build_manifest(reversed(candidates), specs=specs)

    assert first == second
    assert first["summary"]["selected_units"] == 3
    assert len({item["group_id"] for item in first["selected"]}) == 3


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


@pytest.mark.parametrize(
    ("dataset", "stratum", "mutate", "message"),
    [
        (
            "pubtables_v2",
            "pages_2",
            lambda item: item["inputs"][0].__setitem__("member", "../escape.jpg"),
            "unsafe path segment",
        ),
        (
            "varex",
            "Flat",
            lambda item: item["inputs"][0]["container"].__setitem__(
                "uri", item["inputs"][0]["container"]["uri"].replace("00000", "99999")
            ),
            "pinned benchmark parquet shard",
        ),
        (
            "varex",
            "Flat",
            lambda item: item["inputs"][0].__setitem__("field", "thumbnail"),
            "must reference VAREX parquet field",
        ),
        (
            "ai2d_rst",
            "diagram",
            lambda item: item["inputs"][0]["container"].__setitem__("format", "tar.gz"),
            "exact pinned source container",
        ),
        (
            "varex",
            "Flat",
            lambda item: item["metadata"]["source_record"].__setitem__(
                "row_index", item["metadata"]["source_record"]["row_index"] + 1
            ),
            "bind the selected parquet row",
        ),
    ],
)
def test_manifest_rejects_adversarial_materialization_references(
    dataset: str, stratum: str, mutate: Any, message: str
) -> None:
    candidate = _candidate(dataset, stratum, "sample")
    mutate(candidate)
    with pytest.raises(ValueError, match=message):
        build_manifest([candidate], specs=_specs_for(dataset, {stratum: 1}))


def test_record_fields_must_be_nonempty_sorted_and_source_exact() -> None:
    candidate = _candidate("mws_vision_bench", "document_parsing_ru", "qa-1")
    candidate["metadata"]["source_record"]["fields"].reverse()
    with pytest.raises(ValueError, match="non-empty sorted list"):
        build_manifest(
            [candidate], specs=_specs_for("mws_vision_bench", {"document_parsing_ru": 1})
        )


def test_varex_source_record_hash_is_canonical_extracted_json() -> None:
    candidate = _candidate("varex", "Table", "form-a")
    candidate["metadata"]["source_record"]["sha256"] = _sha("wrong record")
    with pytest.raises(ValueError, match="canonical JSON"):
        build_manifest([candidate], specs=_specs_for("varex", {"Table": 1}))


def test_direct_mws_image_uses_exact_pinned_path_and_legacy_record() -> None:
    candidate = _candidate("mws_vision_bench", "document_parsing_ru", "qa-1")
    manifest = build_manifest(
        [candidate], specs=_specs_for("mws_vision_bench", {"document_parsing_ru": 1})
    )
    selected = manifest["selected"][0]
    assert selected["inputs"][0]["kind"] == "direct"
    assert selected["metadata"]["source_record"]["container"] == MWS_RECORDS


def test_source_specific_metadata_and_record_hashes_are_required() -> None:
    for dataset, stratum, key in (
        ("pubtables_v2", "pages_2", "annotation"),
        ("varex", "Flat", "source_record"),
        ("ai2d_rst", "diagram", "annotation"),
        ("mws_vision_bench", "document_parsing_ru", "source_record"),
    ):
        candidate = _candidate(dataset, stratum, "sample")
        candidate["metadata"].pop(key)
        with pytest.raises(ValueError, match=f"metadata.{key}"):
            build_manifest([candidate], specs=_specs_for(dataset, {stratum: 1}))


@pytest.mark.parametrize("stratum", ["Flat", "Nested", "Table"])
def test_varex_requires_non_empty_schema_and_ground_truth(stratum: str) -> None:
    candidate = _candidate("varex", stratum, "empty")
    candidate["metadata"]["schema"] = {}
    with pytest.raises(ValueError, match="schema must be a non-empty object"):
        build_manifest([candidate], specs=_specs_for("varex", {stratum: 1}))


def test_varex_nested_accepts_a_pinned_local_defs_object_reference() -> None:
    candidate = _candidate("varex", "Nested", "local-ref")
    schema = {
        "$defs": {
            "Person": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            }
        },
        "type": "object",
        "properties": {"person": {"$ref": "#/$defs/Person"}},
    }
    candidate["metadata"]["schema"] = schema
    record = {
        "doc_id": candidate["metadata"]["doc_id"],
        "ground_truth": candidate["metadata"]["ground_truth"],
        "schema": schema,
        "split": "Nested",
    }
    candidate["metadata"]["source_record"]["sha256"] = _canonical_json_hash(record)

    manifest = build_manifest([candidate], specs=_specs_for("varex", {"Nested": 1}))

    assert manifest["summary"]["selected_units"] == 1


def test_varex_table_accepts_an_array_nested_below_a_local_defs_reference() -> None:
    candidate = _candidate("varex", "Table", "nested-table")
    schema = {
        "$defs": {
            "Residents": {
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "array",
                        "items": {"type": "object", "properties": {}},
                    }
                },
            }
        },
        "type": "object",
        "properties": {"residents": {"$ref": "#/$defs/Residents"}},
    }
    ground_truth = {"residents": {"rows": [{"value": "1"}]}}
    candidate["metadata"]["schema"] = schema
    candidate["metadata"]["ground_truth"] = ground_truth
    record = {
        "doc_id": candidate["metadata"]["doc_id"],
        "ground_truth": ground_truth,
        "schema": schema,
        "split": "Table",
    }
    candidate["metadata"]["source_record"]["sha256"] = _canonical_json_hash(record)

    manifest = build_manifest([candidate], specs=_specs_for("varex", {"Table": 1}))

    assert manifest["summary"]["selected_units"] == 1


def test_varex_manifest_fixes_canonical_schema_and_ground_truth_hashes() -> None:
    candidate = _candidate("varex", "Table", "form-a")
    manifest = build_manifest([candidate], specs=_specs_for("varex", {"Table": 1}))
    metadata = manifest["selected"][0]["metadata"]
    assert metadata["schema_canonical_sha256"] == _canonical_json_hash(candidate["metadata"]["schema"])
    assert metadata["ground_truth_canonical_sha256"] == _canonical_json_hash(
        candidate["metadata"]["ground_truth"]
    )


def test_full_pinned_corpus_builds_exact_86_units_and_138_inputs() -> None:
    manifest = build_manifest(_complete_candidates())
    assert manifest["summary"]["selected_units"] == 86
    assert manifest["summary"]["input_count"] == 138
    assert manifest["summary"]["by_dataset"] == {
        "ai2d_rst": {"selected_units": 12, "input_count": 12, "by_stratum": {"diagram": 12}},
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
    assert manifest["verification_state"] == "metadata_only_unverified"
    assert manifest["materialization_requirements"][
        "verify_every_referenced_object_bytes_against_sha256"
    ] is True
    json.loads(json.dumps(manifest))


def test_dataset_licenses_and_revisions_are_exactly_pinned() -> None:
    assert DATASETS["ai2d_rst"].license == "CC-BY-4.0 AND CC-BY-SA-4.0"
    assert DATASETS["mws_vision_bench"].license == "MIT AND CC-BY-4.0"
    assert DATASETS["mws_vision_bench"].license_components == {
        "benchmark_code": "MIT",
        "source_assets": "CC-BY-4.0",
    }
    assert DATASETS["pubtables_v2"].revision == "aa575e798cb00a296925e2086addb3e3fd9a1903"
    assert DATASETS["varex"].revision == "2dfc3386a4567c7d56bf1abf4d12ff42afed27b6"


def test_manifest_rejects_overridden_license_metadata() -> None:
    dataset = "mws_vision_bench"
    candidate = _candidate(dataset, "document_parsing_ru", "qa-1")
    spec = replace(DATASETS[dataset], license="MIT")
    with pytest.raises(ValueError, match="must match the pinned spec"):
        build_manifest([candidate], specs={dataset: spec})


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_manifest_rejects_non_finite_json_values(non_finite: float) -> None:
    candidate = _candidate("varex", "Flat", "non-finite")
    candidate["metadata"]["ground_truth"]["field"] = non_finite

    with pytest.raises(ValueError, match="Out of range float values"):
        build_manifest([candidate], specs=_specs_for("varex", {"Flat": 1}))


def test_mws_selected_rows_must_be_physically_unique() -> None:
    first = _candidate("mws_vision_bench", "document_parsing_ru", "qa-a")
    second = _candidate("mws_vision_bench", "document_parsing_ru", "qa-b")
    second["metadata"]["source_record"]["row_index"] = first["metadata"]["source_record"][
        "row_index"
    ]

    with pytest.raises(ValueError, match="duplicate selected metadata reference"):
        build_manifest(
            [first, second],
            specs=_specs_for("mws_vision_bench", {"document_parsing_ru": 2}),
        )


@pytest.mark.parametrize("quota", [True, 11])
def test_custom_specs_cannot_exceed_or_bypass_pinned_quotas(quota: int) -> None:
    spec = replace(DATASETS["varex"], quotas={"Flat": quota}, expected_inputs={"Flat": 2})

    with pytest.raises(ValueError, match="quota"):
        build_manifest([], specs={"varex": spec})
