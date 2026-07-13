from __future__ import annotations

import asyncio
import importlib.util
import json
import stat
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts" / "generate_private_rag_eval.py"
    spec = importlib.util.spec_from_file_location("generate_private_rag_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _chunk(
    module,
    *,
    doc: int,
    idx: int,
    text: str | None = None,
    owner: str = "owner-a",
    source_lang: str = "en",
    kind: str = "section",
    document_kind: str = "pdf_text",
):
    body = text or f"Общий технический параметр document {doc}, section {idx}, value {idx + 10}."
    return module.CorpusChunk(
        id=uuid.UUID(int=doc * 100 + idx + 1),
        document_id=uuid.UUID(int=doc + 1),
        filename=f"document-{doc}.pdf",
        idx=idx,
        kind=kind,
        heading_path=f"Раздел {idx}",
        page_start=idx,
        page_end=idx,
        text=body,
        source_text=body,
        owner_sub=owner,
        source_lang=source_lang,
        document_kind=document_kind,
        s3_key_original=f"private/{doc}.pdf",
        page_count=10,
        meta={},
    )


def test_loopback_guards_reject_external_endpoints() -> None:
    module = _module()

    assert module.require_loopback_url("http://127.0.0.1:8006/v1", name="test")
    assert module.require_loopback_url("http://[::1]:8006/v1", name="test")
    with pytest.raises(ValueError, match="loopback"):
        module.require_loopback_url("https://models.example/v1", name="test")
    with pytest.raises(ValueError, match="credential-free"):
        module.require_loopback_url("http://user:secret@localhost:8006/v1", name="test")
    with pytest.raises(ValueError, match="loopback"):
        module.require_loopback_database_url(
            "postgresql+asyncpg://rag:secret@db.internal:5432/rag_app"
        )


def test_source_plans_are_deterministic_and_stratified() -> None:
    module = _module()
    chunks = [_chunk(module, doc=doc, idx=idx) for doc in range(3) for idx in range(3)]

    first = module.plan_source_sets(chunks, seed=42, pool_per_stratum=4)
    second = module.plan_source_sets(chunks, seed=42, pool_per_stratum=4)

    assert first == second
    assert all(len(source.chunks) == 1 for source in first["single_hop"])
    assert all(
        source.chunks[0].document_id == source.chunks[1].document_id
        for source in first["multi_hop"]
    )
    assert all(
        source.chunks[0].document_id != source.chunks[1].document_id
        for source in first["cross_document"]
    )


def test_cross_document_planning_never_crosses_owner_scope() -> None:
    module = _module()
    chunks = [
        _chunk(module, doc=0, idx=0, owner="owner-a"),
        _chunk(module, doc=1, idx=0, owner="owner-a"),
        _chunk(module, doc=2, idx=0, owner="owner-b"),
    ]

    planned = module.plan_source_sets(chunks, seed=7, pool_per_stratum=20)

    assert planned["cross_document"]
    assert all(
        source.chunks[0].owner_sub == source.chunks[1].owner_sub
        for source in planned["cross_document"]
    )
    assert all(source.chunks[0].scope_id.startswith("scope-sha256:") for source in planned["cross_document"])


def test_language_schedule_is_deterministic_and_balanced() -> None:
    module = _module()

    first = module.language_schedule(20260713, 20)
    second = module.language_schedule(20260713, 20)
    counts = {language: first.count(language) for language in ("ru", "en", "zh")}

    assert first == second
    assert max(counts.values()) - min(counts.values()) <= 1


def test_case_variant_key_is_stable_and_slot_specific() -> None:
    module = _module()

    first = module.case_variant_key("single_hop", seed=7, attempt=100_000)
    assert first == module.case_variant_key("single_hop", seed=7, attempt=100_000)
    assert first != module.case_variant_key("single_hop", seed=7, attempt=100_001)
    assert first != module.case_variant_key("multi_hop", seed=7, attempt=100_000)
    assert len(first) == 16


def test_case_variant_directive_changes_fact_selection_after_resume() -> None:
    module = _module()

    initial = module.case_variant_directive(
        "cross_document", seed=2026071313, attempt=2_200_000
    )
    resumed = module.case_variant_directive(
        "cross_document", seed=2026071313, attempt=2_200_008
    )
    other_slot = module.case_variant_directive(
        "cross_document", seed=2026071313, attempt=2_300_008
    )

    assert initial != resumed
    assert resumed != other_slot
    assert "ВНУТРЕННЯЯ ДИРЕКТИВА" in resumed
    assert "кандидат номер" in resumed
    assert resumed == module.case_variant_directive(
        "cross_document", seed=2026071313, attempt=2_200_008
    )
    with pytest.raises(ValueError, match="non-negative"):
        module.case_variant_directive("cross_document", seed=1, attempt=-1)


def test_source_rotation_is_deterministic_and_slot_distributed() -> None:
    module = _module()

    assert module.rotated_source_indices(4, slot=0) == (0, 1, 2, 3)
    assert module.rotated_source_indices(4, slot=1) == (1, 2, 3, 0)
    assert module.rotated_source_indices(4, slot=5) == (1, 2, 3, 0)
    with pytest.raises(ValueError, match="positive"):
        module.rotated_source_indices(0, slot=0)


def test_continuation_window_draws_deterministically_from_full_pool() -> None:
    module = _module()
    sources = [
        module.SourceSet("single_hop", (_chunk(module, doc=index, idx=0),))
        for index in range(12)
    ]
    target = module.CaseTarget("ru")

    windows = [
        module.continuation_source_window(
            sources,
            target,
            seed=2026071313,
            stratum="single_hop",
            slot=slot,
        )
        for slot in range(4)
    ]

    assert all(len(window) == 8 for window in windows)
    assert windows[0] == module.continuation_source_window(
        sources,
        target,
        seed=2026071313,
        stratum="single_hop",
        slot=0,
    )
    first_eight = {source.chunks[0].document_id for source in sources[:8]}
    selected = {
        source.chunks[0].document_id for window in windows for source in window
    }
    assert selected - first_eight


def test_unique_registry_claim_is_atomic_across_parallel_strata() -> None:
    module = _module()
    registry = module.UniqueCaseRegistry()
    record = SimpleNamespace(case_id="case-a", question_sha256="question-a")

    async def claim_after_yield():
        await asyncio.sleep(0)
        return registry.claim(record)

    async def run_claims():
        return await asyncio.gather(claim_after_yield(), claim_after_yield())

    results = asyncio.run(run_claims())

    assert results.count(None) == 1
    assert results.count("duplicate_case") == 1


def test_retry_budget_and_continuation_schedule_after_eight_duplicate_rejects() -> None:
    module = _module()

    assert module.retry_limit_per_source(source_count=1, max_attempts=48) == 48
    assert module.retry_limit_per_source(source_count=8, max_attempts=48) == 8
    assert module.retry_limit_per_source(source_count=8, max_attempts=4) == 4
    assert module.generation_attempt_schedule(
        3,
        slot=1,
        next_attempts=(8, 8, 8),
        max_attempts=10,
    ) == (
        (1, 8),
        (2, 8),
        (0, 8),
        (1, 9),
        (2, 9),
        (0, 9),
    )
    assert module.generation_attempt_schedule(
        3,
        slot=0,
        next_attempts=(10, 10, 10),
        max_attempts=10,
    ) == ()
    with pytest.raises(ValueError, match="positive"):
        module.retry_limit_per_source(source_count=0, max_attempts=4)
    with pytest.raises(ValueError, match="match source_count"):
        module.generation_attempt_schedule(
            2, slot=0, next_attempts=(8,), max_attempts=10
        )


def test_duplicate_checkpoint_slots_are_invalidated_deterministically() -> None:
    module = _module()
    targets = [
        SimpleNamespace(key=f"single_hop-{index:04d}") for index in range(3)
    ]
    slots = (
        SimpleNamespace(
            target=targets[0],
            record=SimpleNamespace(case_id="case-a", question_sha256="question-a"),
        ),
        SimpleNamespace(
            target=targets[1],
            record=SimpleNamespace(case_id="case-a", question_sha256="question-a"),
        ),
        SimpleNamespace(
            target=targets[2],
            record=SimpleNamespace(case_id="case-b", question_sha256="question-b"),
        ),
    )

    class FakeCheckpoint:
        invalidated = []

        def iter_slots(self):
            return slots

        def invalidate_slot(self, target, *, reason_code):
            self.invalidated.append((target.key, reason_code))

    checkpoint = FakeCheckpoint()

    assert module.invalidate_duplicate_checkpoint_slots(checkpoint) == 1
    assert checkpoint.invalidated == [(targets[1].key, "duplicate_case")]


def test_overlay_merge_is_exact_disjoint_and_unique() -> None:
    module = _module()
    base_target = SimpleNamespace(
        key="single_hop-0000", stratum="single_hop", slot=0
    )
    continuation_target = SimpleNamespace(
        key="single_hop-0001", stratum="single_hop", slot=1
    )
    base_slot = SimpleNamespace(
        target=base_target,
        record=SimpleNamespace(case_id="case-a", question_sha256="question-a"),
    )
    continuation_slot = SimpleNamespace(
        target=continuation_target,
        record=SimpleNamespace(case_id="case-b", question_sha256="question-b"),
    )

    class FakeStore:
        def __init__(self, slots):
            self._slots = slots

        def iter_slots(self):
            return tuple(self._slots)

    merged = module.merge_overlay_slots(
        FakeStore([base_slot]),
        FakeStore([continuation_slot]),
        {base_target.key: base_target, continuation_target.key: continuation_target},
        {continuation_target.key: continuation_target},
    )

    assert merged == (base_slot, continuation_slot)
    with pytest.raises(module.CheckpointError, match="overlap"):
        module.merge_overlay_slots(
            FakeStore([base_slot]),
            FakeStore([base_slot]),
            {base_target.key: base_target},
            {base_target.key: base_target},
        )


def test_english_script_repair_rejects_foreign_and_transliterated_prose() -> None:
    module = _module()
    scalar = module.normalize_english_answer("16 MPa")
    assert scalar == "The resulting answer is 16 MPa."
    assert module.answer_matches_language(scalar, "en")
    assert not module.answer_matches_language("16 MPa", "en")
    assert not module.answer_matches_language("42", "en")
    assert not module.answer_matches_language("ГОСТ", "en")
    assert not module.answer_matches_language("中文", "en")
    assert not module.text_matches_language(
        "Raschetnoe davlenie sostavlyaet shestnadtsat megapaskalei.", "en"
    )
    assert module.has_forbidden_english_script("Extended Cyrillic Ӂ remains")
    assert module.has_forbidden_english_script("CJK compatibility \uf900 remains")
    assert not module.text_matches_language("English words remain 中文", "en")
    quantities = module._quantities(
        "First source: pressure is 16 MPa. Second source: the material is steel."
    )
    assert all(item["value"] not in {"1", "2"} for item in quantities)
    passage_quantities = module._quantities(
        "First passage: pressure is 16 MPa. Second passage: the material is steel."
    )
    assert passage_quantities == quantities
    assert module.text_matches_language("Каково расчетное давление?", "ru")
    assert module.text_matches_language("What is the specified design pressure?", "en")
    assert module.text_matches_language("规定的设计压力是多少？", "zh")


def test_case_targets_are_availability_aware_and_cover_trial_classes() -> None:
    module = _module()
    chunks = [
        _chunk(module, doc=0, idx=0, text="Technical shared narrative with value 10."),
        _chunk(module, doc=0, idx=1, text="Technical shared pressure is 16 MPa under ISO 9001."),
        _chunk(module, doc=1, idx=0, text="Technical | shared | 20", kind="table"),
        _chunk(module, doc=1, idx=1, text="Technical shared x = 20 MPa"),
        _chunk(module, doc=2, idx=0, text="Technical shared figure caption 30", kind="image"),
        _chunk(
            module,
            doc=3,
            idx=0,
            text="Technical shared scanned sheet 40",
            kind="image",
            document_kind="pdf_scan",
        ),
        _chunk(module, doc=3, idx=1, text="Technical shared scan companion section 50"),
    ]
    strata = ("single_hop", "multi_hop", "cross_document", "no_answer")
    plans = module.plan_source_sets(chunks, seed=11, pool_per_stratum=100)

    targets = module.build_case_targets(11, strata, 4, plans)
    flattened = [target for stratum in strata for target in targets[stratum]]

    assert {target.content_type for target in flattened if target.content_type} == {
        "text",
        "table",
        "formula",
        "figure",
        "scan",
    }
    assert {target.challenge_tag for target in flattened if target.challenge_tag} >= {
        "numbers",
        "units",
        "standards",
        "prompt_injection",
        "leakage",
    }
    assert all(
        any(module.source_matches_target(source, target) for source in plans[stratum])
        for stratum in strata
        for target in targets[stratum]
    )

    roomy_targets = module.build_case_targets(11, strata, 5, plans)
    assert {
        target.content_type
        for target in roomy_targets["single_hop"]
        if target.content_type is not None
    } == {"text", "table", "formula", "figure", "scan"}
    assert all(
        target.content_type is None
        for stratum in ("multi_hop", "cross_document", "no_answer")
        for target in roomy_targets[stratum]
    )


def test_positive_payload_requires_exact_quote_and_all_labels() -> None:
    module = _module()
    chunks = (
        _chunk(module, doc=0, idx=0, text="Давление составляет ровно 16,5 МПа при 20 °C."),
        _chunk(module, doc=0, idx=1, text="Испытание длится не менее 120 минут."),
    )
    source = module.SourceSet("multi_hop", chunks)
    payload = {
        "question": "Каковы давление и минимальная длительность испытания?",
        "answer": "16,5 МПа и 120 минут.",
        "evidence": [
            {"label": "E1", "supporting_quote": "Давление составляет ровно 16,5 МПа"},
            {"label": "E2", "supporting_quote": "Испытание длится не менее 120 минут"},
        ],
    }

    question, answer, evidence = module.validate_positive_payload(payload, source)
    assert question == payload["question"]
    assert answer == payload["answer"]
    assert [item["label"] for item in evidence] == ["E1", "E2"]

    payload["evidence"][1]["supporting_quote"] = "Испытание длится примерно 120 минут"
    with pytest.raises(ValueError, match="exact source substring"):
        module.validate_positive_payload(payload, source)


def test_deterministic_evidence_uses_exact_prompt_excerpt() -> None:
    module = _module()
    chunk = _chunk(
        module,
        doc=0,
        idx=0,
        text="Рабочее давление составляет 16,5 МПа. Испытание продолжается 120 минут.",
    )
    evidence = module.deterministic_evidence(
        module.SourceSet("single_hop", (chunk,)), max_chars=40
    )

    assert evidence == [
        {
            "label": "E1",
            "supporting_quote": chunk.text[:40].strip(),
        }
    ]
    context = module._context_block((chunk,), max_chars=40)
    assert context == f"[E1]\nTEXT:\n{evidence[0]['supporting_quote']}"
    assert str(chunk.id) not in context
    assert str(chunk.document_id) not in context
    assert chunk.filename not in context
    assert chunk.heading_path not in context


def test_generated_case_binds_strict_gold_and_sidecar_contracts() -> None:
    module = _module()
    chunk = _chunk(
        module,
        doc=0,
        idx=0,
        text="Расчетное давление корпуса составляет шестнадцать мегапаскалей.",
        owner="owner-a",
        source_lang="ru",
    )
    snapshot = module.DocumentSnapshot(
        document_ref=f"doc-sha256:{'a' * 64}",
        source_sha256="a" * 64,
        parsed_content_sha256="b" * 64,
        page_count=10,
    )
    generator = module.PrivateRagGenerator(
        None,
        model="local-test-model",
        base_url="http://127.0.0.1:8006/v1",
        seed=17,
        concurrency=1,
        corpus=[chunk],
        snapshots={chunk.document_id: snapshot},
    )

    async def fake_completion(_system, user, *, call_seed, max_tokens=1200):
        del call_seed, max_tokens
        if "Rewrite the question and answer" in user:
            return {
                "question": "设备外壳的设计压力是多少？",
                "answer": "十六兆帕。",
            }
        if "response_schema" in user:
            return {
                "answer_supported": True,
                "question_unambiguous": True,
                "uses_all_evidence": True,
            }
        return {
            "question": "Каково расчетное давление корпуса?",
            "answer": "Шестнадцать мегапаскалей.",
            "evidence": [
                {
                    "label": "E1",
                    "supporting_quote": "Расчетное давление корпуса составляет шестнадцать мегапаскалей",
                }
            ],
        }

    generator._json_completion = fake_completion
    try:
        generated = asyncio.run(
            generator.generate_positive(
                module.SourceSet("single_hop", (chunk,)),
                attempt=0,
                target=module.CaseTarget("zh"),
            )
        )
    finally:
        asyncio.run(generator.close())

    gold = module.GoldRecord.model_validate_json(json.dumps(generated.record), strict=True)
    sidecar = module.PrivateSidecarRecord.model_validate_json(
        json.dumps(generated.metadata), strict=True
    )
    assert module.bind_gold_sidecar([gold], [sidecar])[gold.case_id] == sidecar
    assert sidecar.scope_id == module.make_scope_id("owner-a")
    assert sidecar.exact_evidence[0].exact_quote in chunk.text
    assert gold.language == "zh"
    assert module.text_matches_language(gold.question, "zh")


def test_generate_stratum_resumes_an_accepted_slot_without_model_call(
    tmp_path: Path,
) -> None:
    module = _module()
    chunk = _chunk(
        module,
        doc=0,
        idx=0,
        text="Расчетное давление корпуса составляет шестнадцать мегапаскалей.",
        owner="owner-a",
        source_lang="ru",
    )
    snapshot = module.DocumentSnapshot(
        document_ref=f"doc-sha256:{'a' * 64}",
        source_sha256="a" * 64,
        parsed_content_sha256="b" * 64,
        page_count=10,
    )
    generator = module.PrivateRagGenerator(
        None,
        model="local-test-model",
        base_url="http://127.0.0.1:8006/v1",
        seed=17,
        concurrency=1,
        corpus=[chunk],
        snapshots={chunk.document_id: snapshot},
    )

    async def fake_completion(_system, user, *, call_seed, max_tokens=1200):
        del call_seed, max_tokens
        if "response_schema" in user:
            return {
                "answer_supported": True,
                "question_unambiguous": True,
                "uses_all_evidence": True,
            }
        return {
            "question": "Каково расчетное давление корпуса?",
            "answer": "Шестнадцать мегапаскалей.",
        }

    generator._json_completion = fake_completion
    source = module.SourceSet("single_hop", (chunk,))
    case_target = module.CaseTarget("ru")
    checkpoint_target = module.SlotTarget(
        stratum="single_hop",
        slot=0,
        language="ru",
        source_plan_sha256=module.canonical_sha256(
            module._source_plan_payload([source])
        ),
        source_count=1,
    )
    identity = module.RunIdentity(
        seed=17,
        corpus_sha256="c" * 64,
        snapshots_sha256="d" * 64,
        plan_sha256="e" * 64,
        model="local-test-model",
        model_revision="test-revision",
        generator_contract_version="test-v1",
        per_stratum=1,
        min_chars=1,
        trial=True,
    )
    checkpoint = module.PrivateCheckpointStore.create(
        tmp_path / "checkpoint", identity, max_attempts=2
    )
    try:
        first, _ = asyncio.run(
            module.generate_stratum(
                generator,
                [source],
                target=1,
                max_attempts_per_source=2,
                case_targets=[case_target],
                checkpoint_targets=[checkpoint_target],
                checkpoint=checkpoint,
            )
        )
    finally:
        asyncio.run(generator.close())

    class ModelMustNotRun:
        seed = 17

        async def generate_positive(self, *_args, **_kwargs):
            raise AssertionError("model was called for a checkpointed slot")

        async def generate_no_answer(self, *_args, **_kwargs):
            raise AssertionError("model was called for a checkpointed slot")

    resumed, _ = asyncio.run(
        module.generate_stratum(
            ModelMustNotRun(),
            [source],
            target=1,
            max_attempts_per_source=2,
            case_targets=[case_target],
            checkpoint_targets=[checkpoint_target],
            checkpoint=checkpoint,
        )
    )

    assert resumed == first
    assert checkpoint.accepted_slots == 1
    slot_path = tmp_path / "checkpoint" / "slots" / "single_hop-0000.json"
    assert stat.S_IMODE(slot_path.stat().st_mode) == 0o600


def test_private_artifacts_are_atomic_and_owner_only(tmp_path: Path) -> None:
    module = _module()
    candidate = {
        "candidate_id": "one",
        "stratum": "single_hop",
        "question": "Вопрос?",
        "answer": "Ответ.",
        "evidence": [],
    }
    jsonl_path, generator_path, manifest_path = module.write_artifacts(
        tmp_path / "private",
        seed=7,
        records=[candidate],
        generator_metadata=[{"case_id": "one"}],
        manifest={"schema_version": 1},
        overwrite=False,
    )

    assert stat.S_IMODE((tmp_path / "private").stat().st_mode) == 0o700
    assert stat.S_IMODE(jsonl_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(generator_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    assert json.loads(jsonl_path.read_text().strip()) == candidate
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["gold_artifact_sha256"]) == 64
    assert len(manifest["generator_artifact_sha256"]) == 64
    with pytest.raises(FileExistsError):
        module.write_artifacts(
            tmp_path / "private",
            seed=7,
            records=[candidate],
            generator_metadata=[{"case_id": "one"}],
            manifest={"schema_version": 1},
            overwrite=False,
        )
