from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


qualification = _load_script("qualify_selective_citation_models")
_load_script("qualify_generated_citation_shadow")
quantity = _load_script("qualify_quantity_guard")


def _case() -> object:
    return qualification.QualificationCase(
        case_id="ragq-case-0001",
        answerable=True,
        language="ru",
        question="Какое давление?",
        positive_context=("Рабочее давление равно 5 МПа.",),
    )


class _Spec:
    def __init__(self, value: str, unit: str) -> None:
        self.value = value
        self.unit = unit

    def model_dump(self, *, mode: str) -> dict[str, str]:
        assert mode == "python"
        return {"value": self.value, "unit": self.unit}


def _sidecar() -> object:
    spec = _Spec("5", "MPa")
    return SimpleNamespace(quantities=SimpleNamespace(expected=(spec,), supported=(spec,)))


def test_repair_prompt_requires_evidence_bound_numbers() -> None:
    messages = quantity.repair_messages(_case(), "Черновик: 50 МПа.")

    assert "never invent replacement values" in messages[0]["content"]
    assert "Рабочее давление равно 5 МПа." in messages[1]["content"]
    assert "Черновик: 50 МПа." in messages[1]["content"]


def test_observation_is_content_free_and_guard_uses_exact_evidence() -> None:
    private_question = "Какое секретное давление?"
    private_answer = "Секретное давление 50 МПа."
    private_evidence = "Секретное давление 5 МПа."
    case = qualification.QualificationCase(
        case_id="ragq-case-0001",
        answerable=True,
        language="ru",
        question=private_question,
        positive_context=(private_evidence,),
    )
    item = quantity.GeneratedQuantityCase(
        case=case,
        primary_answer=private_answer,
        final_answer="Давление 5 МПа.",
        repair_attempted=True,
    )

    observation = quantity.build_observation(item, _sidecar())
    serialized = json.dumps(observation, ensure_ascii=False)

    assert observation["primary_guard"]["unsupported_value_count"] == 1
    assert observation["final_guard"]["unsupported_value_count"] == 0
    assert private_question not in serialized
    assert private_answer not in serialized
    assert private_evidence not in serialized


def test_decision_is_fail_closed_on_weak_repair_or_recall_regression() -> None:
    summary = {
        "empty_primary_count": 0,
        "empty_final_count": 0,
        "primary": {
            "unsupported_value_rate": 0.40,
            "unsafe_case_rate": 0.30,
            "mean_quantity_unit_recall": 0.90,
        },
        "final": {
            "unsupported_value_rate": 0.20,
            "unsafe_case_rate": 0.15,
            "mean_quantity_unit_recall": 0.70,
        },
    }

    decision = quantity.decide(
        summary,
        max_final_unsupported_rate=0.05,
        min_unsupported_reduction=0.75,
        max_final_unsafe_case_rate=0.05,
        max_recall_drop=0.01,
        request_errors=0,
    )

    assert decision["candidate_gate"] == "NO-GO"
    assert len(decision["blockers"]) == 4


def test_decision_allows_candidate_only_after_strong_repair() -> None:
    summary = {
        "empty_primary_count": 0,
        "empty_final_count": 0,
        "primary": {
            "unsupported_value_rate": 0.40,
            "unsafe_case_rate": 0.30,
            "mean_quantity_unit_recall": 0.90,
        },
        "final": {
            "unsupported_value_rate": 0.02,
            "unsafe_case_rate": 0.03,
            "mean_quantity_unit_recall": 0.90,
        },
    }

    decision = quantity.decide(
        summary,
        max_final_unsupported_rate=0.05,
        min_unsupported_reduction=0.75,
        max_final_unsafe_case_rate=0.05,
        max_recall_drop=0.01,
        request_errors=0,
    )

    assert decision["candidate_gate"] == "GO"
    assert decision["production_gate"] == "PENDING_IMPLEMENTATION_AND_SHADOW"
