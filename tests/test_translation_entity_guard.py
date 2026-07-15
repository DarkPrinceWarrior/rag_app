from __future__ import annotations

import re

import pytest

from rag_app.config import settings
from rag_app.llm.client import SegmentContext
from rag_app.workers.tasks import _translate_validated


class _Translator:
    def __init__(self, transform):
        self.transform = transform
        self.calls: list[tuple[str, str | None]] = []

    async def translate(self, text, context, feedback=None):
        self.calls.append((text, feedback))
        return self.transform(text)


@pytest.mark.asyncio
async def test_entity_guard_enforce_restores_entities(monkeypatch) -> None:
    monkeypatch.setattr(settings, "translation_entity_guard_mode", "enforce")
    translator = _Translator(lambda text: text.replace("Pressure", "Давление"))

    translated, result = await _translate_validated(
        translator,
        "Pressure 16.5 MPa per ISO 9001 and $Q=A v$.",
        SegmentContext(source_lang="en", target_lang="ru"),
    )

    assert translated == "Давление 16.5 MPa per ISO 9001 and $Q=A v$."
    assert result.ok
    assert result.entity_guard == {
        "schema_version": 1,
        "mode": "enforce",
        "protected": {"formula": 1, "measurement": 1, "standard": 1},
        "protected_total": 3,
        "unconfirmed": {},
        "unconfirmed_total": 0,
        "unconfirmed_rate": 0.0,
        "placeholder_errors": 0,
    }
    assert "16.5" not in translator.calls[0][0]


@pytest.mark.asyncio
async def test_entity_guard_enforce_fails_closed_when_placeholder_is_lost(monkeypatch) -> None:
    monkeypatch.setattr(settings, "translation_entity_guard_mode", "enforce")
    translator = _Translator(lambda text: re.sub(r"⟪DRG_[A-Z]+⟫", "", text, count=1))

    _translated, result = await _translate_validated(
        translator,
        "Pressure 10 MPa.",
        SegmentContext(source_lang="en", target_lang="ru"),
    )

    assert not result.ok
    assert result.entity_guard is not None
    assert result.entity_guard["placeholder_errors"] == 1
    assert len(translator.calls) == 2
    assert translator.calls[1][1] is not None


@pytest.mark.asyncio
async def test_entity_guard_enforce_fails_closed_on_unknown_placeholder(monkeypatch) -> None:
    monkeypatch.setattr(settings, "translation_entity_guard_mode", "enforce")
    translator = _Translator(lambda text: text + " ⟪DRG_Z⟫")

    _translated, result = await _translate_validated(
        translator,
        "Pressure 10 MPa.",
        SegmentContext(source_lang="en", target_lang="ru"),
    )

    assert not result.ok
    assert result.entity_guard is not None
    assert result.entity_guard["placeholder_errors"] == 1


@pytest.mark.asyncio
async def test_entity_guard_shadow_measures_without_masking(monkeypatch) -> None:
    monkeypatch.setattr(settings, "translation_entity_guard_mode", "shadow")
    translator = _Translator(lambda text: text.replace("10 MPa", "10 МПа"))

    translated, result = await _translate_validated(
        translator,
        "Pressure 10 MPa.",
        SegmentContext(source_lang="en", target_lang="ru"),
    )

    assert translated == "Pressure 10 МПа."
    assert result.ok
    assert result.entity_guard is not None
    assert result.entity_guard["unconfirmed"] == {"measurement": 1}
    assert result.entity_guard["unconfirmed_rate"] == 1.0
    assert translator.calls[0][0] == "Pressure 10 MPa."


@pytest.mark.asyncio
async def test_translation_memory_enforce_exact_bypasses_model(monkeypatch) -> None:
    monkeypatch.setattr(settings, "translation_memory_mode", "enforce")
    translator = _Translator(lambda text: "MODEL:" + text)
    context = SegmentContext(
        source_lang="en",
        target_lang="ru",
        translation_memory_exact=("00000000-0000-0000-0000-000000000001", "Давление 10 MPa."),
    )

    translated, result = await _translate_validated(translator, "Pressure 10 MPa.", context)

    assert translated == "Давление 10 MPa."
    assert translator.calls == []
    assert result.ok
    assert result.translation_memory is not None
    assert result.translation_memory["origin"] == "exact"


@pytest.mark.asyncio
async def test_translation_memory_shadow_never_bypasses_model(monkeypatch) -> None:
    monkeypatch.setattr(settings, "translation_memory_mode", "shadow")
    translator = _Translator(lambda _text: "Модельный перевод 10 MPa.")
    context = SegmentContext(
        source_lang="en",
        target_lang="ru",
        translation_memory_exact=("00000000-0000-0000-0000-000000000001", "Память 10 MPa."),
    )

    translated, result = await _translate_validated(translator, "Pressure 10 MPa.", context)

    assert translated == "Модельный перевод 10 MPa."
    assert len(translator.calls) == 1
    assert result.translation_memory is not None
    assert result.translation_memory["origin"] == "model"
    assert result.translation_memory["exact_candidate"] is True


@pytest.mark.asyncio
async def test_translation_memory_rejects_invalid_exact_and_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(settings, "translation_memory_mode", "enforce")
    translator = _Translator(lambda _text: "Давление 10 MPa.")
    context = SegmentContext(
        source_lang="en",
        target_lang="ru",
        translation_memory_exact=("00000000-0000-0000-0000-000000000001", "Давление 11 MPa."),
    )

    translated, result = await _translate_validated(translator, "Pressure 10 MPa.", context)

    assert translated == "Давление 10 MPa."
    assert len(translator.calls) == 1
    assert result.ok
    assert result.translation_memory is not None
    assert result.translation_memory["exact_rejected"] is True
