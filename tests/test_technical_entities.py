from __future__ import annotations

import re

from rag_app.pipeline.technical_entities import (
    audit_unconfirmed_entities,
    protect_entities,
    restore_entities,
)


def test_protect_and_restore_all_entity_classes() -> None:
    source = "Pressure 16.5 MPa, ISO 9001, section 4 and formula $Q=A\\cdot v$."
    protected = protect_entities(source)

    assert protected.counts == {
        "formula": 1,
        "measurement": 1,
        "number": 1,
        "standard": 1,
    }
    assert not re.search(r"\d", protected.text)
    restored = restore_entities(protected.text.replace("Pressure", "Давление"), protected)
    assert restored.ok
    assert restored.text == "Давление 16.5 MPa, ISO 9001, section 4 and formula $Q=A\\cdot v$."


def test_restore_reports_missing_and_duplicated_placeholders() -> None:
    protected = protect_entities("Use 10 MPa and ISO 9001.")
    first, second = protected.entities
    malformed = f"{first.token} {first.token}"

    result = restore_entities(malformed, protected)

    assert not result.ok
    assert result.duplicated_tokens == (first.token,)
    assert result.missing_tokens == (second.token,)


def test_placeholder_namespace_avoids_source_collision() -> None:
    protected = protect_entities("Literal ⟪DRG_A⟫ and pressure 10 MPa.")

    assert protected.entities[0].token == "⟪DRGX_A⟫"
    assert restore_entities(protected.text, protected).text == "Literal ⟪DRG_A⟫ and pressure 10 MPa."


def test_shadow_audit_reports_changed_unit_but_accepts_preserved_standard() -> None:
    missing = audit_unconfirmed_entities(
        "Pressure 10 MPa per ISO 9001.",
        "Давление 10 МПа по ISO 9001.",
    )

    assert missing == {"measurement": ["10 MPa"]}


def test_chinese_measurement_with_fullwidth_digits_is_protected() -> None:
    protected = protect_entities("设计压力为１６．５兆帕，温度２０摄氏度。")

    assert protected.counts == {"measurement": 2}
    assert restore_entities(protected.text, protected).text == "设计压力为１６．５兆帕，温度２０摄氏度。"


def test_signed_measurements_are_protected_with_their_signs() -> None:
    source = "Design temperature −40 °C, test pressure +16 MPa, 最低温度－２０摄氏度。"

    protected = protect_entities(source)

    assert protected.counts == {"measurement": 3}
    assert "−40" not in protected.text
    assert "+16" not in protected.text
    assert "－２０" not in protected.text
    assert restore_entities(protected.text, protected).text == source
