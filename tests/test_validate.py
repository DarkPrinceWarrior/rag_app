from __future__ import annotations

from rag_app.pipeline.validate import extract_numbers, validate_numbers


def test_decimal_comma_equivalence() -> None:
    assert validate_numbers(
        "pressure 16.5 MPa at 120 C", "давление 16,5 МПа при 120 °C"
    ).ok


def test_thousands_separators() -> None:
    assert extract_numbers("1,000 bolts") == extract_numbers("1000 болтов")
    assert extract_numbers("1 000 000") == extract_numbers("1000000")


def test_missing_number_fails() -> None:
    r = validate_numbers("hold for 60 minutes at 23.6 MPa", "выдержать при 23,6 МПа")
    assert not r.ok
    assert r.missing == ["60"]


def test_distorted_number_fails() -> None:
    r = validate_numbers("thickness 48 mm", "толщина 84 мм")
    assert not r.ok
    assert "48" in r.missing


def test_extra_numbers_allowed() -> None:
    # «three specimens» → «3 образца»: лишняя цифра в переводе — не ошибка
    assert validate_numbers("three specimens of 27 J", "3 образца по 27 Дж").ok


def test_section_numbering() -> None:
    assert validate_numbers("4.1 Scope of ISO 15156", "4.1 Область применения ISO 15156").ok
