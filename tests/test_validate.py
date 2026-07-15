from __future__ import annotations

from rag_app.pipeline.validate import extract_numbers, validate_numbers, validate_standards


def test_decimal_comma_equivalence() -> None:
    assert validate_numbers(
        "pressure 16.5 MPa at 120 C", "давление 16,5 МПа при 120 °C"
    ).ok


def test_fullwidth_decimal_equivalence() -> None:
    assert validate_numbers("压力１６．５兆帕", "давление 16,5 МПа").ok


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


def test_standard_prefixes_do_not_match_inside_words() -> None:
    for text in ("between 10 and 20", "when 3 pumps", "согласно пункту 5", "место 3"):
        assert validate_standards(text, "") == []


def test_real_standard_is_still_detected() -> None:
    assert validate_standards("Comply with ISO 9001.", "Соблюдать стандарт 9001.") == ["9001"]
