"""Числовая валидация перевода (roadmap § 3.4 п.3).

Дёшево и обязательно: все числа исходного сегмента должны присутствовать
в переводе. Нормализация: десятичная запятая → точка (16,5 ≡ 16.5),
разделители тысяч (1 000 / 1,000) → слитно.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# Число с возможными разделителями тысяч и десятичной частью.
_NUMBER = re.compile(r"\d+(?:[  ,]\d{3})*(?:[.,]\d+)?")


def extract_numbers(text: str) -> Counter[str]:
    """Мультимножество нормализованных чисел текста."""
    out: Counter[str] = Counter()
    for m in _NUMBER.finditer(text):
        raw = m.group(0)
        # убрать разделители тысяч (пробел/неразрывный пробел/запятая перед тройкой цифр)
        norm = re.sub(r"[  ,](?=\d{3}(?:\D|$))", "", raw)
        # десятичная запятая → точка
        norm = norm.replace(",", ".")
        # каноническая форма: без хвостовых нулей дробной части и ведущих нулей
        if "." in norm:
            norm = norm.rstrip("0").rstrip(".")
        norm = norm.lstrip("0") or "0"
        out[norm] += 1
    return out


@dataclass
class ValidationResult:
    ok: bool
    missing: list[str] = field(default_factory=list)  # есть в оригинале, нет в переводе
    extra: list[str] = field(default_factory=list)  # появились в переводе
    standards: list[str] = field(default_factory=list)  # потерянные обозначения стандартов (§4.3.5)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "missing": self.missing, "extra": self.extra, "standards": self.standards}


def validate_numbers(source: str, translated: str) -> ValidationResult:
    src = extract_numbers(source)
    dst = extract_numbers(translated)
    missing = sorted((src - dst).elements())
    extra = sorted((dst - src).elements())
    # «лишние» числа не валят проверку: перевод может легитимно расписать
    # числительное цифрой; критично только исчезновение/искажение исходных
    return ValidationResult(ok=not missing, missing=missing, extra=extra)


# ТЗ §4.3.5: обозначения стандартов (ГОСТ/ISO/API/ASTM/…) должны переноситься в
# перевод без искажения. Латиница и кириллица (ISO/ИСО, API/АПИ); сверяем по
# номеру стандарта рядом с префиксом — это ловит «де-обозначивание»
# («ISO 9001» → «стандарт качества»), не давая ложных срабатываний на обычных
# числах (их проверяет validate_numbers).
_STD_PREFIX = (
    r"ГОСТ\s?Р|ГОСТ|ОСТ|СНиП|СТО|СП|ТУ|РД|ВСН|НПБ|ПБ|ФНП|СанПиН"
    r"|ISO|ИСО|IEC|МЭК|API|АПИ|ASTM|ASME|ANSI|DIN|EN|BS|NACE|NORSOK|UL|NFPA"
)
_STANDARD = re.compile(rf"(?:{_STD_PREFIX})\s?[-—–]?\s?(\d[\d.\-—–/]*)", re.IGNORECASE)


def _std_numbers(text: str) -> Counter[str]:
    """Номера стандартов (нормализованные), привязанные к префиксу-обозначению."""
    out: Counter[str] = Counter()
    for m in _STANDARD.finditer(text):
        num = re.sub(r"[—–\s]", "-", m.group(1)).strip("-.")
        if num:
            out[num] += 1
    return out


def validate_standards(source: str, translated: str) -> list[str]:
    """Обозначения стандартов, потерянные/искажённые в переводе (есть в оригинале
    как «ПРЕФИКС НОМЕР», но номер не найден при префиксе в переводе)."""
    src = _std_numbers(source)
    dst = _std_numbers(translated)
    return sorted((src - dst).elements())
