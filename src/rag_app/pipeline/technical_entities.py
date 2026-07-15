"""Детерминированная защита технических сущностей при машинном переводе.

Слой не пытается переводить или нормализовать содержимое сущностей: он заменяет
их стабильными плейсхолдерами перед вызовом модели и восстанавливает исходное
написание после. Это даёт обратимый режим ``enforce`` и безопасный для трафика
режим ``shadow``, в котором те же сущности только измеряются.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

EntityKind = Literal["formula", "standard", "measurement", "number"]

_CYRILLIC_STANDARD_PREFIX = (
    r"ГОСТ\s?Р|ГОСТ|ОСТ|СНиП|СТО|СП|ТУ|РД|ВСН|НПБ|ПБ|ФНП|СанПиН|ИСО|МЭК|АПИ"
)
_LATIN_STANDARD_PREFIX = r"ISO|IEC|API|ASTM|ASME|ANSI|DIN|EN|BS|NACE|NORSOK|UL|NFPA"

# Граница слева обязательна: без неё EN/ТУ/СТО находились внутри between,
# «пункту» и «место». Для коротких латинских префиксов регистр значим; для
# кириллицы сохраняем удобное нечувствительное сопоставление.
STANDARD_PATTERN = re.compile(
    rf"(?<!\w)(?:(?i:{_CYRILLIC_STANDARD_PREFIX})|{_LATIN_STANDARD_PREFIX})"
    rf"\s?[-—–]?\s?(\d[\d.\-—–/]*)"
)

_NUMBER_TEXT = r"[+\-−＋－]?\d+(?:[ \u00a0,，]\d{3})*(?:[.,，．]\d+)?"

# Сначала идут составные единицы и длинные варианты. Условие справа не даёт
# коротким m/s/A совпасть с началом обычного слова.
_UNIT = (
    r"(?:mm|cm|km|мкм|мм|см|км|m|м)(?:[²³23])?/(?:s|h|с|ч)"
    r"|kg/m(?:[²³23])|кг/м(?:[²³23])|m³/h|м³/ч"
    r"|MPa|kPa|GPa|Pa|МПа|кПа|ГПа|Па|兆帕|千帕|帕|psi|ksi|bar|бар|巴"
    r"|mmHg|мм\s*рт\.?\s*ст\."
    r"|°\s?[CFС]|deg\s?[CF]|摄氏度|华氏度"
    r"|kW|MW|W|кВт|МВт|Вт|千瓦|兆瓦|瓦|kV|V|mV|кВ|В|мВ|千伏|伏"
    r"|mA|kA|A|мА|кА|А|毫安|千安|安培"
    r"|kN|MN|N|кН|МН|Н|N[·⋅ ]?m|Н[·⋅ ]?м"
    r"|Hz|kHz|MHz|Гц|кГц|МГц|rpm|об/мин"
    r"|kg|mg|g|t|lb|lbs|oz|кг|мг|г|т|千克|公斤|毫克|克|吨"
    r"|mm|cm|km|µm|um|m|in|inch(?:es)?|ft|yd|мм|см|км|мкм|м|дюйм(?:а|ов)?"
    r"|毫米|厘米|千米|微米|米"
    r"|mL|ml|L|l|gal|bbl|boe|scf|mcf|л|мл|毫升|升"
    r"|s|ms|min|h|day(?:s)?|с|мс|мин|ч|сут|毫秒|秒|分钟|小时|天"
    r"|%|％|‰|ppm|ppb"
)
_MEASUREMENT_PATTERN = re.compile(
    rf"(?<![A-Za-zА-Яа-я0-9０-９_]){_NUMBER_TEXT}"
    rf"(?:\s*[-–—…]\s*{_NUMBER_TEXT})?\s*(?i:{_UNIT})(?![A-Za-zА-Яа-я0-9０-９_])"
)
_NUMBER_PATTERN = re.compile(_NUMBER_TEXT)

_FORMULA_PATTERNS = (
    re.compile(r"\$\$(?=\S)(?:.|\n)*?(?<=\S)\$\$"),
    re.compile(r"(?<!\\)\$(?=\S)[^$\n]+?(?<=\S)\$"),
    re.compile(r"\\\((?=\S).*?(?<=\S)\\\)"),
    re.compile(r"\\\[(?=\S)(?:.|\n)*?(?<=\S)\\\]"),
    # Короткие неделимитированные формулы: E=mc^2, Q=A*v. Ограничение длины
    # намеренно консервативно, чтобы не замораживать обычные предложения.
    re.compile(r"(?<!\w)[A-Za-zА-Яа-я][A-Za-zА-Яа-я0-9_]{0,9}\s*=\s*[A-Za-zА-Яа-я0-9][A-Za-zА-Яа-я0-9_*/^()+.\-]{1,40}"),
)

_TOKEN_PATTERN = re.compile(r"⟪DRG[A-Z]*_[A-Z]+⟫")


@dataclass(frozen=True)
class ProtectedEntity:
    token: str
    value: str
    kind: EntityKind


@dataclass(frozen=True)
class ProtectedText:
    text: str
    entities: tuple[ProtectedEntity, ...]

    @property
    def counts(self) -> dict[str, int]:
        counts = Counter(entity.kind for entity in self.entities)
        return {kind: counts[kind] for kind in sorted(counts)}


@dataclass(frozen=True)
class RestorationResult:
    text: str
    missing_tokens: tuple[str, ...] = ()
    duplicated_tokens: tuple[str, ...] = ()
    unknown_tokens: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not (self.missing_tokens or self.duplicated_tokens or self.unknown_tokens)


def _token_suffix(index: int) -> str:
    """0-based индекс в буквенный код без цифр: A..Z, AA..AZ, ..."""
    chars: list[str] = []
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(ord("A") + remainder))
    return "".join(reversed(chars))


def _candidate_spans(text: str) -> list[tuple[int, int, EntityKind]]:
    candidates: list[tuple[int, int, EntityKind, int]] = []
    priority = {"formula": 0, "standard": 1, "measurement": 2, "number": 3}
    for pattern in _FORMULA_PATTERNS:
        candidates.extend(
            (m.start(), m.end(), "formula", priority["formula"])
            for m in pattern.finditer(text)
        )
    candidates.extend(
        (m.start(), m.end(), "standard", priority["standard"])
        for m in STANDARD_PATTERN.finditer(text)
    )
    candidates.extend(
        (m.start(), m.end(), "measurement", priority["measurement"])
        for m in _MEASUREMENT_PATTERN.finditer(text)
    )
    candidates.extend(
        (m.start(), m.end(), "number", priority["number"])
        for m in _NUMBER_PATTERN.finditer(text)
    )

    # Более приоритетная/длинная сущность поглощает вложенные числа и единицы.
    selected: list[tuple[int, int, EntityKind]] = []
    occupied: list[tuple[int, int]] = []
    for start, end, kind, _rank in sorted(
        candidates, key=lambda item: (item[3], item[0], -(item[1] - item[0]))
    ):
        if any(start < used_end and end > used_start for used_start, used_end in occupied):
            continue
        selected.append((start, end, kind))
        occupied.append((start, end))
    return sorted(selected)


def protect_entities(text: str) -> ProtectedText:
    """Заменить формулы/стандарты/измерения/числа обратимыми плейсхолдерами."""
    namespace = "DRG"
    while f"⟪{namespace}_" in text:
        namespace += "X"
    spans = _candidate_spans(text)
    entities = tuple(
        ProtectedEntity(f"⟪{namespace}_{_token_suffix(index)}⟫", text[start:end], kind)
        for index, (start, end, kind) in enumerate(spans)
    )
    protected = text
    for (start, end, _kind), entity in reversed(list(zip(spans, entities, strict=True))):
        protected = protected[:start] + entity.token + protected[end:]
    return ProtectedText(protected, entities)


def restore_entities(translated: str, protected: ProtectedText) -> RestorationResult:
    """Восстановить сущности и зафиксировать потерю/дублирование плейсхолдеров."""
    expected = {entity.token for entity in protected.entities}
    present = Counter(_TOKEN_PATTERN.findall(translated))
    missing = tuple(sorted(token for token in expected if present[token] == 0))
    duplicated = tuple(sorted(token for token in expected if present[token] > 1))
    unknown = tuple(sorted(token for token in present if token not in expected))
    restored = translated
    for entity in protected.entities:
        restored = restored.replace(entity.token, entity.value)
    return RestorationResult(restored, missing, duplicated, unknown)


def audit_unconfirmed_entities(source: str, translated: str) -> dict[str, list[str]]:
    """Сущности источника, не подтверждённые дословно в результате (shadow-метрика)."""
    src = protect_entities(source)
    dst_values = Counter(
        (kind, translated[start:end]) for start, end, kind in _candidate_spans(translated)
    )
    missing: dict[str, list[str]] = {}
    for entity in src.entities:
        key = (entity.kind, entity.value)
        if dst_values[key]:
            dst_values[key] -= 1
        else:
            missing.setdefault(entity.kind, []).append(entity.value)
    return {kind: values for kind, values in sorted(missing.items())}
