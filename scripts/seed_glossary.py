"""Сид глоссария: базовая терминология нефтегаз/строительство/договоры.

Запуск: uv run python scripts/seed_glossary.py
Повторный запуск безопасен (upsert по en_term).
"""

from __future__ import annotations

import asyncio
import uuid

from sqlalchemy.dialects.postgresql import insert as pg_insert

from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.db.models import GlossaryTerm

TERMS: list[tuple[str, str, str]] = [
    ("pressure vessel", "сосуд под давлением", "нефтегаз"),
    ("sour service", "сероводородная среда", "нефтегаз"),
    ("design pressure", "расчётное давление", "нефтегаз"),
    ("design temperature", "расчётная температура", "нефтегаз"),
    ("maximum allowable working pressure", "максимально допустимое рабочее давление", "нефтегаз"),
    ("corrosion allowance", "припуск на коррозию", "нефтегаз"),
    ("hydrostatic testing", "гидравлическое испытание", "нефтегаз"),
    ("hydrostatic test", "гидравлическое испытание", "нефтегаз"),
    ("welding procedure specification", "технологическая карта сварки", "сварка"),
    ("weld joint", "сварное соединение", "сварка"),
    ("welded joint", "сварное соединение", "сварка"),
    ("heat affected zone", "зона термического влияния", "сварка"),
    ("radiographic examination", "радиографический контроль", "контроль"),
    ("impact testing", "испытание на ударный изгиб", "контроль"),
    ("carbon steel", "углеродистая сталь", "материалы"),
    ("stainless steel", "нержавеющая сталь", "материалы"),
    ("normalized condition", "нормализованное состояние", "материалы"),
    ("battery limits", "границы установки", "проектирование"),
    (
        "Overall Time Schedule for PD Development",
        "Общий график разработки проектной документации",
        "проектирование",
    ),
    (
        "Kick-off Meeting of PD Development",
        "установочное совещание по разработке проектной документации",
        "проектирование",
    ),
    (
        "CHINA KUNLUN CONTRACTING AND ENGINEERING COORPERATION",
        "КИТАЙСКАЯ ПРОЕКТНО-СТРОИТЕЛЬНАЯ КОМПАНИЯ «КУНЬЛУНЬ»",
        "организации",
    ),
    (
        "CHINA KUNLUN CONTRACTING AND ENGINEERING COOPERATION",
        "КИТАЙСКАЯ ПРОЕКТНО-СТРОИТЕЛЬНАЯ КОМПАНИЯ «КУНЬЛУНЬ»",
        "организации",
    ),
    (
        "CHINA KUNLUN CONTRACTING & ENGINEERING CORPORATION",
        "КИТАЙСКАЯ ПРОЕКТНО-СТРОИТЕЛЬНАЯ КОМПАНИЯ «КУНЬЛУНЬ»",
        "организации",
    ),
    ("Project Documentation", "проектная документация", "проектирование"),
    ("piping system", "трубопроводная система", "проектирование"),
    ("spiral wound gasket", "спирально-навитая прокладка", "трубопроводы"),
    ("flanged connection", "фланцевое соединение", "трубопроводы"),
    ("shell thickness", "толщина обечайки", "нефтегаз"),
    ("dew point", "точка росы", "нефтегаз"),
    ("gas processing facility", "установка подготовки газа", "нефтегаз"),
    ("construction site", "строительная площадка", "строительство"),
    ("technical specification", "техническое задание", "договоры"),
    ("scope of work", "объём работ", "договоры"),
    ("acceptance criteria", "критерии приёмки", "договоры"),
]


async def main() -> None:
    engine = create_engine()
    sessionmaker = create_sessionmaker(engine)
    async with sessionmaker() as session:
        for en, ru, domain in TERMS:
            stmt = (
                pg_insert(GlossaryTerm)
                .values(id=uuid.uuid4(), en_term=en, ru_term=ru, domain=domain)
                .on_conflict_do_update(
                    index_elements=[GlossaryTerm.en_term], set_={"ru_term": ru, "domain": domain}
                )
            )
            await session.execute(stmt)
        await session.commit()
    await engine.dispose()
    print(f"OK: {len(TERMS)} терминов")


if __name__ == "__main__":
    asyncio.run(main())
