"""Генерация тестового текстового PDF (EN) для e2e-проверки этапа 1.

Критерий этапа 1 (roadmap § 11): 50-страничный текстовый PDF → DOCX < 10 мин.
Запуск: uv run python scripts/make_test_pdf.py [страниц] [выход.pdf]
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PARAGRAPHS = [
    "This specification defines the minimum technical requirements for the design, "
    "fabrication, inspection and testing of pressure vessels intended for sour service "
    "in accordance with ISO 15156 and ASME BPVC Section VIII Division 1. The maximum "
    "allowable working pressure shall not exceed 16.5 MPa at a design temperature of 120 °C.",
    "All welded joints shall be subject to 100% radiographic examination. Acceptance "
    "criteria shall comply with ASME Section V Article 2. Welders shall be qualified in "
    "accordance with ASME Section IX. The contractor shall submit welding procedure "
    "specifications for approval at least 30 days prior to the start of fabrication.",
    "Carbon steel materials shall conform to ASTM A516 Grade 70 normalized condition. "
    "Impact testing shall be performed at minus 46 °C with minimum absorbed energy of "
    "27 J average for three specimens. Hardness of the weld metal and heat affected zone "
    "shall not exceed 248 HV10 as required by NACE MR0175.",
    "The vendor shall provide hydrostatic testing of each vessel at 1.43 times the design "
    "pressure, held for a minimum of 60 minutes. Test water chloride content shall not "
    "exceed 50 ppm. After testing, vessels shall be completely drained and dried to a "
    "dew point of minus 40 °C before shipment to the construction site.",
    "Piping systems within the battery limits shall be designed in accordance with "
    "ASME B31.3 for a design life of 25 years. Corrosion allowance shall be 3 mm for "
    "carbon steel lines in wet hydrocarbon service and 1.5 mm for utility services. "
    "All flanged connections shall use spiral wound gaskets with 316L windings.",
]

SPEC_ROWS = [
    ["Item", "Parameter", "Value", "Unit"],
    ["1", "Design pressure", "16.5", "MPa"],
    ["2", "Design temperature", "120", "°C"],
    ["3", "Corrosion allowance", "3.0", "mm"],
    ["4", "Test pressure", "23.6", "MPa"],
    ["5", "Shell thickness", "48", "mm"],
]


def build_story_pdf(n_pages: int) -> bytes:
    """PDF в память — для нагрузочного теста (scripts/load_test.py)."""
    import io

    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=14, spaceAfter=8)
    story = [Paragraph("Load Test Specification", styles["Title"])]
    for page in range(1, n_pages + 1):
        story.append(Paragraph(f"Section {page}. Requirements for Unit {100 + page}", styles["Heading1"]))
        for j, text in enumerate(PARAGRAPHS):
            story.append(Paragraph(f"{page}.{j + 1} {text}", body))
        story.append(PageBreak())
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4).build(story)
    return buf.getvalue()


def main() -> None:
    n_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/test_50p.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=14, spaceAfter=8)

    story = [Paragraph("Technical Specification No. 3086-TS-001", styles["Title"]),
             Paragraph("Pressure Vessels and Piping for Gas Processing Facility", h2),
             Spacer(1, 12)]

    for page in range(1, n_pages + 1):
        story.append(Paragraph(f"Section {page}. Requirements for Unit {100 + page}", h1))
        for j, text in enumerate(PARAGRAPHS):
            story.append(Paragraph(f"{page}.{j + 1} {text}", body))
        if page % 5 == 0:
            story.append(Paragraph(f"Table {page // 5}. Design parameters summary", h2))
            t = Table(SPEC_ROWS, colWidths=[1.5 * cm, 6 * cm, 3 * cm, 2 * cm])
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]))
            story.append(t)
        story.append(PageBreak())

    SimpleDocTemplate(str(out), pagesize=A4).build(story)
    print(f"OK: {out} ({n_pages} стр.)")


if __name__ == "__main__":
    main()
