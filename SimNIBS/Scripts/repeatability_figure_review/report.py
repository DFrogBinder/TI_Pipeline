"""Render the illustrated figure guide as a portable PDF."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def render_pdf(spec_path: Path, output_path: Path) -> None:
    """Render a JSON guide specification with the system ReportLab runtime."""

    spec_path = Path(spec_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "/usr/bin/python3",
            str(Path(__file__).resolve()),
            "--worker",
            str(spec_path),
            str(output_path),
        ],
        check=True,
    )


def _worker(spec_path: Path, output_path: Path) -> None:
    import html

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    regular_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", regular_font))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", bold_font))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "DejaVu"
    styles["Title"].fontName = "DejaVu-Bold"
    styles["Heading1"].fontName = "DejaVu-Bold"
    styles["Heading2"].fontName = "DejaVu-Bold"
    styles.add(
        ParagraphStyle(
            "FigureTitle",
            parent=styles["Heading1"],
            fontName="DejaVu-Bold",
            fontSize=14,
            leading=17,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "Small",
            parent=styles["BodyText"],
            fontName="DejaVu",
            fontSize=8.5,
            leading=11,
        )
    )
    styles.add(
        ParagraphStyle(
            "CoverSubtitle",
            parent=styles["BodyText"],
            fontName="DejaVu",
            alignment=TA_CENTER,
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#44546A"),
        )
    )

    def safe(value: object) -> str:
        return html.escape(str(value)).replace("\n", "<br/>")

    def page_number(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("DejaVu", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawRightString(A4[0] - 0.55 * inch, 0.38 * inch, f"Page {document.page}")
        canvas.restoreState()

    def scaled_image(path: str, max_width: float = 7.0 * inch, max_height: float = 4.5 * inch) -> Image:
        width, height = ImageReader(path).getSize()
        scale = min(max_width / width, max_height / height)
        return Image(path, width=width * scale, height=height * scale)

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title=spec["title"],
        author="Repeatability figure review workflow",
    )
    story = [
        Spacer(1, 0.8 * inch),
        Paragraph(safe(spec["title"]), styles["Title"]),
        Spacer(1, 0.2 * inch),
        Paragraph(safe(spec.get("subtitle", "")), styles["CoverSubtitle"]),
        Spacer(1, 0.45 * inch),
    ]
    for paragraph in spec.get("overview", []):
        story.extend([Paragraph(safe(paragraph), styles["BodyText"]), Spacer(1, 0.12 * inch)])
    story.append(PageBreak())

    story.append(Paragraph("Current-paper figure equivalence", styles["Heading1"]))
    table_data = [["Paper figure", "Status", "Companion output", "Interpretation"]]
    for row in spec.get("paper_equivalence", []):
        table_data.append(
            [
                Paragraph(safe(row["paper_figure"]), styles["Small"]),
                Paragraph(safe(row["status"]), styles["Small"]),
                Paragraph(safe(row["companion_path"]), styles["Small"]),
                Paragraph(safe(row["rationale"]), styles["Small"]),
            ]
        )
    table = Table(table_data, colWidths=[1.35 * inch, 1.05 * inch, 1.65 * inch, 2.75 * inch], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB4BE")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8FA")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([table, PageBreak()])

    story.append(Paragraph("Original downloaded analysis figures", styles["Heading1"]))
    story.append(
        Paragraph(
            "Each entry describes what the source figure encodes, what it shows, why it may be misleading, and which companion figure should be used instead.",
            styles["BodyText"],
        )
    )
    story.append(PageBreak())
    for index, entry in enumerate(spec.get("original_figures", []), start=1):
        story.extend(
            [
                Paragraph(f"{index}. {safe(entry['title'])}", styles["FigureTitle"]),
                Paragraph(f"<b>Source:</b> {safe(entry['source_path'])}", styles["Small"]),
                Spacer(1, 0.08 * inch),
                scaled_image(entry["image_path"]),
                Spacer(1, 0.12 * inch),
                Paragraph(f"<b>Purpose:</b> {safe(entry['purpose'])}", styles["BodyText"]),
                Paragraph(f"<b>How to read it:</b> {safe(entry['encoding'])}", styles["BodyText"]),
                Paragraph(f"<b>Result:</b> {safe(entry['finding'])}", styles["BodyText"]),
                Paragraph(f"<b>Audit finding:</b> {safe(entry['audit'])}", styles["BodyText"]),
                Paragraph(f"<b>Recommended use:</b> {safe(entry['replacement'])}", styles["BodyText"]),
                PageBreak(),
            ]
        )

    story.append(Paragraph("Revised and paper-equivalent figures", styles["Heading1"]))
    story.append(PageBreak())
    for index, entry in enumerate(spec.get("revised_figures", []), start=1):
        story.extend(
            [
                Paragraph(f"R{index}. {safe(entry['title'])}", styles["FigureTitle"]),
                Paragraph(f"<b>Path:</b> {safe(entry['relative_path'])}", styles["Small"]),
                Spacer(1, 0.08 * inch),
                scaled_image(entry["image_path"]),
                Spacer(1, 0.12 * inch),
                Paragraph(safe(entry["description"]), styles["BodyText"]),
                PageBreak(),
            ]
        )

    document.build(story, onFirstPage=page_number, onLaterPages=page_number)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not args.worker:
        parser.error("report.py is an internal PDF worker")
    _worker(args.spec, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

