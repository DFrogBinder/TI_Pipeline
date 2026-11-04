from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image

from repeatability_figure_review.audit import EXPECTED_ANALYSIS_FIGURES
from repeatability_figure_review.report import render_pdf


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 180), "white").save(path)


def test_render_pdf_includes_every_original_figure_entry(tmp_path: Path) -> None:
    entries = []
    for index, relative in enumerate(EXPECTED_ANALYSIS_FIGURES, start=1):
        image_path = tmp_path / "figures" / f"{index:02d}.png"
        _image(image_path)
        entries.append(
            {
                "title": f"Original figure {index}",
                "image_path": str(image_path),
                "source_path": relative,
                "purpose": "Purpose text.",
                "encoding": "Encoding text.",
                "finding": "Finding text.",
                "audit": "Audit text.",
                "replacement": "Replacement text.",
            }
        )
    spec = {
        "title": "Figure guide",
        "subtitle": "Test package",
        "overview": ["Overview paragraph."],
        "paper_equivalence": [],
        "original_figures": entries,
        "revised_figures": [],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    pdf_path = tmp_path / "guide.pdf"

    render_pdf(spec_path, pdf_path)

    assert pdf_path.is_file() and pdf_path.stat().st_size > 10_000
    result = subprocess.run(
        ["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True
    )
    pages_line = next(line for line in result.stdout.splitlines() if line.startswith("Pages:"))
    assert int(pages_line.split(":", 1)[1]) >= 22

