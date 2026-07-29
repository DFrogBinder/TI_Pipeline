import csv
import hashlib
import json
from pathlib import Path

import pytest

from post import package_camcan_publication_inputs as publication
from post import build_camcan_supervisor_revision_figures as revision


def _write_csv(path: Path, columns: set[str], rows: int) -> None:
    ordered = sorted(columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        for index in range(rows):
            writer.writerow({column: index for column in ordered})


def configured_files(role: str) -> list[str]:
    return [
        "analysis_manifest.json",
        *publication.ROLE_CONFIG[role]["tables"].keys(),
    ]


def _write_role_inputs(root: Path, role: str) -> None:
    config = publication.ROLE_CONFIG[role]
    (root / "analysis_manifest.json").write_text(
        json.dumps(config["manifest_expectations"]),
        encoding="utf-8",
    )
    for name, table in config["tables"].items():
        _write_csv(root / name, table["columns"], table["rows"])


@pytest.mark.parametrize("role", ["cohort", "personalized"])
def test_package_is_self_verifying_and_contains_exact_renderer(tmp_path, role):
    results = tmp_path / role
    results.mkdir()
    _write_role_inputs(results, role)
    renderer = tmp_path / "renderer.py"
    renderer.write_text("print('exact renderer')\n", encoding="utf-8")

    payload = publication.package_results(role, results, renderer)

    assert payload["status"] == "complete"
    assert payload["package_role"] == role
    assert payload["expected_outputs"]["png_count"] == 32
    assert payload["expected_outputs"]["pdf_count"] == 32
    packaged_renderer = results / "publication_tools" / publication.RENDERER_NAME
    assert packaged_renderer.read_bytes() == renderer.read_bytes()
    assert payload["renderer"]["sha256"] == hashlib.sha256(
        renderer.read_bytes()
    ).hexdigest()
    assert (results / "README_PUBLICATION_FIGURES.md").is_file()
    assert (results / "publication_figure_input_manifest.json").is_file()
    assert (results / "publication_requirements.txt").is_file()

    checksum_lines = (
        results / "publication_inputs.sha256"
    ).read_text(encoding="utf-8").splitlines()
    assert len(checksum_lines) == len(configured_files(role)) + 2
    for line in checksum_lines:
        expected_hash, relative_path = line.split("  ", maxsplit=1)
        assert publication.sha256_file(results / relative_path) == expected_hash


def test_package_refuses_missing_figure_source_table(tmp_path):
    results = tmp_path / "cohort"
    results.mkdir()
    config = publication.ROLE_CONFIG["cohort"]
    (results / "analysis_manifest.json").write_text(
        json.dumps(config["manifest_expectations"]),
        encoding="utf-8",
    )
    renderer = tmp_path / "renderer.py"
    renderer.write_text("print('renderer')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing or empty"):
        publication.package_results("cohort", results, renderer)


def test_collectors_package_and_verify_before_archiving():
    root = Path(__file__).resolve().parents[1]
    collectors = [
        root / "cohort_pipeline" / "cohort_post_manuscript_collect.slurm",
        root
        / "cohort_pipeline"
        / "cohort_personalized_comparison_collect.slurm",
    ]
    for path in collectors:
        text = path.read_text(encoding="utf-8")
        assert "package_camcan_publication_inputs.py" in text
        assert "sha256sum -c publication_inputs.sha256" in text
        assert text.index("package_camcan_publication_inputs.py") < text.index(
            "tar -C"
        )


def test_packaged_output_contract_matches_renderer_captions():
    assert set(publication.EXPECTED_FIGURE_STEMS) == set(revision.captions())
