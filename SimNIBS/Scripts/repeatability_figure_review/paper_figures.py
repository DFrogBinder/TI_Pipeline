"""Build the clarity-reviewed, image-only repeatability-paper handoff."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from .audit import audit_sources, sha256_file
from .build import tree_manifest
from .figures import render_revised_figures


PAPER_FIGURE_MAPPING = {
    "01_simnibs_v3/v3_repeatability.png": (
        "revised_figures/within_simnibs326/deterministic_repeatability.png"
    ),
    "02_simnibs_v4/v4_bootstrap_precision.png": (
        "paper_equivalents/figure_bootstrap_precision.png"
    ),
    "02_simnibs_v4/v4_legacy_mesh_element_count_variability.png": (
        "legacy_paper_figures/figure_mesh_element_count_variability.png"
    ),
    "02_simnibs_v4/v4_legacy_tissue_volume_deviations.png": (
        "legacy_paper_figures/figure_tissue_volume_deviations.png"
    ),
    "02_simnibs_v4/v4_ranking_uncertainty.png": (
        "paper_equivalents/figure_single_run_ranking_uncertainty.png"
    ),
    "02_simnibs_v4/v4_repeatability.png": (
        "paper_equivalents/figure_target_field_repeatability.png"
    ),
    "03_v3_vs_v4/comparison_central_values.png": (
        "revised_figures/comprehensive/central_values.png"
    ),
    "03_v3_vs_v4/comparison_paired_version_contrasts.png": (
        "revised_figures/comprehensive/paired_version_contrasts.png"
    ),
    "03_v3_vs_v4/comparison_remesh_distributions.png": (
        "revised_figures/cross_version/remesh_distribution_overlay.png"
    ),
    "03_v3_vs_v4/comparison_technical_variability.png": (
        "revised_figures/comprehensive/technical_sd_log_scale.png"
    ),
    "03_v3_vs_v4/comparison_v3_percentile_within_v4.png": (
        "revised_figures/cross_version/v3_percentile_location.png"
    ),
    "03_v3_vs_v4/comparison_version_condition_interaction.png": (
        "revised_figures/comprehensive/version_condition_interaction.png"
    ),
}
PAPER_FIGURE_PATHS = tuple(PAPER_FIGURE_MAPPING)

LEGACY_DESTINATIONS = {
    "02_simnibs_v4/v4_legacy_mesh_element_count_variability.png",
    "02_simnibs_v4/v4_legacy_tissue_volume_deviations.png",
}


def _verify_png(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Required paper figure is missing or empty: {path}")
    with Image.open(path) as image:
        image.verify()


def build_clarity_directory(
    source_root: Path,
    manuscript_root: Path,
    output_root: Path,
    *,
    bootstrap_iterations: int = 20_000,
    bootstrap_max_repeats: int = 40,
) -> dict[str, object]:
    """Build exactly the twelve approved PNGs without modifying either source."""

    source_root = Path(source_root).expanduser().resolve()
    manuscript_root = Path(manuscript_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Output root already exists: {output_root}")

    source_before = tree_manifest(source_root)
    manuscript_before = tree_manifest(manuscript_root)
    bundle = audit_sources(source_root, manuscript_root)

    with tempfile.TemporaryDirectory(prefix="repeatability-clarity-") as temporary:
        staging = Path(temporary) / "rendered"
        render_revised_figures(
            bundle,
            staging,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_max_repeats=bootstrap_max_repeats,
        )
        try:
            output_root.mkdir(parents=True)
            for destination_relative, source_relative in PAPER_FIGURE_MAPPING.items():
                source = staging / source_relative
                destination = output_root / destination_relative
                _verify_png(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                _verify_png(destination)

            actual = {
                path.relative_to(output_root).as_posix()
                for path in output_root.rglob("*")
                if path.is_file()
            }
            if actual != set(PAPER_FIGURE_PATHS):
                raise RuntimeError(
                    "Image-only handoff has an unexpected file set: "
                    f"expected {sorted(PAPER_FIGURE_PATHS)}, observed {sorted(actual)}"
                )
            for relative in LEGACY_DESTINATIONS:
                source_relative = PAPER_FIGURE_MAPPING[relative]
                if sha256_file(output_root / relative) != sha256_file(
                    staging / source_relative
                ):
                    raise RuntimeError(f"Legacy figure changed during copy: {relative}")
        except Exception:
            shutil.rmtree(output_root, ignore_errors=True)
            raise

    if tree_manifest(source_root) != source_before:
        raise RuntimeError("Downloaded analysis changed while building paper figures")
    if tree_manifest(manuscript_root) != manuscript_before:
        raise RuntimeError("Manuscript changed while building paper figures")

    return {
        "status": "complete",
        "output_root": str(output_root),
        "figure_count": len(PAPER_FIGURE_PATHS),
        "regenerated_figure_count": len(PAPER_FIGURE_PATHS)
        - len(LEGACY_DESTINATIONS),
        "legacy_figure_count": len(LEGACY_DESTINATIONS),
        "source_outputs_modified": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the clarity-reviewed repeatability-paper PNG directory."
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--manuscript-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    parser.add_argument("--bootstrap-max-repeats", type=int, default=40)
    args = parser.parse_args(argv)
    result = build_clarity_directory(
        args.source_root,
        args.manuscript_root,
        args.output_root,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_max_repeats=args.bootstrap_max_repeats,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
