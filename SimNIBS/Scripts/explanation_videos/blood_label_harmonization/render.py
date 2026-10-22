"""Render the blood-label harmonization animation reproducibly."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


SCENE_NAME = "BloodLabelHarmonization"
PACKAGE_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output"

_QUALITY_SETTINGS = {
    "low": ("854,480", "15"),
    "final": ("1920,1080", "30"),
}


def manim_command(
    scene_file: Path,
    output_file: Path,
    *,
    quality: str,
) -> list[str]:
    """Build a shell-safe Manim invocation for one delivery profile."""

    try:
        resolution, fps = _QUALITY_SETTINGS[quality]
    except KeyError as exc:
        supported = ", ".join(_QUALITY_SETTINGS)
        raise ValueError(
            f"Unsupported quality {quality!r}; expected one of: {supported}"
        ) from exc

    return [
        sys.executable,
        "-m",
        "manim",
        "--renderer",
        "cairo",
        "--format",
        "mp4",
        "--resolution",
        resolution,
        "--fps",
        fps,
        "--media_dir",
        str(output_file.parent / ".manim"),
        "--output_file",
        str(output_file),
        "--disable_caching",
        "--progress_bar",
        "none",
        str(scene_file),
        SCENE_NAME,
    ]


def gif_command(mp4_file: Path, gif_file: Path) -> list[str]:
    """Build a compact, palette-optimised GIF conversion command."""

    if not mp4_file.is_file():
        raise FileNotFoundError(f"MP4 does not exist: {mp4_file}")

    palette_filter = (
        "fps=12,scale=960:-2:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=128:stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=3"
    )
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(mp4_file),
        "-an",
        "-vf",
        palette_filter,
        "-loop",
        "0",
        str(gif_file),
    ]


def render_environment(
    *,
    cache_dir: Path,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a render environment with import and font-cache paths pinned."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ if base_environment is None else base_environment)
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(SCRIPTS_ROOT) + (
        os.pathsep + python_path if python_path else ""
    )
    environment["XDG_CACHE_HOME"] = str(cache_dir)
    return environment


def _run(command: Sequence[str]) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    cache_dir = Path(
        os.environ.get("EXPLANATION_VIDEO_CACHE_DIR", "/tmp/explanation-videos-cache")
    )
    environment = render_environment(cache_dir=cache_dir)
    subprocess.run(
        list(command),
        check=True,
        cwd=SCRIPTS_ROOT,
        env=environment,
    )


def _recover_nested_manim_output(output_file: Path) -> None:
    """Copy Manim's nested output into the requested path if needed."""

    if output_file.is_file():
        return
    candidates = tuple(
        path
        for path in output_file.parent.rglob(output_file.name)
        if path != output_file and path.is_file()
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Manim did not create {output_file}; nested candidates: {candidates}"
        )
    shutil.copy2(candidates[0], output_file)


def render(
    *,
    quality: str,
    output_dir: Path,
    make_gif: bool,
) -> tuple[Path, Path | None]:
    """Render the MP4 and optionally derive the shareable GIF."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "blood_label_harmonization" if quality == "final" else "blood_label_harmonization_smoke"
    mp4_file = output_dir / f"{stem}.mp4"
    gif_file = output_dir / f"{stem}.gif" if make_gif else None
    scene_file = PACKAGE_DIR / "scene.py"

    _run(manim_command(scene_file, mp4_file, quality=quality))
    _recover_nested_manim_output(mp4_file)
    if mp4_file.stat().st_size == 0:
        raise RuntimeError(f"Manim created an empty MP4: {mp4_file}")

    if gif_file is not None:
        _run(gif_command(mp4_file, gif_file))
        if gif_file.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg created an empty GIF: {gif_file}")

    return mp4_file, gif_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality",
        choices=tuple(_QUALITY_SETTINGS),
        default="final",
        help="low is a fast smoke render; final is 1920x1080 at 30 fps",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"delivery directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-gif",
        action="store_true",
        help="render only the MP4",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mp4_file, gif_file = render(
        quality=args.quality,
        output_dir=args.output_dir,
        make_gif=not args.skip_gif,
    )
    print(f"MP4: {mp4_file}")
    if gif_file is not None:
        print(f"GIF: {gif_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
