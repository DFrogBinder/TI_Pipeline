from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from explanation_videos.blood_label_harmonization.render import (
    SCENE_NAME,
    gif_command,
    manim_command,
    render_environment,
)


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_final_manim_command_is_path_safe_and_pins_delivery_settings(
    tmp_path: Path,
) -> None:
    scene_file = tmp_path / "source with spaces" / "scene.py"
    output_file = tmp_path / "output with spaces" / "animation.mp4"

    command = manim_command(scene_file, output_file, quality="final")

    assert command[:3] == [sys.executable, "-m", "manim"]
    assert option_value(command, "--resolution") == "1920,1080"
    assert option_value(command, "--fps") == "30"
    assert option_value(command, "--format") == "mp4"
    assert option_value(command, "--output_file") == str(output_file)
    assert option_value(command, "--media_dir") == str(output_file.parent / ".manim")
    assert command[-2:] == [str(scene_file), SCENE_NAME]
    assert command.count(str(scene_file)) == 1
    assert command.count(str(output_file)) == 1


def test_low_quality_manim_command_uses_fast_smoke_settings(tmp_path: Path) -> None:
    command = manim_command(
        tmp_path / "scene.py",
        tmp_path / "smoke.mp4",
        quality="low",
    )

    assert option_value(command, "--resolution") == "854,480"
    assert option_value(command, "--fps") == "15"


def test_manim_command_rejects_unknown_quality(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported quality"):
        manim_command(
            tmp_path / "scene.py",
            tmp_path / "animation.mp4",
            quality="cinema",
        )


def test_gif_command_is_path_safe_and_uses_shareable_settings(tmp_path: Path) -> None:
    mp4_file = tmp_path / "source with spaces.mp4"
    mp4_file.write_bytes(b"placeholder")
    gif_file = tmp_path / "preview with spaces.gif"

    command = gif_command(mp4_file, gif_file)

    assert command[:4] == ["ffmpeg", "-y", "-i", str(mp4_file)]
    assert command[-1] == str(gif_file)
    assert "-an" in command
    video_filter = option_value(command, "-vf")
    assert "fps=12" in video_filter
    assert "scale=960:-2" in video_filter
    assert "palettegen" in video_filter
    assert "paletteuse" in video_filter


def test_gif_command_rejects_missing_mp4(tmp_path: Path) -> None:
    missing_mp4 = tmp_path / "missing.mp4"

    with pytest.raises(FileNotFoundError, match="MP4 does not exist"):
        gif_command(missing_mp4, tmp_path / "preview.gif")


def test_render_environment_provides_writable_font_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "font cache"

    environment = render_environment(
        cache_dir=cache_dir,
        base_environment={"PYTHONPATH": "/existing/python/path"},
    )

    assert cache_dir.is_dir()
    assert os.access(cache_dir, os.W_OK)
    assert environment["XDG_CACHE_HOME"] == str(cache_dir)
    assert environment["PYTHONPATH"].endswith(
        os.pathsep + "/existing/python/path"
    )
