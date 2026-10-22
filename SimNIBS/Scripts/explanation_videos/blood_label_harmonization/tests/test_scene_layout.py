from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="'audioop' is deprecated and slated for removal in Python 3.13",
    category=DeprecationWarning,
    module="pydub.utils",
)

from manim import config

from explanation_videos.blood_label_harmonization.scene import (
    final_comparison_header,
    opening_title,
)


def test_long_titles_fit_inside_the_video_frame() -> None:
    title = opening_title()
    final_title, _, _, _ = final_comparison_header()

    assert title.width <= config.frame_width - 0.8
    assert final_title.width <= config.frame_width - 0.8


def test_final_caveat_does_not_overlap_comparison_labels() -> None:
    _, caveat, before_label, after_label = final_comparison_header()

    assert caveat.get_bottom()[1] > before_label.get_top()[1] + 0.05
    assert caveat.get_bottom()[1] > after_label.get_top()[1] + 0.05
