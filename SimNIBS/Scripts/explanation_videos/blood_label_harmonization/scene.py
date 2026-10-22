"""Manim scene explaining local blood-label inheritance."""

from __future__ import annotations

from dataclasses import dataclass

from manim import (
    Arc,
    Arrow,
    Circle,
    Create,
    DOWN,
    Ellipse,
    FadeIn,
    FadeOut,
    GrowArrow,
    Indicate,
    LEFT,
    Line,
    ORIGIN,
    PI,
    RIGHT,
    RoundedRectangle,
    Scene,
    ShowPassingFlash,
    Text,
    Transform,
    UP,
    VGroup,
    VMobject,
    WHITE,
    YELLOW,
    config,
)

from explanation_videos.blood_label_harmonization.storyboard import (
    TISSUES,
    local_blood_replacement,
    storyboard,
)


BACKGROUND = "#111827"
PANEL = "#1F2937"
MUTED = "#B8C1CC"
ACCENT = "#7DD3FC"
DANGER = "#F87171"


@dataclass(slots=True)
class HeadDiagram:
    group: VGroup
    layers: dict[str, VMobject]
    vessels: dict[str, VGroup]
    muscles: VGroup


def tissue_colour(identifier: str) -> str:
    return TISSUES[identifier].colour


def vessel_segment(points: list[tuple[float, float, float]]) -> VGroup:
    main = VMobject(stroke_color=tissue_colour("blood"), stroke_width=10)
    main.set_points_as_corners(points)
    branches = VGroup()
    if len(points) >= 3:
        joint = points[1]
        branches.add(
            Line(
                joint,
                (joint[0] - 0.28, joint[1] + 0.30, 0),
                color=tissue_colour("blood"),
                stroke_width=7,
            ),
            Line(
                joint,
                (joint[0] + 0.32, joint[1] - 0.24, 0),
                color=tissue_colour("blood"),
                stroke_width=7,
            ),
        )
    return VGroup(main, branches)


def corrected_head(*, center=ORIGIN, scale: float = 1.0) -> HeadDiagram:
    """Build the explicit corrected-v4 tissue schematic."""

    layer_specs = (
        ("scalp", 5.20, 6.40),
        ("compact_bone", 4.68, 5.82),
        ("spongy_bone", 4.36, 5.48),
        ("csf", 4.02, 5.14),
        ("grey_matter", 3.66, 4.70),
        ("white_matter", 2.82, 3.72),
    )
    layers: dict[str, VMobject] = {}
    layer_group = VGroup()
    for identifier, width, height in layer_specs:
        shape = Ellipse(
            width=width,
            height=height,
            fill_color=tissue_colour(identifier),
            fill_opacity=0.92,
            stroke_color=WHITE,
            stroke_opacity=0.42,
            stroke_width=1.6,
        )
        layers[identifier] = shape
        layer_group.add(shape)

    eye = Circle(
        radius=0.24,
        fill_color=tissue_colour("eyes"),
        fill_opacity=1,
        stroke_color=WHITE,
        stroke_width=1.4,
    ).move_to((2.23, 0.62, 0))
    layers["eyes"] = eye

    muscles = VGroup(
        Arc(
            radius=2.42,
            start_angle=-PI / 3.2,
            angle=PI / 1.55,
            stroke_color=tissue_colour("muscle"),
            stroke_width=13,
        ).stretch(1.23, dim=1),
        Arc(
            radius=2.35,
            start_angle=PI / 1.45,
            angle=PI / 2.7,
            stroke_color=tissue_colour("muscle"),
            stroke_width=11,
        ).stretch(1.23, dim=1),
    )

    vessels = {
        "white_matter": vessel_segment(
            [(-0.75, -0.55, 0), (-0.18, -0.12, 0), (0.72, 0.42, 0)]
        ),
        "grey_matter": vessel_segment(
            [(0.92, 1.43, 0), (1.34, 1.16, 0), (1.55, 0.72, 0)]
        ),
        "csf": vessel_segment(
            [(-1.89, -0.48, 0), (-1.79, 0.05, 0), (-1.76, 0.72, 0)]
        ),
        "external_soft_tissue": vessel_segment(
            [(2.35, 0.02, 0), (2.48, -0.48, 0), (2.38, -1.12, 0)]
        ),
    }
    vessel_group = VGroup(*vessels.values())
    group = VGroup(layer_group, muscles, eye, vessel_group)
    group.scale(scale).move_to(center)
    return HeadDiagram(group=group, layers=layers, vessels=vessels, muscles=muscles)


def harmonized_head(*, center=ORIGIN, scale: float = 1.0) -> HeadDiagram:
    """Build the v3-compatible six-tissue schematic with local fills."""

    layer_specs = (
        ("external_soft_tissue", 5.20, 6.40),
        ("generic_bone", 4.55, 5.67),
        ("csf", 4.02, 5.14),
        ("grey_matter", 3.66, 4.70),
        ("white_matter", 2.82, 3.72),
    )
    layers: dict[str, VMobject] = {}
    layer_group = VGroup()
    for identifier, width, height in layer_specs:
        shape = Ellipse(
            width=width,
            height=height,
            fill_color=tissue_colour(identifier),
            fill_opacity=0.92,
            stroke_color=WHITE,
            stroke_opacity=0.42,
            stroke_width=1.6,
        )
        layers[identifier] = shape
        layer_group.add(shape)

    eye = Circle(
        radius=0.24,
        fill_color=tissue_colour("eyes"),
        fill_opacity=1,
        stroke_color=WHITE,
        stroke_width=1.4,
    ).move_to((2.23, 0.62, 0))
    layers["eyes"] = eye

    vessels = {
        "white_matter": vessel_segment(
            [(-0.75, -0.55, 0), (-0.18, -0.12, 0), (0.72, 0.42, 0)]
        ),
        "grey_matter": vessel_segment(
            [(0.92, 1.43, 0), (1.34, 1.16, 0), (1.55, 0.72, 0)]
        ),
        "csf": vessel_segment(
            [(-1.89, -0.48, 0), (-1.79, 0.05, 0), (-1.76, 0.72, 0)]
        ),
        "external_soft_tissue": vessel_segment(
            [(2.35, 0.02, 0), (2.48, -0.48, 0), (2.38, -1.12, 0)]
        ),
    }
    for enclosure, segment in vessels.items():
        segment.set_color(tissue_colour(local_blood_replacement(enclosure)))

    vessel_group = VGroup(*vessels.values())
    muscles = VGroup()
    group = VGroup(layer_group, eye, vessel_group)
    group.scale(scale).move_to(center)
    return HeadDiagram(group=group, layers=layers, vessels=vessels, muscles=muscles)


def caption_card(text: str) -> VGroup:
    caption = Text(text, font_size=28, color=WHITE, weight="MEDIUM")
    if caption.width > 12.2:
        caption.scale_to_fit_width(12.2)
    backing = RoundedRectangle(
        width=max(caption.width + 0.58, 5.2),
        height=0.72,
        corner_radius=0.14,
        fill_color=PANEL,
        fill_opacity=0.96,
        stroke_color="#475569",
        stroke_width=1.2,
    )
    return VGroup(backing, caption).move_to(DOWN * 3.48)


def opening_title() -> Text:
    title = Text(
        "From corrected v4 labels to a v3-compatible head model",
        font_size=38,
        color=WHITE,
        weight="BOLD",
    )
    if title.width > 13.2:
        title.scale_to_fit_width(13.2)
    return title.to_edge(UP, buff=0.26)


def final_comparison_header() -> tuple[Text, Text, VGroup, VGroup]:
    final_title = Text(
        "Local enclosure, not a global blood-to-tissue conversion",
        font_size=38,
        color=WHITE,
        weight="BOLD",
    )
    if final_title.width > 13.2:
        final_title.scale_to_fit_width(13.2)
    final_title.to_edge(UP, buff=0.28)
    caveat = Text(
        "Conceptual schematic - not subject-specific anatomy",
        font_size=17,
        color=MUTED,
    ).next_to(final_title, DOWN, buff=0.10)
    before_label = VGroup(
        Text("Corrected v4 labels", font_size=27, color=WHITE, weight="BOLD"),
        Text("blood + muscle + two bone classes", font_size=18, color=MUTED),
    ).arrange(DOWN, buff=0.08).move_to(LEFT * 3.45 + UP * 2.45)
    after_label = VGroup(
        Text("Harmonized six-tissue model", font_size=27, color=WHITE, weight="BOLD"),
        Text("local fills + generic bone + soft tissue", font_size=18, color=MUTED),
    ).arrange(DOWN, buff=0.08).move_to(RIGHT * 3.45 + UP * 2.45)
    return final_title, caveat, before_label, after_label


def legend(identifiers: tuple[str, ...]) -> VGroup:
    rows = VGroup()
    for identifier in identifiers:
        swatch = RoundedRectangle(
            width=0.32,
            height=0.25,
            corner_radius=0.05,
            fill_color=tissue_colour(identifier),
            fill_opacity=1,
            stroke_color=WHITE,
            stroke_width=0.8,
        )
        label = Text(TISSUES[identifier].display_name, font_size=20, color=WHITE)
        rows.add(VGroup(swatch, label).arrange(RIGHT, buff=0.18))
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.16)
    rows.move_to((3.35, 0.15, 0)).align_to((1.65, 0, 0), LEFT)
    return rows


def mapping_card(text: str, *, y: float) -> tuple[VGroup, VGroup]:
    backing = RoundedRectangle(
        width=3.65,
        height=1.03,
        corner_radius=0.14,
        fill_color=PANEL,
        fill_opacity=1,
        stroke_color="#64748B",
        stroke_width=1.2,
    ).move_to((3.35, y, 0))
    label = Text(text, font_size=27, color=WHITE).move_to(backing)
    cross = VGroup(
        Line(LEFT * 0.31 + UP * 0.31, RIGHT * 0.31 + DOWN * 0.31),
        Line(LEFT * 0.31 + DOWN * 0.31, RIGHT * 0.31 + UP * 0.31),
    ).set_stroke(DANGER, width=7).move_to(backing.get_right() + LEFT * 0.38)
    return VGroup(backing, label), cross


def local_mapping_rows() -> VGroup:
    rows = VGroup()
    labels = {
        "white_matter": "Inside WM  ->  WM",
        "grey_matter": "Inside GM  ->  GM",
        "csf": "Inside CSF  ->  CSF",
        "external_soft_tissue": "Outside skull  ->  soft tissue",
    }
    for identifier, text in labels.items():
        blood = Circle(
            radius=0.095,
            fill_color=tissue_colour("blood"),
            fill_opacity=1,
            stroke_width=0,
        )
        arrow = Arrow(LEFT * 0.20, RIGHT * 0.20, buff=0, color=MUTED, stroke_width=2)
        target = Circle(
            radius=0.095,
            fill_color=tissue_colour(identifier),
            fill_opacity=1,
            stroke_width=0,
        )
        label = Text(text, font_size=19, color=WHITE)
        rows.add(VGroup(blood, arrow, target, label).arrange(RIGHT, buff=0.12))
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
    rows.move_to((3.38, 0.2, 0)).align_to((1.55, 0, 0), LEFT)
    return rows


class BloodLabelHarmonization(Scene):
    """Silent scientific schematic of local blood-label inheritance."""

    def construct(self) -> None:
        config.background_color = BACKGROUND
        beats = storyboard()
        title = opening_title()
        subtitle = Text(
            "How the explicit blood label is removed",
            font_size=23,
            color=ACCENT,
        ).next_to(title, DOWN, buff=0.10)
        head = corrected_head(center=LEFT * 2.62 + UP * 0.05, scale=0.84)
        corrected_legend = legend(
            (
                "white_matter",
                "grey_matter",
                "csf",
                "compact_bone",
                "spongy_bone",
                "scalp",
                "muscle",
                "blood",
                "eyes",
            )
        )
        caption = caption_card(beats[0].caption)

        self.play(FadeIn(title), FadeIn(subtitle), run_time=0.8)
        self.play(FadeIn(head.group, shift=RIGHT * 0.15), run_time=2.0)
        self.play(FadeIn(corrected_legend), FadeIn(caption), run_time=1.2)
        self.wait(2.0)

        # Reject the two tempting but incorrect global assignments.
        wrong_csf, cross_csf = mapping_card("All blood  ->  CSF", y=0.72)
        wrong_scalp, cross_scalp = mapping_card("All blood  ->  scalp", y=-0.72)
        wrong_cards = VGroup(wrong_csf, wrong_scalp)
        wrong_crosses = VGroup(cross_csf, cross_scalp)
        next_caption = caption_card(beats[1].caption)
        self.play(
            FadeOut(corrected_legend),
            FadeIn(wrong_cards),
            Transform(caption, next_caption),
            run_time=1.5,
        )
        self.play(Create(wrong_crosses), run_time=1.5)
        self.wait(3.0)

        # Reveal the nested local compartments from inside to outside.
        envelope_labels = VGroup(
            *[
                Text(text, font_size=23, color=WHITE)
                for text in (
                    "1   White-matter envelope",
                    "2   + Grey-matter envelope",
                    "3   + CSF envelope",
                    "4   + Bone envelope",
                    "5   Whole-head soft-tissue envelope",
                )
            ]
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.34)
        envelope_labels.move_to((3.35, 0.12, 0)).align_to((1.38, 0, 0), LEFT)
        next_caption = caption_card(beats[2].caption)
        self.play(
            FadeOut(wrong_cards),
            FadeOut(wrong_crosses),
            FadeIn(envelope_labels),
            Transform(caption, next_caption),
            run_time=1.0,
        )
        envelope_shapes = (
            head.layers["white_matter"],
            head.layers["grey_matter"],
            head.layers["csf"],
            head.layers["compact_bone"],
            head.layers["scalp"],
        )
        for shape, label in zip(envelope_shapes, envelope_labels, strict=True):
            outline = shape.copy().set_fill(opacity=0).set_stroke(YELLOW, width=7)
            self.play(
                ShowPassingFlash(outline, time_width=0.55),
                Indicate(label, color=YELLOW, scale_factor=1.03),
                run_time=0.8,
            )
        self.wait(3.0)

        # Recolour every former blood region according to its own enclosure.
        mapping_rows = local_mapping_rows()
        next_caption = caption_card(beats[3].caption)
        self.play(
            FadeOut(envelope_labels),
            FadeIn(mapping_rows),
            Transform(caption, next_caption),
            run_time=1.5,
        )
        for index, identifier in enumerate(
            ("white_matter", "grey_matter", "csf", "external_soft_tissue")
        ):
            replacement = local_blood_replacement(identifier)
            self.play(
                head.vessels[identifier].animate.set_color(tissue_colour(replacement)),
                Indicate(mapping_rows[index], color=tissue_colour(replacement), scale_factor=1.03),
                run_time=1.0,
            )
        self.wait(2.5)

        # Merge the remaining unsupported explicit classes.
        bone_card = Text(
            "Compact bone + spongy bone  ->  generic bone",
            font_size=22,
            color=WHITE,
        )
        muscle_card = Text(
            "Scalp + muscle  ->  external soft tissue",
            font_size=22,
            color=WHITE,
        )
        merge_cards = VGroup(bone_card, muscle_card).arrange(DOWN, aligned_edge=LEFT, buff=0.48)
        merge_cards.move_to((3.30, 0.16, 0))
        next_caption = caption_card(beats[4].caption)
        self.play(
            FadeOut(mapping_rows),
            FadeIn(merge_cards),
            Transform(caption, next_caption),
            run_time=1.0,
        )
        self.play(
            head.layers["compact_bone"].animate.set_fill(
                tissue_colour("generic_bone"), opacity=0.92
            ),
            head.layers["spongy_bone"].animate.set_fill(
                tissue_colour("generic_bone"), opacity=0.92
            ),
            FadeOut(head.muscles),
            run_time=2.0,
        )
        self.wait(2.0)

        # End with the explicit and harmonized representations side by side.
        self.play(
            FadeOut(title),
            FadeOut(subtitle),
            FadeOut(head.group),
            FadeOut(merge_cards),
            FadeOut(caption),
            run_time=0.8,
        )
        final_title, caveat, before_label, after_label = final_comparison_header()
        self.play(FadeIn(final_title), run_time=0.7)

        before = corrected_head(center=LEFT * 3.45 + UP * 0.05, scale=0.66)
        after = harmonized_head(center=RIGHT * 3.45 + UP * 0.05, scale=0.66)
        transition = Arrow(LEFT * 0.55, RIGHT * 0.55, color=ACCENT, stroke_width=5)
        self.play(
            FadeIn(before.group),
            FadeIn(after.group),
            FadeIn(before_label),
            FadeIn(after_label),
            GrowArrow(transition),
            run_time=2.0,
        )

        final_caption = caption_card(beats[5].caption)
        self.play(FadeIn(final_caption), FadeIn(caveat), run_time=1.0)
        self.wait(2.5)
