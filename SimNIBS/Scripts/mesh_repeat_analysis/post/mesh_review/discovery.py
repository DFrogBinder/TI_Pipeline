from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_SUBJECT_REGEX = r"(?i)(sub-[a-z0-9]+)"

_TAG_COMPONENT = re.compile(r"(?i)^tag[_-](\d+)[_-](.+)$")
_GENERIC_PATH_COMPONENTS = {
    "render",
    "renders",
    "tissue",
    "tissues",
    "tissues_back",
    "front",
    "back",
    "images",
}
_KNOWN_TISSUES = (
    "compact_bone",
    "spongy_bone",
    "white_matter",
    "gray_matter",
    "eye_balls",
    "cartilage",
    "muscle",
    "scalp",
    "blood",
    "bone",
    "csf",
    "fat",
)


@dataclass(frozen=True)
class DiscoveredImage:
    relative_path: str
    subject_id: str
    tissue: str
    view: str
    file_size: int
    mtime_ns: int


@dataclass(frozen=True)
class DiscoveryIssue:
    relative_path: str
    reason: str


@dataclass(frozen=True)
class DiscoveryResult:
    images: tuple[DiscoveredImage, ...]
    skipped: tuple[DiscoveryIssue, ...]


def _match_value(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1) if match.lastindex else match.group(0)


def _slug(value: str) -> str:
    value = value.strip().lower().replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def infer_tissue(
    relative_path: Path, tissue_pattern: re.Pattern[str] | None = None
) -> str:
    path_text = relative_path.as_posix()
    if tissue_pattern is not None:
        custom = _match_value(tissue_pattern, path_text)
        if custom:
            return _slug(custom)

    for component in reversed(relative_path.parts[:-1]):
        match = _TAG_COMPONENT.fullmatch(component)
        if match:
            name = _slug(match.group(2)) or "tissue"
            return f"tag_{int(match.group(1)):02d}_{name}"

    for token in relative_path.stem.split("__"):
        match = _TAG_COMPONENT.fullmatch(token)
        if match:
            name = _slug(match.group(2)) or "tissue"
            return f"tag_{int(match.group(1)):02d}_{name}"

    normalized = _slug(path_text)
    for tissue in _KNOWN_TISSUES:
        if re.search(rf"(?:^|_){re.escape(tissue)}(?:_|$)", normalized):
            return tissue

    for component in reversed(relative_path.parts[:-1]):
        candidate = _slug(component)
        if candidate and candidate not in _GENERIC_PATH_COMPONENTS:
            if not re.fullmatch(r"left_hippocampus_data_\d+", candidate):
                return candidate
    return "unknown_tissue"


def infer_view(relative_path: Path) -> str:
    normalized_parts = {_slug(part) for part in relative_path.parts}
    stem_tokens = {_slug(token) for token in relative_path.stem.split("__")}
    if (
        "tissues_back" in normalized_parts
        or "back" in stem_tokens
        or "posterior" in stem_tokens
    ):
        return "back"
    if "front" in stem_tokens or "anterior" in stem_tokens:
        return "front"
    return "front"


def discover_images(
    image_root: Path,
    *,
    subject_regex: str = DEFAULT_SUBJECT_REGEX,
    tissue_regex: str | None = None,
) -> DiscoveryResult:
    root = Path(image_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")

    subject_pattern = re.compile(subject_regex)
    tissue_pattern = re.compile(tissue_regex) if tissue_regex else None
    images: list[DiscoveredImage] = []
    skipped: list[DiscoveryIssue] = []

    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    for path in candidates:
        relative = path.relative_to(root)
        subject_id = _match_value(subject_pattern, relative.as_posix())
        if not subject_id:
            skipped.append(DiscoveryIssue(relative.as_posix(), "subject ID not found"))
            continue
        try:
            stat = path.stat()
        except OSError as exc:
            skipped.append(
                DiscoveryIssue(relative.as_posix(), f"could not stat image: {exc}")
            )
            continue
        images.append(
            DiscoveredImage(
                relative_path=relative.as_posix(),
                subject_id=subject_id,
                tissue=infer_tissue(relative, tissue_pattern),
                view=infer_view(relative),
                file_size=int(stat.st_size),
                mtime_ns=int(stat.st_mtime_ns),
            )
        )

    return DiscoveryResult(tuple(images), tuple(skipped))
