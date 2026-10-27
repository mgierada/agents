"""Graph state and per-image records."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from cleanup.config import Settings

Verdict = Literal["pass", "fail", "borderline"]


@dataclass(frozen=True)
class ImageRecord:
    """One image/annotation pair and the result of assessing it."""

    image_path: Path
    # Paired annotation (YOLO txt or Darwin JSON); None if the image has none.
    label_path: Path | None
    # Image path relative to data_dir; mirrored under output_dir.
    rel_path: Path
    width: int = 0
    height: int = 0
    blur: float | None = None
    brightness: float | None = None
    verdict: Verdict = "pass"
    reasons: tuple[str, ...] = ()


class CleanupState(TypedDict):
    """State passed between graph nodes."""

    settings: Settings
    records: list[ImageRecord]
    summary: NotRequired[dict[str, int]]
