"""Parse the batch's YOLO label files into one row per box, keyed like the gallery crops.

A crop id is `<label stem>_<n>` where n indexes the non-empty lines of the label file.
Line formats:
    original:  `cls cx cy w h is_overlay`
    pseudo:    `cls cx cy w h is_overlay conf [run_name=<model>]`
The distillation teacher's boxes are pseudo lines whose run_name starts with the teacher
prefix; they are tagged `teacher` so the two sources can be audited separately.
"""

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import polars as pl

from cleanup.console import progress

Origin = Literal["original", "pseudo", "teacher"]

RUN_NAME = re.compile(r"\[run_name=([^\]]+)\]")
CONF_FIELD = 6
# Normalised coordinates may overshoot [0, 1] by float noise.
BOUNDS_EPS = 1e-3
# Lower rank wins when two boxes duplicate each other.
ORIGIN_RANK: dict[str, int] = {"original": 0, "pseudo": 1, "teacher": 2}
SCHEMA: dict[str, type[pl.DataType]] = {
    "crop_id": pl.String,
    "class_id": pl.Int64,
    "cx": pl.Float64,
    "cy": pl.Float64,
    "w": pl.Float64,
    "h": pl.Float64,
    "origin": pl.String,
    "confidence": pl.Float64,
    "run_name": pl.String,
    "out_of_bounds": pl.Boolean,
    "duplicate_of": pl.String,
    "conflicts_with": pl.String,
}


@dataclass(frozen=True)
class LabelBox:
    """One label line."""

    crop_id: str
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    origin: Origin
    confidence: float | None
    run_name: str | None

    @property
    def out_of_bounds(self) -> bool:
        """True if the box reaches outside the image."""
        return (
            self.cx - self.w / 2 < -BOUNDS_EPS
            or self.cy - self.h / 2 < -BOUNDS_EPS
            or self.cx + self.w / 2 > 1 + BOUNDS_EPS
            or self.cy + self.h / 2 > 1 + BOUNDS_EPS
        )


def stem_of(crop_id: str) -> str:
    """Label-file stem of a crop id."""
    return crop_id.rsplit("_", maxsplit=1)[0]


def load_label_boxes(
    labels_dir: Path,
    stems: Iterable[str],
    teacher_prefix: str,
    duplicate_iou: float,
    workers: int,
) -> pl.DataFrame:
    """Parse the label files of `stems` in parallel; one row per box (see SCHEMA)."""
    paths: list[Path] = [labels_dir / f"{stem}.txt" for stem in sorted(set(stems))]
    parse = partial(label_rows, teacher_prefix=teacher_prefix, duplicate_iou=duplicate_iou)
    rows: list[dict[str, object]] = []
    with Pool(processes=workers) as pool:
        for file_rows in progress(
            pool.imap_unordered(parse, paths, chunksize=256),
            desc="labels",
            total=len(paths),
            unit="file",
        ):
            rows.extend(file_rows)
    return pl.DataFrame(rows, schema=SCHEMA)


def label_rows(path: Path, teacher_prefix: str, duplicate_iou: float) -> list[dict[str, object]]:
    """Rows of one label file, with out-of-bounds and overlap tags. Missing file: no rows."""
    if not path.is_file():
        return []
    boxes: list[LabelBox] = parse_label_file(path, teacher_prefix)
    duplicates, conflicts = find_overlaps(boxes, duplicate_iou)
    return [
        asdict(box)
        | {
            "out_of_bounds": box.out_of_bounds,
            "duplicate_of": duplicates.get(box.crop_id),
            "conflicts_with": conflicts.get(box.crop_id),
        }
        for box in boxes
    ]


def parse_label_file(path: Path, teacher_prefix: str) -> list[LabelBox]:
    """Boxes of one label file, numbered like the gallery crops."""
    lines: list[str] = [line for line in path.read_text().splitlines() if line.strip()]
    return [parse_line(f"{path.stem}_{n}", line, teacher_prefix) for n, line in enumerate(lines)]


def parse_line(crop_id: str, line: str, teacher_prefix: str) -> LabelBox:
    """Parse one label line (original or pseudo format)."""
    parts: list[str] = line.split()
    match: re.Match[str] | None = RUN_NAME.search(line)
    run_name: str | None = match.group(1) if match else None
    return LabelBox(
        crop_id=crop_id,
        class_id=int(parts[0]),
        cx=float(parts[1]),
        cy=float(parts[2]),
        w=float(parts[3]),
        h=float(parts[4]),
        origin=origin_of(run_name, teacher_prefix),
        confidence=float(parts[CONF_FIELD]) if len(parts) > CONF_FIELD and run_name else None,
        run_name=run_name,
    )


def origin_of(run_name: str | None, teacher_prefix: str) -> Origin:
    """Where a box came from: human label, an older pseudo-labeller, or the teacher."""
    if run_name is None:
        return "original"
    return "teacher" if run_name.startswith(teacher_prefix) else "pseudo"


def find_overlaps(
    boxes: list[LabelBox], iou_threshold: float
) -> tuple[dict[str, str], dict[str, str]]:
    """Boxes overlapping at `iou_threshold` or more: (duplicates, conflicts).

    duplicates: a later box of the same class -> the box it repeats. Boxes are visited
    original-first, then by line, so the human label is the one kept.
    conflicts: both boxes of a different-class pair -> the other box. The same logo carries
    two brands, so at least one is wrong; neither side is dropped automatically.
    """
    ordered: list[LabelBox] = sorted(boxes, key=lambda b: ORIGIN_RANK[b.origin])
    kept: list[LabelBox] = []
    duplicates: dict[str, str] = {}
    conflicts: dict[str, str] = {}
    for box in ordered:
        overlapping: list[LabelBox] = [k for k in kept if iou(box, k) >= iou_threshold]
        for other in (k for k in overlapping if k.class_id != box.class_id):
            conflicts.setdefault(box.crop_id, other.crop_id)
            conflicts.setdefault(other.crop_id, box.crop_id)
        same: LabelBox | None = next((k for k in overlapping if k.class_id == box.class_id), None)
        if same is None:
            kept.append(box)
        else:
            duplicates[box.crop_id] = same.crop_id
    return duplicates, conflicts


def iou(a: LabelBox, b: LabelBox) -> float:
    """Intersection over union of two normalised center-format boxes."""
    ix: float = max(0.0, min(a.cx + a.w / 2, b.cx + b.w / 2) - max(a.cx - a.w / 2, b.cx - b.w / 2))
    iy: float = max(0.0, min(a.cy + a.h / 2, b.cy + b.h / 2) - max(a.cy - a.h / 2, b.cy - b.h / 2))
    inter: float = ix * iy
    union: float = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0
