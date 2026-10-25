"""LangGraph nodes. Each takes the state and returns a partial state update."""

import json
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal

from cleanup.checks.image import (
    blur_score,
    brightness,
    brightness_issues,
    grade_blur,
    load_gray,
    resolution_issues,
)
from cleanup.checks.labels import label_issues
from cleanup.config import Settings
from cleanup.judge import Judge, policy_judge
from cleanup.state import CleanupState, ImageRecord

REPORT_NAME = "report.json"


def load_dataset(state: CleanupState) -> dict[str, Any]:
    """Discover images under `data_dir/images` and pair them with `data_dir/labels`."""
    settings: Settings = state["settings"]
    images_root: Path = settings.data_dir / "images"
    labels_root: Path = settings.data_dir / "labels"
    paths: list[Path] = sorted(
        p for p in images_root.rglob("*") if p.suffix.lower() in settings.image_exts
    )
    records: list[ImageRecord] = [
        ImageRecord(
            image_path=p,
            label_path=(labels_root / p.relative_to(images_root)).with_suffix(".txt"),
            rel_path=p.relative_to(images_root),
        )
        for p in paths
    ]
    return {"records": records}


def assess_images(state: CleanupState) -> dict[str, Any]:
    """Run deterministic checks on every record and assign a verdict."""
    settings: Settings = state["settings"]
    return {"records": [assess_record(r, settings) for r in state["records"]]}


def assess_record(record: ImageRecord, settings: Settings) -> ImageRecord:
    """Assess a single image/label pair."""
    gray = load_gray(record.image_path)
    if gray is None:
        return replace(record, verdict="fail", reasons=("unreadable",))

    height, width = gray.shape[:2]
    blur: float = blur_score(gray)
    bright: float = brightness(gray)
    blur_verdict = grade_blur(blur, settings.blur_threshold, settings.borderline_margin)

    failures: list[str] = [
        *resolution_issues(width, height, settings.min_width, settings.min_height),
        *brightness_issues(bright, settings.brightness_min, settings.brightness_max),
        *label_issues(record.label_path, settings.min_box_area, settings.allow_empty_labels),
        *(["blurry"] if blur_verdict == "fail" else []),
    ]
    is_borderline: bool = not failures and blur_verdict == "borderline"
    return replace(
        record,
        width=width,
        height=height,
        blur=blur,
        brightness=bright,
        verdict="fail" if failures else ("borderline" if is_borderline else "pass"),
        reasons=tuple(failures) or (("borderline_blur",) if is_borderline else ()),
    )


def route_after_assess(state: CleanupState) -> Literal["judge_borderline", "write_output"]:
    """Send the run through the judge only if any record is borderline."""
    if any(r.verdict == "borderline" for r in state["records"]):
        return "judge_borderline"
    return "write_output"


def make_judge_node(judge: Judge | None) -> Callable[[CleanupState], dict[str, Any]]:
    """Build the judge node. Falls back to `settings.borderline_policy` when judge is None."""

    def judge_borderline(state: CleanupState) -> dict[str, Any]:
        decide: Judge = judge or policy_judge(state["settings"].borderline_policy)
        return {"records": [resolve_borderline(r, decide) for r in state["records"]]}

    return judge_borderline


def resolve_borderline(record: ImageRecord, decide: Judge) -> ImageRecord:
    """Turn a borderline record into pass/fail using the judge."""
    if record.verdict != "borderline":
        return record
    keep: bool = decide(record)
    return replace(
        record,
        verdict="pass" if keep else "fail",
        reasons=(*record.reasons, "judge:keep" if keep else "judge:drop"),
    )


def write_output(state: CleanupState) -> dict[str, Any]:
    """Copy passing images and labels into `output_dir`, mirroring the source layout."""
    settings: Settings = state["settings"]
    if settings.dry_run:
        return {}
    # TODO: also copy dataset metadata (data.yaml / classes.txt) when present.
    for record in (r for r in state["records"] if r.verdict == "pass"):
        copy_file(record.image_path, settings.output_dir / "images" / record.rel_path)
        if record.label_path.exists():
            label_rel: Path = record.rel_path.with_suffix(".txt")
            copy_file(record.label_path, settings.output_dir / "labels" / label_rel)
    return {}


def write_report(state: CleanupState) -> dict[str, Any]:
    """Write a JSON report of every record and return verdict counts."""
    settings: Settings = state["settings"]
    summary: dict[str, int] = dict(Counter(r.verdict for r in state["records"]))
    report: dict[str, Any] = {
        "summary": summary,
        "records": [asdict(r) for r in state["records"]],
    }
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    report_path: Path = settings.output_dir / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2, default=str))
    return {"summary": summary}


def copy_file(src: Path, dst: Path) -> None:
    """Copy a file, creating parent directories."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
