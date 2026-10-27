"""LangGraph nodes. Each takes the state and returns a partial state update."""

import json
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal

import structlog

from cleanup.checks.image import (
    blur_score,
    brightness,
    brightness_issues,
    grade_blur,
    load_gray,
    resolution_issues,
)
from cleanup.checks.labels import label_issues
from cleanup.config import JudgeScope, Settings
from cleanup.console import progress
from cleanup.dataset import find_label, is_image, metadata_files, walk_files
from cleanup.judge import Judge, JudgeVerdict, policy_judge
from cleanup.state import CleanupState, ImageRecord

REPORT_NAME = "report.json"

log = structlog.get_logger()


def load_dataset(state: CleanupState) -> dict[str, Any]:
    """Discover images anywhere under `data_dir` (following symlinks) and pair annotations."""
    settings: Settings = state["settings"]
    data_dir: Path = settings.data_dir
    records: list[ImageRecord] = [
        ImageRecord(
            image_path=path,
            label_path=find_label(image_path=path, data_dir=data_dir),
            rel_path=path.relative_to(data_dir),
        )
        for path in progress(iterable=walk_files(root=data_dir), desc="scan", unit="file")
        if is_image(path, settings.image_exts)
    ]
    labelled: int = sum(r.label_path is not None for r in records)
    log.info("dataset_loaded", data_dir=str(data_dir), images=len(records), labelled=labelled)
    return {"records": records}


def assess_images(state: CleanupState) -> dict[str, Any]:
    """Run deterministic checks on every record and assign a verdict."""
    settings: Settings = state["settings"]
    records: list[ImageRecord] = map_records(
        records=state["records"],
        fn=lambda r: assess_record(record=r, settings=settings),
        desc="assess",
    )
    return {"records": records}


def map_records(
    records: list[ImageRecord],
    fn: Callable[[ImageRecord], ImageRecord],
    desc: str,
    select: Callable[[ImageRecord], bool] = lambda _r: True,
) -> list[ImageRecord]:
    """Apply `fn` to selected records with a progress bar showing a live verdict tally."""
    counts: Counter[str] = Counter()
    result: list[ImageRecord] = []
    with progress(desc=desc, total=sum(map(select, records))) as bar:
        for record in records:
            if not select(record):
                result.append(record)
                continue
            updated: ImageRecord = fn(record)
            counts[updated.verdict] += 1
            bar.set_postfix(counts, refresh=False)
            bar.update()
            result.append(updated)
    log.info(f"{desc}_done", **counts)
    return result


def assess_record(record: ImageRecord, settings: Settings) -> ImageRecord:
    """Assess a single image/label pair."""
    gray = load_gray(record.image_path)
    if gray is None:
        return replace(record, verdict="fail", reasons=("unreadable",))

    height, width = gray.shape[:2]
    blur: float = blur_score(gray)
    bright: float = brightness(gray)
    blur_verdict = grade_blur(
        score=blur, threshold=settings.blur_threshold, margin=settings.borderline_margin
    )

    failures: list[str] = [
        *resolution_issues(
            width=width, height=height, min_width=settings.min_width, min_height=settings.min_height
        ),
        *brightness_issues(value=bright, min=settings.brightness_min, max=settings.brightness_max),
        *annotation_issues(record.label_path, settings),
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


def annotation_issues(label_path: Path | None, settings: Settings) -> list[str]:
    """Validate YOLO txt labels. Other formats (Darwin JSON) are copied unvalidated."""
    if label_path is None:
        return [] if settings.allow_empty_labels else ["missing_label"]
    if label_path.suffix != ".txt":
        return []
    return label_issues(label_path, settings.min_box_area, settings.allow_empty_labels)


def route_after_assess(state: CleanupState) -> Literal["judge_borderline", "write_output"]:
    """Send the run through the judge only if any record needs judging."""
    scope: JudgeScope = state["settings"].judge_scope
    if any(needs_judging(record=r, scope=scope) for r in state["records"]):
        return "judge_borderline"
    return "write_output"


def needs_judging(record: ImageRecord, scope: JudgeScope) -> bool:
    """Borderline records are always judged; passing ones too when scope is `survivors`."""
    return record.verdict == "borderline" or (scope == "survivors" and record.verdict == "pass")


def make_judge_node(judge: Judge | None) -> Callable[[CleanupState], dict[str, Any]]:
    """Build the judge node. Falls back to `settings.borderline_policy` when judge is None."""

    def judge_borderline(state: CleanupState) -> dict[str, Any]:
        settings: Settings = state["settings"]
        decide: Judge = judge or policy_judge(settings.borderline_policy)
        records: list[ImageRecord] = map_records(
            records=state["records"],
            fn=lambda r: resolve_record(record=r, decide=decide),
            desc="judge",
            select=lambda r: needs_judging(record=r, scope=settings.judge_scope),
        )
        return {"records": records}

    return judge_borderline


def resolve_record(record: ImageRecord, decide: Judge) -> ImageRecord:
    """Turn a record into pass/fail using the judge, keeping its reason."""
    decision: JudgeVerdict = decide(record)
    action: str = "keep" if decision.keep else "drop"
    log.debug("judged", image=str(record.rel_path), action=action, reason=decision.reason)
    return replace(
        record,
        verdict="pass" if decision.keep else "fail",
        reasons=(*record.reasons, f"judge:{action}:{decision.reason}"),
    )


def write_output(state: CleanupState) -> dict[str, Any]:
    """Mirror `data_dir` into `output_dir` as real files: passing images, their annotations,
    and dataset metadata (e.g. `classes/`). Symlinks in the source are followed, not copied.
    """
    settings: Settings = state["settings"]
    if settings.dry_run:
        return {}
    data_dir: Path = settings.data_dir
    out: Path = settings.output_dir
    passing: list[ImageRecord] = [r for r in state["records"] if r.verdict == "pass"]
    for record in progress(passing, desc="copy"):
        copy_file(src=record.image_path, dst=out / record.rel_path)
        if record.label_path is not None:
            copy_file(src=record.label_path, dst=out / record.label_path.relative_to(data_dir))
    metadata: list[Path] = list(metadata_files(data_dir, settings.image_exts))
    for path in metadata:
        copy_file(src=path, dst=out / path.relative_to(data_dir))
    log.info("output_written", output_dir=str(out), images=len(passing), metadata=len(metadata))
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
    report_path.write_text(data=json.dumps(report, indent=2, default=str))
    log.info("report_written", path=str(report_path), **summary)
    return {"summary": summary}


def copy_file(src: Path, dst: Path) -> None:
    """Copy a file's contents (following symlinks), creating parent directories."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst, follow_symlinks=True)
