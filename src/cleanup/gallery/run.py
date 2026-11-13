"""Create a run: pick crops, compute flags and the brand priority (no model calls)."""

import json
from pathlib import Path
from typing import Any

import polars as pl
import structlog

from cleanup.gallery.config import (
    ANALYSIS_NAME,
    FOCUS_BRANDS,
    SUMMARY_NAME,
    GallerySettings,
    RunPaths,
)
from cleanup.gallery.flags import build_flags
from cleanup.gallery.jsonl import now
from cleanup.gallery.priority import brand_priority, parse_confusions
from cleanup.gallery.subset import Subset, select_crops

log = structlog.get_logger()

# analysis.parquet's crop_path points at the training host; local crops live in crops/.
DROPPED_COLUMNS: tuple[str, ...] = ("crop_path",)


def init_run(settings: GallerySettings, run: str, subset: Subset) -> RunPaths:
    """Write run.json, crops.parquet, flags.parquet and priority.csv for a new run."""
    paths: RunPaths = settings.run_paths(run)
    if paths.crops.exists():
        raise FileExistsError(f"run {run!r} exists at {paths.root}; pick another name")
    analysis: pl.DataFrame = load_analysis(settings.gallery_dir)
    confusions: pl.DataFrame = parse_confusions(load_summary(settings.gallery_dir))
    priority: pl.DataFrame = brand_priority(analysis, confusions, FOCUS_BRANDS)
    confusion_brands: set[str] = set(confusions["brand"]) | set(confusions["partner"])
    crops: pl.DataFrame = select_crops(analysis, subset, priority, confusion_brands)
    if crops.is_empty():
        raise ValueError("subset selects no crops")

    paths.root.mkdir(parents=True, exist_ok=True)
    flags: pl.DataFrame = build_flags(crops, settings)
    crops.write_parquet(paths.crops)
    flags.write_parquet(paths.flags)
    write_priority(priority, flags, paths.priority)
    write_run_json(paths, settings, subset, crops)
    log.info(
        "run_created",
        run=str(paths.root),
        crops=crops.height,
        brands=crops["brand"].n_unique(),
        gallery_unusable=int(flags["gallery_unusable"].sum()),
    )
    return paths


def load_analysis(gallery_dir: Path) -> pl.DataFrame:
    """The logo audit's per-crop table, without host-specific columns."""
    analysis: pl.DataFrame = pl.read_parquet(gallery_dir / ANALYSIS_NAME)
    return analysis.drop([c for c in DROPPED_COLUMNS if c in analysis.columns])


def load_summary(gallery_dir: Path) -> dict[str, Any]:
    """The logo audit's summary.json."""
    return json.loads((gallery_dir / SUMMARY_NAME).read_text())


def write_priority(priority: pl.DataFrame, flags: pl.DataFrame, path: Path) -> None:
    """Brand priority with the run's crop and flag counts per brand."""
    in_run: pl.DataFrame = flags.group_by("brand").agg(
        run_crops=pl.len(), run_unusable=pl.col("gallery_unusable").sum()
    )
    priority.join(in_run, on="brand", how="left").with_columns(
        pl.col("run_crops", "run_unusable").fill_null(0)
    ).write_csv(path)


def write_run_json(
    paths: RunPaths, settings: GallerySettings, subset: Subset, crops: pl.DataFrame
) -> None:
    """Record how the run was made, so its outputs can be reproduced."""
    meta: dict[str, object] = {
        "created_at": now(),
        "gallery_dir": str(settings.gallery_dir),
        "subset": subset.describe(),
        "n_crops": crops.height,
        "n_brands": crops["brand"].n_unique(),
        "settings": settings.model_dump(mode="json"),
    }
    paths.run_json.write_text(json.dumps(meta, indent=2))


def load_run(settings: GallerySettings, run: str) -> tuple[RunPaths, pl.DataFrame]:
    """Paths of an existing run and its crops joined with their flags."""
    paths: RunPaths = settings.run_paths(run)
    if not paths.crops.exists():
        raise FileNotFoundError(f"no run {run!r} at {paths.root}; run `cleanup gallery init`")
    flags: pl.DataFrame = pl.read_parquet(paths.flags).drop("brand", "box_w", "box_h")
    return paths, pl.read_parquet(paths.crops).join(flags, on="crop_id", how="left")
