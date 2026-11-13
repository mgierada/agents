"""Deterministic per-crop flags (no model).

Flags only judge gallery usability. They never remove detector boxes: a blurry but real logo
must stay a box, or the detector learns it as background.
"""

from pathlib import Path

import polars as pl

from cleanup.gallery.config import GallerySettings
from cleanup.gallery.labels import load_label_boxes, stem_of
from cleanup.gallery.stats import STATS_VERSION, load_crop_stats

ORIGIN_UNKNOWN = "unknown"


def build_flags(crops: pl.DataFrame, settings: GallerySettings) -> pl.DataFrame:
    """Stats, label info and flags for every crop of the run (one row per crop)."""
    crop_ids: list[str] = crops["crop_id"].to_list()
    stats: pl.DataFrame = load_crop_stats(
        crops_dir=settings.crops_dir,
        crop_ids=crop_ids,
        cache_path=settings.cache_dir / f"crop_stats_v{STATS_VERSION}.parquet",
        workers=settings.workers,
    )
    table: pl.DataFrame = crops.select("crop_id", "brand", "box_w", "box_h").join(
        stats, on="crop_id", how="left"
    )
    table = join_label_boxes(table, settings)
    return add_flags(table, settings)


def join_label_boxes(table: pl.DataFrame, settings: GallerySettings) -> pl.DataFrame:
    """Add origin, teacher confidence, out-of-bounds and duplicate info from the label files.

    Without a labels dir every crop gets origin `unknown` and no label-based flags.
    """
    labels_dir: Path = settings.labels_dir
    if not labels_dir.is_dir():
        return table.with_columns(
            origin=pl.lit(ORIGIN_UNKNOWN),
            confidence=pl.lit(None, dtype=pl.Float64),
            out_of_bounds=pl.lit(None, dtype=pl.Boolean),
            duplicate_of=pl.lit(None, dtype=pl.String),
            conflicts_with=pl.lit(None, dtype=pl.String),
            label_found=pl.lit(None, dtype=pl.Boolean),
        )
    boxes: pl.DataFrame = load_label_boxes(
        labels_dir=labels_dir,
        stems=map(stem_of, table["crop_id"].to_list()),
        teacher_prefix=settings.teacher_run_prefix,
        duplicate_iou=settings.duplicate_iou,
        workers=settings.workers,
    )
    joined: pl.DataFrame = table.join(
        boxes.select(
            "crop_id", "origin", "confidence", "out_of_bounds", "duplicate_of", "conflicts_with"
        ),
        on="crop_id",
        how="left",
    )
    return joined.with_columns(
        label_found=pl.col("origin").is_not_null(),
        origin=pl.col("origin").fill_null(ORIGIN_UNKNOWN),
    )


def add_flags(table: pl.DataFrame, settings: GallerySettings) -> pl.DataFrame:
    """Add a bool column per flag, the `flags` list and `gallery_unusable`."""
    rules: dict[str, pl.Expr] = flag_rules(table, settings)
    flagged: pl.DataFrame = table.with_columns(
        short_side=pl.min_horizontal("box_w", "box_h"),
        **{f"flag_{name}": rule.fill_null(False) for name, rule in rules.items()},
    )
    names: list[str] = list(rules)
    hard: list[str] = [n for n in names if n in settings.hard_flags]
    return flagged.with_columns(
        flags=pl.concat_list(
            [pl.when(pl.col(f"flag_{n}")).then(pl.lit(n)).otherwise(None) for n in names]
        ).list.drop_nulls(),
        gallery_unusable=pl.any_horizontal([pl.col(f"flag_{n}") for n in hard])
        if hard
        else pl.lit(False),
    )


def flag_rules(table: pl.DataFrame, settings: GallerySettings) -> dict[str, pl.Expr]:
    """Flag name -> boolean expression. Nulls (no stats / no labels) count as not flagged."""
    short: pl.Expr = pl.min_horizontal("box_w", "box_h")
    long: pl.Expr = pl.max_horizontal("box_w", "box_h")
    return {
        "unreadable": pl.col("readable").not_(),
        "small": short < settings.min_side_px,
        "empty": pl.col("contrast") < settings.empty_std,
        "low_blur": pl.col("blur") < settings.blur_min,
        "low_blur_rel": brand_relative_blur(table, settings),
        "low_contrast": pl.col("contrast") < settings.contrast_min,
        "too_dark": pl.col("brightness") < settings.brightness_min,
        "too_bright": pl.col("brightness") > settings.brightness_max,
        "extreme_aspect": long > short * settings.max_aspect,
        "out_of_bounds": pl.col("out_of_bounds"),
        "duplicate": pl.col("duplicate_of").is_not_null(),
        "conflicting_label": pl.col("conflicts_with").is_not_null(),
        "missing_label": pl.col("label_found").not_(),
    }


def brand_relative_blur(table: pl.DataFrame, settings: GallerySettings) -> pl.Expr:
    """Blur below the brand's `blur_brand_quantile`, for brands with enough crops in the run.

    Flat-colour logos are legitimately low-variance, so this is relative to the brand.
    """
    by_brand = pl.col("blur").over("brand")
    enough: pl.Expr = pl.len().over("brand") >= settings.blur_brand_min_crops
    threshold: pl.Expr = by_brand.quantile(settings.blur_brand_quantile, interpolation="linear")
    return enough & (pl.col("blur") < threshold)
