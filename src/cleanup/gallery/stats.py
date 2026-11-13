"""Per-crop image statistics, computed once per gallery and cached for later runs."""

from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
import structlog
from PIL import Image

from cleanup.checks.image import blur_score, brightness, contrast, load_gray
from cleanup.console import progress

log = structlog.get_logger()

# Blur is measured after resizing every crop to this longest edge. On the raw crops the
# Laplacian variance grows as crops get smaller (tiny crops are saved at native size, so
# pixel noise reads as sharpness); a fixed scale makes scores comparable across sizes.
BLUR_EVAL_EDGE = 128
# Bump when a stat's definition changes; old caches are then ignored.
STATS_VERSION = 1

SCHEMA: dict[str, type[pl.DataType]] = {
    "crop_id": pl.String,
    "readable": pl.Boolean,
    "crop_w": pl.Int64,
    "crop_h": pl.Int64,
    "blur": pl.Float64,
    "contrast": pl.Float64,
    "brightness": pl.Float64,
}


def load_crop_stats(
    crops_dir: Path, crop_ids: list[str], cache_path: Path, workers: int
) -> pl.DataFrame:
    """Stats for `crop_ids`; computes only crops missing from the cache, then updates it."""
    cached: pl.DataFrame = (
        pl.read_parquet(cache_path) if cache_path.exists() else pl.DataFrame(schema=SCHEMA)
    )
    missing: list[str] = sorted(set(crop_ids) - set(cached["crop_id"].to_list()))
    if missing:
        fresh: pl.DataFrame = compute_stats(crops_dir, missing, workers)
        cached = pl.concat([cached, fresh])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached.write_parquet(cache_path)
    log.info("crop_stats", requested=len(crop_ids), computed=len(missing))
    return cached.filter(pl.col("crop_id").is_in(crop_ids))


def compute_stats(crops_dir: Path, crop_ids: list[str], workers: int) -> pl.DataFrame:
    """Stats of the given crops, in parallel."""
    stats = partial(crop_stats, crops_dir=crops_dir)
    with Pool(processes=workers) as pool:
        rows: list[dict[str, object]] = list(
            progress(
                pool.imap(stats, crop_ids, chunksize=256),
                desc="crop stats",
                total=len(crop_ids),
            )
        )
    return pl.DataFrame(rows, schema=SCHEMA)


def crop_stats(crop_id: str, crops_dir: Path) -> dict[str, object]:
    """Blur, contrast and brightness of one crop; nulls if the file is unreadable."""
    gray = load_gray(crops_dir / f"{crop_id}.jpg")
    if gray is None:
        return {"crop_id": crop_id, "readable": False}
    height, width = gray.shape[:2]
    return {
        "crop_id": crop_id,
        "readable": True,
        "crop_w": width,
        "crop_h": height,
        "blur": blur_score(resize_longest(gray, BLUR_EVAL_EDGE)),
        "contrast": contrast(gray),
        "brightness": brightness(gray),
    }


def resize_longest(gray: np.ndarray, edge: int) -> np.ndarray:
    """Bilinear resize of a grayscale array so its longest side equals `edge`."""
    height, width = gray.shape[:2]
    scale: float = edge / max(height, width)
    size: tuple[int, int] = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(Image.fromarray(gray).resize(size, resample=Image.Resampling.BILINEAR))
