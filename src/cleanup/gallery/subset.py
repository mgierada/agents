"""Choose which crops a run works on: the whole gallery, a filtered subset, or a stratified
calibration sample for hand labelling."""

from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

STRATA: tuple[str, ...] = ("suspect", "confusion_pair", "random")


@dataclass(frozen=True)
class Subset:
    """Filters applied in order: crop ids, brands, top brands, suspects, then sampling."""

    crop_ids_file: Path | None = None
    brands: tuple[str, ...] = ()
    top_brands: int | None = None
    suspects_only: bool = False
    # Stratified sample of this many crops (a third each: suspects, confusion-pair brands,
    # random). Takes precedence over frac/limit.
    calibration_sample: int | None = None
    frac: float | None = None
    limit: int | None = None
    seed: int = 0

    def describe(self) -> dict[str, object]:
        """JSON-friendly form for run.json."""
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def select_crops(
    analysis: pl.DataFrame,
    subset: Subset,
    priority: pl.DataFrame,
    confusion_brands: set[str],
) -> pl.DataFrame:
    """Crops of the run, with a `stratum` column (null unless a calibration sample)."""
    crops: pl.DataFrame = filter_crops(analysis, subset, priority)
    if subset.calibration_sample is not None:
        return stratified_sample(crops, subset.calibration_sample, confusion_brands, subset.seed)
    if subset.frac is not None:
        crops = crops.sample(fraction=subset.frac, seed=subset.seed)
    if subset.limit is not None and subset.limit < crops.height:
        crops = crops.sample(n=subset.limit, seed=subset.seed)
    return crops.with_columns(stratum=pl.lit(None, dtype=pl.String)).sort("crop_id")


def filter_crops(analysis: pl.DataFrame, subset: Subset, priority: pl.DataFrame) -> pl.DataFrame:
    """Apply the crop-id, brand, top-brand and suspect filters."""
    crops: pl.DataFrame = analysis
    if subset.crop_ids_file is not None:
        ids: list[str] = subset.crop_ids_file.read_text().split()
        crops = crops.filter(pl.col("crop_id").is_in(ids))
    if subset.brands:
        crops = crops.filter(pl.col("brand").is_in(list(subset.brands)))
    if subset.top_brands is not None:
        top: list[str] = priority.head(subset.top_brands)["brand"].to_list()
        crops = crops.filter(pl.col("brand").is_in(top))
    if subset.suspects_only:
        crops = crops.filter(pl.col("suspect"))
    return crops


def stratified_sample(
    crops: pl.DataFrame, n: int, confusion_brands: set[str], seed: int
) -> pl.DataFrame:
    """About n/3 suspects, n/3 non-suspect crops of confusion-pair brands, rest random.

    A stratum short of crops is topped up from the random stratum.
    """
    suspects: pl.DataFrame = crops.filter(pl.col("suspect"))
    paired: pl.DataFrame = crops.filter(
        ~pl.col("suspect") & pl.col("brand").is_in(list(confusion_brands))
    )
    picked: list[pl.DataFrame] = [
        take(suspects, n // 3, seed).with_columns(stratum=pl.lit("suspect")),
        take(paired, n // 3, seed).with_columns(stratum=pl.lit("confusion_pair")),
    ]
    taken: pl.Series = pl.concat(picked)["crop_id"]
    rest: pl.DataFrame = crops.filter(~pl.col("crop_id").is_in(taken.implode()))
    remaining: int = n - sum(p.height for p in picked)
    picked.append(take(rest, remaining, seed).with_columns(stratum=pl.lit("random")))
    return pl.concat(picked).sort("crop_id")


def take(frame: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    """Random n rows, or all rows if fewer."""
    return frame.sample(n=min(n, frame.height), seed=seed)
