"""Brand review order. Suspect flags and kNN agreement only order the review; they are the
embedding judging itself, so they never decide a crop's fate."""

import re
from collections.abc import Iterable
from typing import Any

import polars as pl

CONFUSION = re.compile(r"^(?P<brand>.+) -> (?P<partner>.+): (?P<count>\d+)$")


def parse_confusions(summary: dict[str, Any]) -> pl.DataFrame:
    """Rows of summary.json `top_confusions` ('a -> b: 123') of the best backbone."""
    best: dict[str, Any] = summary[summary["best"]]
    lines: list[str] = best.get("top_confusions", [])
    rows: list[dict[str, str | int]] = [
        {"brand": m["brand"], "partner": m["partner"], "count": int(m["count"])}
        for m in (CONFUSION.match(line) for line in lines)
        if m is not None
    ]
    return pl.DataFrame(rows, schema={"brand": pl.String, "partner": pl.String, "count": pl.Int64})


def brand_priority(
    analysis: pl.DataFrame, confusions: pl.DataFrame, focus: Iterable[str]
) -> pl.DataFrame:
    """One row per brand, focus brands first, then by score (higher = review sooner).

    score = confused_share + suspect_rate + (1 - mean_agreement); every term is in [0, 1].
    confused_share is the brand's confusions (both directions) per crop, capped at 1.
    """
    both_ways: pl.DataFrame = pl.concat(
        [
            confusions,
            confusions.select(
                pl.col("partner").alias("brand"), pl.col("brand").alias("partner"), "count"
            ),
        ]
    )
    confused: pl.DataFrame = both_ways.group_by("brand").agg(
        confusion_count=pl.col("count").sum(),
        confusion_partners=pl.col("partner").unique().sort().str.join(", "),
    )
    per_brand: pl.DataFrame = analysis.group_by("brand").agg(
        n_crops=pl.len(),
        suspect_rate=pl.col("suspect").mean(),
        mean_agreement=pl.col("knn_agreement").mean(),
    )
    focus_set: list[str] = list(focus)
    return (
        per_brand.join(confused, on="brand", how="left")
        .with_columns(
            pl.col("confusion_count").fill_null(0),
            pl.col("confusion_partners").fill_null(""),
            focus=pl.col("brand").is_in(focus_set),
        )
        .with_columns(
            score=(pl.col("confusion_count") / pl.col("n_crops")).clip(upper_bound=1.0)
            + pl.col("suspect_rate")
            + (1 - pl.col("mean_agreement"))
        )
        .sort(["focus", "score"], descending=True)
        .with_row_index(name="rank", offset=1)
    )
