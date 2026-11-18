"""Brand-level statistics for the UI: how often the VLM disagrees with the annotations.

"Annotation" is the crop's original brand label in the gallery. The VLM agrees when its
latest verdict is `brand_match`; `different_brand` means the crop is mislabelled (another
brand), `unrecognisable` / `not_a_logo` mean it should not be in the gallery at all.
"""

from typing import Any, Literal, get_args

import polars as pl

from cleanup.gallery.judge import TILE_LABELS

BrandSort = Literal[
    "disagree_rate",
    "brand_match_rate",
    "mislabel_rate",
    "unrecognisable_rate",
    "not_a_logo_rate",
    "mislabelled",
    "disagree",
    "net_change",
]
BRAND_SORTS: tuple[str, ...] = get_args(BrandSort)
# Sort key of each verdict's share of judged crops (the UI legend ranks by these).
VERDICT_SORTS: dict[str, str] = {
    "brand_match": "brand_match_rate",
    "different_brand": "mislabel_rate",
    "unrecognisable": "unrecognisable_rate",
    "not_a_logo": "not_a_logo_rate",
}
# desc: highest value of the sort key first; asc: lowest first.
RankOrder = Literal["desc", "asc"]
RANK_ORDERS: tuple[str, ...] = get_args(RankOrder)
DISAGREE: tuple[str, ...] = tuple(v for v in TILE_LABELS if v != "brand_match")
# Human labels with this note copy the VLM's suggestion (UI bulk accept): not an independent
# check, so they are left out of "you vs VLM".
BULK_NOTE = "accepted (bulk)"
# Decisions that take a crop out of its labelled brand towards another one.
RELABEL_DECISIONS: tuple[str, ...] = ("relabel", "suggested_relabel")


def brand_stats(
    items: pl.DataFrame,
    human: dict[str, dict[str, Any]],
    sort: str = "disagree_rate",
    top: int = 20,
    min_judged: int = 20,
    order: str = "desc",
    offset: int = 0,
) -> dict[str, Any]:
    """Overall numbers, `top` brands by `sort` from `offset`, and the most common confusions.

    Rates are over judged crops; brands with fewer than `min_judged` judged crops are left
    out of the ranking (a 2-crop brand at 100% says little). `order` "desc" ranks the
    highest `sort` value first, "asc" the lowest; ties go to the brand with more judged
    crops either way.
    """
    if sort not in BRAND_SORTS:
        raise ValueError(f"sort must be one of {BRAND_SORTS}")
    if order not in RANK_ORDERS:
        raise ValueError(f"order must be one of {RANK_ORDERS}")
    per_brand: pl.DataFrame = brand_table(items)
    eligible: pl.DataFrame = per_brand.filter(pl.col("judged") >= min_judged)
    ranked: pl.DataFrame = eligible.sort(
        [sort, "judged"], descending=[order == "desc", True]
    ).slice(offset, top)
    return {
        "overall": overall(items, human),
        "brands": ranked.to_dicts(),
        "offset": offset,
        "pairs": confusion_pairs(items, per_brand).head(top).to_dicts(),
        "n_brands": per_brand.height,
        "n_ranked": eligible.height,
        "verdicts": list(TILE_LABELS),
        "sorts": list(BRAND_SORTS),
        "verdict_sorts": VERDICT_SORTS,
    }


def overall(items: pl.DataFrame, human: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Run-wide counts: crops, verdict mix, decisions, and your labels vs the VLM."""
    judged: pl.DataFrame = items.filter(pl.col("judge_verdict").is_not_null())
    verdicts: dict[str, int] = dict(judged["judge_verdict"].value_counts().iter_rows())
    decisions: dict[str, int] = (
        dict(items["decision"].value_counts().iter_rows()) if "decision" in items.columns else {}
    )
    by_crop: dict[str, str | None] = dict(items.select("crop_id", "judge_verdict").iter_rows())
    compared: list[tuple[str, str]] = [
        (label["verdict"], verdict)
        for crop_id, label in human.items()
        if label.get("note") != BULK_NOTE and (verdict := by_crop.get(crop_id)) is not None
    ]
    return {
        "crops": items.height,
        "judged": judged.height,
        "verdicts": {v: verdicts.get(v, 0) for v in TILE_LABELS},
        "agree": verdicts.get("brand_match", 0),
        "disagree": sum(verdicts.get(v, 0) for v in DISAGREE),
        "decisions": dict(sorted(decisions.items(), key=lambda kv: -kv[1])),
        "human_labels": len(human),
        "bulk_accepted": sum(label.get("note") == BULK_NOTE for label in human.values()),
        "human_vs_vlm": {
            "compared": len(compared),
            "agree": sum(mine == vlm for mine, vlm in compared),
        },
    }


def brand_table(items: pl.DataFrame) -> pl.DataFrame:
    """One row per labelled brand: verdict counts, rates, relabels out/in, after and delta."""
    judged: pl.Expr = pl.col("judge_verdict").is_not_null()
    counts: list[pl.Expr] = [(pl.col("judge_verdict") == v).sum().alias(v) for v in TILE_LABELS]
    has_decisions: bool = "decision" in items.columns
    moved_out: pl.Expr = (
        pl.col("decision").is_in(RELABEL_DECISIONS).sum() if has_decisions else pl.lit(0)
    )
    table: pl.DataFrame = (
        items.group_by("brand")
        .agg(
            pl.len().alias("crops"),
            judged.sum().alias("judged"),
            *counts,
            moved_out.alias("relabelled_out"),
        )
        .with_columns(
            disagree=pl.sum_horizontal(*DISAGREE),
            mislabelled=pl.col("different_brand"),
        )
        .with_columns(
            disagree_rate=rate(pl.col("disagree")),
            mislabel_rate=rate(pl.col("mislabelled")),
            brand_match_rate=rate(pl.col("brand_match")),
            unrecognisable_rate=rate(pl.col("unrecognisable")),
            not_a_logo_rate=rate(pl.col("not_a_logo")),
        )
    )
    table = table.join(after_counts(items), on="brand", how="left").with_columns(
        relabelled_in=pl.col("relabelled_in").fill_null(0),
        after=pl.col("after").fill_null(0),
    )
    change: pl.Expr = pl.col("after").cast(pl.Int64) - pl.col("crops").cast(pl.Int64)
    return table.with_columns(delta=change, net_change=change.abs()).join(
        top_suggestion(items), on="brand", how="left"
    )


def rate(count: pl.Expr) -> pl.Expr:
    """count / judged, rounded; 0 for brands with nothing judged."""
    return pl.when(pl.col("judged") > 0).then(count / pl.col("judged")).otherwise(0.0).round(4)


def after_counts(items: pl.DataFrame) -> pl.DataFrame:
    """Per brand: crops after the last resolve, and how many were relabelled into it."""
    if "brand_after" not in items.columns:
        return pl.DataFrame(
            {"brand": [], "after": [], "relabelled_in": []},
            schema={"brand": pl.String, "after": pl.UInt32, "relabelled_in": pl.UInt32},
        )
    return (
        items.filter(pl.col("brand_after").is_not_null())
        .group_by(pl.col("brand_after").alias("brand"))
        .agg(
            after=pl.len(),
            relabelled_in=(pl.col("decision") == "relabel").sum(),
        )
    )


def suggested_target() -> pl.Expr:
    """The brand the VLM named: the canonical relabel target, else its own text."""
    raw: pl.Expr = pl.col("judge_suggested_brand").str.strip_chars().str.to_lowercase()
    return pl.coalesce(pl.col("relabel_target"), raw.replace("", None))


def mislabelled(items: pl.DataFrame) -> pl.DataFrame:
    """Crops the VLM called different_brand, with the brand it named (`suggested`)."""
    frame: pl.DataFrame = (
        items
        if "relabel_target" in items.columns
        else items.with_columns(relabel_target=pl.lit(None, pl.String))
    )
    return frame.filter(pl.col("judge_verdict") == "different_brand").with_columns(
        suggested=suggested_target().fill_null("(none named)")
    )


def top_suggestion(items: pl.DataFrame) -> pl.DataFrame:
    """Per brand: the brand the VLM most often named instead, and how often."""
    return (
        mislabelled(items)
        .group_by("brand", "suggested")
        .agg(n=pl.len())
        .sort(["brand", "n", "suggested"], descending=[False, True, False])
        .group_by("brand", maintain_order=True)
        .first()
        .rename({"suggested": "top_suggested", "n": "top_suggested_n"})
    )


def confusion_pairs(items: pl.DataFrame, per_brand: pl.DataFrame) -> pl.DataFrame:
    """Labelled brand -> brand the VLM named, most frequent first, with share of judged."""
    return (
        mislabelled(items)
        .group_by("brand", "suggested")
        .agg(n=pl.len())
        .join(per_brand.select("brand", "judged"), on="brand")
        .with_columns(share_of_judged=(pl.col("n") / pl.col("judged")).round(4))
        .sort(["n", "brand"], descending=[True, False])
    )
