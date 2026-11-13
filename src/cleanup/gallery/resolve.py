"""Step 5: turn flags, judge verdicts and human labels into one decision per crop.

Precedence: human label > hard flag > judge verdict > pending (no verdict yet).
Relabels only ever target brands on the target list; a brand the judge names that is not
on the list becomes a new-class candidate and the crop leaves the gallery. Nothing here
invents class ids, and source files are never edited: decisions.parquet is a new artefact.

A judge's different_brand verdict is trusted to take a crop out of its labelled brand, but not
to name the new one (calibration: right brand in only ~60% of calls). Such crops become
`suggested_relabel` (out of the gallery, target kept in `relabel_target`) and are relabelled
only when you confirm them in the UI, when another judge named the same brand, or with
`trust_judge_relabels`.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import polars as pl
import structlog

from cleanup.gallery import jsonl
from cleanup.gallery.brands import BrandCatalog, load_catalog
from cleanup.gallery.config import GallerySettings, RunPaths
from cleanup.gallery.run import load_run

log = structlog.get_logger()

Decision = Literal[
    "keep",
    "relabel",
    "suggested_relabel",
    "drop_from_gallery",
    "unrecognisable",
    "not_a_logo",
    "pending",
]
CONFIDENCE_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}
# Decisions whose crop stays in the gallery (under brand_after).
IN_GALLERY: tuple[str, ...] = ("keep", "relabel", "pending")
SCHEMA: dict[str, type[pl.DataType]] = {
    "crop_id": pl.String,
    "brand_before": pl.String,
    "decision": pl.String,
    "brand_after": pl.String,
    "decided_by": pl.String,
    "reason": pl.String,
    "suggested_brand": pl.String,
    "visible_text": pl.String,
    "confidence": pl.String,
    "new_class_candidate": pl.String,
    "relabel_target": pl.String,
}


def resolve_run(
    settings: GallerySettings,
    run: str,
    judge: str | None,
    min_confidence: str,
    trust_judge_relabels: bool = False,
) -> pl.DataFrame:
    """Write decisions, brand changes, new-class candidates and summary for the run.

    `judge` picks which judge's verdicts to use (default: the most recent per crop).
    Judge verdicts below `min_confidence` leave the crop pending. Every other judge's latest
    verdict counts as a second opinion for confirming a relabel.
    """
    paths, crops = load_run(settings, run)
    catalog: BrandCatalog = load_catalog(
        settings.target_list, settings.aliases, fallback=crops["brand"].unique().to_list()
    )
    judgements: list[dict[str, Any]] = jsonl.read(paths.judgements)
    verdicts: dict[str, dict[str, Any]] = jsonl.latest_by(
        [r for r in judgements if judge is None or r["judge"] == judge], key="crop_id"
    )
    opinions: dict[str, list[dict[str, Any]]] = all_opinions(judgements)
    human: dict[str, dict[str, Any]] = latest_human_labels(paths)
    rows: list[dict[str, Any]] = [
        decide(
            crop=crop,
            verdict=verdicts.get(crop["crop_id"]),
            human=human.get(crop["crop_id"]),
            catalog=catalog,
            min_rank=CONFIDENCE_RANK[min_confidence],
            opinions=opinions.get(crop["crop_id"], []),
            trust_judge_relabels=trust_judge_relabels,
        )
        for crop in crops.select("crop_id", "brand", "flags", "gallery_unusable").iter_rows(
            named=True
        )
    ]
    decisions: pl.DataFrame = pl.DataFrame(rows, schema=SCHEMA)
    write_outputs(decisions, paths, judge, min_confidence)
    return decisions


def all_opinions(judgements: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """crop_id -> latest verdict of every judge that reviewed it."""
    latest: dict[str, dict[str, Any]] = jsonl.latest_by(
        [r | {"_key": f"{r['crop_id']}\x00{r['judge']}"} for r in judgements], key="_key"
    )
    out: dict[str, list[dict[str, Any]]] = {}
    for row in latest.values():
        out.setdefault(row["crop_id"], []).append(row)
    return out


def latest_human_labels(paths: RunPaths) -> dict[str, dict[str, Any]]:
    """Latest UI label per crop; a cleared label removes the crop's entry."""
    latest: dict[str, dict[str, Any]] = jsonl.latest_by(jsonl.read(paths.events), key="crop_id")
    return {k: v for k, v in latest.items() if v.get("verdict")}


def decide(
    crop: Mapping[str, Any],
    verdict: Mapping[str, Any] | None,
    human: Mapping[str, Any] | None,
    catalog: BrandCatalog,
    min_rank: int,
    opinions: Sequence[Mapping[str, Any]] = (),
    trust_judge_relabels: bool = False,
) -> dict[str, Any]:
    """Decision for one crop (see module docstring for precedence and relabel rules)."""
    brand: str = crop["brand"]
    base: dict[str, Any] = {"crop_id": crop["crop_id"], "brand_before": brand}
    if human is not None:
        return base | from_label(human, brand, catalog, decided_by="human", confirmed=True)
    if crop["gallery_unusable"]:
        reason: str = ",".join(crop["flags"])
        return base | outcome("drop_from_gallery", None, f"rule:{reason}", reason)
    if verdict is None or CONFIDENCE_RANK[verdict["confidence"]] < min_rank:
        why: str = "no verdict" if verdict is None else f"confidence {verdict['confidence']}"
        return base | outcome("pending", brand, "none", why)
    confirmed: bool = trust_judge_relabels or seconded(verdict, opinions, catalog)
    return base | from_label(
        verdict, brand, catalog, decided_by=f"judge:{verdict['judge']}", confirmed=confirmed
    )


def seconded(
    verdict: Mapping[str, Any], opinions: Sequence[Mapping[str, Any]], catalog: BrandCatalog
) -> bool:
    """True when another judge also called different_brand and named the same target brand."""
    target: str | None = catalog.resolve(verdict.get("suggested_brand") or "")
    return target is not None and any(
        o["judge"] != verdict["judge"]
        and o["verdict"] == "different_brand"
        and catalog.resolve(o.get("suggested_brand") or "") == target
        for o in opinions
    )


def from_label(
    label: Mapping[str, Any],
    brand: str,
    catalog: BrandCatalog,
    decided_by: str,
    confirmed: bool,
) -> dict[str, Any]:
    """Decision from a judge verdict or human label (same vocabulary).

    An on-list different_brand is a `relabel` when `confirmed`, else a `suggested_relabel`.
    """
    raw: str = label.get("suggested_brand") or ""
    details: dict[str, Any] = {
        "suggested_brand": raw or None,
        "visible_text": label.get("visible_text") or None,
        "confidence": label.get("confidence"),
    }
    match label["verdict"]:
        case "brand_match":
            return outcome("keep", brand, decided_by, "brand_match") | details
        case "unrecognisable":
            return outcome("unrecognisable", None, decided_by, "unrecognisable") | details
        case "not_a_logo":
            return outcome("not_a_logo", None, decided_by, "not_a_logo") | details
    target: str | None = catalog.resolve(raw)
    if target == catalog.canonical(brand):
        return outcome("keep", brand, decided_by, f"suggested {raw!r} is the label") | details
    if target is not None and confirmed:
        return (
            outcome("relabel", target, decided_by, f"shows {raw!r}")
            | details
            | {"relabel_target": target}
        )
    if target is not None:
        reason: str = f"shows {raw!r}; relabel needs your confirmation or a second judge"
        return (
            outcome("suggested_relabel", None, decided_by, reason)
            | details
            | {"relabel_target": target}
        )
    return (
        outcome("drop_from_gallery", None, decided_by, f"shows {raw!r}, not on target list")
        | details
        | {"new_class_candidate": catalog.canonical(raw) if raw.strip() else None}
    )


def outcome(
    decision: Decision, brand_after: str | None, decided_by: str, reason: str
) -> dict[str, Any]:
    """The decision fields of a row."""
    return {
        "decision": decision,
        "brand_after": brand_after,
        "decided_by": decided_by,
        "reason": reason,
    }


def write_outputs(
    decisions: pl.DataFrame, paths: RunPaths, judge: str | None, min_confidence: str
) -> None:
    """decisions.parquet, brand_changes.csv, new_class_candidates.csv and summary.json."""
    decisions.write_parquet(paths.decisions)
    brand_changes(decisions).write_csv(paths.brand_changes)
    new_class_candidates(decisions).write_csv(paths.new_classes)
    counts: dict[str, int] = dict(decisions["decision"].value_counts().iter_rows())
    summary: dict[str, Any] = {
        "resolved_at": jsonl.now(),
        "judge": judge or "latest",
        "min_confidence": min_confidence,
        "n_crops": decisions.height,
        "decisions": counts,
        "decided_by": dict(
            decisions["decided_by"].str.split(":").list.first().value_counts().iter_rows()
        ),
    }
    paths.summary.write_text(json.dumps(summary, indent=2))
    log.info("resolved", run=str(paths.root), **counts)


def brand_changes(decisions: pl.DataFrame) -> pl.DataFrame:
    """Per brand: crops before, kept, relabelled away / in, removed, pending, after."""
    before: pl.DataFrame = decisions.group_by(pl.col("brand_before").alias("brand")).agg(
        before=pl.len(),
        kept=(pl.col("decision") == "keep").sum(),
        relabelled_out=(pl.col("decision") == "relabel").sum(),
        suggested_out=(pl.col("decision") == "suggested_relabel").sum(),
        removed=(~pl.col("decision").is_in(IN_GALLERY)).sum(),
        pending=(pl.col("decision") == "pending").sum(),
    )
    relabelled_in: pl.DataFrame = (
        decisions.filter(pl.col("decision") == "relabel")
        .group_by(pl.col("brand_after").alias("brand"))
        .agg(relabelled_in=pl.len())
    )
    return (
        before.join(relabelled_in, on="brand", how="full", coalesce=True)
        .fill_null(0)
        .with_columns(pl.exclude("brand").cast(pl.Int64))
        .with_columns(after=pl.col("kept") + pl.col("pending") + pl.col("relabelled_in"))
        .with_columns(delta=pl.col("after") - pl.col("before"))
        .sort("delta")
    )


def new_class_candidates(decisions: pl.DataFrame) -> pl.DataFrame:
    """Off-list brands the judge or a human named: counts, labelled-as brands, examples."""
    return (
        decisions.filter(pl.col("new_class_candidate").is_not_null())
        .group_by(pl.col("new_class_candidate").alias("candidate"))
        .agg(
            n_crops=pl.len(),
            labelled_as=pl.col("brand_before").unique().sort().str.join(", "),
            example_crop_ids=pl.col("crop_id").head(10).str.join(" "),
        )
        .sort("n_crops", descending=True)
    )
