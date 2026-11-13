"""Measure the judge and the deterministic flags against human labels from the UI.

Run on the hand-labelled calibration sample before trusting a judge at scale. The numbers to
watch are the judge's precision on `different_brand` and `unrecognisable`: those are the
verdicts that remove or move crops.
"""

import json
from typing import Any

import polars as pl
import structlog

from cleanup.gallery import jsonl
from cleanup.gallery.brands import BrandCatalog, load_catalog
from cleanup.gallery.config import GallerySettings
from cleanup.gallery.judge import TILE_LABELS
from cleanup.gallery.resolve import latest_human_labels
from cleanup.gallery.run import load_run

log = structlog.get_logger()

# Human verdicts that mean "this crop does not belong in the gallery".
GALLERY_BAD: tuple[str, ...] = ("unrecognisable", "not_a_logo", "different_brand")


def calibrate_run(settings: GallerySettings, run: str) -> dict[str, Any]:
    """Write calibration.json: per-judge confusion and precision/recall, per-flag precision."""
    paths, crops = load_run(settings, run)
    human: dict[str, dict[str, Any]] = latest_human_labels(paths)
    if not human:
        raise ValueError(f"no human labels in {paths.events}; label crops in the UI first")
    catalog: BrandCatalog = load_catalog(
        settings.target_list, settings.aliases, fallback=crops["brand"].unique().to_list()
    )
    judgements: list[dict[str, Any]] = jsonl.read(paths.judgements)
    report: dict[str, Any] = {
        "n_human_labels": len(human),
        "judges": {
            judge: judge_report(
                jsonl.latest_by([r for r in judgements if r["judge"] == judge], "crop_id"),
                human,
                catalog,
            )
            for judge in sorted({r["judge"] for r in judgements})
        },
        "flags": flag_report(crops, human),
    }
    paths.calibration.write_text(json.dumps(report, indent=2))
    log.info("calibrated", run=run, path=str(paths.calibration))
    return report


def judge_report(
    verdicts: dict[str, dict[str, Any]],
    human: dict[str, dict[str, Any]],
    catalog: BrandCatalog,
) -> dict[str, Any]:
    """Confusion matrix (human rows x judge columns), per-verdict precision/recall, and how
    often a `different_brand` call named the same brand as the human."""
    both: list[str] = sorted(set(verdicts) & set(human))
    pairs: list[tuple[str, str]] = [(human[c]["verdict"], verdicts[c]["verdict"]) for c in both]
    matrix: dict[str, dict[str, int]] = {
        h: {j: sum(p == (h, j) for p in pairs) for j in TILE_LABELS} for h in TILE_LABELS
    }
    brand_hits: list[bool] = [
        catalog.resolve(verdicts[c].get("suggested_brand") or "")
        == catalog.resolve(human[c].get("suggested_brand") or "")
        for c in both
        if human[c]["verdict"] == verdicts[c]["verdict"] == "different_brand"
    ]
    return {
        "n_compared": len(both),
        "accuracy": ratio(sum(h == j for h, j in pairs), len(pairs)),
        "confusion": matrix,
        "per_verdict": {label: precision_recall(pairs, label) for label in TILE_LABELS},
        "different_brand_name_accuracy": ratio(sum(brand_hits), len(brand_hits)),
    }


def precision_recall(pairs: list[tuple[str, str]], label: str) -> dict[str, float | int | None]:
    """Precision and recall of one verdict, treating the human label as truth."""
    predicted: int = sum(j == label for _, j in pairs)
    actual: int = sum(h == label for h, _ in pairs)
    hits: int = sum(h == j == label for h, j in pairs)
    return {
        "precision": ratio(hits, predicted),
        "recall": ratio(hits, actual),
        "n_predicted": predicted,
        "n_actual": actual,
    }


def flag_report(crops: pl.DataFrame, human: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per flag: how many labelled crops it fired on and the share humans called bad."""
    labelled: pl.DataFrame = crops.filter(pl.col("crop_id").is_in(list(human))).with_columns(
        human_bad=pl.col("crop_id").replace_strict(
            {c: h["verdict"] in GALLERY_BAD for c, h in human.items()}, return_dtype=pl.Boolean
        )
    )
    flag_cols: list[str] = [c for c in labelled.columns if c.startswith("flag_")]
    return {
        col.removeprefix("flag_"): {
            "n_flagged": int(labelled[col].sum()),
            "precision_bad": ratio(
                int((labelled[col] & labelled["human_bad"]).sum()), int(labelled[col].sum())
            ),
        }
        for col in flag_cols
    } | {
        "gallery_unusable": {
            "n_flagged": int(labelled["gallery_unusable"].sum()),
            "precision_bad": ratio(
                int((labelled["gallery_unusable"] & labelled["human_bad"]).sum()),
                int(labelled["gallery_unusable"].sum()),
            ),
            "base_rate_bad": ratio(int(labelled["human_bad"].sum()), labelled.height),
        }
    }


def ratio(numerator: int, denominator: int) -> float | None:
    """numerator / denominator rounded to 3 places; None when the denominator is 0."""
    return round(numerator / denominator, 3) if denominator else None
