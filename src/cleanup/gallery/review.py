"""Step 4: VLM review of a run's crops, one numbered montage per call.

Crops are grouped by (brand, embedding cluster) so a cluster that is really another brand
shows up as one montage. Noise crops (cluster -1) are grouped by their kNN prediction so
look-alikes still land together. Every call is appended to units.jsonl and every crop
verdict to judgements.jsonl; a re-run skips crops this judge already reviewed.
"""

import hashlib
import json
import re
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl
import structlog
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda
from PIL import Image
from pydantic import ValidationError
from tenacity import RetryCallState, Retrying, retry_if_exception_type, stop_after_attempt

from cleanup.console import progress
from cleanup.errors import InvalidOutputError
from cleanup.gallery import jsonl
from cleanup.gallery.brands import BrandCatalog, load_catalog
from cleanup.gallery.config import GallerySettings, RunPaths
from cleanup.gallery.judge import MontageVerdict, build_messages, system_prompt
from cleanup.gallery.montage import review_sheet
from cleanup.gallery.run import load_analysis, load_run
from cleanup.llm import UsageTally, build_chain

log = structlog.get_logger()

NOISE_CLUSTER = -1
UNSAFE_CHARS = re.compile(r"[^a-z0-9]+")
MONTAGE_QUALITY = 90
# Answers that do not match the schema; another call may well succeed. Auth errors, missing
# models and refusals are not retried.
RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    InvalidOutputError,
    ValidationError,
    OutputParserException,
)


@dataclass(frozen=True)
class ReviewUnit:
    """Crops of one brand reviewed together in one call."""

    unit_id: str
    brand: str
    cluster: int
    crop_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReviewScope:
    """Which of the run's crops to review now."""

    brands: tuple[str, ...] = ()
    top_brands: int | None = None
    max_units: int | None = None
    include_flagged: bool = False
    # Review crops again even if this judge already did; the newest verdict wins.
    rejudge: bool = False


def review_run(settings: GallerySettings, run: str, scope: ReviewScope) -> dict[str, int]:
    """Review the run's pending crops with `settings.judge`; returns token usage."""
    paths, crops = load_run(settings, run)
    pending: pl.DataFrame = pending_crops(crops, paths, settings.judge, scope)
    units: list[ReviewUnit] = order_units(
        build_units(pending, settings.montage_size), paths.priority
    )[: scope.max_units]
    if not units:
        log.info("nothing_to_review", run=run, judge=settings.judge)
        return {}
    catalog: BrandCatalog = load_catalog(
        settings.target_list, settings.aliases, fallback=crops["brand"].unique().to_list()
    )
    references: dict[str, list[str]] = ensure_references(
        settings, paths, sorted({u.brand for u in units})
    )
    chain: Runnable[list[BaseMessage], Any] = build_chain(
        spec=settings.judge, schema=MontageVerdict, effort=settings.judge_effort
    )
    reviewer = UnitReviewer(
        settings=settings,
        paths=paths,
        chain=chain,
        system=system_prompt(catalog.targets),
        references=references,
    )
    log.info("review_start", run=run, judge=settings.judge, units=len(units), crops=pending.height)
    run_parallel(units, reviewer, settings.review_workers, paths)
    usage: dict[str, int] = chain_usage(chain)
    jsonl.append(
        paths.review_log,
        [
            {
                "ts": jsonl.now(),
                "judge": settings.judge,
                "effort": settings.judge_effort,
                "scope": asdict(scope),
                "units": len(units),
                "crops": sum(len(u.crop_ids) for u in units),
                "usage": usage,
            }
        ],
    )
    log.info("review_done", run=run, **usage)
    return usage


def pending_crops(
    crops: pl.DataFrame, paths: RunPaths, judge: str, scope: ReviewScope
) -> pl.DataFrame:
    """Crops in scope that this judge has not reviewed yet (all in scope with `rejudge`)."""
    selected: pl.DataFrame = crops
    if not scope.include_flagged:
        selected = selected.filter(~pl.col("gallery_unusable"))
    if scope.brands:
        selected = selected.filter(pl.col("brand").is_in(list(scope.brands)))
    if scope.top_brands is not None:
        top: list[str] = pl.read_csv(paths.priority).sort("rank")["brand"].to_list()
        in_run: list[str] = [b for b in top if b in set(crops["brand"])][: scope.top_brands]
        selected = selected.filter(pl.col("brand").is_in(in_run))
    if scope.rejudge:
        return selected
    done: set[str] = {r["crop_id"] for r in jsonl.read(paths.judgements) if r["judge"] == judge}
    return selected.filter(~pl.col("crop_id").is_in(list(done)))


def build_units(crops: pl.DataFrame, size: int) -> list[ReviewUnit]:
    """Chunk crops into montages per (brand, cluster); least-agreeing crops first."""
    ordered: pl.DataFrame = crops.with_columns(
        group=pl.when(pl.col("cluster") == NOISE_CLUSTER)
        .then(pl.lit("noise:") + pl.col("knn_pred"))
        .otherwise(pl.lit("cluster:") + pl.col("cluster").cast(pl.String))
    ).sort(["brand", "group", "knn_agreement", "crop_id"])
    units: list[ReviewUnit] = []
    for (brand, _group), frame in ordered.group_by(["brand", "group"], maintain_order=True):
        ids: list[str] = frame["crop_id"].to_list()
        cluster: int = int(frame["cluster"][0])
        units += [
            make_unit(str(brand), cluster, ids[i : i + size]) for i in range(0, len(ids), size)
        ]
    return units


def make_unit(brand: str, cluster: int, crop_ids: list[str]) -> ReviewUnit:
    """Unit with a stable, filename-safe id derived from its crops."""
    digest: str = hashlib.sha1("|".join(crop_ids).encode()).hexdigest()[:10]
    slug: str = UNSAFE_CHARS.sub("-", brand.lower()).strip("-")
    return ReviewUnit(
        unit_id=f"{slug}_c{cluster}_{digest}",
        brand=brand,
        cluster=cluster,
        crop_ids=tuple(crop_ids),
    )


def order_units(units: list[ReviewUnit], priority_csv: Path) -> list[ReviewUnit]:
    """Sort units by their brand's review priority (stable within a brand)."""
    rank: dict[str, int] = dict(pl.read_csv(priority_csv).select("brand", "rank").iter_rows())
    return sorted(units, key=lambda u: rank.get(u.brand, len(rank) + 1))


def ensure_references(
    settings: GallerySettings, paths: RunPaths, brands: list[str]
) -> dict[str, list[str]]:
    """Reference crop ids per brand from references.json, picking missing brands automatically.

    Edit references.json to replace a brand's references (e.g. for heavily polluted brands);
    an empty list sends no references for that brand.
    """
    existing: dict[str, list[str]] = (
        json.loads(paths.references.read_text()) if paths.references.exists() else {}
    )
    missing: list[str] = [b for b in brands if b not in existing]
    if missing:
        existing |= pick_references(load_analysis(settings.gallery_dir), missing, settings)
        paths.references.write_text(json.dumps(existing, indent=2, sort_keys=True))
    return existing


def pick_references(
    analysis: pl.DataFrame, brands: list[str], settings: GallerySettings
) -> dict[str, list[str]]:
    """Large, high-kNN-agreement crops per brand, at most one per video so the references
    show different scenes (and ideally logo variants)."""
    candidates: pl.DataFrame = (
        analysis.filter(
            pl.col("brand").is_in(brands)
            & (pl.col("knn_agreement") >= settings.reference_min_agreement)
            & (pl.min_horizontal("box_w", "box_h") >= settings.reference_min_side_px)
        )
        .sort(["knn_agreement", "knn_top_sim", "crop_id"], descending=[True, True, False])
        .unique(subset=["brand", "video_group"], keep="first", maintain_order=True)
        .group_by("brand", maintain_order=True)
        .head(settings.reference_count)
    )
    picked: dict[str, list[str]] = {b: [] for b in brands}
    for brand, crop_id in candidates.select("brand", "crop_id").iter_rows():
        picked[brand].append(crop_id)
    return picked


@dataclass(frozen=True)
class UnitReviewer:
    """Builds a unit's montages, asks the judge and turns the answer into log rows."""

    settings: GallerySettings
    paths: RunPaths
    chain: Runnable[list[BaseMessage], Any]
    system: str
    references: dict[str, list[str]]

    def __call__(self, unit: ReviewUnit) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """(per-crop judgement rows, unit row). Errors give no crop rows and an error unit."""
        base: dict[str, Any] = {
            "unit_id": unit.unit_id,
            "brand": unit.brand,
            "cluster": unit.cluster,
            "judge": self.settings.judge,
            "ts": jsonl.now(),
        }
        try:
            verdict: MontageVerdict = self.judge(unit)
        except Exception as exc:  # noqa: BLE001 - one failed call must not abort the run
            log.warning("unit_error", unit=unit.unit_id, error=repr(exc))
            return [], base | {"n_tiles": len(unit.crop_ids), "error": repr(exc)}
        rows: list[dict[str, Any]] = tile_rows(unit, verdict, base)
        return rows, base | {
            "n_tiles": len(unit.crop_ids),
            "n_judged": len(rows),
            "cluster_verdict": verdict.cluster_verdict,
            "cluster_suggested_brand": verdict.cluster_suggested_brand,
            "crop_ids": list(unit.crop_ids),
        }

    def judge(self, unit: ReviewUnit) -> MontageVerdict:
        """Render and save the review sheet, then call the judge."""
        s: GallerySettings = self.settings
        sheet: Image.Image = review_sheet(
            crops=self.crop_paths(unit.crop_ids),
            references=self.crop_paths(self.references.get(unit.brand, [])),
            brand=unit.brand,
            tile_px=s.tile_px,
            cols=s.montage_cols,
        )
        self.save(unit.unit_id, sheet)
        messages: list[BaseMessage] = build_messages(
            self.system, unit.brand, sheet, len(unit.crop_ids)
        )
        retrying = Retrying(
            retry=retry_if_exception_type(RETRYABLE_ERRORS),
            stop=stop_after_attempt(s.judge_retries + 1),
            before_sleep=lambda state: log_retry(unit.unit_id, state),
            reraise=True,
        )
        return retrying(self.ask, messages)

    def ask(self, messages: list[BaseMessage]) -> MontageVerdict:
        """One judge call, validated against the montage schema."""
        return MontageVerdict.model_validate(self.chain.invoke(input=messages))

    def crop_paths(self, crop_ids: tuple[str, ...] | list[str]) -> list[Path]:
        """Local crop files."""
        return [self.settings.crops_dir / f"{c}.jpg" for c in crop_ids]

    def save(self, unit_id: str, sheet: Image.Image) -> None:
        """Keep exactly what the judge saw, for the review UI."""
        self.paths.montages.mkdir(parents=True, exist_ok=True)
        sheet.save(self.paths.montages / f"{unit_id}.jpg", quality=MONTAGE_QUALITY)


def tile_rows(
    unit: ReviewUnit, verdict: MontageVerdict, base: dict[str, Any]
) -> list[dict[str, Any]]:
    """One row per judged crop. Out-of-range or repeated tile numbers are ignored; crops the
    judge skipped get no row and stay pending for the next run."""
    by_tile: dict[int, Any] = {}
    for tile in verdict.tiles:
        if 1 <= tile.tile <= len(unit.crop_ids) and tile.tile not in by_tile:
            by_tile[tile.tile] = tile
    return [
        base
        | {
            "crop_id": unit.crop_ids[n - 1],
            "tile": n,
            "verdict": tile.verdict,
            "visible_text": tile.visible_text,
            "suggested_brand": tile.suggested_brand,
            "confidence": tile.confidence,
            "cluster_verdict": verdict.cluster_verdict,
        }
        for n, tile in sorted(by_tile.items())
    ]


def run_parallel(
    units: list[ReviewUnit],
    reviewer: Callable[[ReviewUnit], tuple[list[dict[str, Any]], dict[str, Any]]],
    workers: int,
    paths: RunPaths,
) -> None:
    """Review units concurrently, appending results as each one finishes."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: list[Future[tuple[list[dict[str, Any]], dict[str, Any]]]] = [
            pool.submit(reviewer, u) for u in units
        ]
        for future in progress(completed(futures), desc="review", total=len(units), unit="unit"):
            rows, unit_row = future.result()
            jsonl.append(paths.judgements, rows)
            jsonl.append(paths.units, [unit_row])


def completed(futures: list[Future[Any]]) -> Iterator[Future[Any]]:
    """Futures in completion order."""
    yield from as_completed(futures)


def chain_usage(chain: Runnable[list[BaseMessage], Any]) -> dict[str, int]:
    """Token usage of the run so far (Claude, Bedrock, OpenAI); empty for Ollama."""
    func: Any = chain.func if isinstance(chain, RunnableLambda) else None
    return dict(func.usage) if isinstance(func, UsageTally) else {}


def log_retry(unit_id: str, state: RetryCallState) -> None:
    """Log a judge retry with the error that triggered it."""
    error: BaseException | None = state.outcome.exception() if state.outcome else None
    log.warning("unit_retry", unit=unit_id, attempt=state.attempt_number, error=repr(error))
