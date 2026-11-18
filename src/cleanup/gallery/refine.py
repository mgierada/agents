"""Step 6: a refined gallery from a run's resolved decisions.

Writes a new logo-audit dir next to the source one, in the layout highfive-models'
`logo-audit` (highfive_models/data_processing/fiftyone/main.py) produces after its crops and
embed stages: manifest.parquet, emb_<backbone>.npy (rows aligned with the manifest) and crops/.
Only crops whose decision is `keep` or `relabel` go in, under their brand after the resolve.
Embeddings are sliced from the source (a relabel does not change the image), crops/ is a
symlink to the source's, and nothing of the source is modified.

The kNN analysis (summary.json, analysis.parquet with suspect/cluster, suspects, duplicates,
contact sheets, brand scaling) depends on the new labels, so it is left to highfive-models'
`analyze` stage, run on the new dir (see `analyze_command`).
"""

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog

from cleanup.gallery import jsonl
from cleanup.gallery.config import CROPS_DIR_NAME, GallerySettings, RunPaths
from cleanup.gallery.progress import Progress, no_progress, report, scaled

log = structlog.get_logger()

MANIFEST_NAME = "manifest.parquet"
REFINE_NAME = "refine.json"
EMBEDDING_GLOB = "emb_*.npy"
# Decisions whose crop is in the refined gallery: confirmed by a judge or by you.
REFINED_DECISIONS: tuple[str, ...] = ("keep", "relabel")
SAFE_GALLERY_NAME = re.compile(r"^[A-Za-z0-9._\-]+$")
# Embedding rows copied per slice, bounding memory (768 float16 dims -> ~150 MB).
EMB_CHUNK_ROWS = 100_000
# Share of the run time per stage: copying embedding rows (~1 GB) dominates.
STAGES: dict[str, tuple[float, float]] = {
    "reading manifest": (0.0, 0.05),
    "writing manifest": (0.05, 0.15),
    "slicing embeddings": (0.15, 0.98),
    "linking crops": (0.98, 1.0),
}


def refine_gallery(
    settings: GallerySettings, run: str, name: str, progress: Progress = no_progress
) -> Path:
    """Write the refined gallery `name` (a sibling of the source gallery) and return its dir.

    The run must be resolved and cover the whole source gallery. Raises FileExistsError
    when the target exists, so an earlier refined gallery is never overwritten. `progress`
    gets the stage and overall fraction done (embeddings: every chunk).
    """
    source: Path = settings.gallery_dir.resolve()
    target: Path = refined_dir(source=source, name=name)
    paths: RunPaths = settings.run_paths(run)
    report(progress=progress, stages=STAGES, stage="reading manifest", fraction=0.0)
    manifest: pl.DataFrame = pl.read_parquet(source / MANIFEST_NAME)
    decisions: pl.DataFrame = load_decisions(paths=paths, expected=manifest.height)
    warn_if_stale(paths)
    report(progress=progress, stages=STAGES, stage="writing manifest", fraction=0.0)
    refined, rows = refined_manifest(
        manifest=manifest, decisions=decisions, source_name=source.name, name=name
    )
    target.mkdir()
    refined.write_parquet(target / MANIFEST_NAME)
    embeddings: list[Path] = sorted(source.glob(EMBEDDING_GLOB))
    backbones: list[str] = [
        slice_embeddings(
            source=emb,
            target=target / emb.name,
            rows=rows,
            progress=scaled(
                progress=lambda stage, f: report(
                    progress=progress, stages=STAGES, stage=stage, fraction=f
                ),
                start=i / len(embeddings),
                end=(i + 1) / len(embeddings),
            ),
        )
        for i, emb in enumerate(embeddings)
    ]
    report(progress=progress, stages=STAGES, stage="linking crops", fraction=0.0)
    (target / CROPS_DIR_NAME).symlink_to(os.path.relpath(source / CROPS_DIR_NAME, target))
    info: dict[str, Any] = provenance(run, source, paths, decisions, refined, backbones)
    (target / REFINE_NAME).write_text(json.dumps(info, indent=2))
    report(progress=progress, stages=STAGES, stage="linking crops", fraction=1.0)
    log.info("refined", gallery=str(target), crops=refined.height, of=manifest.height)
    return target


def refined_dir(source: Path, name: str) -> Path:
    """Target dir `name` next to `source`; ValueError for path-like names, error if taken."""
    if not SAFE_GALLERY_NAME.match(name) or name in {".", ".."}:
        raise ValueError(f"gallery name must be a plain dir name, got {name!r}")
    target: Path = source.parent / name
    if target.exists():
        raise FileExistsError(f"{target} exists; pick another --name or remove it")
    return target


def load_decisions(paths: RunPaths, expected: int) -> pl.DataFrame:
    """The run's decisions; errors when it is not resolved or does not cover the gallery."""
    if not paths.decisions.exists():
        raise FileNotFoundError(f"{paths.decisions} missing; run `cleanup gallery resolve`")
    decisions: pl.DataFrame = pl.read_parquet(paths.decisions)
    if decisions.height != expected:
        raise ValueError(
            f"run covers {decisions.height} of {expected} gallery crops; refine needs a run "
            "over the whole gallery (crops outside it would be neither kept nor dropped)"
        )
    return decisions


def warn_if_stale(paths: RunPaths) -> None:
    """Log a warning when verdicts or labels changed after the last resolve."""
    resolved: float = paths.decisions.stat().st_mtime
    newer: list[str] = [
        p.name
        for p in (paths.judgements, paths.events)
        if p.exists() and p.stat().st_mtime > resolved
    ]
    if newer:
        log.warning("resolve_is_stale", changed_since=newer, hint="re-run resolve first")


def refined_manifest(
    manifest: pl.DataFrame, decisions: pl.DataFrame, source_name: str, name: str
) -> tuple[pl.DataFrame, np.ndarray]:
    """Kept manifest rows (source order) with brand_after, and their source row indices.

    crop_path is moved under the new dir (`/<source>/crops/` -> `/<name>/crops/`), which
    resolves through the crops symlink wherever both dirs sit side by side.
    """
    joined: pl.DataFrame = (
        manifest.with_row_index("_row")
        .join(decisions.select("crop_id", "decision", "brand_after"), on="crop_id", how="left")
        .filter(pl.col("decision").is_in(REFINED_DECISIONS))
    )
    rows: np.ndarray = joined["_row"].to_numpy()
    refined: pl.DataFrame = joined.with_columns(
        brand=pl.col("brand_after"),
        crop_path=pl.col("crop_path").str.replace(
            f"/{source_name}/{CROPS_DIR_NAME}/", f"/{name}/{CROPS_DIR_NAME}/", literal=True
        ),
    ).select(manifest.columns)
    return refined, rows


def slice_embeddings(
    source: Path, target: Path, rows: np.ndarray, progress: Progress = no_progress
) -> str:
    """Copy `rows` of a .npy embedding file into `target` in chunks; returns the backbone.

    `progress("slicing embeddings", fraction of this file)` after every chunk.
    """
    emb: np.ndarray = np.load(source, mmap_mode="r")
    out: np.ndarray = np.lib.format.open_memmap(
        target, mode="w+", dtype=emb.dtype, shape=(len(rows), *emb.shape[1:])
    )
    for start in range(0, len(rows), EMB_CHUNK_ROWS):
        out[start : start + EMB_CHUNK_ROWS] = emb[rows[start : start + EMB_CHUNK_ROWS]]
        progress("slicing embeddings", min(1.0, (start + EMB_CHUNK_ROWS) / len(rows)))
    out.flush()
    progress("slicing embeddings", 1.0)
    return source.stem.removeprefix("emb_")


def provenance(
    run: str,
    source: Path,
    paths: RunPaths,
    decisions: pl.DataFrame,
    refined: pl.DataFrame,
    backbones: list[str],
) -> dict[str, Any]:
    """refine.json: where the gallery came from, what was kept, and the next command."""
    resolve_summary: dict[str, Any] = (
        json.loads(paths.summary.read_text()) if paths.summary.exists() else {}
    )
    return {
        "created_at": jsonl.now(),
        "source_gallery": str(source),
        "run": run,
        "resolve": {k: resolve_summary.get(k) for k in ("resolved_at", "judge", "min_confidence")},
        "kept_decisions": list(REFINED_DECISIONS),
        "n_source": decisions.height,
        "n_refined": refined.height,
        "decisions": dict(decisions["decision"].value_counts().iter_rows()),
        "n_brands": refined["brand"].n_unique(),
        "backbones": backbones,
        "analyze_command": analyze_command(source.name, refined, backbones),
    }


def analyze_command(source_name: str, refined: pl.DataFrame, backbones: list[str]) -> str:
    """highfive-models command recomputing the kNN outputs of the refined dir.

    Runs where crop_path resolves (the training host): the output dir is taken from the
    refined crop_path. `logo-audit` needs the source batch name only to check that the batch
    exists; with `--stage analyze` it reads and writes nothing but the output dir.
    """
    output_dir: Path = Path(refined["crop_path"][0]).parent.parent if refined.height else Path()
    batch: str = source_name.removeprefix("logo_audit_")
    flags: str = " ".join(f"--backbones {b}" for b in backbones)
    return (
        "uv run highfive_models/data_processing/fiftyone/main.py logo-audit "
        f"{batch} --stage analyze --output-dir {output_dir} {flags}"
    )
