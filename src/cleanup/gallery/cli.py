"""`cleanup gallery ...`: audit a logo-identification gallery in file-backed steps.

init       pick crops (whole gallery, subset or calibration sample), flags, priority
review     VLM verdicts on montages (Claude or Ollama)
resolve    flags + judge + your labels -> decisions, before/after per brand
calibrate  judge and flag precision against your labels
ui         label crops, compare before/after, browse the change log
"""

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from cleanup.console import configure_logging
from cleanup.gallery.calibrate import calibrate_run
from cleanup.gallery.config import GallerySettings, RunPaths
from cleanup.gallery.resolve import CONFIDENCE_RANK, resolve_run
from cleanup.gallery.review import ReviewScope, review_run
from cleanup.gallery.run import init_run
from cleanup.gallery.server import serve
from cleanup.gallery.subset import Subset

app = typer.Typer(no_args_is_help=True, help=__doc__)

GalleryDir = Annotated[
    Path | None, typer.Option(help="Logo-audit dir (analysis.parquet, crops/, labels)")
]
RunsDir = Annotated[Path | None, typer.Option(help="Where runs go [<gallery-dir>/agent_audit]")]
TargetList = Annotated[
    Path | None, typer.Option(help="Target list, one brand per line [gallery brands]")
]
Aliases = Annotated[Path | None, typer.Option(help='Alias JSON {"old": "new"}')]
Run = Annotated[str, typer.Option("--run", help="Run name")]
Brands = Annotated[str | None, typer.Option(help="Comma-separated brands")]
TopBrands = Annotated[int | None, typer.Option(help="Only the N highest-priority brands")]


def settings_from(**overrides: Any) -> GallerySettings:
    """Settings with CLI values taking precedence over env / .env (None = not given)."""
    return GallerySettings(**{k: v for k, v in overrides.items() if v is not None})


def split(value: str | None) -> tuple[str, ...]:
    """'a, b' -> ('a', 'b')."""
    return tuple(v.strip() for v in value.split(",") if v.strip()) if value else ()


@app.command()
def init(
    run: Run,
    gallery_dir: GalleryDir = None,
    runs_dir: RunsDir = None,
    target_list: TargetList = None,
    aliases: Aliases = None,
    brands: Brands = None,
    top_brands: TopBrands = None,
    crop_ids: Annotated[Path | None, typer.Option(help="File of crop ids to keep")] = None,
    suspects_only: Annotated[bool, typer.Option(help="Only kNN mislabel suspects")] = False,
    calibration_sample: Annotated[
        int | None, typer.Option(help="Stratified sample of N crops for hand labelling")
    ] = None,
    frac: Annotated[float | None, typer.Option(help="Random fraction of crops")] = None,
    limit: Annotated[int | None, typer.Option(help="At most N random crops")] = None,
    seed: Annotated[int, typer.Option(help="Sampling seed")] = 0,
) -> None:
    """Steps 1-3: select crops, compute deterministic flags and the brand priority."""
    settings: GallerySettings = settings_from(
        gallery_dir=gallery_dir, runs_dir=runs_dir, target_list=target_list, aliases=aliases
    )
    configure_logging()
    subset = Subset(
        crop_ids_file=crop_ids,
        brands=split(value=brands),
        top_brands=top_brands,
        suspects_only=suspects_only,
        calibration_sample=calibration_sample,
        frac=frac,
        limit=limit,
        seed=seed,
    )
    paths: RunPaths = init_run(settings=settings, run=run, subset=subset)
    typer.echo(f"run: {paths.root}")


@app.command()
def review(
    run: Run,
    judge: Annotated[
        str | None,
        typer.Option(help="claude:<model> | bedrock:<model> | openai:<model> | ollama:<model>"),
    ] = None,
    effort: Annotated[str | None, typer.Option(help="Claude effort: low|medium|high")] = None,
    gallery_dir: GalleryDir = None,
    runs_dir: RunsDir = None,
    target_list: TargetList = None,
    aliases: Aliases = None,
    brands: Brands = None,
    top_brands: TopBrands = None,
    max_units: Annotated[int | None, typer.Option(help="Stop after N montages (budget)")] = None,
    include_flagged: Annotated[
        bool, typer.Option(help="Also review crops the flags already removed")
    ] = False,
    rejudge: Annotated[
        bool,
        typer.Option(help="Re-review crops this judge already did; the newest verdict wins"),
    ] = False,
    workers: Annotated[int | None, typer.Option(help="Concurrent judge calls")] = None,
) -> None:
    """Step 4: judge pending crops of the run, montage by montage. Safe to re-run."""
    settings: GallerySettings = settings_from(
        gallery_dir=gallery_dir,
        runs_dir=runs_dir,
        target_list=target_list,
        aliases=aliases,
        judge=judge,
        judge_effort=effort,
        review_workers=workers,
    )
    configure_logging()
    scope = ReviewScope(
        brands=split(value=brands),
        top_brands=top_brands,
        max_units=max_units,
        include_flagged=include_flagged,
        rejudge=rejudge,
    )
    usage: dict[str, int] = review_run(settings, run, scope)
    typer.echo(f"usage: {usage}")


@app.command()
def resolve(
    run: Run,
    judge: Annotated[str | None, typer.Option(help="Use this judge's verdicts [latest]")] = None,
    min_confidence: Annotated[
        str, typer.Option(help="Judge verdicts below this stay pending: low|medium|high")
    ] = "low",
    gallery_dir: GalleryDir = None,
    runs_dir: RunsDir = None,
    target_list: TargetList = None,
    aliases: Aliases = None,
) -> None:
    """Step 5: decisions per crop, before/after per brand, new-class candidates."""
    if min_confidence not in CONFIDENCE_RANK:
        raise typer.BadParameter(f"min-confidence must be one of {list(CONFIDENCE_RANK)}")
    settings: GallerySettings = settings_from(
        gallery_dir=gallery_dir, runs_dir=runs_dir, target_list=target_list, aliases=aliases
    )
    configure_logging()
    resolve_run(settings, run, judge=judge, min_confidence=min_confidence)
    paths = settings.run_paths(run)
    typer.echo(paths.summary.read_text())


@app.command()
def calibrate(
    run: Run,
    gallery_dir: GalleryDir = None,
    runs_dir: RunsDir = None,
    target_list: TargetList = None,
    aliases: Aliases = None,
) -> None:
    """Judge and flag precision against your labels (label in `ui` first)."""
    settings: GallerySettings = settings_from(
        gallery_dir=gallery_dir, runs_dir=runs_dir, target_list=target_list, aliases=aliases
    )
    configure_logging()
    report: dict[str, Any] = calibrate_run(settings, run)
    typer.echo(json.dumps(report, indent=2))


@app.command()
def ui(
    run: Run,
    port: Annotated[int, typer.Option(help="Local port")] = 8765,
    gallery_dir: GalleryDir = None,
    runs_dir: RunsDir = None,
    target_list: TargetList = None,
    aliases: Aliases = None,
) -> None:
    """Browser UI on 127.0.0.1: label crops, see before/after, browse every change."""
    settings: GallerySettings = settings_from(
        gallery_dir=gallery_dir, runs_dir=runs_dir, target_list=target_list, aliases=aliases
    )
    configure_logging()
    serve(settings, run, port)
