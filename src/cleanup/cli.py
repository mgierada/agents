"""Command-line entry point."""

from pathlib import Path
from typing import Annotated, Any

import typer

from cleanup.config import Settings
from cleanup.graph import build_graph

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    data_dir: Annotated[Path | None, typer.Option(help="Dataset root (images/, labels/)")] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Where the clean copy goes")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Report only, copy nothing")] = False,
) -> None:
    """Run the cleanup pipeline on a YOLO dataset."""
    overrides: dict[str, Any] = {"data_dir": data_dir, "output_dir": output_dir}
    if dry_run:
        overrides["dry_run"] = True
    settings = Settings(**{k: v for k, v in overrides.items() if v is not None})
    result: dict[str, Any] = build_graph().invoke({"settings": settings, "records": []})
    typer.echo(f"summary: {result.get('summary', {})}")
    typer.echo(f"report: {settings.output_dir / 'report.json'}")


@app.command()
def graph() -> None:
    """Print the pipeline graph as Mermaid."""
    typer.echo(build_graph().get_graph().draw_mermaid())
