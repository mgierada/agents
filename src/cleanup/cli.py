"""Command-line entry point."""

from pathlib import Path
from typing import Annotated, Any

import typer

from cleanup.config import Settings
from cleanup.graph import build_graph
from cleanup.judge import Judge
from cleanup.vlm import ollama_judge

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    data_dir: Annotated[Path | None, typer.Option(help="Dataset root (images/, labels/)")] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Where the clean copy goes")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Report only, copy nothing")] = False,
    judge_model: Annotated[str | None, typer.Option(help="Ollama VLM, e.g. qwen2.5vl:7b")] = None,
    judge_scope: Annotated[str | None, typer.Option(help="borderline | survivors")] = None,
) -> None:
    """Run the cleanup pipeline on a YOLO dataset."""
    overrides: dict[str, Any] = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "judge_model": judge_model,
        "judge_scope": judge_scope,
    }
    if dry_run:
        overrides["dry_run"] = True
    settings = Settings(**{k: v for k, v in overrides.items() if v is not None})
    result: dict[str, Any] = build_graph(judge=build_judge(settings)).invoke(
        {"settings": settings, "records": []}
    )
    typer.echo(f"summary: {result.get('summary', {})}")
    typer.echo(f"report: {settings.output_dir / 'report.json'}")


@app.command()
def graph() -> None:
    """Print the pipeline graph as Mermaid."""
    typer.echo(build_graph().get_graph().draw_mermaid())


def build_judge(settings: Settings) -> Judge | None:
    """Build the VLM judge if a model is configured, else None (policy fallback)."""
    if settings.judge_model is None:
        return None
    return ollama_judge(
        settings.judge_model,
        fallback_keep=settings.borderline_policy == "keep",
        max_edge=settings.judge_max_edge,
    )
