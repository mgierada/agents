"""Charts of calibration.json: how each judge and each flag compares with your labels.

Writes PNGs to `<run>/plots/`. Colours follow one rule per job: judges are categorical (fixed
slot order, validated for colour-blind separation), confusion cells are sequential (one hue).
Two of the judge colours sit below 3:1 contrast on white, so every bar carries its value.
"""

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # files only, no display needed

import matplotlib.pyplot as plt  # noqa: E402 - backend must be set first
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import structlog  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.container import BarContainer  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from cleanup.gallery.config import GallerySettings  # noqa: E402
from cleanup.gallery.judge import TILE_LABELS  # noqa: E402
from cleanup.gallery.run import load_run  # noqa: E402

log = structlog.get_logger()

# Reference categorical palette, fixed order (never cycled); validated for 4 adjacent slots.
JUDGE_COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MAX_JUDGES = len(JUDGE_COLORS)
SINGLE_SERIES = JUDGE_COLORS[0]
# Sequential blue ramp, near-zero recedes toward the surface.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
DPI = 150
FLAG_NAMES: dict[str, str] = {"gallery_unusable": "any hard flag (gallery_unusable)"}
# Below this many compared crops a judge's accuracy is too noisy to rank; drawn hatched.
MIN_RELIABLE_N = 30
VENDOR_PREFIX = re.compile(r"^((global|us|eu|apac)\.)?(anthropic|qwen|amazon|meta|mistral)\.")


def plot_run(settings: GallerySettings, run: str) -> list[Path]:
    """Render every chart for the run's calibration.json; returns the written files."""
    paths, _ = load_run(settings, run)
    if not paths.calibration.exists():
        raise FileNotFoundError(f"{paths.calibration} missing; run `cleanup gallery calibrate`")
    report: dict[str, Any] = json.loads(paths.calibration.read_text())
    out: Path = paths.root / "plots"
    out.mkdir(exist_ok=True)
    apply_theme()
    judges: list[str] = list(report["judges"])
    if len(judges) > MAX_JUDGES:
        log.warning("too_many_judges", shown=judges[:MAX_JUDGES], dropped=judges[MAX_JUDGES:])
        judges = judges[:MAX_JUDGES]
    figures: dict[str, Figure] = {
        "confusion.png": confusion_figure(report, judges),
        "precision_recall.png": precision_recall_figure(report, judges),
        "accuracy.png": accuracy_figure(report, judges),
        "flags.png": flags_figure(report),
    }
    written: list[Path] = []
    for name, figure in figures.items():
        figure.savefig(out / name, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
        plt.close(figure)
        written.append(out / name)
    log.info("plotted", run=run, dir=str(out), files=len(written))
    return written


def apply_theme() -> None:
    """Quiet axes and grid, text in ink colours rather than series colours."""
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "grid.color": GRID,
            "text.color": TEXT,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.titleweight": "bold",
        },
    )


def short_judge(spec: str) -> str:
    """Readable judge label: vendor and routing prefixes dropped, provider in brackets.

    `bedrock-converse:global.anthropic.claude-haiku-5-5` -> `claude-haiku-5-5 (bedrock-converse)`
    """
    provider, _, model = spec.partition(":")
    return f"{VENDOR_PREFIX.sub('', model)} ({provider})"


def confusion_figure(report: dict[str, Any], judges: list[str]) -> Figure:
    """One heatmap per judge: rows = your label, columns = judge verdict.

    Cells show the share of each of your labels (row-normalised) plus the raw count, so a judge
    is compared on how it treats each kind of crop regardless of how common that kind is.
    """
    figure, axes = plt.subplots(
        1, len(judges), figsize=(4.2 * len(judges), 4.4), squeeze=False, sharey=True
    )
    for ax, judge in zip(axes[0], judges, strict=True):
        counts: pd.DataFrame = pd.DataFrame(report["judges"][judge]["confusion"]).T.loc[
            list(TILE_LABELS), list(TILE_LABELS)
        ]
        shares: pd.DataFrame = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
        labels: pd.DataFrame = shares.map(lambda v: f"{v:.0%}") + counts.map(lambda n: f"\n{n}")
        sns.heatmap(
            shares,
            ax=ax,
            cmap=SEQUENTIAL,
            vmin=0,
            vmax=1,
            annot=labels,
            fmt="",
            cbar=False,
            linewidths=2,
            linecolor=SURFACE,
            annot_kws={"fontsize": 9},
        )
        info: dict[str, Any] = report["judges"][judge]
        ax.set_title(
            f"{short_judge(judge)}\naccuracy {pct(info['accuracy'])}, n={info['n_compared']}",
            fontsize=10,
        )
        ax.set_xlabel("judge verdict")
        ax.set_ylabel("your label" if ax is axes[0][0] else "")
        ax.tick_params(axis="x", rotation=30)
    figure.suptitle("Judge vs your labels (row share, count)", fontweight="bold")
    figure.tight_layout()
    return figure


def precision_recall_figure(report: dict[str, Any], judges: list[str]) -> Figure:
    """Precision and recall per verdict, one bar per judge.

    Precision on different_brand / unrecognisable / not_a_logo is what matters most: those
    verdicts move or remove crops.
    """
    rows: list[dict[str, Any]] = [
        {"judge": short_judge(judge), "verdict": verdict, "metric": metric, "value": stats[metric]}
        for judge in judges
        for verdict, stats in report["judges"][judge]["per_verdict"].items()
        for metric in ("precision", "recall")
        if stats[metric] is not None
    ]
    data = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    palette: dict[str, str] = judge_palette(judges)
    for ax, metric in zip(axes, ("precision", "recall"), strict=True):
        sns.barplot(
            data=data[data["metric"] == metric],
            x="verdict",
            y="value",
            hue="judge",
            hue_order=list(palette),
            order=list(TILE_LABELS),
            palette=palette,
            ax=ax,
            width=0.8,
            gap=0.1,
            legend=ax is axes[0],
        )
        label_bars(ax)
        ax.set_title(metric)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel("")
        ax.set_ylabel("share" if ax is axes[0] else "")
    axes[0].legend(
        title="judge", loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, frameon=False
    )
    figure.suptitle("Per-verdict precision and recall (your labels = truth)", fontweight="bold")
    figure.tight_layout()
    return figure


def accuracy_figure(report: dict[str, Any], judges: list[str]) -> Figure:
    """Overall agreement with your labels, one bar per judge."""
    data = pd.DataFrame(
        {
            "judge": [short_judge(j) for j in judges],
            "accuracy": [report["judges"][j]["accuracy"] or 0 for j in judges],
            "n": [report["judges"][j]["n_compared"] for j in judges],
        }
    ).sort_values("accuracy", ascending=False)
    figure, ax = plt.subplots(figsize=(8, 0.6 * len(judges) + 1.4))
    sns.barplot(data=data, x="accuracy", y="judge", color=SINGLE_SERIES, ax=ax, width=0.6)
    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(
                container,
                labels=[
                    f"{a:.0%}  (n={n})" for a, n in zip(data["accuracy"], data["n"], strict=True)
                ],
                padding=4,
                color=TEXT,
                fontsize=9,
            )
    for bar, n in zip(ax.patches, data["n"], strict=False):
        if n < MIN_RELIABLE_N:
            bar.set_hatch("///")
            bar.set_facecolor(SURFACE)
            bar.set_edgecolor(SINGLE_SERIES)
    if (data["n"] < MIN_RELIABLE_N).any():
        figure.text(
            0.01,
            0.0,
            f"Hatched: fewer than {MIN_RELIABLE_N} crops compared, too few to rank.",
            color=MUTED,
            fontsize=8,
        )
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("share of labelled crops where the judge agrees with you")
    ax.set_ylabel("")
    ax.set_title("Judge accuracy")
    figure.tight_layout()
    return figure


def flags_figure(report: dict[str, Any]) -> Figure:
    """Per flag: share of flagged crops you labelled bad, against the base rate.

    A flag well above the base-rate line is a good removal rule; one near or below it removes
    good crops about as often as a random pick would.
    """
    flags: dict[str, Any] = report["flags"]
    base: float | None = flags.get("gallery_unusable", {}).get("base_rate_bad")
    data = pd.DataFrame(
        [
            {
                "flag": FLAG_NAMES.get(name, name),
                "precision_bad": stats["precision_bad"],
                "n": stats["n_flagged"],
            }
            for name, stats in flags.items()
            if stats["n_flagged"]
        ]
    )
    if data.empty:
        return empty_figure("Flag precision vs base rate", "No labelled crop has a flag yet.")
    data = data.sort_values("precision_bad", ascending=False)
    figure, ax = plt.subplots(figsize=(8, 0.42 * len(data) + 1.6))
    sns.barplot(data=data, x="precision_bad", y="flag", color=SINGLE_SERIES, ax=ax, width=0.6)
    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(
                container,
                labels=[
                    f"{p:.0%}  (n={n})"
                    for p, n in zip(data["precision_bad"], data["n"], strict=True)
                ],
                padding=4,
                color=TEXT,
                fontsize=9,
            )
    if base is not None:
        ax.axvline(
            base,
            color=MUTED,
            linestyle="--",
            linewidth=1.5,
            label=f"base rate: {base:.0%} of all labelled crops are bad",
        )
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel(
        "share of flagged crops you labelled bad (unrecognisable, not a logo, other brand)"
    )
    ax.set_ylabel("")
    ax.set_title("Flag precision vs base rate")
    figure.tight_layout()
    return figure


def empty_figure(title: str, message: str) -> Figure:
    """Placeholder chart when there is nothing to plot yet."""
    figure, ax = plt.subplots(figsize=(8, 2))
    ax.set_axis_off()
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED, transform=ax.transAxes)
    return figure


def judge_palette(judges: list[str]) -> dict[str, str]:
    """Judge label -> colour, in calibration.json order (color follows the judge, not rank)."""
    return {short_judge(j): c for j, c in zip(judges, JUDGE_COLORS, strict=False)}


def label_bars(ax: Axes) -> None:
    """Value on top of every bar (needed: two judge colours are low-contrast on white)."""
    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.2f", padding=2, fontsize=7, color=TEXT)


def pct(value: float | None) -> str:
    """0.758 -> '76%'; None -> 'n/a'."""
    return "n/a" if value is None else f"{value:.0%}"
