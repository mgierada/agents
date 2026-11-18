"""Local review UI: label crops, compare before/after, and see every change.

Serves one static page plus a small JSON API on 127.0.0.1. Human labels are appended to the
run's events.jsonl (never rewritten), so the full history of every crop stays visible.
"""

import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import polars as pl
import structlog

from cleanup.gallery import jsonl
from cleanup.gallery.brands import load_catalog
from cleanup.gallery.config import GallerySettings, RunPaths
from cleanup.gallery.judge import TILE_LABELS
from cleanup.gallery.pricing import spend_report
from cleanup.gallery.resolve import Progress, no_progress, resolve_run
from cleanup.gallery.review import judge_one
from cleanup.gallery.run import load_run

log = structlog.get_logger()

SAFE_NAME = re.compile(r"^[A-Za-z0-9._%\-]+$")
DEFAULT_PAGE_SIZE = 60
MAX_PAGE_SIZE = 500
# judgements.jsonl field -> UI column.
JUDGE_FIELDS: dict[str, str] = {
    "judge": "judge",
    "verdict": "judge_verdict",
    "suggested_brand": "judge_suggested_brand",
    "visible_text": "judge_visible_text",
    "confidence": "judge_confidence",
    "unit_id": "judge_unit_id",
    "cluster_verdict": "judge_cluster_verdict",
}
ITEM_COLUMNS: tuple[str, ...] = (
    "crop_id",
    "brand",
    "stratum",
    "cluster",
    "knn_pred",
    "knn_agreement",
    "suspect",
    "box_w",
    "box_h",
    "blur",
    "origin",
    "confidence",
    "flags",
    "gallery_unusable",
)


# Share of a UI resolve spent in resolve_run; the rest reloads the UI tables (~5 s on 1M crops).
RESOLVE_SHARE = 0.9


@dataclass
class ResolveJob:
    """A resolve running in the background for one run, polled by the UI for its progress."""

    status: str = "idle"  # idle | running | done | error
    stage: str = ""
    fraction: float = 0.0
    started: str | None = None
    finished: str | None = None
    error: str | None = None
    summary: dict[str, Any] | None = None
    accepted: int | None = None  # relabels accepted before this resolve, if any
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe state, with the progress as a whole percentage."""
        with self.lock:
            return {
                "status": self.status,
                "stage": self.stage,
                "percent": round(100 * self.fraction),
                "started": self.started,
                "finished": self.finished,
                "error": self.error,
                "summary": self.summary,
                "accepted": self.accepted,
            }

    def update(self, stage: str, fraction: float) -> None:
        """Progress callback for resolve_run."""
        with self.lock:
            self.stage, self.fraction = stage, fraction


@dataclass
class RunState:
    """Everything the UI shows for one run.

    Crops and decisions are loaded once (and again after a resolve); judge verdicts follow
    judgements.jsonl as it grows, so a review running in another process shows up live.
    """

    settings: GallerySettings
    run: str
    paths: RunPaths = field(init=False)
    items: pl.DataFrame = field(init=False)
    targets: list[str] = field(init=False)
    # (file, mtime_ns, size) -> parsed content, so polling large runs stays cheap.
    file_cache: dict[tuple[str, int, int], Any] = field(init=False, default_factory=dict)
    # Crops + decisions without judge columns; verdicts are joined on top in `refresh_judgements`.
    base: pl.DataFrame = field(init=False)
    verdicts: pl.DataFrame = field(init=False)
    # Bytes of judgements.jsonl already folded into `verdicts`.
    judged_offset: int = field(init=False, default=0)
    lock: threading.RLock = field(init=False, default_factory=threading.RLock, repr=False)
    resolve_job: ResolveJob = field(init=False, default_factory=ResolveJob)

    def __post_init__(self) -> None:
        """Load the run's tables."""
        self.reload()

    def reload(self) -> None:
        """Join crops, flags and the last resolve's decisions, then every judge verdict."""
        with self.lock:
            self.paths, crops = load_run(self.settings, self.run)
            base: pl.DataFrame = crops.select(
                [c for c in ITEM_COLUMNS if c in crops.columns]
            ).rename({"confidence": "teacher_conf"})
            if self.paths.decisions.exists():
                decisions: pl.DataFrame = pl.read_parquet(self.paths.decisions).drop(
                    "brand_before", "suggested_brand", "visible_text", "confidence"
                )
                base = base.join(decisions, on="crop_id", how="left")
            self.base = base
            self.verdicts, self.judged_offset = judgements_frame([]), 0
            self.refresh_judgements(force=True)
        self.targets = sorted(
            load_catalog(
                self.settings.target_list,
                self.settings.aliases,
                fallback=crops["brand"].unique().to_list(),
            ).targets
        )

    def refresh_judgements(self, force: bool = False) -> None:
        """Fold verdicts appended to judgements.jsonl since the last call into `items`.

        Reads only the new bytes (complete lines), so it is cheap enough to run on every
        request even while a full-gallery review is appending.
        """
        with self.lock:
            path: Path = self.paths.judgements
            size: int = path.stat().st_size if path.exists() else 0
            if size < self.judged_offset:  # file replaced or truncated: start over
                self.verdicts, self.judged_offset, force = judgements_frame([]), 0, True
            if size == self.judged_offset and not force:
                return
            rows, consumed = read_new_lines(path, self.judged_offset)
            self.judged_offset += consumed
            if rows or force:
                new: pl.DataFrame = judgements_frame(rows).unique(
                    "crop_id", keep="last", maintain_order=True
                )
                kept: pl.DataFrame = self.verdicts.join(new, on="crop_id", how="anti")
                self.verdicts = pl.concat([kept, new])
                self.items = self.base.join(self.verdicts, on="crop_id", how="left")


def read_new_lines(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    """JSONL rows after byte `offset`, up to the last complete line; and the bytes consumed.

    A line still being written (no newline yet) is left for the next call. Unparseable
    complete lines (a writer killed mid-line earlier) are skipped.
    """
    if not path.exists():
        return [], 0
    with path.open("rb") as f:
        f.seek(offset)
        data: bytes = f.read()
    end: int = data.rfind(b"\n") + 1
    rows: list[dict[str, Any]] = []
    for line in data[:end].decode().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("judgement_line_skipped", path=str(path))
    return rows, end


def judgements_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """judgements.jsonl rows (append order) as `crop_id` + `judge_*` columns."""
    schema: dict[str, type[pl.DataType]] = {"crop_id": pl.String} | {
        dst: pl.String for dst in JUDGE_FIELDS.values()
    }
    return pl.DataFrame(
        [
            {"crop_id": row["crop_id"]}
            | {dst: row.get(src) or None for src, dst in JUDGE_FIELDS.items()}
            for row in rows
        ],
        schema=schema,
    )


@dataclass(frozen=True)
class ReviewActivity:
    """Crops the running `review` has not finished: queued or being judged right now."""

    in_review: frozenset[str] = frozenset()
    judging: frozenset[str] = frozenset()

    def status(self, crop_id: str) -> str | None:
        """'judging', 'queued' or None (not part of a running review)."""
        if crop_id in self.judging:
            return "judging"
        return "queued" if crop_id in self.in_review else None


NO_ACTIVITY = ReviewActivity()


def review_activity(state: "RunState") -> ReviewActivity:
    """What the running review still has to do; empty when no review is running."""
    paths: RunPaths = state.paths
    if not paths.progress.exists() or not paths.review_queue.exists():
        return NO_ACTIVITY
    progress: dict[str, Any] = json.loads(paths.progress.read_text())
    if progress.get("status") != "running":
        return NO_ACTIVITY
    queue: dict[str, Any] = cached(state, paths.review_queue, lambda p: json.loads(p.read_text()))
    if queue.get("started") != progress["started"]:
        return NO_ACTIVITY  # queue of an older review
    offset: int = queue.get("units_offset", 0)
    finished: set[str] = {
        crop_id
        for row in cached(state, paths.units, lambda p: read_from(p, offset), tag=str(offset))
        if row["judge"] == progress["judge"]
        for crop_id in row.get("crop_ids", [])
    }
    judging: frozenset[str] = frozenset(c for ids in progress["judging"].values() for c in ids)
    return ReviewActivity(
        in_review=frozenset(set(queue["crop_ids"]) - finished) | judging, judging=judging
    )


def read_from(path: Path, offset: int) -> list[dict[str, Any]]:
    """JSONL rows written after byte `offset` (this review's rows)."""
    with path.open("rb") as f:
        f.seek(offset)
        return [json.loads(line) for line in f.read().decode().splitlines() if line.strip()]


def cached(state: "RunState", path: Path, load: Callable[[Path], Any], tag: str = "") -> Any:
    """`load(path)`, reused until the file changes (`tag` separates different loads of it)."""
    stat = path.stat() if path.exists() else None
    name: str = f"{path}#{tag}"
    key: tuple[str, int, int] = (name, stat.st_mtime_ns if stat else 0, stat.st_size if stat else 0)
    if key not in state.file_cache:
        state.file_cache = {k: v for k, v in state.file_cache.items() if k[0] != name}
        state.file_cache[key] = load(path) if stat else []
    return state.file_cache[key]


def filter_items(
    items: pl.DataFrame,
    human: dict[str, dict[str, Any]],
    query: dict[str, str],
    activity: ReviewActivity = NO_ACTIVITY,
) -> pl.DataFrame:
    """Apply the UI's filters (all optional)."""
    out: pl.DataFrame = items
    resolved: bool = "decision" in out.columns
    if brand := query.get("brand"):
        in_brand: pl.Expr = pl.col("brand") == brand
        out = out.filter(in_brand | (pl.col("brand_after") == brand) if resolved else in_brand)
    if (decision := query.get("decision")) and resolved:
        out = out.filter(pl.col("decision") == decision)
    if query.get("changed") == "1" and resolved:
        out = out.filter(~pl.col("decision").is_in(["keep", "pending"]))
    if verdict := query.get("verdict"):
        out = out.filter(pl.col("judge_verdict") == verdict)
    if flag := query.get("flag"):
        out = out.filter(pl.col("flags").list.contains(flag))
    if stratum := query.get("stratum"):
        out = out.filter(pl.col("stratum") == stratum)
    labelled: list[str] = list(human)
    if query.get("labelled") == "1":
        out = out.filter(pl.col("crop_id").is_in(labelled))
    if query.get("labelled") == "0":
        out = out.filter(~pl.col("crop_id").is_in(labelled))
    if review := query.get("review"):
        out = out.filter(review_filter(review, activity))
    if query.get("disagree") == "1":
        human_verdict: pl.Expr = pl.col("crop_id").replace_strict(
            {c: h["verdict"] for c, h in human.items()}, default=None, return_dtype=pl.String
        )
        out = out.filter(human_verdict.is_not_null() & (human_verdict != pl.col("judge_verdict")))
    return out


def filter_unit(state: RunState, items: pl.DataFrame, unit: str) -> pl.DataFrame:
    """Crops of one review unit (montage), in tile order; or units whose id contains `unit`.

    An exact unit id is looked up in units.jsonl, so it also finds units whose crops were
    re-judged since; a partial id (e.g. `devon-energy_c-1`) matches the crops' latest unit.
    """
    unit = unit.strip()
    if not unit:
        return items
    members: list[str] | None = unit_members(state, unit)
    if members is None:
        # Whole segments first: `devon-energy_c-1` is cluster -1, not also cluster -100.
        segment: pl.DataFrame = items.filter(pl.col("judge_unit_id").str.starts_with(f"{unit}_"))
        if not segment.is_empty():
            return segment
        return items.filter(pl.col("judge_unit_id").str.contains(unit, literal=True))
    order: dict[str, int] = {crop_id: i for i, crop_id in enumerate(members)}
    return (
        items.filter(pl.col("crop_id").is_in(members))
        .with_columns(_tile=pl.col("crop_id").replace_strict(order, return_dtype=pl.Int64))
        .sort("_tile")
        .drop("_tile")
    )


def unit_members(state: RunState, unit_id: str) -> list[str] | None:
    """Crop ids of a unit in tile order (latest record of it), or None if no such unit."""
    rows: list[dict[str, Any]] = cached(state, state.paths.units, jsonl.read)
    for row in reversed(rows):
        if row.get("unit_id") == unit_id and row.get("crop_ids"):
            return list(row["crop_ids"])
    return None


def review_filter(review: str, activity: ReviewActivity) -> pl.Expr:
    """pending = not judged yet or still in the running review; the rest as named."""
    not_judged: pl.Expr = pl.col("judge_verdict").is_null()
    in_review: pl.Expr = pl.col("crop_id").is_in(list(activity.in_review))
    match review:
        case "pending":
            return not_judged | in_review
        case "not_judged":
            return not_judged & ~in_review
        case "in_review":
            return in_review
        case "judging":
            return pl.col("crop_id").is_in(list(activity.judging))
        case "judged":
            return ~not_judged & ~in_review
    raise ValueError(f"unknown review filter {review!r}")


@dataclass
class RunRegistry:
    """Runs of one gallery, loaded on first use; each request names its run (`?run=`).

    Every browser tab can show a different run; requests without a run get `default`.
    """

    settings: GallerySettings
    default: str
    states: dict[str, RunState] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get(self, run: str | None) -> RunState:
        """The run's state, loading it the first time; ValueError for unknown runs."""
        name: str = run or self.default
        if not SAFE_NAME.match(name) or not self.settings.run_paths(name).run_json.exists():
            raise ValueError(f"unknown run {name!r}")
        with self.lock:
            if name not in self.states:
                self.states[name] = RunState(settings=self.settings, run=name)
            state: RunState = self.states[name]
        state.refresh_judgements()  # pick up verdicts a running review appended meanwhile
        return state


def api_runs(registry: RunRegistry) -> list[dict[str, Any]]:
    """Every run of the gallery, newest first, with size and review status."""
    rows: list[dict[str, Any]] = []
    for run_json in registry.settings.audit_root.glob("*/run.json"):
        info: dict[str, Any] = json.loads(run_json.read_text())
        progress: Path = run_json.parent / "review_progress.json"
        status: str | None = (
            json.loads(progress.read_text()).get("status") if progress.exists() else None
        )
        rows.append(
            {
                "run": run_json.parent.name,
                "created_at": info.get("created_at"),
                "n_crops": info.get("n_crops"),
                "n_brands": info.get("n_brands"),
                "review_status": status,
            }
        )
    return sorted(rows, key=lambda r: r["created_at"] or "", reverse=True)


def make_handler(registry: RunRegistry) -> type[BaseHTTPRequestHandler]:
    """Request handler; each request picks its run with `?run=` (default: the served run)."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib name
            """Route access logs to debug."""
            log.debug("http", line=format % args)

        def do_GET(self) -> None:  # noqa: N802 - stdlib name
            """Static page, images and read-only API."""
            url = urlparse(self.path)
            query: dict[str, str] = {k: v[-1] for k, v in parse_qs(url.query).items()}
            if url.path in ("/", "/index.html"):
                return self.send_bytes(page(), "text/html; charset=utf-8")
            if url.path == "/api/runs":
                return self.send_json(api_runs(registry))
            try:
                state: RunState = registry.get(query.pop("run", None))
            except ValueError as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            routes: dict[str, Any] = {
                "/api/meta": lambda: api_meta(state),
                "/api/items": lambda: api_items(state, query),
                "/api/history": lambda: api_history(state, query.get("crop_id", "")),
                "/api/changes": lambda: api_changes(state),
                "/api/brand_changes": lambda: api_brand_changes(state),
                "/api/spend": lambda: api_spend(state),
                "/api/progress": lambda: api_progress(state),
                "/api/resolve_status": lambda: api_resolve_status(state),
            }
            if url.path in routes:
                return self.send_json(routes[url.path]())
            for prefix, root in (
                ("/crops/", state.settings.crops_dir),
                ("/montages/", state.paths.montages),
            ):
                if url.path.startswith(prefix):
                    return self.send_file(root, url.path.removeprefix(prefix))
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - stdlib name
            """Write endpoints: add a label, re-run resolve."""
            length: int = int(self.headers.get("Content-Length", 0))
            body: dict[str, Any] = json.loads(self.rfile.read(length) or b"{}")
            url = urlparse(self.path)
            path: str = url.path
            run: str | None = parse_qs(url.query).get("run", [None])[-1]
            try:
                state: RunState = registry.get(run)
                if path == "/api/label":
                    return self.send_json(api_label(state, body))
                if path == "/api/accept_relabels":
                    return self.send_json(api_accept_relabels(state, body, start_resolve))
                if path == "/api/rejudge":
                    return self.send_json(api_rejudge(state, body))
                if path == "/api/resolve":
                    return self.send_json(start_resolve(state, body))
            except ValueError as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            self.send_error(HTTPStatus.NOT_FOUND)

        def send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            """JSON response."""
            self.send_bytes(json.dumps(data, default=str).encode(), "application/json", status)

        def send_file(self, root: Path, name: str) -> None:
            """A JPEG from `root`; only plain file names, so no path traversal."""
            path: Path = root / name
            if not SAFE_NAME.match(name) or not path.is_file():
                return self.send_error(HTTPStatus.NOT_FOUND)
            self.send_bytes(path.read_bytes(), "image/jpeg", cache=True)

        def send_bytes(
            self,
            data: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
            cache: bool = False,
        ) -> None:
            """Raw response."""
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def page() -> bytes:
    """The single-page UI."""
    return resources.files("cleanup.gallery").joinpath("static/index.html").read_bytes()


def human_labels(state: RunState) -> dict[str, dict[str, Any]]:
    """Latest non-cleared human label per crop."""
    latest: dict[str, dict[str, Any]] = jsonl.latest_by(jsonl.read(state.paths.events), "crop_id")
    return {k: v for k, v in latest.items() if v.get("verdict")}


def api_meta(state: RunState) -> dict[str, Any]:
    """Run overview for the header and filter dropdowns."""
    items: pl.DataFrame = state.items
    summary: dict[str, Any] = (
        json.loads(state.paths.summary.read_text()) if state.paths.summary.exists() else {}
    )
    run_meta: dict[str, Any] = json.loads(state.paths.run_json.read_text())
    return {
        "run": state.run,
        "subset": run_meta.get("subset"),
        "n_crops": items.height,
        "n_labelled": len(human_labels(state)),
        "n_judged": int(items["judge_verdict"].is_not_null().sum()),
        "has_decisions": "decision" in items.columns,
        "summary": summary,
        "brands": sorted(items["brand"].unique().to_list()),
        "strata": sorted(s for s in items["stratum"].unique().to_list() if s),
        "flags": sorted({f for fs in items["flags"].to_list() for f in fs}),
        "verdicts": list(TILE_LABELS),
        "hard_flags": list(state.settings.hard_flags),
        "targets": state.targets,
        "refresh_seconds": state.settings.ui_refresh_seconds,
        "judges": judge_choices(state),
        "default_judge": state.settings.judge,
    }


def api_items(state: RunState, query: dict[str, str]) -> dict[str, Any]:
    """One page of filtered crops, each with its latest human label."""
    human: dict[str, dict[str, Any]] = human_labels(state)
    activity: ReviewActivity = review_activity(state)
    filtered: pl.DataFrame = filter_unit(
        state, filter_items(state.items, human, query, activity), query.get("unit", "")
    )
    offset: int = max(0, int(query.get("offset", 0)))
    limit: int = min(MAX_PAGE_SIZE, int(query.get("limit", DEFAULT_PAGE_SIZE)))
    page_rows: list[dict[str, Any]] = filtered.slice(offset, limit).to_dicts()
    return {
        "total": filtered.height,
        "offset": offset,
        "items": [
            row
            | {"human": human.get(row["crop_id"]), "review_status": activity.status(row["crop_id"])}
            for row in page_rows
        ],
    }


def api_history(state: RunState, crop_id: str) -> dict[str, Any]:
    """Every judge verdict and human event for one crop, oldest first."""
    return {
        "judgements": [r for r in jsonl.read(state.paths.judgements) if r["crop_id"] == crop_id],
        "events": [r for r in jsonl.read(state.paths.events) if r["crop_id"] == crop_id],
    }


def api_changes(state: RunState) -> list[dict[str, Any]]:
    """All human events, newest first."""
    return list(reversed(jsonl.read(state.paths.events)))


def api_brand_changes(state: RunState) -> list[dict[str, Any]]:
    """Per-brand before/after counts from the last resolve."""
    if not state.paths.brand_changes.exists():
        return []
    return pl.read_csv(state.paths.brand_changes).to_dicts()


def api_progress(state: RunState) -> dict[str, Any]:
    """Live progress of the current or last `review` of this run ({"status": "none"} if never)."""
    if not state.paths.progress.exists():
        return {"status": "none"}
    return json.loads(state.paths.progress.read_text())


def api_spend(state: RunState) -> dict[str, Any]:
    """Estimated judge spend: this run, all runs of the gallery, recent review calls, prices."""
    return spend_report(state.settings.audit_root, state.run)


def api_label(state: RunState, body: dict[str, Any]) -> dict[str, Any]:
    """Append a human label (or clear one with verdict null); records the previous label."""
    crop_id: str = str(body.get("crop_id", ""))
    verdict: str | None = body.get("verdict")
    if state.items.filter(pl.col("crop_id") == crop_id).is_empty():
        raise ValueError(f"unknown crop_id {crop_id!r}")
    if verdict is not None and verdict not in TILE_LABELS:
        raise ValueError(f"verdict must be one of {TILE_LABELS} or null")
    suggested: str = str(body.get("suggested_brand") or "").strip()
    if verdict == "different_brand" and not suggested:
        raise ValueError("different_brand needs suggested_brand")
    judge_row: dict[str, Any] = state.items.filter(pl.col("crop_id") == crop_id).row(0, named=True)
    previous: dict[str, Any] | None = human_labels(state).get(crop_id)
    event: dict[str, Any] = {
        "crop_id": crop_id,
        "ts": jsonl.now(),
        "verdict": verdict,
        "suggested_brand": suggested if verdict == "different_brand" else "",
        "note": str(body.get("note") or ""),
        "brand": judge_row["brand"],
        "judge_verdict": judge_row["judge_verdict"],
        "previous_verdict": previous["verdict"] if previous else None,
        "previous_suggested_brand": previous.get("suggested_brand") if previous else None,
    }
    jsonl.append(state.paths.events, [event])
    return event


def api_accept_relabels(
    state: RunState,
    body: dict[str, Any],
    then_resolve: Callable[[RunState, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Confirm every `suggested_relabel` matching the UI's filters, then re-resolve.

    `then_resolve` runs the resolve (default: `api_resolve`, inline; the server starts it in
    the background instead) and its result is returned as `summary`. Each acceptance is an
    ordinary human label (different_brand + the judge's target, note `accepted (bulk)`)
    appended to events.jsonl, so any crop can be changed back individually.
    """
    if "decision" not in state.items.columns:
        raise ValueError("run Resolve first")
    if state.resolve_job.snapshot()["status"] == "running":
        raise ValueError("a resolve is running; accept relabels when it is done")
    filters: dict[str, str] = {k: str(v) for k, v in (body.get("filters") or {}).items()}
    candidates: pl.DataFrame = filter_unit(
        state, filter_items(state.items, human_labels(state), filters), filters.get("unit", "")
    ).filter(pl.col("decision") == "suggested_relabel")
    for row in candidates.select("crop_id", "relabel_target").iter_rows(named=True):
        api_label(
            state,
            {
                "crop_id": row["crop_id"],
                "verdict": "different_brand",
                "suggested_brand": row["relabel_target"],
                "note": "accepted (bulk)",
            },
        )
    summary: dict[str, Any] = (then_resolve or api_resolve)(
        state, body | {"accepted": candidates.height}
    )
    return {"accepted": candidates.height, "summary": summary}


def judge_choices(state: RunState) -> list[str]:
    """Judges offered for a second opinion: configured ones, then any already used in the run."""
    used: list[str] = sorted({r["judge"] for r in jsonl.read(state.paths.judgements)})
    return list(dict.fromkeys([*state.settings.ui_judges, state.settings.judge, *used]))


def api_rejudge(state: RunState, body: dict[str, Any]) -> dict[str, Any]:
    """Ask `judge` about one crop now (second opinion); reload so the card shows the verdict."""
    crop_id: str = str(body.get("crop_id", ""))
    judge: str = str(body.get("judge") or "").strip()
    if not judge:
        raise ValueError("pick a judge")
    try:
        verdict: dict[str, Any] = judge_one(state.settings, state.run, crop_id, judge)
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - report setup errors (missing package, auth) to the UI
        raise ValueError(f"{judge}: {exc!r}") from exc
    state.reload()
    item: dict[str, Any] = state.items.filter(pl.col("crop_id") == crop_id).row(0, named=True)
    return {"verdict": verdict, "item": item | {"human": human_labels(state).get(crop_id)}}


def api_resolve(
    state: RunState, body: dict[str, Any], progress: Progress = no_progress
) -> dict[str, Any]:
    """Re-run resolve (human labels included) and reload the tables."""
    resolve_run(
        state.settings,
        state.run,
        judge=body.get("judge") or None,
        min_confidence=body.get("min_confidence") or "low",
        trust_judge_relabels=bool(body.get("trust_judge_relabels")),
        progress=lambda stage, fraction: progress(stage, RESOLVE_SHARE * fraction),
    )
    progress("reloading UI", RESOLVE_SHARE)
    state.reload()
    progress("reloading UI", 1.0)
    return json.loads(state.paths.summary.read_text())


def start_resolve(state: RunState, body: dict[str, Any]) -> dict[str, Any]:
    """Start `api_resolve` in a background thread; poll `api_resolve_status` for progress.

    One resolve per run at a time: ValueError while one is running.
    """
    job: ResolveJob = state.resolve_job
    with job.lock:
        if job.status == "running":
            raise ValueError("a resolve is already running for this run")
        job.status, job.stage, job.fraction = "running", "starting", 0.0
        job.started, job.finished, job.error, job.summary = jsonl.now(), None, None, None
        job.accepted = body.get("accepted")
    threading.Thread(target=run_resolve_job, args=(state, body), daemon=True).start()
    return job.snapshot()


def run_resolve_job(state: RunState, body: dict[str, Any]) -> None:
    """Thread body of `start_resolve`: run it and record the outcome on the job."""
    job: ResolveJob = state.resolve_job
    try:
        summary: dict[str, Any] = api_resolve(state, body, progress=job.update)
    except Exception as exc:  # noqa: BLE001 - shown in the UI instead of killing the thread
        log.exception("resolve_failed", run=state.run)
        with job.lock:
            job.status, job.error, job.finished = "error", repr(exc), jsonl.now()
        return
    with job.lock:
        job.status, job.summary, job.finished = "done", summary, jsonl.now()


def api_resolve_status(state: RunState) -> dict[str, Any]:
    """Progress of the run's current or last UI resolve."""
    return state.resolve_job.snapshot()


def serve(settings: GallerySettings, run: str, port: int) -> None:
    """Serve the UI (opening on `run`; any run of the gallery can be picked) until interrupted."""
    registry = RunRegistry(settings=settings, default=run)
    state: RunState = registry.get(run)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(registry))
    log.info("ui", url=f"http://127.0.0.1:{port}", run=run, crops=state.items.height)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
