"""Local review UI: label crops, compare before/after, and see every change.

Serves one static page plus a small JSON API on 127.0.0.1. Human labels are appended to the
run's events.jsonl (never rewritten), so the full history of every crop stays visible.
"""

import json
import re
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
from cleanup.gallery.resolve import resolve_run
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


@dataclass
class RunState:
    """Everything the UI shows for one run; reloaded after a resolve."""

    settings: GallerySettings
    run: str
    paths: RunPaths = field(init=False)
    items: pl.DataFrame = field(init=False)
    targets: list[str] = field(init=False)
    # (file, mtime_ns, size) -> parsed content, so polling large runs stays cheap.
    file_cache: dict[tuple[str, int, int], Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Load the run's tables."""
        self.reload()

    def reload(self) -> None:
        """Join crops, flags, latest judge verdict and the last resolve's decisions."""
        self.paths, crops = load_run(self.settings, self.run)
        items: pl.DataFrame = crops.select([c for c in ITEM_COLUMNS if c in crops.columns]).rename(
            {"confidence": "teacher_conf"}
        )
        items = items.join(latest_judgements(self.paths), on="crop_id", how="left")
        if self.paths.decisions.exists():
            decisions: pl.DataFrame = pl.read_parquet(self.paths.decisions).drop(
                "brand_before", "suggested_brand", "visible_text", "confidence"
            )
            items = items.join(decisions, on="crop_id", how="left")
        self.items = items
        self.targets = sorted(
            load_catalog(
                self.settings.target_list,
                self.settings.aliases,
                fallback=crops["brand"].unique().to_list(),
            ).targets
        )


def latest_judgements(paths: RunPaths) -> pl.DataFrame:
    """Latest judge verdict per crop, as `judge_*` columns."""
    latest: dict[str, dict[str, Any]] = jsonl.latest_by(jsonl.read(paths.judgements), "crop_id")
    rows: list[dict[str, Any]] = [
        {"crop_id": crop_id} | {dst: row.get(src) or None for src, dst in JUDGE_FIELDS.items()}
        for crop_id, row in latest.items()
    ]
    schema: dict[str, type[pl.DataType]] = {"crop_id": pl.String} | {
        dst: pl.String for dst in JUDGE_FIELDS.values()
    }
    return pl.DataFrame(rows, schema=schema)


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


def make_handler(state: RunState) -> type[BaseHTTPRequestHandler]:
    """Request handler bound to one run."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib name
            """Route access logs to debug."""
            log.debug("http", line=format % args)

        def do_GET(self) -> None:  # noqa: N802 - stdlib name
            """Static page, images and read-only API."""
            url = urlparse(self.path)
            query: dict[str, str] = {k: v[-1] for k, v in parse_qs(url.query).items()}
            routes: dict[str, Any] = {
                "/api/meta": lambda: api_meta(state),
                "/api/items": lambda: api_items(state, query),
                "/api/history": lambda: api_history(state, query.get("crop_id", "")),
                "/api/changes": lambda: api_changes(state),
                "/api/brand_changes": lambda: api_brand_changes(state),
                "/api/spend": lambda: api_spend(state),
                "/api/progress": lambda: api_progress(state),
            }
            if url.path in routes:
                return self.send_json(routes[url.path]())
            if url.path in ("/", "/index.html"):
                return self.send_bytes(page(), "text/html; charset=utf-8")
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
            path: str = urlparse(self.path).path
            try:
                if path == "/api/label":
                    return self.send_json(api_label(state, body))
                if path == "/api/accept_relabels":
                    return self.send_json(api_accept_relabels(state, body))
                if path == "/api/rejudge":
                    return self.send_json(api_rejudge(state, body))
                if path == "/api/resolve":
                    return self.send_json(api_resolve(state, body))
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
    filtered: pl.DataFrame = filter_items(state.items, human, query, activity)
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


def api_accept_relabels(state: RunState, body: dict[str, Any]) -> dict[str, Any]:
    """Confirm every `suggested_relabel` matching the UI's filters, then re-resolve.

    Each acceptance is an ordinary human label (different_brand + the judge's target, note
    `accepted (bulk)`) appended to events.jsonl, so any crop can be changed back individually.
    """
    if "decision" not in state.items.columns:
        raise ValueError("run Resolve first")
    filters: dict[str, str] = {k: str(v) for k, v in (body.get("filters") or {}).items()}
    candidates: pl.DataFrame = filter_items(state.items, human_labels(state), filters).filter(
        pl.col("decision") == "suggested_relabel"
    )
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
    summary: dict[str, Any] = api_resolve(state, body)
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


def api_resolve(state: RunState, body: dict[str, Any]) -> dict[str, Any]:
    """Re-run resolve (human labels included) and reload the tables."""
    resolve_run(
        state.settings,
        state.run,
        judge=body.get("judge") or None,
        min_confidence=body.get("min_confidence") or "low",
        trust_judge_relabels=bool(body.get("trust_judge_relabels")),
    )
    state.reload()
    return json.loads(state.paths.summary.read_text())


def serve(settings: GallerySettings, run: str, port: int) -> None:
    """Serve the UI for a run until interrupted."""
    state = RunState(settings=settings, run=run)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    log.info("ui", url=f"http://127.0.0.1:{port}", run=run, crops=state.items.height)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
