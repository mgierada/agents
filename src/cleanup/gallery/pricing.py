"""Estimated judge spend from the token usage in review_log.jsonl.

Prices are USD per 1M tokens, standard (on-demand) tier, prompts under 100K tokens. They are a
snapshot (PRICES_AS_OF) and drift; check the sources before relying on totals. A judge missing
from PRICES is reported as unpriced, never guessed.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cleanup.gallery import jsonl

PRICES_AS_OF = "2026-11-13"
PER_TOKEN = 1_000_000
# Providers whose `input_tokens` excludes cache reads/writes (Anthropic usage). LangChain-based
# providers report `input_tokens` as the total, so cached tokens are subtracted for those.
ANTHROPIC_USAGE_PROVIDERS: frozenset[str] = frozenset({"claude", "bedrock"})
FREE_PROVIDERS: frozenset[str] = frozenset({"ollama"})
ANTHROPIC_SOURCE = "Anthropic API price list"
AWS_SOURCE = "AWS Price List API, us-east-1"


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens. Cache write is the 5-minute TTL rate."""

    input: float
    output: float
    cache_read: float
    cache_write: float
    source: str


PRICES: dict[str, Price] = {
    "claude:claude-haiku-5-5": Price(0.10, 0.50, 0.01, 0.125, ANTHROPIC_SOURCE),
    "claude:claude-sonnet-5-5": Price(2.00, 10.00, 0.10, 2.50, ANTHROPIC_SOURCE),
    "claude:claude-opus-5-5": Price(4.00, 20.00, 0.20, 5.00, ANTHROPIC_SOURCE),
    # Mantle model ids are in-region ("Standard") on Bedrock.
    "bedrock:claude-opus-5-5": Price(4.40, 22.00, 0.22, 5.50, AWS_SOURCE),
    "bedrock-converse:global.anthropic.claude-haiku-5-5": Price(
        0.10, 0.50, 0.01, 0.125, AWS_SOURCE
    ),
    "bedrock-converse:us.anthropic.claude-haiku-5-5": Price(0.11, 0.55, 0.011, 0.1375, AWS_SOURCE),
    "bedrock-converse:global.anthropic.claude-sonnet-5-5": Price(
        2.00, 10.00, 0.10, 2.50, AWS_SOURCE
    ),
    "bedrock-converse:us.anthropic.claude-sonnet-5-5": Price(2.20, 11.00, 0.11, 2.75, AWS_SOURCE),
    "bedrock-converse:qwen.qwen3-vl-235b-a22b": Price(0.53, 2.66, 0.0, 0.0, AWS_SOURCE),
}


def spend_report(audit_root: Path, run: str, recent: int = 10) -> dict[str, Any]:
    """Spend of `run` and of all runs under `audit_root`, the newest review calls, and prices."""
    rows: list[dict[str, Any]] = [
        priced(entry=entry, run=run_dir.name)
        for run_dir in sorted(p for p in audit_root.iterdir() if p.is_dir())
        for entry in [*jsonl.read(path=run_dir / "review_log.jsonl"), *unlogged_review(run_dir)]
    ]
    rows.sort(key=lambda r: r["ts"])
    mine: list[dict[str, Any]] = [r for r in rows if r["run"] == run]
    return {
        "run_usd": total(mine),
        "all_usd": total(rows),
        "unpriced_judges": sorted({r["judge"] for r in rows if r["usd"] is None}),
        "recent": list(reversed(mine[-recent:])),
        "prices": price_table(),
        "prices_as_of": PRICES_AS_OF,
    }


def unlogged_review(run_dir: Path) -> list[dict[str, Any]]:
    """The run's last review as a review_log-like entry when it never wrote one.

    That is a review still running, or one killed with SIGKILL; its review_progress.json
    holds the usage up to the last finished unit. Empty when the review already logged itself.
    """
    progress_path: Path = run_dir / "review_progress.json"
    if not progress_path.exists():
        return []
    progress: dict[str, Any] = json.loads(progress_path.read_text())
    logged: set[str] = {e.get("started", "") for e in jsonl.read(path=run_dir / "review_log.jsonl")}
    if "usage" not in progress or progress["started"] in logged:
        return []
    status: str = "running" if progress["status"] == "running" else "killed"
    return [
        {
            "ts": progress["updated"],
            "started": progress["started"],
            "status": status,
            "judge": progress["judge"],
            "units": progress["total_units"],
            "units_done": progress["done_units"],
            "crops": progress["total_crops"],
            "usage": progress["usage"],
        }
    ]


def priced(entry: dict[str, Any], run: str) -> dict[str, Any]:
    """One review_log entry flattened, with its estimated cost (None when unpriced)."""
    usage: dict[str, Any] = entry.get("usage") or {}
    requests: int = usage.get("requests", 0)
    # No successful request (e.g. every unit failed auth or 404): nothing was billed.
    usd: float | None = estimate_usd(judge=entry["judge"], usage=usage) if requests else 0.0
    return {
        "run": run,
        "ts": entry["ts"],
        "status": entry.get("status", "finished"),  # entries before status was logged finished
        "units_done": entry.get("units_done", entry.get("units", 0)),
        "judge": entry["judge"],
        "units": entry.get("units", 0),
        "crops": entry.get("crops", 0),
        "requests": requests,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_read": usage.get("cache_read_input_tokens", 0),
        "cache_write": usage.get("cache_creation_input_tokens", 0),
        "usd": usd,
        "usd_per_request": usd / requests if usd is not None and requests else None,
    }


def estimate_usd(judge: str, usage: dict[str, Any]) -> float | None:
    """Estimated USD for one usage tally; 0 for local models, None for unknown judges."""
    provider: str = judge.partition(":")[0]
    if provider in FREE_PROVIDERS:
        return 0.0
    price: Price | None = PRICES.get(judge)
    if price is None:
        return None
    read: int = usage.get("cache_read_input_tokens", 0)
    write: int = usage.get("cache_creation_input_tokens", 0)
    uncached: int = usage.get("input_tokens", 0)
    if provider not in ANTHROPIC_USAGE_PROVIDERS:
        uncached = max(0, uncached - read - write)
    return (
        uncached * price.input
        + read * price.cache_read
        + write * price.cache_write
        + usage.get("output_tokens", 0) * price.output
    ) / PER_TOKEN


def total(rows: list[dict[str, Any]]) -> float:
    """Sum of priced rows (unpriced rows are skipped and listed separately)."""
    return sum(r["usd"] for r in rows if r["usd"] is not None)


def price_table() -> list[dict[str, Any]]:
    """PRICES as rows for the UI."""
    return [{"judge": judge} | asdict(price) for judge, price in PRICES.items()]


# Tokens per judge call, fitted on measured Haiku 5.5 calls (1 crop and 16 crops, Nov 2026):
# input = 17,600 + 27 per crop, of which 16,995 come from the prompt cache (Claude only);
# output = 91 + 51 per crop. Used only to estimate reviews that died without a log entry.
EST_INPUT_BASE = 17_600
EST_INPUT_PER_CROP = 27
EST_CACHED = 16_995
EST_OUTPUT_BASE = 91
EST_OUTPUT_PER_CROP = 51
CACHING_PROVIDERS: frozenset[str] = frozenset({"claude", "bedrock", "bedrock-converse"})


def backfill_spend(run_dir: Path) -> list[dict[str, Any]]:
    """Append an estimated review_log entry per judge for units no log entry accounts for.

    Units saved in units.jsonl by reviews that were killed (old code wrote the log only at
    the end) are otherwise missing from spend. Count-based and idempotent: units already
    covered by log entries (estimated ones included) are not counted again, and units of a
    review still running are left for that review to log. Failed units are not charged.
    """
    log_path: Path = run_dir / "review_log.jsonl"
    units: list[dict[str, Any]] = settled_units(run_dir)
    logged: dict[str, int] = {}
    for entry in jsonl.read(path=log_path):
        done: int = entry.get("units_done", entry.get("units", 0))
        logged[entry["judge"]] = logged.get(entry["judge"], 0) + done
    entries: list[dict[str, Any]] = []
    for judge in sorted({u["judge"] for u in units}):
        mine: list[dict[str, Any]] = [u for u in units if u["judge"] == judge]
        missing: int = len(mine) - logged.get(judge, 0)
        if missing <= 0:
            continue
        unlogged: list[dict[str, Any]] = mine[-missing:]
        crops: list[int] = [u.get("n_tiles", 0) for u in unlogged if "error" not in u]
        entries.append(
            {
                "ts": jsonl.now(),
                "status": "killed (estimated)",
                "estimated": True,
                "judge": judge,
                "scope": {"source": "backfill_spend"},
                "units": missing,
                "units_done": missing,
                "crops": sum(u.get("n_tiles", 0) for u in unlogged),
                "usage": estimated_usage(judge, crops),
            }
        )
    jsonl.append(path=log_path, rows=entries)
    return entries


def settled_units(run_dir: Path) -> list[dict[str, Any]]:
    """units.jsonl rows, minus those of a review that is still running."""
    units_path: Path = run_dir / "units.jsonl"
    progress_path: Path = run_dir / "review_progress.json"
    queue_path: Path = run_dir / "review_queue.json"
    running: bool = (
        progress_path.exists() and json.loads(progress_path.read_text())["status"] == "running"
    )
    if not (running and queue_path.exists() and units_path.exists()):
        return jsonl.read(path=units_path)
    offset: int = json.loads(queue_path.read_text()).get("units_offset", 0)
    with units_path.open("rb") as f:
        head: str = f.read(offset).decode()
    return [json.loads(line) for line in head.splitlines() if line.strip()]


def estimated_usage(judge: str, crops_per_call: list[int]) -> dict[str, int]:
    """Token usage of successful calls with these crop counts, in the provider's own format."""
    calls: int = len(crops_per_call)
    crops: int = sum(crops_per_call)
    total_input: int = calls * EST_INPUT_BASE + crops * EST_INPUT_PER_CROP
    provider: str = judge.partition(":")[0]
    cached: int = calls * EST_CACHED if provider in CACHING_PROVIDERS else 0
    # Anthropic usage reports input without cached tokens; LangChain's includes them.
    input_tokens: int = (
        total_input - cached if provider in ANTHROPIC_USAGE_PROVIDERS else total_input
    )
    return {
        "requests": calls,
        "input_tokens": input_tokens,
        "output_tokens": calls * EST_OUTPUT_BASE + crops * EST_OUTPUT_PER_CROP,
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,
    }
