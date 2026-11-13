"""Estimated judge spend from the token usage in review_log.jsonl.

Prices are USD per 1M tokens, standard (on-demand) tier, prompts under 100K tokens. They are a
snapshot (PRICES_AS_OF) and drift; check the sources before relying on totals. A judge missing
from PRICES is reported as unpriced, never guessed.
"""

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
        priced(entry, log.parent.name)
        for log in sorted(audit_root.glob("*/review_log.jsonl"))
        for entry in jsonl.read(log)
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


def priced(entry: dict[str, Any], run: str) -> dict[str, Any]:
    """One review_log entry flattened, with its estimated cost (None when unpriced)."""
    usage: dict[str, Any] = entry.get("usage") or {}
    requests: int = usage.get("requests", 0)
    # No successful request (e.g. every unit failed auth or 404): nothing was billed.
    usd: float | None = estimate_usd(entry["judge"], usage) if requests else 0.0
    return {
        "run": run,
        "ts": entry["ts"],
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
