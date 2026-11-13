"""Append-only JSON-lines logs: judge results and human events are never rewritten, so every
change stays traceable."""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_lock = threading.Lock()


def now() -> str:
    """UTC timestamp, ISO 8601."""
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def append(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows to a JSONL file (thread-safe within the process)."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    text: str = "".join(json.dumps(row, default=str) + "\n" for row in rows)
    with _lock, path.open("a") as f:
        f.write(text)


def read(path: Path) -> list[dict[str, Any]]:
    """All rows of a JSONL file; empty if it does not exist."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Last row per key value (rows are in append order)."""
    return {row[key]: row for row in rows}
