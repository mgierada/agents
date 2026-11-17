import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

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
    with _lock, path.open(mode="a+b") as f:
        if f.tell() and not ends_with_newline(f):
            f.write(b"\n")
        f.write(text.encode())


def ends_with_newline(f: Any) -> bool:
    """Whether the open binary file (positioned at its end) ends with a newline."""
    f.seek(-1, 2)
    last: bytes = f.read(1)
    return last == b"\n"


def read(path: Path) -> list[dict[str, Any]]:
    """All rows of a JSONL file; empty if it does not exist.

    Lines that are not valid JSON (half a line left by a process killed mid-write) are skipped
    with a warning instead of making the whole log unreadable.
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    broken: list[int] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            broken.append(number)
    if broken:
        log.warning("jsonl_broken_lines_skipped", path=str(path), lines=broken[:10], n=len(broken))
    return rows


def latest_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Last row per key value (rows are in append order)."""
    return {row[key]: row for row in rows}
