from pathlib import Path

YOLO_BOX_FIELDS = 5


def label_issues(path: Path, min_box_area: float, allow_empty: bool) -> list[str]:
    """Validate a YOLO txt label file and return a list of failure reasons."""
    if not path.exists():
        return [] if allow_empty else ["missing_label"]
    lines: list[str] = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        return [] if allow_empty else ["empty_label"]
    return [issue for i, ln in enumerate(lines, 1) for issue in line_issues(ln, i, min_box_area)]


def line_issues(line: str, line_no: int, min_box_area: float) -> list[str]:
    """Validate one `class cx cy w h` line (normalized coordinates)."""
    parts: list[str] = line.split()
    if len(parts) != YOLO_BOX_FIELDS:
        return [f"malformed_line:{line_no}"]
    try:
        cls = int(parts[0])
        cx, cy, w, h = (float(p) for p in parts[1:])
    except ValueError:
        return [f"malformed_line:{line_no}"]
    if cls < 0:
        return [f"negative_class:{line_no}"]
    if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
        return [f"out_of_bounds:{line_no}"]
    if w * h < min_box_area:
        return [f"tiny_box:{line_no}"]
    return []
