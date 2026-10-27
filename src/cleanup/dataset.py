import os
from collections.abc import Iterator
from pathlib import Path

IMAGES_DIR = "images"
# (directory, suffix) pairs tried in order when looking for an image's annotation.
LABEL_DIRS: tuple[tuple[str, str], ...] = (("labels", ".txt"), ("annotations", ".json"))


def walk_files(root: Path) -> Iterator[Path]:
    """Yield every file under root in sorted order, following symlinks (cycle-safe).

    Yielded paths keep the symlinked (unresolved) form so they stay relative to `root`.
    """
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(top=root, followlinks=True):
        real: str = os.path.realpath(filename=dirpath)
        if real in seen:
            dirnames.clear()
            continue
        seen.add(real)
        dirnames.sort()
        yield from (Path(dirpath) / name for name in sorted(filenames))


def is_image(path: Path, image_exts: tuple[str, ...]) -> bool:
    """True for non-hidden files with an image extension."""
    return not path.name.startswith(".") and path.suffix.lower() in image_exts


def find_label(image_path: Path, data_dir: Path) -> Path | None:
    """Return the annotation file paired with an image, or None if there is none."""
    parts: tuple[str, ...] = image_path.relative_to(data_dir).parts
    if IMAGES_DIR not in parts[:-1]:
        return None
    idx: int = len(parts) - 1 - parts[::-1].index(IMAGES_DIR)
    root: Path = data_dir.joinpath(*parts[:idx])
    sub: Path = Path(*parts[idx + 1 :])
    candidates: Iterator[Path] = (
        root / dir_name / sub.with_suffix(suffix) for dir_name, suffix in LABEL_DIRS
    )
    return next((c for c in candidates if c.is_file()), None)


def metadata_files(data_dir: Path, image_exts: tuple[str, ...]) -> Iterator[Path]:
    """Files that are neither images nor annotations, e.g. `classes/*.txt` or `data.yaml`."""
    label_dirs: set[str] = {dir_name for dir_name, _ in LABEL_DIRS}
    for path in walk_files(data_dir):
        rel_parts: tuple[str, ...] = path.relative_to(data_dir).parts
        if path.name.startswith(".") or path.suffix.lower() in image_exts:
            continue
        if label_dirs.intersection(rel_parts[:-1]) or IMAGES_DIR in rel_parts[:-1]:
            continue
        yield path
