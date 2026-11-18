from collections.abc import Callable, Iterable, Iterator, Mapping

# (stage, overall fraction done 0..1)
Progress = Callable[[str, float], None]


def no_progress(stage: str, fraction: float) -> None:
    """Default callback: report nothing."""


def report(
    progress: Progress, stages: Mapping[str, tuple[float, float]], stage: str, fraction: float
) -> None:
    """Call `progress` with `fraction` of `stage` mapped onto its (start, end) share."""
    start, end = stages[stage]
    progress(stage, start + (end - start) * fraction)


def scaled(progress: Progress, start: float, end: float) -> Progress:
    """A callback mapping 0..1 onto start..end of `progress` (a step inside a bigger job)."""
    return lambda stage, fraction: progress(stage, start + (end - start) * fraction)


def tracked[T](rows: Iterable[T], total: int, on_step: Callable[[float], None]) -> Iterator[T]:
    """Yield `rows`, calling `on_step(fraction done)` every 1% of `total`."""
    step: int = max(1, total // 100)
    for i, row in enumerate(rows):
        if i % step == 0:
            on_step(i / total)
        yield row
