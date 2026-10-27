"""Console output: structlog setup and tqdm progress bars that don't tear each other up."""

import logging
import sys
from collections.abc import Iterable
from typing import Any

import structlog
from tqdm import tqdm


class TqdmLogger:
    """structlog logger that prints through `tqdm.write`, keeping progress bars intact."""

    def msg(self, message: str) -> None:
        """Write one rendered log line to stderr."""
        tqdm.write(message, file=sys.stderr)

    log = debug = info = warning = warn = error = critical = exception = msg


def tqdm_logger_factory(*_args: Any) -> TqdmLogger:
    """structlog logger factory returning a TqdmLogger."""
    return TqdmLogger()


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog for human-readable console output at the given level."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=tqdm_logger_factory,
        cache_logger_on_first_use=True,
    )


def progress(
    iterable: Iterable[Any] | None = None,
    *,
    desc: str,
    total: int | None = None,
    unit: str = "img",
) -> tqdm:
    """Progress bar on stderr. Auto-disabled when stderr is not a terminal."""
    return tqdm(iterable, desc=desc, total=total, unit=unit, dynamic_ncols=True, disable=None)
