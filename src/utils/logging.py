"""Logging and stage timing shared by every pipeline stage."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

ROOT_LOGGER = "videoseek"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Configure the project logger once per process.

    Repeated calls are safe: previous handlers are closed and replaced.
    """
    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT, _DATE_FORMAT)

    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        logger.addHandler(stream)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if not name:
        return logging.getLogger(ROOT_LOGGER)
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes:02d}m {secs:04.1f}s"


class StageTimer:
    """Context manager that logs how long a stage took."""

    def __init__(self, stage: str, logger: Optional[logging.Logger] = None) -> None:
        self.stage = stage
        self.logger = logger or get_logger("timer")
        self.elapsed_s = 0.0
        self._start = 0.0

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        self.logger.info("start %s", self.stage)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.elapsed_s = time.perf_counter() - self._start
        if exc_type is None:
            self.logger.info("done  %s in %s", self.stage, format_duration(self.elapsed_s))
        else:
            self.logger.error("fail  %s after %s", self.stage, format_duration(self.elapsed_s))
        return False


@contextmanager
def stage(name: str, logger: Optional[logging.Logger] = None) -> Iterator[StageTimer]:
    timer = StageTimer(name, logger)
    with timer:
        yield timer
