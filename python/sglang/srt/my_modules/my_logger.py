"""
Centralized file-logger for the my_modules prefetch system.

Usage
-----
Call :func:`setup` once at startup (inside ``init_my_prefetch_controller``).
After that, every ``logging.getLogger(__name__)`` call inside the
``sglang.srt.my_modules`` namespace will automatically write to the configured
file **in addition to** sglang's normal console output.

Log format
----------
Each line looks like::

    2026-03-09 14:23:01 | INFO     | prefetch_controller | [Predict] session=abc | ...
    2026-03-09 14:23:01 | DEBUG    | hidden_hook         | [Hook] captured [512, 2048]
    2026-03-09 14:23:01 | INFO     | prefetch_controller | [Prefetch] 'Classify' p=0.65 → TRIGGERED

Level guide
-----------
- DEBUG  — low-level events: hook fires, skip reasons, node state checks
- INFO   — business events: prediction results, prefetch triggers, tool registration
- WARNING — unexpected states: load_back failure, tokenization error
"""

from __future__ import annotations

import logging
import os
from typing import Optional

# The namespace that covers all my_modules sub-loggers.
_NAMESPACE = "sglang.srt.my_modules"

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Singleton reference so we don't add duplicate handlers on repeated calls.
_file_handler: Optional[logging.FileHandler] = None


def setup(log_file: str) -> None:
    """
    Configure a ``FileHandler`` on the ``sglang.srt.my_modules`` logger.

    Parameters
    ----------
    log_file:
        Absolute or relative path to the target log file.  The parent
        directory is created automatically if it does not exist.
        If *log_file* is empty, the function is a no-op.
    """
    global _file_handler

    if not log_file:
        return

    if _file_handler is not None:
        # Already configured — avoid duplicate handlers.
        return

    # Ensure parent directory exists.
    parent = os.path.dirname(os.path.abspath(log_file))
    os.makedirs(parent, exist_ok=True)

    # Create handler.
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    # Attach to the namespace root logger.
    ns_logger = logging.getLogger(_NAMESPACE)
    ns_logger.setLevel(logging.DEBUG)  # allow DEBUG to reach the handler
    ns_logger.addHandler(fh)

    _file_handler = fh

    ns_logger.info(
        "[MyLogger] File logging enabled → %s", os.path.abspath(log_file)
    )


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper — returns ``logging.getLogger(name)``."""
    return logging.getLogger(name)
