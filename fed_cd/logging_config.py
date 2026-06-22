"""
Loguru logger setup for federated training.

Writes to:
  - results/<project_name>/train.log   (persistent, with rotation)
  - stderr                             (colored, real-time console)

Usage:
    from fed_cd.logging_config import setup_logger
    logger = setup_logger(args.project_name, args.checkpoint_root)
    logger.info(f"Round {r}: loss={loss:.4f}")
"""

from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    project_name: str,
    checkpoint_root: str = "results",
    log_to_file: bool = True,
    console: bool = True,
    level: str = "INFO",
) -> "logger":
    """Configure loguru logger with file + console sinks.

    Args:
        project_name: experiment name, used as subdirectory under checkpoint_root
        checkpoint_root: root directory for results (default: "results")
        log_to_file: whether to write to results/<project_name>/train.log
        console: whether to echo to stderr
        level: log level (DEBUG / INFO / WARNING / ERROR)

    Returns:
        Configured loguru logger instance (singleton).
    """
    # Remove any pre-existing sinks (avoid duplicate output when re-called)
    logger.remove()

    # File sink: structured, with rotation & retention
    if log_to_file:
        log_dir = Path(checkpoint_root) / project_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "train.log"
        logger.add(
            str(log_file),
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7s} | {message}",
            rotation="50 MB",
            retention=10,
            encoding="utf-8",
            enqueue=True,  # thread-safe for federated multi-client scenarios
        )

    # Console sink: colored, concise (no timestamp to reduce noise)
    if console:
        logger.add(
            lambda msg: print(msg, end=""),
            level=level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level:<7s}</level> | {message}",
            colorize=True,
        )

    return logger

