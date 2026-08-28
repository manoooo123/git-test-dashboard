"""
Enterprise Structured Logging Module for Pearls AQI Predictor.
"""

import functools
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
from typing import Callable, Any

# Log directory configuration
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"


def setup_logger(name: str = "pearls_aqi", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a structured logger with console and rotating file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Log format specifications
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(name)s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating File Handler (5 MB per file, max 3 backups)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Default application logger instance
logger = setup_logger()


def log_execution_time(func: Callable) -> Callable:
    """Decorator to measure and log function execution time."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        func_name = func.__qualname__
        logger.debug("Executing function: %s", func_name)
        try:
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info("Completed %s in %.2f ms", func_name, elapsed)
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error("Failed %s after %.2f ms with error: %s", func_name, elapsed, e, exc_info=True)
            raise
    return wrapper
