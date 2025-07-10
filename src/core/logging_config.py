import os
from pathlib import Path
import sys

from loguru import logger

from src.core.config import settings


ENV = settings.ENV


def configure_logging() -> None:
    """Centralized logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Remove default handler
    logger.remove()

    # Main application log
    logger.add(
        log_dir / "app.log",
        rotation="10 MB",
        retention="30 days",
        level=os.getenv("LOG_LEVEL", "INFO"),
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} |"
        " {level: <8} |"
        " {name}:{function}:{line} - {message}",
        backtrace=True,
        diagnose=True,
        compression="zip",
    )

    # Development console output
    if ENV == "development":
        logger.add(
            sys.stderr,
            level="DEBUG",
            colorize=True,
            format="<green>{time:HH:mm:ss}</green> |"
            " <level>{level: <8}</level> |"
            " <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
            " - <level>{message}</level>",
        )
