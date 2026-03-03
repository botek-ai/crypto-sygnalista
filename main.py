"""crypto-sygnalista — main entry point."""

from __future__ import annotations

import asyncio
import sys

from loguru import logger

from config.settings import get_settings
from db.models import init_db
from scheduler.jobs import CryptoScheduler


def setup_logging() -> None:
    """Configure loguru logger with level from settings."""
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    logger.add(
        "logs/crypto_sygnalista.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
    )


async def main() -> None:
    """Application entry point.

    1. Sets up logging
    2. Initializes the database
    3. Starts the scheduler (runs forever until interrupted)
    """
    setup_logging()
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("crypto-sygnalista starting up")
    logger.info("dry_run={}, log_level={}", settings.dry_run, settings.log_level)
    logger.info("=" * 60)

    # Initialize database
    await init_db()
    logger.success("Database initialized.")

    # Start scheduler
    scheduler = CryptoScheduler(settings)
    await scheduler.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
