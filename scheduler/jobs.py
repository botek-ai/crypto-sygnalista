"""APScheduler job definitions for crypto-sygnalista."""

from __future__ import annotations

import asyncio
from typing import Optional

import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from config.settings import Settings, get_settings
from data.fetcher import BinanceFetcher
from notifications.telegram import TelegramNotifier
from signals.engine import SignalEngine


class CryptoScheduler:
    """Manages scheduled scanning jobs using APScheduler.

    Jobs:
    - ``scan_signals``: Scans all watchlist symbols every N seconds (default 60s).
    - ``purge_cache``: Clears stale cache entries periodically.
    - ``health_ping``: Optional heartbeat log message.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize the scheduler.

        Args:
            settings: Application settings.
        """
        self._settings = settings or get_settings()
        self._scheduler = AsyncIOScheduler()
        self._fetcher: Optional[BinanceFetcher] = None
        self._engine: Optional[SignalEngine] = None
        self._notifier: Optional[TelegramNotifier] = None
        self._symbols: list[str] = []
        self._signal_tf: str = "4h"
        self._daily_tf: str = "1d"

    async def _load_symbols(self) -> list[str]:
        """Load watchlist from symbols.yaml config file.

        Supports two modes:
        1. Explicit ``watchlist`` key — pairs listed directly (e.g. ``BTC/USDC``).
        2. Auto-build from ``pools`` + ``base_currency`` — each pool lists bare
           coin symbols (e.g. ``BTC``) and the method appends ``/<base_currency>``.

        Returns:
            List of trading pair strings.
        """
        path = self._settings.symbols_config_path
        try:
            with open(path) as f:
                config = yaml.safe_load(f)

            tf_config = config.get("timeframes", {})
            self._signal_tf = tf_config.get("signal", "4h")
            self._daily_tf = tf_config.get("daily", "1d")

            blacklist: set[str] = set(config.get("blacklist", []))

            # Mode 1: explicit watchlist
            explicit: list[str] = config.get("watchlist", [])
            if explicit:
                symbols = [s for s in explicit if s not in blacklist]
                logger.info(
                    "Loaded {} symbols from {} (explicit watchlist, signal={}, daily={})",
                    len(symbols),
                    path,
                    self._signal_tf,
                    self._daily_tf,
                )
                return symbols

            # Mode 2: build pairs from pools + base_currency
            base: str = config.get("base_currency", "USDC")
            pools: dict = config.get("pools", {})
            pairs: list[str] = []
            for pool_cfg in pools.values():
                for coin in pool_cfg.get("symbols", []):
                    pair = f"{coin}/{base}"
                    if pair not in blacklist:
                        pairs.append(pair)

            logger.info(
                "Loaded {} symbols from {} (base={}, signal={}, daily={})",
                len(pairs),
                path,
                base,
                self._signal_tf,
                self._daily_tf,
            )
            return pairs
        except Exception as exc:
            logger.error("Failed to load symbols from {}: {}", path, exc)
            return []

    async def _scan_signals_job(self) -> None:
        """Job: scan all symbols and send notifications for triggered signals."""
        if not self._engine or not self._symbols:
            logger.warning("scan_signals: engine or symbols not ready")
            return

        logger.debug("Starting signal scan for {} symbols...", len(self._symbols))
        try:
            events = await self._engine.scan_all(
                self._symbols,
                signal_tf=self._signal_tf,
                daily_tf=self._daily_tf,
            )
            for event in events:
                logger.info("Signal: {}", event)
                if self._notifier:
                    await self._notifier.send_signal(event)
        except Exception as exc:
            logger.error("scan_signals_job error: {}", exc)

    async def _purge_cache_job(self) -> None:
        """Job: purge expired cache entries."""
        if self._engine and hasattr(self._engine, "_cache"):
            removed = self._engine._cache.purge_expired()
            if removed:
                logger.debug("Cache purge: removed {} expired entries", removed)

    async def _health_ping_job(self) -> None:
        """Job: log a heartbeat message to confirm scheduler is alive."""
        logger.info(
            "❤️ Scheduler alive | symbols={} | dry_run={}",
            len(self._symbols),
            self._settings.dry_run,
        )

    async def start(self) -> None:
        """Initialize components and start the scheduler.

        This method loads symbols, creates the fetcher/engine/notifier,
        registers all jobs, and starts the APScheduler.
        """
        logger.info("Starting CryptoScheduler...")
        self._symbols = await self._load_symbols()

        self._fetcher = BinanceFetcher(self._settings)
        self._engine = SignalEngine(self._fetcher, self._settings)
        self._notifier = TelegramNotifier(self._settings)

        # Job: signal scan
        self._scheduler.add_job(
            self._scan_signals_job,
            trigger=IntervalTrigger(seconds=self._settings.scan_interval_seconds),
            id="scan_signals",
            name="Scan signals",
            max_instances=1,
            replace_existing=True,
        )

        # Job: cache purge
        self._scheduler.add_job(
            self._purge_cache_job,
            trigger=IntervalTrigger(seconds=120),
            id="purge_cache",
            name="Purge cache",
            max_instances=1,
            replace_existing=True,
        )

        # Job: health ping every 10 minutes
        self._scheduler.add_job(
            self._health_ping_job,
            trigger=IntervalTrigger(minutes=10),
            id="health_ping",
            name="Health ping",
            max_instances=1,
            replace_existing=True,
        )

        self._scheduler.start()
        logger.success(
            "Scheduler started — {} jobs registered, scan interval={}s",
            len(self._scheduler.get_jobs()),
            self._settings.scan_interval_seconds,
        )

        if self._notifier and self._settings.telegram_chat_id:
            await self._notifier.send_text(
                f"🚀 <b>crypto-sygnalista started</b>\n"
                f"Watching <b>{len(self._symbols)}</b> pairs "
                f"({'DRY RUN' if self._settings.dry_run else 'LIVE'})"
            )

    async def stop(self) -> None:
        """Stop the scheduler and clean up resources."""
        logger.info("Stopping CryptoScheduler...")
        self._scheduler.shutdown(wait=False)
        if self._fetcher:
            await self._fetcher.close()
        logger.info("Scheduler stopped.")

    async def run_forever(self) -> None:
        """Start the scheduler and block until interrupted."""
        await self.start()
        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Interrupt received, shutting down...")
        finally:
            await self.stop()
