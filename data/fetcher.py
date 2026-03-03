"""Binance OHLCV fetcher using ccxt."""

from __future__ import annotations

import asyncio
from typing import Any

import ccxt.pro as ccxtpro
import pandas as pd
from loguru import logger

from config.settings import Settings, get_settings


class BinanceFetcher:
    """Async OHLCV fetcher for Binance via ccxt.

    Handles rate limiting, retries, and converts raw OHLCV data to
    pandas DataFrames with proper column names and dtypes.
    """

    OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the fetcher with application settings.

        Args:
            settings: Application settings. Defaults to get_settings().
        """
        self._settings = settings or get_settings()
        self._exchange: ccxtpro.binance | None = None

    async def _get_exchange(self) -> ccxtpro.binance:
        """Lazily initialize and return the ccxt Binance exchange instance."""
        if self._exchange is None:
            config: dict[str, Any] = {
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
            if self._settings.binance_api_key:
                config["apiKey"] = self._settings.binance_api_key
                config["secret"] = self._settings.binance_api_secret
            if self._settings.binance_testnet:
                config["sandbox"] = True

            self._exchange = ccxtpro.binance(config)
            logger.debug(
                "Initialized Binance exchange (testnet={})",
                self._settings.binance_testnet,
            )
        return self._exchange

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15m",
        limit: int = 100,
        retries: int = 3,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles for a symbol and timeframe.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT".
            timeframe: Candle timeframe, e.g. "15m", "1h", "1d".
            limit: Number of candles to fetch.
            retries: Number of retry attempts on failure.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
            Index is datetime (UTC).

        Raises:
            ccxt.NetworkError: After exhausting retries.
            ccxt.ExchangeError: On exchange-side errors.
        """
        exchange = await self._get_exchange()
        last_exc: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                raw: list[list[float]] = await exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, limit=limit
                )
                df = pd.DataFrame(raw, columns=self.OHLCV_COLUMNS)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.set_index("timestamp")
                df = df.astype(float)
                logger.debug(
                    "Fetched {} candles for {} @ {}",
                    len(df),
                    symbol,
                    timeframe,
                )
                return df
            except Exception as exc:
                last_exc = exc
                wait = 2**attempt
                logger.warning(
                    "fetch_ohlcv failed for {} (attempt {}/{}): {}. Retrying in {}s.",
                    symbol,
                    attempt,
                    retries,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"fetch_ohlcv failed for {symbol} after {retries} retries"
        ) from last_exc

    async def fetch_many(
        self,
        symbols: list[str],
        timeframe: str = "15m",
        limit: int = 100,
        concurrency: int = 5,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple symbols concurrently.

        Args:
            symbols: List of trading pairs.
            timeframe: Candle timeframe.
            limit: Number of candles per symbol.
            concurrency: Max simultaneous requests.

        Returns:
            Dict mapping symbol → DataFrame. Failed symbols are omitted.
        """
        semaphore = asyncio.Semaphore(concurrency)
        results: dict[str, pd.DataFrame] = {}

        async def _fetch_one(sym: str) -> None:
            async with semaphore:
                try:
                    results[sym] = await self.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
                except Exception as exc:
                    logger.error("Skipping {} — fetch error: {}", sym, exc)

        await asyncio.gather(*[_fetch_one(s) for s in symbols])
        logger.info(
            "fetch_many: {}/{} symbols fetched successfully",
            len(results),
            len(symbols),
        )
        return results

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """Fetch current ticker data for a symbol.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT".

        Returns:
            ccxt ticker dict with keys: last, bid, ask, volume, etc.
        """
        exchange = await self._get_exchange()
        ticker = await exchange.fetch_ticker(symbol)
        logger.debug("Ticker {}: last={}", symbol, ticker.get("last"))
        return ticker  # type: ignore[return-value]

    async def close(self) -> None:
        """Close the underlying exchange connection."""
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None
            logger.debug("Binance exchange connection closed.")

    async def __aenter__(self) -> "BinanceFetcher":
        """Support async context manager usage."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close exchange on context manager exit."""
        await self.close()
