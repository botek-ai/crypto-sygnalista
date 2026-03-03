"""In-memory OHLCV cache with TTL support."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from loguru import logger


@dataclass
class _CacheEntry:
    """Single cache entry with expiry tracking."""

    data: pd.DataFrame
    fetched_at: float = field(default_factory=time.monotonic)
    ttl_seconds: float = 55.0  # slightly below 1-min candle interval

    @property
    def is_expired(self) -> bool:
        """Return True if this entry has exceeded its TTL."""
        return (time.monotonic() - self.fetched_at) > self.ttl_seconds


class OHLCVCache:
    """Thread-safe (single-threaded asyncio) in-memory cache for OHLCV DataFrames.

    Stores per (symbol, timeframe) entries. Entries expire after ``ttl_seconds``.
    Useful to avoid redundant Binance API calls within a single scan cycle.
    """

    def __init__(self, ttl_seconds: float = 55.0) -> None:
        """Initialize cache.

        Args:
            ttl_seconds: Cache entry lifetime. Defaults to 55s (just under 1-min candle).
        """
        self._ttl = ttl_seconds
        self._store: dict[tuple[str, str], _CacheEntry] = {}

    def _key(self, symbol: str, timeframe: str) -> tuple[str, str]:
        return (symbol, timeframe)

    def get(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Return cached DataFrame if present and not expired.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT".
            timeframe: Candle timeframe.

        Returns:
            Cached DataFrame or None if missing/expired.
        """
        entry = self._store.get(self._key(symbol, timeframe))
        if entry is None:
            return None
        if entry.is_expired:
            del self._store[self._key(symbol, timeframe)]
            logger.trace("Cache expired for {} @ {}", symbol, timeframe)
            return None
        logger.trace("Cache hit for {} @ {}", symbol, timeframe)
        return entry.data

    def set(self, symbol: str, timeframe: str, data: pd.DataFrame) -> None:
        """Store a DataFrame in the cache.

        Args:
            symbol: Trading pair.
            timeframe: Candle timeframe.
            data: OHLCV DataFrame to cache.
        """
        self._store[self._key(symbol, timeframe)] = _CacheEntry(
            data=data, ttl_seconds=self._ttl
        )
        logger.trace("Cache set for {} @ {}", symbol, timeframe)

    def invalidate(self, symbol: str, timeframe: str) -> None:
        """Remove a specific entry from the cache.

        Args:
            symbol: Trading pair.
            timeframe: Candle timeframe.
        """
        self._store.pop(self._key(symbol, timeframe), None)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._store.clear()
        logger.debug("OHLCV cache cleared.")

    def purge_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        expired = [k for k, v in self._store.items() if v.is_expired]
        for k in expired:
            del self._store[k]
        if expired:
            logger.debug("Purged {} expired cache entries.", len(expired))
        return len(expired)

    def __len__(self) -> int:
        """Return number of cached entries (including possibly expired ones)."""
        return len(self._store)
