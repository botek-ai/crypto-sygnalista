"""Signal engine — orchestrates fetching, indicator computation and rule evaluation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

from loguru import logger

from config.settings import Settings, get_settings
from data.cache import OHLCVCache
from data.fetcher import BinanceFetcher
from indicators.technical import TechnicalIndicators
from signals.rules import BuyRules, SellRules, SignalEvent, SignalType


class PositionFilter:
    """Tracks open positions and enforces position filters from Jacek's strategy.

    Filters:
    - Cooldown 12 min after buy/sell on the same coin
    - Max 8 open positions simultaneously
    - Max 15% exposure per coin
    - Min 15 USDC position size
    - Position size = 20% of free capital
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize with settings."""
        self._settings = settings
        # symbol → last signal timestamp
        self._last_signal: dict[str, datetime] = {}
        # symbol → position size in USDC
        self._open_positions: dict[str, float] = {}
        self._total_capital: float = settings.initial_capital

    @property
    def open_count(self) -> int:
        """Number of currently open positions."""
        return len(self._open_positions)

    @property
    def free_capital(self) -> float:
        """Available capital not tied up in open positions."""
        used = sum(self._open_positions.values())
        return max(0.0, self._total_capital - used)

    def can_buy(self, symbol: str) -> tuple[bool, str]:
        """Check all filters before allowing a buy.

        Args:
            symbol: Trading pair.

        Returns:
            Tuple (allowed: bool, reason: str).
        """
        s = self._settings

        # Cooldown check
        if symbol in self._last_signal:
            cooldown_end = self._last_signal[symbol] + timedelta(minutes=s.cooldown_minutes)
            if datetime.now(tz=timezone.utc) < cooldown_end:
                remaining = (cooldown_end - datetime.now(tz=timezone.utc)).seconds // 60
                return False, f"Cooldown active for {symbol} ({remaining}min remaining)"

        # Max positions
        if self.open_count >= s.max_open_positions:
            return False, f"Max open positions reached ({s.max_open_positions})"

        # Min position size
        position_size = self.free_capital * s.position_size_pct
        if position_size < s.min_position_usdc:
            return False, (
                f"Position size too small ({position_size:.2f} < {s.min_position_usdc} USDC)"
            )

        # Max coin exposure
        current_exposure = self._open_positions.get(symbol, 0.0)
        new_exposure = current_exposure + position_size
        if new_exposure / self._total_capital > s.max_coin_exposure_pct:
            return False, (
                f"Max exposure exceeded for {symbol} "
                f"({new_exposure / self._total_capital:.1%} > {s.max_coin_exposure_pct:.0%})"
            )

        return True, "OK"

    def on_buy(self, symbol: str, price: float) -> float:
        """Record a buy and return the position size in USDC.

        Args:
            symbol: Trading pair.
            price: Entry price.

        Returns:
            Position size in USDC.
        """
        size = self.free_capital * self._settings.position_size_pct
        self._open_positions[symbol] = self._open_positions.get(symbol, 0.0) + size
        self._last_signal[symbol] = datetime.now(tz=timezone.utc)
        logger.info("Position opened: {} @ {} — size={:.2f} USDC", symbol, price, size)
        return size

    def on_sell(self, symbol: str, price: float, entry_price: float, size_usdc: float) -> None:
        """Record a sell and update capital.

        Args:
            symbol: Trading pair.
            price: Exit price.
            entry_price: Entry price.
            size_usdc: Position size in USDC.
        """
        pnl_pct = (price - entry_price) / entry_price
        realized_pnl = size_usdc * pnl_pct
        self._total_capital += realized_pnl
        self._open_positions.pop(symbol, None)
        self._last_signal[symbol] = datetime.now(tz=timezone.utc)
        logger.info(
            "Position closed: {} @ {} — PnL={:.2%} ({:+.2f} USDC), capital={:.2f}",
            symbol,
            price,
            pnl_pct,
            realized_pnl,
            self._total_capital,
        )


class SignalEngine:
    """Main signal engine: fetches data, computes indicators, evaluates rules.

    Scans all symbols on signal timeframe (15m) and emits BUY/SELL events.
    """

    def __init__(
        self,
        fetcher: BinanceFetcher,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the signal engine.

        Args:
            fetcher: BinanceFetcher instance.
            settings: Application settings.
        """
        self._fetcher = fetcher
        self._settings = settings or get_settings()
        self._cache = OHLCVCache(ttl_seconds=55)
        self._indicators = TechnicalIndicators(
            bb_squeeze_threshold=self._settings.bb_squeeze_width
        )
        self._buy_rules = BuyRules()
        self._sell_rules = SellRules()
        self._position_filter = PositionFilter(self._settings)

    async def scan_symbol(
        self,
        symbol: str,
        signal_tf: str = "15m",
        daily_tf: str = "1d",
    ) -> Optional[SignalEvent]:
        """Scan a single symbol and return a signal event if triggered.

        Args:
            symbol: Trading pair.
            signal_tf: Signal timeframe (default 15m).
            daily_tf: Daily timeframe for trend filter (default 1d).

        Returns:
            SignalEvent if BUY triggered, None otherwise.
        """
        # Fetch signal-timeframe data (with cache)
        df = self._cache.get(symbol, signal_tf)
        if df is None:
            df = await self._fetcher.fetch_ohlcv(symbol, timeframe=signal_tf, limit=100)
            self._cache.set(symbol, signal_tf, df)

        # Fetch daily data (with cache)
        daily_df = self._cache.get(symbol, daily_tf)
        if daily_df is None:
            daily_df = await self._fetcher.fetch_ohlcv(symbol, timeframe=daily_tf, limit=60)
            self._cache.set(symbol, daily_tf, daily_df)

        ind = self._indicators.compute(df, daily_df=daily_df)

        # Check buy rules
        buy_ok, buy_reason = self._buy_rules.check(
            ind, rsi_threshold=self._settings.rsi_buy_threshold
        )
        if buy_ok:
            # Apply position filters
            can_buy, filter_reason = self._position_filter.can_buy(symbol)
            if not can_buy:
                logger.info("BUY signal for {} blocked: {}", symbol, filter_reason)
                return None

            price = ind.close or 0.0
            event = SignalEvent(
                symbol=symbol,
                signal=SignalType.BUY,
                price=price,
                timestamp=datetime.now(tz=timezone.utc),
                reason=buy_reason,
                indicators=ind,
            )
            self._position_filter.on_buy(symbol, price)
            logger.success("BUY signal: {}", event)
            return event

        return None

    async def scan_all(
        self,
        symbols: list[str],
        signal_tf: str = "15m",
        daily_tf: str = "1d",
        concurrency: int = 5,
    ) -> list[SignalEvent]:
        """Scan all symbols and return triggered signal events.

        Args:
            symbols: List of trading pairs.
            signal_tf: Signal timeframe.
            daily_tf: Daily timeframe.
            concurrency: Max concurrent scans.

        Returns:
            List of triggered SignalEvents (may be empty).
        """
        semaphore = asyncio.Semaphore(concurrency)
        events: list[SignalEvent] = []

        async def _scan(sym: str) -> None:
            async with semaphore:
                try:
                    event = await self.scan_symbol(sym, signal_tf=signal_tf, daily_tf=daily_tf)
                    if event:
                        events.append(event)
                except Exception as exc:
                    logger.error("Error scanning {}: {}", sym, exc)

        await asyncio.gather(*[_scan(s) for s in symbols])
        logger.info(
            "Scan complete: {}/{} symbols → {} signals",
            len(symbols),
            len(symbols),
            len(events),
        )
        return events
