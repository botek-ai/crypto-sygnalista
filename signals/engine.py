"""Signal engine — orchestrates fetching, indicator computation and v2 rule evaluation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

from loguru import logger

from backtest.engine import POOLS, get_pool_for_symbol
from config.settings import Settings, get_settings
from data.cache import OHLCVCache
from data.fetcher import BinanceFetcher
from indicators.technical import TechnicalIndicators
from signals.rules import BuyRules, SellRules, SignalEvent, SignalType


class PositionFilter:
    """Tracks open positions and enforces position filters.

    Filters:
    - Cooldown 12 min after buy/sell on the same coin
    - Max 8 open positions simultaneously
    - Max 15% exposure per coin
    - Min 30 USDC position size (v3)
    - Position size = 20% of free capital
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._last_signal: dict[str, datetime] = {}
        self._open_positions: dict[str, float] = {}
        self._total_capital: float = settings.initial_capital

    @property
    def open_count(self) -> int:
        return len(self._open_positions)

    @property
    def free_capital(self) -> float:
        used = sum(self._open_positions.values())
        return max(0.0, self._total_capital - used)

    def can_buy(self, symbol: str) -> tuple[bool, str]:
        s = self._settings

        if symbol in self._last_signal:
            cooldown_end = self._last_signal[symbol] + timedelta(minutes=s.cooldown_minutes)
            if datetime.now(tz=timezone.utc) < cooldown_end:
                remaining = (cooldown_end - datetime.now(tz=timezone.utc)).seconds // 60
                return False, f"Cooldown {symbol} ({remaining}min)"

        if self.open_count >= s.max_open_positions:
            return False, f"Max positions ({s.max_open_positions})"

        position_size = self.free_capital * s.position_size_pct
        min_size = 30.0  # v3 minimum
        if position_size < min_size:
            return False, f"Position too small ({position_size:.2f} < {min_size} USDC)"

        current_exposure = self._open_positions.get(symbol, 0.0)
        new_exposure = current_exposure + position_size
        if new_exposure / self._total_capital > s.max_coin_exposure_pct:
            return False, f"Max exposure {symbol} ({new_exposure/self._total_capital:.1%})"

        return True, "OK"

    def on_buy(self, symbol: str, price: float) -> float:
        size = self.free_capital * self._settings.position_size_pct
        self._open_positions[symbol] = self._open_positions.get(symbol, 0.0) + size
        self._last_signal[symbol] = datetime.now(tz=timezone.utc)
        logger.info("Position opened: {} @ {} — size={:.2f} USDC", symbol, price, size)
        return size

    def on_sell(self, symbol: str, price: float, entry_price: float, size_usdc: float) -> None:
        pnl_pct = (price - entry_price) / entry_price
        realized_pnl = size_usdc * pnl_pct
        self._total_capital += realized_pnl
        self._open_positions.pop(symbol, None)
        self._last_signal[symbol] = datetime.now(tz=timezone.utc)
        logger.info(
            "Position closed: {} @ {} — PnL={:.2%} ({:+.2f} USDC), capital={:.2f}",
            symbol, price, pnl_pct, realized_pnl, self._total_capital,
        )


class SignalEngine:
    """Main signal engine: fetches 4h data, computes indicators, runs v2 scoring.

    BUY: score >= 3/4 v2 conditions on 4h candles.
    SELL: per-pool TSL/SL tracked via 4h candle close prices in live mode.
    """

    def __init__(
        self,
        fetcher: BinanceFetcher,
        settings: Optional[Settings] = None,
    ) -> None:
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
        signal_tf: str = "4h",
        daily_tf: str = "1d",
    ) -> Optional[SignalEvent]:
        """Scan a single symbol on 4h timeframe, return BUY event if triggered."""
        df = self._cache.get(symbol, signal_tf)
        if df is None:
            df = await self._fetcher.fetch_ohlcv(symbol, timeframe=signal_tf, limit=100)
            self._cache.set(symbol, signal_tf, df)

        daily_df = self._cache.get(symbol, daily_tf)
        if daily_df is None:
            daily_df = await self._fetcher.fetch_ohlcv(symbol, timeframe=daily_tf, limit=60)
            self._cache.set(symbol, daily_tf, daily_df)

        ind = self._indicators.compute(df, daily_df=daily_df)

        # Get pool params for this symbol
        _, pool_params = get_pool_for_symbol(symbol)

        # V2 scoring buy check
        triggered, reason, score, conditions = self._buy_rules.check(
            ind,
            min_score=3,
            rsi_threshold=self._settings.rsi_buy_threshold,
            bb_pctb_threshold=0.3,
            volume_multiplier=self._settings.volume_multiplier,
            use_f1_ema_filter=False,
            use_f2_daily_filter=False,
        )

        if triggered:
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
                reason=reason,
                score=score,
                conditions=conditions,
                indicators=ind,
            )
            self._position_filter.on_buy(symbol, price)
            logger.success(
                "BUY signal: {} score={}/4 [{}] @ {:.4f}",
                symbol, score, ", ".join(conditions), price,
            )
            return event

        logger.debug("No signal for {}: {}", symbol, reason)
        return None

    async def scan_all(
        self,
        symbols: list[str],
        signal_tf: str = "4h",
        daily_tf: str = "1d",
        concurrency: int = 5,
    ) -> list[SignalEvent]:
        """Scan all symbols concurrently. Returns triggered signal events."""
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
            "Scan complete: {} symbols → {} signals",
            len(symbols), len(events),
        )
        return events
