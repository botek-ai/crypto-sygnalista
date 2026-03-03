"""Technical indicator calculations using pandas-ta."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pandas_ta as ta  # type: ignore[import]
from loguru import logger
from pydantic import BaseModel


class IndicatorResult(BaseModel):
    """Computed indicator values for a single candle (latest)."""

    # RSI
    rsi: Optional[float] = None
    rsi_prev: Optional[float] = None  # previous candle RSI (to detect rising)

    # MACD
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_prev: Optional[float] = None
    macd_signal_prev: Optional[float] = None

    # EMA
    ema9: Optional[float] = None
    ema21: Optional[float] = None
    ema50_daily: Optional[float] = None  # from daily timeframe

    # Bollinger Bands
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None

    # Volume
    volume: Optional[float] = None
    volume_avg20: Optional[float] = None

    # Price
    close: Optional[float] = None
    close_daily: Optional[float] = None  # latest daily close

    @property
    def rsi_rising(self) -> bool:
        """Return True if RSI is rising (current > previous)."""
        if self.rsi is None or self.rsi_prev is None:
            return False
        return self.rsi > self.rsi_prev

    @property
    def macd_bullish_crossover(self) -> bool:
        """Return True if MACD crossed above signal line (bullish crossover).

        Crossover: previous MACD <= previous signal AND current MACD > current signal.
        """
        if None in (self.macd, self.macd_signal, self.macd_prev, self.macd_signal_prev):
            return False
        assert self.macd is not None
        assert self.macd_signal is not None
        assert self.macd_prev is not None
        assert self.macd_signal_prev is not None
        return self.macd_prev <= self.macd_signal_prev and self.macd > self.macd_signal

    @property
    def macd_bearish_crossover(self) -> bool:
        """Return True if MACD crossed below signal line (bearish crossover)."""
        if None in (self.macd, self.macd_signal, self.macd_prev, self.macd_signal_prev):
            return False
        assert self.macd is not None
        assert self.macd_signal is not None
        assert self.macd_prev is not None
        assert self.macd_signal_prev is not None
        return self.macd_prev >= self.macd_signal_prev and self.macd < self.macd_signal

    @property
    def bb_squeeze(self) -> bool:
        """Return True if Bollinger Bands are in squeeze (width < threshold)."""
        if self.bb_width is None:
            return False
        return self.bb_width < 0.03

    @property
    def volume_spike(self) -> bool:
        """Return True if current volume > 1.5× 20-candle average."""
        if self.volume is None or self.volume_avg20 is None or self.volume_avg20 == 0:
            return False
        return self.volume > 1.5 * self.volume_avg20

    @property
    def daily_trend_positive(self) -> bool:
        """Return True if daily close > EMA50 (daily trend is bullish)."""
        if self.close_daily is None or self.ema50_daily is None:
            return False
        return self.close_daily > self.ema50_daily


class TechnicalIndicators:
    """Compute technical indicators from OHLCV DataFrames.

    Uses pandas-ta for all indicator calculations.
    Supports both signal-timeframe and daily-timeframe DataFrames.
    """

    def __init__(self, bb_squeeze_threshold: float = 0.03) -> None:
        """Initialize.

        Args:
            bb_squeeze_threshold: BB width below which squeeze is active.
        """
        self._bb_squeeze_threshold = bb_squeeze_threshold

    def compute(
        self,
        df: pd.DataFrame,
        daily_df: Optional[pd.DataFrame] = None,
    ) -> IndicatorResult:
        """Compute all indicators from an OHLCV DataFrame.

        Args:
            df: Signal-timeframe OHLCV DataFrame (e.g., 15m).
                Must have columns: open, high, low, close, volume.
            daily_df: Optional daily OHLCV DataFrame for daily trend check.

        Returns:
            IndicatorResult with all computed values (None if insufficient data).
        """
        if len(df) < 50:
            logger.warning("DataFrame too short ({} rows), need >= 50", len(df))
            return IndicatorResult()

        result_data: dict[str, float | None] = {}

        # --- RSI ---
        rsi_series = ta.rsi(df["close"], length=14)
        if rsi_series is not None and not rsi_series.empty:
            result_data["rsi"] = self._last(rsi_series)
            result_data["rsi_prev"] = self._prev(rsi_series)

        # --- MACD ---
        macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            macd_col = [c for c in macd_df.columns if c.startswith("MACD_") and "s" not in c.lower() and "h" not in c.lower()]
            signal_col = [c for c in macd_df.columns if "MACDs_" in c]
            if macd_col and signal_col:
                result_data["macd"] = self._last(macd_df[macd_col[0]])
                result_data["macd_prev"] = self._prev(macd_df[macd_col[0]])
                result_data["macd_signal"] = self._last(macd_df[signal_col[0]])
                result_data["macd_signal_prev"] = self._prev(macd_df[signal_col[0]])

        # --- EMA 9, 21 ---
        ema9 = ta.ema(df["close"], length=9)
        ema21 = ta.ema(df["close"], length=21)
        if ema9 is not None:
            result_data["ema9"] = self._last(ema9)
        if ema21 is not None:
            result_data["ema21"] = self._last(ema21)

        # --- Bollinger Bands ---
        bb = ta.bbands(df["close"], length=20, std=2.0)
        if bb is not None and not bb.empty:
            upper_col = [c for c in bb.columns if "BBU_" in c]
            lower_col = [c for c in bb.columns if "BBL_" in c]
            mid_col = [c for c in bb.columns if "BBM_" in c]
            if upper_col and lower_col and mid_col:
                upper = self._last(bb[upper_col[0]])
                lower = self._last(bb[lower_col[0]])
                middle = self._last(bb[mid_col[0]])
                result_data["bb_upper"] = upper
                result_data["bb_lower"] = lower
                result_data["bb_middle"] = middle
                # BB width = (upper - lower) / middle
                if upper is not None and lower is not None and middle and middle != 0:
                    result_data["bb_width"] = (upper - lower) / middle

        # --- Volume ---
        result_data["volume"] = float(df["volume"].iloc[-1])
        result_data["volume_avg20"] = float(df["volume"].rolling(20).mean().iloc[-1])

        # --- Close ---
        result_data["close"] = float(df["close"].iloc[-1])

        # --- Daily trend: EMA50 ---
        if daily_df is not None and len(daily_df) >= 50:
            ema50_d = ta.ema(daily_df["close"], length=50)
            if ema50_d is not None:
                result_data["ema50_daily"] = self._last(ema50_d)
            result_data["close_daily"] = float(daily_df["close"].iloc[-1])

        return IndicatorResult(**result_data)

    @staticmethod
    def _last(series: pd.Series) -> Optional[float]:
        """Return last non-NaN value or None."""
        val = series.dropna().iloc[-1] if not series.dropna().empty else None
        return float(val) if val is not None else None

    @staticmethod
    def _prev(series: pd.Series) -> Optional[float]:
        """Return second-to-last non-NaN value or None."""
        clean = series.dropna()
        if len(clean) < 2:
            return None
        return float(clean.iloc[-2])
