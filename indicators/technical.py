"""Technical indicator calculations using pandas-ta."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore[import]
from loguru import logger
from pydantic import BaseModel


class IndicatorResult(BaseModel):
    """Computed indicator values for a single candle (latest)."""

    # RSI
    rsi: Optional[float] = None
    rsi_prev: Optional[float] = None

    # StochRSI
    stochrsi_k: Optional[float] = None
    stochrsi_k_prev: Optional[float] = None
    stochrsi_d: Optional[float] = None

    # MACD
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_prev: Optional[float] = None
    macd_signal_prev: Optional[float] = None

    # EMA
    ema9: Optional[float] = None
    ema21: Optional[float] = None
    ema21_daily: Optional[float] = None  # from daily timeframe
    ema50_daily: Optional[float] = None  # from daily timeframe (legacy)

    # Bollinger Bands
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None
    bb_pct_b: Optional[float] = None  # BB%B

    # RSI divergence
    rsi_bullish_divergence: Optional[bool] = None

    # Volume
    volume: Optional[float] = None
    volume_avg20: Optional[float] = None

    # Price
    close: Optional[float] = None
    close_daily: Optional[float] = None

    @property
    def rsi_rising(self) -> bool:
        if self.rsi is None or self.rsi_prev is None:
            return False
        return self.rsi > self.rsi_prev

    @property
    def macd_bullish_crossover(self) -> bool:
        if None in (self.macd, self.macd_signal, self.macd_prev, self.macd_signal_prev):
            return False
        return self.macd_prev <= self.macd_signal_prev and self.macd > self.macd_signal  # type: ignore[operator]

    @property
    def macd_bearish_crossover(self) -> bool:
        if None in (self.macd, self.macd_signal, self.macd_prev, self.macd_signal_prev):
            return False
        return self.macd_prev >= self.macd_signal_prev and self.macd < self.macd_signal  # type: ignore[operator]

    @property
    def stochrsi_k_cross_up(self) -> bool:
        """K crossed above D from below 20."""
        if None in (self.stochrsi_k, self.stochrsi_k_prev, self.stochrsi_d):
            return False
        return (
            self.stochrsi_k_prev < 20  # type: ignore[operator]
            and self.stochrsi_k > self.stochrsi_d  # type: ignore[operator]
        )

    @property
    def bb_squeeze(self) -> bool:
        if self.bb_width is None:
            return False
        return self.bb_width < 0.03

    @property
    def volume_spike(self) -> bool:
        if self.volume is None or self.volume_avg20 is None or self.volume_avg20 == 0:
            return False
        return self.volume > 1.5 * self.volume_avg20

    @property
    def daily_trend_positive(self) -> bool:
        if self.close_daily is None or self.ema50_daily is None:
            return False
        return self.close_daily > self.ema50_daily

    @property
    def daily_trend_ema21(self) -> bool:
        """Daily close above EMA21 daily."""
        if self.close_daily is None or self.ema21_daily is None:
            return False
        return self.close_daily > self.ema21_daily


class TechnicalIndicators:
    """Compute technical indicators from OHLCV DataFrames."""

    def __init__(self, bb_squeeze_threshold: float = 0.03) -> None:
        self._bb_squeeze_threshold = bb_squeeze_threshold

    def compute(
        self,
        df: pd.DataFrame,
        daily_df: Optional[pd.DataFrame] = None,
    ) -> IndicatorResult:
        if len(df) < 50:
            logger.warning("DataFrame too short ({} rows), need >= 50", len(df))
            return IndicatorResult()

        result_data: dict[str, object] = {}

        # --- RSI ---
        rsi_series = ta.rsi(df["close"], length=14)
        if rsi_series is not None and not rsi_series.empty:
            result_data["rsi"] = self._last(rsi_series)
            result_data["rsi_prev"] = self._prev(rsi_series)

        # --- StochRSI ---
        stochrsi_df = ta.stochrsi(df["close"], length=14, rsi_length=14, k=3, d=3)
        if stochrsi_df is not None and not stochrsi_df.empty:
            k_cols = [c for c in stochrsi_df.columns if "STOCHRSIk_" in c]
            d_cols = [c for c in stochrsi_df.columns if "STOCHRSId_" in c]
            if k_cols:
                result_data["stochrsi_k"] = self._last(stochrsi_df[k_cols[0]])
                result_data["stochrsi_k_prev"] = self._prev(stochrsi_df[k_cols[0]])
            if d_cols:
                result_data["stochrsi_d"] = self._last(stochrsi_df[d_cols[0]])

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
            pctb_col = [c for c in bb.columns if "BBP_" in c]
            if upper_col and lower_col and mid_col:
                upper = self._last(bb[upper_col[0]])
                lower = self._last(bb[lower_col[0]])
                middle = self._last(bb[mid_col[0]])
                result_data["bb_upper"] = upper
                result_data["bb_lower"] = lower
                result_data["bb_middle"] = middle
                if upper is not None and lower is not None and middle and middle != 0:
                    result_data["bb_width"] = (upper - lower) / middle
            if pctb_col:
                result_data["bb_pct_b"] = self._last(bb[pctb_col[0]])

        # --- RSI Bullish Divergence (window=10) ---
        result_data["rsi_bullish_divergence"] = self._detect_rsi_divergence(df, rsi_series)

        # --- Volume ---
        result_data["volume"] = float(df["volume"].iloc[-1])
        result_data["volume_avg20"] = float(df["volume"].rolling(20).mean().iloc[-1])

        # --- Close ---
        result_data["close"] = float(df["close"].iloc[-1])

        # --- Daily trend: EMA21 + EMA50 ---
        if daily_df is not None and len(daily_df) >= 21:
            ema21_d = ta.ema(daily_df["close"], length=21)
            if ema21_d is not None:
                result_data["ema21_daily"] = self._last(ema21_d)
            if len(daily_df) >= 50:
                ema50_d = ta.ema(daily_df["close"], length=50)
                if ema50_d is not None:
                    result_data["ema50_daily"] = self._last(ema50_d)
            result_data["close_daily"] = float(daily_df["close"].iloc[-1])

        return IndicatorResult(**result_data)

    @staticmethod
    def _detect_rsi_divergence(df: pd.DataFrame, rsi_series: Optional[pd.Series], window: int = 10) -> bool:
        """Detect RSI bullish divergence: price makes lower low, RSI makes higher low."""
        if rsi_series is None or len(df) < window + 1:
            return False
        prices = df["close"].values[-window:]
        rsiv = rsi_series.values[-window:]
        if np.any(np.isnan(rsiv)):
            return False
        # Price lower low: last close < min of previous window-1
        price_lower_low = prices[-1] < np.min(prices[:-1])
        # RSI higher low: last RSI > min of previous window-1 RSI
        rsi_higher_low = rsiv[-1] > np.min(rsiv[:-1])
        return bool(price_lower_low and rsi_higher_low)

    @staticmethod
    def _last(series: pd.Series) -> Optional[float]:
        val = series.dropna().iloc[-1] if not series.dropna().empty else None
        return float(val) if val is not None else None

    @staticmethod
    def _prev(series: pd.Series) -> Optional[float]:
        clean = series.dropna()
        if len(clean) < 2:
            return None
        return float(clean.iloc[-2])
