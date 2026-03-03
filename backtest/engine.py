"""Backtest engine — iterates over historical OHLCV candles without look-ahead bias."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pandas_ta as ta  # type: ignore[import]
from loguru import logger

from backtest.portfolio import Portfolio, Position, Trade
from config.settings import Settings, get_settings

# ---------------------------------------------------------------------------
# Volatility pools — per-pool trailing stop and hard stop-loss parameters
# ---------------------------------------------------------------------------

POOLS: dict[str, dict] = {
    "high": {
        "symbols": ["INJ/USDT", "RENDER/USDT", "SEI/USDT"],
        "trailing_stop": 0.013,  # 1.3%
        "stop_loss": 0.025,      # 2.5%
    },
    "mid": {
        "symbols": [
            "LINK/USDT", "SOL/USDT", "AVAX/USDT", "AAVE/USDT", "ATOM/USDT",
            "NEAR/USDT", "OP/USDT", "FIL/USDT", "ICP/USDT",
        ],
        "trailing_stop": 0.011,  # 1.1%
        "stop_loss": 0.025,      # 2.5%
    },
    "low": {
        "symbols": ["BNB/USDT", "BTC/USDT", "ETH/USDT"],
        "trailing_stop": 0.009,  # 0.9%
        "stop_loss": 0.025,      # 2.5%
    },
}

ALL_SYMBOLS = [s for pool in POOLS.values() for s in pool["symbols"]]


def get_pool_for_symbol(symbol: str) -> tuple[str, dict]:
    """Return (pool_name, pool_params) for a given symbol."""
    for pool_name, pool in POOLS.items():
        if symbol in pool["symbols"]:
            return pool_name, pool
    # Fallback: mid pool
    return "mid", POOLS["mid"]


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""

    symbol: str
    timeframe: str = "4h"
    initial_capital: float = 1000.0
    # Position management
    max_positions: int = 8
    position_size_pct: float = 0.20
    min_position_usdc: float = 15.0
    max_coin_exposure_pct: float = 0.15
    cooldown_minutes: int = 12
    # Sell rules — per-pool trailing stop + hard stop-loss
    trailing_stop_pct: float = 0.011   # % drop from peak price to trigger close
    stop_loss_pct: float = 0.025       # hard stop-loss from entry
    emergency_minutes: int = 90        # emergency exit: stuck position
    emergency_threshold_pct: float = 0.01
    # Buy rules
    rsi_threshold: float = 45.0        # RSI < 45 (relaxed from 35 to generate more signals)
    use_ema_filter: bool = False        # If True, require EMA9 > EMA21 for entry (more restrictive)
    bb_squeeze_width: float = 0.03
    volume_multiplier: float = 1.5


def _timeframe_minutes(tf: str) -> int:
    """Convert timeframe string to minutes."""
    units = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    for suffix, mult in units.items():
        if tf.endswith(suffix):
            try:
                return int(tf[:-1]) * mult
            except ValueError:
                pass
    return 60  # default


class BacktestEngine:
    """Candle-by-candle backtest engine for multi-indicator strategy.

    No look-ahead bias: at step i, only candles 0..i are visible.
    Indicators are precomputed on the full DataFrame (all are backward-looking)
    but accessed at index i.

    Trailing stop logic (per pool):
        - Position tracks ``peak_price`` — highest close seen since entry.
        - peak_price is updated each candle when close > previous peak.
        - When close drops ``trailing_stop_pct`` % below peak_price → close position.

    Args:
        df: Signal-timeframe OHLCV DataFrame (must have: open, high, low, close, volume).
            Index must be DatetimeIndex (UTC).
        daily_df: Daily OHLCV DataFrame for daily trend filter.
        config: Backtest configuration.
    """

    # Minimum candles needed before signals can fire
    MIN_CANDLES = 60

    def __init__(
        self,
        df: pd.DataFrame,
        daily_df: Optional[pd.DataFrame],
        config: Optional[BacktestConfig] = None,
    ) -> None:
        self.df = df.copy()
        self.daily_df = daily_df.copy() if daily_df is not None else None
        self.config = config or BacktestConfig(symbol="UNKNOWN")
        self.tf_minutes = _timeframe_minutes(self.config.timeframe)

        self.portfolio = Portfolio(
            initial_capital=self.config.initial_capital,
            max_positions=self.config.max_positions,
            position_size_pct=self.config.position_size_pct,
            min_position_usdc=self.config.min_position_usdc,
            max_coin_exposure_pct=self.config.max_coin_exposure_pct,
            cooldown_minutes=self.config.cooldown_minutes,
        )

        # Precomputed indicator columns (populated in _precompute)
        self._ind: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        """Run the full backtest and return statistics.

        Returns:
            dict with trades list and portfolio statistics.
        """
        logger.info(
            "Starting backtest: {} | {} | {} candles | capital={:.0f} USDC | "
            "trailing={:.1%} | sl={:.1%} | rsi<{} | ema_filter={}",
            self.config.symbol,
            self.config.timeframe,
            len(self.df),
            self.config.initial_capital,
            self.config.trailing_stop_pct,
            self.config.stop_loss_pct,
            self.config.rsi_threshold,
            self.config.use_ema_filter,
        )

        self._precompute_indicators()

        symbol = self.config.symbol
        tf_min = self.tf_minutes

        for i in range(self.MIN_CANDLES, len(self.df)):
            row = self.df.iloc[i]
            ind_row = self._ind.iloc[i]
            now: datetime = self.df.index[i].to_pydatetime()  # type: ignore[union-attr]
            close = float(row["close"])
            high = float(row["high"])

            # --- Update peak prices for open positions ---
            for pos in self.portfolio.positions.get(symbol, []):
                if high > pos.peak_price:
                    pos.peak_price = high

            # --- Check sell conditions for open positions ---
            positions_snapshot = list(self.portfolio.positions.get(symbol, []))
            for pos in positions_snapshot:
                sell, reason = self._check_sell(pos, close, now, tf_min)
                if sell:
                    self.portfolio.close_position(pos, close, now, reason)

            # --- Record equity snapshot ---
            self.portfolio.record_equity(now, {symbol: close})

            # --- Check buy conditions ---
            if self._check_buy(ind_row):
                self.portfolio.open_position(symbol, close, now)

        # Force-close any remaining open positions at last price
        last_close = float(self.df["close"].iloc[-1])
        last_time = self.df.index[-1].to_pydatetime()  # type: ignore[union-attr]
        for pos in list(self.portfolio.positions.get(symbol, [])):
            self.portfolio.close_position(pos, last_close, last_time, "End of backtest")

        stats = self.portfolio.get_stats()
        stats["symbol"] = symbol
        stats["timeframe"] = self.config.timeframe
        stats["candles"] = len(self.df)
        stats["period_start"] = str(self.df.index[0])
        stats["period_end"] = str(self.df.index[-1])
        stats["trailing_stop_pct"] = self.config.trailing_stop_pct
        stats["stop_loss_pct"] = self.config.stop_loss_pct
        stats["rsi_threshold"] = self.config.rsi_threshold
        stats["use_ema_filter"] = self.config.use_ema_filter

        # Condition trigger analysis
        stats["condition_analysis"] = self._analyze_conditions()

        logger.success(
            "Backtest complete: {} trades | return={:.2%} | win_rate={:.1%} | max_dd={:.2%}",
            stats.get("total_trades", 0),
            stats.get("total_return_pct", 0.0),
            stats.get("win_rate", 0.0),
            stats.get("max_drawdown_pct", 0.0),
        )

        return {
            "stats": stats,
            "trades": self.portfolio.trades,
            "equity_curve": self.portfolio._equity_curve,
        }

    # ------------------------------------------------------------------ #
    # Indicator precomputation                                             #
    # ------------------------------------------------------------------ #

    def _precompute_indicators(self) -> None:
        """Precompute all indicator columns on the full DataFrame.

        All pandas-ta indicators are purely backward-looking (they only use
        past data), so computing them on the full df is safe — no look-ahead.
        """
        df = self.df
        ind = pd.DataFrame(index=df.index)

        # RSI
        rsi = ta.rsi(df["close"], length=14)
        if rsi is not None:
            ind["rsi"] = rsi
            ind["rsi_prev"] = rsi.shift(1)

        # MACD
        macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            macd_cols = [c for c in macd_df.columns if c.startswith("MACD_") and "s" not in c.lower() and "h" not in c.lower()]
            signal_cols = [c for c in macd_df.columns if "MACDs_" in c]
            if macd_cols:
                ind["macd"] = macd_df[macd_cols[0]]
                ind["macd_prev"] = macd_df[macd_cols[0]].shift(1)
            if signal_cols:
                ind["macd_signal"] = macd_df[signal_cols[0]]
                ind["macd_signal_prev"] = macd_df[signal_cols[0]].shift(1)

        # EMA
        ind["ema9"] = ta.ema(df["close"], length=9)
        ind["ema21"] = ta.ema(df["close"], length=21)

        # Bollinger Bands
        bb = ta.bbands(df["close"], length=20, std=2.0)
        if bb is not None and not bb.empty:
            upper_col = [c for c in bb.columns if "BBU_" in c]
            lower_col = [c for c in bb.columns if "BBL_" in c]
            mid_col = [c for c in bb.columns if "BBM_" in c]
            if upper_col and lower_col and mid_col:
                ind["bb_upper"] = bb[upper_col[0]]
                ind["bb_lower"] = bb[lower_col[0]]
                ind["bb_middle"] = bb[mid_col[0]]
                mid = bb[mid_col[0]]
                ind["bb_width"] = (bb[upper_col[0]] - bb[lower_col[0]]) / mid.replace(0, float("nan"))

        # Volume
        ind["volume"] = df["volume"]
        ind["volume_avg20"] = df["volume"].rolling(20).mean()

        # Close
        ind["close"] = df["close"]

        # Daily trend: EMA50 on daily df, merged into signal df
        if self.daily_df is not None and len(self.daily_df) >= 50:
            daily_ema50 = ta.ema(self.daily_df["close"], length=50)
            daily_close = self.daily_df["close"]
            # Use merge_asof to align daily data with signal candles (no look-ahead)
            daily_merged = pd.DataFrame({
                "ema50_daily": daily_ema50,
                "close_daily": daily_close,
            })
            # Shift daily by 1 to avoid same-day look-ahead (use previous day's data)
            daily_merged = daily_merged.shift(1)
            # Resample/reindex: forward-fill daily values into signal timeframe
            combined = daily_merged.reindex(
                daily_merged.index.union(ind.index)
            ).ffill()
            ind["ema50_daily"] = combined["ema50_daily"].reindex(ind.index)
            ind["close_daily"] = combined["close_daily"].reindex(ind.index)

        self._ind = ind
        logger.debug("Indicators precomputed: {} rows, {} columns", len(ind), len(ind.columns))

    # ------------------------------------------------------------------ #
    # Signal evaluation                                                    #
    # ------------------------------------------------------------------ #

    def _check_buy(self, row: pd.Series) -> bool:
        """Check all BUY conditions for a single candle's indicator row.

        BUY conditions (updated — RSI < 45, EMA9>EMA21 optional):
            C1. RSI < threshold (45) and rising
            C2. MACD bullish crossover
            C3. EMA9 > EMA21 (only if config.use_ema_filter=True)
            C4. Price > lower BB after squeeze
            C5. Volume spike (1.5× avg)
            C6. Daily trend positive (close > EMA50 daily)
        """
        cfg = self.config

        def v(col: str) -> Optional[float]:
            val = row.get(col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            return float(val)

        rsi = v("rsi")
        rsi_prev = v("rsi_prev")
        macd = v("macd")
        macd_prev = v("macd_prev")
        macd_sig = v("macd_signal")
        macd_sig_prev = v("macd_signal_prev")
        ema9 = v("ema9")
        ema21 = v("ema21")
        bb_lower = v("bb_lower")
        bb_width = v("bb_width")
        close = v("close")
        volume = v("volume")
        vol_avg = v("volume_avg20")
        ema50_d = v("ema50_daily")
        close_d = v("close_daily")

        # 1. RSI < threshold (45) and rising
        if rsi is None or rsi_prev is None:
            return False
        if not (rsi < cfg.rsi_threshold and rsi > rsi_prev):
            return False

        # 2. MACD bullish crossover
        if None in (macd, macd_prev, macd_sig, macd_sig_prev):
            return False
        if not (macd_prev <= macd_sig_prev and macd > macd_sig):  # type: ignore[operator]
            return False

        # 3. EMA9 > EMA21 (optional — only if use_ema_filter enabled)
        if cfg.use_ema_filter:
            if ema9 is None or ema21 is None or ema9 <= ema21:
                return False

        # 4. Price > lower BB after squeeze
        if bb_width is None or bb_lower is None or close is None:
            return False
        if not (bb_width < cfg.bb_squeeze_width and close > bb_lower):
            return False

        # 5. Volume spike
        if volume is None or vol_avg is None or vol_avg == 0:
            return False
        if volume <= cfg.volume_multiplier * vol_avg:
            return False

        # 6. Daily trend positive
        if ema50_d is None or close_d is None:
            return False
        if close_d <= ema50_d:
            return False

        logger.debug(
            "BUY signal: RSI={:.1f} (thr={}) MACD_cross BB_squeeze Vol_spike Daily_up close={}",
            rsi, cfg.rsi_threshold, close,
        )
        return True

    def _analyze_conditions(self) -> dict:
        """Count how many candles trigger each individual buy condition.

        Provides analysis for both approaches:
        - Approach A: RSI<45, no EMA filter (current strategy)
        - Approach B: RSI<45, with EMA9>EMA21 filter
        - Approach C: original RSI<35, with EMA9>EMA21 filter (old strategy)
        """
        ind = self._ind
        cfg = self.config

        def cnt(mask: "pd.Series") -> int:
            return int(mask.sum())

        # Individual conditions
        c1_45 = (ind.get("rsi", pd.Series(dtype=float)) < 45) & \
                (ind.get("rsi", pd.Series(dtype=float)) > ind.get("rsi_prev", pd.Series(dtype=float)))
        c1_35 = (ind.get("rsi", pd.Series(dtype=float)) < 35) & \
                (ind.get("rsi", pd.Series(dtype=float)) > ind.get("rsi_prev", pd.Series(dtype=float)))
        c2 = (ind.get("macd_prev", pd.Series(dtype=float)) <= ind.get("macd_signal_prev", pd.Series(dtype=float))) & \
             (ind.get("macd", pd.Series(dtype=float)) > ind.get("macd_signal", pd.Series(dtype=float)))
        c3 = ind.get("ema9", pd.Series(dtype=float)) > ind.get("ema21", pd.Series(dtype=float))
        c4 = (ind.get("bb_width", pd.Series(dtype=float)) < cfg.bb_squeeze_width) & \
             (ind.get("close", pd.Series(dtype=float)) > ind.get("bb_lower", pd.Series(dtype=float)))
        c5 = ind.get("volume", pd.Series(dtype=float)) > cfg.volume_multiplier * ind.get("volume_avg20", pd.Series(dtype=float))
        c6 = ind.get("close_daily", pd.Series(dtype=float)) > ind.get("ema50_daily", pd.Series(dtype=float))

        total = len(ind)
        return {
            "total_candles": total,
            # Individual
            "c1_rsi45_rising": cnt(c1_45),
            "c1_rsi35_rising": cnt(c1_35),
            "c2_macd_crossover": cnt(c2),
            "c3_ema9_above_ema21": cnt(c3),
            "c4_bb_squeeze_above_lower": cnt(c4),
            "c5_volume_spike": cnt(c5),
            "c6_daily_trend_positive": cnt(c6),
            # Approach A: RSI<45, NO EMA filter (current — more signals)
            "approach_A_rsi45_no_ema": cnt(c1_45 & c2 & c4 & c5 & c6),
            # Approach B: RSI<45, WITH EMA9>EMA21 filter (more restrictive)
            "approach_B_rsi45_with_ema": cnt(c1_45 & c2 & c3 & c4 & c5 & c6),
            # Approach C: Original RSI<35 + EMA9>EMA21 (old strategy — baseline)
            "approach_C_rsi35_with_ema": cnt(c1_35 & c2 & c3 & c4 & c5 & c6),
            # Used in actual backtest
            "active_approach": "A (RSI<45, no EMA filter)" if not cfg.use_ema_filter else "B (RSI<45, with EMA filter)",
        }

    def _check_sell(
        self,
        pos: Position,
        current_price: float,
        now: datetime,
        tf_minutes: int,
    ) -> tuple[bool, str]:
        """Check all SELL conditions for an open position.

        Sell rules (per-pool params):
            1. Hard stop-loss: price dropped stop_loss_pct% from entry
            2. Trailing stop: price dropped trailing_stop_pct% from peak_price
            3. Emergency exit: position stuck >90min with <1% move
        """
        cfg = self.config
        entry = pos.entry_price
        peak = pos.peak_price
        pnl = (current_price - entry) / entry
        peak_pnl = (peak - entry) / entry
        elapsed_min = (now - pos.entry_time).total_seconds() / 60

        # 1. Hard stop-loss from entry
        if pnl <= -cfg.stop_loss_pct:
            return True, f"Stop-loss {pnl:.2%} (SL={cfg.stop_loss_pct:.1%})"

        # 2. Trailing stop from peak
        # When price drops trailing_stop_pct% below peak → close
        trail_trigger = peak * (1 - cfg.trailing_stop_pct)
        if current_price <= trail_trigger:
            return True, (
                f"Trailing stop {pnl:.2%} "
                f"(peak={peak:.4f}, trigger={trail_trigger:.4f}, trail={cfg.trailing_stop_pct:.1%})"
            )

        # 3. Emergency exit: 90 min without ±1% move
        if elapsed_min >= cfg.emergency_minutes and abs(pnl) <= cfg.emergency_threshold_pct:
            return True, f"Emergency exit: {elapsed_min:.0f}min, PnL={pnl:.2%}"

        return False, f"Hold (PnL={pnl:.2%})"
