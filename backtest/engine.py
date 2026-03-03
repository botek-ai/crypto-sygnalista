"""Backtest engine — iterates over historical OHLCV candles without look-ahead bias."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
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
        "trailing_stop": 0.013,   # 1.3% drop from peak triggers close
        "stop_loss": 0.025,       # 2.5% hard SL from entry
        "trailing_activation": 0.02,  # activate trailing after +2% gain
    },
    "mid": {
        "symbols": [
            "LINK/USDT", "SOL/USDT", "AVAX/USDT", "AAVE/USDT", "ATOM/USDT",
            "NEAR/USDT", "OP/USDT", "FIL/USDT", "ICP/USDT",
        ],
        "trailing_stop": 0.020,   # 2.0% drop from peak
        "stop_loss": 0.025,
        "trailing_activation": 0.02,
    },
    "low": {
        "symbols": ["BNB/USDT", "BTC/USDT", "ETH/USDT"],
        "trailing_stop": 0.009,   # 0.9% drop from peak
        "stop_loss": 0.025,
        "trailing_activation": 0.02,
    },
}

ALL_SYMBOLS = [s for pool in POOLS.values() for s in pool["symbols"]]


def get_pool_for_symbol(symbol: str) -> tuple[str, dict]:
    for pool_name, pool in POOLS.items():
        if symbol in pool["symbols"]:
            return pool_name, pool
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
    trailing_stop_pct: float = 0.020     # % drop from peak (overridden per-pool)
    trailing_activation_pct: float = 0.02  # activate after this gain
    stop_loss_pct: float = 0.025          # hard stop-loss from entry
    emergency_minutes: int = 96           # 4h x 24 candles = 4 days
    emergency_threshold_pct: float = 0.01
    # V2 scoring strategy
    min_score: int = 3   # minimum conditions out of 4 to trigger BUY
    # Filters (blocking)
    use_f1_ema_filter: bool = True    # F1: EMA9 > EMA21 on 4h
    use_f2_daily_filter: bool = False  # F2: close_daily > EMA21_daily (disabled by default)
    # Individual thresholds
    rsi_threshold: float = 45.0
    bb_pctb_threshold: float = 0.3
    volume_multiplier: float = 1.5
    # Legacy compat
    rsi_threshold_legacy: float = 45.0
    use_ema_filter: bool = False
    bb_squeeze_width: float = 0.03


def _timeframe_minutes(tf: str) -> int:
    units = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    for suffix, mult in units.items():
        if tf.endswith(suffix):
            try:
                return int(tf[:-1]) * mult
            except ValueError:
                pass
    return 60


class BacktestEngine:
    """Candle-by-candle backtest engine with v2 scoring strategy.

    V2 BUY strategy — score >= 3/4 conditions:
        C1: RSI < 45
        C2: BB%B < 0.3 OR RSI bullish divergence
        C3: StochRSI K<20 crossup OR MACD bullish crossover
        C4: Volume > 1.5× avg20

    Blocking filters:
        F1: EMA9 > EMA21 (4h) — required
        F2: close_daily > EMA21_daily — optional (disabled by default)

    Trailing stop (per pool):
        - Activates after +2% gain from entry
        - Tracks peak price; closes when price drops trailing_stop_pct% below peak
    """

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

        self._ind: pd.DataFrame = pd.DataFrame()

    def run(self) -> dict:
        logger.info(
            "Starting backtest: {} | {} | {} candles | capital={:.0f} | "
            "trailing={:.1%} (act@{:.0%}) | sl={:.1%} | min_score={}/4 | f1={} f2={}",
            self.config.symbol,
            self.config.timeframe,
            len(self.df),
            self.config.initial_capital,
            self.config.trailing_stop_pct,
            self.config.trailing_activation_pct,
            self.config.stop_loss_pct,
            self.config.min_score,
            self.config.use_f1_ema_filter,
            self.config.use_f2_daily_filter,
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

            # Update peak prices
            for pos in self.portfolio.positions.get(symbol, []):
                if high > pos.peak_price:
                    pos.peak_price = high

            # Check sell conditions
            positions_snapshot = list(self.portfolio.positions.get(symbol, []))
            for pos in positions_snapshot:
                sell, reason = self._check_sell(pos, close, now, tf_min)
                if sell:
                    self.portfolio.close_position(pos, close, now, reason)

            # Record equity
            self.portfolio.record_equity(now, {symbol: close})

            # Check buy conditions
            score, triggered_conditions = self._score_buy(ind_row)
            if score >= self.config.min_score:
                logger.debug(
                    "BUY signal: {} score={}/4 conditions={} close={}",
                    symbol, score, triggered_conditions, close
                )
                self.portfolio.open_position(symbol, close, now)

        # Force-close remaining
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
        stats["use_ema_filter"] = self.config.use_f1_ema_filter
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

    def _precompute_indicators(self) -> None:
        df = self.df
        ind = pd.DataFrame(index=df.index)

        # RSI
        rsi = ta.rsi(df["close"], length=14)
        if rsi is not None:
            ind["rsi"] = rsi
            ind["rsi_prev"] = rsi.shift(1)

        # StochRSI
        stochrsi_df = ta.stochrsi(df["close"], length=14, rsi_length=14, k=3, d=3)
        if stochrsi_df is not None and not stochrsi_df.empty:
            k_cols = [c for c in stochrsi_df.columns if "STOCHRSIk_" in c]
            d_cols = [c for c in stochrsi_df.columns if "STOCHRSId_" in c]
            if k_cols:
                ind["stochrsi_k"] = stochrsi_df[k_cols[0]]
                ind["stochrsi_k_prev"] = stochrsi_df[k_cols[0]].shift(1)
            if d_cols:
                ind["stochrsi_d"] = stochrsi_df[d_cols[0]]

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
            pctb_col = [c for c in bb.columns if "BBP_" in c]
            if upper_col and lower_col and mid_col:
                ind["bb_upper"] = bb[upper_col[0]]
                ind["bb_lower"] = bb[lower_col[0]]
                ind["bb_middle"] = bb[mid_col[0]]
                mid = bb[mid_col[0]]
                ind["bb_width"] = (bb[upper_col[0]] - bb[lower_col[0]]) / mid.replace(0, float("nan"))
            if pctb_col:
                ind["bb_pct_b"] = bb[pctb_col[0]]

        # RSI bullish divergence (rolling window 10)
        if rsi is not None:
            div_arr = np.zeros(len(df), dtype=bool)
            win = 10
            close_arr = df["close"].values
            rsi_arr = rsi.values
            for i in range(win, len(df)):
                if np.any(np.isnan(rsi_arr[i - win: i + 1])):
                    continue
                p_slice = close_arr[i - win: i]
                r_slice = rsi_arr[i - win: i]
                # price lower low: current < all in window
                # rsi higher low: current > min of window
                if close_arr[i] < np.min(p_slice) and rsi_arr[i] > np.min(r_slice):
                    div_arr[i] = True
            ind["rsi_divergence"] = div_arr

        # Volume
        ind["volume"] = df["volume"]
        ind["volume_avg20"] = df["volume"].rolling(20).mean()

        # Close
        ind["close"] = df["close"]

        # Daily data
        if self.daily_df is not None and len(self.daily_df) >= 21:
            daily_ema21 = ta.ema(self.daily_df["close"], length=21)
            daily_close = self.daily_df["close"]
            daily_merged = pd.DataFrame({
                "ema21_daily": daily_ema21,
                "close_daily": daily_close,
            })
            if len(self.daily_df) >= 50:
                daily_ema50 = ta.ema(self.daily_df["close"], length=50)
                daily_merged["ema50_daily"] = daily_ema50
            # shift by 1 to avoid look-ahead
            daily_merged = daily_merged.shift(1)
            combined = daily_merged.reindex(
                daily_merged.index.union(ind.index)
            ).ffill()
            ind["ema21_daily"] = combined["ema21_daily"].reindex(ind.index)
            ind["close_daily"] = combined["close_daily"].reindex(ind.index)
            if "ema50_daily" in combined.columns:
                ind["ema50_daily"] = combined["ema50_daily"].reindex(ind.index)

        self._ind = ind
        logger.debug("Indicators precomputed: {} rows, {} columns", len(ind), len(ind.columns))

    def _score_buy(self, row: pd.Series) -> tuple[int, list[str]]:
        """Score v2 buy conditions. Returns (score, triggered_conditions).

        V2 conditions (score >= min_score to buy):
            C1: RSI < 45
            C2: BB%B < 0.3 OR RSI bullish divergence
            C3: StochRSI K<20+crossup OR MACD bullish crossover
            C4: Volume > 1.5× avg20

        Blocking filters:
            F1: EMA9 > EMA21 (if use_f1_ema_filter)
            F2: close_daily > EMA21_daily (if use_f2_daily_filter)
        """
        cfg = self.config

        def v(col: str) -> Optional[float]:
            val = row.get(col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            return float(val)

        def b(col: str) -> bool:
            val = row.get(col)
            if val is None:
                return False
            if isinstance(val, (bool, np.bool_)):
                return bool(val)
            if isinstance(val, float) and pd.isna(val):
                return False
            return bool(val)

        # --- Blocking filters ---
        if cfg.use_f1_ema_filter:
            ema9 = v("ema9")
            ema21 = v("ema21")
            if ema9 is None or ema21 is None or ema9 <= ema21:
                return 0, []

        if cfg.use_f2_daily_filter:
            close_d = v("close_daily")
            ema21_d = v("ema21_daily")
            if close_d is None or ema21_d is None or close_d <= ema21_d:
                return 0, []

        # --- Score conditions ---
        score = 0
        triggered: list[str] = []

        # C1: RSI < 45
        rsi = v("rsi")
        if rsi is not None and rsi < cfg.rsi_threshold:
            score += 1
            triggered.append(f"C1:RSI={rsi:.1f}")

        # C2: BB%B < 0.3 OR RSI divergence
        bb_pctb = v("bb_pct_b")
        rsi_div = b("rsi_divergence")
        if (bb_pctb is not None and bb_pctb < cfg.bb_pctb_threshold) or rsi_div:
            score += 1
            detail = []
            if bb_pctb is not None and bb_pctb < cfg.bb_pctb_threshold:
                detail.append(f"BBP={bb_pctb:.2f}")
            if rsi_div:
                detail.append("RSI_div")
            triggered.append(f"C2:{'+'.join(detail)}")

        # C3: StochRSI K<20+crossup OR MACD bullish crossover
        stochrsi_k = v("stochrsi_k")
        stochrsi_k_prev = v("stochrsi_k_prev")
        stochrsi_d = v("stochrsi_d")
        macd = v("macd")
        macd_prev = v("macd_prev")
        macd_sig = v("macd_signal")
        macd_sig_prev = v("macd_signal_prev")

        stoch_cross = (
            stochrsi_k is not None
            and stochrsi_k_prev is not None
            and stochrsi_d is not None
            and stochrsi_k_prev < 20
            and stochrsi_k > stochrsi_d
        )
        macd_cross = (
            None not in (macd, macd_prev, macd_sig, macd_sig_prev)
            and macd_prev <= macd_sig_prev  # type: ignore[operator]
            and macd > macd_sig  # type: ignore[operator]
        )
        if stoch_cross or macd_cross:
            score += 1
            detail = []
            if stoch_cross:
                detail.append(f"StochK={stochrsi_k:.1f}")
            if macd_cross:
                detail.append("MACD_x")
            triggered.append(f"C3:{'+'.join(detail)}")

        # C4: Volume > 1.5× avg20
        volume = v("volume")
        vol_avg = v("volume_avg20")
        if volume is not None and vol_avg is not None and vol_avg > 0 and volume > cfg.volume_multiplier * vol_avg:
            score += 1
            triggered.append(f"C4:Vol={volume/vol_avg:.1f}x")

        return score, triggered

    def _analyze_conditions(self) -> dict:
        """Count candles triggering each individual condition for analysis."""
        ind = self._ind
        cfg = self.config

        def cnt(mask: pd.Series) -> int:
            return int(mask.fillna(False).sum())

        rsi = ind.get("rsi", pd.Series(dtype=float))
        bb_pctb = ind.get("bb_pct_b", pd.Series(dtype=float))
        rsi_div = ind.get("rsi_divergence", pd.Series(dtype=bool)).fillna(False)
        stochrsi_k = ind.get("stochrsi_k", pd.Series(dtype=float))
        stochrsi_k_prev = ind.get("stochrsi_k_prev", pd.Series(dtype=float))
        stochrsi_d = ind.get("stochrsi_d", pd.Series(dtype=float))
        macd = ind.get("macd", pd.Series(dtype=float))
        macd_prev = ind.get("macd_prev", pd.Series(dtype=float))
        macd_sig = ind.get("macd_signal", pd.Series(dtype=float))
        macd_sig_prev = ind.get("macd_signal_prev", pd.Series(dtype=float))
        volume = ind.get("volume", pd.Series(dtype=float))
        vol_avg = ind.get("volume_avg20", pd.Series(dtype=float))
        ema9 = ind.get("ema9", pd.Series(dtype=float))
        ema21 = ind.get("ema21", pd.Series(dtype=float))
        close_d = ind.get("close_daily", pd.Series(dtype=float))
        ema21_d = ind.get("ema21_daily", pd.Series(dtype=float))

        c1 = rsi < cfg.rsi_threshold
        c2 = (bb_pctb < cfg.bb_pctb_threshold) | rsi_div
        c3_stoch = (stochrsi_k_prev < 20) & (stochrsi_k > stochrsi_d)
        c3_macd = (macd_prev <= macd_sig_prev) & (macd > macd_sig)
        c3 = c3_stoch | c3_macd
        c4 = volume > cfg.volume_multiplier * vol_avg
        f1 = ema9 > ema21
        f2 = close_d > ema21_d

        # with F1 filter
        base = f1 if cfg.use_f1_ema_filter else pd.Series(True, index=ind.index)
        if cfg.use_f2_daily_filter:
            base = base & f2

        score_ge3 = cnt(base & (c1.astype(int) + c2.astype(int) + c3.astype(int) + c4.astype(int) >= 3))
        score_ge2 = cnt(base & (c1.astype(int) + c2.astype(int) + c3.astype(int) + c4.astype(int) >= 2))

        return {
            "total_candles": len(ind),
            "c1_rsi_lt45": cnt(c1),
            "c2_bbpctb_or_div": cnt(c2),
            "c3_stochrsi_or_macd": cnt(c3),
            "c3_stochrsi_cross": cnt(c3_stoch),
            "c3_macd_cross": cnt(c3_macd),
            "c4_volume_spike": cnt(c4),
            "f1_ema9_gt_ema21": cnt(f1),
            "f2_daily_above_ema21": cnt(f2),
            "score_ge3_with_filters": score_ge3,
            "score_ge2_with_filters": score_ge2,
            # Legacy for RSI report
            "approach_A_rsi45_no_ema": cnt(c1),
            "approach_B_rsi45_with_ema": cnt(c1 & f1),
            "approach_C_rsi35_with_ema": cnt((rsi < 35) & f1),
        }

    def _check_sell(
        self,
        pos: Position,
        current_price: float,
        now: datetime,
        tf_minutes: int,
    ) -> tuple[bool, str]:
        cfg = self.config
        entry = pos.entry_price
        peak = pos.peak_price
        pnl = (current_price - entry) / entry
        peak_pnl = (peak - entry) / entry
        elapsed_min = (now - pos.entry_time).total_seconds() / 60

        # Hard stop-loss
        if pnl <= -cfg.stop_loss_pct:
            return True, f"Stop-loss {pnl:.2%} (SL={cfg.stop_loss_pct:.1%})"

        # Trailing stop — activates after trailing_activation_pct gain
        if peak_pnl >= cfg.trailing_activation_pct:
            trail_trigger = peak * (1 - cfg.trailing_stop_pct)
            if current_price <= trail_trigger:
                return True, (
                    f"Trailing stop {pnl:.2%} "
                    f"(peak={peak:.4f}, trigger={trail_trigger:.4f}, trail={cfg.trailing_stop_pct:.1%})"
                )

        # Emergency exit
        if elapsed_min >= cfg.emergency_minutes and abs(pnl) <= cfg.emergency_threshold_pct:
            return True, f"Emergency exit: {elapsed_min:.0f}min, PnL={pnl:.2%}"

        return False, f"Hold (PnL={pnl:.2%})"
