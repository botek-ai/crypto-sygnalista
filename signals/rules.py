"""Signal rules — v2 scoring buy strategy + per-pool trailing stop exits.

BUY conditions (score >= min_score out of 4):
    C1: RSI(14) < 45
    C2: BB%B < 0.3 OR RSI bullish divergence
    C3: StochRSI K<20 crossup OR MACD bullish crossover
    C4: Volume > 1.5× avg20

Blocking filter (optional):
    F1: EMA9 > EMA21 (4h trend filter — disabled by default in live mode)
    F2: close_daily > EMA21_daily (disabled by default)

EXIT (per-pool trailing stop, applied on 4h candles in live mode):
    - SL: -2.5% from entry (hard stop)
    - TSL: activates at +pool_activation%, then closes at -trailing_stop_pct% from peak
    - Emergency: 4 days, PnL within ±1%
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from indicators.technical import IndicatorResult


class SignalType(str, Enum):
    """Signal type enum."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalEvent(BaseModel):
    """A signal event emitted by the rules engine."""

    symbol: str
    signal: SignalType
    price: float
    timestamp: datetime
    reason: str
    score: int = 0
    conditions: list[str] = []
    indicators: IndicatorResult

    model_config = {"arbitrary_types_allowed": True}

    def __str__(self) -> str:
        """Human-readable signal summary."""
        conds = ", ".join(self.conditions) if self.conditions else self.reason
        return (
            f"[{self.signal}] {self.symbol} @ {self.price:.4f} "
            f"score={self.score}/4 ({conds}) "
            f"({self.timestamp.strftime('%Y-%m-%d %H:%M')})"
        )


class BuyRules:
    """V2 scoring buy rules.

    BUY when score >= min_score (default 3/4) conditions are met,
    subject to optional F1/F2 blocking filters.
    """

    def check(
        self,
        ind: IndicatorResult,
        min_score: int = 3,
        rsi_threshold: float = 45.0,
        bb_pctb_threshold: float = 0.3,
        volume_multiplier: float = 1.5,
        use_f1_ema_filter: bool = False,
        use_f2_daily_filter: bool = False,
    ) -> tuple[bool, str, int, list[str]]:
        """Evaluate v2 scoring buy conditions.

        Returns:
            (triggered, reason_str, score, triggered_conditions)
        """
        # --- Blocking filters ---
        if use_f1_ema_filter:
            if ind.ema9 is None or ind.ema21 is None or ind.ema9 <= ind.ema21:
                return False, f"F1_BLOCKED: EMA9={ind.ema9} <= EMA21={ind.ema21}", 0, []

        if use_f2_daily_filter:
            if (
                ind.close_daily is None
                or ind.ema21_daily is None
                or ind.close_daily <= ind.ema21_daily
            ):
                return False, f"F2_BLOCKED: daily={ind.close_daily} <= EMA21d={ind.ema21_daily}", 0, []

        # --- Score conditions ---
        score = 0
        triggered: list[str] = []

        # C1: RSI < 45
        if ind.rsi is not None and ind.rsi < rsi_threshold:
            score += 1
            triggered.append(f"C1:RSI={ind.rsi:.1f}")

        # C2: BB%B < 0.3 OR RSI bullish divergence
        bb_ok = ind.bb_pct_b is not None and ind.bb_pct_b < bb_pctb_threshold
        div_ok = bool(ind.rsi_bullish_divergence)
        if bb_ok or div_ok:
            score += 1
            detail = []
            if bb_ok:
                detail.append(f"BBP={ind.bb_pct_b:.2f}")
            if div_ok:
                detail.append("RSI_div")
            triggered.append(f"C2:{'+'.join(detail)}")

        # C3: StochRSI K<20 crossup OR MACD bullish crossover
        if ind.stochrsi_k_cross_up or ind.macd_bullish_crossover:
            score += 1
            detail = []
            if ind.stochrsi_k_cross_up:
                detail.append(f"StochK={ind.stochrsi_k:.1f}")
            if ind.macd_bullish_crossover:
                detail.append("MACD_x")
            triggered.append(f"C3:{'+'.join(detail)}")

        # C4: Volume > multiplier × avg20
        if ind.volume_spike:
            score += 1
            ratio = ind.volume / ind.volume_avg20 if ind.volume_avg20 else 0
            triggered.append(f"C4:Vol={ratio:.1f}x")

        if score >= min_score:
            reason = f"score={score}/4 [{', '.join(triggered)}]"
            return True, reason, score, triggered

        reason = f"score={score}/4 (need {min_score}) [{', '.join(triggered)}]"
        return False, reason, score, triggered


class SellRules:
    """Per-pool trailing stop sell rules (mirrors backtest engine logic).

    SELL when:
    - SL: current price <= entry * (1 - stop_loss_pct)
    - TSL: peak gain >= trailing_activation_pct AND price <= peak * (1 - trailing_stop_pct)
    - Emergency: elapsed >= emergency_minutes AND |PnL| <= emergency_threshold_pct
    """

    def check(
        self,
        ind: IndicatorResult,
        entry_price: float,
        peak_price: float,
        entry_time: datetime,
        trailing_stop_pct: float = 0.020,
        trailing_activation_pct: float = 0.015,
        stop_loss_pct: float = 0.025,
        emergency_minutes: int = 5760,   # 96h = 4 days
        emergency_threshold_pct: float = 0.01,
        tsl_active: bool = False,
    ) -> tuple[bool, str]:
        """Evaluate exit conditions.

        Returns:
            (should_sell, reason)
        """
        if ind.close is None:
            return False, "No close price"

        current = ind.close
        pnl = (current - entry_price) / entry_price
        peak_pnl = (peak_price - entry_price) / entry_price
        elapsed_min = (datetime.now(tz=timezone.utc) - entry_time).total_seconds() / 60

        sl_price = entry_price * (1.0 - stop_loss_pct)

        # 1. Hard stop-loss
        if current <= sl_price:
            return True, f"SL: {pnl:.2%} (limit={-stop_loss_pct:.1%})"

        # 2. TSL: activates after trailing_activation_pct gain
        if peak_pnl >= trailing_activation_pct:
            trail_trigger = peak_price * (1.0 - trailing_stop_pct)
            if current <= trail_trigger:
                return True, (
                    f"TSL: {pnl:.2%} (peak={peak_price:.4f}, "
                    f"trigger={trail_trigger:.4f}, trail={trailing_stop_pct:.1%})"
                )

        # 3. Emergency exit
        if elapsed_min >= emergency_minutes and abs(pnl) <= emergency_threshold_pct:
            return True, f"Emergency: {elapsed_min:.0f}min, PnL={pnl:.2%}"

        return False, f"HOLD (PnL={pnl:.2%}, elapsed={elapsed_min:.0f}min)"
