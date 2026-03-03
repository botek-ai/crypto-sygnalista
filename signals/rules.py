"""Signal rules — buy/sell conditions."""

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
    indicators: IndicatorResult

    model_config = {"arbitrary_types_allowed": True}

    def __str__(self) -> str:
        """Human-readable signal summary."""
        return (
            f"[{self.signal}] {self.symbol} @ {self.price:.4f} "
            f"({self.timestamp.strftime('%Y-%m-%d %H:%M')}) — {self.reason}"
        )


class BuyRules:
    """Buy signal rules.

    BUY when ALL conditions are met:
    1. RSI(14) < 35 and rising
    2. MACD bullish crossover
    3. EMA9 > EMA21
    4. Price above lower BB after squeeze (BB width < 0.03)
    5. Volume > 1.5× 20-candle average
    6. Daily trend positive (close > EMA50 on 1d)
    """

    def check(self, ind: IndicatorResult, rsi_threshold: float = 35.0) -> tuple[bool, str]:
        """Evaluate all buy conditions.

        Args:
            ind: Computed indicator values.
            rsi_threshold: RSI level below which buy is considered.

        Returns:
            Tuple of (triggered: bool, reason: str describing result).
        """
        reasons: list[str] = []
        failures: list[str] = []

        # 1. RSI < threshold and rising
        if ind.rsi is not None and ind.rsi < rsi_threshold and ind.rsi_rising:
            reasons.append(f"RSI={ind.rsi:.1f} (<{rsi_threshold}, rising)")
        else:
            failures.append(
                f"RSI fail (rsi={ind.rsi}, rising={ind.rsi_rising})"
            )

        # 2. MACD bullish crossover
        if ind.macd_bullish_crossover:
            reasons.append("MACD bullish crossover")
        else:
            failures.append(
                f"MACD no crossover (macd={ind.macd:.6f if ind.macd else None}, "
                f"signal={ind.macd_signal:.6f if ind.macd_signal else None})"
            )

        # 3. EMA9 > EMA21
        if ind.ema9 is not None and ind.ema21 is not None and ind.ema9 > ind.ema21:
            reasons.append(f"EMA9({ind.ema9:.4f})>EMA21({ind.ema21:.4f})")
        else:
            failures.append(f"EMA fail (ema9={ind.ema9}, ema21={ind.ema21})")

        # 4. Price above lower BB after squeeze
        if (
            ind.bb_squeeze
            and ind.close is not None
            and ind.bb_lower is not None
            and ind.close > ind.bb_lower
        ):
            reasons.append(
                f"Above BBL after squeeze (width={ind.bb_width:.4f})"
            )
        else:
            failures.append(
                f"BB fail (squeeze={ind.bb_squeeze}, "
                f"close={ind.close}, bbl={ind.bb_lower}, width={ind.bb_width})"
            )

        # 5. Volume spike
        if ind.volume_spike:
            reasons.append(
                f"Vol spike ({ind.volume:.0f} > 1.5×avg {ind.volume_avg20:.0f})"
            )
        else:
            failures.append(
                f"Vol fail (vol={ind.volume}, avg={ind.volume_avg20})"
            )

        # 6. Daily trend positive
        if ind.daily_trend_positive:
            reasons.append(
                f"Daily trend+ (close={ind.close_daily:.4f} > EMA50={ind.ema50_daily:.4f})"
            )
        else:
            failures.append(
                f"Daily trend fail (close={ind.close_daily}, ema50={ind.ema50_daily})"
            )

        all_triggered = len(failures) == 0
        detail = "; ".join(reasons) if all_triggered else "BLOCKED: " + "; ".join(failures)
        return all_triggered, detail


class SellRules:
    """Sell signal rules.

    SELL when ANY of these conditions are met:
    - Stop-loss: -3%
    - Take-profit: +150%
    - Trailing stop: break-even after +2%, 1.5% from peak after +3.5%
    - Emergency exit: after 90 min if ±1%
    - RSI > 70 and falling + MACD bearish crossover
    """

    def check(
        self,
        ind: IndicatorResult,
        entry_price: float,
        peak_price: float,
        entry_time: datetime,
        rsi_threshold: float = 70.0,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 1.50,
        trailing_activation_pct: float = 0.02,
        trailing2_activation_pct: float = 0.035,
        trailing2_distance_pct: float = 0.015,
        emergency_minutes: int = 90,
        emergency_threshold_pct: float = 0.01,
    ) -> tuple[bool, str]:
        """Evaluate all sell conditions.

        Args:
            ind: Current indicator values.
            entry_price: Position entry price.
            peak_price: Highest price seen since entry.
            entry_time: Position entry timestamp.
            rsi_threshold: RSI level above which sell is considered.
            stop_loss_pct: Stop-loss percentage (0.03 = 3%).
            take_profit_pct: Take-profit percentage (1.50 = 150%).
            trailing_activation_pct: Trailing stop 1 activation.
            trailing2_activation_pct: Trailing stop 2 activation.
            trailing2_distance_pct: Trailing stop 2 distance from peak.
            emergency_minutes: Emergency exit time limit.
            emergency_threshold_pct: Emergency exit price threshold.

        Returns:
            Tuple of (triggered: bool, reason: str).
        """
        if ind.close is None:
            return False, "No close price"

        current = ind.close
        pnl = (current - entry_price) / entry_price
        elapsed = (datetime.now(tz=timezone.utc) - entry_time).total_seconds() / 60

        # 1. Stop-loss
        if pnl <= -stop_loss_pct:
            return True, f"Stop-loss: {pnl:.2%} <= -{stop_loss_pct:.0%}"

        # 2. Take-profit
        if pnl >= take_profit_pct:
            return True, f"Take-profit: {pnl:.2%} >= +{take_profit_pct:.0%}"

        # 3. Trailing stop — break-even after +2%
        peak_pnl = (peak_price - entry_price) / entry_price
        if peak_pnl >= trailing_activation_pct and current <= entry_price:
            return True, f"Trailing stop BE: peak was +{peak_pnl:.2%}, back to entry"

        # 4. Trailing stop — 1.5% from peak after +3.5%
        if peak_pnl >= trailing2_activation_pct:
            trail_sl = peak_price * (1 - trailing2_distance_pct)
            if current <= trail_sl:
                return True, (
                    f"Trailing stop 1.5%: price {current:.4f} <= trail_sl {trail_sl:.4f} "
                    f"(peak={peak_price:.4f})"
                )

        # 5. Emergency exit: 90 min if ±1%
        if elapsed >= emergency_minutes and abs(pnl) <= emergency_threshold_pct:
            return True, f"Emergency exit: {elapsed:.0f}min elapsed, PnL={pnl:.2%}"

        # 6. RSI > threshold + falling + MACD bearish crossover
        rsi_falling = (
            ind.rsi is not None
            and ind.rsi_prev is not None
            and ind.rsi > rsi_threshold
            and ind.rsi < ind.rsi_prev
        )
        if rsi_falling and ind.macd_bearish_crossover:
            return True, (
                f"RSI overbought+falling ({ind.rsi:.1f}) + MACD bearish crossover"
            )

        return False, f"HOLD (PnL={pnl:.2%}, elapsed={elapsed:.0f}min)"
