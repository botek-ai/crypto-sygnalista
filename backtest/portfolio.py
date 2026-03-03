"""Portfolio tracking — positions, P&L, statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger


@dataclass
class Position:
    """An open position in the portfolio."""

    symbol: str
    entry_price: float
    entry_time: datetime
    size_usdc: float          # capital allocated (entry cost)
    qty: float                # coins purchased
    peak_price: float = 0.0  # highest close seen since entry
    trailing_be_active: bool = False    # break-even trailing activated
    trailing2_active: bool = False      # 1.5%-from-peak trailing activated

    def __post_init__(self) -> None:
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price

    @property
    def current_value(self) -> float:
        """Current USDC value at peak price (not real-time)."""
        return self.qty * self.peak_price


@dataclass
class Trade:
    """A closed trade record."""

    symbol: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    size_usdc: float    # initial cost
    qty: float
    pnl_usdc: float
    pnl_pct: float
    exit_reason: str

    @property
    def is_winner(self) -> bool:
        return self.pnl_usdc > 0


class Portfolio:
    """Tracks USDC cash, open positions, and trade history.

    Args:
        initial_capital: Starting USDC balance.
        max_positions: Maximum number of simultaneously open positions.
        position_size_pct: Fraction of free capital per trade.
        min_position_usdc: Minimum position size in USDC.
        max_coin_exposure_pct: Max fraction of total capital per coin.
        cooldown_minutes: Cooldown between trades per symbol.
    """

    def __init__(
        self,
        initial_capital: float = 1000.0,
        max_positions: int = 8,
        position_size_pct: float = 0.20,
        min_position_usdc: float = 15.0,
        max_coin_exposure_pct: float = 0.15,
        cooldown_minutes: int = 12,
    ) -> None:
        self.initial_capital = initial_capital
        self.cash: float = initial_capital
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.min_position_usdc = min_position_usdc
        self.max_coin_exposure_pct = max_coin_exposure_pct
        self.cooldown_minutes = cooldown_minutes

        self.positions: dict[str, list[Position]] = {}   # symbol → list of positions
        self.trades: list[Trade] = []
        self._last_trade_time: dict[str, datetime] = {}  # cooldown tracking
        self._equity_curve: list[tuple[datetime, float]] = []

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    @property
    def open_count(self) -> int:
        return sum(len(ps) for ps in self.positions.values())

    @property
    def invested_usdc(self) -> float:
        return sum(p.size_usdc for ps in self.positions.values() for p in ps)

    @property
    def total_equity(self) -> float:
        """Cash + current market value of all open positions (at peak — approximate)."""
        market_value = sum(
            p.qty * p.peak_price for ps in self.positions.values() for p in ps
        )
        return self.cash + market_value

    def coin_exposure_usdc(self, symbol: str) -> float:
        return sum(p.size_usdc for p in self.positions.get(symbol, []))

    def can_buy(self, symbol: str, now: datetime) -> tuple[bool, str]:
        """Check all filters before opening a position."""
        # Cooldown
        if symbol in self._last_trade_time:
            elapsed = (now - self._last_trade_time[symbol]).total_seconds() / 60
            if elapsed < self.cooldown_minutes:
                return False, f"Cooldown {elapsed:.1f}/{self.cooldown_minutes}min"

        # Max positions
        if self.open_count >= self.max_positions:
            return False, f"Max positions ({self.max_positions})"

        # Position size check
        size = self.cash * self.position_size_pct
        if size < self.min_position_usdc:
            return False, f"Position too small ({size:.2f} < {self.min_position_usdc} USDC)"

        # Coin exposure
        exposure = self.coin_exposure_usdc(symbol) + size
        total = self.initial_capital  # use initial for stable % calc
        if exposure / total > self.max_coin_exposure_pct:
            return False, f"Max exposure for {symbol} ({exposure/total:.1%})"

        return True, "OK"

    # ------------------------------------------------------------------ #
    # Mutations                                                            #
    # ------------------------------------------------------------------ #

    def open_position(self, symbol: str, price: float, now: datetime) -> Optional[Position]:
        """Open a new position. Returns Position if opened, None if blocked."""
        ok, reason = self.can_buy(symbol, now)
        if not ok:
            logger.debug("Cannot buy {}: {}", symbol, reason)
            return None

        size = self.cash * self.position_size_pct
        size = max(size, self.min_position_usdc)
        qty = size / price
        self.cash -= size

        pos = Position(
            symbol=symbol,
            entry_price=price,
            entry_time=now,
            size_usdc=size,
            qty=qty,
            peak_price=price,
        )

        if symbol not in self.positions:
            self.positions[symbol] = []
        self.positions[symbol].append(pos)

        logger.debug(
            "BUY {} @ {:.6f} | size={:.2f} USDC | qty={:.6f} | cash left={:.2f}",
            symbol, price, size, qty, self.cash,
        )
        return pos

    def close_position(
        self, pos: Position, exit_price: float, exit_time: datetime, reason: str
    ) -> Trade:
        """Close an open position and record the trade."""
        pnl_usdc = pos.qty * (exit_price - pos.entry_price)
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        proceeds = pos.qty * exit_price
        self.cash += proceeds

        trade = Trade(
            symbol=pos.symbol,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            size_usdc=pos.size_usdc,
            qty=pos.qty,
            pnl_usdc=pnl_usdc,
            pnl_pct=pnl_pct,
            exit_reason=reason,
        )
        self.trades.append(trade)

        # Remove from open positions
        symbol = pos.symbol
        if symbol in self.positions:
            self.positions[symbol] = [p for p in self.positions[symbol] if p is not pos]
            if not self.positions[symbol]:
                del self.positions[symbol]

        self._last_trade_time[symbol] = exit_time

        logger.debug(
            "SELL {} @ {:.6f} | PnL={:+.2%} ({:+.2f} USDC) | reason={} | cash={:.2f}",
            symbol, exit_price, pnl_pct, pnl_usdc, reason, self.cash,
        )
        return trade

    def record_equity(self, now: datetime, price_map: dict[str, float]) -> None:
        """Snapshot equity curve at a given timestamp."""
        market_val = 0.0
        for sym, pos_list in self.positions.items():
            price = price_map.get(sym, 0.0)
            market_val += sum(p.qty * price for p in pos_list)
        equity = self.cash + market_val
        self._equity_curve.append((now, equity))

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    def get_pnl(self) -> float:
        """Total realized P&L in USDC."""
        return sum(t.pnl_usdc for t in self.trades)

    def get_stats(self) -> dict:
        """Compute comprehensive backtest statistics."""
        trades = self.trades
        final_cap = self.cash + self.invested_usdc
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "total_pnl_usdc": 0.0,
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "avg_trade_duration_h": 0.0,
                "best_trade_pct": 0.0,
                "worst_trade_pct": 0.0,
                "final_capital": final_cap,
            }

        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]

        win_rate = len(winners) / len(trades)
        avg_win = sum(t.pnl_pct for t in winners) / len(winners) if winners else 0.0
        avg_loss = sum(t.pnl_pct for t in losers) / len(losers) if losers else 0.0

        total_pnl = self.get_pnl()
        total_return = total_pnl / self.initial_capital

        # Max drawdown from equity curve
        max_dd = self._calc_max_drawdown()

        # Sharpe ratio (annualized, using trade returns)
        sharpe = self._calc_sharpe(trades)

        # Profit factor
        gross_profit = sum(t.pnl_usdc for t in winners)
        gross_loss = abs(sum(t.pnl_usdc for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Average trade duration
        durations = [
            (t.exit_time - t.entry_time).total_seconds() / 3600
            for t in trades
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "total_trades": len(trades),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_pnl_usdc": total_pnl,
            "total_return_pct": total_return,
            "max_drawdown_pct": max_dd,
            "sharpe_ratio": sharpe,
            "profit_factor": profit_factor,
            "avg_trade_duration_h": avg_duration,
            "best_trade_pct": max((t.pnl_pct for t in trades), default=0.0),
            "worst_trade_pct": min((t.pnl_pct for t in trades), default=0.0),
            "final_capital": self.cash + self.invested_usdc,
        }

    def _calc_max_drawdown(self) -> float:
        """Calculate max drawdown from equity curve."""
        if not self._equity_curve:
            return 0.0
        equities = [e for _, e in self._equity_curve]
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    def _calc_sharpe(self, trades: list[Trade], risk_free: float = 0.0) -> float:
        """Annualized Sharpe ratio based on per-trade returns."""
        if len(trades) < 2:
            return 0.0
        returns = [t.pnl_pct for t in trades]
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(variance) if variance > 0 else 0.0
        if std_r == 0:
            return 0.0
        # Approximate annualization: assume ~252 trades/year
        trades_per_year = 252
        return (mean_r - risk_free) / std_r * math.sqrt(trades_per_year)
