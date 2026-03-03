"""Backtest report generation — text, CSV, and Notion-ready output."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from backtest.portfolio import Trade


class BacktestReport:
    """Generate and persist backtest results.

    Args:
        results: Dict returned by BacktestEngine.run().
        output_dir: Directory to write report files.
    """

    def __init__(
        self,
        results: dict[str, Any],
        output_dir: str | Path = "backtest_results",
    ) -> None:
        self.results = results
        self.stats: dict = results.get("stats", {})
        self.trades: list[Trade] = results.get("trades", [])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H-%M")

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def text_report(self) -> str:
        """Build a human-readable text report."""
        s = self.stats
        lines: list[str] = []
        add = lines.append

        add("=" * 60)
        add(f"BACKTEST REPORT — {s.get('symbol', '?')} {s.get('timeframe', '?')}")
        add("=" * 60)
        add(f"Period:       {s.get('period_start', '?')} → {s.get('period_end', '?')}")
        add(f"Candles:      {s.get('candles', 0)}")
        add("")
        add("── Capital ─────────────────────────────────")
        add(f"Initial:      {self.results.get('initial_capital', 1000.0):.2f} USDC")
        add(f"Final:        {s.get('final_capital', 0):.2f} USDC")
        add(f"Total P&L:    {s.get('total_pnl_usdc', 0):+.2f} USDC ({s.get('total_return_pct', 0):.2%})")
        add(f"Max Drawdown: {s.get('max_drawdown_pct', 0):.2%}")
        add("")
        add("── Trades ──────────────────────────────────")
        add(f"Total trades: {s.get('total_trades', 0)}")
        add(f"Winning:      {s.get('winning_trades', 0)}")
        add(f"Losing:       {s.get('losing_trades', 0)}")
        add(f"Win rate:     {s.get('win_rate', 0):.1%}")
        add(f"Avg win:      {s.get('avg_win', 0):.2%}")
        add(f"Avg loss:     {s.get('avg_loss', 0):.2%}")
        add(f"Best trade:   {s.get('best_trade_pct', 0):.2%}")
        add(f"Worst trade:  {s.get('worst_trade_pct', 0):.2%}")
        add(f"Profit factor:{s.get('profit_factor', 0):.2f}")
        add(f"Avg duration: {s.get('avg_trade_duration_h', 0):.1f}h")
        add("")
        add("── Risk ─────────────────────────────────────")
        add(f"Sharpe ratio: {s.get('sharpe_ratio', 0):.2f}")
        add("")

        # Condition analysis
        ca = s.get("condition_analysis", {})
        if ca:
            total = ca.get("total_candles", 1)
            add("── Condition analysis (buy signals) ─────────")
            add(f"C1 RSI<35 rising:         {ca.get('c1_rsi_oversold_rising', 0):>4} / {total} ({ca.get('c1_rsi_oversold_rising', 0)/total:.1%})")
            add(f"C2 MACD bullish crossover:{ca.get('c2_macd_crossover', 0):>4} / {total} ({ca.get('c2_macd_crossover', 0)/total:.1%})")
            add(f"C3 EMA9 > EMA21:          {ca.get('c3_ema9_above_ema21', 0):>4} / {total} ({ca.get('c3_ema9_above_ema21', 0)/total:.1%})")
            add(f"C4 BB squeeze + above BBL:{ca.get('c4_bb_squeeze_above_lower', 0):>4} / {total} ({ca.get('c4_bb_squeeze_above_lower', 0)/total:.1%})")
            add(f"C5 Volume spike (1.5x):   {ca.get('c5_volume_spike', 0):>4} / {total} ({ca.get('c5_volume_spike', 0)/total:.1%})")
            add(f"C6 Daily trend positive:  {ca.get('c6_daily_trend_positive', 0):>4} / {total} ({ca.get('c6_daily_trend_positive', 0)/total:.1%})")
            add(f"C1+C2:                    {ca.get('c1_and_c2', 0):>4}")
            add(f"C1+C2+C3:                 {ca.get('c1_and_c2_and_c3', 0):>4}")
            add(f"ALL conditions met:       {ca.get('all_conditions', 0):>4}")
            if ca.get("all_conditions", 0) == 0:
                add("")
                add("⚠  FINDING: Strategy fired 0 buy signals.")
                add("   C1 (RSI<35 oversold) and C3 (EMA9>EMA21 bullish)")
                add("   are structurally contradictory during downtrends.")
                add("   Consider relaxing RSI threshold or removing EMA filter.")
            add("")

        if self.trades:
            add("── Trade log ────────────────────────────────")
            add(f"{'#':>3}  {'Symbol':<12}  {'Entry':>10}  {'Exit':>10}  {'PnL':>8}  {'Reason'}")
            add("-" * 70)
            for i, t in enumerate(self.trades, 1):
                add(
                    f"{i:>3}  {t.symbol:<12}  {t.entry_price:>10.4f}  {t.exit_price:>10.4f}"
                    f"  {t.pnl_pct:>7.2%}  {t.exit_reason}"
                )

        add("=" * 60)
        return "\n".join(lines)

    def save_text(self) -> Path:
        """Save text report to file and return path."""
        symbol_safe = self.stats.get("symbol", "UNKNOWN").replace("/", "-")
        tf = self.stats.get("timeframe", "?")
        filename = f"{self._timestamp}_{symbol_safe}_{tf}.txt"
        path = self.output_dir / filename
        path.write_text(self.text_report(), encoding="utf-8")
        logger.info("Report saved: {}", path)
        return path

    def save_csv(self) -> Path:
        """Save trade-level CSV report and return path."""
        symbol_safe = self.stats.get("symbol", "UNKNOWN").replace("/", "-")
        tf = self.stats.get("timeframe", "?")
        filename = f"{self._timestamp}_{symbol_safe}_{tf}_trades.csv"
        path = self.output_dir / filename

        fieldnames = [
            "symbol", "entry_time", "exit_time", "entry_price", "exit_price",
            "size_usdc", "qty", "pnl_usdc", "pnl_pct", "exit_reason",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in self.trades:
                writer.writerow({
                    "symbol": t.symbol,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "entry_price": f"{t.entry_price:.8f}",
                    "exit_price": f"{t.exit_price:.8f}",
                    "size_usdc": f"{t.size_usdc:.4f}",
                    "qty": f"{t.qty:.8f}",
                    "pnl_usdc": f"{t.pnl_usdc:.4f}",
                    "pnl_pct": f"{t.pnl_pct:.4f}",
                    "exit_reason": t.exit_reason,
                })
        logger.info("CSV saved: {}", path)
        return path

    def notion_blocks(self) -> list[dict]:
        """Return Notion block objects representing the report.

        Suitable for appending to a Notion page via the Blocks API.
        """
        s = self.stats
        blocks: list[dict] = []

        def heading(text: str, level: int = 2) -> dict:
            htype = f"heading_{level}"
            return {
                "object": "block",
                "type": htype,
                htype: {"rich_text": [{"type": "text", "text": {"content": text}}]},
            }

        def para(text: str) -> dict:
            return {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]},
            }

        def bullet(text: str) -> dict:
            return {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": text}}]},
            }

        blocks.append(heading(f"📊 {s.get('symbol', '?')} {s.get('timeframe', '?')} Backtest", 2))
        blocks.append(para(f"Period: {s.get('period_start', '?')} → {s.get('period_end', '?')} | {s.get('candles', 0)} candles"))
        blocks.append(heading("Capital", 3))
        blocks.append(bullet(f"Total P&L: {s.get('total_pnl_usdc', 0):+.2f} USDC ({s.get('total_return_pct', 0):.2%})"))
        blocks.append(bullet(f"Final capital: {s.get('final_capital', 0):.2f} USDC"))
        blocks.append(bullet(f"Max Drawdown: {s.get('max_drawdown_pct', 0):.2%}"))
        blocks.append(heading("Trades", 3))
        blocks.append(bullet(f"Total: {s.get('total_trades', 0)} | Winners: {s.get('winning_trades', 0)} | Losers: {s.get('losing_trades', 0)}"))
        blocks.append(bullet(f"Win rate: {s.get('win_rate', 0):.1%} | Avg win: {s.get('avg_win', 0):.2%} | Avg loss: {s.get('avg_loss', 0):.2%}"))
        blocks.append(bullet(f"Profit factor: {s.get('profit_factor', 0):.2f} | Sharpe: {s.get('sharpe_ratio', 0):.2f}"))
        blocks.append(bullet(f"Best trade: {s.get('best_trade_pct', 0):.2%} | Worst: {s.get('worst_trade_pct', 0):.2%}"))
        blocks.append(bullet(f"Avg duration: {s.get('avg_trade_duration_h', 0):.1f}h"))

        ca = s.get("condition_analysis", {})
        if ca:
            total = ca.get("total_candles", 1)
            blocks.append(heading("Signal Condition Analysis", 3))
            blocks.append(bullet(f"C1 RSI<35 rising: {ca.get('c1_rsi_oversold_rising', 0)} / {total} ({ca.get('c1_rsi_oversold_rising', 0)/total:.1%})"))
            blocks.append(bullet(f"C2 MACD bullish crossover: {ca.get('c2_macd_crossover', 0)} / {total} ({ca.get('c2_macd_crossover', 0)/total:.1%})"))
            blocks.append(bullet(f"C3 EMA9 > EMA21: {ca.get('c3_ema9_above_ema21', 0)} / {total} ({ca.get('c3_ema9_above_ema21', 0)/total:.1%})"))
            blocks.append(bullet(f"C4 BB squeeze + above BBL: {ca.get('c4_bb_squeeze_above_lower', 0)} / {total} ({ca.get('c4_bb_squeeze_above_lower', 0)/total:.1%})"))
            blocks.append(bullet(f"C5 Volume spike (1.5×): {ca.get('c5_volume_spike', 0)} / {total} ({ca.get('c5_volume_spike', 0)/total:.1%})"))
            blocks.append(bullet(f"C6 Daily trend positive: {ca.get('c6_daily_trend_positive', 0)} / {total} ({ca.get('c6_daily_trend_positive', 0)/total:.1%})"))
            blocks.append(bullet(f"C1 ∧ C2: {ca.get('c1_and_c2', 0)} | C1 ∧ C2 ∧ C3: {ca.get('c1_and_c2_and_c3', 0)} | ALL: {ca.get('all_conditions', 0)}"))
            if ca.get("all_conditions", 0) == 0:
                blocks.append(para(
                    "⚠️ FINDING: Strategia nie wygenerowała żadnych sygnałów BUY w badanym okresie. "
                    "Warunki C1 (RSI<35 = oversold/downtrend) i C3 (EMA9>EMA21 = bullish short-term) "
                    "są strukturalnie sprzeczne podczas silnych spadków. "
                    "Rekomendacja: rozważyć podniesienie progu RSI lub usunięcie warunku EMA."
                ))

        if self.trades:
            blocks.append(heading("Trade Log (first 20)", 3))
            for t in self.trades[:20]:
                icon = "✅" if t.is_winner else "❌"
                blocks.append(bullet(
                    f"{icon} {t.symbol} | {t.entry_time.strftime('%Y-%m-%d %H:%M')} → "
                    f"{t.exit_time.strftime('%Y-%m-%d %H:%M')} | "
                    f"{t.entry_price:.4f} → {t.exit_price:.4f} | "
                    f"PnL: {t.pnl_pct:.2%} ({t.pnl_usdc:+.2f} USDC) | {t.exit_reason}"
                ))

        return blocks
