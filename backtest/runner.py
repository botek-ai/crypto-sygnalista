"""CLI runner for the backtest engine.

Usage:
    python -m backtest.runner --symbol BTC/USDT --timeframe 4h --days 90
    python -m backtest.runner  # runs default symbols
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

# Ensure project root is importable when run as __main__
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine import BacktestConfig, BacktestEngine
from backtest.report import BacktestReport
from config.settings import get_settings
from data.fetcher import BinanceFetcher


DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
DEFAULT_TIMEFRAME = "4h"
DEFAULT_DAYS = 90


def _candles_for_days(timeframe: str, days: int) -> int:
    """Calculate number of candles needed to cover N days."""
    units = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    tf_min = 60
    for suffix, mult in units.items():
        if timeframe.endswith(suffix):
            try:
                tf_min = int(timeframe[:-1]) * mult
                break
            except ValueError:
                pass
    return int(days * 1440 / tf_min) + 100  # +100 for indicator warmup


async def run_backtest(
    symbol: str,
    timeframe: str,
    days: int,
    initial_capital: float,
    fetcher: BinanceFetcher,
    output_dir: str = "backtest_results",
) -> dict:
    """Fetch data and run backtest for a single symbol."""
    logger.info("Fetching {} @ {} ({} days)...", symbol, timeframe, days)

    limit = _candles_for_days(timeframe, days)
    daily_limit = days + 100  # extra for daily EMA50 warmup

    try:
        df = await fetcher.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        daily_df = await fetcher.fetch_ohlcv(symbol, timeframe="1d", limit=daily_limit)
    except Exception as exc:
        logger.error("Failed to fetch data for {}: {}", symbol, exc)
        return {"error": str(exc), "symbol": symbol}

    logger.info(
        "Fetched {} candles ({} to {})",
        len(df),
        df.index[0].strftime("%Y-%m-%d"),
        df.index[-1].strftime("%Y-%m-%d"),
    )

    settings = get_settings()
    config = BacktestConfig(
        symbol=symbol,
        timeframe=timeframe,
        initial_capital=initial_capital,
        max_positions=settings.max_open_positions,
        position_size_pct=settings.position_size_pct,
        min_position_usdc=settings.min_position_usdc,
        max_coin_exposure_pct=settings.max_coin_exposure_pct,
        cooldown_minutes=settings.cooldown_minutes,
        stop_loss_pct=settings.stop_loss_pct,
        take_profit_pct=settings.take_profit_pct,
        trailing_be_activation_pct=settings.trailing_stop_activation_pct,
        trailing2_activation_pct=settings.trailing_stop_2_activation_pct,
        trailing2_distance_pct=settings.trailing_stop_2_distance_pct,
        emergency_minutes=settings.emergency_exit_minutes,
        emergency_threshold_pct=settings.emergency_exit_threshold_pct,
        rsi_threshold=settings.rsi_buy_threshold,
        bb_squeeze_width=settings.bb_squeeze_width,
        volume_multiplier=settings.volume_multiplier,
    )

    engine = BacktestEngine(df=df, daily_df=daily_df, config=config)
    results = engine.run()
    results["initial_capital"] = initial_capital

    report = BacktestReport(results, output_dir=output_dir)
    text = report.text_report()
    print(text)

    txt_path = report.save_text()
    csv_path = report.save_csv()
    logger.info("Saved: {} | {}", txt_path, csv_path)

    results["report_text"] = text
    results["report_path"] = str(txt_path)
    results["notion_blocks"] = report.notion_blocks()

    return results


async def main(args: argparse.Namespace) -> None:
    """Main async entry point."""
    logger.remove()
    logger.add(sys.stderr, level=args.log_level, format="<level>{level}</level> | {message}")

    symbols = args.symbols if args.symbols else DEFAULT_SYMBOLS
    timeframe = args.timeframe
    days = args.days
    capital = args.capital
    output_dir = args.output_dir

    all_results: list[dict] = []

    async with BinanceFetcher() as fetcher:
        for symbol in symbols:
            logger.info("=" * 50)
            logger.info("Running backtest: {} | {} | {}d", symbol, timeframe, days)
            logger.info("=" * 50)
            result = await run_backtest(
                symbol=symbol,
                timeframe=timeframe,
                days=days,
                initial_capital=capital,
                fetcher=fetcher,
                output_dir=output_dir,
            )
            all_results.append(result)

    # Summary if multiple symbols
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY — ALL SYMBOLS")
        print("=" * 60)
        print(f"{'Symbol':<14} {'Trades':>7} {'Win%':>7} {'Return':>9} {'MaxDD':>8} {'Sharpe':>7}")
        print("-" * 60)
        for r in all_results:
            if "error" in r:
                print(f"{r.get('symbol', '?'):<14}  ERROR: {r['error']}")
                continue
            s = r.get("stats", {})
            print(
                f"{s.get('symbol', '?'):<14}"
                f"  {s.get('total_trades', 0):>6}"
                f"  {s.get('win_rate', 0):>6.1%}"
                f"  {s.get('total_return_pct', 0):>8.2%}"
                f"  {s.get('max_drawdown_pct', 0):>7.2%}"
                f"  {s.get('sharpe_ratio', 0):>6.2f}"
            )
        print("=" * 60)

    return all_results  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crypto backtest engine — test multi-indicator strategy on historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbol", dest="symbols", action="append",
        help="Trading pair (e.g. BTC/USDT). Repeatable. Default: BTC/USDT ETH/USDT SOL/USDT",
    )
    parser.add_argument(
        "--timeframe", default=DEFAULT_TIMEFRAME,
        help=f"OHLCV timeframe (default: {DEFAULT_TIMEFRAME})",
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_DAYS,
        help=f"Number of historical days (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--capital", type=float, default=1000.0,
        help="Initial capital in USDC (default: 1000.0)",
    )
    parser.add_argument(
        "--output-dir", default="backtest_results",
        help="Directory for output files (default: backtest_results)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
