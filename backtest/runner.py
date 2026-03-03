"""CLI runner for the backtest engine — multi-pool, all symbols.

Usage:
    python -m backtest.runner                          # all 15 symbols, 4h, 90 days
    python -m backtest.runner --days 60 --timeframe 1h
    python -m backtest.runner --symbol BTC/USDT        # single symbol override
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

from backtest.engine import POOLS, ALL_SYMBOLS, BacktestConfig, BacktestEngine, get_pool_for_symbol
from backtest.report import BacktestReport
from config.settings import get_settings
from data.fetcher import BinanceFetcher


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
    """Fetch data and run backtest for a single symbol using its pool params."""
    pool_name, pool_params = get_pool_for_symbol(symbol)
    trailing_stop = pool_params["trailing_stop"]
    stop_loss = pool_params["stop_loss"]

    logger.info(
        "Fetching {} @ {} ({} days) [pool={}, trail={:.1%}, sl={:.1%}]...",
        symbol, timeframe, days, pool_name, trailing_stop, stop_loss,
    )

    limit = _candles_for_days(timeframe, days)
    daily_limit = days + 100  # extra for daily EMA50 warmup
    h1_limit = days * 24 + 200  # 1h candles for dual-timeframe exits

    try:
        df = await fetcher.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        daily_df = await fetcher.fetch_ohlcv(symbol, timeframe="1d", limit=daily_limit)
        h1_df = await fetcher.fetch_ohlcv(symbol, timeframe="1h", limit=h1_limit)
        logger.info("Fetched {} 1h candles for {}", len(h1_df), symbol)
    except Exception as exc:
        logger.error("Failed to fetch data for {}: {}", symbol, exc)
        return {"error": str(exc), "symbol": symbol, "pool": pool_name}

    logger.info(
        "Fetched {} candles ({} to {})",
        len(df),
        df.index[0].strftime("%Y-%m-%d"),
        df.index[-1].strftime("%Y-%m-%d"),
    )

    settings = get_settings()
    trailing_activation = pool_params.get("trailing_activation", 0.015)

    config = BacktestConfig(
        symbol=symbol,
        timeframe=timeframe,
        initial_capital=initial_capital,
        max_positions=settings.max_open_positions,
        position_size_pct=settings.position_size_pct,
        min_position_usdc=30.0,  # v3: raised minimum
        max_coin_exposure_pct=max(settings.max_coin_exposure_pct, settings.position_size_pct),
        cooldown_minutes=settings.cooldown_minutes,
        # Per-pool TPSL
        trailing_stop_pct=trailing_stop,
        trailing_activation_pct=trailing_activation,
        stop_loss_pct=stop_loss,
        # Emergency exit
        emergency_minutes=settings.emergency_exit_minutes,
        emergency_threshold_pct=settings.emergency_exit_threshold_pct,
        # Commission
        commission_pct=0.00075,
        # V2 scoring strategy
        # F1 disabled: EMA9<EMA21 in downtrend → too restrictive, 0 trades
        # F2 disabled: close_daily > EMA21_daily → too restrictive
        min_score=3,
        use_f1_ema_filter=False,
        use_f2_daily_filter=False,
        rsi_threshold=45.0,
        bb_pctb_threshold=0.3,
        volume_multiplier=settings.volume_multiplier,
    )

    engine = BacktestEngine(df=df, daily_df=daily_df, config=config, h1_df=h1_df)
    results = engine.run()
    results["initial_capital"] = initial_capital
    results["pool"] = pool_name
    results["pool_params"] = pool_params

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


def _print_pool_summary(pool_name: str, results: list[dict]) -> None:
    """Print summary table for one pool."""
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    pool_cfg = POOLS[pool_name]
    print(f"\n{'═' * 65}")
    print(f"  POOL: {pool_name.upper()} | trailing={pool_cfg['trailing_stop']:.1%} | sl={pool_cfg['stop_loss']:.1%}")
    print(f"{'═' * 65}")
    print(f"{'Symbol':<14} {'Trades':>7} {'Win%':>7} {'Return':>9} {'MaxDD':>8} {'Sharpe':>7} {'PnL USDC':>10}")
    print(f"{'-' * 65}")

    pool_trades = pool_pnl = 0
    pool_winners = 0
    for r in valid:
        s = r.get("stats", {})
        trades = s.get("total_trades", 0)
        winners = s.get("winning_trades", 0)
        pnl = s.get("total_pnl_usdc", 0.0)
        pool_trades += trades
        pool_pnl += pnl
        pool_winners += winners
        print(
            f"{s.get('symbol', '?'):<14}"
            f"  {trades:>6}"
            f"  {s.get('win_rate', 0):>6.1%}"
            f"  {s.get('total_return_pct', 0):>8.2%}"
            f"  {s.get('max_drawdown_pct', 0):>7.2%}"
            f"  {s.get('sharpe_ratio', 0):>6.2f}"
            f"  {pnl:>+9.2f}"
        )
    for r in errors:
        print(f"{r.get('symbol', '?'):<14}  ERROR: {r.get('error', '?')}")

    if valid:
        pool_winrate = pool_winners / pool_trades if pool_trades > 0 else 0.0
        print(f"{'-' * 65}")
        print(f"{'POOL TOTAL':<14}  {pool_trades:>6}  {pool_winrate:>6.1%}  {'':>8}  {'':>7}  {'':>6}  {pool_pnl:>+9.2f}")


def _print_grand_summary(all_results: list[dict]) -> None:
    """Print grand summary across all pools and symbols."""
    print(f"\n{'═' * 70}")
    print("  GRAND SUMMARY — ALL POOLS & SYMBOLS")
    print(f"{'═' * 70}")

    grand_trades = grand_pnl = grand_winners = 0
    for pool_name in POOLS:
        pool_results = [r for r in all_results if r.get("pool") == pool_name]
        valid = [r for r in pool_results if "error" not in r]
        for r in valid:
            s = r.get("stats", {})
            grand_trades += s.get("total_trades", 0)
            grand_pnl += s.get("total_pnl_usdc", 0.0)
            grand_winners += s.get("winning_trades", 0)

    grand_winrate = grand_winners / grand_trades if grand_trades > 0 else 0.0
    total_symbols = len([r for r in all_results if "error" not in r])
    error_count = len([r for r in all_results if "error" in r])

    print(f"Symbols tested:   {total_symbols} / {len(all_results)} ({error_count} errors)")
    print(f"Total trades:     {grand_trades}")
    print(f"Total winners:    {grand_winners} ({grand_winrate:.1%})")
    print(f"Total P&L:        {grand_pnl:+.2f} USDC")
    print(f"{'═' * 70}")

    # Condition analysis summary (first valid result per pool)
    print("\n── RSI Threshold Analysis (signals across approaches) ──────────────")
    print(f"{'Symbol':<14} {'Appr.A RSI<45':>13} {'Appr.B RSI<45+EMA':>17} {'Appr.C RSI<35+EMA':>17}")
    print(f"{'-' * 65}")
    for r in all_results:
        if "error" in r:
            continue
        s = r.get("stats", {})
        ca = s.get("condition_analysis", {})
        sym = s.get("symbol", "?")
        a = ca.get("approach_A_rsi45_no_ema", 0)
        b = ca.get("approach_B_rsi45_with_ema", 0)
        c = ca.get("approach_C_rsi35_with_ema", 0)
        print(f"{sym:<14}  {a:>12}  {b:>16}  {c:>16}")
    print(f"{'═' * 70}")


async def main(args: argparse.Namespace) -> list[dict]:
    """Main async entry point."""
    logger.remove()
    logger.add(sys.stderr, level=args.log_level, format="<level>{level}</level> | {message}")

    # Symbol selection
    if args.symbols:
        symbols = args.symbols
        # Group them by pool (or run all under "custom")
        pool_groups: dict[str, list[str]] = {"custom": symbols}
    else:
        # Default: all pools, all symbols
        pool_groups = {name: pool["symbols"] for name, pool in POOLS.items()}

    timeframe = args.timeframe
    days = args.days
    capital = args.capital
    output_dir = args.output_dir

    all_results: list[dict] = []

    async with BinanceFetcher() as fetcher:
        for pool_name, symbols in pool_groups.items():
            pool_results: list[dict] = []
            for symbol in symbols:
                logger.info("=" * 55)
                logger.info("Pool={} | {} | {} | {}d", pool_name, symbol, timeframe, days)
                logger.info("=" * 55)
                result = await run_backtest(
                    symbol=symbol,
                    timeframe=timeframe,
                    days=days,
                    initial_capital=capital,
                    fetcher=fetcher,
                    output_dir=output_dir,
                )
                # Tag with pool if custom run
                if args.symbols:
                    p, _ = get_pool_for_symbol(symbol)
                    result["pool"] = p
                pool_results.append(result)
                all_results.append(result)

            if pool_name != "custom":
                _print_pool_summary(pool_name, pool_results)

    # Grand summary
    _print_grand_summary(all_results)

    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crypto backtest engine — multi-pool strategy test\n"
            "Pools: high (INJ/RENDER/SEI), mid (9 symbols), low (BNB/BTC/ETH)\n"
            "Default: all 15 symbols, 4h, 90 days"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbol", dest="symbols", action="append",
        help="Trading pair override (e.g. BTC/USDT). Repeatable. Default: all 15 pool symbols.",
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
