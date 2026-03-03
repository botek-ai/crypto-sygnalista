"""Telegram notification sender using python-telegram-bot."""

from __future__ import annotations

from typing import Optional

from loguru import logger
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

from config.settings import Settings, get_settings
from signals.rules import SignalEvent, SignalType


class TelegramNotifier:
    """Send signal alerts to a Telegram chat.

    Formats BUY and SELL messages with emoji and relevant indicator data.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize with application settings.

        Args:
            settings: App settings. Defaults to get_settings().
        """
        self._settings = settings or get_settings()
        self._bot: Optional[Bot] = None

    def _get_bot(self) -> Optional[Bot]:
        """Lazily initialize the Bot instance."""
        if not self._settings.telegram_bot_token:
            return None
        if self._bot is None:
            self._bot = Bot(token=self._settings.telegram_bot_token)
        return self._bot

    async def send_signal(self, event: SignalEvent) -> bool:
        """Send a formatted signal notification.

        Args:
            event: Signal event to notify about.

        Returns:
            True if message was sent successfully, False otherwise.
        """
        bot = self._get_bot()
        if bot is None:
            logger.warning("Telegram not configured — skipping notification for {}", event.symbol)
            return False

        text = self._format_message(event)
        return await self._send(text)

    async def send_text(self, text: str) -> bool:
        """Send a raw text message.

        Args:
            text: Message text (supports HTML).

        Returns:
            True on success.
        """
        bot = self._get_bot()
        if bot is None:
            logger.warning("Telegram not configured — skipping message")
            return False
        return await self._send(text)

    async def _send(self, text: str) -> bool:
        """Internal: send message and log result."""
        bot = self._get_bot()
        if bot is None:
            return False
        try:
            await bot.send_message(
                chat_id=self._settings.telegram_chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
            )
            logger.debug("Telegram message sent ({} chars)", len(text))
            return True
        except TelegramError as exc:
            logger.error("Telegram send failed: {}", exc)
            return False

    def _format_message(self, event: SignalEvent) -> str:
        """Format a signal event as a Telegram HTML message.

        Args:
            event: Signal event.

        Returns:
            Formatted HTML string.
        """
        ind = event.indicators
        ts = event.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        dry_run_tag = " <i>[DRY RUN]</i>" if self._settings.dry_run else ""

        if event.signal == SignalType.BUY:
            emoji = "🟢"
            header = f"{emoji} <b>BUY: {event.symbol}</b>{dry_run_tag}"
        elif event.signal == SignalType.SELL:
            emoji = "🔴"
            header = f"{emoji} <b>SELL: {event.symbol}</b>{dry_run_tag}"
        else:
            emoji = "⚪"
            header = f"{emoji} <b>HOLD: {event.symbol}</b>{dry_run_tag}"

        lines = [
            header,
            f"💰 Price: <code>{event.price:.4f} USDT</code>",
            f"🕐 Time: {ts}",
            "",
            "<b>Indicators:</b>",
        ]

        if ind.rsi is not None:
            lines.append(f"  RSI(14): <code>{ind.rsi:.1f}</code> {'↑' if ind.rsi_rising else '↓'}")
        if ind.macd is not None and ind.macd_signal is not None:
            lines.append(f"  MACD: <code>{ind.macd:.6f}</code> / Signal: <code>{ind.macd_signal:.6f}</code>")
        if ind.ema9 is not None and ind.ema21 is not None:
            lines.append(f"  EMA9/EMA21: <code>{ind.ema9:.4f}</code> / <code>{ind.ema21:.4f}</code>")
        if ind.bb_width is not None:
            lines.append(f"  BB Width: <code>{ind.bb_width:.4f}</code>")
        if ind.volume is not None and ind.volume_avg20 is not None:
            lines.append(
                f"  Vol: <code>{ind.volume:.0f}</code> (avg20: <code>{ind.volume_avg20:.0f}</code>)"
            )
        if ind.close_daily is not None and ind.ema50_daily is not None:
            lines.append(
                f"  Daily: close=<code>{ind.close_daily:.4f}</code> EMA50=<code>{ind.ema50_daily:.4f}</code>"
            )

        lines += ["", f"📋 <i>{event.reason}</i>"]
        return "\n".join(lines)
