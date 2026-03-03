"""Application settings using Pydantic BaseSettings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Binance
    binance_api_key: str = Field(default="", description="Binance API key")
    binance_api_secret: str = Field(default="", description="Binance API secret")
    binance_testnet: bool = Field(default=False, description="Use Binance testnet")

    # Telegram
    telegram_bot_token: str = Field(default="", description="Telegram bot token")
    telegram_chat_id: str = Field(default="", description="Telegram chat ID")

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./crypto_sygnalista.db",
        description="SQLAlchemy database URL",
    )

    # Capital management
    initial_capital: float = Field(default=1000.0, ge=0, description="Initial capital in USDC")
    position_size_pct: float = Field(
        default=0.20, ge=0.01, le=1.0, description="Position size as fraction of free capital"
    )
    min_position_usdc: float = Field(
        default=15.0, ge=1.0, description="Minimum position size in USDC"
    )
    max_open_positions: int = Field(default=8, ge=1, le=50, description="Max open positions")
    max_coin_exposure_pct: float = Field(
        default=0.15, ge=0.01, le=1.0, description="Max exposure per coin as fraction of capital"
    )

    # Signal settings
    cooldown_minutes: int = Field(
        default=12, ge=1, description="Cooldown in minutes after buy/sell"
    )
    rsi_buy_threshold: float = Field(default=35.0, description="RSI threshold for buy signal")
    rsi_sell_threshold: float = Field(default=70.0, description="RSI threshold for sell signal")

    # Sell rules
    stop_loss_pct: float = Field(default=0.03, description="Stop-loss percentage (3%)")
    take_profit_pct: float = Field(default=1.50, description="Take-profit percentage (150%)")
    trailing_stop_activation_pct: float = Field(
        default=0.02, description="Trailing stop activation at +2%"
    )
    trailing_stop_2_activation_pct: float = Field(
        default=0.035, description="Second trailing stop activation at +3.5%"
    )
    trailing_stop_2_distance_pct: float = Field(
        default=0.015, description="Trailing stop distance at 1.5% from peak"
    )
    emergency_exit_minutes: int = Field(
        default=90, description="Emergency exit after N minutes if ±1%"
    )
    emergency_exit_threshold_pct: float = Field(
        default=0.01, description="Emergency exit price threshold (±1%)"
    )
    trailing_stop_buy_pct: float = Field(
        default=0.005, description="Trailing stop-buy distance (0.5%)"
    )

    # BB squeeze
    bb_squeeze_width: float = Field(
        default=0.03, description="Bollinger Band width threshold for squeeze"
    )

    # Volume filter
    volume_multiplier: float = Field(
        default=1.5, description="Volume must be > X × 20-candle average"
    )
    volume_avg_period: int = Field(default=20, description="Volume moving average period")

    # Scheduler
    scan_interval_seconds: int = Field(default=60, description="Scan interval in seconds")
    daily_trend_interval_seconds: int = Field(
        default=300, description="Daily trend update interval in seconds"
    )

    # Paths
    symbols_config_path: Path = Field(
        default=Path("config/symbols.yaml"), description="Path to symbols YAML config"
    )

    # Environment
    log_level: str = Field(default="INFO", description="Log level")
    dry_run: bool = Field(default=True, description="Dry run mode — no real orders")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a recognized loguru level."""
        valid = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid}")
        return v.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
