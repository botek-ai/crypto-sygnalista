"""SQLAlchemy 2.x async models for crypto-sygnalista."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config.settings import get_settings


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class Signal(Base):
    """Recorded signal events (BUY/SELL/HOLD)."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    signal_type: Mapped[str] = mapped_column(String(10), nullable=False)  # BUY / SELL / HOLD
    price: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)

    # Indicators snapshot
    rsi: Mapped[float | None] = mapped_column(Float)
    macd: Mapped[float | None] = mapped_column(Float)
    macd_signal: Mapped[float | None] = mapped_column(Float)
    ema9: Mapped[float | None] = mapped_column(Float)
    ema21: Mapped[float | None] = mapped_column(Float)
    bb_width: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    volume_avg20: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Signal {self.signal_type} {self.symbol} @ {self.price} [{self.created_at}]>"


class Position(Base):
    """Tracked open/closed positions."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float | None] = mapped_column(Float)
    size_usdc: Mapped[float] = mapped_column(Float, nullable=False)
    peak_price: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False, default="OPEN")  # OPEN / CLOSED

    entry_signal_id: Mapped[int | None] = mapped_column(Integer)
    exit_signal_id: Mapped[int | None] = mapped_column(Integer)

    pnl_pct: Mapped[float | None] = mapped_column(Float)
    pnl_usdc: Mapped[float | None] = mapped_column(Float)

    entry_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    exit_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<Position {self.symbol} entry={self.entry_price} "
            f"size={self.size_usdc}USDC status={self.status}>"
        )


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_async_engine() -> AsyncEngine:
    """Return (or create) the singleton async SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            future=True,
        )
    return _engine


def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Async generator yielding a database session.

    Usage::

        async with get_async_session() as session:
            ...
    """
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_async_engine(), expire_on_commit=False
        )
    return _session_factory()  # type: ignore[return-value]


async def init_db() -> None:
    """Create all database tables if they don't exist."""
    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
