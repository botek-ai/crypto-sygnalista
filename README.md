# crypto-sygnalista 🤖

Crypto signals tracker — RSI, MACD, EMA, Bollinger Bands. Built by Botek AI team.

## Strategia sygnałów

### BUY (wszystkie warunki muszą być spełnione)
- RSI(14) < 35 i rośnie
- MACD bullish crossover (MACD przebija sygnał od dołu)
- EMA9 > EMA21
- Cena powyżej dolnej Bollinger Band po squeeze (BB width < 0.03)
- Wolumen świecy > 1.5× średnia z 20 świec
- Trend dzienny pozytywny (close > EMA50 na 1d)

### SELL
- Stop-loss: -3%
- Take-profit: 150%
- Trailing stop po +2%: SL → break-even
- Trailing stop po +3.5%: 1.5% od szczytu
- Wyjście awaryjne: po 90 min jeśli ±1%
- RSI > 70 i spada + MACD crossover w dół

### Filtry pozycji
- Cooldown 12 min po buy/sell na danym coinie
- Max 8 otwartych pozycji jednocześnie
- Max 15% ekspozycji na jeden coin
- Trailing stop-buy: 0.5%
- Rozmiar pozycji: 20% wolnego kapitału (min 15 USDC)

## Setup

```bash
# Zainstaluj uv (jeśli nie masz)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sklonuj i skonfiguruj
git clone git@github.com:botek-ai/crypto-sygnalista.git
cd crypto-sygnalista

# Stwórz środowisko i zainstaluj zależności
uv venv
uv sync

# Skonfiguruj zmienne środowiskowe
cp .env.example .env
# Uzupełnij .env swoimi kluczami API

# Uruchom
uv run python main.py
```

## Struktura projektu

```
crypto-sygnalista/
├── main.py              # Punkt wejścia
├── config/
│   ├── settings.py      # Konfiguracja (Pydantic BaseSettings)
│   └── symbols.yaml     # Lista par do śledzenia
├── data/
│   ├── fetcher.py       # Pobieranie danych z Binance (ccxt)
│   └── cache.py         # Cache OHLCV
├── indicators/
│   └── technical.py     # RSI, MACD, EMA, BB (pandas-ta)
├── signals/
│   ├── engine.py        # Silnik sygnałów
│   └── rules.py         # Reguły buy/sell
├── db/
│   └── models.py        # SQLAlchemy models
├── notifications/
│   └── telegram.py      # Powiadomienia Telegram
└── scheduler/
    └── jobs.py          # APScheduler jobs
```

## Stack

- **Python 3.12+** z `uv`
- **ccxt** — dane z Binance
- **pandas-ta** — wskaźniki techniczne
- **APScheduler** — scheduler
- **SQLAlchemy 2.x + aiosqlite** — baza danych
- **Pydantic BaseSettings** — konfiguracja
- **python-telegram-bot** — powiadomienia
- **loguru** — logowanie
