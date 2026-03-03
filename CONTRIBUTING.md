# Contributing

## Commit messages

Używamy [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

Format: `<type>[scope]: <description>`

### Typy
- `feat` — nowa funkcjonalność
- `fix` — naprawa błędu
- `docs` — dokumentacja
- `style` — formatowanie (bez zmiany logiki)
- `refactor` — refaktoryzacja
- `test` — testy
- `chore` — konfiguracja, zależności

### Przykłady
```
feat(signals): add multi-indicator buy signal engine
fix(fetcher): handle Binance 429 rate limit with backoff
docs(strategy): document RSI threshold rationale
refactor(indicators): extract base indicator class
test(signals): add unit tests for MACD crossover detection
chore(deps): add pandas-ta and ccxt to pyproject.toml
```
