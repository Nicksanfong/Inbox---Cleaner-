# Algorithmic Trading Platform

A modular, Python-based algorithmic trading platform built for paper and live trading.

## Project Layout

```
.
├── config/        # Config loader, YAML settings, logging setup
├── data/          # Market data cache and raw downloads
├── strategies/    # Trading strategy implementations
├── indicators/    # Technical indicators (RSI, MACD, BB, etc.)
├── patterns/      # Chart pattern detection
├── risk/          # Position sizing and risk management
├── execution/     # Order routing and broker integration
├── dashboard/     # Web or CLI performance dashboard
├── logs/          # Rotating log files (auto-created)
├── tests/         # Unit and integration tests
├── main.py        # Entry point
├── config/config.yaml    # App configuration (no secrets)
└── .env           # Secrets (never commit this file)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create your secrets file

```bash
cp .env.example .env
# Edit .env with your broker API keys
```

### 3. Run the platform

```bash
python main.py
```

You should see "System ready" in the output.

## Configuration

Edit `config/config.yaml` to change:
- Which symbols to trade
- Risk limits (max position size, drawdown limits)
- Logging verbosity

Secrets (API keys, passwords) go in `.env` only — never in YAML.

## Running Tests

```bash
pytest tests/ -v
```
